"""Convert a RECOMP_CAPTURE directory into a glTF 2.0 binary (.glb).

    py -3 tools/capture/capture2gltf.py <capture dir> <out.glb>

The capture (src/kernel/nv2a_pb_exec.c, RECOMP_CAPTURE) holds, per draw,
each vertex's input position (model space), the vertex program's output
position (screen x, y, depth, and w), its colour and texture coordinates,
plus the draw's state and its textures. For each draw this tool:

  1. recovers clip space from the screen output (viewport scale k, offset o);
  2. fits the 4x4 matrix M with clip = M * model (least squares);
  3. takes each frame's view-projection VP as the M that most vertices share
     (static scenery is drawn with one world matrix), and gives the draw the
     world matrix inv(VP) * M: a model-space mesh under a node transform;
  4. writes the draws one matrix does not explain (skinned meshes) in world
     space instead, as posed in that frame.

Draws captured in several frames (a camera moving through a stage) are
merged: a draw seen again with the same vertex buffer, textures and world
matrix is kept once. Draws that sample a render target (full-screen passes,
reflections) are left out. Positions are converted from Direct3D's
left-handed space to glTF's right-handed one by negating z.

Needs numpy and Pillow. The output is made from your own copy of the game:
do not redistribute it.
"""

import argparse
import io
import json
import os
import struct
from collections import Counter, OrderedDict

import numpy as np
from PIL import Image

REC_FLOATS = 29                       # index, v0[4], pos[4], d0[4], tex[4][4]
TRI_PRIMS = (5, 6, 7, 8, 9, 10)
# GL blend / compare enums the capture records
GL_ZERO, GL_ONE = 0, 1
GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0x0302, 0x0303
GL_FUNC_REVERSE_SUBTRACT = 0x800B


def triangles(prim, idx):
    """NV2A primitive -> list of triangles (local vertex numbers)."""
    n = len(idx)
    out = []
    if prim == 5:
        out = [(idx[i], idx[i + 1], idx[i + 2]) for i in range(0, n - 2, 3)]
    elif prim == 6:
        for i in range(n - 2):
            a, b, c = idx[i], idx[i + 1], idx[i + 2]
            out.append((b, a, c) if i & 1 else (a, b, c))
    elif prim in (7, 10):
        out = [(idx[0], idx[i], idx[i + 1]) for i in range(1, n - 1)]
    elif prim == 8:
        for i in range(0, n - 3, 4):
            out += [(idx[i], idx[i + 1], idx[i + 2]), (idx[i], idx[i + 2], idx[i + 3])]
    elif prim == 9:
        for i in range(0, n - 3, 2):
            out += [(idx[i], idx[i + 1], idx[i + 3]), (idx[i], idx[i + 3], idx[i + 2])]
    return [t for t in out if len(set(t)) == 3]


def load(capdir):
    draws = []
    blob = open(os.path.join(capdir, "draws.bin"), "rb").read()
    with open(os.path.join(capdir, "draws.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue                  # cut off when the game was closed
            if d["off"] + d["nvert"] * REC_FLOATS * 4 + d["nidx"] * 4 <= len(blob):
                draws.append(d)
    for d in draws:
        nv, ni, off = d["nvert"], d["nidx"], d["off"]
        rec = np.frombuffer(blob, dtype=np.float32, count=nv * REC_FLOATS,
                            offset=off).reshape(nv, REC_FLOATS)
        d["v0"] = rec[:, 1:5].astype(np.float64)
        d["pos"] = rec[:, 5:9].astype(np.float64)
        d["col"] = rec[:, 9:13].astype(np.float64)
        d["uvq"] = rec[:, 13:29].reshape(nv, 4, 4).astype(np.float64)
        d["local"] = np.frombuffer(blob, dtype=np.uint32, count=ni,
                                   offset=off + nv * REC_FLOATS * 4).tolist()
    return draws


def clip_space(d):
    k = np.array(d["k"], dtype=np.float64)
    o = np.array(d["o"], dtype=np.float64)
    k = np.where(k == 0, 1.0, k)
    p = d["pos"]
    w = p[:, 3:4]
    return np.hstack([(p[:, :3] - o) / k * w, w])


def fit_matrix(model, clip):
    """clip ~= M @ [x y z 1]; returns M (4x4), relative error and rank."""
    a = np.hstack([model[:, :3], np.ones((len(model), 1))])
    m, _r, rank, _s = np.linalg.lstsq(a, clip, rcond=None)
    m = m.T
    err = np.abs(a @ m.T - clip).max() / (np.abs(clip).max() + 1e-9)
    return m, err, rank


def to_trs(w, model):
    """Nearest translation-rotation-scale to the affine matrix w, if it moves
    the model's vertices by less than 0.1 % of their extent (the per-frame
    fit carries a little float noise, glTF nodes allow no shear). Returns
    (t, quaternion xyzw, scale) or None."""
    if np.abs(w[3] - [0, 0, 0, 1]).max() > 1e-3 * max(1.0, np.abs(w).max()):
        return None
    a = w[:3, :3]
    sc = np.linalg.norm(a, axis=0)
    if sc.min() < 1e-9:
        return None
    u, _s, vt = np.linalg.svd(a / sc)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r[:, 2] = -r[:, 2]
        sc = sc * np.array([1.0, 1.0, -1.0])
    snapped = r * sc
    ext = (model.max(axis=0) - model.min(axis=0)).max() + 1e-9
    err = np.abs(model @ (snapped - a).T).max()
    if err > 1e-3 * ext * np.abs(sc).max():
        return None
    # rotation matrix -> quaternion (x, y, z, w)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        q4 = np.sqrt(tr + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / q4, (r[0, 2] - r[2, 0]) / q4, (r[1, 0] - r[0, 1]) / q4, 0.25 * q4]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        q4 = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * q4, (r[0, 1] + r[1, 0]) / q4, (r[0, 2] + r[2, 0]) / q4, (r[2, 1] - r[1, 2]) / q4]
    elif r[1, 1] > r[2, 2]:
        q4 = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / q4, 0.25 * q4, (r[1, 2] + r[2, 1]) / q4, (r[0, 2] - r[2, 0]) / q4]
    else:
        q4 = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / q4, (r[1, 2] + r[2, 1]) / q4, 0.25 * q4, (r[1, 0] - r[0, 1]) / q4]
    q = np.array(q) / np.linalg.norm(q)
    return w[:3, 3].tolist(), q.tolist(), sc.tolist()


def matrix_key(m):
    s = np.abs(m).max() + 1e-12
    return tuple(np.round(m / s, 4).ravel())


def texture_png(capdir, t, cache):
    key = (t["off"], t["fmt"], t["w"], t["h"])
    if key in cache:
        return cache[key]
    path = os.path.join(capdir, "tex_%s_%02X_%dx%d.argb" % (t["off"], t["fmt"], t["w"], t["h"]))
    png = None
    if os.path.exists(path):
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == t["w"] * t["h"] * 4:
            bgra = raw.reshape(t["h"], t["w"], 4)
            rgba = bgra[:, :, [2, 1, 0, 3]]
            buf = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(buf, "PNG")
            png = buf.getvalue()
    cache[key] = png
    return png


class Glb:
    def __init__(self):
        self.bin = bytearray()
        self.j = OrderedDict(asset={"version": "2.0", "generator": "xboxrecomp capture2gltf"},
                             scene=0, scenes=[{"nodes": []}], nodes=[], meshes=[],
                             materials=[], textures=[], images=[], samplers=[],
                             accessors=[], bufferViews=[], buffers=[])
        self.j["samplers"].append({"magFilter": 9729, "minFilter": 9729,
                                   "wrapS": 10497, "wrapT": 10497})

    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.j["bufferViews"].append(v)
        return len(self.j["bufferViews"]) - 1

    def accessor(self, arr, kind, target):
        arr = np.ascontiguousarray(arr)
        comp = 5126 if arr.dtype == np.float32 else 5125
        a = {"bufferView": self.view(arr.tobytes(), target), "componentType": comp,
             "count": int(arr.shape[0]), "type": kind}
        if kind == "VEC3" and comp == 5126:
            a["min"] = arr.min(axis=0).tolist()
            a["max"] = arr.max(axis=0).tolist()
        self.j["accessors"].append(a)
        return len(self.j["accessors"]) - 1

    def image(self, png):
        self.j["images"].append({"bufferView": self.view(png), "mimeType": "image/png"})
        self.j["textures"].append({"sampler": 0, "source": len(self.j["images"]) - 1})
        return len(self.j["textures"]) - 1

    def write(self, path):
        self.j["buffers"] = [{"byteLength": len(self.bin)}]
        for k in ("materials", "textures", "images"):
            if not self.j[k]:
                del self.j[k]
        js = json.dumps(self.j, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        bn = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bn)))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(bn), 0x004E4942) + bn)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capdir")
    ap.add_argument("out")
    ap.add_argument("--max-error", type=float, default=2e-3,
                    help="largest relative error for a one-matrix fit")
    ap.add_argument("--frame", type=int, help="only this captured frame (1, 2, ...)")
    ap.add_argument("--skinned-only", action="store_true",
                    help="only draws one rigid matrix does not explain: skinned meshes "
                         "such as the player character, as posed in that frame")
    args = ap.parse_args()

    draws = load(args.capdir)
    if args.frame is not None:
        draws = [d for d in draws if d["frame"] == args.frame]
    surfs = {int(d["surf"], 16) & 0x0FFFFFFF for d in draws}
    frames = sorted({d["frame"] for d in draws})
    print(f"{len(draws)} draws in {len(frames)} frames")

    # Per draw: clip space and the model->clip fit M (clip = VP * W * model).
    for d in draws:
        d["clip"] = clip_space(d)
        if d["nvert"] >= 4:
            d["M"], d["err"], d["rank"] = fit_matrix(d["v0"], d["clip"])
        else:
            d["M"], d["err"], d["rank"] = None, 1.0, 0
        d["key"] = (d["vb"], d["nvert"], d["nidx"],
                    tuple(t["off"] for t in d["tex"] if t["on"]))
        d["good"] = d["M"] is not None and d["rank"] == 4 and d["err"] < args.max_error

    # One world space for all frames. The camera (VP) is never known, but
    # inv(M_a) * M_b = inv(W_a) * W_b for two draws of one frame. So a static
    # reference object R defines the world (W_R = I): a frame that draws an
    # object whose world matrix G is known gets T_f = G * inv(M), the map from
    # its clip space to world, and then every draw in it is placed:
    # W = T_f * M (model space + node matrix), or T_f * clip per vertex.
    byframe = {}
    for d in draws:
        byframe.setdefault(d["frame"], []).append(d)
    seen_in = Counter()
    for f, ds in byframe.items():
        for k in {d["key"] for d in ds if d["good"]}:
            seen_in[k] += 1
    nverts = {d["key"]: d["nvert"] for d in draws}
    ref = max(seen_in, key=lambda k: (seen_in[k], nverts[k])) if seen_in else None
    known = {ref: np.eye(4)} if ref else {}
    xform = {}
    progress = True
    while progress:
        progress = False
        for f, ds in byframe.items():
            if f in xform:
                continue
            anchor = max((d for d in ds if d["good"] and d["key"] in known),
                         key=lambda d: d["nvert"], default=None)
            if anchor is None:
                continue
            t = known[anchor["key"]] @ np.linalg.inv(anchor["M"])
            xform[f] = t
            progress = True
            for d in ds:
                if d["good"] and d["key"] not in known:
                    w = t @ d["M"]
                    if np.abs(w[3] - [0, 0, 0, 1]).max() < 1e-3 * max(1.0, np.abs(w).max()):
                        known[d["key"]] = w
    print(f"world space from {len(xform)} of {len(frames)} frames "
          f"(reference: vertex buffer {ref[0] if ref else '-'})")

    g = Glb()
    # Direct3D is left-handed, glTF right-handed: one root node mirrors z.
    root = {"name": "capture", "scale": [1.0, 1.0, -1.0], "children": []}
    g.j["nodes"].append(root)
    g.j["scenes"][0]["nodes"].append(0)
    texcache, matcache = {}, {}
    skipped_rt = skipped_other = 0

    # Pass 1: place every draw. Model space + TRS node where one clean matrix
    # explains it, else its vertices in world space (planar, skinned).
    cands = {}                                  # object key -> {frame: [placements]}
    for d in draws:
        tex = d["tex"]
        if d["prim"] not in TRI_PRIMS or d["frame"] not in xform:
            skipped_other += 1
            continue
        if any(t["on"] and (int(t["off"], 16) & 0x0FFFFFFF) in surfs for t in tex):
            skipped_rt += 1
            continue
        tris = triangles(d["prim"], d["local"])
        if not tris:
            skipped_other += 1
            continue
        t = xform[d["frame"]]
        world = t @ d["M"] if d["good"] else None
        trs = to_trs(world, d["v0"][:, :3]) if world is not None else None
        skinned = d["M"] is not None and d["rank"] == 4 and d["err"] >= args.max_error
        if args.skinned_only and not skinned:
            continue
        if trs is not None:
            pos = d["v0"][:, :3]
        else:
            # Vertices behind the camera or the near plane were clipped by the
            # GPU; their world position from clip space is meaningless.
            c = d["clip"]
            wmin = max(1e-6, 1e-2 * float(np.median(np.abs(c[:, 3]))))
            vis = (c[:, 3] > wmin) & (c[:, 2] >= -1e-4 * np.abs(c[:, 3]))
            tris = [tr for tr in tris if vis[tr[0]] and vis[tr[1]] and vis[tr[2]]]
            if not tris:
                skipped_other += 1
                continue
            h = (t @ c.T).T
            pos = h[:, :3] / np.where(np.abs(h[:, 3:4]) < 1e-12, 1.0, h[:, 3:4])
        uv = d["uvq"][:, 0, :]
        q = np.where(np.abs(uv[:, 3]) < 1e-12, 1.0, uv[:, 3])
        st = np.stack([uv[:, 0] / q, uv[:, 1] / q], axis=1)
        t0 = tex[0] if tex and tex[0]["on"] else None
        if t0 and not t0["swz"]:                # linear texture: texel units
            st = st / np.array([max(t0["w"], 1), max(t0["h"], 1)])
        used = sorted({i for tr in tris for i in tr})
        if not (np.isfinite(pos[used]).all() and np.isfinite(st[used]).all()):
            skipped_other += 1
            continue
        cands.setdefault(d["key"], {}).setdefault(d["frame"], []).append(
            dict(d=d, tris=tris, pos=pos, st=st, trs=trs, t0=t0))

    # Pass 2: one frame per object. Copies placed in the same frame are real
    # instances; the same object in other frames is either the same placement
    # again or a moving object, which would otherwise leave a trail.
    kept = posed = 0
    for key, byf in cands.items():
        best = max(byf.values(), key=len)
        for c in best:
            d, tex, t0 = c["d"], c["d"]["tex"], c["t0"]
            blend = d["blend"][0] and d["blend"][2] == GL_ONE_MINUS_SRC_ALPHA
            additive = d["blend"][0] and d["blend"][2] == GL_ONE
            mkey = (t0["off"] if t0 else None, t0 and t0["fmt"], bool(blend), bool(additive),
                    d["alpha"][0], d["cull"][0])
            if mkey not in matcache:
                mat = {"name": "mat_%s" % (t0["off"] if t0 else "untextured"),
                       "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
                       "doubleSided": not d["cull"][0]}
                if t0:
                    png = texture_png(args.capdir, t0, texcache)
                    if png:
                        tk = ("idx", t0["off"], t0["fmt"])
                        if tk not in texcache:
                            texcache[tk] = g.image(png)
                        mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": texcache[tk]}
                if blend or additive:
                    mat["alphaMode"] = "BLEND"
                elif d["alpha"][0]:
                    mat["alphaMode"] = "MASK"
                    mat["alphaCutoff"] = max(d["alpha"][2], 1) / 255.0
                g.j["materials"].append(mat)
                matcache[mkey] = len(g.j["materials"]) - 1
            col = np.clip(d["col"], 0.0, 1.0)
            prim = {"attributes": {
                        "POSITION": g.accessor(c["pos"].astype(np.float32), "VEC3", 34962),
                        "TEXCOORD_0": g.accessor(c["st"].astype(np.float32), "VEC2", 34962),
                        "COLOR_0": g.accessor(col.astype(np.float32), "VEC4", 34962)},
                    "indices": g.accessor(np.array(c["tris"], dtype=np.uint32).ravel(), "SCALAR", 34963),
                    "material": matcache[mkey]}
            g.j["meshes"].append({"name": "vb%s_d%d" % (d["vb"], d["draw"]), "primitives": [prim]})
            node = {"name": "vb%s_f%d_d%d" % (d["vb"], d["frame"], d["draw"]),
                    "mesh": len(g.j["meshes"]) - 1}
            if c["trs"] is not None:
                node["translation"], node["rotation"], node["scale"] = c["trs"]
            else:
                posed += 1
            g.j["nodes"].append(node)
            root["children"].append(len(g.j["nodes"]) - 1)
            kept += 1

    g.write(args.out)
    print(f"wrote {args.out}: {kept} meshes ({posed} posed), "
          f"{len(g.j.get('materials', []))} materials, {len(g.j.get('images', []))} textures; "
          f"left out {skipped_rt} render-target draws, {skipped_other} other")


if __name__ == "__main__":
    main()
