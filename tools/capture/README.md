# Capture to glTF

Export what a game draws to glTF 2.0 (`.glb`), for Blender, Unity, Unreal or
Godot.

1. Start the game with `RECOMP_CAPTURE=<dir>` (an existing folder). Optional:
   - `RECOMP_CAPTURE_EVERY=<n>`: capture every n-th frame (default 1);
   - `RECOMP_CAPTURE_MINW=<w>`: only surfaces at least w pixels wide
     (default 640: the screen, not render targets).
2. Create `<dir>\capture.flag` where you want to capture, move the camera
   around, and delete the flag. Frames are captured while it exists, and the
   log says `[CAPTURE] started` / `stopped`. Each frame takes a few MB.
3. Convert:

   ```
   py -3 tools/capture/capture2gltf.py <dir> scene.glb
   ```

   It needs numpy and Pillow.

What you get:

- the drawn geometry with its textures (PNG), vertex colours and alpha mode;
- objects drawn with one rigid transform become model-space meshes under
  translation/rotation/scale nodes;
- planar and skinned draws are written in world space, as posed in the
  frame they come from;
- all frames are merged into one world space, anchored on a static object;
- each object is kept from the frame where it has the most instances, so
  moving objects leave no trails.

Limits:

- only what the game draws: culled or never-visited parts are missing, so
  fly around;
- no collision and no object placement data;
- lighting is baked into the vertex colours as the game computed it;
- render-target passes (reflections, full-screen effects) are left out.

The output is made from your own copy of the game. Do not redistribute it.
