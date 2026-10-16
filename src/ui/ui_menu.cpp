/*
 * ui_menu.cpp -- the in-game menu: Escape opens it over the game.
 *
 * Dear ImGui with its Win32 backend for input, and a small software
 * rasteriser of its own for output: ImGui's triangles are drawn into a
 * premultiplied overlay the size of the window, which the framebuffer window
 * composites over the frame. That works the same whether the frame reaches
 * the screen through GDI or the Vulkan swapchain, and needs no device of its
 * own. Everything here runs on the framebuffer window's thread.
 *
 * Sections: Controllers (per-port bindings, deadzones, which host pad drives
 * which port), Graphics (options other modules declare), Cheats (panels a
 * cheat engine adds), About.
 */
#include "xbox_ui.h"
#include "xbox_input_map.h"

#include <windows.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "imgui.h"
#include "imgui_impl_win32.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
extern "C" int xbox_NetplayActive(void);           /* src/net */
extern "C" int xbox_NetplayPlayer(void);
extern "C" int xbox_NetplayPlayers(void);
extern "C" int xbox_settings_enum(const char *section, int i, const char **key,
                                  const char **label, int *type, int *lo, int *hi,
                                  const char *const **names, int *n,
                                  unsigned *flags, int *env_set);

namespace {

volatile LONG s_open;
HWND s_hwnd;
bool s_ctx_ready;
int s_scale;                        /* integer UI scale for the window size */
std::vector<uint32_t> s_overlay;    /* back buffer: premultiplied 0xAARRGGBB */
int s_ow, s_oh;
/* What readers get (the window, a Vulkan present on another thread): the
 * last finished frame, swapped in under the lock, held shared by a reader
 * from xbox_ui_overlay() to xbox_ui_overlay_release(). */
std::vector<uint32_t> s_front;
int s_fw, s_fh;
SRWLOCK s_front_lock = SRWLOCK_INIT;
volatile LONG s_serial;
const unsigned char *s_tex;         /* font atlas, RGBA8 */
int s_tw, s_th;

/* "Press a key / a pad button" for a binding. */
int s_cap_port = -1, s_cap_ctl = -1, s_cap_pad;   /* s_cap_pad: 1 pad, 0 key */
int s_port = 0;                                   /* the port being edited */

struct Panel { const char *section, *title; void (*draw)(void *); void *ctx; };
std::vector<Panel> s_panels;

struct Deferred { void (*fn)(void *); void *ctx; };
std::vector<Deferred> s_defer;
SRWLOCK s_defer_lock = SRWLOCK_INIT;

char s_about[1024] = "";

/* ---- the rasteriser --------------------------------------------------- */

inline float edge(const ImVec2 &a, const ImVec2 &b, float x, float y)
{
    return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
}

/* One triangle of ImGui output: vertex colour times the font atlas (nearest),
 * blended "over" into the premultiplied overlay, inside the clip rectangle. */
void raster_tri(const ImDrawVert &v0, const ImDrawVert &v1, const ImDrawVert &v2,
                int cx0, int cy0, int cx1, int cy1)
{
    float area = edge(v0.pos, v1.pos, v2.pos.x, v2.pos.y);
    if (fabsf(area) < 1e-6f)
        return;
    int x0 = (int)floorf(fminf(v0.pos.x, fminf(v1.pos.x, v2.pos.x)));
    int y0 = (int)floorf(fminf(v0.pos.y, fminf(v1.pos.y, v2.pos.y)));
    int x1 = (int)ceilf(fmaxf(v0.pos.x, fmaxf(v1.pos.x, v2.pos.x)));
    int y1 = (int)ceilf(fmaxf(v0.pos.y, fmaxf(v1.pos.y, v2.pos.y)));
    if (x0 < cx0) x0 = cx0;
    if (y0 < cy0) y0 = cy0;
    if (x1 > cx1) x1 = cx1;
    if (y1 > cy1) y1 = cy1;
    if (x0 >= x1 || y0 >= y1)
        return;
    float inv = 1.0f / area;
    float c0[4], c1[4], c2[4];
    for (int k = 0; k < 4; k++) {
        c0[k] = (float)((v0.col >> (8 * k)) & 0xFF);
        c1[k] = (float)((v1.col >> (8 * k)) & 0xFF);
        c2[k] = (float)((v2.col >> (8 * k)) & 0xFF);
    }
    /* Solid fills sample the atlas's white texel; skip the lookup then. */
    bool flat_uv = v0.uv.x == v1.uv.x && v1.uv.x == v2.uv.x
                && v0.uv.y == v1.uv.y && v1.uv.y == v2.uv.y;
    for (int y = y0; y < y1; y++) {
        uint32_t *row = &s_overlay[(size_t)y * s_ow];
        float py = y + 0.5f;
        for (int x = x0; x < x1; x++) {
            float px = x + 0.5f;
            float w0 = edge(v1.pos, v2.pos, px, py) * inv;
            float w1 = edge(v2.pos, v0.pos, px, py) * inv;
            float w2 = 1.0f - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0)
                continue;
            float r = c0[0] * w0 + c1[0] * w1 + c2[0] * w2;
            float g = c0[1] * w0 + c1[1] * w1 + c2[1] * w2;
            float b = c0[2] * w0 + c1[2] * w1 + c2[2] * w2;
            float a = c0[3] * w0 + c1[3] * w1 + c2[3] * w2;
            float u = flat_uv ? v0.uv.x : v0.uv.x * w0 + v1.uv.x * w1 + v2.uv.x * w2;
            float v = flat_uv ? v0.uv.y : v0.uv.y * w0 + v1.uv.y * w1 + v2.uv.y * w2;
            int tx = (int)(u * s_tw), ty = (int)(v * s_th);
            if (tx < 0) tx = 0;
            if (ty < 0) ty = 0;
            if (tx >= s_tw) tx = s_tw - 1;
            if (ty >= s_th) ty = s_th - 1;
            const unsigned char *t = s_tex + ((size_t)ty * s_tw + tx) * 4;
            r *= t[0] / 255.0f;
            g *= t[1] / 255.0f;
            b *= t[2] / 255.0f;
            a *= t[3] / 255.0f;
            if (a < 0.5f)
                continue;
            float sa = a / 255.0f, ia = 1.0f - sa;
            uint32_t d = row[x];
            float da = (float)(d >> 24), dr = (float)((d >> 16) & 0xFF),
                  dg = (float)((d >> 8) & 0xFF), db = (float)(d & 0xFF);
            uint32_t oa = (uint32_t)(a + da * ia + 0.5f);
            uint32_t orr = (uint32_t)(r * sa + dr * ia + 0.5f);
            uint32_t og = (uint32_t)(g * sa + dg * ia + 0.5f);
            uint32_t ob = (uint32_t)(b * sa + db * ia + 0.5f);
            if (oa > 255) oa = 255;
            if (orr > 255) orr = 255;
            if (og > 255) og = 255;
            if (ob > 255) ob = 255;
            row[x] = (oa << 24) | (orr << 16) | (og << 8) | ob;
        }
    }
}

void raster(ImDrawData *dd)
{
    std::fill(s_overlay.begin(), s_overlay.end(), 0u);
    for (int n = 0; n < dd->CmdListsCount; n++) {
        const ImDrawList *cl = dd->CmdLists[n];
        const ImDrawVert *vb = cl->VtxBuffer.Data;
        const ImDrawIdx *ib = cl->IdxBuffer.Data;
        for (int c = 0; c < cl->CmdBuffer.Size; c++) {
            const ImDrawCmd &cmd = cl->CmdBuffer[c];
            if (cmd.UserCallback)
                continue;
            int cx0 = (int)(cmd.ClipRect.x - dd->DisplayPos.x);
            int cy0 = (int)(cmd.ClipRect.y - dd->DisplayPos.y);
            int cx1 = (int)ceilf(cmd.ClipRect.z - dd->DisplayPos.x);
            int cy1 = (int)ceilf(cmd.ClipRect.w - dd->DisplayPos.y);
            if (cx0 < 0) cx0 = 0;
            if (cy0 < 0) cy0 = 0;
            if (cx1 > s_ow) cx1 = s_ow;
            if (cy1 > s_oh) cy1 = s_oh;
            for (unsigned i = 0; i + 2 < cmd.ElemCount; i += 3) {
                const ImDrawIdx *p = ib + cmd.IdxOffset + i;
                raster_tri(vb[cmd.VtxOffset + p[0]], vb[cmd.VtxOffset + p[1]],
                           vb[cmd.VtxOffset + p[2]], cx0, cy0, cx1, cy1);
            }
        }
    }
}

/* ---- sections ------------------------------------------------------------ */

const char *key_name(int vk, char *buf, size_t n)
{
    if (!vk) {
        snprintf(buf, n, "-");
        return buf;
    }
    UINT sc = MapVirtualKeyA((UINT)vk, MAPVK_VK_TO_VSC);
    LONG lp = (LONG)(sc << 16);
    switch (vk) {                     /* extended keys GetKeyNameText needs */
    case VK_LEFT: case VK_RIGHT: case VK_UP: case VK_DOWN: case VK_PRIOR:
    case VK_NEXT: case VK_END: case VK_HOME: case VK_INSERT: case VK_DELETE:
    case VK_DIVIDE: case VK_NUMLOCK: case VK_RCONTROL: case VK_RMENU:
        lp |= 1 << 24;
    }
    if (!GetKeyNameTextA(lp, buf, (int)n))
        snprintf(buf, n, "key 0x%02X", vk);
    return buf;
}

void section_controllers()
{
    XboxPortMap m;
    bool changed = false;
    if (ImGui::BeginTabBar("ports")) {
        for (int p = 0; p < 4; p++) {
            char t[16];
            snprintf(t, sizeof t, "Port %d", p + 1);
            if (ImGui::BeginTabItem(t)) {
                s_port = p;
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
    xbox_input_map_get(s_port, &m);

    {
        static const char *const pads[] = { "None", "Host pad 1", "Host pad 2",
                                            "Host pad 3", "Host pad 4" };
        int cur = m.pad + 1;
        ImGui::SetNextItemWidth(160.0f * s_scale);
        if (ImGui::Combo("Host pad", &cur, pads, 5)) {
            m.pad = cur - 1;
            changed = true;
        }
        ImGui::SameLine();
        if (m.pad >= 0)
            ImGui::TextDisabled(xbox_input_host_pad_connected(m.pad)
                                ? "(connected)" : "(not connected)");
        bool kb = m.keyboard != 0;
        if (ImGui::Checkbox("Keyboard drives this port", &kb)) {
            m.keyboard = kb;
            changed = true;
        }
        const char *e = getenv("RECOMP_KEYBOARD");
        if (!(e && *e && *e != '0')) {
            ImGui::SameLine();
            ImGui::TextDisabled("(keyboard off: RECOMP_KEYBOARD unset)");
        }
        ImGui::SetNextItemWidth(220.0f * s_scale);
        changed |= ImGui::SliderInt("Left stick deadzone", &m.dz_left, 0, 32767);
        ImGui::SetNextItemWidth(220.0f * s_scale);
        changed |= ImGui::SliderInt("Right stick deadzone", &m.dz_right, 0, 32767);
        ImGui::SetNextItemWidth(220.0f * s_scale);
        changed |= ImGui::SliderInt("Trigger threshold", &m.trigger, 0, 255);
    }

    /* Waiting for a pad press to bind. */
    if (s_cap_pad && s_cap_port == s_port && s_cap_ctl >= 0) {
        int src = xbox_input_capture_source(m.pad);
        if (src != XSRC_NONE) {
            m.src[s_cap_ctl] = src;
            s_cap_ctl = -1;
            changed = true;
        }
    }

    ImGui::Separator();
    if (ImGui::BeginTable("bind", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY
                          | ImGuiTableFlags_BordersInnerV,
                          ImVec2(0, ImGui::GetContentRegionAvail().y - 40.0f * s_scale))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Xbox control");
        ImGui::TableSetupColumn("Keyboard");
        ImGui::TableSetupColumn("Host pad");
        ImGui::TableHeadersRow();
        for (int c = 0; c < XIN_CONTROLS; c++) {
            char kn[64], lab[96];
            ImGui::PushID(c);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(xbox_input_control_name(c));
            ImGui::TableNextColumn();
            bool cap_k = !s_cap_pad && s_cap_port == s_port && s_cap_ctl == c;
            snprintf(lab, sizeof lab, "%s###k", cap_k ? "press a key..." : key_name(m.key[c], kn, sizeof kn));
            if (ImGui::Button(lab, ImVec2(140.0f * s_scale, 0))) {
                s_cap_port = s_port; s_cap_ctl = c; s_cap_pad = 0;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("x##kx")) { m.key[c] = 0; changed = true; }
            ImGui::TableNextColumn();
            bool cap_p = s_cap_pad && s_cap_port == s_port && s_cap_ctl == c;
            snprintf(lab, sizeof lab, "%s###p", cap_p ? "press on the pad..." : xbox_input_source_name(m.src[c]));
            if (ImGui::Button(lab, ImVec2(140.0f * s_scale, 0))) {
                s_cap_port = s_port; s_cap_ctl = c; s_cap_pad = 1;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("x##px")) { m.src[c] = XSRC_NONE; changed = true; }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (ImGui::Button("Reset this port to defaults")) {
        xbox_input_map_reset(s_port);
        s_cap_ctl = -1;
        return;
    }
    if (changed)
        xbox_input_map_set(s_port, &m);
}

/* Options other modules declared in a section, as widgets. */
void declared_options(const char *section)
{
    const char *key, *label;
    const char *const *names;
    int type, lo, hi, n, env_set;
    unsigned flags;
    for (int i = 0; xbox_settings_enum(section, i, &key, &label, &type, &lo, &hi,
                                       &names, &n, &flags, &env_set); i++) {
        int v = xbox_setting_get(section, key, 0);
        bool ch = false;
        ImGui::PushID(i);
        ImGui::BeginDisabled(env_set != 0);
        if (type == 1) {                                  /* bool */
            bool b = v != 0;
            if (ImGui::Checkbox(label, &b)) { v = b; ch = true; }
        } else if (type == 2) {                           /* int */
            ImGui::SetNextItemWidth(220.0f * s_scale);
            ch = ImGui::SliderInt(label, &v, lo, hi);
        } else if (type == 3 && names) {                  /* choice */
            ImGui::SetNextItemWidth(220.0f * s_scale);
            ch = ImGui::Combo(label, &v, names, n);
        }
        ImGui::EndDisabled();
        if (env_set) {
            ImGui::SameLine();
            ImGui::TextDisabled("(set by environment)");
        } else if (flags & XBOX_SETTING_RESTART) {
            ImGui::SameLine();
            ImGui::TextDisabled("(after restart)");
        }
        if (ch) {
            xbox_setting_set(section, key, v);
            xbox_settings_save();
        }
        ImGui::PopID();
    }
}

int panels(const char *section)
{
    int shown = 0;
    for (const Panel &p : s_panels)
        if (!strcmp(p.section, section)) {
            if (p.title && ImGui::CollapsingHeader(p.title, ImGuiTreeNodeFlags_DefaultOpen))
                p.draw(p.ctx);
            else if (!p.title)
                p.draw(p.ctx);
            shown++;
        }
    return shown;
}

void section_graphics()
{
    declared_options("graphics");
    if (!panels("graphics")) {
        const char *k;
        const char *l;
        const char *const *nm;
        int t, lo, hi, n, es;
        unsigned f;
        if (!xbox_settings_enum("graphics", 0, &k, &l, &t, &lo, &hi, &nm, &n, &f, &es))
            ImGui::TextDisabled("No graphics options in this build yet.");
    }
}

void section_cheats()
{
    if (!xbox_ui_guest_writes_allowed())
        ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1),
                           "A replay (or a deterministic run) is running: cheats are off.");
    declared_options("cheats");
    if (!panels("cheats"))
        ImGui::TextDisabled("No cheat engine in this build yet.");
}

/* Network play (src/net): host or join a lockstep game. A network game
 * starts from power-on for everyone, so starting one restarts the game with
 * RECOMP_NET set; during one, this shows who is who. */
void start_network_game(int host, const char *addr, int port, int players, int delay)
{
    char v[96], exe[MAX_PATH], dir[MAX_PATH], *slash;
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    if (host)
        snprintf(v, sizeof v, "host:%d", port);
    else
        snprintf(v, sizeof v, "join:%s:%d", addr, port);
    SetEnvironmentVariableA("RECOMP_NET", v);
    snprintf(v, sizeof v, "%d", players);
    SetEnvironmentVariableA("RECOMP_NET_PLAYERS", v);
    snprintf(v, sizeof v, "%d", delay);
    SetEnvironmentVariableA("RECOMP_NET_DELAY", v);
    GetModuleFileNameA(NULL, exe, MAX_PATH);
    snprintf(dir, sizeof dir, "%s", exe);
    if ((slash = strrchr(dir, '\\')) != nullptr)
        *slash = 0;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    if (CreateProcessA(exe, nullptr, nullptr, nullptr, FALSE, 0, nullptr, dir, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        TerminateProcess(GetCurrentProcess(), 0);
    }
}

void section_network()
{
    static bool loaded;
    static int mode, port, players, delay;
    static char addr[64];
    bool changed = false;

    if (xbox_NetplayActive()) {
        ImGui::Text("Network game: you are player %d of %d.", xbox_NetplayPlayer() + 1,
                    xbox_NetplayPlayers());
        ImGui::TextWrapped("Every player runs the whole game and only pad input travels, "
                           "so everyone sees the same game. If the games ever drift apart, "
                           "the game stops and says so (netplay_report.txt next to the exe). "
                           "Your own saves are not touched: a network game plays on a copy.");
        return;
    }
    if (!loaded) {
        mode = xbox_setting_get("network", "mode", 0);
        port = xbox_setting_get("network", "port", 27350);
        players = xbox_setting_get("network", "players", 2);
        delay = xbox_setting_get("network", "delay", 3);
        xbox_setting_get_str("network", "host", "192.168.1.2", addr, sizeof addr);
        loaded = true;
    }
    changed |= ImGui::RadioButton("Host a game", &mode, 0);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Join a game", &mode, 1);
    ImGui::Separator();
    ImGui::SetNextItemWidth(160.0f * s_scale);
    changed |= ImGui::InputInt("Port (UDP and TCP)", &port);
    if (port < 1024) port = 1024;
    if (port > 65535) port = 65535;
    if (mode == 0) {
        ImGui::SetNextItemWidth(160.0f * s_scale);
        changed |= ImGui::SliderInt("Players", &players, 2, 4);
        ImGui::SetNextItemWidth(160.0f * s_scale);
        changed |= ImGui::SliderInt("Input delay (frames)", &delay, 1, 10);
        ImGui::TextWrapped("You are player 1. Others join with your IP address and this "
                           "port (allow it through the firewall). Your saves are sent to "
                           "every player for this game. More delay hides more lag; 3 "
                           "frames (50 ms) suits a LAN or a VPN.");
    } else {
        ImGui::SetNextItemWidth(220.0f * s_scale);
        changed |= ImGui::InputText("Host's address", addr, sizeof addr);
        ImGui::TextWrapped("The host decides the number of players and the delay, and "
                           "sends its saves. You get the next free controller port.");
    }
    if (changed) {
        xbox_setting_set("network", "mode", mode);
        xbox_setting_set("network", "port", port);
        xbox_setting_set("network", "players", players);
        xbox_setting_set("network", "delay", delay);
        xbox_setting_set_str("network", "host", addr);
        xbox_settings_save();
    }
    ImGui::Separator();
    ImGui::TextWrapped("Starting restarts the game: everyone begins from power-on together, "
                       "and the host's game waits until every player has joined.");
    if (ImGui::Button(mode == 0 ? "Start network game (host)" : "Join network game"))
        start_network_game(mode == 0, addr, port, players, delay);
}

void section_about()
{
    if (s_about[0])
        ImGui::TextWrapped("%s", s_about);
    ImGui::Separator();
    ImGui::TextWrapped("Escape opens and closes this menu; while it is open the game "
                       "sees no input. Settings are saved next to the game and load "
                       "at start; an environment variable named for a setting "
                       "overrides it.");
    ImGui::TextWrapped("Alt+Enter or F11: fullscreen.  F9: smooth / sharp scaling.");
    ImGui::TextWrapped("Built with Dear ImGui %s (MIT licence).", IMGUI_VERSION);
}

void draw_menu(int w, int h)
{
    float mw = fminf(w * 0.86f, 900.0f * s_scale), mh = fminf(h * 0.86f, 680.0f * s_scale);
    ImGui::SetNextWindowPos(ImVec2((w - mw) * 0.5f, (h - mh) * 0.5f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(mw, mh), ImGuiCond_Always);
    ImGui::Begin("Menu  (Esc to return to the game)", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize
                 | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
    if (ImGui::BeginTabBar("sections")) {
        if (ImGui::BeginTabItem("Controllers")) { section_controllers(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Graphics"))    { section_graphics();    ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Cheats"))      { section_cheats();      ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Network"))     { section_network();     ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("About"))       { section_about();       ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void ensure_context(HWND hwnd, int h)
{
    int scale = h >= 1400 ? 3 : h >= 900 ? 2 : 1;
    if (!s_ctx_ready) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        io.IniFilename = nullptr;               /* no imgui.ini next to the exe */
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard
                        | ImGuiConfigFlags_NavEnableGamepad;
        ImGui_ImplWin32_Init(hwnd);
        unsigned char *px;
        io.Fonts->GetTexDataAsRGBA32(&px, &s_tw, &s_th);
        io.Fonts->SetTexID((ImTextureID)1);
        s_tex = px;
        s_hwnd = hwnd;
        s_ctx_ready = true;
        s_scale = 0;
    }
    if (scale != s_scale) {
        ImGui::GetStyle() = ImGuiStyle();
        ImGui::StyleColorsDark();
        ImGui::GetStyle().WindowRounding = 4.0f;
        ImGui::GetStyle().ScaleAllSizes((float)scale);
        ImGui::GetIO().FontGlobalScale = (float)scale;
        s_scale = scale;
    }
}

void set_open(bool open)
{
    InterlockedExchange(&s_open, open ? 1 : 0);
    xbox_input_set_blocked(open ? 1 : 0);
    s_cap_ctl = -1;
    if (!open)
        InterlockedIncrement(&s_serial);
}

} // namespace

/* ---- C API ------------------------------------------------------------------ */

extern "C" {

int xbox_ui_menu_open(void) { return s_open != 0; }

void xbox_ui_menu_toggle(void) { set_open(!s_open); }

int xbox_ui_wndproc(void *hwnd, unsigned msg, uintptr_t wparam, intptr_t lparam)
{
    HWND h = (HWND)hwnd;
    bool key = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
    if (key && wparam == VK_ESCAPE && !(lparam & (1 << 30))) {
        if (s_open && s_cap_ctl >= 0)
            s_cap_ctl = -1;                     /* cancel a "press a key" */
        else
            set_open(!s_open);
        return 1;
    }
    if (!s_open)
        return 0;
    /* Binding a key: the next key pressed is it. */
    if (key && s_cap_ctl >= 0 && !s_cap_pad && !(lparam & (1 << 30))) {
        XboxPortMap m;
        int vk = (int)wparam;
        if (vk == VK_SHIFT || vk == VK_CONTROL || vk == VK_MENU || vk < 256) {
            xbox_input_map_get(s_cap_port, &m);
            m.key[s_cap_ctl] = vk & 0xFF;
            xbox_input_map_set(s_cap_port, &m);
        }
        s_cap_ctl = -1;
        return 1;
    }
    /* Mouse messages posted to a window that is not in front (agents
     * driving the menu with PostMessage while the human uses another app):
     * the backend answers WM_MOUSEMOVE with TrackMouseEvent, and with the
     * real cursor elsewhere Windows sends WM_MOUSELEAVE at once, so the
     * click lands nowhere. A window that is not in front keeps its last
     * mouse position instead; in front, nothing changes. */
    if (msg == WM_MOUSELEAVE && GetForegroundWindow() != h)
        return 1;
    if (s_ctx_ready)
        ImGui_ImplWin32_WndProcHandler(h, msg, (WPARAM)wparam, (LPARAM)lparam);
    switch (msg) {                              /* input is the menu's now */
    case WM_KEYDOWN: case WM_KEYUP: case WM_CHAR:
    case WM_SYSCHAR:
    case WM_MOUSEMOVE: case WM_LBUTTONDOWN: case WM_LBUTTONUP:
    case WM_RBUTTONDOWN: case WM_RBUTTONUP: case WM_MBUTTONDOWN:
    case WM_MBUTTONUP: case WM_MOUSEWHEEL: case WM_MOUSEHWHEEL:
    case WM_LBUTTONDBLCLK:
        return 1;
    }
    return 0;
}

int xbox_ui_render(void *hwnd, int w, int h)
{
    if (!s_open || w <= 0 || h <= 0)
        return 0;
    ensure_context((HWND)hwnd, h);
    if (w != s_ow || h != s_oh) {
        s_ow = w;
        s_oh = h;
        s_overlay.assign((size_t)w * h, 0u);
    }
    ImGui_ImplWin32_NewFrame();
    ImGui::GetIO().DisplaySize = ImVec2((float)w, (float)h);
    ImGui::NewFrame();
    draw_menu(w, h);
    ImGui::Render();
    raster(ImGui::GetDrawData());
    AcquireSRWLockExclusive(&s_front_lock);
    s_front.swap(s_overlay);
    std::swap(s_fw, s_ow);
    std::swap(s_fh, s_oh);
    InterlockedIncrement(&s_serial);
    ReleaseSRWLockExclusive(&s_front_lock);
    return 1;
}

int xbox_ui_overlay(const uint32_t **px, int *w, int *h, uint32_t *serial)
{
    if (!s_open)
        return 0;
    AcquireSRWLockShared(&s_front_lock);
    if (s_front.empty() || !s_open) {
        ReleaseSRWLockShared(&s_front_lock);
        return 0;
    }
    *px = s_front.data();
    *w = s_fw;
    *h = s_fh;
    if (serial)
        *serial = (uint32_t)s_serial;
    return 1;                           /* held until xbox_ui_overlay_release */
}

void xbox_ui_overlay_release(void)
{
    ReleaseSRWLockShared(&s_front_lock);
}

void xbox_ui_add_panel(const char *section, const char *title,
                       void (*draw)(void *ctx), void *ctx)
{
    s_panels.push_back(Panel{ section, title, draw, ctx });
}

int xbox_ui_button(const char *label) { return ImGui::Button(label) ? 1 : 0; }

int xbox_ui_checkbox(const char *label, int *v)
{
    bool b = *v != 0;
    if (!ImGui::Checkbox(label, &b))
        return 0;
    *v = b;
    return 1;
}

int xbox_ui_slider_int(const char *label, int *v, int lo, int hi)
{
    return ImGui::SliderInt(label, v, lo, hi) ? 1 : 0;
}

int xbox_ui_input_int(const char *label, int *v)
{
    return ImGui::InputInt(label, v) ? 1 : 0;
}

int xbox_ui_combo(const char *label, int *cur, const char *const *items, int n)
{
    return ImGui::Combo(label, cur, items, n) ? 1 : 0;
}

void xbox_ui_text(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    ImGui::TextWrappedV(fmt, ap);
    va_end(ap);
}

void xbox_ui_separator(void) { ImGui::Separator(); }
void xbox_ui_same_line(void) { ImGui::SameLine(); }

void xbox_ui_defer(void (*fn)(void *ctx), void *ctx)
{
    AcquireSRWLockExclusive(&s_defer_lock);
    s_defer.push_back(Deferred{ fn, ctx });
    ReleaseSRWLockExclusive(&s_defer_lock);
}

void xbox_ui_run_deferred(void)
{
    std::vector<Deferred> run;
    if (s_defer.empty())
        return;
    AcquireSRWLockExclusive(&s_defer_lock);
    run.swap(s_defer);
    ReleaseSRWLockExclusive(&s_defer_lock);
    for (const Deferred &d : run)
        d.fn(d.ctx);
}

/* Not during a replay or a deterministic (virtual-mode) run. A recording is
 * fine: the cheat engine writes each applied cheat into the input recording,
 * and a replay applies it again at the same poll. */
int xbox_ui_guest_writes_allowed(void)
{
    extern int xbox_SchedReplaying(void);
    return !xbox_SchedReplaying();
}

void xbox_ui_set_about(const char *text)
{
    snprintf(s_about, sizeof s_about, "%s", text ? text : "");
}

} // extern "C"
