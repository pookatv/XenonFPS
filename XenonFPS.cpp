// XenonFPS.xex: FPS overlay drawn by the GPU
//
// How it works:
//   1. A Worker thread patches the first 4 instructions of VdSwap (kernel) to jump to HookVdSwap
//   2. HookVdSwap counts frames, calls the real VdSwap through a trampoline, then edits the
//      GPU ring buffer: it replaces the ring's NOP area with an INDIRECT_BUFFER packet that
//      points at a command stream of MEM_WRITE packets. The GPU runs those writes and the
//      64x32 box (FPS digits) lands in the front buffer, in order with the game's own commands
//   3. The command streams are cached per (front buffer, image version), so on a normal frame
//      the hook only rewrites 4 ring dwords. A stream is rebuilt only when the FPS number
//      changes or a front buffer is seen for the first time
//   4. The Worker measures FPS twice a second and builds a new box image into the spare
//      one of two images, then flips the version number, so the hook never reads a half-built image
//
// Diagnostics: read g_diag with getmem (search for the magic 0xD1A60100, see the D_* indices)

#include <xtl.h>

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------

// Box position: pick a corner and, if your TV crops the edges, a margin
#define CORNER_TOP_LEFT       0
#define CORNER_TOP_RIGHT      1
#define CORNER_BOTTOM_LEFT    2
#define CORNER_BOTTOM_RIGHT   3

#define BOX_CORNER            CORNER_TOP_LEFT   // <-- change this to move the box
#define BOX_MARGIN_TILES      0                 // move the box inward from the edges

// The corner is resolved per frame against the frame's real width and height (see HookVdSwap).
#if   BOX_CORNER == CORNER_TOP_LEFT
  #define BOX_AT_RIGHT    0
  #define BOX_AT_BOTTOM   0
#elif BOX_CORNER == CORNER_TOP_RIGHT
  #define BOX_AT_RIGHT    1
  #define BOX_AT_BOTTOM   0
#elif BOX_CORNER == CORNER_BOTTOM_LEFT
  #define BOX_AT_RIGHT    0
  #define BOX_AT_BOTTOM   1
#elif BOX_CORNER == CORNER_BOTTOM_RIGHT
  #define BOX_AT_RIGHT    1
  #define BOX_AT_BOTTOM   1
#else
  #error BOX_CORNER must be one of the CORNER_* values
#endif

#if BOX_MARGIN_TILES < 0 || BOX_MARGIN_TILES > 10
  #error BOX_MARGIN_TILES must be 0..10
#endif

// The box image is always built at tile (0,0): the layout inside a tile does not depend on where
// the tile is, so the same image works at every position and every surface width. Where the box
// is drawn is worked out per frame from that frame's own size and pitch.
#define BOX_X0          0
#define BOX_Y0          0
#define BOX_TILE_OFF    0

#define BOX_DWORDS      2048                          // the box is 2 tiles = 8192 bytes = 2048 dwords
#define STREAM_DWORDS   (BOX_DWORDS * 3)              // one MEM_WRITE (3 dwords) per box dword
#define SLOT_BYTES      (STREAM_DWORDS * 4)           // 0x6000, a multiple of 128

// Black background trimmed by this many pixels on each side of the 64x32 box (0 = the full box)
// Only the pixels inside the trimmed rectangle are written each frame
#define BOX_PAD_X       8           // 0..15; the digits need at least 10
#define BOX_PAD_Y       4           // 0..5; the digits need at least 6
#if BOX_PAD_X < 0 || BOX_PAD_X > 10 || BOX_PAD_Y < 0 || BOX_PAD_Y > 6
  #error BOX_PAD_X must be 0..10 and BOX_PAD_Y 0..6 so the digits are not clipped
#endif
#define FB_ENTRIES      4                             // front buffers remembered
#define SLOT_COUNT      (FB_ENTRIES * 2)              // two image versions per buffer
#define START_DELAY_MS  20000                         // let video come up before touching VdSwap

// Controller toggle: press these buttons together to hide or show the overlay (any of the 4 pads).
// Buttons: XINPUT_GAMEPAD_DPAD_UP / DOWN / LEFT / RIGHT, START, BACK, LEFT_THUMB, RIGHT_THUMB
// (the stick clicks), LEFT_SHOULDER, RIGHT_SHOULDER, A, B, X, Y. Join them with |.
#define TOGGLE_BUTTONS    (XINPUT_GAMEPAD_RIGHT_THUMB | XINPUT_GAMEPAD_DPAD_UP)
#define OVERLAY_START_ON  1         // 1 = shown at start, 0 = hidden until toggled on
#define INPUT_POLL_MS     50        // how often the pads are checked (the FPS update rate is unchanged)

typedef LONG NTSTATUS;

extern "C" {
    NTSTATUS ExCreateThread(PHANDLE, DWORD, LPDWORD, PVOID, LPTHREAD_START_ROUTINE, LPVOID, DWORD);
    NTSTATUS KeDelayExecutionThread(CHAR WaitMode, BOOL Alertable, PLARGE_INTEGER Interval);
    VOID XapiThreadStartup(VOID (__cdecl *)(VOID *), PVOID, DWORD);
}

// Kernel-only sleep. Do not use Sleep(): it needs xapi runtime state
static VOID DelayMs(DWORD ms)
{
    LARGE_INTEGER t;
    t.QuadPart = -(LONGLONG)ms * 10000;   // negative = relative, 100 ns units
    KeDelayExecutionThread(0, FALSE, &t); // 0 = KernelMode
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

// Diagnostics block.
enum {
    D_MAGIC,      // 0xD1A60100
    D_STATE,      // 1 = worker started, 2 = command memory allocated
    D_PATCH,      // 0xBAD = VdSwap did not match (nothing patched), 2 = trampoline ready, 3 = patch installed
    D_HOOKADDR,   // address of HookVdSwap
    D_TRAMP,      // address of the trampoline
    D_FRAMES,     // VdSwap calls seen
    D_FPS,        // last measured FPS
    D_QUEUED,     // frames that got the box
    D_SKIPPED,    // frames where the gate did not match (start-up, menus, unknown layouts)
    D_FB,         // last front buffer
    D_BUILDS,     // box image builds
    D_STREAMS,    // command streams built (cache misses)
    D_VERSION,    // current image version
    D_CMDVA,      // command memory, CPU address
    D_CMDPHYS,    // command memory, physical address
    D_REJ_GATE,   // frames rejected by the 1280x720 32-bit check
    D_LAST_D0,    // texture fetch dword 0 of the last rejected frame
    D_LAST_D1,    // dword 1
    D_LAST_D2,    // dword 2
    D_LAST_FB,    // front buffer address of the last rejected frame
    D_REJ_PTR,    // frames passed but the ring pointer was unaligned or below 0xA0000000
    D_LAST_A,     // last ring pointer seen
    D_REJ_NOCMD,  // skipped: no command memory
    D_REJ_NOIMG,  // skipped: no box image yet
    D_REJ_FB,     // skipped: front buffer below 0xE0000000
    D_REJ_RING,   // skipped: ring signature did not match
    D_LAST_RB0,   // rb[0] at the last ring mismatch
    D_LAST_RB4,   // rb[4] at the last ring mismatch
    D_EXC,        // exceptions inside the hook
    D_OK_D1,      // texture fetch dword 1 of the last accepted frame (format and endian)
    D_BAD_A,      // ring pointer at the last ring mismatch
    D_BAD0,       // rb[0..15] at the last ring mismatch
    D_BAD_LAST = D_BAD0 + 15,
    D_ENABLED,    // 1 = overlay shown, 0 = hidden
    D_TOGGLES,    // times the combo has toggled the overlay
    D_PADS,       // bitmask of controllers XInput can see (bit 0 = pad 1)
    D_HIDDEN,     // frames that would have got the box but the overlay was hidden
    D_COUNT
};
static volatile DWORD g_diag[D_COUNT] = { 0xD1A60100 };

__declspec(align(128)) static volatile DWORD g_tramp[32];

// Command memory: SLOT_COUNT streams, GPU readable
static volatile DWORD g_cmdVa   = 0;
static volatile DWORD g_cmdPhys = 0;

// Stream cache. Slot = bufferEntry * 2 + (version & 1). A slot is valid when its tags match
static DWORD g_fbTab[FB_ENTRIES];
static DWORD g_fbNext = 0;
static DWORD g_slotFb[SLOT_COUNT];
static DWORD g_slotVer[SLOT_COUNT];
static DWORD g_slotOff[SLOT_COUNT];     // where in the buffer the stream draws, so a mode change rebuilds it

// Which box dwords get written each frame (indices into the box image), built once by the Worker
static DWORD g_writeIdx[BOX_DWORDS];
static volatile DWORD g_writeCount = 0;

// Two box images. The Worker builds into the one that is not current, then bumps g_boxVer
// Current image = g_boxImgs[g_boxVer & 1]. g_boxVer = 0 means no image yet
__declspec(align(128)) static DWORD g_boxImgs[2][BOX_DWORDS];
static volatile DWORD g_boxVer = 0;

// Overlay on/off, flipped by the controller combo (see PollToggle)
static volatile DWORD g_overlayOn = OVERLAY_START_ON;
static bool g_comboWasHeld = false;

// CPU address -> physical address as the GPU sees it. 0 if the range is not one we understand
// 0xA0000000-0xDFFFFFFF is a plain mask; 0xE0000000+ has the extra 0x1000 the front buffer needs
static DWORD PhysOf(DWORD va)
{
    if (va >= 0xA0000000u && va < 0xE0000000u) return va & 0x1FFFFFFFu;
    if (va >= 0xE0000000u)                     return (va & 0x1FFFFFFFu) + 0x1000u;
    return 0;
}

// Reads a dword only from kernel/data space we trust. 0xDEADDEAD otherwise
static DWORD Peek(PVOID p)
{
    DWORD a = (DWORD)p;
    if ((a & 3) || a < 0x70000000 || a >= 0x80000000) return 0xDEADDEAD;
    return *(volatile DWORD *)p;
}

// ---------------------------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------------------------

typedef DWORD (*VdSwapFn)(PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID);

// Builds (or reuses) the command stream for this buffer and image version, and returns its slot
// Returns -1 if it cannot be used
static int StreamFor(DWORD fb, DWORD ver, DWORD boxOff)
{
    // Find or assign this front buffer's entry
    DWORD e = 0;
    for (; e < FB_ENTRIES; e++) if (g_fbTab[e] == fb) break;
    if (e == FB_ENTRIES) {
        e = g_fbNext++ % FB_ENTRIES;
        g_fbTab[e] = fb;
    }
    DWORD slot = e * 2 + (ver & 1);

    if (g_slotFb[slot] != fb || g_slotVer[slot] != ver || g_slotOff[slot] != boxOff) {
        volatile DWORD *cb = (volatile DWORD *)(g_cmdVa + slot * SLOT_BYTES);
        const DWORD *img   = g_boxImgs[ver & 1];
        DWORD x = PhysOf(fb + boxOff);
        DWORD n = 0;
        DWORD cnt = g_writeCount;
        for (DWORD w = 0; w < cnt; w++) {
            DWORD k = g_writeIdx[w];
            cb[n++] = 0xC0013D00;           // MEM_WRITE, one data dword
            cb[n++] = (x + k * 4) | 2;      // physical address | endian 8in32
            cb[n++] = img[k];
        }
        for (DWORD o = 0; o < n; o += 32)   // the GPU reads RAM, not our cache: flush every 128-byte line
            __dcbf(0, (void *)&cb[o]);
        __sync();
        g_slotFb[slot]  = fb;
        g_slotVer[slot] = ver;
        g_slotOff[slot] = boxOff;
        g_diag[D_STREAMS] = g_diag[D_STREAMS] + 1;
    }
    return (int)slot;
}

static DWORD HookVdSwap(PVOID a, PVOID b, PVOID c, PVOID d, PVOID e, PVOID f, PVOID g, PVOID h, PVOID i, PVOID j)
{
    g_diag[D_FRAMES] = g_diag[D_FRAMES] + 1;

    // Describe the buffer the game is presenting: f -> front buffer address, b -> texture fetch constant
    DWORD fb    = Peek(f);
    DWORD d0    = Peek(b);
    DWORD d1    = Peek((PBYTE)b + 4);
    DWORD d2    = Peek((PBYTE)b + 8);
    DWORD pitch = ((d0 >> 22) & 0x1FF) * 32;
    DWORD fmt   = d1 & 0x3F;
    DWORD wid   = (d2 & 0x1FFF) + 1;
    DWORD hgt   = ((d2 >> 13) & 0x1FFF) + 1;

    // Only handle 32-bit frame buffers of a size we have seen: 1280x720 and 960x720.
    // To add a mode, read its width/height/pitch with Read-Diag and add it here.
    bool ok = ((d0 >> 31) == 1 && (fmt == 6 || fmt == 54) && pitch == wid &&
               ((wid == 1280 && hgt == 720) || (wid == 960 && hgt == 720)) &&
               fb >= 0xA0000000 && (fb & 0xFFF) == 0 && fb == (d1 & 0xFFFFF000));

    // Byte offset of the box inside this frame buffer: the corner and margin, in whole 32x32 tiles.
    DWORD boxOff = 0;
    if (ok) {
        DWORD cols = wid >> 5;                   // tiles across (40 for 1280, 30 for 960)
        DWORD rows = hgt >> 5;                   // whole tile rows (22 for 720)
        DWORD tpr  = pitch >> 5;                 // tiles per row in memory
        DWORD col  = BOX_AT_RIGHT  ? (cols - 2 - BOX_MARGIN_TILES) : BOX_MARGIN_TILES;
        DWORD row  = BOX_AT_BOTTOM ? (rows - 2 - BOX_MARGIN_TILES) : BOX_MARGIN_TILES;
        boxOff = (row * tpr + col) << 12;        // 4096 bytes per tile
    }

    if (!ok) {
        g_diag[D_REJ_GATE] = g_diag[D_REJ_GATE] + 1;
        g_diag[D_LAST_D0]  = d0;
        g_diag[D_LAST_D1]  = d1;
        g_diag[D_LAST_D2]  = d2;
        g_diag[D_LAST_FB]  = fb;
    } else {
        g_diag[D_OK_D1]    = d1;            // format and endian of the last accepted frame
    }

    DWORD ret = ((VdSwapFn)(PVOID)&g_tramp[0])(a, b, c, d, e, f, g, h, i, j);

    // a = the GPU ring: the kernel's fence packet at rb[0..3], a 56-dword NOP area at rb[4..63]
    if (ok && !g_overlayOn) {
        g_diag[D_HIDDEN] = g_diag[D_HIDDEN] + 1;
    } else if (ok && (((DWORD)a) & 3) == 0 && ((DWORD)a) >= 0xA0000000) {
        __try {
            volatile DWORD *rb = (volatile DWORD *)a;
            DWORD ver = g_boxVer;           // read once: the image index and tags must agree
            DWORD r0  = rb[0];
            DWORD r4  = rb[4];
            if (g_cmdPhys && ver && fb >= 0xA0000000u &&
                r0 == 0xC0025800 && r4 == 0xC03A1000) {
                int slot = StreamFor(fb, ver, boxOff);
                rb[4] = 0xC0013F00;                         // INDIRECT_BUFFER
                rb[5] = g_cmdPhys + (DWORD)slot * SLOT_BYTES;
                rb[6] = g_writeCount * 3;                   // 3 dwords per MEM_WRITE
                rb[7] = 0xC0371000;                         // NOP over the rest: 56 payload dwords (rb[8]..rb[63])
                __sync();
                g_diag[D_FB]     = fb;
                g_diag[D_QUEUED] = g_diag[D_QUEUED] + 1;
            } else {
                g_diag[D_SKIPPED] = g_diag[D_SKIPPED] + 1;
                if (!g_cmdPhys)                g_diag[D_REJ_NOCMD] = g_diag[D_REJ_NOCMD] + 1;
                else if (!ver)                 g_diag[D_REJ_NOIMG] = g_diag[D_REJ_NOIMG] + 1;
                else if (fb < 0xA0000000u)     g_diag[D_REJ_FB]    = g_diag[D_REJ_FB] + 1;
                else {
                    g_diag[D_REJ_RING] = g_diag[D_REJ_RING] + 1;
                    g_diag[D_LAST_RB0] = r0;
                    g_diag[D_LAST_RB4] = r4;
                    g_diag[D_BAD_A]    = (DWORD)a;
                    for (DWORD q = 0; q < 16; q++)
                        g_diag[D_BAD0 + q] = rb[q];
                }
            }
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            g_diag[D_SKIPPED] = g_diag[D_SKIPPED] + 1;
            g_diag[D_EXC]     = g_diag[D_EXC] + 1;
        }
    } else if (ok) {
        g_diag[D_REJ_PTR] = g_diag[D_REJ_PTR] + 1;
        g_diag[D_LAST_A]  = (DWORD)a;
    }
    return ret;
}

// ---------------------------------------------------------------------------------------------
// Box image
// ---------------------------------------------------------------------------------------------

static const DWORD kDigits[10] = {
    0x7B6F, 0x2C97, 0x73E7, 0x73CF, 0x5BC9, 0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BCF
};

// Byte offset of pixel (x, y) in a 1280-wide, 32-bit tiled surface (Xenia texture_address.h Tiled2D)
static DWORD TiledOff(DWORD x, DWORD y)
{
    DWORD outer = ((y >> 5) * 40 + (x >> 5)) << 6;
    DWORD inner = (((y >> 1) & 7) << 3) | (x & 7);
    DWORD oib   = (outer | inner) << 2;
    DWORD bank  = (y >> 4) & 1;
    DWORD pipe  = ((x >> 3) & 3) ^ (((y >> 3) & 1) << 1);
    return ((y & 1) << 4) | (pipe << 6) | (bank << 11)
         | (oib & 15) | (((oib >> 4) & 1) << 5) | (((oib >> 5) & 7) << 8)
         | ((oib >> 8) << 12);
}

// Draws the 64x32 box (black background, white digits) into a surface at address buf
static void DrawBox(DWORD buf, DWORD fps)
{
    DWORD d[3]  = { (fps / 100) % 10, (fps / 10) % 10, fps % 10 };
    DWORD first = (fps >= 100) ? 0 : ((fps >= 10) ? 1 : 2);   // skip leading zeros
    DWORD n     = 3 - first;                                  // digits shown
    DWORD tw    = n * 16 - 4;                                 // text width (12 px digits, 4 px gaps)
    DWORD sx    = BOX_X0 + (64 - tw) / 2;                     // centered in the 64 px box

    for (DWORD y = BOX_Y0; y < BOX_Y0 + 32; y++) {
        for (DWORD x = BOX_X0; x < BOX_X0 + 64; x++) {
            DWORD col = 0x00000003;                           // black (B10 G10 R10 A2 layout)
            if (y >= BOX_Y0 + 6 && y < BOX_Y0 + 26) {
                DWORD row = (y - (BOX_Y0 + 6)) >> 2;
                for (DWORD k = first; k < 3; k++) {
                    DWORD x0 = sx + (k - first) * 16;
                    if (x >= x0 && x < x0 + 12) {
                        DWORD c = (x - x0) >> 2;
                        if ((kDigits[d[k]] >> (14 - (row * 3 + c))) & 1) col = 0xFFFFFFFF;
                    }
                }
            }
            *(volatile DWORD *)(buf + TiledOff(x, y)) = col;
        }
    }
}

// Builds the image into the spare slot by running DrawBox against the array instead of a frame buffer
static void BuildBoxImage(DWORD fps)
{
    DWORD ver = g_boxVer;
    DWORD nxt = (ver + 1) & 1;
    DrawBox((DWORD)&g_boxImgs[nxt][0] - BOX_TILE_OFF, fps);
    __sync();
    g_boxVer = ver + 1;
    g_diag[D_VERSION] = ver + 1;
    g_diag[D_BUILDS]  = g_diag[D_BUILDS] + 1;
}

// Lists the box dwords inside the trimmed rectangle. Uses the same TiledOff() math as DrawBox,
// so each index points at the dword of the box image that DrawBox filled for that pixel
static void BuildWriteList()
{
    DWORD cnt = 0;
    for (DWORD y = BOX_Y0 + BOX_PAD_Y; y < BOX_Y0 + 32 - BOX_PAD_Y; y++) {
        for (DWORD x = BOX_X0 + BOX_PAD_X; x < BOX_X0 + 64 - BOX_PAD_X; x++) {
            DWORD k = (TiledOff(x, y) - BOX_TILE_OFF) >> 2;
            if (k < BOX_DWORDS) g_writeIdx[cnt++] = k;
        }
    }
    g_writeCount = cnt;
}

// ---------------------------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------------------------

// Patches the first 4 instructions of VdSwap. Refuses unless they match what was dumped
static void InstallHook()
{
    volatile DWORD *kp = (volatile DWORD *)0x800C8B90;

    if (kp[0] != 0x7D8802A6 || kp[1] != 0x48044021 ||
        kp[2] != 0x9421FF40 || kp[3] != 0x3D608017) {
        g_diag[D_PATCH] = 0xBAD;
        return;
    }

    // Cache-flush helper at g_tramp[16..19]: icbi 0,r3 ; sync ; isync ; blr
    g_tramp[16] = 0x7C001FAC;
    g_tramp[17] = 0x7C0004AC;
    g_tramp[18] = 0x4C00012C;
    g_tramp[19] = 0x4E800020;

    // Trampoline: the 4 overwritten instructions replayed, then a jump back to 0x800C8BA0
    g_tramp[0]  = 0x7D8802A6;   // mflr r12
    g_tramp[1]  = 0x3D608010;   // lis r11,0x8010
    g_tramp[2]  = 0x616BCBB4;   // ori r11,r11,0xCBB4  (register-save helper)
    g_tramp[3]  = 0x7D6903A6;   // mtctr r11
    g_tramp[4]  = 0x4E800421;   // bctrl  (replaces the original bl)
    g_tramp[5]  = 0x9421FF40;   // stwu r1,-0xC0(r1)
    g_tramp[6]  = 0x3D608017;   // lis r11,0x8017
    g_tramp[7]  = 0x3D80800C;   // lis r12,0x800C
    g_tramp[8]  = 0x618C8BA0;   // ori r12,r12,0x8BA0
    g_tramp[9]  = 0x7D8903A6;   // mtctr r12
    g_tramp[10] = 0x4E800420;   // bctr
    __dcbst(0, (void *)&g_tramp[0]);
    __sync();

    typedef VOID (*FlushFn)(PVOID);
    FlushFn flush = (FlushFn)(PVOID)&g_tramp[16];
    flush((PVOID)&g_tramp[0]);

    DWORD hk = (DWORD)(PVOID)&HookVdSwap;
    g_diag[D_HOOKADDR] = hk;
    g_diag[D_TRAMP]    = (DWORD)&g_tramp[0];
    g_diag[D_PATCH]    = 2; // trampoline ready, about to patch

    kp[1] = 0x616B0000 | (hk & 0xFFFF);    // ori r11,r11,lo
    kp[2] = 0x7D6903A6;                    // mtctr r11
    kp[3] = 0x4E800420;                    // bctr
    kp[0] = 0x3D600000 | (hk >> 16);       // lis r11,hi  (written last)
    __dcbst(0, (void *)kp);
    __sync();
    flush((PVOID)kp);

    g_diag[D_PATCH] = 3; // patch installed
}

// Checks the controllers for the toggle combo and flips the overlay once per press (not while held)
static void PollToggle()
{
    DWORD pads = 0;
    bool held  = false;
    for (DWORD u = 0; u < 4; u++) {
        XINPUT_STATE st;
        if (XInputGetState(u, &st) == ERROR_SUCCESS) {
            pads |= (1u << u);
            if ((st.Gamepad.wButtons & TOGGLE_BUTTONS) == TOGGLE_BUTTONS) held = true;
        }
    }
    g_diag[D_PADS] = pads;
    if (held && !g_comboWasHeld) {
        g_overlayOn = g_overlayOn ? 0 : 1;
        g_diag[D_ENABLED] = g_overlayOn;
        g_diag[D_TOGGLES] = g_diag[D_TOGGLES] + 1;
    }
    g_comboWasHeld = held;
}

static DWORD WINAPI Worker(LPVOID)
{
    g_diag[D_STATE] = 1;
    g_diag[D_ENABLED] = g_overlayOn;

    // GPU-readable memory for the command streams.
    PVOID cb = XPhysicalAlloc(SLOT_COUNT * SLOT_BYTES, (ULONG_PTR)(~(ULONG_PTR)0), 0, PAGE_READWRITE);
    DWORD va = (DWORD)cb;
    if (cb && PhysOf(va)) {
        g_cmdVa   = va;
        g_cmdPhys = PhysOf(va);
        g_diag[D_CMDVA]   = va;
        g_diag[D_CMDPHYS] = g_cmdPhys;
        g_diag[D_STATE]   = 2;
    }

    BuildWriteList();
    DelayMs(START_DELAY_MS);
    InstallHook();

    // FPS from the CPU timebase (49.875 MHz), measured every 0.5 s. The box image is rebuilt
    // only when the displayed number changes. The GPU does all the drawing
    ULONGLONG t0      = __mftb();
    DWORD     f0      = g_diag[D_FRAMES];
    DWORD     imgFps  = 0xFFFFFFFF;
    for (;;) {
        ULONGLONG t  = __mftb();
        ULONGLONG el = t - t0;
        if (el < 24937500ULL) {
            // Sleep until the next pad check or the end of the 0.5 s window, whichever comes first
            DWORD left = (DWORD)(((24937500ULL - el) * 1000ULL) / 49875000ULL) + 1;
            DelayMs(left < INPUT_POLL_MS ? left : INPUT_POLL_MS);
            PollToggle();
            continue;
        }

        DWORD f   = g_diag[D_FRAMES];
        DWORD fps = (DWORD)(((ULONGLONG)(f - f0) * 49875000ULL + (t - t0) / 2) / (t - t0));
        g_diag[D_FPS] = fps;
        f0 = f;
        t0 = t;

        DWORD shown = (fps > 999) ? 999 : fps; // the box has 3 digits
        if (shown != imgFps) { BuildBoxImage(shown); imgFps = shown; }
    }
}

extern "C" BOOL WINAPI DllMain(HANDLE, DWORD reason, LPVOID)
{
    if (reason != DLL_PROCESS_ATTACH)
        return TRUE;

    HANDLE h = NULL;
    DWORD tid = 0;
    if (ExCreateThread(&h, 0, &tid, (PVOID)XapiThreadStartup, (LPTHREAD_START_ROUTINE)Worker, NULL, 2 | CREATE_SUSPENDED) >= 0 && h != NULL) {
        XSetThreadProcessor(h, 5);
        ResumeThread(h);
        CloseHandle(h);
    }
    return TRUE;
}
