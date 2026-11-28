// XenonFPS.xex: Fast GPU-drawn FPS overlay (DashLaunch plugin)
//
//  1. The Worker thread waits START_DELAY_MS, allocates memory, then patches the first 4 instructions of the
//     kernel's VdSwap to jump to HookVdSwap
//  2. HookVdSwap counts frames, calls the real VdSwap through a trampoline, then turns the start of the NOP area
//     in VdSwap's 64-dword ring block into an INDIRECT_BUFFER to a stream of MEM_WRITE packets: the GPU writes
//     the overlay into the front buffer after the game's own commands
//  3. Streams are cached per (front buffer, display kind, image version parity); a normal frame rewrites only a
//     few ring dwords
//  4. The Worker measures FPS every 0.5 s, builds the new image into the spare of two, prebuilds its streams,
//     then bumps the version, so the hook never reads a half built image
//  5. Modes (MODE_BUTTONS): 0 = FPS only, 1 = detailed panel, 2 = panel + frame-time / frame-rate graphs
//     Sensors and statistics are gathered only in modes 1 and 2
//
// Every MEM_WRITE carries exactly one data dword: the console writes only the first dword of a longer packet
// Diagnostics: read g_diag with getmem (magic 0xD1A60100, indices D_*)

#include <xtl.h>

// ======== Settings ========
#define CORNER_TOP_LEFT       0
#define CORNER_TOP_RIGHT      1
#define CORNER_BOTTOM_LEFT    2
#define CORNER_BOTTOM_RIGHT   3
#define BOX_CORNER            CORNER_TOP_LEFT   // starting corner (CORNER_BUTTONS cycles it in game)
#define BOX_MARGIN_TILES      0                 // inward offset from the screen edges, in 32 px tiles (overscan)
#define BG_MARGIN_X           2                 // FPS box background margin left / right of the digits, 0..10 px
#define BG_MARGIN_Y           2                 // above / below the digits, 0..6 px
#define BOX_EDGE_GAP          6                 // gap between the screen corner and the background rectangle, 0..8 px
#define BOX_BACKGROUND        1                 // FPS mode: 1 = solid background, 0 = digit pixels only (fewer writes)
#define BOX_BG_COLOR          0x000000          // 0xRRGGBB, opaque; also the panel and graph background
#define BOX_TEXT_COLOR        0xFFFFFF

#if BOX_CORNER < 0 || BOX_CORNER > 3
  #error BOX_CORNER must be one of the CORNER_* values
#endif
#if BOX_MARGIN_TILES < 0 || BOX_MARGIN_TILES > 10
  #error BOX_MARGIN_TILES must be 0..10
#endif
#if BG_MARGIN_X < 0 || BG_MARGIN_X > 10 || BG_MARGIN_Y < 0 || BG_MARGIN_Y > 6
  #error BG_MARGIN_X must be 0..10 and BG_MARGIN_Y 0..6 so the rectangle stays inside the box
#endif
#if BOX_EDGE_GAP < 0 || BOX_EDGE_GAP > 8 || \
    BOX_EDGE_GAP + 44 + 2 * BG_MARGIN_X > 64 || BOX_EDGE_GAP + 20 + 2 * BG_MARGIN_Y > 32
  #error BOX_EDGE_GAP does not fit: the rectangle for three digits plus the gap must stay inside the 64x32 box
#endif
#if (BOX_BG_COLOR) > 0xFFFFFF || (BOX_TEXT_COLOR) > 0xFFFFFF
  #error BOX_BG_COLOR and BOX_TEXT_COLOR must be 0xRRGGBB
#endif

#define BOX_BG_PIX        ColorToPixel(BOX_BG_COLOR)
#define BOX_TEXT_PIX      ColorToPixel(BOX_TEXT_COLOR)

// ---- Controller (any of the 4 pads; join XINPUT_GAMEPAD_* values with |) ----
#define CONTROLLER_INPUT  1     // 0 = XInput is never called: overlay stays at OVERLAY_START_ON / BOX_CORNER / mode 0
#define OVERLAY_START_ON  1
#define TOGGLE_BUTTONS    (XINPUT_GAMEPAD_RIGHT_THUMB | XINPUT_GAMEPAD_DPAD_UP)      // show / hide
#define CORNER_BUTTONS    (XINPUT_GAMEPAD_RIGHT_THUMB | XINPUT_GAMEPAD_DPAD_LEFT)    // next corner
#define MODE_BUTTONS      (XINPUT_GAMEPAD_RIGHT_THUMB | XINPUT_GAMEPAD_DPAD_RIGHT)   // next display mode
#define DISPLAY_MODES     3
#define INPUT_POLL_MS     50    // pad poll interval
#define PAD_RESCAN_MS     1000  // pads that did not answer are re-checked this often
#if CONTROLLER_INPUT != 0 && CONTROLLER_INPUT != 1
  #error CONTROLLER_INPUT must be 0 or 1
#endif

// ---- Worker ----
#define START_DELAY_MS  15000                         // before this: no allocation, sensors, input or hook
#define WORKER_THREAD   5                             // hardware thread 0..5 (4 and 5 share a core); pick the game's idlest
#define WORKER_PRIORITY THREAD_PRIORITY_BELOW_NORMAL

// ---- Diagnostics ----
#define DIAG_ENABLED      0     // 0 = drop the hook's timing reads and counters (keep 1 in development builds)
#define SLOW_US           50    // a hook pass longer than this counts as slow
#define SLOW_TICKS        ((SLOW_US * 49875) / 1000)    // timebase ticks (49.875 MHz)

// ---- Accepted frame sizes ----
#define MODE_MIN_W        640
#define MODE_MAX_W        1920
#define MODE_MIN_H        480
#define MODE_MAX_H        1080

// ---- Detailed panel (modes 1 and 2) ----
#define DETAIL_OUTLINE_MAX  24000       // packets per detailed stream (text and graph pixels plus a 1 px background halo)
#define DETAIL_TEMP_EVERY   20          // SMC temperatures are read once per this many panel updates (0.5 s each)
#define DETAIL_GPU_COLOR    0x3CB44B    // GPU labels
#define DETAIL_CPU_COLOR    0x2F9BE3    // CPU labels
#define DETAIL_RAM_COLOR    0xB57BFF
#define DETAIL_FPS_COLOR    0xD9A19B    // frame rate labels and graph titles
#define DETAIL_VAL_COLOR    0xFF8C1A    // usage, memory and temperature values
#define DETAIL_FPSVAL_COLOR 0xFFFFFF    // frame rate values, graph readouts and axis labels

// ---- Graphs (mode 2) ----
#define GRAPH_FT_TOP_MS     50          // top of the frame-time graph (longer frames clip); grid at 16.7 and 33.3 ms
#define GRAPH_COL_TICKS     4987500     // timebase ticks per graph column (100 ms)
#define GRAPH_LINE_COLOR    0x3CB44B
#define GRAPH_GRID_COLOR    0x404040
#define GRAPH_BORDER_COLOR  0x909090

// ======== Derived sizes ========

// FPS box: 64x32 px = 2 tiles, built at tile (0,0); the screen position is applied per frame
#define BOX_DWORDS        2048

// Worst case MEM_WRITE packets per stream: the background rectangle around 3 digits, or 3 x '8' lit pixels
#if BOX_BACKGROUND
  #define BOX_MAX_WRITES  ((44 + 2 * BG_MARGIN_X) * (20 + 2 * BG_MARGIN_Y))
#else
  #define BOX_MAX_WRITES  (3 * 13 * 16)
#endif
#define STREAM_DWORDS     (BOX_MAX_WRITES * 3)
#define SLOT_BYTES        (((STREAM_DWORDS * 4) + 127) & ~127)

// Panel layout: the 5x7 font at 1.25x (6x9 glyphs, advances alternate 7 and 8 px). TEXT_W(n) = width of n characters
#define DET_ROWS        16
#define DET_CHARS       12
#define DET_GLYPH_W     6
#define DET_GLYPH_H     9
#define TEXT_W(n)       ((n) ? (((n) - 1) * 15 / 2 + DET_GLYPH_W) : 0)
#define DET_PITCH       12                                  // row advance: 9 px glyph + 3 px gap
#define DET_PAD         4
#define DET_RECT_W      (TEXT_W(DET_CHARS) + 2 * DET_PAD)
#define DET_RECT_H      ((DET_ROWS - 1) * DET_PITCH + DET_GLYPH_H + 2 * DET_PAD)

// Graph blocks: title row, bordered plot, axis labels to the right
#define GB_PAD          DET_PAD
#define GB_LBL_W        TEXT_W(3)
#define GB_LBL_GAP      4
#define GB_PLOT_W       (DET_RECT_W - 2 * GB_PAD)           // including the border
#define GB_RECT_W       (2 * GB_PAD + GB_PLOT_W + GB_LBL_GAP + GB_LBL_W)
#define GB_PLOT_H       40
#define GB_IN_W         (GB_PLOT_W - 2)
#define GB_IN_H         (GB_PLOT_H - 2)
#define GB_TITLE_GAP    3                                   // between title row and plot
#define GB_H            (GB_PAD + DET_GLYPH_H + GB_TITLE_GAP + GB_PLOT_H + GB_PAD)
#define GB_GAP          5                                   // above each block
#define GRAPH_COLS      GB_IN_W                             // one history column per plot pixel
#define DET_STACK_H     (DET_RECT_H + 2 * (GB_GAP + GB_H))  // panel + both blocks

// Detailed box in whole 32x32 tiles: sized for the mode 2 stack when graphs exist (mode 1 draws only the panel)
#if DISPLAY_MODES >= 3
  #define DET_BOX_W     GB_RECT_W
  #define DET_BOX_H     DET_STACK_H
#else
  #define DET_BOX_W     DET_RECT_W
  #define DET_BOX_H     DET_RECT_H
#endif
#define DET_TILES_W     ((DET_BOX_W + BOX_EDGE_GAP + 31) / 32)
#define DET_TILES_H     ((DET_BOX_H + BOX_EDGE_GAP + 31) / 32)
#define DET_W           (DET_TILES_W * 32)
#define DET_H           (DET_TILES_H * 32)
#define DET_IMG_DWORDS  (DET_TILES_W * DET_TILES_H * 1024)
#define DET_MASK_DWORDS (DET_H * DET_TILES_W)               // one bit per pixel

// A detailed stream is split into fixed-capacity regions: region 0 = all static content (labels, units, titles,
// borders), then one per number field and three per graph (readout, axis labels, plot). A dynamic region's capacity
// is its rectangle's area, so it cannot overflow. Static + dynamic is at most 22,648 packets: recheck on layout changes
#define DET_MAX_WRITES  DETAIL_OUTLINE_MAX
#if DETAIL_OUTLINE_MAX < 22700
  #error DETAIL_OUTLINE_MAX is below the worst case of the regioned stream
#endif
#define DET_STATIC_RGNS DET_TILES_H                         // static content: one region per 32 px band
#define DET_PANEL_RGNS  (DET_STATIC_RGNS + DET_ROWS)        // static bands + number fields
#define DET_REGIONS     (DET_PANEL_RGNS + 2 * 3)            // + 2 graphs x (readout, axis, plot)
#define DSLOT_BYTES     (((DET_MAX_WRITES * 3 * 4) + 127) & ~127)
#if DET_H + 32 * BOX_MARGIN_TILES > MODE_MIN_H
  #error The detailed panel plus BOX_MARGIN_TILES does not fit the smallest accepted frame height
#endif
#if DET_IMG_DWORDS > 65536
  #error The pixel lists hold image dword indexes in a WORD
#endif

// Frame statistics
#define FT_RING         512                   // frame times kept (power of two)
#define FT_WORST        8                     // 1% low averages the worst n/100 frames, at most this many
#define FT_MAX_TICKS    24937500              // a gap of 0.5 s or more (pause, loading) is not a frame time
#define FPS_HIST        128                   // FPS readouts kept for MIN / MAX

// Stream slots: FB_ENTRIES front buffers x 2 image parities, once for FPS mode and once for detailed
// Keep 3: with 2 (saves ~454 KiB of GPU memory) a title cycling 3 buffers evicts and rebuilds a stream on every
// return, the hook skips frames without a ready stream (D_STREAM_MISS) and the overlay flickers
#define FB_ENTRIES      3
#define SLOT_COUNT      (FB_ENTRIES * 2)
#define DET_SLOT_COUNT  SLOT_COUNT

// GPU memory for the detailed mode: its streams + 128 bytes for the GPU probe words
#define DETAIL_MEM_BYTES    (DET_SLOT_COUNT * DSLOT_BYTES + 128)

// Cached CPU memory, detailed part: 2 images, one pixel list per image, 3 masks per image
#define DET_IMGS_BYTES      (2 * DET_IMG_DWORDS * 4)
#define DET_PIXLIST_BYTES   (((2 * DET_MAX_WRITES * 2) + 127) & ~127)
#define DET_MASKS_BYTES     (((6 * DET_MASK_DWORDS * 4) + 127) & ~127)   // outline, lit, static

// Packet positions: (tile of the box << 10) | dword offset inside the destination tile. A list is stored only when
// several slots are filled from the same image; the other slots replay it
#if DET_TILES_W * (DET_TILES_H + 1) <= 64
  typedef WORD PktLoc;                  // 6 bits of tile index + 10 bits of offset
#else
  typedef DWORD PktLoc;
#endif
#define DET_LOC_BYTES       (((2 * DET_MAX_WRITES * sizeof(PktLoc)) + 127) & ~127)
#define DET_WORK_BYTES      (DET_IMGS_BYTES + DET_PIXLIST_BYTES + DET_MASKS_BYTES + DET_LOC_BYTES)

// FPS part: 2 images, 2 write masks, pixel lists
#define WK_BOXIMGS_BYTES    (2 * BOX_DWORDS * 4)
#define WK_WMASK_BYTES      (2 * 64 * 4)
#define WK_SLOTPIX_BYTES    (((SLOT_COUNT * BOX_MAX_WRITES * 2) + 127) & ~127)
#define FPS_WORK_BYTES      (WK_BOXIMGS_BYTES + WK_WMASK_BYTES + WK_SLOTPIX_BYTES)

typedef LONG NTSTATUS;
extern "C" {
    NTSTATUS ExCreateThread(PHANDLE, DWORD, LPDWORD, PVOID, LPTHREAD_START_ROUTINE, LPVOID, DWORD);
    NTSTATUS KeDelayExecutionThread(CHAR WaitMode, BOOL Alertable, PLARGE_INTEGER Interval);
    VOID XapiThreadStartup(VOID (__cdecl *)(VOID *), PVOID, DWORD);
    VOID HalSendSMCMessage(PVOID Command, PVOID Response);
}

// Kernel-only sleep: Sleep() needs xapi runtime state
static VOID DelayMs(DWORD ms) {
    LARGE_INTEGER t;
    t.QuadPart = -(LONGLONG)ms * 10000;   // negative = relative, 100 ns units
    KeDelayExecutionThread(0, FALSE, &t); // 0 = KernelMode
}

// ======== Diagnostics ========
enum {
    D_MAGIC,                                            // 0xD1A60100
    D_STATE,                                            // 1 = worker started, 2 = command memory allocated
    D_PATCH,                                            // 0xBAD = VdSwap mismatch (unpatched), 2 = trampoline ready, 3 = installed
    D_HOOKADDR, D_TRAMP,                                // addresses of HookVdSwap and the trampoline
    D_FRAMES, D_FPS,                                    // VdSwap calls seen (g_frames itself), last measured FPS
    D_QUEUED,                                           // frames that got the overlay
    D_SKIPPED,                                          // accepted frames that got nothing (no stream / memory / image, ring mismatch, exception)
    D_FB,                                               // front buffer of the last accepted frame (set when the gate record is recomputed)
    D_BUILDS, D_VERSION,                                // images published, current image version
    D_CMDVA, D_CMDPHYS,                                 // command memory: CPU and physical address
    D_REJ_GATE,                                         // frames rejected by the frame check
    D_LAST_D0, D_LAST_D1, D_LAST_D2, D_LAST_FB,         // fetch dwords and front buffer of the last rejected frame
    D_REJ_PTR, D_LAST_A,                                // accepted frames with a bad ring pointer, and that pointer
    D_REJ_NOCMD, D_REJ_NOIMG, D_REJ_FB, D_REJ_RING,     // skipped: no memory, no image, front buffer < 0xA0000000, ring mismatch
    D_LAST_RB0, D_LAST_RB4,                             // rb[0], rb[4] at the last ring mismatch
    D_EXC,                                              // exceptions inside the hook
    D_OK_D1,                                            // fetch dword 1 of the last accepted frame (set with D_FB)
    D_BAD_A,                                            // ring pointer at the last ring mismatch
    D_BAD0, D_BAD_LAST = D_BAD0 + 15,                   // rb[0..15] at the last ring mismatch
    D_ENABLED, D_TOGGLES,                               // 1 = shown; show / hide presses
    D_PADS,                                             // controllers XInput sees (bit 0 = pad 1)
    D_HIDDEN,                                           // frames not drawn: overlay hidden, or image in another pixel layout
    D_RUNS, D_STREAMLEN,                                // MEM_WRITE packets of the last published image, its stream length in dwords
    D_HOOK_TB_SUM, D_HOOK_TB_MAX, D_HOOK_TB_N,          // hook time after the real VdSwap (ticks, 49.875 MHz): sum, max, passes
    D_PREBUILT,                                         // streams prebuilt by the Worker
    D_PRE_TB_SUM, D_PRE_TB_MAX,                         // hook time before the real VdSwap: sum, max (ticks)
    D_SLOW_POST, D_SLOW_PRE,                            // after / before swap sections longer than SLOW_US
    D_CORNER,                                           // 0 top left, 1 top right, 2 bottom left, 3 bottom right
    D_FAST_MISS,                                        // StreamFor calls that missed both fast records
    D_MODE, D_MODE_SW,                                  // display mode 0..2, mode switches
    D_DET_MEM,                                          // detailed memory: 0 not tried, 1 ready, 2 unavailable
    D_ST_MIN, D_ST_AVG, D_ST_MAX, D_ST_LOW,             // last frame statistics (fps)
    D_WSTAGE,                                           // Worker: 1 idle, 4 stats, 5 RAM, 6 SMC, 7 detailed image, 8 prebuilding, 9 done
    D_WBEAT,                                            // Worker 0.5 s ticks (stops if it is stuck)
    D_SMC_N,                                            // SMC temperature reads
    D_PRB_CLK, D_PRB_B0, D_PRB_B1,                      // GPU probe: last raw words (CP clock, RBBM 0, RBBM 1)
    D_PRB_DCLK, D_PRB_DB0, D_PRB_DB1,                   // their deltas between the last two reads
    D_PRB_DTB,                                          // timebase delta of that interval (dClk ~ dTb * 10.02)
    D_PRB_PCT0, D_PRB_PCT1,                             // 100 * dB / dClk
    D_PRB_SETUP,                                        // counter setup packets sent
    D_OK_D3,                                            // fetch dword 3 of the last accepted frame (bits 1-12 = swizzle)
    D_LAST_B, D_LAST_F,                                 // VdSwap args b and f of the last rejected frame
    // Timing statistics, 3 dwords each (sum, max, count), in timebase ticks (D_G_IB: GPU CP clock, ~500 MHz)
    // Hook phases: pacing + fetch reads, gate check / recompute + RecordFrame, the real VdSwap (for scale), ring
    // signature check, StreamFor, writing the ring packets, barrier after the ring edit
    D_T_PEEK, D_T_GATE = D_T_PEEK + 3, D_T_SWAP = D_T_GATE + 3, D_T_RINGCHK = D_T_SWAP + 3,
    D_T_STREAM = D_T_RINGCHK + 3, D_T_WRITE = D_T_STREAM + 3, D_T_FENCE = D_T_WRITE + 3,
    // Worker: one 0.5 s tick; GatherDetail with its parts (FrameStats + GraphUpdate, ProbeGpu, CpuUsageUpdate,
    // ReadRamMB, ReadTemps on reading ticks only); image build; PrebuildStreams after a build; one fill inside it; PollToggle
    D_T_W_TICK = D_T_FENCE + 3, D_T_W_GATHER = D_T_W_TICK + 3, D_T_W_FSTATS = D_T_W_GATHER + 3,
    D_T_W_PROBE = D_T_W_FSTATS + 3, D_T_W_CPU = D_T_W_PROBE + 3, D_T_W_RAM = D_T_W_CPU + 3, D_T_W_SMC = D_T_W_RAM + 3,
    D_T_W_BUILD = D_T_W_SMC + 3, D_T_W_PREBUILD = D_T_W_BUILD + 3, D_T_W_FILL = D_T_W_PREBUILD + 3,
    D_T_W_POLL = D_T_W_FILL + 3,
    D_G_IB         = D_T_W_POLL + 3,       // GPU: CP clock ticks on the overlay stream
    // Frame pacing, measured in the hook while the overlay is shown
    D_F_SUM        = D_G_IB + 3,           // sum of frame intervals (0.1 ms units, intervals over 1 s excluded)
    D_F_SUMSQ,                             // sum of their squares
    D_F_MAX,                               // longest interval in ticks (never reset by the plugin)
    D_F_B0, D_F_B1, D_F_B2, D_F_B3, D_F_B4, D_F_B5,   // histogram: <= 18, 25, 34, 50, 100 ms, longer
    // Event counters: gate record recomputed; stream fills that rewrote everything; partial fills (FPS data only, or
    // only the changed detailed regions); detailed regions replayed from a cached list (no tile walk); prebuilds
    // skipped (buffer inactive); detailed images built from scratch / updated incrementally; FPS-only images built;
    // frames skipped with no ready stream (the Worker is asked to build it)
    D_GATE_MISS, D_FILL_FULL, D_FILL_DATA, D_FILL_LIST, D_PRE_SKIP, D_IMG_FULL, D_IMG_INCR, D_IMG_FPS, D_STREAM_MISS,
    // Packet and memory use (DIAG_ENABLED only)
    D_PKT_PEAK0, D_PKT_PEAK1, D_PKT_PEAK2,  // highest packet count of any published image: FPS-only, panel, panel + graphs
    D_MEM_WORK,                            // bytes requested for the cached CPU work memory
    D_MEM_CMD,                             // bytes requested for the write-combined GPU command memory
    D_DET_CLAMP,                           // unused (stays 0): regions cannot exceed their capacity
    D_W_STALL_N, D_W_STALL_SUM,            // Worker timed sections over W_STALL_TICKS (preempted, kept out of the averages): count, ticks
    D_FILL_PKTS,                           // MEM_WRITE packets written by detailed fills
    D_COUNT
};

static volatile DWORD g_diag[D_COUNT] = { 0xD1A60100 };
#define COUNT(i)    (g_diag[i] = g_diag[i] + 1)     // always on
#define STAGE(n)    (g_diag[D_WSTAGE] = (n))        // Worker only, always on

// Switchable diagnostics. TMARK starts a timer, TSTAT adds a duration to the 3 dwords at i, TLAP adds the time
// since the timer and restarts it from the same timebase read
#if DIAG_ENABLED
  static __forceinline void DiagMax(DWORD i, DWORD v) { if (v > g_diag[i]) g_diag[i] = v; }
  static __forceinline void DiagStat(DWORD i, DWORD ticks) {
      g_diag[i] = g_diag[i] + ticks;
      if (ticks > g_diag[i + 1]) g_diag[i + 1] = ticks;
      COUNT(i + 2);
  }

  #define DIAG_SET(i, v)  (g_diag[i] = (v))
  #define DIAG_MAX(i, v)  DiagMax((i), (DWORD)(v))
  #define DIAG_ADD(i)     COUNT(i)
  #define TNOW()          ((DWORD)__mftb())
  #define TMARK(v)        DWORD v = TNOW(); (void)v
  #define TSTAT(i, ticks) DiagStat((i), (DWORD)(ticks))
  static __forceinline void DiagLap(DWORD i, DWORD &v) {
      DWORD n = TNOW(); DiagStat(i, n - v); v = n;
  }

  #define TLAP(i, v)      DiagLap((i), (v))

  // Worker timers: a sample over W_STALL_TICKS is a preemption and updates only the max. leaf = also count it in
  // D_W_STALL_* (outer timers pass false, so one stall is not counted at every nesting level)
  #define W_STALL_TICKS   199500                    // 4 ms
  static __forceinline void DiagWork(DWORD i, DWORD ticks, bool leaf) {
      if (ticks <= W_STALL_TICKS) { DiagStat(i, ticks); return; }
      DiagMax(i + 1, ticks);
      if (leaf) { COUNT(D_W_STALL_N); g_diag[D_W_STALL_SUM] = g_diag[D_W_STALL_SUM] + ticks; }
  }

  static __forceinline void DiagWorkLap(DWORD i, DWORD &v) {
      DWORD n = TNOW(); DiagWork(i, n - v, true); v = n;
  }

  #define WSTAT(i, ticks)       DiagWork((i), (DWORD)(ticks), true)
  #define WSTAT_OUTER(i, ticks) DiagWork((i), (DWORD)(ticks), false)
  #define WLAP(i, v)            DiagWorkLap((i), (v))
#else
  #define DIAG_SET(i, v)  ((void)0)
  #define DIAG_MAX(i, v)  ((void)0)
  #define DIAG_ADD(i)     ((void)0)
  #define TNOW()          0
  #define TMARK(v)        DWORD v = 0; (void)v
  #define TSTAT(i, ticks) ((void)0)
  #define TLAP(i, v)      ((void)0)
  #define WSTAT(i, ticks)       ((void)0)
  #define WSTAT_OUTER(i, ticks) ((void)0)
  #define WLAP(i, v)            ((void)0)
#endif

// ======== State ========
// The frame counter lives in g_diag (D_FRAMES) so the hook does one store per frame. It drives the FPS number, so
// it is not part of the diagnostics switch
#define g_frames    (g_diag[D_FRAMES])
#if DIAG_ENABLED
// Hook-entry intervals of shown frames: sum and sum of squares, histogram, maximum. Hidden frames are not timed
// (the hook clears g_fLast), so the first shown frame is not an interval
static DWORD g_fLast = 0;
static __forceinline void FramePacing(DWORD now) {
    DWORD had = g_fLast; DWORD dt = now - had; g_fLast = now;
    if (!had) return;
    // 0.1 ms units without a divide: floor((dt + 1) * 861060 / 2^32) = dt / 4988 exactly up to 268 M ticks (5.4 s);
    // beyond that it may be one low, which only lands in the last bucket and the excluded sums
    DWORD ms10 = (DWORD)((((ULONGLONG)dt + 1) * 861060ULL) >> 32);
    DWORD bkt  = (ms10 <= 180) ? 0 : (ms10 <= 250) ? 1 : (ms10 <= 340) ? 2 : (ms10 <= 500) ? 3 : (ms10 <= 1000) ? 4 : 5;
    COUNT(D_F_B0 + bkt);
    if (dt > g_diag[D_F_MAX]) g_diag[D_F_MAX] = dt;
    if (ms10 <= 10000) {
        g_diag[D_F_SUM] = g_diag[D_F_SUM] + ms10; g_diag[D_F_SUMSQ] = g_diag[D_F_SUMSQ] + ms10 * ms10;
    }
}
#endif

// Frame times: written by the hook while the overlay is shown, read by the Worker
static volatile DWORD g_ftRing[FT_RING];
static volatile DWORD g_ftIdx  = 0;     // frames recorded since the last reset
static DWORD          g_ftLast = 0;     // timebase low word of the previous recorded frame

// Worker only: past FPS readouts, each with the ring index where its window ended
static DWORD g_fpsIdx = 0;              // ring index the next FPS window starts at
static DWORD g_fhFps[FPS_HIST];
static DWORD g_fhEnd[FPS_HIST];
static DWORD g_fhN    = 0;              // entries pushed since the last reset
__declspec(align(128)) static volatile DWORD g_tramp[32];
static volatile DWORD g_cmdVa = 0;      // FPS-only streams, write combined

// Stream cache: slot = ((detailed ? FB_ENTRIES : 0) + entry) * 2 + (version & 1); FPS-only slots come first
//   entry[e]  a front buffer and its shape: geom = height | tiles across << 12 | endian << 20
//   slot[s]   what the stream was written for: fb, key = boxOff | boxYo << 2 | endian (see BoxPos),
//             key2 = boxOff1 (detailed: | tiles across), ver = image version of its data (0 = unusable),
//             lay = pixel set of its headers
//   fast[0..1]  the hook's last two (fb, ver, key, key2) -> slot answers, so a game alternating two buffers always
//             hits. At most one record per slot; the Worker refills only invalid slots, so a matching record is never stale
struct FbEntry { volatile DWORD fb; volatile DWORD geom; };
struct SlotTag { volatile DWORD fb; volatile DWORD key; volatile DWORD key2; volatile DWORD ver; volatile DWORD lay; };

// FastRec and GateRec are touched only by the hook, so they are not volatile (the compiler may merge accesses)
// phys = the slot's stream physical address (fixed after start-up)
struct FastRec { DWORD fb; DWORD ver; DWORD key; DWORD key2; DWORD slot; DWORD phys; };

// Verdict for the last frame, reused while the five raw dwords and the image cfg are unchanged. A zeroed record
// is a valid "rejected" answer
struct GateRec {
    DWORD fb, d0, d1, d2, d3;            // raw inputs
    DWORD cfg;                           // image cfg the box position was computed for
    DWORD ok;                            // passed the frame check
    DWORD pf;                            // PixDescOf(fmt, d3)
    DWORD geom;
    DWORD boxOff, boxOff1, boxYo;
};

// Everything the hook reads on a normal frame. The first 128 bytes (header 32, fast 48, gate 48) are one cache
// line; HotLineCheck breaks the build if a field pushes the gate out of it
struct HotState {
    volatile DWORD overlayOn;     // 1 = shown
    volatile DWORD boxVer;        // current image version, 0 = none yet
    volatile DWORD cmdPhys;       // physical address of the command memory, 0 = none
    volatile DWORD listCnt[2];    // packets per stream, per image parity
    volatile DWORD imgCfg[2];     // per parity: corner (bits 0-1) | display mode << 2 the image was laid out for
    volatile DWORD seen;          // bit (slot >> 1): buffer drawn this tick (set by the hook, cleared by the Worker)
    FastRec        fast[2];
    GateRec        gate;
    FbEntry        entry[FB_ENTRIES];
    SlotTag        slot[SLOT_COUNT * 2];
    volatile DWORD slotPhys[SLOT_COUNT * 2];    // stream physical address, 0 = no memory
};

__declspec(align(128)) static HotState g_hot = { OVERLAY_START_ON, 0, 0, { 0, 0 } };
typedef char HotLineCheck[(8 * 4 + 2 * sizeof(FastRec) + sizeof(GateRec) <= 128) ? 1 : -1];
static volatile DWORD g_corner = BOX_CORNER;             // corner the user asked for (Worker only)
#define g_overlayOn   (g_hot.overlayOn)
#define g_boxVer      (g_hot.boxVer)
#define g_cmdPhys     (g_hot.cmdPhys)
#define g_listCnt     (g_hot.listCnt)
static DWORD g_fbNext = 0;                              // round robin entry replacement
static volatile DWORD g_missReq = 0;                    // hook -> Worker: a frame found no ready stream

// Buffers the Worker prebuilds for: HotState::seen of the last two ticks (all while hidden)
static volatile DWORD g_activeMask = 0xFFFFFFFF;
static DWORD          g_seenPrev   = 0xFFFFFFFF;
// Pointers into the Worker's cached CPU allocation. The current image is index g_boxVer & 1

static DWORD *g_boxImgs[2] = { 0, 0 };     // FPS box images, BOX_DWORDS each
static DWORD *g_wMask[2] = { 0, 0 };       // pixels to write: bit (x & 31) of [(y << 1) | (x >> 5)]
static DWORD  g_listLay[2];                // FPS image layout (digit count and corner), fixes the pixel set
static WORD  *g_slotPix = 0;               // per slot: image dword of each packet, for data-only rebuilds
static DWORD *g_detImgs = 0;               // two images of DET_IMG_DWORDS
static DWORD *g_detMask[2] = { 0, 0 };     // pixels that get packets: lit grown by the 1 px halo
static DWORD *g_detLit[2] = { 0, 0 };      // text and graph pixels
static DWORD *g_detStat[2] = { 0, 0 };     // g_detMask without the dynamic regions
static WORD  *g_dPix = 0;                  // one pixel list per image (DET_MAX_WRITES each), indexed like the stream
static PktLoc *g_dLoc = 0;                 // packet position of each list entry (tile << 10 | dword offset)

// One region of a detailed image: box rectangle (x1 / y1 excluded), place in the stream, packets used, and a stamp
// that changes whenever its content does
struct DetRgn { DWORD x0, y0, x1, y1, off, cap, used, stamp; };
static DetRgn g_rgn[2][DET_REGIONS];
static DWORD  g_rgnN[2];                   // regions in each image (panel only, or with graphs)
static DWORD  g_detLayId[2];               // changes on every full build of that image (never 0 once built)
static DWORD  g_detStampCtr = 0;
static DWORD  g_slotStamp[DET_SLOT_COUNT][DET_REGIONS];   // region stamps each detailed slot holds
static DWORD  g_lstStamp[2][DET_REGIONS];  // region stamp each cached list was stored for (0 = none)
static DWORD  g_lstYo[2][DET_REGIONS];     // ... and its box row offset
static volatile DWORD g_dCmdVa = 0;        // detailed streams
static WORD g_tiledTab[BOX_DWORDS];        // byte offset of each pixel of a 64x32 (2 tile) block, see InitTiledTab
static WORD g_invTab[1024];                // inverse for one tile: dword offset -> (y << 5) | x

// GPU probe: four dwords written by the GPU in the spare end of the detailed memory (so unavailable without it):
// [0] CP clock after the stream, [1] RBBM 0, [2] RBBM 1, [3] CP clock before
static volatile DWORD *g_probeVa   = 0;
static volatile DWORD  g_probePhys = 0;
static volatile DWORD  g_probeArm  = 1;                  // send the counter setup packets with the next sample
static volatile DWORD  g_probeWant = 1;                  // Worker wants a sample (one per tick); cleared by the hook
static volatile DWORD  g_probeSel[2] = { 1, 13 };        // RBBM events: NRT_BUSY (1), SX_BUSY (13)
static DWORD     g_prbClk, g_prbB0, g_prbB1, g_prbHave;
static ULONGLONG g_prbTb;
static DWORD     g_prbPct0, g_prbPct1;
static DWORD g_mode = 0;        // display mode, owned by the Worker (the hook sees it through imgCfg)

// ======== Pixel layout ========
// The layout is read from each frame. Descriptor: bit 0 = kind (0: 8:8:8:8, formats 6/50; 1: A2:10:10:10, formats
// 7/54), bits 1-2 / 3-4 / 5-6 = channel slot of red / green / blue (0 = lowest bits). Alpha takes the top bits, always
// opaque. PIX_DEFAULT = 8:8:8:8 as 0xAARRGGBB
#define PIX_DEFAULT  ((2u << 1) | (1u << 3) | (0u << 5))
static volatile DWORD g_pixFmt = PIX_DEFAULT;                              // last accepted frame (set by the hook)
static DWORD          g_buildFmt = PIX_DEFAULT;                            // latched once per image build
static volatile DWORD g_imgFmt[2] = { PIX_DEFAULT, PIX_DEFAULT };          // layout of each published image
static DWORD Ch10(DWORD c8) { return ((c8 << 2) | (c8 >> 6)) & 0x3FF; }

// d3 bits 1-12 = swizzle, 3 bits per output (R, G, B, A) naming the source slot. Unless R, G, B use three different
// slots among 0..2, the default order is assumed
static DWORD PixDescOf(DWORD fmt, DWORD d3) {
    DWORD kind = (fmt == 7 || fmt == 54) ? 1u : 0u; DWORD sw = (d3 >> 1) & 0xFFF;
    DWORD sR = sw & 7, sG = (sw >> 3) & 7, sB = (sw >> 6) & 7;
    if (sR > 2 || sG > 2 || sB > 2 || sR == sG || sR == sB || sG == sB) return kind | PIX_DEFAULT;
    return kind | (sR << 1) | (sG << 3) | (sB << 5);
}

// 0xRRGGBB -> frame buffer dword in the layout being built
static DWORD ColorToPixel(DWORD rgb) {
    DWORD p = g_buildFmt; DWORD kind = p & 1; DWORD bits = kind ? 10 : 8;
    DWORD r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF, b = rgb & 0xFF;
    if (kind) { r = Ch10(r); g = Ch10(g); b = Ch10(b); }
    DWORD out = kind ? 0xC0000000u : 0xFF000000u; out |= r << (((p >> 1) & 3) * bits);
    out |= g << (((p >> 3) & 3) * bits); out |= b << (((p >> 5) & 3) * bits);
    return out;
}

// ======== Frame access ========
// CPU address -> GPU physical address, 0 if unknown (0xE0000000+ needs an extra 0x1000)
static DWORD PhysOf(DWORD va) {
    if (va >= 0xA0000000u && va < 0xE0000000u) return va & 0x1FFFFFFFu;
    if (va >= 0xE0000000u)                     return (va & 0x1FFFFFFFu) + 0x1000u;
    return 0;
}

// Refuses unaligned addresses, the null page and the register window (reads there can have side effects)
static __forceinline bool PeekOk(DWORD a) {
    return !((a & 3) || a < 0x10000 || (a >= 0x7FC00000 && a < 0x80000000));
}

// ======== Streams ========
// Box position for an image cfg (corner | mode << 2). Horizontally whole tiles; vertically the top edge may fall
// inside a tile row (720 and 1080 are not multiples of 32): boxYo = pixels down inside the first row, boxOff1 = the
// next tile row (0 if boxYo is 0). Offsets are in bytes; the gate guarantees pitch == width
static __forceinline void BoxPos(DWORD cfg, DWORD cols, DWORD hgt, DWORD &boxOff, DWORD &boxOff1, DWORD &boxYo) {
    DWORD corner = cfg & 3;
    DWORD bw   = (cfg >> 2) ? DET_TILES_W : 2;           // width in tiles
    DWORD bh   = (cfg >> 2) ? DET_H : 32;                // height in pixels
    DWORD col  = (corner & 1) ? (cols - bw - BOX_MARGIN_TILES) : BOX_MARGIN_TILES;
    DWORD y0 = (corner & 2) ? (hgt - bh - 32 * BOX_MARGIN_TILES) : (32 * BOX_MARGIN_TILES); DWORD tr0 = y0 >> 5;
    boxYo   = y0 & 31;
    boxOff  = (tr0 * cols + col) << 12;                  // 4096 bytes per tile
    boxOff1 = boxYo ? (((tr0 + 1) * cols + col) << 12) : 0;
}

// One frame-time sample; now = timebase low word at hook entry
static __forceinline void RecordFrame(DWORD now) {
    DWORD dt = now - g_ftLast; g_ftLast = now;
    if (dt < FT_MAX_TICKS) {                             // drops the first frame after a pause or restart
        DWORD i = g_ftIdx; g_ftRing[i & (FT_RING - 1)] = dt; g_ftIdx = i + 1;
    }
}

static bool SlotValid(DWORD slot, DWORD fb, DWORD ver, DWORD key, DWORD key2) {
    const SlotTag &t = g_hot.slot[slot];
    return t.fb == fb && t.key == key && t.key2 == key2 && t.ver == ver;
}

static __forceinline void CommitSlot(SlotTag &t, bool hdrOk, DWORD fb, DWORD key, DWORD key2, DWORD lay, DWORD ver) {
    __sync();                               // drain the write-combine buffers
    if (!hdrOk) { t.fb = fb; t.key = key; t.key2 = key2; t.lay = lay; }
    t.ver = ver;                            // written last: marks the slot usable
}

// Index of the lowest set bit of a nonzero dword (de Bruijn)
static const BYTE kLowBit[32] = {
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};

static __forceinline DWORD LowBit(DWORD m) { return kLowBit[((m & (0u - m)) * 0x077CB531u) >> 27]; }
// Fills an FPS-only slot (Worker prebuild). off0 / off1 / yo: see BoxPos. A full build walks destination memory in
// address order, one packet per masked pixel. With a background the pixel set is fixed per layout, so later versions
// rewrite data only
static void FillSlot(DWORD slot, DWORD fb, DWORD ver, DWORD off0, DWORD off1, DWORD yo, DWORD endian) {
    volatile DWORD *cb = (volatile DWORD *)(g_cmdVa + slot * SLOT_BYTES); const DWORD *img = g_boxImgs[ver & 1];
    WORD *pix = g_slotPix + slot * BOX_MAX_WRITES; DWORD cnt = g_listCnt[ver & 1];
    DWORD lay = g_listLay[ver & 1];
    DWORD key = off0 | (yo << 2) | endian;  // off0 is a multiple of 4096, so the low bits are free
    SlotTag &t = g_hot.slot[slot];
#if BOX_BACKGROUND
    bool hdrOk = (t.fb == fb && t.key == key && t.key2 == off1 && t.lay == lay);
#else
    bool hdrOk = false;                     // the lit pixels change with the number
#endif
    t.ver = 0;                              // data is about to change
    if (hdrOk) {
        DIAG_ADD(D_FILL_DATA);
        for (DWORD w = 0; w < cnt; w++) cb[w * 3 + 2] = img[pix[w]];
    } else {
        DIAG_ADD(D_FILL_FULL); t.fb = 0; const DWORD *mask = g_wMask[ver & 1];
        DWORD base[2] = { PhysOf(fb + off0), off1 ? PhysOf(fb + off1) : 0 };
        DWORD rows = yo ? 2 : 1;            // tile rows touched
        DWORD n = 0, w = 0;
        for (DWORD r = 0; r < rows; r++) {
            for (DWORD tile = 0; tile < 2; tile++) {
                DWORD any = 0;                      // nothing masked in this tile: no packets
                for (DWORD ly = 0; ly < 32; ly++) {
                    DWORD by = ly + (r ? 32 : 0) - yo;
                    if (by < 32) any |= mask[(by << 1) | tile];
                }
                if (!any) continue;
                DWORD tb = base[r] + tile * 4096;
                for (DWORD o = 0; o < 1024; o++) {
                    DWORD inv = g_invTab[o]; DWORD lx = inv & 31;
                    DWORD by  = (inv >> 5) + (r ? 32 : 0) - yo;   // box row; out of range (wraps) outside the box
                    if (by >= 32) continue;
                    if (!((mask[(by << 1) | tile] >> lx) & 1)) continue;
                    DWORD ii = g_tiledTab[(by << 6) | (tile << 5) | lx] >> 2; pix[w++] = (WORD)ii;
                    cb[n++]  = 0xC0013D00;                    // MEM_WRITE, one data dword
                    cb[n++]  = (tb + (o << 2)) | endian;      // physical address | frame endian bits
                    cb[n++]  = img[ii];
                }
            }
        }
    }
    CommitSlot(t, hdrOk, fb, key, off1, lay, ver);
}

// Bits of mask word w that lie inside columns x0..x1-1 (the word must overlap them)
static __forceinline DWORD ColMask(DWORD w, DWORD x0, DWORD x1) {
    DWORD base = w << 5; DWORD lo = (x0 > base) ? x0 - base : 0; DWORD hi = (x1 < base + 32) ? x1 - base : 32;
    return (0xFFFFFFFFu >> (32 - (hi - lo))) << lo;
}

// Writes one packet per masked pixel of a region, tile by tile in address order, and returns the count. pix / loc
// (both 0 = off) receive the list for replaying the region into other slots
static DWORD WalkRegion(volatile DWORD *cb, const DWORD *img, const DWORD *mask, const DetRgn &g, DWORD yo, DWORD endian, const DWORD *tbase, WORD *pix, PktLoc *loc) {
    DWORD n = 0;
    for (DWORD r = (g.y0 + yo) >> 5; r <= (g.y1 - 1 + yo) >> 5; r++) {
        for (DWORD tx = g.x0 >> 5; tx <= (g.x1 - 1) >> 5; tx++) {
            // occ: one bit per dword offset of this destination tile that gets a packet
            DWORD occ[32]; DWORD any = 0; DWORD cm = ColMask(tx, g.x0, g.x1);
            for (DWORD q = 0; q < 32; q++) occ[q] = 0;
            for (DWORD ly = 0; ly < 32; ly++) {
                DWORD by = ly + (r << 5) - yo;
                if (by < g.y0 || by >= g.y1) continue;
                DWORD m = mask[by * DET_TILES_W + tx] & cm; any |= m;
                while (m) {
                    DWORD o = g_tiledTab[(ly << 6) | LowBit(m)] >> 2; occ[o >> 5] |= 1u << (o & 31); m &= m - 1;
                }
            }
            if (!any) continue;                                 // empty tile
            DWORD tile = r * DET_TILES_W + tx; DWORD tb = tbase[tile];
            for (DWORD q = 0; q < 32; q++) {                    // ascending offsets = address order
                DWORD m = occ[q];
                while (m) {
                    DWORD o = (q << 5) | LowBit(m); m &= m - 1; DWORD inv = g_invTab[o]; DWORD lx = inv & 31;
                    DWORD by  = (inv >> 5) + (r << 5) - yo;
                    DWORD ii  = ((by >> 5) * DET_TILES_W + tx) * 1024 + (g_tiledTab[((by & 31) << 6) | lx] >> 2);
                    if (pix) { pix[n] = (WORD)ii; loc[n] = (PktLoc)((tile << 10) | o); }
                    cb[n * 3 + 0] = 0xC0013D00;                 // MEM_WRITE, one data dword
                    cb[n * 3 + 1] = (tb + (o << 2)) | endian; cb[n * 3 + 2] = img[ii]; n++;
                }
            }
        }
    }
    return n;
}

// FillSlot for the detailed box; tpr = tiles per memory row. A slot already holding this layout for this buffer gets
// only the regions whose stamp changed, otherwise all of them. Unused capacity of a region is covered by one NOP
// packet. record = more slots will be filled from this image, so keep each walked region's list for them. Only the
// Worker fills streams, so lists are never read while rewritten
static void FillDetailSlot(DWORD slot, DWORD fb, DWORD ver, DWORD off0, DWORD off1, DWORD yo, DWORD endian, DWORD tpr, bool record) {
    DWORD ds = slot - SLOT_COUNT; DWORD par = ver & 1;
    volatile DWORD *cb = (volatile DWORD *)(g_dCmdVa + ds * DSLOT_BYTES);
    const DWORD *img = g_detImgs + par * DET_IMG_DWORDS; DWORD lay = g_detLayId[par];
    DWORD key = off0 | (yo << 2) | endian; DWORD key2 = off1 | tpr; SlotTag &t = g_hot.slot[slot];
    DWORD *have = g_slotStamp[ds]; bool hdrOk = (t.fb == fb && t.key == key && t.key2 == key2 && t.lay == lay);
    t.ver = 0;                                      // data is about to change
    if (hdrOk) {
        DIAG_ADD(D_FILL_DATA);
    } else {
        DIAG_ADD(D_FILL_FULL); t.fb = 0;
        for (DWORD r = 0; r < DET_REGIONS; r++) have[r] = 0;
    }
    DWORD p0 = PhysOf(fb + off0); DWORD rows = DET_TILES_H + (yo ? 1 : 0);
    DWORD tbase[(DET_TILES_H + 1) * DET_TILES_W];   // physical address of each tile of the box
    for (DWORD r = 0; r < rows; r++)
        for (DWORD tx = 0; tx < DET_TILES_W; tx++) tbase[r * DET_TILES_W + tx] = p0 + (r * tpr + tx) * 4096;
    for (DWORD r = 0; r < g_rgnN[par]; r++) {
        const DetRgn &g = g_rgn[par][r];
        if (have[r] == g.stamp) continue;           // the slot already holds this content
        volatile DWORD *q = cb + g.off * 3; WORD *pix = g_dPix + par * DET_MAX_WRITES + g.off;
        PktLoc *loc = g_dLoc + par * DET_MAX_WRITES + g.off; DWORD n;
        if (g_lstStamp[par][r] == g.stamp && g_lstYo[par][r] == yo) {
            DIAG_ADD(D_FILL_LIST); n = g.used;
            for (DWORD w = 0; w < n; w++) {
                DWORD l = loc[w];
                q[w * 3 + 0] = 0xC0013D00;          // MEM_WRITE, one data dword
                q[w * 3 + 1] = (tbase[l >> 10] + ((l & 1023) << 2)) | endian; q[w * 3 + 2] = img[pix[w]];
            }
        } else {
            const DWORD *mask = (r >= DET_STATIC_RGNS) ? g_detMask[par] : g_detStat[par];
            n = WalkRegion(q, img, mask, g, yo, endian, tbase, record ? pix : 0, record ? loc : 0);
            if (record) { g_lstStamp[par][r] = g.stamp; g_lstYo[par][r] = yo; }
        }
        if (n < g.cap) q[n * 3] = 0xC0001000 | (((g.cap - n) * 3 - 2) << 16);   // NOP over the rest of the region
        DIAG_SET(D_FILL_PKTS, g_diag[D_FILL_PKTS] + n); have[r] = g.stamp;
    }
    CommitSlot(t, hdrOk, fb, key, key2, lay, ver);
}

static __forceinline DWORD SlotIndex(DWORD mode, DWORD e, DWORD ver) {
    return ((mode ? 1 : 0) * FB_ENTRIES + e) * 2 + (ver & 1);
}

static __forceinline void FillStream(DWORD mode, DWORD slot, DWORD fb, DWORD ver, DWORD off, DWORD off1, DWORD yo,
                                     DWORD endian, DWORD cols, bool record) {
    if (mode) FillDetailSlot(slot, fb, ver, off, off1, yo, endian, cols, record);
    else      FillSlot(slot, fb, ver, off, off1, yo, endian);
}

// Hook side, slow path (both fast records missed): finds or assigns this buffer's entry. Returns the fast record
// holding the slot and its stream address, or 0 when the buffer has no stream for this image version yet. The hook
// never builds streams (a full detailed fill is thousands of packets): it sets g_missReq, the Worker builds the
// stream within one poll, and the overlay is skipped for those frames
static const FastRec *StreamSlow(DWORD fb, DWORD ver, DWORD mode, DWORD key, DWORD key2, DWORD geom) {
    DIAG_ADD(D_FAST_MISS);
    DWORD e = 0;                            // find or assign this buffer's entry
    for (; e < FB_ENTRIES; e++) if (g_hot.entry[e].fb == fb) break;
    if (e == FB_ENTRIES) {
        e = g_fbNext++ % FB_ENTRIES;
        g_hot.fast[0].fb    = 0;            // the Worker may now rebuild this entry's slots for the new buffer,
        g_hot.fast[1].fb    = 0;            // so no record may still point at them
        g_hot.entry[e].fb   = 0;            // unpublish, so the Worker never pairs the old buffer with the new shape
        __sync();
        g_hot.entry[e].geom = geom;
        __sync();
        g_hot.entry[e].fb   = fb;           // published last
    } else if (g_hot.entry[e].geom != geom) {
        g_hot.entry[e].geom = geom;
    }
    DWORD slot = SlotIndex(mode, e, ver);
    if (!SlotValid(slot, fb, ver, key, key2)) {
        g_missReq = 1; DIAG_ADD(D_STREAM_MISS);
        return 0;
    }
    // Remember the answer: f0 moves down to f1 unless it is for the same slot
    FastRec &f0 = g_hot.fast[0]; FastRec &f1 = g_hot.fast[1];
    if (f0.slot != slot) {
        f1.fb = f0.fb; f1.ver = f0.ver; f1.key = f0.key; f1.key2 = f0.key2; f1.slot = f0.slot; f1.phys = f0.phys;
    }
    f0.fb = fb; f0.ver = ver; f0.key = key; f0.key2 = key2; f0.slot = slot; f0.phys = g_hot.slotPhys[slot];
    return &f0;
}

// Hook side: the fast record (slot + stream address) for this buffer and image version, 0 if there is no stream yet
static __forceinline const FastRec *StreamFor(DWORD fb, DWORD ver, DWORD mode, DWORD boxOff, DWORD boxOff1, DWORD boxYo, DWORD endian, DWORD geom) {
    DWORD key = boxOff | (boxYo << 2) | endian; DWORD key2 = boxOff1 | (mode ? ((geom >> 12) & 0xFF) : 0);
    const FastRec &f0 = g_hot.fast[0];
    if (f0.fb == fb && f0.ver == ver && f0.key == key && f0.key2 == key2) return &f0;
    const FastRec &f1 = g_hot.fast[1];
    if (f1.fb == fb && f1.ver == ver && f1.key == key && f1.key2 == key2) return &f1;
    return StreamSlow(fb, ver, mode, key, key2, geom);
}

// Worker side: builds streams of one image version for the known buffers. Called with ver + 1 after a build (those
// slots have the other parity, last used at least 0.5 s ago) and with the current version for invalid slots only.
// force = also while hidden and for inactive buffers
static void PrebuildStreams(DWORD ver, bool force) {
    if (!g_cmdVa || !ver || !g_listCnt[ver & 1]) return;
    if (!force && !g_overlayOn) return;
    DWORD cfg = g_hot.imgCfg[ver & 1]; DWORD mode = cfg >> 2;
    if (mode && !g_dCmdVa) return;
    DWORD active = g_activeMask;
    DWORD pending = 0;                      // slots to fill: pass 0 counts them, pass 1 fills
    for (DWORD pass = 0; pass < 2; pass++) {
        for (DWORD e = 0; e < FB_ENTRIES; e++) {
            DWORD fb = g_hot.entry[e].fb;
            if (!fb) continue;
            // skip buffers not presented lately (either mode's bit counts, so a mode switch still prebuilds)
            if (!force && !(((active >> e) | (active >> (FB_ENTRIES + e))) & 1)) {
                if (pass) DIAG_ADD(D_PRE_SKIP);
                continue;
            }
            DWORD geom = g_hot.entry[e].geom; DWORD cols = (geom >> 12) & 0xFF;
            if (!cols) continue;
            DWORD hgt = geom & 0xFFF; DWORD endi = (geom >> 20) & 3; DWORD off, off1, yo;
            BoxPos(cfg, cols, hgt, off, off1, yo);      // position for this image's corner and mode
            DWORD key = off | (yo << 2) | endi; DWORD slot = SlotIndex(mode, e, ver);
            if (SlotValid(slot, fb, ver, key, off1 | (mode ? cols : 0))) continue;
            if (!pass) { pending++; continue; }
            TMARK(tf); FillStream(mode, slot, fb, ver, off, off1, yo, endi, cols, pending > 1);
            WSTAT(D_T_W_FILL, TNOW() - tf); DIAG_ADD(D_PREBUILT);
        }
    }
}

// ======== Hook ========
typedef DWORD (*VdSwapFn)(PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID, PVOID);
// REG_TO_MEM: the GPU copies one register to physical address dst (| 2 = endian bits)
static __forceinline void RegToMem(volatile DWORD *q, DWORD &u, DWORD reg, DWORD dst) {
    q[u++] = 0xC0013E00; q[u++] = reg; q[u++] = dst | 2;
}

// a = VdSwap's 64-dword ring block, b = front buffer texture fetch constant, f = front buffer address pointer
static DWORD HookVdSwap(PVOID a, PVOID b, PVOID c, PVOID d, PVOID e, PVOID f, PVOID g, PVOID h, PVOID i, PVOID j) {
    g_frames = g_frames + 1;                     // D_FRAMES
    if (!g_overlayOn) {                          // hidden: count the frame and let VdSwap run untouched
        DIAG_ADD(D_HIDDEN);
#if DIAG_ENABLED
        g_fLast = 0;                             // pacing is timed only while shown
#endif
        return ((VdSwapFn)(PVOID)&g_tramp[0])(a, b, c, d, e, f, g, h, i, j);
    }
#if DIAG_ENABLED
    ULONGLONG tp0 = __mftb();   // hook entry
    DWORD tl = (DWORD)tp0;      // lap timer (TLAP)
    FramePacing((DWORD)tp0);
#endif
    // *f (front buffer address) and the four fetch constant dwords at b are read in one guarded block inline (no
    // call or by-reference outputs: __forceinline is often refused for a function holding __try). A bad address
    // leaves 0xDEADDEAD in all five, which the frame check rejects. All loads are issued before any store, so
    // their cache misses overlap
    DWORD fb = 0xDEADDEAD, d0 = 0xDEADDEAD, d1 = 0xDEADDEAD, d2 = 0xDEADDEAD, d3 = 0xDEADDEAD;
    {
        DWORD fa = (DWORD)f, ba = (DWORD)b;
        if (PeekOk(fa) && PeekOk(ba) && PeekOk(ba + 12)) {
            __try {
                volatile DWORD *bp = (volatile DWORD *)ba; DWORD tf = *(volatile DWORD *)fa;
                DWORD t0 = bp[0], t1 = bp[1], t2 = bp[2], t3 = bp[3]; fb = tf; d0 = t0; d1 = t1; d2 = t2; d3 = t3;
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                fb = d0 = d1 = d2 = d3 = 0xDEADDEAD;
            }
        }
    }
    TLAP(D_T_PEEK, tl);
    DWORD ver = g_boxVer;                        // read once: image, cfg and stream tags must agree
    DWORD cfg = g_hot.imgCfg[ver & 1]; GateRec &gr = g_hot.gate;
    if (gr.fb != fb || gr.d0 != d0 || gr.d1 != d1 || gr.d2 != d2 || gr.d3 != d3 || gr.cfg != cfg) {
        DIAG_ADD(D_GATE_MISS); DWORD pitch = ((d0 >> 22) & 0x1FF) * 32; DWORD fmt = d1 & 0x3F;
        DWORD wid = (d2 & 0x1FFF) + 1; DWORD hgt = ((d2 >> 13) & 0x1FFF) + 1;
        // Frame check: tiled 32-bit supported format, pitch == width (multiple of 32), size within MODE_*
        bool okNew = ((d0 >> 31) == 1 && (fmt == 6 || fmt == 50 || fmt == 7 || fmt == 54) && pitch == wid &&
                      (wid & 31) == 0 && wid >= MODE_MIN_W && wid <= MODE_MAX_W &&
                      hgt >= MODE_MIN_H && hgt <= MODE_MAX_H &&
                      fb >= 0xA0000000 && (fb & 0xFFF) == 0 && fb == (d1 & 0xFFFFF000));
        DWORD cols = wid >> 5;                   // tiles across
        DWORD bo = 0, bo1 = 0, byo = 0;
        if (okNew) BoxPos(cfg, cols, hgt, bo, bo1, byo);
        gr.ok = okNew ? 1 : 0; gr.pf = okNew ? PixDescOf(fmt, d3) : 0;
        gr.geom = hgt | (cols << 12) | (((d1 >> 6) & 3) << 20); gr.boxOff = bo; gr.boxOff1 = bo1;
        gr.boxYo   = byo;
        gr.fb = fb; gr.d0 = d0; gr.d1 = d1; gr.d2 = d2; gr.d3 = d3; gr.cfg = cfg;   // key last
#if DIAG_ENABLED
        if (okNew) { g_diag[D_OK_D1] = d1; g_diag[D_OK_D3] = d3; g_diag[D_FB] = fb; }   // they only change with the record
#endif
    }
    bool ok = (gr.ok != 0); DWORD pf = gr.pf; DWORD geom = gr.geom;
    DWORD boxOff = gr.boxOff, boxOff1 = gr.boxOff1, boxYo = gr.boxYo; DWORD mode = cfg >> 2;
    DWORD fbEndian = (geom >> 20) & 3;
    bool  fmtOk    = true;                       // false = the image was built for another pixel layout
    if (ok) {
        if (g_pixFmt != pf) g_pixFmt = pf;       // the Worker rebuilds the image for it
        fmtOk = (g_imgFmt[ver & 1] == pf);
        if (g_overlayOn) {
#if DIAG_ENABLED
            RecordFrame((DWORD)tp0);
#else
            RecordFrame((DWORD)__mftb());
#endif
        }
    }
    TLAP(D_T_GATE, tl);
#if DIAG_ENABLED
    if (!ok) {
        COUNT(D_REJ_GATE);
        g_diag[D_LAST_D0] = d0; g_diag[D_LAST_D1] = d1; g_diag[D_LAST_D2] = d2; g_diag[D_LAST_FB] = fb;
        g_diag[D_LAST_B] = (DWORD)b; g_diag[D_LAST_F] = (DWORD)f;
    }
    ULONGLONG tpre = __mftb(); DWORD pre = (DWORD)(tpre - tp0); g_diag[D_PRE_TB_SUM] = g_diag[D_PRE_TB_SUM] + pre;
    if (pre > g_diag[D_PRE_TB_MAX]) g_diag[D_PRE_TB_MAX] = pre;
    if (pre > SLOW_TICKS) COUNT(D_SLOW_PRE);
    tl = (DWORD)tpre;
#endif
    DWORD ret = ((VdSwapFn)(PVOID)&g_tramp[0])(a, b, c, d, e, f, g, h, i, j);
#if DIAG_ENABLED
    ULONGLONG tb0 = __mftb();   // from here on only the overlay's own work is timed
    TSTAT(D_T_SWAP, (DWORD)tb0 - tl); tl = (DWORD)tb0;
#endif
    if (ok && (!g_overlayOn || !fmtOk)) {
        DIAG_ADD(D_HIDDEN);
    } else if (ok && (((DWORD)a) & 3) == 0 && ((DWORD)a) >= 0xA0000000) {
        __try {
            volatile DWORD *rb = (volatile DWORD *)a; DWORD r0 = rb[0]; DWORD r4 = rb[4];
            // Where VdSwap put its fence packet (kernel VdSwap dump, 0x800C8B90):
            //   k = 0   normal: rb[0..3] fence, rb[4] NOP header (C03A1000)
            //   k = 10  D3D passed its own swap command buffer: rb[0..6] fetch constant 15 write,
            //           rb[7..9] INDIRECT_BUFFER, rb[10..13] fence, rb[14] NOP header (C0301000)
            DWORD k = 0xFFFFFFFF;
            if (r0 == 0xC0025800 && r4 == 0xC03A1000) k = 0;
            else if (rb[7] == 0xC0013F00 && rb[10] == 0xC0025800 && rb[14] == 0xC0301000) k = 10;
            TLAP(D_T_RINGCHK, tl);
            const FastRec *sr = 0;                          // slot and stream address of this frame's stream
            if (g_cmdPhys && ver && fb >= 0xA0000000u && k != 0xFFFFFFFF)
                sr = StreamFor(fb, ver, mode, boxOff, boxOff1, boxYo, fbEndian, geom);
            if (sr) {
                TLAP(D_T_STREAM, tl);
                volatile DWORD *q = rb + k;                 // q[0..3] = fence packet, q[4] = NOP header
                DWORD u = 4;                                // next free dword
                // One probe sample per Worker tick: CP clock before and after the stream, RBBM counters after
                bool sample = (g_probePhys != 0) && (g_probeWant || g_probeArm);
                if (sample) {
                    if (g_probeArm) {                       // counter setup as D3D's P_SPCE sends it (header = register)
                        g_probeArm = 0;
                        q[u++] = 0x000001E6; q[u++] = 0;                            // CP counter 0: CP_COUNT
                        q[u++] = 0x00000395; q[u++] = g_probeSel[0] & 0xFF;         // RBBM counter 0 event
                        q[u++] = 0x00000396; q[u++] = g_probeSel[1] & 0xFF;         // RBBM counter 1 event
                        q[u++] = 0x000001F5; q[u++] = 0;                            // CP_PERFMON_CNTL = 0
                        q[u++] = 0x000021F9; q[u++] = 23;                           // PERFCOUNTER_START
                        q[u++] = 0x000001F5; q[u++] = 1;                            // CP_PERFMON_CNTL = 1
                        DIAG_ADD(D_PRB_SETUP);
                    }
                    g_probeWant = 0; RegToMem(q, u, 0x1E7, g_probePhys + 12);
                }
                q[u++] = 0xC0013F00;                        // INDIRECT_BUFFER
                q[u++] = sr->phys;
                q[u++] = g_listCnt[ver & 1] * 3;            // length in dwords
                if (sample) {
                    RegToMem(q, u, 0x1E7, g_probePhys + 0); RegToMem(q, u, 0x397, g_probePhys + 4);
                    RegToMem(q, u, 0x399, g_probePhys + 8);
                }
                q[u] = 0xC0001000 | ((62 - k - u) << 16);   // NOP over the rest, up to rb[63]
                TLAP(D_T_WRITE, tl);
                __sync();                                   // always the full barrier: __lwsync / __eieio measured no faster and risk missed frames or flicker
                TLAP(D_T_FENCE, tl);
                DWORD sb = 1u << (sr->slot >> 1);           // tell the Worker this buffer is live
                if (!(g_hot.seen & sb)) g_hot.seen |= sb;
                DIAG_ADD(D_QUEUED);
            } else {
#if DIAG_ENABLED
                COUNT(D_SKIPPED);
                if (!g_cmdPhys)             COUNT(D_REJ_NOCMD);
                else if (!ver)              COUNT(D_REJ_NOIMG);
                else if (fb < 0xA0000000u)  COUNT(D_REJ_FB);
                else if (k != 0xFFFFFFFF)   { }     // no ready stream yet: counted as D_STREAM_MISS in StreamSlow
                else {
                    COUNT(D_REJ_RING); g_diag[D_LAST_RB0] = r0; g_diag[D_LAST_RB4] = r4;
                    g_diag[D_BAD_A]    = (DWORD)a;
                    for (DWORD q = 0; q < 16; q++) g_diag[D_BAD0 + q] = rb[q];
                }
#endif
            }
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            DIAG_ADD(D_SKIPPED); DIAG_ADD(D_EXC);
        }
    } else if (ok) {
        DIAG_ADD(D_REJ_PTR); DIAG_SET(D_LAST_A, (DWORD)a);
    }
#if DIAG_ENABLED
    DWORD dt = (DWORD)(__mftb() - tb0); g_diag[D_HOOK_TB_SUM] = g_diag[D_HOOK_TB_SUM] + dt; COUNT(D_HOOK_TB_N);
    if (dt > g_diag[D_HOOK_TB_MAX]) g_diag[D_HOOK_TB_MAX] = dt;
    if (dt > SLOW_TICKS) COUNT(D_SLOW_POST);
#endif
    return ret;
}

// ======== FPS image ========
// 3x5 digits, 15 bits each, top left = bit 14, drawn at 4x
static const DWORD kDigits[10] = {
    0x7B6F, 0x2C97, 0x73E7, 0x73CF, 0x5BC9, 0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BCF
};

// Byte offset of pixel (x, y) in a 32-bit tiled surface (Xenia texture_address.h Tiled2D)
static DWORD TiledOffCalc(DWORD x, DWORD y) {
    DWORD outer = ((y >> 5) * 40 + (x >> 5)) << 6; DWORD inner = (((y >> 1) & 7) << 3) | (x & 7);
    DWORD oib = (outer | inner) << 2; DWORD bank = (y >> 4) & 1;
    DWORD pipe  = ((x >> 3) & 3) ^ (((y >> 3) & 1) << 1);
    return ((y & 1) << 4) | (pipe << 6) | (bank << 11)
         | (oib & 15) | (((oib >> 4) & 1) << 5) | (((oib >> 5) & 7) << 8)
         | ((oib >> 8) << 12);
}

static void InitTiledTab() {
    for (DWORD y = 0; y < 32; y++)
        for (DWORD x = 0; x < 64; x++) g_tiledTab[(y << 6) | x] = (WORD)TiledOffCalc(x, y);
    for (DWORD y = 0; y < 32; y++)
        for (DWORD x = 0; x < 32; x++) g_invTab[g_tiledTab[(y << 6) | x] >> 2] = (WORD)((y << 5) | x);
}

// Builds the 64x32 image (tiled order) and its write mask. The digits plus margins form a rectangle BOX_EDGE_GAP px
// from the corner's screen edges. Returns the mask pixel count; *lay = digits | corner << 2
static DWORD BuildImage(DWORD *img, DWORD *mask, DWORD fps, DWORD corner, DWORD *lay) {
    DWORD d[3]  = { (fps / 100) % 10, (fps / 10) % 10, fps % 10 };
    DWORD first = (fps >= 100) ? 0 : ((fps >= 10) ? 1 : 2);   // skip leading zeros
    DWORD n     = 3 - first;
    DWORD rw    = n * 16 - 4 + 2 * BG_MARGIN_X;               // 12 px digits, 4 px gaps, plus margins
    DWORD rh = 20 + 2 * BG_MARGIN_Y; DWORD rx0 = (corner & 1) ? 64 - BOX_EDGE_GAP - rw : BOX_EDGE_GAP;
    DWORD ry0   = (corner & 2) ? 32 - BOX_EDGE_GAP - rh : BOX_EDGE_GAP;
    DWORD sx    = rx0 + BG_MARGIN_X;                          // top left of the digits
    DWORD sy = ry0 + BG_MARGIN_Y; DWORD cnt = 0;
    const DWORD textPix = BOX_TEXT_PIX, bgPix = BOX_BG_PIX;   // converted once, not per pixel
    for (DWORD i = 0; i < 64; i++) mask[i] = 0;
    for (DWORD y = 0; y < 32; y++) {
        for (DWORD x = 0; x < 64; x++) {
            bool lit = false;
            if (y >= sy && y < sy + 20) {
                DWORD row = (y - sy) >> 2;
                for (DWORD k = first; k < 3; k++) {
                    DWORD x0 = sx + (k - first) * 16;
                    if (x >= x0 && x < x0 + 12) {
                        DWORD c = (x - x0) >> 2;
                        if ((kDigits[d[k]] >> (14 - (row * 3 + c))) & 1) lit = true;
                    }
                }
            }
            img[g_tiledTab[(y << 6) | x] >> 2] = lit ? textPix : bgPix;
#if BOX_BACKGROUND
            bool inList = (x >= rx0 && x < rx0 + rw && y >= ry0 && y < ry0 + rh);
#else
            bool inList = lit;
#endif
            if (inList) { mask[(y << 1) | (x >> 5)] |= 1u << (x & 31); cnt++; }
        }
    }
    *lay = n | (corner << 2);
    return cnt;
}

// Publishes the image just built into the spare parity: records its metadata, prebuilds its streams, flips the
// version. cnt = stream length in packets (detailed: including NOP padding), used = MEM_WRITE packets
static void PublishImage(DWORD cnt, DWORD used, DWORD lay, DWORD cfg) {
    DWORD ver = g_boxVer; DWORD nxt = (ver + 1) & 1; g_listCnt[nxt] = cnt; g_listLay[nxt] = lay;
    g_hot.imgCfg[nxt] = cfg; g_imgFmt[nxt] = g_buildFmt; g_diag[D_RUNS] = used;
    g_diag[D_STREAMLEN] = cnt * 3;
#if DIAG_ENABLED
    {
        DWORD pm = (cfg >> 2) & 3;          // display mode of this image (FPS-only images pass the corner alone: mode 0)
        if (pm > 2) pm = 2;
        DIAG_MAX(D_PKT_PEAK0 + pm, used);
    }
#endif
    __sync();
    STAGE(8); TMARK(tp); PrebuildStreams(ver + 1, false); WSTAT_OUTER(D_T_W_PREBUILD, TNOW() - tp); STAGE(9);
    __sync();
    g_boxVer = ver + 1; g_diag[D_VERSION] = ver + 1; COUNT(D_BUILDS);
}

static void BuildBoxImage(DWORD fps, DWORD corner) {
    g_buildFmt = g_pixFmt; DWORD nxt = (g_boxVer + 1) & 1; DWORD lay = 0; TMARK(tb);
    DWORD cnt = BuildImage(g_boxImgs[nxt], g_wMask[nxt], fps, corner, &lay); WSTAT(D_T_W_BUILD, TNOW() - tb);
    DIAG_ADD(D_IMG_FPS);
    PublishImage(cnt, cnt, lay, corner);   // cfg = corner, mode 0
}

// ======== Detailed image (modes 1 and 2) ========
// Panel row order
enum { DV_GPU, DV_GPUT, DV_CPU0, DV_CPU1, DV_CPU2, DV_CPU3, DV_CPU4, DV_CPU5, DV_CPU, DV_CPUT, DV_RAM, DV_FPS, DV_MIN, DV_AVG, DV_MAX, DV_LOW };
#if DET_ROWS != 16
  #error DET_ROWS must match the DV_* rows
#endif
struct DetailData { DWORD v[DET_ROWS]; };
static DetailData g_dd;             // what the panel should show (Worker only)
static DetailData g_ddBuilt;        // values of the latest detailed build
static DetailData g_detShown[2];    // values each image holds
static DWORD      g_detKey[2];      // layout of each image (corner, pixel format, mode), 0 = never built
static DWORD      g_shCur[2][2];    // graph readout, axis top and history sequence each image holds
static DWORD      g_shTop[2][2];
static DWORD      g_shSeq[2];
static DWORD      g_dTick = 0;      // paces the SMC reads

// No struct assignment: it could pull in memcpy
static void CopyData(DetailData &dst, const DetailData &src) { for (DWORD i = 0; i < DET_ROWS; i++) dst.v[i] = src.v[i]; }

// 5x7 glyphs, one byte per row, bit 4 = leftmost; '^' is the degree sign
struct Glyph { char ch; BYTE row[7]; };
static const Glyph kGlyphs[] = {
    { ' ', { 0x00,0x00,0x00,0x00,0x00,0x00,0x00 } }, { '0', { 0x0E,0x11,0x13,0x15,0x19,0x11,0x0E } },
    { '1', { 0x04,0x0C,0x04,0x04,0x04,0x04,0x0E } }, { '2', { 0x0E,0x11,0x01,0x02,0x04,0x08,0x1F } },
    { '3', { 0x0E,0x11,0x01,0x06,0x01,0x11,0x0E } }, { '4', { 0x02,0x06,0x0A,0x12,0x1F,0x02,0x02 } },
    { '5', { 0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E } }, { '6', { 0x06,0x08,0x10,0x1E,0x11,0x11,0x0E } },
    { '7', { 0x1F,0x01,0x02,0x04,0x08,0x08,0x08 } }, { '8', { 0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E } },
    { '9', { 0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C } }, { 'A', { 0x0E,0x11,0x11,0x1F,0x11,0x11,0x11 } },
    { 'B', { 0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E } }, { 'C', { 0x0E,0x11,0x10,0x10,0x10,0x11,0x0E } },
    { 'E', { 0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F } }, { 'F', { 0x1F,0x10,0x10,0x1E,0x10,0x10,0x10 } },
    { 'G', { 0x0E,0x11,0x10,0x17,0x11,0x11,0x0F } }, { 'I', { 0x0E,0x04,0x04,0x04,0x04,0x04,0x0E } },
    { 'L', { 0x10,0x10,0x10,0x10,0x10,0x10,0x1F } }, { 'M', { 0x11,0x1B,0x15,0x15,0x11,0x11,0x11 } },
    { 'N', { 0x11,0x19,0x15,0x13,0x11,0x11,0x11 } }, { 'P', { 0x1E,0x11,0x11,0x1E,0x10,0x10,0x10 } },
    { 'R', { 0x1E,0x11,0x11,0x1E,0x14,0x12,0x11 } }, { 'S', { 0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E } },
    { 'T', { 0x1F,0x04,0x04,0x04,0x04,0x04,0x04 } }, { 'U', { 0x11,0x11,0x11,0x11,0x11,0x11,0x0E } },
    { 'V', { 0x11,0x11,0x11,0x11,0x11,0x0A,0x04 } }, { 'X', { 0x11,0x11,0x0A,0x04,0x0A,0x11,0x11 } },
    { 'o', { 0x00,0x00,0x0E,0x11,0x11,0x11,0x0E } }, { 'w', { 0x00,0x00,0x11,0x11,0x15,0x15,0x0A } },
    { 'm', { 0x00,0x00,0x1A,0x15,0x15,0x15,0x15 } }, { 's', { 0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E } },
    { '.', { 0x00,0x00,0x00,0x00,0x00,0x0C,0x0C } }, { '%', { 0x19,0x1A,0x02,0x04,0x08,0x0B,0x13 } },
    { '^', { 0x0C,0x12,0x12,0x0C,0x00,0x00,0x00 } },
};

// 1.25x font: the glyphs are resampled to 6x9 once (InitGlyphTab) by doubling the middle column and rows 1 and 5,
// which stays symmetric and keeps the horizontal bars of E, F, S, 8 etc. 1 px thick
static const BYTE kSrcCol[DET_GLYPH_W] = { 0, 1, 2, 2, 3, 4 };
static const BYTE kSrcRow[DET_GLYPH_H] = { 0, 1, 1, 2, 3, 4, 5, 5, 6 };
static BYTE g_glyphIdx[128];            // glyph number + 1 per character code, 0 = none
static BYTE g_glyphScaled[sizeof(kGlyphs) / sizeof(kGlyphs[0])][DET_GLYPH_H];   // 6x9 rows, bit 0 = leftmost column
static void InitGlyphTab() {
    for (DWORD i = 0; i < sizeof(kGlyphs) / sizeof(kGlyphs[0]); i++) {
        g_glyphIdx[(BYTE)kGlyphs[i].ch & 127] = (BYTE)(i + 1);
        for (DWORD ry = 0; ry < DET_GLYPH_H; ry++) {
            DWORD src = kGlyphs[i].row[kSrcRow[ry]]; DWORD out = 0;
            for (DWORD cx = 0; cx < DET_GLYPH_W; cx++)
                if ((src >> (4 - kSrcCol[cx])) & 1) out |= 1u << cx;
            g_glyphScaled[i][ry] = (BYTE)out;
        }
    }
}

static __forceinline const BYTE *GlyphRows(char ch) {
    DWORD i = ((BYTE)ch < 128) ? g_glyphIdx[(BYTE)ch] : 0;
    return i ? g_glyphScaled[i - 1] : 0;
}

struct DetRow { const char *label; const char *unit; DWORD labelRgb; DWORD valRgb; };
#define ROW_GPU(l, u) { l, u, DETAIL_GPU_COLOR, DETAIL_VAL_COLOR }
#define ROW_CPU(l, u) { l, u, DETAIL_CPU_COLOR, DETAIL_VAL_COLOR }
#define ROW_FPS(l)    { l, "FPS", DETAIL_FPS_COLOR, DETAIL_FPSVAL_COLOR }
static const DetRow kDetRows[DET_ROWS] = {
    ROW_GPU("GPU", "%"),   ROW_GPU("TEMP", "^C"),
    ROW_CPU("CPU0", "%"),  ROW_CPU("CPU1", "%"),  ROW_CPU("CPU2", "%"),
    ROW_CPU("CPU3", "%"),  ROW_CPU("CPU4", "%"),  ROW_CPU("CPU5", "%"),
    ROW_CPU("CPU", "%"),   ROW_CPU("TEMP", "^C"),
    { "RAM", "MB", DETAIL_RAM_COLOR, DETAIL_VAL_COLOR },
    ROW_FPS("FPS"), ROW_FPS("MIN"), ROW_FPS("AVG"), ROW_FPS("MAX"), ROW_FPS("1% Low"),
};

// img = tiled pixel data, lit = text and graph pixels, out = lit grown by 1 px (the pixels that get packets)
struct Canvas { DWORD *img; DWORD *lit; DWORD *out; };

// Pixel (x, y) of the detailed box -> image dword (tiled like the frame buffer)
static __forceinline DWORD DetIdx(DWORD x, DWORD y) {
    return ((y >> 5) * DET_TILES_W + (x >> 5)) * 1024 + (g_tiledTab[((y & 31) << 6) | (x & 31)] >> 2);
}

static void DetPlot(Canvas &c, DWORD x, DWORD y, DWORD pix) {
    if (x >= DET_W || y >= DET_H) return;
    c.img[DetIdx(x, y)] = pix; c.lit[y * DET_TILES_W + (x >> 5)] |= 1u << (x & 31);
}

// Turns lit pixels off; their image data is replaced when the rectangle is outlined again
static void LitClear(Canvas &c, DWORD x0, DWORD y0, DWORD w, DWORD h) {
    for (DWORD wd = x0 >> 5; wd <= (x0 + w - 1) >> 5; wd++) {
        DWORD keep = ~ColMask(wd, x0, x0 + w);
        for (DWORD y = y0; y < y0 + h; y++) c.lit[y * DET_TILES_W + wd] &= keep;
    }
}

static void HLine(Canvas &c, DWORD x0, DWORD y, DWORD w, DWORD pix) { for (DWORD x = 0; x < w; x++) DetPlot(c, x0 + x, y, pix); }

// The mask is updated once per glyph row, the image once per lit pixel
static void DetChar(Canvas &c, DWORD x, DWORD y, char ch, DWORD pix) {
    const BYTE *g = GlyphRows(ch);          // 9 pre-scaled rows, bit 0 = leftmost of 6 columns
    if (!g || x + DET_GLYPH_W > DET_W) return;
    DWORD *img = c.img; DWORD wd = x >> 5, sh = x & 31;
    for (DWORD ry = 0; ry < DET_GLYPH_H; ry++) {
        DWORD bits = g[ry];
        if (!bits) continue;
        DWORD py = y + ry;
        if (py >= DET_H) continue;
        DWORD *m = c.lit + py * DET_TILES_W + wd; m[0] |= bits << sh;
        if (sh > 32 - DET_GLYPH_W) m[1] |= bits >> (32 - sh);   // the row crosses a mask word
        const WORD *rt   = &g_tiledTab[(py & 31) << 6];  // row address data, once per pixel row
        DWORD       base = (py >> 5) * DET_TILES_W * 1024;
        do {
            DWORD px = x + LowBit(bits); bits &= bits - 1; img[base + ((px >> 5) << 10) + (rt[px & 31] >> 2)] = pix;
        } while (bits);
    }
}

// Character i starts at x + i * 7.5 (integer division: advances alternate 7 and 8 px)
static void DetText(Canvas &c, DWORD x, DWORD y, const char *t, DWORD pix) {
    for (DWORD i = 0; *t; t++, i++) DetChar(c, x + (i * 15) / 2, y, *t, pix);
}

static DWORD DetTextW(const char *t) {
    DWORD n = 0;
    while (t[n]) n++;
    return TEXT_W(n);
}

static DWORD FmtNum(DWORD v, char *out) {
    char tmp[12]; DWORD n = 0;
    do { tmp[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    for (DWORD i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    out[n] = 0;
    return n;
}

static DWORD FmtTenths(DWORD v, char *out)      // 167 -> "16.7"
{
    DWORD n = FmtNum(v / 10, out); out[n++] = '.'; out[n++] = (char)('0' + v % 10); out[n] = 0;
    return n;
}

// ---- Graph history and drawing (mode 2) ----
// One column per GRAPH_COL_TICKS, fed from the frame-time ring only in mode 2. Frame time plots each column's
// longest frame (hitches show as spikes), frame rate plots frames / time
static DWORD g_gFt[GRAPH_COLS];     // 0.1 ms units
static DWORD g_gFps[GRAPH_COLS];    // 0.1 FPS units
static DWORD g_gN     = 0;          // columns filled
static DWORD g_gStart = 0;          // oldest column (0 until the history is full)
static DWORD g_gIdx   = 0;          // g_ftIdx consumed so far
static DWORD g_gbSum  = 0;          // column being filled: ticks, longest frame, frame count
static DWORD g_gbMax  = 0;
static DWORD g_gbCnt  = 0;
static bool  g_gDirty = false;      // history changed since the last image build
static DWORD g_gCurFt = 0;          // average frame time since the last update, 0.1 ms
static DWORD g_gSeq   = 1;          // changes with the history: images redraw their plots when it differs

static void GraphReset(DWORD startIdx)      // startIdx = first ring frame to read next
{
    g_gN = 0; g_gStart = 0; g_gIdx = startIdx; g_gbSum = g_gbMax = g_gbCnt = 0; g_gDirty = true; g_gSeq++;
}

static void GraphUpdate() {
    const ULONGLONG TB = 49875000ULL; DWORD idx = g_ftIdx;
    if (idx < g_gIdx) GraphReset(0);                   // the ring was restarted
    DWORD have = idx - g_gIdx;
    if (have > FT_RING) { g_gIdx = idx - FT_RING; have = FT_RING; }   // fell behind: oldest frames are gone
    ULONGLONG tsum = 0;
    for (DWORD i = 0; i < have; i++) {
        DWORD dt = g_ftRing[(g_gIdx + i) & (FT_RING - 1)];
        if (!dt) dt = 1;
        tsum += dt; g_gbSum += dt; g_gbCnt++;
        if (dt > g_gbMax) g_gbMax = dt;
        if (g_gbSum >= GRAPH_COL_TICKS) {
            DWORD ft = g_gbMax / 4988;
            if (ft > 9999) ft = 9999;
            DWORD fp = (DWORD)(((ULONGLONG)g_gbCnt * TB * 10 + g_gbSum / 2) / g_gbSum);
            if (fp > 9999) fp = 9999;
            DWORD gs;
            if (g_gN == GRAPH_COLS) {                  // full: overwrite the oldest
                gs = g_gStart; g_gStart = (gs + 1 == GRAPH_COLS) ? 0 : gs + 1;
            } else {
                gs = g_gN++;
            }
            g_gFt[gs] = ft; g_gFps[gs] = fp; g_gbSum = g_gbMax = g_gbCnt = 0; g_gDirty = true; g_gSeq++;
        }
    }
    if (have) {
        DWORD cf = (DWORD)((tsum / have + 2494) / 4988); g_gCurFt = (cf > 9999) ? 9999 : cf;
    }
    g_gIdx = idx;
}

// Plot row (0 = top) of value v on a 0..top axis, clipped at the top
static DWORD GraphRow(DWORD v, DWORD top) {
    DWORD h = (DWORD)(((ULONGLONG)(v > top ? top : v) * (GB_IN_H - 1)) / top);
    return (GB_IN_H - 1) - h;
}

// Title and plot border; (bx, by) = block top left
static void GraphStatic(Canvas &c, DWORD bx, DWORD by, const char *title) {
    DWORD px = bx + GB_PAD; DWORD py = by + GB_PAD + DET_GLYPH_H + GB_TITLE_GAP;
    DWORD bd = ColorToPixel(GRAPH_BORDER_COLOR); DetText(c, px, by + GB_PAD, title, ColorToPixel(DETAIL_FPS_COLOR));
    for (DWORD x = 0; x < GB_PLOT_W; x++) { DetPlot(c, px + x, py, bd); DetPlot(c, px + x, py + GB_PLOT_H - 1, bd); }
    for (DWORD y = 0; y < GB_PLOT_H; y++) { DetPlot(c, px, py + y, bd); DetPlot(c, px + GB_PLOT_W - 1, py + y, bd); }
}

// Axis top in 0.1 units. Frame rate: the first of 30/60/120 that holds the peak with 5% headroom, else 240
static DWORD GraphTop(bool rate) {
    if (!rate) return GRAPH_FT_TOP_MS * 10;
    DWORD mx = 0;
    for (DWORD i = 0; i < g_gN; i++) if (g_gFps[i] > mx) mx = g_gFps[i];
    DWORD t = 30;
    while (t < 240 && mx > t * 10 + t / 2) t *= 2;
    return t * 10;
}

// Current value right aligned in the title row (room for 6 characters). rate: false = frame time (cur in 0.1 ms),
// true = frame rate (cur in FPS); g = this readout's region
static void GraphReadout(Canvas &c, const DetRgn &g, bool rate, DWORD cur) {
    char num[16]; DWORD n;
    if (rate) {
        n = FmtNum(cur, num);
    } else {
        n = (cur >= 1000) ? FmtNum(cur / 10, num) : FmtTenths(cur, num);   // "16.7ms", "123ms" from 100 ms
        num[n++] = 'm'; num[n++] = 's'; num[n] = 0;
    }
    DetText(c, g.x1 - 1 - TEXT_W(n), g.y0 + 1, num, ColorToPixel(DETAIL_FPSVAL_COLOR));
}

// Axis labels: top, middle, 0
static void GraphAxis(Canvas &c, const DetRgn &g, DWORD top) {
    DWORD ax = g.x0 + 1, ay = g.y0 + 1; DWORD lc = ColorToPixel(DETAIL_FPSVAL_COLOR);
    DWORD yBot = ay + GB_PLOT_H - DET_GLYPH_H; DWORD yMid = (ay + yBot) / 2; char lbl[12];
    FmtNum(top / 10, lbl); DetText(c, ax, ay, lbl, lc);
    FmtNum(top / 20, lbl); DetText(c, ax, yMid, lbl, lc);
    FmtNum(0, lbl); DetText(c, ax, yBot, lbl, lc);
}

// Grid and line inside the border (the region is the plot including its border)
static void GraphPlot(Canvas &c, const DetRgn &g, bool rate, DWORD top) {
    DWORD px = g.x0 + 1, py = g.y0 + 1; const DWORD *vals = rate ? g_gFps : g_gFt;
    DWORD grid = ColorToPixel(GRAPH_GRID_COLOR);
    HLine(c, px, py + GraphRow(rate ? top / 2 : 167, top), GB_IN_W, grid);        // half scale, or 16.7 ms
    if (!rate) HLine(c, px, py + GraphRow(333, top), GB_IN_W, grid);              // 33.3 ms
    DWORD ln    = ColorToPixel(GRAPH_LINE_COLOR);
    DWORD x0    = GB_IN_W - g_gN;                                  // newest column at the right edge
    DWORD yPrev = 0;
    for (DWORD i = 0; i < g_gN; i++) {
        DWORD si = g_gStart + i;
        if (si >= GRAPH_COLS) si -= GRAPH_COLS;
        DWORD y = GraphRow(vals[si], top);
        DWORD a = (i && yPrev < y) ? yPrev : y;                    // vertical run joining the previous column
        DWORD b = (i && yPrev > y) ? yPrev : y;
        for (DWORD yy = a; yy <= b; yy++) DetPlot(c, px + x0 + i, py + yy, ln);
        yPrev = y;
    }
}

static __forceinline DWORD PopCount32(DWORD v) {
    v = v - ((v >> 1) & 0x55555555u); v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return (((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
}

// Recomputes the outline mask inside a rectangle (lit pixels grown by 1 px in all 8 directions), writes the
// background colour into the halo pixels only, and returns the rectangle's packet count. Lit pixels already hold
// their colour and pixels outside the outline are never read, so the image is never cleared
static DWORD OutlineRect(Canvas &c, DWORD x0, DWORD y0, DWORD x1, DWORD y1) {
    const DWORD bg = BOX_BG_PIX; DWORD *img = c.img; DWORD w0 = x0 >> 5, w1 = (x1 - 1) >> 5; DWORD cnt = 0;
    for (DWORD y = y0; y < y1; y++) {
        const DWORD *mid = c.lit + y * DET_TILES_W; const DWORD *up = y ? mid - DET_TILES_W : mid;
        const DWORD *dn = (y + 1 < DET_H) ? mid + DET_TILES_W : mid; DWORD *out = c.out + y * DET_TILES_W;
        const WORD *rt = &g_tiledTab[(y & 31) << 6]; DWORD base = (y >> 5) * DET_TILES_W * 1024;
        DWORD prev = w0 ? (up[w0 - 1] | mid[w0 - 1] | dn[w0 - 1]) : 0; DWORD cur = up[w0] | mid[w0] | dn[w0];
        for (DWORD w = w0; w <= w1; w++) {
            DWORD next = (w + 1 < DET_TILES_W) ? (up[w + 1] | mid[w + 1] | dn[w + 1]) : 0;
            DWORD cm   = ColMask(w, x0, x1);
            DWORD v    = (cur | (cur << 1) | (cur >> 1) | (prev >> 31) | (next << 31)) & cm;
            out[w] = (out[w] & ~cm) | v; cnt += PopCount32(v);
            for (DWORD halo = v & ~mid[w]; halo; halo &= halo - 1) {
                DWORD b = LowBit(halo); img[base + (w << 10) + (rt[b] >> 2)] = bg;
            }
            prev = cur; cur = next;
        }
    }
    return cnt;
}

// Incremental redraw of a dynamic region: RgnClear erases its lit pixels before the content is redrawn, RgnDone
// re-outlines and restamps it afterwards. Full builds do neither (they outline everything at the end)
static void RgnClear(Canvas &c, const DetRgn &g, DWORD w, DWORD h, bool full) {
    if (!full) LitClear(c, g.x0 + 1, g.y0 + 1, w, h);
}

static void RgnDone(Canvas &c, DetRgn &g, bool full) {
    if (!full) { g.used = OutlineRect(c, g.x0, g.y0, g.x1, g.y1); g.stamp = ++g_detStampCtr; }
}

static void SetRgn(DetRgn &g, DWORD x0, DWORD y0, DWORD x1, DWORD y1) {
    g.x0 = x0; g.y0 = y0; g.x1 = x1; g.y1 = y1; g.cap = (x1 - x0) * (y1 - y0);
}

// The number of panel row r, right aligned in its region
static void DetNumber(Canvas &c, const DetRgn &g, DWORD r, DWORD v) {
    DWORD vmax = (r == DV_RAM) ? 9999 : 999; char num[12]; DWORD n = FmtNum(v > vmax ? vmax : v, num);
    DetText(c, g.x1 - 1 - TEXT_W(n), g.y0 + 1, num, ColorToPixel(kDetRows[r].valRgb));
}

// Region rectangles of a layout and the static content. A dynamic region covers everything its content can light
// plus 1 px for the halo, so redrawing it never changes pixels of another region. Region 0's count is set later
static void DetLayout(Canvas &c, DWORD corner, DWORD par, DWORD mode) {
    DWORD stackH = (mode == 2) ? DET_STACK_H : DET_RECT_H;
    DWORD rx0    = (corner & 1) ? DET_W - BOX_EDGE_GAP - DET_RECT_W : BOX_EDGE_GAP;
    DWORD ry0 = (corner & 2) ? DET_H - BOX_EDGE_GAP - stackH : BOX_EDGE_GAP; DWORD left = rx0 + DET_PAD;
    DWORD right  = rx0 + DET_RECT_W - DET_PAD;
    DWORD gx0    = (corner & 1) ? DET_W - BOX_EDGE_GAP - GB_RECT_W : BOX_EDGE_GAP;   // blocks are wider than the panel
    DetRgn *rg = g_rgn[par];
    for (DWORD b = 0; b < DET_STATIC_RGNS; b++) SetRgn(rg[b], 0, b * 32, DET_W, b * 32 + 32);   // static bands
    for (DWORD r = 0; r < DET_ROWS; r++) {
        const DetRow &row = kDetRows[r]; DWORD y = ry0 + DET_PAD + r * DET_PITCH;
        DWORD ux = right - DetTextW(row.unit);
        DWORD nr = ux - 3;                                      // right edge of the number
        DWORD nw = (r == DV_RAM) ? TEXT_W(4) : TEXT_W(3);
        SetRgn(rg[DET_STATIC_RGNS + r], nr - nw - 1, y - 1, nr + 1, y + DET_GLYPH_H + 1);
        DetText(c, left, y, row.label, ColorToPixel(row.labelRgb));
        DetText(c, ux,   y, row.unit,  ColorToPixel(row.valRgb));
    }
    g_rgnN[par] = DET_PANEL_RGNS;
    if (mode == 2) {
        for (DWORD gi = 0; gi < 2; gi++) {
            DWORD by = ry0 + DET_RECT_H + GB_GAP + gi * (GB_H + GB_GAP);   // frame-time block, frame-rate block
            DWORD px = gx0 + GB_PAD; DWORD ly = by + GB_PAD; DWORD py = ly + DET_GLYPH_H + GB_TITLE_GAP;
            DWORD ax = px + GB_PLOT_W + GB_LBL_GAP; DetRgn *g = rg + DET_PANEL_RGNS + gi * 3;
            SetRgn(g[0], px + GB_PLOT_W - TEXT_W(6) - 1, ly - 1, px + GB_PLOT_W + 1, ly + DET_GLYPH_H + 1);
            SetRgn(g[1], ax - 1, py - 1, ax + GB_LBL_W + 1, py + GB_PLOT_H + 1);
            SetRgn(g[2], px, py, px + GB_PLOT_W, py + GB_PLOT_H); GraphStatic(c, gx0, by, gi ? "FPS" : "FTIME");
        }
        g_rgnN[par] = DET_REGIONS;
    }
}

// Updates image par to the values in d. With the same layout the image already holds, only regions whose content
// changed are redrawn, outlined and restamped; otherwise everything is built
static void BuildDetailImg(const DetailData &d, DWORD corner, DWORD par, DWORD mode) {
    Canvas c = { g_detImgs + par * DET_IMG_DWORDS, g_detLit[par], g_detMask[par] };
    DetRgn *rg = g_rgn[par]; DWORD key = corner | (g_buildFmt << 2) | (1u << 20) | ((mode == 2) ? (1u << 21) : 0);
    bool  full  = (g_detKey[par] != key);
    if (full) {
        DIAG_ADD(D_IMG_FULL);
        for (DWORD i = 0; i < DET_MASK_DWORDS; i++) { c.lit[i] = 0; c.out[i] = 0; }
        DetLayout(c, corner, par, mode); g_detLayId[par] = ++g_detStampCtr;
    } else {
        DIAG_ADD(D_IMG_INCR);
    }
    for (DWORD r = 0; r < DET_ROWS; r++) {
        if (!full && d.v[r] == g_detShown[par].v[r]) continue;
        DetRgn &g = rg[DET_STATIC_RGNS + r]; RgnClear(c, g, g.x1 - g.x0 - 2, DET_GLYPH_H, full); DetNumber(c, g, r, d.v[r]);
        RgnDone(c, g, full);
    }
    if (mode == 2) {
        for (DWORD gi = 0; gi < 2; gi++) {
            bool rate = (gi != 0); DWORD cur = rate ? d.v[DV_FPS] : g_gCurFt; DWORD top = GraphTop(rate);
            DetRgn *g  = rg + DET_PANEL_RGNS + gi * 3;
            if (full || cur != g_shCur[par][gi]) {
                RgnClear(c, g[0], g[0].x1 - g[0].x0 - 2, DET_GLYPH_H, full); GraphReadout(c, g[0], rate, cur);
                RgnDone(c, g[0], full);
            }
            if (full || top != g_shTop[par][gi]) {
                RgnClear(c, g[1], GB_LBL_W, GB_PLOT_H, full); GraphAxis(c, g[1], top); RgnDone(c, g[1], full);
            }
            if (full || g_gSeq != g_shSeq[par] || top != g_shTop[par][gi]) {
                RgnClear(c, g[2], GB_IN_W, GB_IN_H, full); GraphPlot(c, g[2], rate, top); RgnDone(c, g[2], full);
            }
            g_shCur[par][gi] = cur; g_shTop[par][gi] = top;
        }
        g_shSeq[par] = g_gSeq;
    }
    if (full) {
        OutlineRect(c, 0, 0, DET_W, DET_H); DWORD *st = g_detStat[par];
        for (DWORD i = 0; i < DET_MASK_DWORDS; i++) st[i] = c.out[i];
        for (DWORD r = DET_STATIC_RGNS; r < g_rgnN[par]; r++) { // count each dynamic region and cut it out of the static bands
            DetRgn &g = rg[r]; DWORD used = 0;
            for (DWORD w = g.x0 >> 5; w <= (g.x1 - 1) >> 5; w++) {
                DWORD cm = ColMask(w, g.x0, g.x1);
                for (DWORD y = g.y0; y < g.y1; y++) {
                    used += PopCount32(c.out[y * DET_TILES_W + w] & cm); st[y * DET_TILES_W + w] &= ~cm;
                }
            }
            g.used = used; g.stamp = ++g_detStampCtr;
        }
        for (DWORD b = 0; b < DET_STATIC_RGNS; b++) {           // static bands: what is left after the cut
            DetRgn &s = rg[b]; DWORD used = 0;
            for (DWORD i = b * 32 * DET_TILES_W; i < (b + 1) * 32 * DET_TILES_W; i++) used += PopCount32(st[i]);
            s.used = used; s.cap = used;                        // static: no padding
            s.stamp = ++g_detStampCtr;
        }
        // Stream order = scan order: each band's static pixels, then the dynamic regions that start in that band,
        // so the top rows are written first (the GPU runs this stream while the display is already scanning out)
        DWORD off = 0;
        for (DWORD b = 0; b < DET_STATIC_RGNS; b++) {
            rg[b].off = off; off += rg[b].cap;
            for (DWORD r = DET_STATIC_RGNS; r < g_rgnN[par]; r++) {
                DWORD rb = rg[r].y0 >> 5; if (rb >= DET_STATIC_RGNS) rb = DET_STATIC_RGNS - 1;
                if (rb == b) { rg[r].off = off; off += rg[r].cap; }
            }
        }
    }
    g_detKey[par] = key; CopyData(g_detShown[par], d);
}

static void BuildDetailImage(DWORD corner, DWORD mode) {
    g_buildFmt = g_pixFmt; DWORD nxt = (g_boxVer + 1) & 1; STAGE(7); TMARK(tb);
    BuildDetailImg(g_dd, corner, nxt, mode); DWORD cnt = 0, used = 0;
    for (DWORD r = 0; r < g_rgnN[nxt]; r++) { cnt += g_rgn[nxt][r].cap; used += g_rgn[nxt][r].used; }
    WSTAT(D_T_W_BUILD, TNOW() - tb); CopyData(g_ddBuilt, g_dd); g_gDirty = false;
    PublishImage(cnt, used, 0, corner | (mode << 2));
}

static bool DetailChanged() {
    for (DWORD i = 0; i < DET_ROWS; i++)
        if (g_dd.v[i] != g_ddBuilt.v[i]) return true;
    return false;
}

// ======== Statistics and sensors ========
static DWORD Cap999(ULONGLONG v) { return v > 999 ? 999 : (DWORD)v; }

// The FPS readout of one tick: frames recorded since the previous tick over the sum of their frame times, so it is
// not quantised to whole frames per window. counted (frames / elapsed) is used when the ring covers less than half
// of the elapsed time (stall, loading). While shown, ring readings are remembered for MIN / MAX
static DWORD RingFps(DWORD counted, ULONGLONG elapsed) {
    DWORD idx = g_ftIdx;
    if (idx < g_fpsIdx) g_fpsIdx = 0;                   // the ring was restarted
    DWORD have = idx - g_fpsIdx;
    if (have > FT_RING) have = FT_RING;
    g_fpsIdx = idx; ULONGLONG sum = 0;
    for (DWORD i = 0; i < have; i++) sum += g_ftRing[(idx - have + i) & (FT_RING - 1)];
    if (!sum || sum * 2 < elapsed) return counted;
    DWORD fps = (DWORD)(((ULONGLONG)have * 49875000ULL + sum / 2) / sum);
    if (g_overlayOn) {
        DWORD j = g_fhN++ % FPS_HIST; g_fhFps[j] = Cap999(fps); g_fhEnd[j] = idx;
    }
    return fps;
}

// Over the last FT_RING frames. AVG and 1% Low use single frames (1% Low = rate of the average of the worst 1%, like
// RTSS). MIN / MAX are the lowest and highest FPS readout whose window has frames among those, including the current
// one, so MIN <= FPS <= MAX always holds
static void FrameStats(DetailData &d, DWORD fps) {
    const ULONGLONG TB = 49875000ULL; DWORD idx = g_ftIdx; DWORD n = (idx < FT_RING) ? idx : FT_RING;
    DWORD f = Cap999(fps); d.v[DV_FPS] = f;
    if (n < 2) { d.v[DV_MIN] = d.v[DV_AVG] = d.v[DV_MAX] = d.v[DV_LOW] = f; return; }
    DWORD k = n / 100;
    if (k < 1) k = 1;
    if (k > FT_WORST) k = FT_WORST;
    DWORD worst[FT_WORST];
    for (DWORD i = 0; i < k; i++) worst[i] = 0;
    // One pass: total time and the k longest frames
    DWORD first = idx - n; ULONGLONG sum = 0;
    for (DWORD i = 0; i < n; i++) {
        DWORD dt = g_ftRing[(first + i) & (FT_RING - 1)];
        if (!dt) dt = 1;
        sum += dt;
        if (dt > worst[k - 1]) {                      // keep the k largest, sorted descending
            DWORD j = k - 1;
            while (j > 0 && worst[j - 1] < dt) { worst[j] = worst[j - 1]; j--; }
            worst[j] = dt;
        }
    }
    ULONGLONG ws = 0;
    for (DWORD i = 0; i < k; i++) ws += worst[i];
    DWORD lo = f, hi = f; DWORD hn = (g_fhN < FPS_HIST) ? g_fhN : FPS_HIST;
    for (DWORD j = 0; j < hn; j++) {
        if (idx - g_fhEnd[j] >= n) continue;          // its frames have left the ring
        if (g_fhFps[j] < lo) lo = g_fhFps[j];
        if (g_fhFps[j] > hi) hi = g_fhFps[j];
    }
    DWORD avg = Cap999((TB * n + sum / 2) / sum);
    if (avg < lo) avg = lo;
    if (avg > hi) avg = hi;
    d.v[DV_MAX] = hi; d.v[DV_MIN] = lo; d.v[DV_AVG] = avg; d.v[DV_LOW] = Cap999((TB * k + ws / 2) / ws);
    g_diag[D_ST_MIN] = d.v[DV_MIN]; g_diag[D_ST_AVG] = d.v[DV_AVG];
    g_diag[D_ST_MAX] = d.v[DV_MAX]; g_diag[D_ST_LOW] = d.v[DV_LOW];
}

static DWORD ReadRamMB()        // used physical memory
{
    MEMORYSTATUS ms; ms.dwLength = sizeof(ms); GlobalMemoryStatus(&ms);
    return (DWORD)((ms.dwTotalPhys - ms.dwAvailPhys) >> 20);
}

// SMC command 0x07 (temperature sensors): little-endian 8.8 fixed point, CPU at bytes 1-2, GPU at 3-4 (EDRAM 5-6,
// board 7-8). UNVERIFIED: compare against a known-good tool before trusting it
static void ReadTemps(DWORD *cpu, DWORD *gpu) {
    __declspec(align(16)) BYTE msg[16]; __declspec(align(16)) BYTE resp[16];
    for (DWORD i = 0; i < 16; i++) { msg[i] = 0; resp[i] = 0; }
    msg[0] = 0x07; HalSendSMCMessage(msg, resp); *cpu = (DWORD)(resp[1] | (resp[2] << 8)) >> 8;
    *gpu = (DWORD)(resp[3] | (resp[4] << 8)) >> 8;
}

// CPU usage: each hardware thread has a kernel control block at 0x801B0000 + n * 0x1000 (idle thread at +0x300);
// +0x358 counts its idle milliseconds. Matched polling within a few points on console. The layout is validated
// once; on a mismatch the CPU rows stay 0
#define CPU_BLOCK_BASE   0x801B0000u
#define CPU_BLOCK_STEP   0x1000u
#define CPU_IDLE_MS_OFF  0x358u
#define CPU_HW           6
static DWORD     g_cpuIdlePrev[CPU_HW];
static ULONGLONG g_cpuTickPrev = 0;
static bool      g_cpuHavePrev = false;
static DWORD     g_cpuChecked  = 0;          // 0 = unchecked, 1 = layout ok, 2 = wrong
static DWORD     g_cpuBusy[CPU_HW];          // busy percent
static bool CpuBlockOk(DWORD n) {
    DWORD pcr = CPU_BLOCK_BASE + n * CPU_BLOCK_STEP; bool ok = false;
    __try {
        ok = (*(volatile BYTE *)(pcr + 0x10C) == n) && (*(volatile DWORD *)(pcr + 0x108) == pcr + 0x300);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ok = false;
    }
    return ok;
}

static bool CpuReadIdle(DWORD n, DWORD *out) {
    __try {
        *out = *(volatile DWORD *)(CPU_BLOCK_BASE + n * CPU_BLOCK_STEP + CPU_IDLE_MS_OFF);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Busy % = 100 - idle ms / elapsed ms since the previous sample
static void CpuUsageUpdate() {
    if (g_cpuChecked == 0) {
        g_cpuChecked = 1;
        for (DWORD n = 0; n < CPU_HW; n++)
            if (!CpuBlockOk(n)) g_cpuChecked = 2;
    }
    if (g_cpuChecked != 1) return;
    DWORD idle[CPU_HW]; ULONGLONG now = __mftb();
    for (DWORD n = 0; n < CPU_HW; n++) {
        if (!CpuReadIdle(n, &idle[n])) { g_cpuHavePrev = false; return; }
    }
    if (g_cpuHavePrev) {
        DWORD ms = (DWORD)((now - g_cpuTickPrev) / 49875);
        if (ms < 250) return;                                  // too short: keep the old baseline
        if (ms <= 3000) {                                      // a longer gap (mode was off) only sets a new baseline
            for (DWORD n = 0; n < CPU_HW; n++) {
                DWORD dIdle = idle[n] - g_cpuIdlePrev[n];      // unsigned: handles wrap
                DWORD idlePm = (DWORD)(((ULONGLONG)dIdle * 1000) / ms);
                if (idlePm > 1000) idlePm = 1000;
                g_cpuBusy[n] = (1000 - idlePm + 5) / 10;
            }
        }
    }
    for (DWORD n = 0; n < CPU_HW; n++) g_cpuIdlePrev[n] = idle[n];
    g_cpuTickPrev = now; g_cpuHavePrev = true;
}

// GPU usage = RBBM counter 0 over CP clock, from the words the hook had the GPU write
static void ProbeGpu() {
    if (!g_probeVa) return;
    DWORD clk = g_probeVa[0], b0 = g_probeVa[1], b1 = g_probeVa[2]; ULONGLONG tb = __mftb();
    g_diag[D_PRB_CLK] = clk; g_diag[D_PRB_B0] = b0; g_diag[D_PRB_B1] = b1; ULONGLONG dTb = tb - g_prbTb;
    if (g_prbHave && dTb < 74812500ULL) {                    // under 1.5 s, else counters may have wrapped unseen
        DWORD dClk = clk - g_prbClk, dB0 = b0 - g_prbB0, dB1 = b1 - g_prbB1;
        g_diag[D_PRB_DCLK] = dClk; g_diag[D_PRB_DB0] = dB0; g_diag[D_PRB_DB1] = dB1; g_diag[D_PRB_DTB] = (DWORD)dTb;
        if (dClk >= 800000000) {
            // restarted, wrapped or sentinel: new baseline only
        } else if (dClk) {
            g_prbPct0 = (DWORD)((ULONGLONG)dB0 * 100 / dClk); g_prbPct1 = (DWORD)((ULONGLONG)dB1 * 100 / dClk);
            g_diag[D_PRB_PCT0] = g_prbPct0; g_diag[D_PRB_PCT1] = g_prbPct1;
        }
        else if (!g_probeWant) g_probeArm = 1;               // a sample was taken but the clock is stuck: set up again
    }
    g_prbClk = clk; g_prbB0 = b0; g_prbB1 = b1; g_prbTb = tb; g_prbHave = 1;
}

#if DIAG_ENABLED
// CP clock (~500 MHz) spent on the overlay stream: reading it and issuing its writes, not memory completion
static DWORD g_ibLastAfter = 0;
static void ReadGpuIb() {
    if (!g_probeVa) return;
    DWORD after = g_probeVa[0], before = g_probeVa[3];
    // If the hook took the last request but nothing new arrived, the CP clock is stopped (a game owning
    // CP_PERFMON_CNTL can do that): request the setup again. This also covers FPS mode, which skips ProbeGpu
    bool taken = (g_probeWant == 0);
    if (after == 0xDEADBEEF || before == 0xDEADBEEF) { if (taken) g_probeArm = 1; return; }   // never written
    if (after == g_ibLastAfter)                      { if (taken) g_probeArm = 1; return; }   // no new sample
    g_ibLastAfter = after; DWORD dt = after - before;
    if (dt > 50000000) return;                                 // implausible (over 100 ms): wrapped
    TSTAT(D_G_IB, dt);
}
#endif

// Collects everything the panel shows (detailed modes only)
static void GatherDetail(DWORD fps) {
    TMARK(tg); TMARK(ts); DWORD t = g_dTick++; STAGE(4); FrameStats(g_dd, fps);
    if (g_mode == 2) GraphUpdate();
    WLAP(D_T_W_FSTATS, ts); ProbeGpu(); g_dd.v[DV_GPU] = (g_prbPct0 > 100) ? 100 : g_prbPct0; WLAP(D_T_W_PROBE, ts);
    CpuUsageUpdate(); DWORD busy = 0;
    for (DWORD n = 0; n < CPU_HW; n++) { g_dd.v[DV_CPU0 + n] = g_cpuBusy[n]; busy += g_cpuBusy[n]; }
    g_dd.v[DV_CPU] = (busy + 3) / CPU_HW; WLAP(D_T_W_CPU, ts); STAGE(5); g_dd.v[DV_RAM] = ReadRamMB();
    WLAP(D_T_W_RAM, ts);
    if ((t % DETAIL_TEMP_EVERY) == 0) {          // the SMC is shared with the system: read rarely
        STAGE(6); ReadTemps(&g_dd.v[DV_CPUT], &g_dd.v[DV_GPUT]); COUNT(D_SMC_N); WLAP(D_T_W_SMC, ts);
    }
    WSTAT_OUTER(D_T_W_GATHER, TNOW() - tg);
}

// Rebuilds the image for the current mode
static void BuildCurrent(DWORD fps, DWORD corner) {
    if (g_mode) { GatherDetail(fps); BuildDetailImage(corner, g_mode); }
    else        BuildBoxImage(fps, corner);
}

// ======== Worker ========
// Patches the first 4 instructions of VdSwap, only if they match the dumped kernel
static void InstallHook() {
    volatile DWORD *kp = (volatile DWORD *)0x800C8B90;
    if (kp[0] != 0x7D8802A6 || kp[1] != 0x48044021 ||
        kp[2] != 0x9421FF40 || kp[3] != 0x3D608017) {
        g_diag[D_PATCH] = 0xBAD;
        return;
    }
    // Cache flush helper at g_tramp[16..19]: icbi 0,r3 ; sync ; isync ; blr
    g_tramp[16] = 0x7C001FAC; g_tramp[17] = 0x7C0004AC; g_tramp[18] = 0x4C00012C; g_tramp[19] = 0x4E800020;
    // Trampoline: replays the 4 overwritten instructions, then jumps to 0x800C8BA0
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
    typedef VOID (*FlushFn)(PVOID); FlushFn flush = (FlushFn)(PVOID)&g_tramp[16]; flush((PVOID)&g_tramp[0]);
    DWORD hk = (DWORD)(PVOID)&HookVdSwap; g_diag[D_HOOKADDR] = hk; g_diag[D_TRAMP] = (DWORD)&g_tramp[0];
    g_diag[D_PATCH]    = 2;
    kp[1] = 0x616B0000 | (hk & 0xFFFF);    // ori r11,r11,lo
    kp[2] = 0x7D6903A6;                    // mtctr r11
    kp[3] = 0x4E800420;                    // bctr
    kp[0] = 0x3D600000 | (hk >> 16);       // lis r11,hi  (written last)
    __dcbst(0, (void *)kp);
    __sync();
    flush((PVOID)kp); g_diag[D_PATCH] = 3;
}
#if CONTROLLER_INPUT
static bool  g_comboWasHeld  = false;
static bool  g_cornerWasHeld = false;
static bool  g_modeWasHeld   = false;
static DWORD g_padMask  = 0;    // pads that answered last time
static DWORD g_padPolls = 0;

// Acts once per press of the toggle, corner or mode combo
static void PollToggle() {
    bool rescan = (g_padPolls++ % (PAD_RESCAN_MS / INPUT_POLL_MS)) == 0; DWORD pads = 0;
    bool held = false, cornerHeld = false, modeHeld = false;
    for (DWORD u = 0; u < 4; u++) {
        if (!rescan && !(g_padMask & (1u << u))) continue;
        XINPUT_STATE st;
        if (XInputGetState(u, &st) == ERROR_SUCCESS) {
            pads |= (1u << u); DWORD b = st.Gamepad.wButtons;
            if ((b & TOGGLE_BUTTONS) == TOGGLE_BUTTONS)       held = true;
            else if ((b & CORNER_BUTTONS) == CORNER_BUTTONS)  cornerHeld = true;
            else if ((b & MODE_BUTTONS) == MODE_BUTTONS)      modeHeld = true;
        }
    }
    g_padMask = pads; g_diag[D_PADS] = pads;
    if (held && !g_comboWasHeld) {
        // Turning on: restart frame stats and prebuild while still hidden, so the first frame is ready
        if (!g_overlayOn) { g_ftIdx = 0; g_fpsIdx = 0; g_fhN = 0; GraphReset(0); PrebuildStreams(g_boxVer, true); }
        g_overlayOn = g_overlayOn ? 0 : 1; g_diag[D_ENABLED] = g_overlayOn; COUNT(D_TOGGLES);
    }
    g_comboWasHeld = held;
    if (cornerHeld && !g_cornerWasHeld) {
        g_corner = (g_corner + 1) & 3; g_diag[D_CORNER] = g_corner;
    }
    g_cornerWasHeld = cornerHeld;
    if (modeHeld && !g_modeWasHeld) {
        DWORD next = (g_mode + 1) % DISPLAY_MODES;
        if (next >= 1 && !g_dCmdVa) next = 0;            // no detailed memory
        if (next >= 1 && g_mode == 0) {                  // leaving FPS only: fresh stats and graphs
            g_ftIdx = 0; g_fpsIdx = 0; g_fhN = 0; GraphReset(0);
        } else if (next == 2) {                          // panel -> graphs: start from the recorded frames
            DWORD have = (g_ftIdx < FT_RING) ? g_ftIdx : FT_RING; GraphReset(g_ftIdx - have);
        }
        g_mode = next; g_diag[D_MODE] = next; COUNT(D_MODE_SW);
    }
    g_modeWasHeld = modeHeld;
}

#else
static void PollToggle() {}
#endif

static PVOID PhysAlloc(DWORD bytes, DWORD protect) { return XPhysicalAlloc(bytes, (ULONG_PTR)(~(ULONG_PTR)0), 0, protect); }
static DWORD WINAPI Worker(LPVOID) {
    DelayMs(START_DELAY_MS);        // nothing at all happens before this returns
    g_diag[D_STATE] = 1; g_diag[D_ENABLED] = g_overlayOn; g_diag[D_CORNER] = g_corner;
    g_diag[D_DET_MEM] = 2;          // until everything for the detailed mode is allocated
    // Cached CPU work memory (read a lot, so not write combined; not static, to keep the load footprint small).
    // Without room for both parts only the FPS mode is offered
    PVOID wk = PhysAlloc(FPS_WORK_BYTES + DET_WORK_BYTES, PAGE_READWRITE); DWORD detBytes = wk ? DETAIL_MEM_BYTES : 0;
    if (!wk) wk = PhysAlloc(FPS_WORK_BYTES, PAGE_READWRITE);
    if (!wk) return 0;                                  // the plugin stays idle
    DIAG_SET(D_MEM_WORK, FPS_WORK_BYTES + (detBytes ? DET_WORK_BYTES : 0)); BYTE *wb = (BYTE *)wk;
    g_boxImgs[0] = (DWORD *)wb; g_boxImgs[1] = g_boxImgs[0] + BOX_DWORDS; wb += WK_BOXIMGS_BYTES;
    g_wMask[0] = (DWORD *)wb; g_wMask[1] = g_wMask[0] + 64; wb += WK_WMASK_BYTES;
    g_slotPix = (WORD *)wb; wb += WK_SLOTPIX_BYTES;
    for (DWORD i = 0; i < 2 * BOX_DWORDS; i++) g_boxImgs[0][i] = 0;
    for (DWORD i = 0; i < 128; i++) g_wMask[0][i] = 0;
    // One write-combined GPU allocation: FPS streams, then detailed streams and probe words.
    // g_cmdPhys is set before the hook is installed
    PVOID cb = PhysAlloc(SLOT_COUNT * SLOT_BYTES + detBytes, PAGE_READWRITE | PAGE_WRITECOMBINE);
    if (cb) DIAG_SET(D_MEM_CMD, SLOT_COUNT * SLOT_BYTES + detBytes);
    DWORD va = (DWORD)cb;
    if (cb && PhysOf(va)) {
        g_cmdVa = va; g_cmdPhys = PhysOf(va);
        for (DWORD s = 0; s < SLOT_COUNT; s++) g_hot.slotPhys[s] = g_cmdPhys + s * SLOT_BYTES;
        if (detBytes) {
            for (DWORD s = 0; s < DET_SLOT_COUNT; s++)
                g_hot.slotPhys[SLOT_COUNT + s] = g_cmdPhys + SLOT_COUNT * SLOT_BYTES + s * DSLOT_BYTES;
            __sync();
            g_detImgs = (DWORD *)wb; g_dPix = (WORD *)(wb + DET_IMGS_BYTES);
            g_detMask[0] = (DWORD *)(wb + DET_IMGS_BYTES + DET_PIXLIST_BYTES);
            g_detMask[1] = g_detMask[0] + DET_MASK_DWORDS; g_detLit[0] = g_detMask[1] + DET_MASK_DWORDS;
            g_detLit[1] = g_detLit[0] + DET_MASK_DWORDS; g_detStat[0] = g_detLit[1] + DET_MASK_DWORDS;
            g_detStat[1] = g_detStat[0] + DET_MASK_DWORDS;
            g_dLoc       = (PktLoc *)(wb + DET_IMGS_BYTES + DET_PIXLIST_BYTES + DET_MASKS_BYTES);
            g_dCmdVa     = va + SLOT_COUNT * SLOT_BYTES;
            g_probeVa    = (volatile DWORD *)(g_dCmdVa + DET_SLOT_COUNT * DSLOT_BYTES);
            for (DWORD i = 0; i < 8; i++) g_probeVa[i] = 0xDEADBEEF;   // sentinel: never written by the GPU
            __sync();
            g_probePhys = g_cmdPhys + SLOT_COUNT * SLOT_BYTES + DET_SLOT_COUNT * DSLOT_BYTES; g_diag[D_DET_MEM] = 1;
        } else {
            __sync();
        }
        g_diag[D_CMDVA] = va; g_diag[D_CMDPHYS] = g_cmdPhys; g_diag[D_STATE] = 2;
    }
    InitTiledTab();                 // before the first image build
    InitGlyphTab(); InstallHook();
    // FPS from the timebase (49.875 MHz) over 0.5 s windows, polling the pads in between
    ULONGLONG t0 = __mftb(); DWORD f0 = g_frames;
    DWORD     imgFps    = 0xFFFFFFFF;       // number the current image stands for, 0xFFFFFFFF = none yet
    DWORD imgCorner = 0xFFFFFFFF; DWORD imgMode = 0; DWORD imgPixFmt = 0;
    DWORD     lastShown = 0xFFFFFFFF;       // newest measurement, built or not
    for (;;) {
        ULONGLONG t = __mftb(); ULONGLONG el = t - t0;
        if (el < 24937500ULL) {
            DWORD left = (DWORD)(((24937500ULL - el) * 1000ULL) / 49875000ULL) + 1;
            DelayMs(left < INPUT_POLL_MS ? left : INPUT_POLL_MS);   // also the latency for a stream the hook asked for
            TMARK(tpl); PollToggle(); WSTAT(D_T_W_POLL, TNOW() - tpl);
            if (g_missReq) {                    // the hook found no ready stream: fill the invalid slots now
                g_missReq = 0;                  // (valid ones are skipped, so nothing the hook uses is touched)
                PrebuildStreams(g_boxVer, true);
            }
            // Rebuild at once on a corner, mode or pixel layout change, or when shown again in FPS mode with an old number
            DWORD cn = g_corner, md = g_mode; DWORD pfNow = g_pixFmt;
            bool stale = g_overlayOn && md == 0 && imgFps != 0xFFFFFFFF && lastShown != imgFps;
            if (imgFps != 0xFFFFFFFF && (cn != imgCorner || md != imgMode || pfNow != imgPixFmt || stale)) {
                BuildCurrent(lastShown, cn);
                imgPixFmt = pfNow;              // if the hook changed it during the build, the next poll rebuilds
                imgCorner = cn; imgMode = md; imgFps = lastShown;
            }
            continue;
        }
        TMARK(twk); DWORD f = g_frames;
        DWORD fps = (DWORD)(((ULONGLONG)(f - f0) * 49875000ULL + (t - t0) / 2) / (t - t0));
        fps = RingFps(fps, t - t0); g_diag[D_FPS] = fps; COUNT(D_WBEAT); f0 = f; t0 = t;
        // Active buffers = those drawn in the last two ticks (all while hidden, since nothing is recorded)
        if (g_overlayOn) {
            DWORD sn = g_hot.seen; g_hot.seen = 0; g_activeMask = sn | g_seenPrev; g_seenPrev = sn;
        } else {
            g_activeMask = 0xFFFFFFFF; g_seenPrev = 0xFFFFFFFF;
        }
        DWORD shown = (fps > 999) ? 999 : fps; lastShown = shown; DWORD cn = g_corner, md = g_mode;
        DWORD pf = g_pixFmt;
        if (md) {
            // Sensors and statistics are read only while visible (or for a corner / mode change)
            if (g_overlayOn || md != imgMode || cn != imgCorner) {
                GatherDetail(shown);
                if (md != imgMode || cn != imgCorner || pf != imgPixFmt || DetailChanged() || (md == 2 && g_gDirty)) {
                    BuildDetailImage(cn, md); imgPixFmt = pf;
                }
            }
            imgFps = shown;
        } else if (g_overlayOn || cn != imgCorner || md != imgMode || pf != imgPixFmt) {
            // Hidden and unchanged: nothing is built and imgFps keeps the number the image really holds
            if (shown != imgFps || cn != imgCorner || md != imgMode || pf != imgPixFmt) {
                BuildBoxImage(shown, cn); imgPixFmt = pf;
            }
            imgFps = shown;
        }
        imgCorner = cn; imgMode = md;
#if DIAG_ENABLED
        ReadGpuIb();
#endif
        g_probeWant = 1;                        // ask the hook for the next probe sample
        WSTAT_OUTER(D_T_W_TICK, TNOW() - twk); STAGE(1);
    }
}

extern "C" BOOL WINAPI DllMain(HANDLE, DWORD reason, LPVOID) {
    if (reason != DLL_PROCESS_ATTACH)
        return TRUE;
    HANDLE h = NULL; DWORD tid = 0;
    if (ExCreateThread(&h, 0, &tid, (PVOID)XapiThreadStartup, (LPTHREAD_START_ROUTINE)Worker, NULL, 2 | CREATE_SUSPENDED) >= 0 && h != NULL) {
        XSetThreadProcessor(h, WORKER_THREAD); SetThreadPriority(h, WORKER_PRIORITY); ResumeThread(h); CloseHandle(h);
    }
    return TRUE;
}
