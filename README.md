# XenonFPS

DashLaunch plugin for Xbox 360 that draws an FPS counter on top of any game as well as the dashboard. It hooks the kernel's `VdSwap` call and has the GPU draw a small box with the frame rate into the front buffer, so there is no extra work on the game's own render path.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4e96c76e-102e-4746-b645-e3a5230f958e" />
*An example of the FPS counter visible on Forza Horizon*

## Setup

1. Download `XenonFPS.xex` from the [latest release](../../releases/latest) and copy it to the root of your USB stick or console drive.
2. Add it to `launch.ini` (example):

```
[Plugins]
plugin1 = Usb:\Xbdm.xex
plugin2 = Usb:\xbGuard.xex
plugin3 = Usb:\XenonFPS.xex
```

3. Run BadUpdate or BadAvatar as usual. The counter appears about 20 seconds after the plugin loads, located at the top-left corner of the screen, which gives the video system time to come up before the hook is installed.

> [!IMPORTANT]
> The hook patches the first four instructions of `VdSwap` at `0x800C8B90` and refuses to install unless they match the kernel it was written against. On a different kernel it does nothing and sets `D_PATCH` to `0xBAD` (see Diagnostics). Tested on: 2.0.17559.0 (BadUpdate)

> [!WARNING]
> For games with an unlocked framerate (no V-Sync), be mindful that the overlay may tear/flicker when the screen tears.

## Settings

All settings are `#define`s at the top of `XenonFPS.cpp`. Change them and rebuild.

| Setting | Default | What it does |
|---|---|---|
| `BOX_CORNER` | `CORNER_TOP_LEFT` | Which corner the box sits in: `CORNER_TOP_LEFT`, `CORNER_TOP_RIGHT`, `CORNER_BOTTOM_LEFT`, `CORNER_BOTTOM_RIGHT` |
| `BOX_MARGIN_TILES` | `0` | Moves the box inward from the edges, 32 px per step (0..10). Useful if your TV crops the edges |
| `BOX_PAD_X` / `BOX_PAD_Y` | `8` / `4` | Trims the black background on each side of the 64x32 box (0..10 and 0..6). Set both to 0 for the full box. Only the pixels inside the trimmed rectangle are written each frame, so a bigger trim means less work for the GPU |
| `START_DELAY_MS` | `20000` | How long the plugin waits before patching `VdSwap` |

The box is a solid black background with white digits. The GPU can only overwrite pixels, not blend with what the game drew, so there is no real transparency.

## How it works

1. A worker thread waits for `START_DELAY_MS`, then patches the first four instructions of `VdSwap` to jump to `HookVdSwap`.
2. `HookVdSwap` counts frames, calls the real `VdSwap` through a trampoline, then edits the GPU ring buffer. It replaces the ring's NOP area with an `INDIRECT_BUFFER` packet that points at a command stream of `MEM_WRITE` packets. The GPU runs those writes in order with the game's own commands, and the box lands in the front buffer.
3. Command streams are cached per front buffer and image version, so on a normal frame the hook only rewrites four ring dwords. A stream is rebuilt only when the FPS number changes or a front buffer is seen for the first time.
4. The worker measures FPS twice a second from the CPU timebase and builds the new box image into the spare one of two images, then flips the version number so the hook never reads a half-built image.

The patch lives in memory only. Nothing is written to storage, so a reboot starts clean.

> [!WARNING]
> This edits the GPU command ring. A bad packet can crash the GPU or hang the console. The one-dword `MEM_WRITE` packets used here are known to work; multi-dword `MEM_WRITE` packets did not on the console this was developed on, so don't try to merge writes into larger packets.

## Build

You need the Xbox 360 XDK. Point `XEDK` at the SDK folder and run `./build.sh` from Git Bash. The result is `build/XenonFPS.xex`.

```
export XEDK="C:\Program Files (x86)\Microsoft Xbox 360 SDK"
./build.sh
```

If your XDK ships extra plain C headers in a separate folder, set `CRT_INC` to it before building.

## Diagnostics

The plugin keeps a diagnostics block in memory (`g_diag`). Search the console's memory for the magic `0xD1A60100` with `getmem` and read the values by index:

| Index | Name | Meaning |
|---|---|---|
| 0 | `D_MAGIC` | `0xD1A60100` |
| 1 | `D_STATE` | 1 = worker started, 2 = command memory allocated |
| 2 | `D_PATCH` | `0xBAD` = `VdSwap` did not match (nothing patched), 2 = trampoline ready, 3 = patch installed |
| 5 | `D_FRAMES` | `VdSwap` calls seen |
| 6 | `D_FPS` | Last measured FPS |
| 7 | `D_QUEUED` | Frames that got the box |
| 8 | `D_SKIPPED` | Frames skipped (start-up, menus, unknown layouts) |

The remaining entries (hook and trampoline addresses, last front buffer, image builds, stream builds, image version, command memory addresses) are listed in the `D_*` enum in the source.
