#!/usr/bin/env bash
# Builds XenonFPS.xex with the Xbox 360 XDK's own compiler. Run from Git Bash
set -e
cd "$(dirname "$0")"

[ -n "$XEDK" ] || { echo "XEDK is not set" >&2; exit 1; }
XEDK_U="$(cygpath -u "$XEDK")"
BIN="$XEDK_U/bin/win32"
[ -f "$BIN/cl.exe" ] || { echo "cl.exe not found in $BIN" >&2; exit 1; }

INC="$(cygpath -w "$XEDK_U/include/xbox")"
INCLUDE_PATH="$INC"

if [ -n "$CRT_INC" ] && [ -d "$CRT_INC" ]; then
    INCLUDE_PATH="$INC;$(cygpath -w "$CRT_INC")"
fi

# Force-include only if your XDK ships this header
FI=()
if [ -f "$XEDK_U/include/xbox/xbox_intellisense_platform.h" ]; then
    FI=(-FI"$INC\\xbox_intellisense_platform.h")
fi

mkdir -p build

INCLUDE="$INCLUDE_PATH" "$BIN/cl.exe" -nologo -c -W4 -Ox -MT -GR- -EHsc -TP \
    -D _XBOX -D NDEBUG "${FI[@]}" -Fobuild/XenonFPS.obj XenonFPS.cpp

LIB="$(cygpath -w "$XEDK_U/lib/xbox")" "$BIN/link.exe" -nologo -RELEASE -OPT:REF -DLL -ENTRY:_DllMainCRTStartup \
    -XEX:NO -ALIGN:128,4096 -MAP:build/XenonFPS.map -OUT:build/XenonFPS.exe build/XenonFPS.obj xboxkrnl.lib xapilib.lib

"$BIN/imagexex.exe" -nologo -config:XenonFPS.xex.xml -out:build/XenonFPS.xex build/XenonFPS.exe
echo build/XenonFPS.xex
