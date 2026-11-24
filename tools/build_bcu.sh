#!/usr/bin/env bash
# Build NXP BCU (board control and power monitor tool) natively for Apple Silicon into tools/bcu/.
# NXP's README describes an x86 Homebrew under Rosetta; this builds arm64 instead, with libftdi 1.5 from
# source and libusb + libyaml from the arm64 Homebrew in /opt/homebrew. cmake comes from pip in a temp venv.
#   tools/build_bcu.sh [work_dir]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${1:-$(mktemp -d)}
OUT=$ROOT/tools/bcu
LIBFTDI_SHA=7c7091e9c86196148bd41177b4590dccb1510bfe6cea5bf7407ff194482eb049   # same as Homebrew's libftdi 1.5
export PATH=/opt/homebrew/bin:$PATH
brew list --versions libusb libyaml pkgconf >/dev/null || { echo "need: brew install libusb libyaml pkgconf"; exit 1; }
mkdir -p "$W" "$OUT"
cd "$W"
python3 -m venv venv && ./venv/bin/pip -q install cmake
curl -sSfL -o libftdi1-1.5.tar.bz2 https://www.intra2net.com/en/developer/libftdi/download/libftdi1-1.5.tar.bz2
echo "$LIBFTDI_SHA  libftdi1-1.5.tar.bz2" | shasum -a 256 -c -
tar xjf libftdi1-1.5.tar.bz2
cmake_() { "$W/venv/bin/cmake" "$@"; }
(mkdir -p libftdi1-1.5/build && cd libftdi1-1.5/build && cmake_ .. -DCMAKE_INSTALL_PREFIX="$W/prefix" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
   -DFTDI_EEPROM=OFF -DEXAMPLES=OFF -DPYTHON_BINDINGS=OFF -DDOCUMENTATION=OFF -DFTDIPP=OFF -DLINK_PYTHON_LIBRARY=OFF >/dev/null && make -j8 install >/dev/null)
git clone -q --depth 1 https://github.com/nxp-imx/bcu.git bcu
(mkdir -p bcu/build && cd bcu/build && PKG_CONFIG_PATH="$W/prefix/lib/pkgconfig:/opt/homebrew/lib/pkgconfig" cmake_ .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null && make -j8 >/dev/null)
cp bcu/build/bcu_mac "$OUT/"
cp "$W/prefix/lib/libftdi1.2.dylib" "$OUT/"
OLD=$(otool -L "$OUT/bcu_mac" | awk '/libftdi1/{print $1}')
install_name_tool -change "$OLD" @executable_path/libftdi1.2.dylib "$OUT/bcu_mac"
install_name_tool -id @executable_path/libftdi1.2.dylib "$OUT/libftdi1.2.dylib"
codesign -f -s - "$OUT/libftdi1.2.dylib" "$OUT/bcu_mac"
{ (cd bcu && git log -1 --format='BCU source commit %H (%cs)'); echo "libftdi 1.5 (sha256 $LIBFTDI_SHA), built arm64"; } > "$OUT/VERSION.txt"
cat "$OUT/VERSION.txt"
"$OUT/bcu_mac" lsftdi | tail -2
