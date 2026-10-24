#!/bin/bash
# Build only the DeepSeek-V4.1 engine (strata-dsv41) for the Linux AMD RX 7000 family (gfx1100;gfx1101;gfx1102).
#
# Upstream ships strata-dsv41 packages for cuda13 and hip-rx9070 only: neither the v0.2.4 release nor the signed
# update feed has a build for the RX 7900 family, so a 7900 can never install a DeepSeek engine on its own. This
# script produces that one binary from the same Strata source tag, with the flags of the DSV41 block in upstream's
# packaging/build_hip_linux.sh, so it can be put BESIDE `strata` inside the released rx7900 engine package folder.
# That is exactly the shape upstream's rx9070 package has (its build sets DSV41=1): setup.py dsv41_exe() looks for
# strata-dsv41 beside the engine, so no environment variable is needed at run time.
#
# No GPU is needed: every code object is compiled ahead of time for ARCHS.
#
# Environment (SRC, LLAMA and HP are required):
#   SRC=<strata checkout>            the Strata source tree (tag v0.2.4 or later)
#   LLAMA=<llama.cpp checkout>       at the commit in <SRC>/third_party/ggml/VERSION.txt (ggml is built static from it)
#   HP=<ROCm devel root>             TheRock: <venv>/**/site-packages/_rocm_sdk_devel
#   LIBW=<ROCm libraries root>       TheRock: <venv>/**/site-packages/_rocm_sdk_libraries_gfx110X_dgpu
#   OUT=<work folder>                (default: build-dsv41; the result is $OUT/strata-dsv41)
#   ARCHS=gfx1100;gfx1101;gfx1102    the code objects
#   JOBS=$(nproc)
#
# The flags below are upstream's packaging/build_hip_linux.sh DSV41 block; keep them in sync when the source tag moves.
set -eu

: "${SRC:?SRC=<strata checkout>}"
: "${LLAMA:?LLAMA=<llama.cpp checkout at third_party/ggml/VERSION.txt>}"
: "${HP:?HP=<ROCm devel root (TheRock _rocm_sdk_devel)>}"
LIBW=${LIBW:-$HP}
OUT=${OUT:-build-dsv41}
ARCHS=${ARCHS:-gfx1100;gfx1101;gfx1102}
JOBS=${JOBS:-$(nproc)}

export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP
# TheRock 7.14.x dists keep libnuma, libz, libdrm ... in lib/rocm_sysdeps; older wheels have no such folder
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
HIPDEFS=(-DCMAKE_HIP_COMPILER="$HP/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$HP"
         "-DCMAKE_PREFIX_PATH=$HP;$LIBW$SYSDEPS"
         "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$HP/lib/llvm/amdgcn/bitcode")

mkdir -p "$OUT"
cmake -G Ninja -S "$SRC" -B "$OUT/b" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON \
  "-DCMAKE_HIP_ARCHITECTURES=$ARCHS" "${HIPDEFS[@]}" \
  -DSTRATA_GGML_DIR="$LLAMA" -DSTRATA_PORTABLE=ON \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags"
cmake --build "$OUT/b" --target strata-dsv41 -j "$JOBS"

[ -x "$OUT/b/strata-dsv41" ] || { echo "build-dsv41: no strata-dsv41 in $OUT/b" >&2; exit 1; }
install -m 0755 "$OUT/b/strata-dsv41" "$OUT/strata-dsv41"
echo "built $OUT/strata-dsv41 (RUNPATH \$ORIGIN; put it beside the engine binary)"
