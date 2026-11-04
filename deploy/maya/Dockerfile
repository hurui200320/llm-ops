# Project Maya (GLM-5.3-Flash + DeepSeek-V4.1-Flash) on RX 7900 XT / XTX (gfx1100), HIP / ROCm.
#
# Upstream (mw00/project-maya) ships no Docker image: the installer (maya.py) compiles the engines on the user's
# machine and downloads the models at first start.  This builds the engines the way that installer does
# (compile_engine_hip / compile_vision: the same cmake flags, the same llama.cpp commit) and the GPU vision
# encoder, inside AMD's ROCm dev container - llama.cpp's .devops/rocm.Dockerfile pattern.  The build stage and
# the runtime stage share one ROCm image, so the engines link the ROCm they run with, and the image keeps the
# full SDK on purpose: maya.py's check_hip_pc runs at every start and requires hipcc >= 7, ROCm's clang and
# libhipblas inside the environment (a libs-only runtime would refuse to start).
#
# The ROCm line is 7.2.1: the one Maya's own RX 7900 validation used (docs/AMD_MAYA.md), and the one whose
# hipBLASLt 1.2.2 (= 100202) matches tools/hip/gfx1100-glm-hipblaslt-100202.txt, the GLM tuning table maya.py
# passes to the engine (the engine accepts only its own arch + hipBLASLt version, else plain hipBLAS).
#
# The build writes build-hip/MAYA-BUILD.json and build-vision/MAYA-BUILD.json with Maya's own source hash
# (deploy/maya/stamps.py), so the installer reuses these engines on first start instead of compiling again.
# The image is built by .github/workflows/maya-image.yml; run recipes and pins: deploy/maya/README.md.

ARG UBUNTU_VERSION=24.04
ARG ROCM_VERSION=7.2.1
ARG BASE_IMAGE=docker.io/rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete
ARG LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d

# ------------------------------------------------------------------------------------------------ engine build
FROM ${BASE_IMAGE} AS build

ARG LLAMA_COMMIT
ARG HIP_ARCH=gfx1100
ARG ENGINES=maya,maya-ds4
ARG MAYA_COMMIT=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build curl ca-certificates \
      python3 python3-venv python3-pip libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/maya
COPY maya/ /opt/maya/
COPY deploy/maya/stamps.py /tmp/stamps.py
RUN rm -rf /opt/maya/.git
# A source archive has no .git: cmake/build_id.cmake reads the commit from BUILD_ID (maya-ds4 --version names it).
RUN printf '%s\n' "$MAYA_COMMIT" > /opt/maya/BUILD_ID

# llama.cpp at the commit setup.py pins, laid out as get_llama_cpp() lays it out (the zip's top folder becomes
# third_party/llama.cpp); no git needed.  The pin is gated against setup.py's LLAMA_CPP_COMMIT by the workflow.
RUN set -eux; \
    cd /opt/maya/third_party; \
    curl -fsSL --retry 5 --retry-delay 2 -o llama.cpp.zip \
      "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.zip"; \
    python3 -c "import pathlib, shutil, zipfile; \
zipfile.ZipFile('llama.cpp.zip').extractall('_unpack'); \
shutil.move(str(next(pathlib.Path('_unpack').iterdir())), 'llama.cpp')"; \
    rm -rf _unpack llama.cpp.zip; \
    test -f llama.cpp/ggml/CMakeLists.txt; \
    test -d llama.cpp/gguf-py

# The GLM engine and DeepSeek's: compile_engine_hip's configure line (maya.py, Linux branch) with gfx1100 only.
# Jobs as maya.py sizes them for HIP builds (half the cores, at most 4: each clang takes 2-4 GB).
# STRATA_PORTABLE=ON: the image must run on every host maya accepts (maya.py's check_pc floor is AVX2), so ggml
# must not inherit the build host's ISA: maya's CMakeLists forces GGML_NATIVE per platform and STRATA_PORTABLE
# switches it to explicit AVX/AVX2/FMA/F16C/BMI2 (the same switch upstream's Windows HIP package build passes).
# Without it, an image built on an AVX-512-capable builder SIGILLs in ggml_cpu_init on AVX2-only hosts at the
# first model load (notes/maya.md 20261104). The assertion below pins this at build time.
RUN set -eux; \
    cd /opt/maya; \
    clang="$(hipconfig -l)/clang++"; \
    [ -x "$clang" ] || clang=/opt/rocm/lib/llvm/bin/clang++; \
    [ -x "$clang" ]; \
    jobs=$(( $(nproc) / 2 )); [ "$jobs" -ge 1 ] || jobs=1; [ "$jobs" -le 4 ] || jobs=4; \
    cmake -S /opt/maya -B /opt/maya/build-hip -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
        -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_PORTABLE=ON \
        -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
        -DCMAKE_HIP_COMPILER="$clang" -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DSTRATA_GGML_DIR=/opt/maya/third_party/llama.cpp; \
    for target in $(printf '%s' "$ENGINES" | tr ',' ' '); do \
        cmake --build /opt/maya/build-hip --target "$target" -j "$jobs"; \
    done

# The GPU vision encoder (compile_vision, Linux HIP branch): llama.cpp's mtmd at the pinned commit with
# GLM-5.3-Flash's vision tower added by the idempotent patch, on ggml's HIP backend for gfx1100.
RUN set -eux; \
    cd /opt/maya; \
    python3 tools/vision/glm5next_patch.py third_party/llama.cpp; \
    jobs=$(( $(nproc) / 2 )); [ "$jobs" -ge 1 ] || jobs=1; [ "$jobs" -le 4 ] || jobs=4; \
    cmake -S tools/vision -B build-vision -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_DIR=/opt/maya/third_party/llama.cpp \
        -DSTRATA_VISION_HIP=ON -DSTRATA_VISION_HIP_TARGETS="${HIP_ARCH}" \
        -DSTRATA_PORTABLE=ON \
        -DAMDGPU_TARGETS="${HIP_ARCH}" -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
        -DCMAKE_EXE_LINKER_FLAGS=-no-pie; \
    cmake --build build-vision --target maya-vision -j "$jobs"

# The build must be ISA-portable: assert the ggml options took STRATA_PORTABLE's values (GGML_NATIVE OFF,
# AVX2 on) - one CMakeCache per configure tree (add_subdirectory shares it) - so a future maya bump that
# changes the mechanism fails here instead of at a user's first load.
RUN set -eux; \
    grep -q '^GGML_NATIVE:BOOL=OFF' /opt/maya/build-hip/CMakeCache.txt; \
    grep -q '^GGML_AVX2:BOOL=ON'    /opt/maya/build-hip/CMakeCache.txt; \
    grep -q '^GGML_NATIVE:BOOL=OFF' /opt/maya/build-vision/CMakeCache.txt; \
    grep -q '^GGML_AVX2:BOOL=ON'    /opt/maya/build-vision/CMakeCache.txt; \
    echo "portable ggml ok (engine + vision)"

# The installer's own "already compiled" stamps (see deploy/maya/stamps.py).
RUN set -eux; \
    python3 /tmp/stamps.py write --root /opt/maya --arch "${HIP_ARCH}" --engines "${ENGINES}" --vision; \
    cat /opt/maya/build-hip/MAYA-BUILD.json; \
    cat /opt/maya/build-vision/MAYA-BUILD.json

# The Python environment maya.sh would have made (.venv): setup.PY_PACKAGES as the installer installs them, the
# setup screen (Textual) from the vendored wheels, and pip's own record so the installer does not reinstall.
RUN set -eux; \
    python3 -m venv /opt/maya/.venv; \
    python3 -c 'import sys; sys.path.insert(0, "/opt/maya"); import setup as S; print("\n".join(S.PY_PACKAGES))' \
        > /tmp/py-packages.txt; \
    xargs -r -a /tmp/py-packages.txt /opt/maya/.venv/bin/pip install \
        --no-cache-dir --disable-pip-version-check; \
    /opt/maya/.venv/bin/pip install --no-cache-dir --disable-pip-version-check \
        --no-index --find-links /opt/maya/third_party/wheels textual; \
    /opt/maya/.venv/bin/python -c 'import json, sys; sys.path.insert(0, "/opt/maya"); import setup as S; \
open(sys.prefix + "/.strata-pip.json", "w").write(json.dumps(sorted(set(S.PY_PACKAGES)), indent=0))'

# The runtime tree: the sources (llama.cpp included: get_llama_cpp() reuses a checkout that has ggml/CMakeLists.txt
# and gguf-py) plus only the built binaries and stamps - no CMake object trees in the image.
RUN set -eux; \
    mkdir -p /opt/runtime; \
    cd /opt/maya; \
    tar -cf - --exclude=build-hip --exclude=build-vision --exclude=.venv . | tar -xf - -C /opt/runtime; \
    mkdir -p /opt/runtime/build-hip /opt/runtime/build-vision/bin; \
    for target in $(printf '%s' "$ENGINES" | tr ',' ' '); do cp -a "build-hip/$target" /opt/runtime/build-hip/; done; \
    cp -a build-hip/MAYA-BUILD.json /opt/runtime/build-hip/; \
    for f in build-hip/*.so build-hip/*.so.*; do [ -e "$f" ] && cp -a "$f" /opt/runtime/build-hip/ || true; done; \
    cp -a build-vision/bin/maya-vision /opt/runtime/build-vision/bin/; \
    cp -a build-vision/MAYA-BUILD.json /opt/runtime/build-vision/; \
    for f in build-vision/*.so build-vision/*.so.*; do [ -e "$f" ] && cp -a "$f" /opt/runtime/build-vision/ || true; done; \
    ls -l /opt/runtime/build-hip /opt/runtime/build-vision/bin

# ---------------------------------------------------------------------------------------------------- runtime
FROM ${BASE_IMAGE}

ARG MAYA_REV=N/A
ARG MAYA_COMMIT=N/A
ARG BUILD_DATE=N/A
ARG ROCM_VERSION

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version="maya-${MAYA_REV}-rocm-${ROCM_VERSION}-gfx1100" \
      org.opencontainers.image.revision=$MAYA_COMMIT \
      org.opencontainers.image.title="llm-ops Maya" \
      org.opencontainers.image.description="Project Maya (GLM-5.3-Flash, DeepSeek-V4.1-Flash) HIP engine for gfx1100, prebuilt the way Maya's installer builds it, on AMD's ROCm dev image" \
      org.opencontainers.image.url="https://github.com/hurui200320/llm-ops" \
      org.opencontainers.image.source="https://github.com/hurui200320/llm-ops"

ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    PYTHONUNBUFFERED=1

# curl: the healthcheck and the model downloads during setup.  build-essential: check_hip_pc refuses a start
# without a `c++` on PATH, even when nothing is compiled.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl libgomp1 build-essential python3 python3-venv \
 && apt-get clean -y \
 && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/*

COPY --from=build /opt/runtime /opt/maya
COPY --from=build /opt/maya/.venv /opt/maya/.venv
COPY deploy/maya/entrypoint.sh /entrypoint.sh
COPY deploy/maya/stamps.py /opt/maya/tools/stamps.py
RUN chmod 0755 /entrypoint.sh /opt/maya/tools/stamps.py \
 && mkdir -p /data/models /data/config

WORKDIR /opt/maya
EXPOSE 8080
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
