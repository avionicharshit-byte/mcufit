#!/bin/bash
# Rebuilds src/mcufit/wasm/tflm.js - the TFLM benchmark compiled to WebAssembly.
# Shipped in the wheel and copied into web/wasm/ at deploy time.
# Requires emscripten (brew install emscripten) and gmake >= 3.82.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$HOME/.cache/mcufit/tflm-wasm}"
[ -d "$SRC" ] || git clone --depth 1 https://github.com/tensorflow/tflite-micro.git "$SRC"
cd "$SRC"
gmake -f tensorflow/lite/micro/tools/make/Makefile tflm_benchmark -j8 \
  BUILD_TYPE=default CC_TOOL=emcc CXX_TOOL=em++ AR_TOOL=emar
OBJ=$(find gen -name generic_model_benchmark.o | head -1)
DIR=$(dirname "$OBJ")
em++ -O2 "$OBJ" "$DIR/metrics.o" \
  "$(find gen -name show_meta_data.o | head -1)" \
  gen/*/lib/libtensorflow-microlite.a \
  -o "$ROOT/src/mcufit/wasm/tflm.js" \
  -sMODULARIZE=1 -sEXPORT_NAME=createTflmModule -sINVOKE_RUN=0 \
  -sEXPORTED_RUNTIME_METHODS=FS,callMain -sALLOW_MEMORY_GROWTH=1 \
  -sENVIRONMENT=web,node -sSINGLE_FILE=1
mkdir -p "$ROOT/web/wasm"
cp "$ROOT/src/mcufit/wasm/tflm.js" "$ROOT/web/wasm/tflm.js"   # for serving web/ locally
echo "wrote $ROOT/src/mcufit/wasm/tflm.js"
