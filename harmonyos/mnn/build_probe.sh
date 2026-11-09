#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${MNN_ROOT:=$HOME/Developer/MNN-harmony}"
: "${DEVECO_HOME:=$HOME/Applications/DevEco-Studio.app/Contents}"
NATIVE="$DEVECO_HOME/sdk/default/openharmony/native"
APP="$ROOT/Stage3App"
mkdir -p "$APP/entry/libs/arm64-v8a"
if [ ! -f "$MNN_ROOT/build-ohos/libMNN.a" ]; then
 echo "Build MNN first with harmonyos/mnn/build_mnn.sh" >&2
 exit 1
fi
if [ ! -f "$APP/build-profile.json5" ]; then
 cp "$APP/build-profile.template.json5" "$APP/build-profile.json5"
fi
"$NATIVE/llvm/bin/clang++" --target=aarch64-linux-ohos --sysroot="$NATIVE/sysroot" \
 -std=c++17 -static-libstdc++ -O2 -fPIC -shared -DENABLE_MNN_PROBE \
 -I"$MNN_ROOT/include" -I"$MNN_ROOT/source" -I"$MNN_ROOT/schema/current" \
 -I"$MNN_ROOT/3rd_party/flatbuffers/include" \
 -I"$MNN_ROOT/tools/train/source/nn" -I"$MNN_ROOT/tools/train/source/optimizer" \
 "$APP/entry/src/main/cpp/batchprep.cpp" \
 -Wl,--whole-archive "$MNN_ROOT/build-ohos/libMNN.a" -Wl,--no-whole-archive \
 -lace_napi.z -lmindspore_lite_ndk -lvulkan -lhilog_ndk.z -ldl \
 -o "$APP/entry/libs/arm64-v8a/libbatchprep.so"
cd "$APP"
export PATH="$DEVECO_HOME/tools/node/bin:$PATH"
export NODE_HOME="$DEVECO_HOME/tools/node"
export DEVECO_SDK_HOME="$DEVECO_HOME/sdk"
"$DEVECO_HOME/tools/ohpm/bin/ohpm" install
"$DEVECO_HOME/tools/hvigor/bin/hvigorw" --mode module -p module=entry@default -p product=default assembleHap --no-daemon
