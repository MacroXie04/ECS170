#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${DEVECO_HOME:=$HOME/Applications/DevEco-Studio.app/Contents}"
NATIVE="$DEVECO_HOME/sdk/default/openharmony/native"
APP="$ROOT/Stage3App"
mkdir -p "$APP/entry/libs/arm64-v8a"
"$NATIVE/llvm/bin/clang++" --target=aarch64-linux-ohos --sysroot="$NATIVE/sysroot" \
  -std=c++17 -O2 -fPIC -shared "$APP/entry/src/main/cpp/batchprep.cpp" \
  -lace_napi.z -o "$APP/entry/libs/arm64-v8a/libbatchprep.so"
if [ ! -f "$APP/build-profile.json5" ]; then
  cp "$APP/build-profile.template.json5" "$APP/build-profile.json5"
fi
cd "$APP"
export PATH="$DEVECO_HOME/tools/node/bin:$PATH"
export NODE_HOME="$DEVECO_HOME/tools/node"
export DEVECO_SDK_HOME="$DEVECO_HOME/sdk"
"$DEVECO_HOME/tools/ohpm/bin/ohpm" install
"$DEVECO_HOME/tools/hvigor/bin/hvigorw" --mode module -p module=entry@default -p product=default assembleHap --no-daemon
