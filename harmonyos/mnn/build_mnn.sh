#!/bin/sh
set -eu
: "${MNN_ROOT:=$HOME/Developer/MNN-harmony}"
: "${DEVECO_HOME:=$HOME/Applications/DevEco-Studio.app/Contents}"
: "${MNN_BUILD_DIR:=$MNN_ROOT/build-ohos}"
: "${MNN_GPU_PROFILE:=ON}"
MNN_COMMIT=5b74a931887ba5b4d9f65b222cc18dd1a16599a8
if [ ! -d "$MNN_ROOT/.git" ]; then
    git init "$MNN_ROOT"
    git -C "$MNN_ROOT" remote add origin https://github.com/alibaba/MNN.git
    git -C "$MNN_ROOT" fetch --depth 1 origin "$MNN_COMMIT"
    git -C "$MNN_ROOT" checkout --detach FETCH_HEAD
fi
if [ "$(git -C "$MNN_ROOT" rev-parse HEAD)" != "$MNN_COMMIT" ]; then
    echo "MNN checkout must be at $MNN_COMMIT; existing checkout was not changed." >&2
    exit 1
fi
cmake -S "$MNN_ROOT" -B "$MNN_BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$DEVECO_HOME/sdk/default/openharmony/native/build/cmake/ohos.toolchain.cmake" \
    -DOHOS_ARCH=arm64-v8a -DOHOS_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_TRAIN=ON \
    -DMNN_OPENCL=ON -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF \
    -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_ARM82=OFF -DMNN_SME2=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DMNN_GPU_TIME_PROFILE="$MNN_GPU_PROFILE" -DMNN_GPU_PROFILE_SILENT=ON
cmake --build "$MNN_BUILD_DIR" --target MNN -j 8
