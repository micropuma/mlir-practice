#!/bin/bash

BUILD_DIR="build"

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"

pushd "$BUILD_DIR"

BUILD_SYSTEM=Ninja
BUILD_DIR=./build-`echo ${BUILD_SYSTEM}| tr '[:upper:]' '[:lower:]'`

LLVM_BUILD_DIR=../llvm/build/

export PATH=/home/douliyang/large/mlir-workspace/mlir-aie/llvm/install/bin:$PATH

cmake .. \
    -DLLVM_DIR=$PWD/../../mlir-aie/build/lib/cmake/llvm \
    -DMLIR_DIR=$PWD/../../mlir-aie/build/lib/cmake/mlir \
    -DAIE_DIR=$PWD/../../mlir-aie/build/lib/cmake/aie \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=/home/douliyang/large/mlir-workspace/mlir-aie/llvm/install


make -j ${nproc}

popd




