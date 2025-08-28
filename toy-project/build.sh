#!/bin/bash

cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/home/leon/tutorial/mlir-tutorial/install -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 

ninja -j $(nproc)
