cd build/

cmake -G Ninja ../ \
   -DLLVM_USE_LINKER=lld \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

ninja -j $(nproc)