# PolyAie
## Intro
This project refers to [polyaie source project](https://github.com/hanchenye/polyaie), tries to migrate codes to MLIR version 20 and AIE Vitis2023. The project aims to reach these targets:  
* migrate to support higher version of aie framework and mlir infrastructure.
* Lower C/C++ code to mlir-aie dialects. Also tries to further fulfill the whole **pipeline** for generating C/C++ code to .exe and .elf code that can run on VCK190 board.
* gain mlir developing skills and experiences. **This project is for self-study use!**

## Todo
1. convert Affine Dialect to MLIR-AIE
2. do optimization

## Build from source
prerequisite: mlir-aie installed, mlir installed

`build polyaie project`
simply run following scripts:
```shell
bash build-poly.sh
```

### Run tests
#### Simple test case
```shell
build/bin/polyaie-opt test/Conversion/simple_vectorize.mlir --polyaie-affine-preprocess --affine-super-vectorize --canonicalize --o build/bin/simple_vectorize.mlir
```
* `polyaie-affine-preprocess`：do preprocess steps for affine dialect.
* `affine-super-vectorize`：do super vectorize pass.
* `--canonicalize`：do canonicalize pass.

#### Lit test
see [test guide](https://mlir.llvm.org/getting_started/TestingGuide/) for detailed reference.

## References
1. [MLIR tutorial](https://www.jeremykun.com/2023/09/07/mlir-using-traits/)
2. [Polyaie project](https://github.com/hanchenye/polyaie)
3. [MLIR vector dialect](https://www.lei.chat/posts/mlir-vector-dialect-and-patterns/)
4. [MLIR AIE project](git@github.com:Xilinx/mlir-aie.git)
5. [MLIR test guide](https://mlir.llvm.org/getting_started/TestingGuide/)