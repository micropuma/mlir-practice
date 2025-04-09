../build/bin/transform-opt ./invalid.mlir --transform-interpreter --split-input-file \
                              --verify-diagnostics

../build/bin/transform-opt ./sequence.mlir \
                --pass-pipeline="builtin.module(transform-interpreter{ \
                debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},\
                canonicalize,cse,symbol-dce)" \
                -o result.mlir
