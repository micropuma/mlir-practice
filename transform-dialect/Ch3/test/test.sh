../build/bin/transform-opt ./sequence.mlir \
                --pass-pipeline="builtin.module(transform-interpreter{ \
                debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},\
                canonicalize,cse,symbol-dce)" \
                -o result.mlir

../build/bin/transform-opt ./ops.mlir --transform-interpreter \
                --allow-unregistered-dialect --split-input-file \
                -o result2.mlir
