mlir-opt fusion.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{ \
  debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},\
  canonicalize,cse,symbol-dce)" -o result1.mlir

mlir-opt test.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{ \
  debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})"

