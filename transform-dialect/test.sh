mlir-opt transform.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{ \
  debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},\
  canonicalize,cse,symbol-dce)" -o transform-next.mlir

