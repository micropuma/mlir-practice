mlir-opt test.mlir --inline \
    --canonicalize \
    --cse \
    -o inline.mlir
