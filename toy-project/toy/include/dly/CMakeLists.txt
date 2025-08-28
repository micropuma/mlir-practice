# 默认的tablegen生成MLIRDlyIncGen
add_mlir_dialect(Dly dly)
set(LLVM_TARGET_DEFINITIONS DlyCombine.td)
mlir_tablegen(DlyCombine.h.inc -gen-rewriters)
add_public_tablegen_target(DlyCombineIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(DlyShapeInferenceIncGen)

add_custom_target(header2 DEPENDS MLIRDlyIncGen DlyCombineIncGen DlyShapeInferenceIncGen)
