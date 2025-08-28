# 默认的tablegen生成MLIRToyIncGen
add_mlir_dialect(Toy toy)

# 利用tablegen的gen-pass-decls显式生成MLIRToyTransformsIncGen
set(LLVM_TARGET_DEFINITIONS ToyPasses.td)
mlir_tablegen(ToyPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRToyTransformsIncGen)

add_custom_target(header DEPENDS MLIRToyIncGen MLIRToyTransformsIncGen)