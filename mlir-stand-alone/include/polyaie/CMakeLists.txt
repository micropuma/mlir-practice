set(LLVM_TARGET_DEFINITIONS Passes.td)

# Generate pass declarations
mlir_tablegen(Passes.h.inc -gen-pass-decls)

# Add the public tablegen target
add_public_tablegen_target(MLIRPolyAIEPassesIncGen)

# Generate pass documentation
mlir_tablegen(Passes.md -gen-pass-doc)