# Tell Tablegen to use MyExtension.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtension.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(MyExtension.h.inc -gen-op-decls)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)

# Tell Tablegen to use MyExtensionTypes.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtensionTypes.td)

# Ask Tablegen to generate typedef declarations and definitions from ODS.
mlir_tablegen(MyExtensionTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MyExtensionTypes.cpp.inc -gen-typedef-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(MyExtensionCh2IncGen)


