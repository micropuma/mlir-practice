get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRIR
        MLIRMlirOptMain
        MLIRSideEffectInterfaces
        MyExtensionCh3
        )
add_llvm_executable(transform-opt transform-opt.cpp)

target_link_libraries(transform-opt PRIVATE ${LIBS})