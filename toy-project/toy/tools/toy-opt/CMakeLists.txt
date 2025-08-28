add_executable(toy-opt toy-opt.cpp)
set_target_properties(toy-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(toy-opt 
  PRIVATE Toy
  ToyTransforms
  MLIRIR MLIRParser MLIRSupport

  MLIROptLib        # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
  MLIRFuncDialect   # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
  MLIRTransforms    # 对应 #include "mlir/Transforms/Passes.h"
)