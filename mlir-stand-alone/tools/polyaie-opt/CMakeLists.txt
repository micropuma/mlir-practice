get_property(all_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(polyaie-opt
  polyaie-opt.cpp
  )

set_target_properties(polyaie-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

llvm_update_compile_flags(polyaie-opt)

target_link_libraries(polyaie-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRSupport
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRFunctionInterfaces
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIROptLib
  MLIRIR
  MLIRCopyOpInterface    # needed by libAIEVecTransforms.a

  LLVMSupport

  MLIROptLib
  ADF
  AIE

  bootgen-lib # 包含mlir-aie项目使用的第三方库
  cdo_driver_mlir_aie # 包含mlir-aie项目使用的第三方库：aie-rt中的所有runtime
  xaienginecdo_static    # 包含mlir-aie项目使用的第三方库：aie-rt中的所有runtime
  AIERT    # 包含AIERTControl类等的runtime库
  AIECAPI  # aie-capi构建的是静态库，所以需要手动链接

  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils

  MLIRAIEVecUtils   # mlir-aie构建的是静态库，所以需要手动链接

  MLIRAIEToConfiguration
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRAIEVecTransformOps
  MLIRAIEVecTransforms
  MLIRAIEVecToLLVM

  MLIRTransformDialect
  MLIRXLLVMDialect
  
  MLIRPolyAIEPasses
  )

target_include_directories(polyaie-opt PRIVATE
  ${AIE_INCLUDE_DIRS}/aie/Conversion/AIEToConfiguration/AIEToConfiguration.h
  ${AIE_INCLUDE_DIRS}/aie/Targets/AIERT.h
  ${AIE_INCLUDE_DIRS}/aie/Dialect/AIEVec/Utils/Utils.h
)
