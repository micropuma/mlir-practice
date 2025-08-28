add_executable(conv-opt conv_opt.cpp)
set_target_properties(conv-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(conv-opt
  PRIVATE
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)