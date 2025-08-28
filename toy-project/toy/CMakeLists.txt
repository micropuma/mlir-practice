set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# 输出调试信息
message(STATUS "cmake current source directories: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "cmake current binary directories: ${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/dly-opt)
# add_subdirectory(tools/toy-opt)
# add_subdirectory(tools/conv-opt)
