cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(polyaie LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(AIE REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

set(POLYAIE_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include )
set(POLYAIE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )

set(POLYAIE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLYAIE_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(POLYAIE_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${AIE_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(AIE_INCLUDE_DIRS "/home/douliyang/large/mlir-workspace/mlir-aie/include")
set(AIE_LIB_DIRS "/home/douliyang/large/mlir-workspace/mlir-aie/install/lib")
set(AIE_INCLUDE_INC_DIRS "/home/douliyang/large/mlir-workspace/mlir-aie/build/include")

# 添加头文件目录
include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${AIE_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${AIE_INCLUDE_INC_DIRS}
  ${CMAKE_SOURCE_DIR}/third-party/aie-rt
)

# 添加链接目录
link_directories(
  ${LLVM_BUILD_LIBRARY_DIR}
  ${AIE_LIB_DIRS}
  ${AIE_CMAKE_DIR}/../..
)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/polyaie)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)