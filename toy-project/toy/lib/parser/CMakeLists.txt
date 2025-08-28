# For a better template to copy, see examples/standalone
set(LLVM_LINK_COMPONENTS
  Support
  )

# 设置源文件
set(PARSER_SOURCES
    AST.cpp
)

# 生成一个静态库
add_library(ParserLib STATIC ${PARSER_SOURCES})

# 包含头文件目录
target_include_directories(ParserLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
