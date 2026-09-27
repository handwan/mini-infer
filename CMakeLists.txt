cmake_minimum_required(VERSION 3.16)
project(mini_infer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)          # 生成 compile_commands.json（clangd 用）

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)                # 默认 Release：压测数据才有意义
endif()

add_compile_options(-Wall -Wextra)

# SYSTEM：当"系统头"对待 —— 编译器不为第三方代码报警告
add_library(httplib INTERFACE)
target_include_directories(httplib SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/third_party)

find_package(Threads REQUIRED)

add_executable(mini-infer
  src/main.cc

)

target_link_libraries(mini-infer PRIVATE httplib Threads::Threads)

# 可执行文件直接落在 build/ 下，方便 ./build/mini-infer 这样跑
set_target_properties(mini-infer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
