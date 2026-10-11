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

# ONNX Runtime：预编译包（用 scripts/fetch_deps.sh 拉）
add_library(onnxruntime INTERFACE)
target_include_directories(onnxruntime SYSTEM INTERFACE
  ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)
target_link_libraries(onnxruntime INTERFACE
  ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so)

find_package(Threads REQUIRED)

add_executable(mini-infer
  src/main.cc
  src/engine/onnx_engine.cc
  src/batch/dynamic_batcher.cc
  src/gateway/gateway.cc
  src/gateway/rate_limiter.cc
)

target_include_directories(mini-infer PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(mini-infer PRIVATE httplib onnxruntime Threads::Threads)

# 可执行文件直接落在 build/ 下，方便 ./build/mini-infer 这样跑
set_target_properties(mini-infer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# KV cache 对比实验（独立小工具，不参与服务）
add_executable(kv_cache_demo src/experiments/kv_cache_demo.cc)
set_target_properties(kv_cache_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 单元测试（默认构建；Docker 镜像里用 -DMINI_INFER_BUILD_TESTS=OFF 关掉）
option(MINI_INFER_BUILD_TESTS "Build unit tests" ON)
if(MINI_INFER_BUILD_TESTS)
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
    message(FATAL_ERROR "third_party/googletest missing — run scripts/fetch_deps.sh first")
  endif()

  enable_testing()
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)   # 只留 gtest
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/googletest)

  add_executable(unit_tests
    tests/rate_limiter_test.cc
    tests/dynamic_batcher_test.cc
    tests/gateway_test.cc
    src/gateway/rate_limiter.cc
    src/gateway/gateway.cc
    src/batch/dynamic_batcher.cc
  )
  target_include_directories(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(unit_tests PRIVATE GTest::gtest_main httplib Threads::Threads)

  include(GoogleTest)
  gtest_discover_tests(unit_tests PROPERTIES TIMEOUT 60)
endif()
