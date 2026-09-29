# TRUSS / Loom trellis engine. Layout and rules: docs/04-architecture.md §8, docs/CODE.md.
cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc CACHE FILEPATH "box's pinned CUDA toolkit")
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "sm_86: RTX 3090 / 3060")
project(truss LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# llama-paw's ggml, used by tests only, as the reference implementation
set(LLAMA_PAW_DIR $ENV{HOME}/llama-paw CACHE PATH "llama-paw checkout (reference for tests)")

# ---- engine kernels
add_library(truss_kernels STATIC
  src/kernels/moe/moe_window.cu)
target_include_directories(truss_kernels PUBLIC src)

# ---- tests (tests/unit: one binary per op, exit code = pass/fail)
add_executable(moe_window_test tests/unit/moe_window_test.cu)
target_include_directories(moe_window_test PRIVATE ${LLAMA_PAW_DIR}/ggml/include)
target_link_directories(moe_window_test PRIVATE ${LLAMA_PAW_DIR}/build/bin)
target_link_libraries(moe_window_test PRIVATE truss_kernels ggml ggml-base ggml-cuda)
set_target_properties(moe_window_test PROPERTIES BUILD_RPATH ${LLAMA_PAW_DIR}/build/bin)

# ---- tools/tk-bench: microbenchmarks
add_executable(tk-bench-ceiling tools/tk-bench/ceiling.cu)
target_include_directories(tk-bench-ceiling PRIVATE src)
add_executable(tk-bench-moe-trace tools/tk-bench/moe_trace.cu)
target_link_libraries(tk-bench-moe-trace PRIVATE truss_kernels)

# ---- tools/codec-lab: codebook studies (quantization MSE vs the Gaussian bound)
add_executable(tk-codec-lab tools/codec-lab/viterbi_mse.cu)
