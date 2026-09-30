# TRUSS / Loom trellis engine. Layout and rules: docs/04-architecture.md §8, docs/CODE.md.
cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc CACHE FILEPATH "box's pinned CUDA toolkit")
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "sm_86: RTX 3090 / 3060")
project(truss LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# llama-paw's ggml, used by tests only, as the reference implementation
set(LLAMA_PAW_DIR $ENV{HOME}/llama-paw CACHE PATH "llama-paw checkout (reference for tests)")

# ---- formats: model file readers (host only)
add_library(truss_formats STATIC
  src/formats/gguf.cc
  src/formats/trellis_table.cc)
target_include_directories(truss_formats PUBLIC src)
target_compile_options(truss_formats PRIVATE -Wall -Wextra)

# ---- core: device memory for model tensors, scratch
add_library(truss_core STATIC
  src/core/device_tensors.cc)
target_include_directories(truss_core PUBLIC src)
target_link_libraries(truss_core PUBLIC truss_formats CUDA::cudart)

# ---- reference ops: plain fp32, the definition the fast kernels are tested against
add_library(truss_reference STATIC
  src/kernels/reference/ref_linear.cu
  src/kernels/reference/ref_norm.cu
  src/kernels/reference/ref_trellis.cu)
target_link_libraries(truss_reference PUBLIC truss_core)

# ---- models: hyperparameters, weight binding, reference forward per architecture
add_library(truss_models STATIC
  src/model/qwen4exp/config.cc
  src/model/qwen4exp/weights.cc
  src/model/qwen4exp/reference.cu)
target_link_libraries(truss_models PUBLIC truss_formats truss_reference)
target_compile_options(truss_models PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# ---- pack-time encoders (Viterbi), used by the pack converter
add_library(truss_encode STATIC
  src/encode/v2one_encode.cu)
target_link_libraries(truss_encode PUBLIC truss_core)

# ---- engine kernels
add_library(truss_kernels STATIC
  src/kernels/moe/moe_window.cu
  src/kernels/moe/moe_prefill.cu
  src/kernels/dense/q8_gemm.cu)
target_include_directories(truss_kernels PUBLIC src)
target_link_libraries(truss_kernels PUBLIC CUDA::cublas)

# ---- tests (tests/unit: one binary per op, exit code = pass/fail)
add_executable(moe_window_test tests/unit/moe_window_test.cu)
target_include_directories(moe_window_test PRIVATE ${LLAMA_PAW_DIR}/ggml/include)
target_link_directories(moe_window_test PRIVATE ${LLAMA_PAW_DIR}/build/bin)
target_link_libraries(moe_window_test PRIVATE truss_kernels ggml ggml-base ggml-cuda)
set_target_properties(moe_window_test PROPERTIES BUILD_RPATH ${LLAMA_PAW_DIR}/build/bin)

add_executable(moe_prefill_test tests/unit/moe_prefill_test.cu)
target_link_libraries(moe_prefill_test PRIVATE truss_kernels)

add_executable(q8_gemm_test tests/unit/q8_gemm_test.cu)
target_link_libraries(q8_gemm_test PRIVATE truss_kernels truss_reference)

add_executable(codec_v2one_test tests/unit/codec_v2one_test.cu)
target_link_libraries(codec_v2one_test PRIVATE truss_encode)

# ---- tests/layer: model blocks vs llama-paw activations
add_executable(qwen4exp_parity tests/layer/qwen4exp_parity.cu)
target_include_directories(qwen4exp_parity PRIVATE .)
target_link_libraries(qwen4exp_parity PRIVATE truss_models truss_kernels)

# ---- tools/tk-bench: microbenchmarks
add_executable(tk-bench-ceiling tools/tk-bench/ceiling.cu)
target_include_directories(tk-bench-ceiling PRIVATE src)
add_executable(tk-bench-moe-trace tools/tk-bench/moe_trace.cu)
target_link_libraries(tk-bench-moe-trace PRIVATE truss_kernels)

# ---- tools/tk-pack: model file catalog, pack converter
add_executable(tk-pack-inspect tools/tk-pack/inspect.cc)
target_link_libraries(tk-pack-inspect PRIVATE truss_models)

# ---- tools/tk-parity: reference activations from llama-paw (CP2 layer parity)
add_executable(tk-parity-llama-dump tools/tk-parity/llama_dump.cpp)
target_include_directories(tk-parity-llama-dump PRIVATE ${LLAMA_PAW_DIR}/include ${LLAMA_PAW_DIR}/ggml/include)
target_link_directories(tk-parity-llama-dump PRIVATE ${LLAMA_PAW_DIR}/build/bin)
target_link_libraries(tk-parity-llama-dump PRIVATE llama ggml ggml-base ggml-cpu)
set_target_properties(tk-parity-llama-dump PROPERTIES BUILD_RPATH ${LLAMA_PAW_DIR}/build/bin)

# ---- tools/codec-lab: codebook studies (quantization MSE vs the Gaussian bound)
add_executable(tk-codec-lab tools/codec-lab/viterbi_mse.cu)
