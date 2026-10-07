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
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # the static libraries also go into libtruss.so

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
  src/kernels/reference/ref_ssm.cu
  src/kernels/reference/ref_attn.cu
  src/kernels/reference/ref_trellis.cu)
target_link_libraries(truss_reference PUBLIC truss_core)

# ---- models: hyperparameters, weight binding, reference forward per architecture
add_library(truss_models STATIC
  src/model/qwen4exp/config.cc
  src/model/qwen4exp/ple.cc
  src/model/qwen4exp/ple_reader.cc
  src/model/qwen4exp/weights.cc
  src/model/qwen4exp/reference.cu)
target_link_libraries(truss_models PUBLIC truss_formats truss_reference)
target_compile_options(truss_models PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# ---- engine: model steps on the fast kernels
# ---- src/cpu: host-side expert tier (AVX2 + FMA + F16C; the build machine is the run machine)
find_package(Threads REQUIRED)
add_library(truss_cpu STATIC src/cpu/expert_q4.cc src/cpu/expert_trellis.cc)
target_include_directories(truss_cpu PUBLIC src)
target_compile_options(truss_cpu PRIVATE -O3 -mavx2 -mfma -mf16c)
target_link_libraries(truss_cpu PUBLIC Threads::Threads)

add_library(truss_engine STATIC
  src/runtime/expert_store.cu
  src/runtime/doorbell.cu
  src/model/qwen4exp/forward.cu)
target_link_libraries(truss_engine PUBLIC truss_models truss_kernels truss_cpu)

# ---- public C API: libtruss.so (include/truss/truss.h), what server/ binds to
add_library(truss SHARED src/api/truss_c.cu)
target_include_directories(truss PUBLIC include)
target_link_libraries(truss PRIVATE truss_engine)
set_target_properties(truss PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# ---- pack-time encoders (Viterbi), used by the pack converter
add_library(truss_encode STATIC
  src/encode/v2one_encode.cu)
target_link_libraries(truss_encode PUBLIC truss_core)

# ---- engine kernels
add_library(truss_kernels STATIC
  src/kernels/moe/moe_window.cu
  src/kernels/moe/moe_prefill.cu
  src/kernels/dense/q8_gemm.cu
  src/kernels/gdn/gdn_prefill.cu
  src/kernels/dsa/dsa_prefill.cu
  src/kernels/dsa/dsa_prepare.cu
  src/kernels/gdn/gdn_prepare.cu
  src/kernels/hc/hc_prefill.cu
  src/kernels/ffn/ffn_ops.cu
  src/kernels/ple/ple_prefill.cu
  src/kernels/sampling/argmax.cu
  src/kernels/sampling/sample.cu
  src/kernels/sampling/penalty.cu
  src/kernels/sampling/spec.cu
  src/kernels/spec/rollback.cu
  src/kernels/mtp/mtp_ops.cu)
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

add_executable(cpu_expert_test tests/unit/cpu_expert_test.cc)
target_compile_options(cpu_expert_test PRIVATE -O2 -mavx2 -mfma -mf16c)
target_link_libraries(cpu_expert_test PRIVATE truss_cpu)
add_executable(cpu_trellis_test tests/unit/cpu_trellis_test.cc)
target_compile_options(cpu_trellis_test PRIVATE -O2 -mavx2 -mfma -mf16c)
target_link_libraries(cpu_trellis_test PRIVATE truss_cpu)
add_executable(q8_gemm_test tests/unit/q8_gemm_test.cu)
target_link_libraries(q8_gemm_test PRIVATE truss_kernels truss_reference)
add_executable(sample_test tests/unit/sample_test.cu)
target_link_libraries(sample_test PRIVATE truss_kernels)
add_executable(spec_test tests/unit/spec_test.cu)
target_link_libraries(spec_test PRIVATE truss_kernels)
add_executable(penalty_rows_test tests/unit/penalty_rows_test.cc)
target_link_libraries(penalty_rows_test PRIVATE truss_kernels)
add_executable(ple_reader_test tests/unit/ple_reader_test.cc)
target_link_libraries(ple_reader_test PRIVATE truss_models)
add_executable(gemv_multi_test tests/unit/gemv_multi_test.cu)
target_link_libraries(gemv_multi_test PRIVATE truss_kernels)

add_executable(gdn_prefill_test tests/unit/gdn_prefill_test.cu)
target_link_libraries(gdn_prefill_test PRIVATE truss_kernels truss_reference)
add_executable(dsa_prefill_test tests/unit/dsa_prefill_test.cu)
target_link_libraries(dsa_prefill_test PRIVATE truss_kernels truss_reference)

add_executable(frac_decode_test tests/unit/frac_decode_test.cu)
target_link_libraries(frac_decode_test PRIVATE truss_kernels truss_reference)
add_executable(codec_v2one_test tests/unit/codec_v2one_test.cu)
target_link_libraries(codec_v2one_test PRIVATE truss_encode)

# ---- tests/layer: model blocks vs llama-paw activations
add_executable(qwen4exp_parity tests/layer/qwen4exp_parity.cu)
target_include_directories(qwen4exp_parity PRIVATE .)
target_link_libraries(qwen4exp_parity PRIVATE truss_models truss_kernels)
add_executable(qwen4exp_dsa_long tests/layer/qwen4exp_dsa_long.cu)
target_include_directories(qwen4exp_dsa_long PRIVATE .)
target_link_libraries(qwen4exp_dsa_long PRIVATE truss_models truss_kernels)

# ---- tools/tk-bench: microbenchmarks
add_executable(tk-bench-ceiling tools/tk-bench/ceiling.cu)
target_include_directories(tk-bench-ceiling PRIVATE src)
add_executable(tk-bench-moe-trace tools/tk-bench/moe_trace.cu)
target_link_libraries(tk-bench-moe-trace PRIVATE truss_kernels)

# Lead Order-2 microbench: trellis K2.5/K3.5 (truss_cpu) vs ggml Q3_K/Q4_K/Q8_0 vec_dot (llama-paw's AVX2 build).
add_executable(tk-bench-cpu-quant tools/tk-bench/cpu_quant.cc)
target_include_directories(tk-bench-cpu-quant PRIVATE src ${LLAMA_PAW_DIR}/ggml/src)
target_link_directories(tk-bench-cpu-quant PRIVATE ${LLAMA_PAW_DIR}/build/bin)
target_link_libraries(tk-bench-cpu-quant PRIVATE truss_cpu ggml-cpu ggml-base Threads::Threads)
set_target_properties(tk-bench-cpu-quant PROPERTIES BUILD_RPATH ${LLAMA_PAW_DIR}/build/bin)
target_compile_options(tk-bench-cpu-quant PRIVATE -O3 -mavx2 -mfma -mf16c)

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
add_executable(qwen4exp_forward tests/layer/qwen4exp_forward.cu)
target_include_directories(qwen4exp_forward PRIVATE .)
target_link_libraries(qwen4exp_forward PRIVATE truss_engine)
add_executable(tk-bench-prefill tools/tk-bench/prefill.cu)
target_link_libraries(tk-bench-prefill PRIVATE truss_engine)
add_executable(tk-bench-decode tools/tk-bench/decode.cu)
target_link_libraries(tk-bench-decode PRIVATE truss_engine)
add_executable(tk-profile tools/tk-bench/profile.cu)
target_link_libraries(tk-profile PRIVATE truss_engine)
add_executable(tk-bench-spec tools/tk-bench/spec.cu)
target_link_libraries(tk-bench-spec PRIVATE truss_engine)
add_executable(tk-parity-kl tools/tk-parity/kl_vs_base.cu)
target_link_libraries(tk-parity-kl PRIVATE truss_engine)
