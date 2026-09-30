// MTP (multi-token prediction) draft block input (qwen4exp::Mtp, llama-paw graph_mtp):
//   join  out [T][hc][2d] fp16 = [RMSNorm(emb [T][d]) * enorm | hn [T][hc][d]] per hc stream, the eh_proj input.
//         hn is the previous hidden state already RMS-normed per stream and scaled by hnorm (hc::norm).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::mtp {

void join(const float * emb, const float * enorm, const half * hn, int T, int hc, int d, float eps, half * out,
          cudaStream_t stream);

}  // namespace truss::mtp
