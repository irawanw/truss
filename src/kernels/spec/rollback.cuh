// Rollback helpers for speculative verify windows (qwen4exp::Forward::verify / accept).
//
// Carried row histories (GDN conv rows, PLE conv history) hold the last H rows of a stream. A verify window appends
// T rows; if only the first n are accepted, the history must be the last H rows of [before-window history | rows
// 0 .. n - 1]. tail_rows computes that from a snapshot of the old history and the saved window rows.
#pragma once
#include <cuda_runtime.h>

namespace truss::spec {

// out [H][width] = the last H rows of concat(old [H][width], rows [n][width]); out must not alias old or rows.
void tail_rows(const float * old, int H, const float * rows, int n, int width, float * out, cudaStream_t stream);

}  // namespace truss::spec
