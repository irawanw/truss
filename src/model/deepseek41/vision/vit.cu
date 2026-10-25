// DeepSeek-V4.1 vision tower. See vit.h. One kernel per elementwise step, cuBLAS for the matrices and the attention.
#include "model/deepseek41/vision/vit.h"

#include "core/cublas_check.h"
#include "core/cuda_check.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace truss::deepseek41 {

namespace {

constexpr int HD = 64;                                   // head dim (hidden 1024 / 16 heads)
constexpr int RD = HD / 2;                               // rotary frequencies per row: 16 of h, then 16 of w

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.7071067811865476f)); }

__device__ float block_sum(float v, float * sh)
{
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0; for (int i = 0; i < (int) (blockDim.x >> 5); i++) r += sh[i];
    return r;
}

__device__ float block_max(float v, float * sh)
{
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = -1e30f; for (int i = 0; i < (int) (blockDim.x >> 5); i++) r = fmaxf(r, sh[i]);
    return r;
}

__device__ __forceinline__ void store(float * p, float v) { *p = v; }
__device__ __forceinline__ void store(__half * p, float v) { *p = __float2half(v); }

__global__ void k_f32_to_f16(const float * a, __half * b, long n) { long i = blockIdx.x * (long) blockDim.x + threadIdx.x; if (i < n) b[i] = __float2half(a[i]); }

// out = RMSNorm(x) * w over rows of D (eps 1e-6), one block per row
template <class T>
__global__ void k_rmsnorm(const float * x, const float * w, T * out, int D)
{
    __shared__ float sh[32];
    const float * r = x + (long) blockIdx.x * D;
    float q = 0; for (int i = threadIdx.x; i < D; i += blockDim.x) q += r[i] * r[i];
    const float inv = rsqrtf(block_sum(q, sh) / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) store(out + (long) blockIdx.x * D + i, r[i] * inv * w[i]);
}

// out = y + bias (bias may be null), f32 or f16 out
template <class T>
__global__ void k_bias(const float * y, const float * bias, T * out, long n, int D)
{
    long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
    if (i < n) store(out + i, y[i] + (bias ? bias[i % D] : 0.f));
}

// x += y + bias (bias may be null)
__global__ void k_add_res(float * x, const float * y, const float * bias, long n, int D)
{
    long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i] + (bias ? bias[i % D] : 0.f);
}

__global__ void k_bias_gelu(const float * y, const float * bias, __half * out, long n, int D)
{
    long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(gelu_erf(y[i] + bias[i % D]));
}

// y [N][2I] = gate | up -> out [N][I] = silu(gate) * up
__global__ void k_swiglu(const float * y, __half * out, long n, int I)
{
    long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long t = i / I; const int j = (int) (i % I);
    const float g = y[t * 2 * I + j], u = y[t * 2 * I + I + j];
    out[i] = __float2half(g / (1.f + __expf(-g)) * u);
}

// qkv [N][3][D] (+bias) -> q, k with 2D rotary (rotate-half over the head's 64 dims, freq index d mod 32), v; f16 [N][D]
__global__ void k_qkv_rope(const float * qkv, const float * bias, const float * cs, const float * sn, __half * q, __half * k, __half * v, int D)
{
    const int t = blockIdx.x;
    const float * r = qkv + (long) t * 3 * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const int d = j % HD, base = j - d, f = d % RD;
        const float cc = cs[t * RD + f], ss = sn[t * RD + f];
        for (int which = 0; which < 2; which++) {
            const int o = which * D + base;
            const float xv = r[o + d] + bias[o + d];
            const int pd = d < RD ? d + RD : d - RD;
            const float xp = r[o + pd] + bias[o + pd];
            const float y = d < RD ? xv * cc - xp * ss : xv * cc + xp * ss;
            (which ? k : q)[(long) t * D + j] = __float2half(y);
        }
        v[(long) t * D + j] = __float2half(r[2 * D + j] + bias[2 * D + j]);
    }
}

// one block per score row: p = softmax(s) over N keys, f16 out
__global__ void k_softmax(const float * s, __half * p, int N)
{
    __shared__ float sh[32];
    const float * r = s + (long) blockIdx.x * N;
    float mx = -1e30f; for (int j = threadIdx.x; j < N; j += blockDim.x) mx = fmaxf(mx, r[j]);
    mx = block_max(mx, sh);
    float sum = 0; for (int j = threadIdx.x; j < N; j += blockDim.x) sum += __expf(r[j] - mx);
    const float inv = 1.f / block_sum(sum, sh);
    __half * o = p + (long) blockIdx.x * N;
    for (int j = threadIdx.x; j < N; j += blockDim.x) o[j] = __float2half(__expf(r[j] - mx) * inv);
}

// aligner input: vit_out [nh][nw][C] -> [lh*lw][C*9], feature c*9 + ki*3 + kj (F.unfold order), zero past the grid
__global__ void k_unfold(const float * x, __half * out, int nh, int nw, int lw, int C)
{
    const int l = blockIdx.x, bi = l / lw, bj = l % lw;
    for (int f = threadIdx.x; f < C * 9; f += blockDim.x) {
        const int c = f / 9, ki = (f % 9) / 3, kj = f % 3, r = bi * 3 + ki, cc = bj * 3 + kj;
        out[(long) l * C * 9 + f] = __float2half(r < nh && cc < nw ? x[((long) r * nw + cc) * C + c] : 0.f);
    }
}

// span row r: START, then per line lw aligner rows and NEWLINE, then END
__global__ void k_span(const float * rows, const float * st, const float * nl, const float * en, float * out, int lw, int span, int W)
{
    const int r = blockIdx.x;
    const float * src;
    if (r == 0) src = st;
    else if (r == span - 1) src = en;
    else {
        const int i = r - 1, line = i / (lw + 1), pos = i % (lw + 1);
        src = pos == lw ? nl : rows + ((long) line * lw + pos) * W;
    }
    for (int j = threadIdx.x; j < W; j += blockDim.x) out[(long) r * W + j] = src[j];
}

void gemm(cublasHandle_t h, const void * w16, const __half * x16, float * y, int in, int out, int n)
{
    const float one = 1.f, zero = 0.f;     // y[n][out] = x[n][in] * w[out][in]^T
    TRUSS_CUBLAS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in, &one, w16, CUDA_R_16F, in, x16, CUDA_R_16F, in, &zero, y, CUDA_R_32F, out,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int nblk(long n) { return (int) ((n + 255) / 256); }

}  // namespace

void * Vit::dev(const gguf::Tensor & t, bool f32)
{
    if (f32 != (t.type == gguf::Type::F32)) throw std::runtime_error("vision tensor " + t.name + " has an unexpected type");
    void * p = nullptr;
    TRUSS_CUDA(cudaMalloc(&p, t.bytes));
    TRUSS_CUDA(cudaMemcpy(p, t.data, t.bytes, cudaMemcpyHostToDevice));
    owned_.push_back(p);
    return p;
}

Vit::Vit(const gguf::File & f, int device, bool reserve) : dev_(device)
{
    TRUSS_CUDA(cudaSetDevice(dev_));
    const std::string P = "deepseek41.vision.";
    depth_ = (int) f.get_int(P + "num_hidden_layers"); d_ = (int) f.get_int(P + "hidden_size");
    heads_ = (int) f.get_int(P + "num_attention_heads"); inter_ = (int) f.get_int(P + "intermediate_size");
    patch_ = (int) f.get_int(P + "patch_size"); ratio_ = (int) f.get_int(P + "downsample_ratio");
    out_ = (int) f.get_int(P + "out_hidden_size");
    theta_ = (float) f.get_float(P + "rope_theta");
    max_tokens_ = (int) f.get_int(P + "max_image_tokens");
    max_patches_ = max_tokens_ * ratio_ * ratio_;
    if (d_ / heads_ != HD || ratio_ != 3) throw std::runtime_error("deepseek41::Vit: unsupported vision config");
    TRUSS_CUBLAS(cublasCreate(&blas_));
    auto W = [&](const std::string & n) { return dev(f.at(n), false); };
    auto F = [&](const std::string & n) { return (const float *) dev(f.at(n), true); };
    patch_w_ = W("v.patch_embed.proj.weight"); patch_b_ = F("v.patch_embed.proj.bias");
    for (int i = 0; i < depth_; i++) {
        const std::string b = "v.blocks." + std::to_string(i) + ".";
        blocks_.push_back({ W(b + "attn.wqkv.weight"), W(b + "attn.wo.weight"), W(b + "mlp.w1.weight"), W(b + "mlp.w2.weight"),
                            F(b + "attn.wqkv.bias"), F(b + "attn.wo.bias"), F(b + "norm1.weight"), F(b + "norm2.weight") });
    }
    norm_w_ = F("v.norm.weight");
    a1_w_ = W("a.w1.weight"); a1_b_ = F("a.w1.bias"); a2_w_ = W("a.w2.weight"); a2_b_ = F("a.w2.bias");
    span_start_ = F("span.start"); span_nl_ = F("span.newline"); span_end_ = F("span.end");
    if (reserve) grow(max_patches_);
}

int Vit::span_rows(int n_vit_h, int n_vit_w) const
{
    const int lh = (n_vit_h + ratio_ - 1) / ratio_, lw = (n_vit_w + ratio_ - 1) / ratio_;
    return lh * (lw + 1) + 2;
}

void Vit::release()
{
    for (void ** p : { (void **) &x_, (void **) &y32_, (void **) &s32_, (void **) &rows_, (void **) &span_, (void **) &cos_, (void **) &sin_,
                       (void **) &px_, &q16_, &k16_, &v16_, &a16_, &p16_, &px16_ })
        if (*p) { cudaFree(*p); *p = nullptr; }
    cap_ = 0;
}

// workspace for n patches; the aligner rows L and the span S are at most max_image_tokens (forward() checks the span)
void Vit::grow(int n)
{
    if (n <= cap_) return;
    release();
    const long D = d_, K0 = 3L * patch_ * patch_, S = max_tokens_, L = S;
    const long y32 = std::max({ n * 3 * D, n * 2L * inter_, L * out_ });
    const long a16 = std::max({ n * (long) inter_, n * D, L * 9 * D, L * (long) out_ });
    qchunk_ = (int) std::min<long>(n, std::max<long>(32, (16L << 20) / ((long) heads_ * n)));   // ~64 MB of f32 scores
    TRUSS_CUDA(cudaMalloc(&x_, n * D * 4)); TRUSS_CUDA(cudaMalloc(&y32_, y32 * 4));
    TRUSS_CUDA(cudaMalloc(&q16_, n * D * 2)); TRUSS_CUDA(cudaMalloc(&k16_, n * D * 2)); TRUSS_CUDA(cudaMalloc(&v16_, n * D * 2));
    TRUSS_CUDA(cudaMalloc(&a16_, a16 * 2));
    TRUSS_CUDA(cudaMalloc(&s32_, (long) heads_ * qchunk_ * n * 4)); TRUSS_CUDA(cudaMalloc(&p16_, (long) heads_ * qchunk_ * n * 2));
    TRUSS_CUDA(cudaMalloc(&rows_, L * out_ * 4)); TRUSS_CUDA(cudaMalloc(&span_, S * out_ * 4));
    TRUSS_CUDA(cudaMalloc(&cos_, (long) n * RD * 4)); TRUSS_CUDA(cudaMalloc(&sin_, (long) n * RD * 4));
    TRUSS_CUDA(cudaMalloc(&px_, n * K0 * 4)); TRUSS_CUDA(cudaMalloc(&px16_, n * K0 * 2));
    cap_ = n;
}

const float * Vit::forward(const float * pixel_values, int nh, int nw, std::map<std::string, std::vector<float>> * taps)
{
    const int N = nh * nw;
    if (nh < 1 || nw < 1 || N > max_patches_ || span_rows(nh, nw) > max_tokens_)
        throw std::runtime_error("deepseek41::Vit::forward: bad grid " + std::to_string(nh) + "x" + std::to_string(nw));
    TRUSS_CUDA(cudaSetDevice(dev_));
    grow(N);
    const long D = d_, K0 = 3L * patch_ * patch_;
    const int lh = (nh + ratio_ - 1) / ratio_, lw = (nw + ratio_ - 1) / ratio_, L = lh * lw, span = span_rows(nh, nw);
    // 2D rotary tables, patches row-major: freq i < 16 = row * theta^(-2i/32), i >= 16 the same of the column
    std::vector<float> cs((size_t) N * RD), sn((size_t) N * RD);
    for (int t = 0; t < N; t++)
        for (int i = 0; i < RD; i++) {
            const int half = RD / 2, fi = i % half;
            const float inv = (float) (1.0 / std::pow((double) theta_, (2.0 * fi) / RD));
            const float a = (float) (i < half ? t / nw : t % nw) * inv;
            cs[(size_t) t * RD + i] = std::cos(a); sn[(size_t) t * RD + i] = std::sin(a);
        }
    TRUSS_CUDA(cudaMemcpy(cos_, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(sin_, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
    auto tap = [&](const std::string & name, const float * p, long n) {
        if (!taps) return;
        auto & v = (*taps)[name]; v.resize(n);
        TRUSS_CUDA(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost));
    };
    // patch embed: pixels -> f16 -> GEMM + bias
    TRUSS_CUDA(cudaMemcpy(px_, pixel_values, (size_t) N * K0 * 4, cudaMemcpyHostToDevice));
    k_f32_to_f16<<<nblk(N * K0), 256>>>(px_, (__half *) px16_, N * K0);
    gemm(blas_, patch_w_, (const __half *) px16_, y32_, (int) K0, (int) D, N);
    k_bias<float><<<nblk(N * D), 256>>>(y32_, patch_b_, x_, N * D, (int) D);
    tap("patch_embed", x_, N * D);
    __half * a16 = (__half *) a16_, * q16 = (__half *) q16_, * k16 = (__half *) k16_, * v16 = (__half *) v16_, * p16 = (__half *) p16_;
    const float scale = 1.f / std::sqrt((float) HD), one = 1.f, zero = 0.f;
    for (int b = 0; b < depth_; b++) {
        const Block & B = blocks_[b];
        k_rmsnorm<__half><<<N, 256>>>(x_, B.n1_w, a16, (int) D);
        gemm(blas_, B.qkv_w, a16, y32_, (int) D, (int) (3 * D), N);
        k_qkv_rope<<<N, 256>>>(y32_, B.qkv_b, cos_, sin_, q16, k16, v16, (int) D);
        for (int i0 = 0; i0 < N; i0 += qchunk_) {
            const int nq = std::min(qchunk_, N - i0);
            // s[h][i][j] = scale * q[i0+i][h] . k[j][h]   (column-major: C[N][nq] per head = K_h^T Q_h)
            TRUSS_CUBLAS(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, nq, HD, &scale,
                                                    k16, CUDA_R_16F, (int) D, HD, q16 + (long) i0 * D, CUDA_R_16F, (int) D, HD,
                                                    &zero, s32_, CUDA_R_32F, N, (long) nq * N, heads_, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            k_softmax<<<heads_ * nq, 256>>>(s32_, p16, N);
            // o[i0+i][h] = sum_j p[h][i][j] v[j][h]   (C[HD][nq] = V_h P_h, written into a16 rows i0.. at head offset)
            TRUSS_CUBLAS(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_N, CUBLAS_OP_N, HD, nq, N, &one,
                                                    v16, CUDA_R_16F, (int) D, HD, p16, CUDA_R_16F, N, (long) nq * N,
                                                    &zero, a16 + (long) i0 * D, CUDA_R_16F, (int) D, HD, heads_, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        gemm(blas_, B.wo_w, a16, y32_, (int) D, (int) D, N);
        k_add_res<<<nblk(N * D), 256>>>(x_, y32_, B.wo_b, N * D, (int) D);
        k_rmsnorm<__half><<<N, 256>>>(x_, B.n2_w, a16, (int) D);
        gemm(blas_, B.w1_w, a16, y32_, (int) D, 2 * inter_, N);
        k_swiglu<<<nblk((long) N * inter_), 256>>>(y32_, a16, (long) N * inter_, inter_);
        gemm(blas_, B.w2_w, a16, y32_, inter_, (int) D, N);
        k_add_res<<<nblk(N * D), 256>>>(x_, y32_, nullptr, N * D, (int) D);
        if (b == 0 || b == depth_ / 2 - 1 || b == depth_ - 1) tap("block" + std::to_string(b) + "_out", x_, N * D);
    }
    // final RMSNorm (in place, f32: the aligner gathers from it), then aligner: 3x3 unfold, w1 + GELU(erf), w2
    k_rmsnorm<float><<<N, 256>>>(x_, norm_w_, x_, (int) D);
    tap("vit_out", x_, N * D);
    k_unfold<<<L, 256>>>(x_, a16, nh, nw, lw, (int) D);
    gemm(blas_, a1_w_, a16, y32_, (int) (9 * D), out_, L);
    k_bias_gelu<<<nblk((long) L * out_), 256>>>(y32_, a1_b_, a16, (long) L * out_, out_);
    gemm(blas_, a2_w_, a16, y32_, out_, out_, L);
    k_bias<float><<<nblk((long) L * out_), 256>>>(y32_, a2_b_, rows_, (long) L * out_, out_);
    k_span<<<span, 256>>>(rows_, span_start_, span_nl_, span_end_, span_, lw, span, out_);
    TRUSS_CUDA(cudaGetLastError());
    TRUSS_CUDA(cudaDeviceSynchronize());
    tap("rows", rows_, (long) L * out_);
    return span_;
}

Vit::~Vit()
{
    release();
    for (void * p : owned_) cudaFree(p);
    cublasDestroy(blas_);
}

}  // namespace truss::deepseek41
