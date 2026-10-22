// Flash-Next vision tower. See vit.h. One kernel per elementwise step, cuBLAS for the matrices.
#include "model/qwen4exp/vision/vit.h"

#include "core/cublas_check.h"
#include "core/cuda_check.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace truss::qwen4exp {

namespace {

constexpr int HD = 72;                                   // head dim (hidden 1152 / 16 heads)

__device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
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

__global__ void k_f32_to_f16(const float * a, __half * b, long n) { long i = blockIdx.x * (long) blockDim.x + threadIdx.x; if (i < n) b[i] = __float2half(a[i]); }

// out16 = LayerNorm(x) over rows of D (eps 1e-6), one block per row
__global__ void k_layernorm(const float * x, const float * w, const float * b, __half * out, int D)
{
    __shared__ float sh[32];
    const float * r = x + (long) blockIdx.x * D;
    float s = 0; for (int i = threadIdx.x; i < D; i += blockDim.x) s += r[i];
    const float mean = block_sum(s, sh) / D;
    float q = 0; for (int i = threadIdx.x; i < D; i += blockDim.x) { float d = r[i] - mean; q += d * d; }
    const float inv = rsqrtf(block_sum(q, sh) / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(long) blockIdx.x * D + i] = __float2half((r[i] - mean) * inv * w[i] + b[i]);
}

// x = (y + bias) + pos-embed taps (patch embed)
__global__ void k_patch_post(float * x, const float * y, const float * bias, const __half * pos, const int * pi, const float * pw, int D)
{
    const int t = blockIdx.x;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float a = y[(long) t * D + j] + bias[j];
        for (int u = 0; u < 4; u++) a += pw[t * 4 + u] * __half2float(pos[(long) pi[t * 4 + u] * D + j]);
        x[(long) t * D + j] = a;
    }
}

// x += y + bias (attention proj / mlp fc2)
__global__ void k_add_res(float * x, const float * y, const float * bias, long n, int D)
{
    long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i] + bias[i % D];
}

__global__ void k_bias_gelu(const float * y, const float * bias, __half * out, long n, int D, bool tanh_form)
{
    long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
    if (i < n) { float v = y[i] + bias[i % D]; out[i] = __float2half(tanh_form ? gelu_tanh(v) : gelu_erf(v)); }
}

__global__ void k_bias(const float * y, const float * bias, float * out, long n, int D)
{
    long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
    if (i < n) out[i] = y[i] + bias[i % D];
}

// qkv [N][3][heads][HD] (+bias) -> q,k with rotary (rotate_half over the full 72), v; all [N][heads][HD] f32
__global__ void k_qkv_rope(const float * qkv, const float * bias, const float * cs, const float * sn, float * q, float * k, float * v, int heads)
{
    const int t = blockIdx.x, D = heads * HD;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const int d = j % HD;
        const float * r = qkv + (long) t * 3 * D;
        float cc = cs[t * HD + d], ss = sn[t * HD + d];
        for (int which = 0; which < 2; which++) {
            const int base = which * D + j - d;
            float xv = r[which * D + j] + bias[which * D + j];
            float xo = d < HD / 2 ? r[base + d + HD / 2] + bias[base + d + HD / 2] : r[base + d - HD / 2] + bias[base + d - HD / 2];
            float rot = d < HD / 2 ? -xo : xo;
            (which ? k : q)[(long) t * D + j] = xv * cc + rot * ss;
        }
        v[(long) t * D + j] = r[2 * D + j] + bias[2 * D + j];
    }
}

// full attention: block per (query, head); scores in shared memory (N <= 8192)
__global__ void k_attn(const float * q, const float * k, const float * v, __half * out, int N, int heads)
{
    extern __shared__ float sc[];
    __shared__ float sh[32], qs[HD];
    const int t = blockIdx.x, h = blockIdx.y, D = heads * HD;
    if (threadIdx.x < HD) qs[threadIdx.x] = q[(long) t * D + h * HD + threadIdx.x];
    __syncthreads();
    const float scale = rsqrtf((float) HD);
    float mx = -1e30f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const float * kr = k + (long) j * D + h * HD;
        float s = 0; for (int d = 0; d < HD; d++) s += qs[d] * kr[d];
        s *= scale; sc[j] = s; mx = fmaxf(mx, s);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = -1e30f; for (int i = 0; i < (int) (blockDim.x >> 5); i++) mx = fmaxf(mx, sh[i]);
    float sum = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) { float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
    const float tot = block_sum(sum, sh);
    __syncthreads();
    // output: thread d < HD sums p_j * v[j][h][d]; split the keys over blockDim/HD groups to use all threads
    const int groups = blockDim.x / HD;
    __shared__ float part[8][HD];
    if (threadIdx.x < groups * HD) {
        const int d = threadIdx.x % HD, g = threadIdx.x / HD;
        float a = 0;
        for (int j = g; j < N; j += groups) a += sc[j] * v[(long) j * D + h * HD + d];
        part[g][d] = a;
    }
    __syncthreads();
    if (threadIdx.x < HD) { float a = 0; for (int g = 0; g < groups; g++) a += part[g][threadIdx.x]; out[(long) t * D + h * HD + threadIdx.x] = __float2half(a / tot); }
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
    void * p = nullptr;
    TRUSS_CUDA(cudaMalloc(&p, t.bytes));
    TRUSS_CUDA(cudaMemcpy(p, t.data, t.bytes, cudaMemcpyHostToDevice));
    owned_.push_back(p);
    if (f32 != (t.type == gguf::Type::F32)) throw std::runtime_error("vision tensor " + t.name + " has an unexpected type");
    return p;
}

Vit::Vit(const gguf::File & f, int device) : dev_(device)
{
    TRUSS_CUDA(cudaSetDevice(dev_));
    const std::string P = "qwen4exp.vision.";
    depth_ = (int) f.get_int(P + "depth"); d_ = (int) f.get_int(P + "hidden_size"); heads_ = (int) f.get_int(P + "num_heads");
    mlp_ = (int) f.get_int(P + "intermediate_size"); out_ = (int) f.get_int(P + "out_hidden_size");
    patch_ = (int) f.get_int(P + "patch_size"); tpatch_ = (int) f.get_int(P + "temporal_patch_size");
    merge_ = (int) f.get_int(P + "spatial_merge_size");
    side_ = (int) std::lround(std::sqrt((double) f.get_int(P + "num_position_embeddings")));
    if (d_ / heads_ != HD || merge_ != 2) throw std::runtime_error("Vit: unsupported vision config");
    TRUSS_CUBLAS(cublasCreate(&blas_));
    auto W = [&](const std::string & n) { return dev(f.at("v." + n), false); };
    auto F = [&](const std::string & n) { return (const float *) dev(f.at("v." + n), true); };
    patch_w_ = W("patch_embed.proj.weight"); patch_b_ = F("patch_embed.proj.bias"); pos_ = W("pos_embed.weight");
    for (int i = 0; i < depth_; i++) {
        const std::string b = "blocks." + std::to_string(i) + ".";
        blocks_.push_back({ W(b + "attn.qkv.weight"), W(b + "attn.proj.weight"), W(b + "mlp.linear_fc1.weight"), W(b + "mlp.linear_fc2.weight"),
                            F(b + "attn.qkv.bias"), F(b + "attn.proj.bias"), F(b + "mlp.linear_fc1.bias"), F(b + "mlp.linear_fc2.bias"),
                            F(b + "norm1.weight"), F(b + "norm1.bias"), F(b + "norm2.weight"), F(b + "norm2.bias") });
    }
    m_n_w_ = F("merger.norm.weight"); m_n_b_ = F("merger.norm.bias");
    m_fc1_w_ = W("merger.linear_fc1.weight"); m_fc1_b_ = F("merger.linear_fc1.bias");
    m_fc2_w_ = W("merger.linear_fc2.weight"); m_fc2_b_ = F("merger.linear_fc2.bias");
}

void Vit::grow(int n)
{
    if (n <= cap_) return;
    auto fr = [](void * p) { if (p) cudaFree(p); };
    fr(x_); fr(y32_); fr(q_); fr(k_); fr(v_); fr(out_rows_); fr(cos_); fr(sin_); fr(posw_); fr(posi_); fr(a16_); fr(px16_);
    const long D = d_, big = std::max<long>(3 * D, std::max<long>(mlp_, 4 * D));
    TRUSS_CUDA(cudaMalloc(&x_, n * D * 4)); TRUSS_CUDA(cudaMalloc(&y32_, n * big * 4));
    TRUSS_CUDA(cudaMalloc(&q_, n * D * 4)); TRUSS_CUDA(cudaMalloc(&k_, n * D * 4)); TRUSS_CUDA(cudaMalloc(&v_, n * D * 4));
    TRUSS_CUDA(cudaMalloc(&out_rows_, (long) (n / 4) * out_ * 4));
    TRUSS_CUDA(cudaMalloc(&cos_, (long) n * HD * 4)); TRUSS_CUDA(cudaMalloc(&sin_, (long) n * HD * 4));
    TRUSS_CUDA(cudaMalloc(&posw_, (long) n * 16)); TRUSS_CUDA(cudaMalloc(&posi_, (long) n * 16));
    TRUSS_CUDA(cudaMalloc(&a16_, n * big * 2)); TRUSS_CUDA(cudaMalloc(&px16_, (long) n * 3 * tpatch_ * patch_ * patch_ * 2));
    cap_ = n;
}

const float * Vit::forward(const float * pixel_values, int gh, int gw, std::map<std::string, std::vector<float>> * taps)
{
    const int N = gh * gw, M = merge_;
    if (gh % M || gw % M || N > MAX_TOKENS || N < 4) throw std::runtime_error("Vit::forward: bad grid " + std::to_string(gh) + "x" + std::to_string(gw));
    TRUSS_CUDA(cudaSetDevice(dev_));
    grow(N);
    const long D = d_, K0 = 3L * tpatch_ * patch_ * patch_;
    // host tables: rotary cos/sin and pos-embed taps, tokens in merge-block order
    std::vector<float> cs((size_t) N * HD), sn((size_t) N * HD), pw((size_t) N * 4);
    std::vector<int> pi((size_t) N * 4);
    const int bw = gw / M;
    for (int t = 0; t < N; t++) {
        const int in_col = t % M, in_row = (t / M) % M, bc = (t / (M * M)) % bw, br = t / (M * M * bw);
        const int row = br * M + in_row, col = bc * M + in_col;
        float fr[HD];
        for (int i = 0; i < HD / 4; i++) {
            const float inv = (float) std::pow(10000.0, -(2.0 * i) / (HD / 2));
            fr[i] = row * inv; fr[HD / 4 + i] = col * inv; fr[HD / 2 + i] = fr[i]; fr[HD / 2 + HD / 4 + i] = fr[HD / 4 + i];
        }
        for (int d = 0; d < HD; d++) { cs[(size_t) t * HD + d] = (float) std::cos((double) fr[d]); sn[(size_t) t * HD + d] = (float) std::sin((double) fr[d]); }
        int ti[2][2]; float tw[2][2];
        const int idx2[2] = { row, col }, sz[2] = { gh, gw };
        for (int a = 0; a < 2; a++) {
            const float src = idx2[a] * (float) (side_ - 1) / (float) std::max(sz[a] - 1, 1);
            const float fl = std::floor(src);
            for (int o = 0; o < 2; o++) {
                ti[a][o] = std::min(std::max((int) fl + o, 0), side_ - 1);
                tw[a][o] = std::max(1.f - std::fabs(src - fl - o), 0.f);
            }
        }
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) { pi[(size_t) t * 4 + a * 2 + b] = ti[0][a] * side_ + ti[1][b]; pw[(size_t) t * 4 + a * 2 + b] = tw[0][a] * tw[1][b]; }
    }
    TRUSS_CUDA(cudaMemcpy(cos_, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice)); TRUSS_CUDA(cudaMemcpy(sin_, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(posi_, pi.data(), pi.size() * 4, cudaMemcpyHostToDevice)); TRUSS_CUDA(cudaMemcpy(posw_, pw.data(), pw.size() * 4, cudaMemcpyHostToDevice));
    // pixels -> f16 -> patch GEMM
    float * pxf = nullptr;
    TRUSS_CUDA(cudaMalloc(&pxf, (size_t) N * K0 * 4));
    TRUSS_CUDA(cudaMemcpy(pxf, pixel_values, (size_t) N * K0 * 4, cudaMemcpyHostToDevice));
    k_f32_to_f16<<<nblk((long) N * K0), 256>>>(pxf, (__half *) px16_, (long) N * K0);
    gemm(blas_, patch_w_, (const __half *) px16_, y32_, (int) K0, (int) D, N);
    k_patch_post<<<N, 256>>>(x_, y32_, patch_b_, (const __half *) pos_, posi_, posw_, (int) D);
    auto tap = [&](const char * name, const float * p, long n) {
        if (!taps) return;
        auto & v = (*taps)[name]; v.resize(n);
        TRUSS_CUDA(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost));
    };
    tap("patch_embed", x_, (long) N * D);
    __half * a16 = (__half *) a16_;
    for (int b = 0; b < depth_; b++) {
        const Block & B = blocks_[b];
        k_layernorm<<<N, 256>>>(x_, B.n1_w, B.n1_b, a16, (int) D);
        gemm(blas_, B.qkv_w, a16, y32_, (int) D, (int) (3 * D), N);
        k_qkv_rope<<<N, 256>>>(y32_, B.qkv_b, cos_, sin_, q_, k_, v_, heads_);
        k_attn<<<dim3(N, heads_), 256, (size_t) N * 4>>>(q_, k_, v_, a16, N, heads_);
        gemm(blas_, B.proj_w, a16, y32_, (int) D, (int) D, N);
        k_add_res<<<nblk((long) N * D), 256>>>(x_, y32_, B.proj_b, (long) N * D, (int) D);
        k_layernorm<<<N, 256>>>(x_, B.n2_w, B.n2_b, a16, (int) D);
        gemm(blas_, B.fc1_w, a16, y32_, (int) D, mlp_, N);
        k_bias_gelu<<<nblk((long) N * mlp_), 256>>>(y32_, B.fc1_b, a16, (long) N * mlp_, mlp_, true);
        gemm(blas_, B.fc2_w, a16, y32_, mlp_, (int) D, N);
        k_add_res<<<nblk((long) N * D), 256>>>(x_, y32_, B.fc2_b, (long) N * D, (int) D);
        if (b == 0) tap("block0_out", x_, (long) N * D);
        if (b == 13) tap("block13_out", x_, (long) N * D);
        if (b == 26) tap("block26_out", x_, (long) N * D);
    }
    // merger: LayerNorm per token, regroup 4 consecutive tokens, fc1 + GELU(erf), fc2
    const int n4 = N / 4, D4 = (int) (4 * D);
    k_layernorm<<<N, 256>>>(x_, m_n_w_, m_n_b_, a16, (int) D);
    gemm(blas_, m_fc1_w_, a16, y32_, D4, D4, n4);
    k_bias_gelu<<<nblk((long) n4 * D4), 256>>>(y32_, m_fc1_b_, a16, (long) n4 * D4, D4, false);
    gemm(blas_, m_fc2_w_, a16, y32_, D4, out_, n4);
    k_bias<<<nblk((long) n4 * out_), 256>>>(y32_, m_fc2_b_, out_rows_, (long) n4 * out_, out_);
    TRUSS_CUDA(cudaGetLastError());
    TRUSS_CUDA(cudaDeviceSynchronize());
    cudaFree(pxf);
    tap("vit_out", out_rows_, (long) n4 * out_);
    return out_rows_;
}

Vit::~Vit()
{
    for (void * p : owned_) cudaFree(p);
    for (void * p : { (void *) x_, (void *) y32_, (void *) q_, (void *) k_, (void *) v_, (void *) out_rows_, (void *) cos_, (void *) sin_, (void *) posw_, (void *) posi_, a16_, px16_ }) if (p) cudaFree(p);
    cublasDestroy(blas_);
}

}  // namespace truss::qwen4exp
