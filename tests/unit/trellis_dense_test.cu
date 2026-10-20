// trellis_dense_test: the trellis-dense format (PAW X3.1 mul1 tiles) on real matrices.
//   - the window GEMV (the decode path) against an fp64 reference of the same math: the codec's error plus the
//     fp16 mma accumulator
//   - the columns 1..16 bit-identical to n = 1 (the drafter's verify batch must equal plain decode)
//   - the prompt path (reconstruct + GEMM) against the decode path: the two numerics (half2 vs float Hadamard)
//   - K3.5 and K3 matrices
// usage: trellis_dense_test <PAW-27B-X3.1.gguf> [matrix names...]
#include "blocks/linear.h"
#include "core/scratch.h"
#include "engine/gpu.h"
#include "formats/gguf.h"
#include "kernels/reference/ref.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace truss;

namespace {

// The 128-point Hadamard (natural order, scaled by 1/sqrt(128)) of every 128-value block, in double.
void had128(std::vector<double> & v)
{
    const size_t n = v.size();
    for (size_t b = 0; b < n; b += 128) {
        double t[128];
        for (int i = 0; i < 128; ++i) {
            double s = 0;
            for (int j = 0; j < 128; ++j) s += (__builtin_popcount(i & j) & 1) ? -v[b + j] : v[b + j];
            t[i] = s;
        }
        for (int i = 0; i < 128; ++i) v[b + i] = t[i] * 0.08838834764831845;
    }
}

double rel_err(const std::vector<float> & a, const std::vector<double> & b)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        num += d * d;
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : num;
}

double rel_err2(const std::vector<float> & a, const std::vector<float> & b)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        num += d * d;
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : num;
}

int failures = 0;

void check(bool ok, const char * what, double v, double lim)
{
    printf("%-46s %-8s %.3e (limit %.1e)\n", what, ok ? "PASS" : "FAIL", v, lim);
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char ** argv)
{
    const std::string file = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/tqllm/models/PAW-27B-X3.1.gguf";
    std::vector<std::string> names;
    for (int i = 2; i < argc; ++i) names.push_back(argv[i]);
    if (names.empty())
        names = { "blk.0.ffn_gate", "blk.0.ffn_down", "blk.0.attn_gate", "blk.3.attn_q",  "blk.3.attn_k",
                  "blk.3.attn_output", "blk.0.ssm_out" };

    std::unique_ptr<gguf::File> f = gguf::File::open(file);
    engine::Gpu g(0);
    truss::Scratch sc(64u << 20);   // the stage's scratch (the Linear's temporaries)
    g.sc = &sc;
    cudaStream_t s = g.s;

    for (const std::string & name : names) {
        const gguf::Tensor * w = f->find(name + ".m3_trellis");
        if (!w) {
            printf("%-46s SKIP (no %s.m3_trellis)\n", name.c_str(), name.c_str());
            continue;
        }
        const gguf::TrellisGroup * gr = w->trellis;
        if (!gr) {
            printf("%-46s FAIL (no group)\n", name.c_str());
            ++failures;
            continue;
        }
        const int in = (int) gr->in, out = (int) gr->out, k = gr->k;
        printf("== %s: %d x %d, rate %d\n", name.c_str(), out, in, k);

        // the transformed-basis weights (the reference decode): W_hat[out][in] fp32
        float * dW = nullptr;
        const size_t nw = (size_t) in * out;
        TRUSS_CUDA(cudaMalloc(&dW, nw * sizeof(float)));
        // the reference kernel reads the words on the device: copy them there first
        uint16_t * dWords = nullptr;
        TRUSS_CUDA(cudaMalloc(&dWords, gr->trellis->bytes));
        TRUSS_CUDA(cudaMemcpy(dWords, gr->trellis->data, gr->trellis->bytes, cudaMemcpyHostToDevice));
        ref::trellis_dequant(dWords, k, in, out, dW, s);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        std::vector<float> Wh(nw);
        TRUSS_CUDA(cudaMemcpy(Wh.data(), dW, nw * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dW);
        cudaFree(dWords);

        // the scales (F16 in the file)
        std::vector<half> hsuh(in), hsvh(out);
        memcpy(hsuh.data(), gr->suh->data, (size_t) in * sizeof(half));
        memcpy(hsvh.data(), gr->svh->data, (size_t) out * sizeof(half));
        std::vector<float> suh(in), svh(out);
        for (int i = 0; i < in; ++i) suh[i] = __half2float(hsuh[i]);
        for (int i = 0; i < out; ++i) svh[i] = __half2float(hsvh[i]);

        // the input: a fixed pseudo-random row, repeated over the 16 columns (so the columns of a wider call are
        // the same math and must be bit-identical)
        std::vector<half> x(16 * in);
        unsigned st = 12345;
        for (int i = 0; i < in; ++i) {
            st = st * 1664525u + 1013904223u;
            x[i] = __float2half(((float) ((int) (st >> 10) % 2001) - 1000.0f) / 500.0f);
        }
        for (int t = 1; t < 16; ++t) memcpy(&x[(size_t) t * in], x.data(), in * sizeof(half));
        half * dx = nullptr;
        TRUSS_CUDA(cudaMalloc(&dx, 16 * in * sizeof(half)));
        TRUSS_CUDA(cudaMemcpy(dx, x.data(), 16 * in * sizeof(half), cudaMemcpyHostToDevice));

        // the fp64 reference of the decode path: y[o] = svh[o] * H128( sum_k W_hat[o][k] * H128(suh*x)[k] )
        std::vector<double> a(in);
        for (int i = 0; i < in; ++i) a[i] = (double) __half2float(x[i]) * (double) suh[i];
        had128(a);
        std::vector<double> ref(out), p(out);
        for (int o = 0; o < out; ++o) {
            double s2 = 0;
            const float * row = &Wh[(size_t) o * in];
            for (int i = 0; i < in; ++i) s2 += (double) row[i] * a[i];
            p[o] = s2;
        }
        had128(p);
        for (int o = 0; o < out; ++o) ref[o] = p[o] * (double) svh[o];

        // the engine path: the Linear with the trellis matrix
        blocks::Linear lin(g, blocks::Activations::FP16);
        lin.add(w);
        lin.upload(nullptr, {});
        float * dy = nullptr;
        TRUSS_CUDA(cudaMalloc(&dy, (size_t) 16 * out * sizeof(float)));

        // n = 1
        std::vector<float> y1(out);
        TRUSS_CUDA(cudaMemset(dy, 0xff, (size_t) 16 * out * sizeof(float)));
        lin.lin(w, dx, 1, dy);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        TRUSS_CUDA(cudaMemcpy(y1.data(), dy, out * sizeof(float), cudaMemcpyDeviceToHost));
        check(rel_err(y1, ref) < 5e-3, (name + ": decode n=1 vs fp64").c_str(), rel_err(y1, ref), 5e-3);

        // the columns 1..16 bit-identical to n = 1
        std::vector<float> y16(16 * out);
        TRUSS_CUDA(cudaMemset(dy, 0xff, (size_t) 16 * out * sizeof(float)));
        lin.lin(w, dx, 16, dy);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        TRUSS_CUDA(cudaMemcpy(y16.data(), dy, 16 * out * sizeof(float), cudaMemcpyDeviceToHost));
        bool same = true;
        for (int t = 0; t < 16; ++t)
            for (int o = 0; o < out; ++o)
                if (memcmp(&y16[(size_t) t * out + o], &y1[o], sizeof(float)) != 0) same = false;
        check(same, (name + ": columns 1..16 bit-identical to n=1").c_str(), same ? 0 : 1, 0);

        // the prompt path: 32 rows (reconstruct + GEMM)
        const int rows = 32;
        std::vector<half> xm((size_t) rows * in);
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < in; ++i) {
                st = st * 1664525u + 1013904223u;
                xm[(size_t) r * in + i] = __float2half(((float) ((int) (st >> 10) % 2001) - 1000.0f) / 500.0f);
            }
        half * dxm = nullptr;
        float * dym = nullptr;
        TRUSS_CUDA(cudaMalloc(&dxm, (size_t) rows * in * sizeof(half)));
        TRUSS_CUDA(cudaMalloc(&dym, (size_t) rows * out * sizeof(float)));
        TRUSS_CUDA(cudaMemcpy(dxm, xm.data(), (size_t) rows * in * sizeof(half), cudaMemcpyHostToDevice));
        TRUSS_CUDA(cudaMemset(dym, 0xff, (size_t) rows * out * sizeof(float)));
        lin.lin(w, dxm, rows, dym);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        std::vector<float> ypm((size_t) rows * out);
        TRUSS_CUDA(cudaMemcpy(ypm.data(), dym, (size_t) rows * out * sizeof(float), cudaMemcpyDeviceToHost));

        // the same rows through the decode path (16 at a time) and compared
        std::vector<float> yd((size_t) rows * out);
        for (int r0 = 0; r0 < rows; r0 += 16) {
            const int n = std::min(16, rows - r0);
            TRUSS_CUDA(cudaMemset(dy, 0xff, (size_t) 16 * out * sizeof(float)));
            lin.lin(w, dxm + (size_t) r0 * in, n, dy);
            TRUSS_CUDA(cudaStreamSynchronize(s));
            std::vector<float> tmp((size_t) n * out);
            TRUSS_CUDA(cudaMemcpy(tmp.data(), dy, (size_t) n * out * sizeof(float), cudaMemcpyDeviceToHost));
            for (int t = 0; t < n; ++t)
                memcpy(&yd[(size_t) (r0 + t) * out], &tmp[(size_t) t * out], out * sizeof(float));
        }
        check(rel_err2(ypm, yd) < 5e-3, (name + ": prompt path vs decode path").c_str(), rel_err2(ypm, yd), 5e-3);

        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dxm);
        cudaFree(dym);
    }

    printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
