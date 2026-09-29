// CP2 layer parity: each qwen4exp block of the TRUSS reference forward, fed llama-paw's own input for that block,
// against llama-paw's output (tools/tk-parity/llama_dump on a slice from tools/tk-parity/slice_gguf.py).
// Per-block inputs isolate a mismatch to one block. Metric: rel = |ours - ref| / |ref| (L2 over the tensor).
// Runs in llama-paw's activation mode (Q8_1 activations into Q8_0 matmuls, ref::ActQuant): the fp32 reference
// differs from llama-paw by its activation rounding (~0.5% per matmul), which would hide real mismatches.
// usage: qwen4exp_parity <slice.gguf> <dump dir> [gate, default 1e-3]
#include "core/cuda_check.h"
#include "core/device_tensors.h"
#include "core/scratch.h"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/reference.h"
#include "model/qwen4exp/weights.h"
#include "tests/layer/dump.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

namespace {

struct Checker {
    double gate;
    int fails = 0, cases = 0;

    void check(const std::string & what, const float * dev, const test::DumpTensor & want)
    {
        std::vector<float> got(want.f.size());
        TRUSS_CUDA(cudaMemcpy(got.data(), dev, got.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, maxabs = 0;
        bool finite = true;
        for (size_t i = 0; i < got.size(); ++i) {
            const double e = (double) got[i] - want.f[i];
            finite &= std::isfinite(got[i]);
            num += e * e;
            den += (double) want.f[i] * want.f[i];
            maxabs = std::max(maxabs, std::fabs(e));
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const bool ok = finite && rel <= gate;
        ++cases;
        fails += !ok;
        std::printf("%-28s rel %.2e  maxabs %.2e  %s\n", what.c_str(), rel, maxabs, ok ? "PASS" : "FAIL");
    }
};

// device copy of a dump tensor, NaN-filled slack so reads past the valid range show up
struct DevBuf {
    float * p = nullptr;
    explicit DevBuf(size_t n)
    {
        TRUSS_CUDA(cudaMalloc(&p, n * 4));
        TRUSS_CUDA(cudaMemset(p, 0xff, n * 4));
    }
    DevBuf(const test::DumpTensor & t) : DevBuf(t.f.size())
    {
        TRUSS_CUDA(cudaMemcpy(p, t.f.data(), t.f.size() * 4, cudaMemcpyHostToDevice));
    }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf &) = delete;
};

std::string at(const char * base, int l, const char * suffix = "")
{
    return std::string(base) + "-" + std::to_string(l) + suffix;
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <slice.gguf> <dump dir> [gate]\n", argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        std::vector<const gguf::Tensor *> up;
        for (const auto & t : file->tensors())
            if (&t != w.ple_table && &t != w.ple_scale) up.push_back(&t);   // the PLE table is gathered on the host
        const DeviceTensors dev(up);
        Scratch scratch(512ull << 20);
        const test::Dump dump(argv[2]);
        Checker chk{ argc > 3 ? std::atof(argv[3]) : 1e-3 };
        const q::reference::Ctx x{ c, dev, scratch, nullptr, ref::ActQuant::Q8_1 };
        const int T = (int) dump.get("model.input_embed").ne[1];
        std::printf("slice %d layers, %d tokens, weights on device %.2f GB, gate rel <= %.0e\n", c.n_layer, T,
                    dev.bytes() / 1e9, chk.gate);

        for (int l = 0; l < c.n_layer; ++l) {
            const q::Layer & L = w.layers[l];
            scratch.reset();
            // attention-side mix: its input is the previous layer's output (PLE layers: after the PLE block)
            const std::string res_in = l == 0 ? "hc_init" : at("l_last", l - 1);
            if (!c.is_ple(l)) {
                DevBuf res(dump.get(res_in)), mixed((size_t) T * c.d_model), inj((size_t) T * c.hc);
                q::reference::hc_mix(x, L.hc_attn, res.p, T, mixed.p, inj.p);
                chk.check(at("hc_mix attn", l), mixed.p, dump.get(at("hc_mixed", l, "#1")));
                chk.check(at("hc_inject attn", l), inj.p, dump.get(at("hc_inject", l)));
            }
            {   // FFN-side mix: input is the residual after the attention combine
                DevBuf res(dump.get(at("hc_combine", l))), mixed((size_t) T * c.d_model), inj((size_t) T * c.hc);
                q::reference::hc_mix(x, L.hc_ffn, res.p, T, mixed.p, inj.p);
                chk.check(at("hc_mix ffn", l), mixed.p, dump.get(at("hc_mixed", l, "#2")));
                chk.check(at("hc_inject ffn", l), inj.p, dump.get(at("hc_inject", l, "#2")));
            }
            {   // FFN combine: hc_combine + ffn_out * weights(inject) -> l_last
                DevBuf res(dump.get(at("hc_combine", l))), out(dump.get(at("ffn_out", l))),
                    inj(dump.get(at("hc_inject", l, "#2")));
                q::reference::hc_combine(x, res.p, out.p, inj.p, T);
                chk.check(at("hc_combine ffn", l), res.p, dump.get(at("l_last", l)));
            }
        }
        TRUSS_CUDA(cudaDeviceSynchronize());
        std::printf("%d/%d cases pass\n", chk.cases - chk.fails, chk.cases);
        return chk.fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
