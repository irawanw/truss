// CP2 layer parity: each qwen4exp block of the TRUSS reference forward, fed llama-paw's own input for that block,
// against llama-paw's output (tools/tk-parity/llama_dump on a slice from tools/tk-parity/slice_gguf.py).
// Per-block inputs isolate a mismatch to one block. Metric: rel = |ours - ref| / |ref| (L2 over the tensor).
// Runs with llama-paw's numerics (ref::Numerics::LLAMA, e.g. Q8_1 activations into Q8_0 matmuls): the fp32 reference
// differs from llama-paw by its activation rounding (~0.5% per matmul), which would hide real mismatches.
// Gates (TRACKER #34): 1e-3 where llama's rounding is reproduced (hyper-connections, router); 3e-3 on the FFN
// paths, whose remaining rounding inside llama's ops is not reproduced (llama-paw itself sits 1.0-1.65e-3 from the
// fp32 reference there). A wrong formula shows up as O(1e-1..1).
// usage: qwen4exp_parity <slice.gguf> <dump dir>
#include "core/cuda_check.h"
#include "core/device_tensors.h"
#include "core/scratch.h"
#include "kernels/moe/moe_window.cuh"
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

constexpr double GATE = 1e-3, GATE_FFN = 3e-3;

struct Checker {
    int fails = 0, cases = 0;

    void check(const std::string & what, const float * dev, const test::DumpTensor & want, double gate = GATE)
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

// rel L2 distance of two device vectors (for "fast kernel vs reference" lines)
double rel_dev(const float * a, const float * b, size_t n)
{
    std::vector<float> x(n), y(n);
    TRUSS_CUDA(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) num += (x[i] - y[i]) * (double) (x[i] - y[i]), den += (double) y[i] * y[i];
    return std::sqrt(num / std::max(den, 1e-30));
}

moe::ProjView view(const DeviceTensors & dev, const formats::ExpertTable & t)
{
    return { dev(t.trellis).as<uint16_t>(), dev(t.meta).as<int32_t>(), dev(t.suh).as<half>(), dev(t.svh).as<half>() };
}

std::string at(const char * base, int l, const char * suffix = "")
{
    return std::string(base) + "-" + std::to_string(l) + suffix;
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <slice.gguf> <dump dir>\n", argv[0]);
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
        Checker chk;
        const q::reference::Ctx x{ c, dev, scratch, nullptr, ref::Numerics::LLAMA };
        const q::reference::Ctx x32{ c, dev, scratch, nullptr, ref::Numerics::FP32 };
        void * moe_ws;
        TRUSS_CUDA(cudaMalloc(&moe_ws, moe::workspace_bytes<moe::FlashNext>()));
        moe::workspace_init<moe::FlashNext>(moe_ws, nullptr);
        const int T = (int) dump.get("model.input_embed").ne[1];
        std::printf("slice %d layers, %d tokens, weights on device %.2f GB, gate rel <= %.0e (FFN %.0e)\n",
                    c.n_layer, T, dev.bytes() / 1e9, GATE, GATE_FFN);

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
            if (L.mixer == q::Mixer::GDN) {   // GDN mixer on llama's attention-side mix
                const int kd = c.key_dim(), vd = c.value_dim(), C = 2 * kd + vd, Hv = c.ssm_v_heads;
                DevBuf in(dump.get(at("hc_mixed", l, "#1"))), qkv((size_t) T * C), z((size_t) T * vd),
                    gate((size_t) T * Hv), beta((size_t) T * Hv), conv((size_t) T * C), core((size_t) T * vd),
                    normed((size_t) T * vd), out((size_t) T * c.d_model);
                q::reference::GdnTrace tr;
                tr.qkv = qkv.p; tr.z = z.p; tr.gate = gate.p; tr.beta = beta.p; tr.conv = conv.p; tr.core = core.p;
                tr.normed = normed.p;
                q::reference::gdn(x, L.gdn, in.p, T, out.p, &tr);
                chk.check(at("gdn qkv", l), qkv.p, dump.get(at("linear_attn_qkv_mixed", l)));
                chk.check(at("gdn z", l), z.p, dump.get(at("z", l)));
                chk.check(at("gdn gate", l), gate.p, dump.get(at("gate", l)));
                chk.check(at("gdn beta", l), beta.p, dump.get(at("beta_sigmoid", l)));
                chk.check(at("gdn conv+silu", l), conv.p, dump.get(at("conv_output_silu", l)));
                chk.check(at("gdn delta rule", l), core.p, dump.get(at("attn_output", l)));
                chk.check(at("gdn gated norm", l), normed.p, dump.get(at("final_output", l)));
                chk.check(at("gdn out", l), out.p, dump.get(at("linear_attn_out", l)), GATE_FFN);
                DevBuf lin(dump.get(at("final_output", l)));   // out projection alone, on llama's input
                ref::linear(dev(L.gdn.out), lin.p, out.p, T, nullptr, ref::Numerics::LLAMA);
                chk.check(at("gdn out (llama input)", l), out.p, dump.get(at("linear_attn_out", l)));
                scratch.reset();
            }
            {   // FFN-side mix: input is the residual after the attention combine
                DevBuf res(dump.get(at("hc_combine", l))), mixed((size_t) T * c.d_model), inj((size_t) T * c.hc);
                q::reference::hc_mix(x, L.hc_ffn, res.p, T, mixed.p, inj.p);
                chk.check(at("hc_mix ffn", l), mixed.p, dump.get(at("hc_mixed", l, "#2")));
                chk.check(at("hc_inject ffn", l), inj.p, dump.get(at("hc_inject", l, "#2")));
            }
            {   // FFN: router, routed experts (llama's routing), shared expert, sum
                const test::DumpTensor in = dump.get(at("hc_mixed", l, "#2"));
                DevBuf xin(in), logits((size_t) T * c.n_expert), routed((size_t) T * c.d_model),
                    shared((size_t) T * c.d_model), ffn((size_t) T * c.d_model);
                const q::reference::Routing r = q::reference::route(x, L.moe, xin.p, T, logits.p);
                chk.check(at("router logits", l), logits.p, dump.get(at("ffn_moe_logits", l)));
                const test::DumpTensor ids = dump.get(at("ffn_moe_topk", l, " (cont)"));
                const test::DumpTensor wts = dump.get(at("ffn_moe_weights", l));
                int id_diff = 0;
                double w_err = 0;
                for (size_t i = 0; i < r.ids.size(); ++i) {
                    id_diff += r.ids[i] != ids.i[i];
                    w_err = std::max(w_err, (double) std::fabs(r.weights[i] - wts.f[i]));
                }
                ++chk.cases;
                chk.fails += id_diff != 0;
                std::printf("%-28s ids differ %d/%zu, max |w diff| %.1e  %s\n", at("router top-k", l).c_str(), id_diff,
                            r.ids.size(), w_err, id_diff ? "FAIL" : "PASS");
                q::reference::Routing lr{ std::vector<int32_t>(ids.i.begin(), ids.i.end()), wts.f };
                q::reference::routed(x, L.moe, xin.p, lr, T, routed.p);
                chk.check(at("routed experts", l), routed.p, dump.get(at("ffn_moe_out", l)), GATE_FFN);
                q::reference::shared(x, L.moe, xin.p, T, shared.p);
                chk.check(at("shared expert", l), shared.p, dump.get(at("ffn_shexp_gated", l)), GATE_FFN);
                scratch.reset();
                q::reference::ffn(x, L.moe, xin.p, T, ffn.p);
                chk.check(at("ffn", l), ffn.p, dump.get(at("ffn_out", l)), GATE_FFN);

                // the fast kernel on the same routing: distance to llama-paw and to the fp32 reference (its own
                // compute error; not gated here, the unit test gates it)
                scratch.reset();
                DevBuf fast((size_t) T * c.d_model), ref32((size_t) T * c.d_model);
                int * d_ids;
                float * d_w;
                TRUSS_CUDA(cudaMalloc(&d_ids, ids.i.size() * 4));
                TRUSS_CUDA(cudaMalloc(&d_w, wts.f.size() * 4));
                TRUSS_CUDA(cudaMemcpy(d_ids, ids.i.data(), ids.i.size() * 4, cudaMemcpyHostToDevice));
                TRUSS_CUDA(cudaMemcpy(d_w, wts.f.data(), wts.f.size() * 4, cudaMemcpyHostToDevice));
                moe::Weights mw{ { view(dev, L.moe.gate), view(dev, L.moe.up), view(dev, L.moe.down) }, c.n_expert };
                if (T <= moe::MAX_ROWS) {
                    moe::window<moe::FlashNext>(mw, xin.p, d_ids, d_w, T, fast.p, moe_ws, nullptr);
                } else {   // the window op takes <= MAX_ROWS rows: run the tokens in windows
                    for (int t0 = 0; t0 < T; t0 += moe::MAX_ROWS)
                        moe::window<moe::FlashNext>(mw, xin.p + (size_t) t0 * c.d_model, d_ids + t0 * c.n_expert_used,
                                                    d_w + t0 * c.n_expert_used, std::min(moe::MAX_ROWS, T - t0),
                                                    fast.p + (size_t) t0 * c.d_model, moe_ws, nullptr);
                }
                q::reference::routed(x32, L.moe, xin.p, lr, T, ref32.p);
                const test::DumpTensor want = dump.get(at("ffn_moe_out", l));
                DevBuf lw(want);
                std::printf("%-28s vs llama %.2e, vs fp32 reference %.2e; llama vs fp32 %.2e\n",
                            at("  moe_window kernel", l).c_str(), rel_dev(fast.p, lw.p, want.f.size()),
                            rel_dev(fast.p, ref32.p, want.f.size()), rel_dev(lw.p, ref32.p, want.f.size()));
                cudaFree(d_ids);
                cudaFree(d_w);
                scratch.reset();
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
