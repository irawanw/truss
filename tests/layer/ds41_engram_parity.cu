// blocks::Engram vs tools/tk-parity/ds41_engram_ref.py: the hash ids of a corpus sequence vs the reference
// NgramHashState's (exact), and the gated residual update on a random residual vs dsref Engram math on the same int4
// rows and dense-pack weights (rel gate 1e-2: fp16 rows and wkv input). The sequence runs as two chunks (all but 7
// tokens, then 7), so the second chunk hashes across the boundary from the committed history.
// usage: ds41_engram_parity <ds41-dense.gguf> <dump dir>   (dump dir holds engram-L<L>-sparse.gguf)
#include "blocks/engram.h"
#include "blocks/linear.h"
#include "core/cuda_check.h"
#include "core/scratch.h"
#include "engine/gpu.h"
#include "model/deepseek41/config.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace truss;

namespace {
template <class X> std::vector<X> load(const std::string & path, size_t n)
{
    std::vector<X> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read((char *) v.data(), n * sizeof(X))) throw std::runtime_error("short read " + path);
    return v;
}
}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <ds41-dense.gguf> <dump dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    int L = 1, T = 0;
    {
        std::ifstream m(dir + "/meta.txt");
        std::string k;
        while (m >> k) {
            if (k == "layer") m >> L;
            if (k == "tokens") m >> T;
        }
    }
    const auto f = gguf::File::open(argv[1]);
    f->overlay(dir + "/engram-L" + std::to_string(L) + "-sparse.gguf");
    const auto c = deepseek41::Config::from_gguf(*f);
    const auto & e = c.engram;
    const int ei = e.index(L), B = e.buckets(), hc = c.hc, d = c.d_model;
    const std::string p = "blk." + std::to_string(L) + ".";
    blocks::EngramWeights w { f->find(p + "engram_wkv.weight"), f->find(p + "engram_q.weight"),
                              f->find(p + "engram_k.weight") };
    blocks::EngramDims dims { hc, d, e.heads, e.head_dim, e.max_ngram, e.pad_id, c.rms_eps,
                              e.multipliers.data() + (size_t) ei * e.max_ngram, e.primes.data() + (size_t) ei * B,
                              e.offsets.data() + (size_t) ei * B, e.token_map.data(), (int64_t) e.token_map.size() };
    engine::Gpu g(0);
    blocks::Linear lin(g, blocks::Activations::FP16);
    blocks::Engram::add_tensors(lin, w);
    lin.upload(nullptr, {});
    blocks::Engram eng(lin, w, dims, f->at(p + "engram_embd.q4"));
    eng.build(g);
    eng.alloc(1);
    Scratch sc(blocks::Engram::scratch_per_row(dims) * T + (64u << 20));
    g.sc = &sc;

    const auto tok = load<int32_t>(dir + "/tokens.bin", T), ref_ids = load<int32_t>(dir + "/ids.bin", (size_t) T * B);
    const auto h = load<float>(dir + "/h.bin", (size_t) T * hc * d), ref = load<float>(dir + "/out.bin", h.size());
    float * res = g.alloc<float>(h.size());
    TRUSS_CUDA(cudaMemcpy(res, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    const int T0 = T - 7;
    long bad_ids = 0;
    for (auto [pos0, n] : { std::pair { 0, T0 }, std::pair { T0, T - T0 } }) {
        std::vector<int32_t> ids((size_t) n * B);
        eng.hash(0, tok.data() + pos0, pos0, n, ids.data());
        for (size_t i = 0; i < ids.size(); ++i) bad_ids += ids[i] != ref_ids[(size_t) pos0 * B + i];
        eng.forward(g, 0, tok.data() + pos0, pos0, n, res + (size_t) pos0 * hc * d);
        eng.commit_tokens(0, tok.data() + pos0, n);
    }
    TRUSS_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(h.size());
    TRUSS_CUDA(cudaMemcpy(out.data(), res, out.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0, dn = 0, dd = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        num += (out[i] - (double) ref[i]) * (out[i] - (double) ref[i]), den += (double) ref[i] * ref[i];
        const double a = out[i] - h[i], b = ref[i] - h[i];   // the update alone
        dn += (a - b) * (a - b), dd += b * b;
    }
    const double r_all = std::sqrt(num / den), r_upd = std::sqrt(dn / dd);
    const bool ok = bad_ids == 0 && r_upd <= 1e-2;
    std::printf("L%d T=%d: hash ids %ld of %zu differ; residual rel %.3e, update rel %.3e %s\n", L, T, bad_ids,
                (size_t) T * B, r_all, r_upd, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
