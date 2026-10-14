#include "blocks/head.h"

#include <algorithm>

namespace truss::blocks {

void Head::set_draft_vocab(const std::vector<int32_t> & ids)
{
    if (ids.empty()) return;
    const dense::Q8Matrix & O = lin_.q8(out_);
    const int n = (int) ids.size();
    for (int32_t t : ids) engine::require(t >= 0 && t < O.out, "draft_vocab token id out of range");
    draft_map_ = g_.alloc<int>(n);
    TRUSS_CUDA(cudaMemcpy(draft_map_, ids.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    int8_t * q = g_.alloc<int8_t>((size_t) n * O.in);
    half * dd = g_.alloc<half>((size_t) n * O.in / 32);
    dense::q8_gather(O, draft_map_, n, q, dd, g_.s);
    draft_ = { q, dd, O.in, n };
}

void Head::logits(const half * x16, int n, float * logits, bool draft)
{
    Scratch & sc = *g_.sc;
    const size_t m = sc.mark();
    if (lin_.act() == Activations::Q8_1) {
        if (draft && draft_.q) {   // the draft vocabulary's rows only
            const Linear::Act a = lin_.quant(x16, n, draft_.in);
            if (n <= dense::GEMV_ROWS) dense::q8_gemv(draft_, a.q, a.d, n, logits, g_.s);
            else dense::q8_gemm(draft_, a.q, a.d, n, logits, g_.s);
        } else {
            lin_.lin(out_, x16, n, logits);
        }
        sc.release(m);
        return;
    }
    if (n <= dense::GEMV_ROWS) {   // FP16 decode: the fused gemv reads the int8 output matrix once
        dense::q8_gemv_a16(draft && draft_.q ? draft_ : lin_.q8(out_), x16, n, logits, g_.s);
        sc.release(m);
        return;
    }
    const dense::Q8Matrix & O = lin_.q8(out_);
    const int tile = (int) std::min<size_t>(O.out, lin_.w16_elems() / O.in) / 64 * 64;
    float * part = sc.alloc((size_t) n * tile);
    for (int v0 = 0; v0 < O.out; v0 += tile) {
        const int nv = std::min(tile, O.out - v0);
        const dense::Q8Matrix sub{ O.q + (size_t) v0 * O.in, O.d + (size_t) v0 * (O.in / 32), O.in, nv };
        dense::q8_gemm_a16(sub, x16, n, part, lin_.w16(), g_.blas, g_.s);
        TRUSS_CUDA(cudaMemcpy2DAsync(logits + v0, (size_t) O.out * 4, part, (size_t) nv * 4, (size_t) nv * 4, n,
                                     cudaMemcpyDeviceToDevice, g_.s));
    }
    sc.release(m);
}

}  // namespace truss::blocks
