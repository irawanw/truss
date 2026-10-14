#include "blocks/shortconv.h"

#include "engine/stage.h"
#include "kernels/conv/short_conv.cuh"
#include "kernels/spec/rollback.cuh"

namespace truss::blocks {

void ShortConv::alloc_slots(engine::Stage & st, int n_slots, int, int spec_rows)
{
    s_.resize(n_slots);
    for (Slot & s : s_) {
        s.state = st.gpu().alloc<float>(state_floats());
        if (spec_rows) s.snap = st.gpu().alloc<float>(state_floats()), s.rows = st.gpu().alloc<float>((size_t) spec_rows * d_);
    }
}

void ShortConv::forward(engine::Gpu & g, int slot, const half * in16, int T, float * out, bool tentative)
{
    Slot & s = s_[slot];
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    float * bcx = sc.alloc((size_t) T * 3 * d_);
    half * y16 = sc.alloc<half>((size_t) T * d_);
    lin_.lin(w_.in_proj, in16, T, bcx);
    if (tentative) g.copy(s.snap, s.state, state_floats());
    conv::short_conv(bcx, lin_.f32(w_.conv), s.state, T, d_, L_, y16, tentative ? s.rows : nullptr, g.s);
    lin_.lin(w_.out_proj, y16, T, out);
    sc.release(m);
}

void ShortConv::reset(engine::Stage & st, int slot)
{
    TRUSS_CUDA(cudaMemsetAsync(s_[slot].state, 0, sizeof(float) * state_floats(), st.gpu().s));
}

void ShortConv::accept(engine::Stage & st, int slot, int n, const engine::Window & w)
{
    if (n == w.T) return;   // the window advanced the state over all its rows
    Slot & s = s_[slot];
    if (n > 0) spec::tail_rows(s.snap, L_ - 1, s.rows, n, d_, s.state, st.gpu().s);
    else st.gpu().copy(s.state, s.snap, state_floats());
}

void ShortConv::ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out)
{
    out.push_back({ s_[slot].state, state_floats() });
}

}  // namespace truss::blocks
