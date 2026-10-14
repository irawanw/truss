// RMSNorm of residual rows (kernels/norm): y = w * x / rms(x). Stateless; one per norm weight.
#pragma once
#include "blocks/linear.h"
#include "kernels/norm/rms_norm.cuh"

namespace truss::blocks {

class RmsNorm {
public:
    RmsNorm(Linear & lin, T weight, int d, float eps) : lin_(lin), w_(weight), d_(d), eps_(eps) {}
    // x [T][d] -> y32 and / or y16 (either may be nullptr)
    void forward(engine::Gpu & g, const float * x, int T, float * y32, half * y16) const
    {
        norm::rms(x, lin_.f32(w_), T, d_, eps_, y32, y16, g.s);
    }
    // x += a first (a == nullptr: plain forward), one launch
    void forward_add(engine::Gpu & g, float * x, const float * a, int T, float * y32, half * y16) const
    {
        if (a) norm::add_rms(x, a, lin_.f32(w_), T, d_, eps_, y32, y16, g.s);
        else forward(g, x, T, y32, y16);
    }

private:
    Linear & lin_;
    const T w_;
    const int d_;
    const float eps_;
};

}  // namespace truss::blocks
