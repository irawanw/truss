// engine::GraphStep: capture, per-step refresh and replay of one all-resident decode step (B1).
#include "engine/graph_step.h"

#include "kernels/sampling/argmax.cuh"

namespace truss::engine {

GraphStep::GraphStep(Pipeline & p, int vocab) : p_(p), vocab_(vocab), gi_(p.graph_info()), s_(p.stream())
{
    require(!p.split(), "GraphStep: a layer split is not supported");
    require(p.all_resident(), "GraphStep: needs every expert resident with no CPU tier");
    require(!gi_.pos_funcs.empty(), "GraphStep: the model names no pos-taking nodes");
    for (const auto & f : gi_.pos_funcs)
        require(f.func && f.arg >= 0 && f.nargs > f.arg && f.nargs <= 12, "GraphStep: bad pos-taking node");
    TRUSS_CUDA(cudaMallocHost(&tok_, sizeof(int32_t)));
    TRUSS_CUDA(cudaMallocHost(&next_h_, sizeof(int)));
}

GraphStep::~GraphStep()
{
    if (e_) cudaGraphExecDestroy(e_);
    if (g_) cudaGraphDestroy(g_);
    if (tok_) cudaFreeHost(tok_);
    if (next_h_) cudaFreeHost(next_h_);
}

void GraphStep::capture(int pos, float * logits, int * next_d)
{
    // the host cursor may be stale (no host code runs per replayed step): re-align it so the capture bakes this
    // block's splits/grid/workspace, then refresh pos0 explicitly before every launch (set_pos in step())
    p_.set_position(pos);
    if (e_) {
        TRUSS_CUDA(cudaGraphExecDestroy(e_));
        TRUSS_CUDA(cudaGraphDestroy(g_));
        e_ = nullptr;
    }
    TRUSS_CUDA(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
    p_.run(tok_, 1);
    p_.head(0, 1, logits);
    sampling::argmax(logits, vocab_, next_d, s_);
    TRUSS_CUDA(cudaMemcpyAsync(next_h_, next_d, sizeof(int), cudaMemcpyDeviceToHost, s_));
    TRUSS_CUDA(cudaStreamEndCapture(s_, &g_));
    TRUSS_CUDA(cudaGraphInstantiateWithFlags(&e_, g_, 0));
    nodes_.clear();
    nodes_.reserve(gi_.pos_funcs.size() * 48);   // no moves after this: args[arg] points at the element's storage
    std::vector<cudaGraphNode_t> ns(8192);
    size_t nn = ns.size();
    TRUSS_CUDA(cudaGraphGetNodes(g_, ns.data(), &nn));
    for (size_t i = 0; i < nn; ++i) {
        cudaGraphNodeType ty;
        TRUSS_CUDA(cudaGraphNodeGetType(ns[i], &ty));
        if (ty != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams kp;
        TRUSS_CUDA(cudaGraphKernelNodeGetParams(ns[i], &kp));
        for (const auto & f : gi_.pos_funcs) {
            if (kp.func != f.func) continue;
            PosNode pn;
            pn.node = ns[i], pn.arg = f.arg, pn.nargs = f.nargs, pn.kp = kp;
            for (int a = 0; a < f.nargs; ++a) pn.args[a] = kp.kernelParams[a];
            nodes_.push_back(pn);
            nodes_.back().args[f.arg] = &nodes_.back().pos_store;
            break;
        }
    }
    require(!nodes_.empty(), "GraphStep: no pos-taking node captured (model/graph mismatch?)");
    epoch_ = gi_.epoch ? gi_.epoch(pos) : 0;
    ++captures_;
}

int32_t GraphStep::step(int pos, int32_t tok, float * logits, int * next_d)
{
    if (!e_ || (gi_.epoch && gi_.epoch(pos) != epoch_)) capture(pos, logits, next_d);
    *tok_ = tok;
    for (auto & n : nodes_) {
        n.pos_store = pos;
        n.kp.kernelParams = n.args;
        TRUSS_CUDA(cudaGraphExecKernelNodeSetParams(e_, n.node, &n.kp));
    }
    TRUSS_CUDA(cudaGraphLaunch(e_, s_));
    TRUSS_CUDA(cudaStreamSynchronize(s_));
    return *next_h_;
}

}  // namespace truss::engine
