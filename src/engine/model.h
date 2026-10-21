// What a model family implements (docs/ENGINE.md). Two levels:
//   Family  one per loaded model file: hyperparameters the engine needs, and a StageModel per layer range
//   StageModel  the layers [lb, le) of one stage on one device: its blocks, embed, the layer loop, the head
// plus, optionally, a Drafter (speculative drafts, e.g. Flash-Next's MTP block) living on the last stage.
//
// A family is composition only: it creates blocks (src/blocks/), lists the tensors they read, and calls them in
// order. Slots, verify windows, checkpoints, the layer split, expert residency and the C API are the engine's.
#pragma once
#include "engine/block.h"
#include "formats/gguf.h"
#include "runtime/moe_exec.h"
#include "runtime/moe_kernels.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace truss::engine {

class Stage;

// one chunk of tokens on one stage
struct Chunk {
    const int32_t * tokens = nullptr;          // host
    int pos0 = 0, T = 0;
    int slot = 0;
    bool tentative = false;                    // a verify window (accept() commits)
    bool stream = false;                       // a prompt chunk in stream mode (else fetch-sized)
    const int32_t * next = nullptr;            // lookahead: the next chunk's tokens (prefetch of per-token tables)
    int next_T = 0;
};
using LayerHook = std::function<void(int layer, const float * res, int T)>;   // parity tests: residual after a layer

class Drafter {
public:
    virtual ~Drafter() = default;
    // after a committed chunk on the last stage (residual rows of the chunk in the stage's res)
    virtual void on_commit(Stage &, const Chunk &) = 0;
    // after accept(n) of a verify window: the n rows' commit may be deferred to the next draft()
    virtual void on_accept(Stage &, int slot, int n, const Window & w) = 0;
    virtual void flush(Stage &, int slot) = 0;                       // run a deferred commit now
    virtual void park(Stage &, int slot) = 0;                        // the stage switches away from / back to slot
    virtual void unpark(Stage &, int slot) = 0;
    // up to n drafts after `next` (the token at position pos) into out; returns how many
    struct Sampling {
        const void * params = nullptr;          // sampling::SampleParams, nullptr: greedy
        uint64_t counter = 0;
        float * q = nullptr;
    };
    virtual int draft(Stage &, int slot, int32_t next, int pos, int n, int32_t * out, const Sampling & s) = 0;
    // the model's per-draft confidence p_i (the acceptance probability of draft i, the DeepInfer recipe): the
    // varlen verify schedule uses them. nullptr: no confidence (the fixed-length schedule).
    virtual const float * draft_confidence(int * n) const { *n = 0; return nullptr; }
    virtual Stateful * state() = 0;                                  // its per-slot state (registered with the stage)
};

class StageModel {
public:
    virtual ~StageModel() = default;
    // B1: CUDA-graph decode support. A family whose decode kernels take position-dependent values (pos0, splits)
    // by value describes them here; the engine's GraphStep captures one T=1 step and refreshes the values per
    // call. Empty pos_funcs: graph mode unsupported (the caller runs eager).
    struct GraphInfo {
        struct PosFunc { void * func; int arg, nargs; };   // a kernel taking pos0 by value (arg-th of nargs args)
        std::vector<PosFunc> pos_funcs;
        std::function<int64_t(int pos)> epoch;   // re-capture when epoch(pos) changes (nullptr: capture once)
    };
    virtual GraphInfo graph_info() const { return {}; }
    // tensors to upload (stage.lin().add) and the ones linear's a16 scratch is not sized for; host_rows: a table kept
    // in pinned host memory (an untied embedding), nullptr: none
    virtual void add_tensors(Stage &) = 0;
    virtual const gguf::Tensor * host_rows() const { return nullptr; }
    virtual std::vector<const gguf::Tensor *> no_w16() const { return {}; }
    // the stage's routed-expert layers in store order (a drafter's last), and the expert kernels; empty: dense model
    virtual std::vector<runtime::MoeLayer> moe_layers() const { return {}; }
    virtual runtime::MoeKernels moe_kernels() const { return {}; }
    // peak temporaries of a chunk of T rows. decode: the decode path's (the small scratch: a few rows, the decode
    // only buffers); prompt: the prompt path's (a chunk of rows). They differ: the decode path stacks the grouped
    // output LoRA's rows, the prompt path does not.
    virtual size_t scratch_bytes(int T, bool decode) const = 0;
    virtual int extra_buffers() const { return 0; }   // extra [rows][res_width] buffers per buffer set (a drafter's)
    virtual void build(Stage &) = 0;                  // create the blocks (weights are on the device by now)
    // the stage's expert runtime exists (after build(): its memory plan needs the blocks' state allocated first):
    // create the blocks that run routed experts
    virtual void bind_experts(Stage &) {}
    virtual void embed(Stage &, const Chunk &, float * res) = 0;                      // first stage only
    virtual void layers(Stage &, const Chunk &, float * res, const LayerHook & hook) = 0;   // lb .. le - 1
    virtual void head(Stage &, const float * rows, int n, float * logits, bool draft) = 0;   // last stage only
    virtual Drafter * drafter() { return nullptr; }
    virtual std::vector<Stateful *> states() = 0;     // its blocks' per-slot states, in layer order
    // host-resident per-token tables (Flash-Next's PLE): host ms gathering rows, calls, rows read, and the reader's
    // wait / phase split (added to the arguments; reset zeroes the model's counters)
    virtual void table_stats(double & /*ms*/, long & /*calls*/, long & /*rows*/, bool /*reset*/, double * /*wait_ms*/,
                             double * /*phase_ms*/) {}
};

struct DrafterFiles {          // a speculative drafter's inputs (truss_params.mtp etc.)
    std::string path;
    std::vector<int32_t> vocab;
    float min_p = 0.f;
};

class Family {
public:
    virtual ~Family() = default;
    virtual int n_layer() const = 0;
    virtual int n_vocab() const = 0;
    virtual int n_expert() const { return 0; }        // routed experts per layer (0: dense)
    virtual int res_width() const = 0;                // floats per residual row
    virtual int chunk_align() const { return 1; }     // prompt chunks are a multiple of this
    virtual bool has_drafter() const { return false; }
    virtual std::unique_ptr<StageModel> stage(int lb, int le, int n_ctx) const = 0;
};

// general.architecture -> family (src/model/registry.cc); throws for an unknown architecture
std::unique_ptr<Family> open_family(const gguf::File & f, const DrafterFiles & drafter);

}  // namespace truss::engine
