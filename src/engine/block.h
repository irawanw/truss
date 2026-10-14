// What the engine needs from a block that keeps per-sequence state (a mixer's recurrent state, an n-gram window, a
// drafter's pending row, ...). The engine owns WHEN (which slot, after which chunk, inside or after a verify window,
// checkpoint and restore); the block owns HOW (its buffers, how to replay n accepted rows). Every model gets sequence
// slots, speculative windows, checkpoints and the layer split from this, with no code in its folder.
//
// Each block keeps one state per slot (alloc_slots) and indexes it by the slot the stage passes in; nothing is
// swapped on a slot change. A position-indexed cache (K/V) is one n_ctx array that the slots share as sub-ranges of
// seq_ctx positions; a recurrent piece is one buffer per slot.
#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace truss::engine {

class Stage;

struct Window {   // the open verify window of a slot on a stage
    int pos0 = 0, T = 0;
    const int32_t * tokens = nullptr;
};

class Stateful {
public:
    virtual ~Stateful() = default;
    // called once by the stage after every block exists: n slots of seq_ctx positions each; spec_rows > 0: verify
    // windows of up to that many rows will run (allocate the snapshots a rollback needs)
    virtual void alloc_slots(Stage & st, int n_slots, int seq_ctx, int spec_rows) = 0;
    // a new sequence: zero the recurrent state of `slot`
    virtual void reset(Stage &, int /*slot*/) {}
    // commit the first n rows of the slot's open verify window (0 <= n <= w.T); n == w.T: all of it, n == 0: none
    virtual void accept(Stage &, int /*slot*/, int /*n*/, const Window & /*w*/) {}
    // tokens committed to the sequence (a chunk outside a window, or the accepted rows of one), in order
    virtual void commit(Stage &, int /*slot*/, const int32_t * /*tokens*/, int /*n*/) {}
    // checkpoint: the device buffers holding the slot's recurrent state (fixed order) and any host-side state
    virtual void ck_buffers(int /*slot*/, std::vector<std::pair<float *, size_t>> & /*out*/) {}
    virtual void ck_save(int /*slot*/, std::vector<int32_t> & /*host*/) {}
    virtual void ck_load(int /*slot*/, const std::vector<int32_t> & /*host*/, size_t & /*at*/) {}
    // E6 (one slot only): position-indexed arrays whose unused tail may be lent to the expert ring
    struct Positional {
        void * base;
        size_t bytes, row_bytes;
    };
    virtual void positional(std::vector<Positional> & /*out*/) const {}
};

}  // namespace truss::engine
