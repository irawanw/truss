// Engine-wide size limits every stage, block and model shares.
#pragma once

#include <cstdlib>

namespace truss::engine {

// Chunks of up to FETCH_ROWS rows (decode steps, verify windows, short prompts) fetch only the cold experts their
// routing uses and run on the small resident buffers; longer chunks stream whole expert layers and borrow the
// ExpertStore ring's spare region for their buffers (TRACKER #55, #57). Decode-size fp32 projections are row-invariant
// up to this size (routing must not depend on the window size).
constexpr int FETCH_ROWS = 32;
inline bool fetch_mode(int T) { return T <= FETCH_ROWS; }

// The prompt path's row pieces: a chunk of T rows is processed in pieces of PREFILL_PIECE rows, so the blocks'
// per-row temporaries (the attention's output) are bounded by the piece, not by the chunk. The prompt path's buffers
// are carved from the expert ring's spare region, and the ring's size decides how many experts stay resident
// (TRACKER #201). The pieces must not change the numerics: the dense projections stay whole-chunk.
constexpr int PREFILL_PIECE = 1024;
// the piece size in rows (TRUSS_PIECE_ROWS overrides it: the block and the model must agree)
inline int prefill_piece()
{
    // the env override can only shrink it: the window rings (blocks::AttnDs41::RING) are sized for the default piece
    static const int p = std::getenv("TRUSS_PIECE_ROWS") ? std::atoi(std::getenv("TRUSS_PIECE_ROWS")) : PREFILL_PIECE;
    return p > 0 && p < PREFILL_PIECE ? p : PREFILL_PIECE;
}

}  // namespace truss::engine
