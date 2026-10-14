// Engine-wide size limits every stage, block and model shares.
#pragma once

namespace truss::engine {

// Chunks of up to FETCH_ROWS rows (decode steps, verify windows, short prompts) fetch only the cold experts their
// routing uses and run on the small resident buffers; longer chunks stream whole expert layers and borrow the
// ExpertStore ring's spare region for their buffers (TRACKER #55, #57). Decode-size fp32 projections are row-invariant
// up to this size (routing must not depend on the window size).
constexpr int FETCH_ROWS = 32;
inline bool fetch_mode(int T) { return T <= FETCH_ROWS; }

}  // namespace truss::engine
