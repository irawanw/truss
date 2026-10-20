// The host-visible plan of the trellis-dense window GEMV (kernels/formats/x3/gemv_window.cu in tqLLM, ported).
// Plain C++ so the format layer can build a list without seeing a kernel: one List = the row parts of one matvec
// that share a codec and an input; the codec's rate picks the kernel.
//
// The k split of a part depends on its shape only, never on n (the number of columns), so column t of a matvec is
// bit-identical for every n: a drafter's verify batch equals plain decode (TRACKER #205: spec == plain).
#pragma once
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace truss::trellis_dense {

constexpr int TD_MAX_PARTS = 4;      // row parts fused in one matvec (one call per run of equal-rate parts)
constexpr int TD_MAX_N = 16;         // columns per call (a verify window is at most 16)
constexpr int TD_MAX_ITEM_SLICES = 160;   // k slices per gemv item: the A stage stays <= 41 KB at n = 8

// gemv block geometry: WK warps split an item's k slices, each warp owns WNT adjacent 16-column tiles, PF k slices
// of weights in flight per warp, fp16 accumulators folded to fp32 every FOLD slices.
struct Geom {
    static constexpr int WK = 8, WNT = 4, PF = 4, FOLD = 2;
    static constexpr int THREADS = WK * 32;
    static constexpr int COLS = WNT * 16;
};

// One row part of the matrix as the window kernels see it.
struct Part {
    const uint32_t * B = nullptr;   // trellis words, tiles k-major (x3_tile_offset)
    const half * suh = nullptr;     // [cols]  the input-side scales
    const half * svh = nullptr;     // [rows]  the output-side scales
    half * A = nullptr;             // scratch [n][cols]: H128(x * suh)
    float * P = nullptr;            // scratch [kparts][n][rows]: per-k-part partial sums
    int rows = 0;                   // output rows of the part
    int kparts = 0;                 // k split across items (divides cols / 16)
    int item0 = 0;                  // first gemv item of this part
    float * y = nullptr;            // the part's output: [n][ldy]
    int ldy = 0;                    // the output row stride
};

struct List {
    Part p[TD_MAX_PARTS];
    int count = 0;
    int cols = 0;                   // input size (shared by all parts)
    int n = 0;                      // batch columns 1..TD_MAX_N
    int items = 0;                  // total gemv items
    int out_rows = 0;               // total output rows (the output row stride)
};

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// k split for one part: doubled while the launch is short of items or an item exceeds TD_MAX_ITEM_SLICES, keeping
// >= 8 slices per warp. Depends on (rows, cols) only, never on n.
inline int kparts(int rows, int cols, int num_sms)
{
    const int groups = rows / Geom::COLS, kslices = cols / 16;
    const int target = 4 * num_sms;
    int kp = 1;
    while ((groups * kp < target || kslices / kp > TD_MAX_ITEM_SLICES) && kslices % (2 * kp) == 0 &&
           kslices / (2 * kp) >= 8 * Geom::WK)
        kp *= 2;
    return kp;
}

inline size_t align256(size_t b) { return (b + 255) & ~(size_t) 255; }

// Scratch of one list of parts (rows[i], shared cols), sized for TD_MAX_N: A then the partial sums.
inline size_t scratch_bytes(const int * rows, int count, int cols, int num_sms)
{
    size_t p = 0;
    for (int i = 0; i < count; ++i)
        p += (size_t) kparts(rows[i], cols, num_sms) * TD_MAX_N * rows[i] * sizeof(float);
    return align256((size_t) count * TD_MAX_N * cols * sizeof(half)) + p;
}

inline void list_begin(List & L, int cols, int n, int out_rows)
{
    L.count = 0;
    L.cols = cols;
    L.n = n;
    L.items = 0;
    L.out_rows = out_rows;
}

// Appends a part; returns false when the list is full.
inline bool list_add(List & L, const uint16_t * B, const half * suh, const half * svh, int rows, float * y, int ldy,
                     int num_sms)
{
    if (L.count == TD_MAX_PARTS) return false;
    Part & p = L.p[L.count++];
    p.B = reinterpret_cast<const uint32_t *>(B);
    p.suh = suh;
    p.svh = svh;
    p.rows = rows;
    p.y = y;
    p.ldy = ldy;
    p.kparts = kparts(rows, L.cols, num_sms);
    p.item0 = L.items;
    p.A = nullptr;
    p.P = nullptr;
    L.items += (rows / Geom::COLS) * p.kparts;
    return true;
}

// Points the parts' A / P at `scratch` (at least scratch_bytes of the same parts).
inline void list_bind_scratch(List & L, void * scratch)
{
    char * base = static_cast<char *>(scratch);
    float * P = reinterpret_cast<float *>(base + align256((size_t) L.count * TD_MAX_N * L.cols * sizeof(half)));
    for (int i = 0; i < L.count; ++i) {
        L.p[i].A = reinterpret_cast<half *>(base) + (size_t) i * TD_MAX_N * L.cols;
        L.p[i].P = P;
        P += (size_t) L.p[i].kparts * TD_MAX_N * L.p[i].rows;
    }
}

// The one definition of the trellis tile order (the expert tables' order, tqLLM's x3_tile_offset): tile
// (k tile, row tile) lives at (k tile * (rows / 16) + row tile) * words, i.e. k-tile major, row-tile minor.
__host__ __device__ constexpr int tile_offset(int k_tile, int row_tile, int row_tiles, int words)
{
    return (k_tile * row_tiles + row_tile) * words;
}

}  // namespace truss::trellis_dense
