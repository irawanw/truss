// Trellis rate codes (TRACKER #118, E3). A projection's meta K is either an integer rate 1..8 or a half-integer
// rate stored as 10 * KA + 5: 15 = K1.5, 25 = K2.5, 35 = K3.5 (exllamav3 v1.5.1 "frac": KA, MASK 0xAAAA).
//
// Integer K: a 16x16 tile is a ring of 256 * K bits; weight j's 16-bit state is the ring window ending at bit
// (j + 1) * K. Half-integer K: weight j takes KA + (j & 1) fresh bits, so its window ends at
// S(j) = (2 KA + 1) * (j >> 1) + KA + (j & 1) * (KA + 1), and the ring is 128 * (2 KA + 1) bits = 16 * KA + 8 uint16.
// j is the same ring position in both (lane * 8 + m of the tensor-core fragment order).
#pragma once

#if defined(__CUDACC__)
#define TRUSS_HD __host__ __device__
#else
#define TRUSS_HD
#endif

namespace truss::formats {

TRUSS_HD constexpr bool k_frac(int k) { return k == 15 || k == 25 || k == 35; }
TRUSS_HD constexpr bool k_valid(int k) { return (k >= 1 && k <= 8) || k_frac(k); }
TRUSS_HD constexpr int k_ka(int k) { return k_frac(k) ? k / 10 : k; }                 // whole bits per weight
TRUSS_HD constexpr int k_tile_u16(int k) { return k_frac(k) ? 16 * (k / 10) + 8 : 16 * k; }
TRUSS_HD constexpr int k_tile_bits(int k) { return 16 * k_tile_u16(k); }
TRUSS_HD constexpr double k_bits(int k) { return k_frac(k) ? k / 10 + 0.5 : k; }       // bits per weight
// ring bit where weight j's window ends
TRUSS_HD constexpr int k_window_end(int k, int j)
{
    return k_frac(k) ? (2 * (k / 10) + 1) * (j >> 1) + k / 10 + (j & 1) * (k / 10 + 1) : (j + 1) * k;
}

}  // namespace truss::formats

#undef TRUSS_HD
