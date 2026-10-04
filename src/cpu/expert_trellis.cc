#include "cpu/expert_trellis.h"
#include "formats/trellis_k.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace truss::cpu {
namespace {

constexpr int MAXR = 8, MAX_IN = 4096, MAX_COLS = 4096;
// kernel lane order: after the two 64-bit shifts and _mm256_shuffle_ps(s0, s1, 0x88), element e holds weight m
constexpr int ORDER[8] = { 0, 1, 4, 5, 2, 3, 6, 7 };

// codebook constants: h * 1/147.7 - 10.39 in fp16 (codec_mul1.cuh)
inline float k_inv() { return _cvtsh_ss(0x1eee); }
inline float k_bias() { return _cvtsh_ss(0xc931); }

// TRUSS_TRELLIS_ROUND=1: decoded weights rounded to fp16 like the GPU (gemv_tiles4 / gemv_tiles only)
bool round_weights()
{
    static const bool v = getenv("TRUSS_TRELLIS_ROUND") && atoi(getenv("TRUSS_TRELLIS_ROUND"));
    return v;
}

inline float round16(float x) { return _cvtsh_ss(_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT)); }

// fp32 -> nearest-even fp16 precision (10 mantissa bits) on the bits: 3 integer ops instead of cvtps_ph + cvtph_ps
// (~3 cycles per vector on Zen 2). Exact for values in the fp16 normal range, which every mul1 codebook value is
// (|v| <= 3.5, the closest to 0 is far above 6.1e-5); same result as the F16C round trip there.
inline __m256 round16_bits(__m256 v)
{
    const __m256i b = _mm256_castps_si256(v);
    const __m256i odd = _mm256_and_si256(_mm256_srli_epi32(b, 13), _mm256_set1_epi32(1));
    const __m256i r = _mm256_add_epi32(b, _mm256_add_epi32(odd, _mm256_set1_epi32(0xfff)));
    return _mm256_castsi256_ps(_mm256_and_si256(r, _mm256_set1_epi32(~0x1fff)));
}

// 128-point Walsh-Hadamard (Sylvester order, y[i] = sum_j (-1)^popcount(i & j) x[j]) scaled 1/sqrt(128), in place
void hadamard128(float * v)
{
    for (int h = 1; h < 8; h <<= 1)   // strides 1, 2, 4: scalar butterflies inside each 8-group
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b, v[j + h] = a - b;
            }
    for (int h = 8; h < 128; h <<= 1)   // strides 8 .. 64: 8-wide
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; j += 8) {
                const __m256 a = _mm256_loadu_ps(v + j), b = _mm256_loadu_ps(v + j + h);
                _mm256_storeu_ps(v + j, _mm256_add_ps(a, b));
                _mm256_storeu_ps(v + j + h, _mm256_sub_ps(a, b));
            }
    const __m256 s = _mm256_set1_ps(0.088388347648f);
    for (int j = 0; j < 128; j += 8) _mm256_storeu_ps(v + j, _mm256_mul_ps(_mm256_loadu_ps(v + j), s));
}

// 64 bits of a tile's cyclic stream (`words` uint32, MSB first) starting at bit p (p >= -32), MSB first in the result
inline uint64_t window_wrap(const uint32_t * t, int words, int p)
{
    const int i = p >> 5, o = p & 31;   // floor division
    auto at = [&](int k) -> uint64_t { return t[k < 0 ? k + words : k >= words ? k - words : k]; };
    const uint64_t hi = (at(i) << 32) | at(i + 1);
    return o ? (hi << o) | (at(i + 2) >> (32 - o)) : hi;
}

template <int R, bool ROUND>
void gemv_tiles(const TrellisMat & W, const float * P, int nt0, int nt1, float * c, int ldc)
{
    // rs: row stride of P, trellis_prep_floats(in, 1) like gemv_tiles4 (the pool preps rows one at a time at that step;
    // this kernel used 2 * in, wrong for R > 1 through the pool)
    const int K = W.K, words = 8 * K, KT = W.in / 16, NT = W.out / 16, rs = trellis_prep_floats(W.in, 1);
    // per lane: where its 64-bit window starts. Lanes >= 1 never wrap and never read past the tile when they need
    // bits from a third word, except the last lanes; those (and lane 0, which starts before bit 0) take the slow path.
    int wi[32], wo[32];
    bool slow[32];
    for (int l = 0; l < 32; ++l) {
        const int p = (l * 8 + 1) * K - 16;
        wi[l] = p >> 5, wo[l] = p & 31;
        slow[l] = p < 0 || wi[l] + 2 >= words;
    }
    const __m256i sh0 = _mm256_setr_epi64x(48, 48 - K, 48 - 2 * K, 48 - 3 * K);
    const __m256i sh1 = _mm256_setr_epi64x(48 - 4 * K, 48 - 5 * K, 48 - 6 * K, 48 - 7 * K);
    const __m256i mask = _mm256_set1_epi32(0xffff), mul = _mm256_set1_epi32((int) 0x83DCD12Du);
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1), h0 = _mm256_set1_epi32(1024);
    const __m256 kinv = _mm256_set1_ps(k_inv()), kbias = _mm256_set1_ps(k_bias());
    for (int nt = nt0; nt < nt1; ++nt)
        for (int n = 0; n < 8; ++n) {
            // two accumulators per row (even / odd lane quad): one chain would serialize on the FMA latency
            __m256 acc[2][R];
            for (int r = 0; r < R; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
            for (int kt = 0; kt < KT; ++kt) {
                const uint32_t * t = W.tiles + ((size_t) kt * NT + nt) * words;
                const float * pk = P + (size_t) kt * 32;
                for (int q = 0; q < 4; ++q) {
                    const int l = n * 4 + q;
                    uint64_t w;
                    if (slow[l]) {
                        w = window_wrap(t, words, (l * 8 + 1) * K - 16);
                    } else {
                        const uint64_t hi = ((uint64_t) t[wi[l]] << 32) | t[wi[l] + 1];
                        w = wo[l] ? (hi << wo[l]) | (t[wi[l] + 2] >> (32 - wo[l])) : hi;
                    }
                    const __m256i wv = _mm256_set1_epi64x((long long) w);
                    const __m256i s0 = _mm256_srlv_epi64(wv, sh0), s1 = _mm256_srlv_epi64(wv, sh1);
                    __m256i st = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(s0), _mm256_castsi256_ps(s1), 0x88));
                    st = _mm256_mullo_epi32(_mm256_and_si256(st, mask), mul);
                    const __m256i sum = _mm256_madd_epi16(_mm256_maddubs_epi16(st, ones8), ones16);
                    __m256 v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, h0)), kinv, kbias);
                    if (ROUND) v = _mm256_cvtph_ps(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
                    for (int r = 0; r < R; ++r)
                        acc[q & 1][r] = _mm256_fmadd_ps(v, _mm256_loadu_ps(pk + (size_t) r * rs + q * 8), acc[q & 1][r]);
                }
            }
            for (int r = 0; r < R; ++r) {
                // e in {0,1,4,5}: output n of the tile, e in {2,3,6,7}: output n + 8
                const __m256 a = _mm256_add_ps(acc[0][r], acc[1][r]);
                const __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
                float f[4];
                _mm_storeu_ps(f, s);
                c[(size_t) r * ldc + (nt - nt0) * 16 + n] = f[0] + f[1];
                c[(size_t) r * ldc + (nt - nt0) * 16 + n + 8] = f[2] + f[3];
            }
        }
}


// K <= 4: the 8 lanes of one octet for one weight index m in one vector. Per nt column, every tile's words are
// byte-swapped once into `col` (the stream then reads as big-endian bytes, MSB first), with the tile's last word in
// front (lane 0's first states start before bit 0) and padding behind. For fixed m, lane l's state starts at bit
// 8*K*l + (m+1)*K - 16: byte K*l + const and the same bit shift for every lane, so one pair of 16-byte loads, one
// vpshufb (which also puts each 32-bit window in value order) and two shifts give 8 states.
// The activation side P2 [kt][m][8] holds, for weight m, element l -> a[16 kt + 2 (l % 4) + (m & 1) + 8 (m >> 1 & 1)];
// lanes 0-3 of an octet o feed output 2o (+ 8 if m >= 4), lanes 4-7 output 2o + 1 (+ 8).
template <int K, int R, bool ROUND>
void gemv_tiles4(const TrellisMat & W, const float * P2, int nt0, int nt1, float * c, int ldc, uint8_t * col)
{
    const int words = 8 * K, KT = W.in / 16, NT = W.out / 16, in = W.in;
    const int stride = 32 * K + 32;   // bytes per tile in col: 4 front + 32 K + padding
    const __m256i bswap = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                           3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m256i pick;   // dword i of each half = big-endian bytes K*i .. K*i + 3
    {
        alignas(32) int8_t idx[32];
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < 4; ++i)
                for (int b = 0; b < 4; ++b) idx[16 * h + 4 * i + b] = (int8_t) (K * i + 3 - b);
        pick = _mm256_load_si256(reinterpret_cast<const __m256i *>(idx));
    }
    // per m: byte offset (from the front word) and left shift, compile-time with K
    auto mb = [](int m) { return ((m + 1) * K + 16) >> 3; };
    auto ms = [](int m) { return ((m + 1) * K + 16) & 7; };
    const __m256i mul = _mm256_set1_epi32((int) 0x83DCD12Du);
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1), h0 = _mm256_set1_epi32(1024);
    const __m256 kinv = _mm256_set1_ps(k_inv()), kbias = _mm256_set1_ps(k_bias());
    for (int nt = nt0; nt < nt1; ++nt) {
        for (int kt = 0; kt < KT; ++kt) {   // byte-swap the column once
            const uint32_t * t = W.tiles + ((size_t) kt * NT + nt) * words;
            uint8_t * d = col + (size_t) kt * stride;
            const uint32_t last = __builtin_bswap32(t[words - 1]);
            std::memcpy(d, &last, 4);
            int w = 0;
            for (; w + 8 <= words; w += 8)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + 4 + 4 * w),
                                    _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(t + w)), bswap));
            for (; w < words; ++w) {
                const uint32_t v = __builtin_bswap32(t[w]);
                std::memcpy(d + 4 + 4 * w, &v, 4);
            }
            std::memset(d + 4 + 4 * words, 0, 28);
        }
        for (int o = 0; o < 4; ++o) {   // octet: lanes 8o .. 8o + 7 -> outputs 2o, 2o + 1 (and + 8)
            __m256 acc[2][R];   // [m >= 4][row]
            for (int r = 0; r < R; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
            for (int kt = 0; kt < KT; ++kt) {
                const uint8_t * d = col + (size_t) kt * stride + 8 * K * o;   // lane 8o starts K * 8o bytes in
                const float * pk = P2 + (size_t) kt * 64;
                auto step = [&](auto mc) {
                    constexpr int m = decltype(mc)::value;
                    const uint8_t * b = d + mb(m);
                    const __m256i raw = _mm256_loadu2_m128i(reinterpret_cast<const __m128i *>(b + 4 * K),
                                                            reinterpret_cast<const __m128i *>(b));
                    __m256i st = _mm256_shuffle_epi8(raw, pick);
                    st = _mm256_srli_epi32(_mm256_slli_epi32(st, ms(m)), 16);
                    st = _mm256_mullo_epi32(st, mul);
                    const __m256i sum = _mm256_madd_epi16(_mm256_maddubs_epi16(st, ones8), ones16);
                    __m256 v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, h0)), kinv, kbias);
                    if (ROUND) v = round16_bits(v);
                    for (int r = 0; r < R; ++r)
                        acc[m >> 2][r] = _mm256_fmadd_ps(v, _mm256_loadu_ps(pk + (size_t) r * in * 4 + m * 8), acc[m >> 2][r]);
                };
                step(std::integral_constant<int, 0>{}), step(std::integral_constant<int, 1>{});
                step(std::integral_constant<int, 2>{}), step(std::integral_constant<int, 3>{});
                step(std::integral_constant<int, 4>{}), step(std::integral_constant<int, 5>{});
                step(std::integral_constant<int, 6>{}), step(std::integral_constant<int, 7>{});
            }
            for (int r = 0; r < R; ++r)
                for (int hi = 0; hi < 2; ++hi) {
                    float f[8];
                    _mm256_storeu_ps(f, acc[hi][r]);
                    float * cr = c + (size_t) r * ldc + (nt - nt0) * 16 + 8 * hi;
                    cr[2 * o] = f[0] + f[1] + f[2] + f[3];
                    cr[2 * o + 1] = f[4] + f[5] + f[6] + f[7];
                }
        }
    }
}

// The int16 kernel (K <= 4, default; TRUSS_TRELLIS_I16=0 selects gemv_tiles4). Zen 2 measured per vector op
// (TRACKER #76): vpmulld 2.3 cycles, every other integer multiply (vpmullw, vpmulhuw, vpmaddubsw, vpmaddwd) and
// every shift 1 cycle on a single pipe each, and/or/add/pshufb ~4 per cycle. gemv_tiles4 spends vpmulld +
// vpmaddubsw + vpmaddwd = ~4.3 multiply-pipe cycles per 8 weights. Here, per 16 weights (two m of one octet):
//   states: m's 32-bit window of gemv_tiles4 holds m + 1's state too (ms + K + 16 <= 32): shifted down (state m in
//           the low half of each dword) and up (state m + 1 in the high half), vpblendw: 16 states in 16-bit lanes
//   hash:   s * 0x83DCD12D mod 2^32 from 16-bit products, low half = mullo(s, 0xD12D), high half =
//           mulhi(s, 0xD12D) + mullo(s, 0x83DC) (mod 2^16; exact because s < 2^16)
//   sum:    bytes of lo by vpmaddubsw (x ones), bytes of hi by (hi & 0xff) + (hi >> 8): measured best balance of the
//           multiply pipe against the ALUs (all-ALU byte sums 3.4, both by vpmaddubsw 3.25 cycles / 8 weights)
//   dot:    one vpmaddwd of the 16 byte sums (0..1020) against int16 activations, int32 accumulators.
// = 5 multiply-pipe ops, ~17 vector ops per 16 weights: 2.8 cycles / 8 weights vs gemv_tiles4's 4.8 (Zen 2, one
// thread from cache, TRACKER #76); now bound by the total op count, not one pipe. The weight is kinv * (1024 + sum) + kbias, so
// sum_i w_i a_i = kinv * sum_i bytesum_i a_i + (1024 kinv + kbias) * sum_i a_i: the codebook's affine part is one
// per-row term. Activations: int16 with one scale per row (amax / 32767) from the fp32 Hadamard output, which is
// finer than the GPU's fp16 activations (error |a|max / 65534 per element vs 2^-12 relative); weights are the exact
// fp32 codebook values, as in gemv_tiles4. Overflow: per pair 2 * 1020 * 32767 = 6.7e7, 2 pairs per accumulator per
// 16-input slice, flushed to fp32 every 8 slices (1.1e9 < 2^31).
// Prepared row (trellis_prep, K <= 4 with i16): int16 [kt][v][16] with v = 0 for m in {0,1,4,5}, 1 for {2,3,6,7},
// lane 2e + j -> a[16 kt + 2 (e % 4) + j + 8 v] (m and m + 4 read the same inputs), then float scale, float sum(a).
bool use_i16()
{
    static const bool v = !getenv("TRUSS_TRELLIS_I16") || atoi(getenv("TRUSS_TRELLIS_I16"));
    return v;
}

template <int K, int R>
void gemv_i16(const TrellisMat & W, const float * P, int nt0, int nt1, float * c, int ldc, uint8_t * col)
{
    const int words = 8 * K, KT = W.in / 16, NT = W.out / 16, in = W.in, rs = trellis_prep_floats(in, 1);
    const int stride = 32 * K + 32;
    const __m256i bswap = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                           3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m256i pick;
    {
        alignas(32) int8_t idx[32];
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < 4; ++i)
                for (int b = 0; b < 4; ++b) idx[16 * h + 4 * i + b] = (int8_t) (K * i + 3 - b);
        pick = _mm256_load_si256(reinterpret_cast<const __m256i *>(idx));
    }
    auto mb = [](int m) { return ((m + 1) * K + 16) >> 3; };
    auto ms = [](int m) { return ((m + 1) * K + 16) & 7; };
    const __m256i mlo = _mm256_set1_epi16((short) 0xD12D), mhi = _mm256_set1_epi16((short) 0x83DC);
    const __m256i lo8 = _mm256_set1_epi16(0x00ff), ones8 = _mm256_set1_epi8(1);
    const float kinv = k_inv(), cb = std::fma(1024.f, k_inv(), k_bias());
    float scale[R], suma[R];
    for (int r = 0; r < R; ++r) {
        const float * tail = P + (size_t) r * rs + in;   // after the 2 * in int16
        scale[r] = tail[0], suma[r] = tail[1];
    }
    // Loop order: blocks of KB slices outer, columns inner. Tiles are k-slice major, so the tiles of one slice for the
    // call's columns are contiguous: each block reads KB contiguous runs instead of walking each column at a stride
    // of NT tiles (3.8 KB for gate/up), which the hardware prefetcher follows poorly (one thread: 1.0 ms / expert
    // from DRAM vs 0.55 from cache, TRACKER #76). fp32 accumulators per column stay in L1 between blocks.
    constexpr int KB = 8;   // slices per block: also the int32 flush interval (overflow bound above)
    const int ncol = nt1 - nt0;
    alignas(32) float facc[512 / 16][4][2][R][8];   // [column][octet][m >= 4][row][lane]; trellis_gemv passes <= 512 columns
    std::memset(facc, 0, sizeof(float) * 8 * R * 8 * ncol);
    auto prefetch_block = [&](int k0) {
        for (int kt = k0; kt < std::min(KT, k0 + KB); ++kt) {
            const char * t = reinterpret_cast<const char *>(W.tiles + ((size_t) kt * NT + nt0) * words);
            const int n = ncol * words * 4;
            for (int b = 0; b < n; b += 64) _mm_prefetch(t + b, _MM_HINT_T0);
        }
    };
    prefetch_block(0);
    for (int k0 = 0; k0 < KT; k0 += KB) {
        const int k1 = std::min(KT, k0 + KB);
        if (k1 < KT) prefetch_block(k1);
        for (int nt = nt0; nt < nt1; ++nt) {
            for (int kt = k0; kt < k1; ++kt) {   // byte-swap the block's tiles of this column (as gemv_tiles4)
                const uint32_t * t = W.tiles + ((size_t) kt * NT + nt) * words;
                uint8_t * d = col + (size_t) (kt - k0) * stride;
                const uint32_t last = __builtin_bswap32(t[words - 1]);
                std::memcpy(d, &last, 4);
                int w = 0;
                for (; w + 8 <= words; w += 8)
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + 4 + 4 * w),
                                        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(t + w)), bswap));
                for (; w < words; ++w) {
                    const uint32_t v = __builtin_bswap32(t[w]);
                    std::memcpy(d + 4 + 4 * w, &v, 4);
                }
                // no padding written: bytes past the stream only reach window bits that the shifts discard
            }
            for (int o = 0; o < 4; ++o) {
                __m256i iacc[2][R];
                for (int r = 0; r < R; ++r) iacc[0][r] = iacc[1][r] = _mm256_setzero_si256();
                for (int kt = k0; kt < k1; ++kt) {
                    const uint8_t * d = col + (size_t) (kt - k0) * stride + 8 * K * o;
                    const int16_t * q = reinterpret_cast<const int16_t *>(P) + (size_t) kt * 32;
                    auto win = [&](auto mc) {
                        constexpr int m = decltype(mc)::value;
                        const uint8_t * b = d + mb(m);
                        const __m256i raw = _mm256_loadu2_m128i(reinterpret_cast<const __m128i *>(b + 4 * K),
                                                                reinterpret_cast<const __m128i *>(b));
                        return _mm256_shuffle_epi8(raw, pick);
                    };
                    auto pair = [&](auto mc) {   // m, m + 1 (m even): both states lie in m's 32-bit window
                        constexpr int m = decltype(mc)::value;
                        const __m256i w = win(mc);
                        const __m256i s = _mm256_blend_epi16(_mm256_srli_epi32(w, 16 - ms(m)), _mm256_slli_epi32(w, ms(m) + K), 0xAA);
                        const __m256i lo = _mm256_mullo_epi16(s, mlo);
                        const __m256i hi = _mm256_add_epi16(_mm256_mulhi_epu16(s, mlo), _mm256_mullo_epi16(s, mhi));
                        // bytes of lo by vpmaddubsw, bytes of hi by and/shift: balances the multiply pipe and the ALUs
                        const __m256i bs = _mm256_add_epi16(_mm256_maddubs_epi16(lo, ones8),
                                                            _mm256_add_epi16(_mm256_and_si256(hi, lo8), _mm256_srli_epi16(hi, 8)));
                        constexpr int v = (m >> 1) & 1, g = m >> 2;
                        for (int r = 0; r < R; ++r) {
                            const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + (size_t) r * rs * 2 + v * 16));
                            iacc[g][r] = _mm256_add_epi32(iacc[g][r], _mm256_madd_epi16(bs, av));
                        }
                    };
                    pair(std::integral_constant<int, 0>{}), pair(std::integral_constant<int, 2>{});
                    pair(std::integral_constant<int, 4>{}), pair(std::integral_constant<int, 6>{});
                }
                for (int r = 0; r < R; ++r)
                    for (int g = 0; g < 2; ++g) {
                        float * f = facc[nt - nt0][o][g][r];
                        _mm256_store_ps(f, _mm256_add_ps(_mm256_load_ps(f), _mm256_cvtepi32_ps(iacc[g][r])));
                    }
            }
        }
    }
    for (int nt = nt0; nt < nt1; ++nt)
        for (int o = 0; o < 4; ++o)
            for (int r = 0; r < R; ++r)
                for (int g = 0; g < 2; ++g) {
                    const float * f = facc[nt - nt0][o][g][r];
                    float * cr = c + (size_t) r * ldc + (nt - nt0) * 16 + 8 * g;
                    const float ks = kinv * scale[r], bias = cb * suma[r];
                    cr[2 * o] = std::fma(ks, f[0] + f[1] + f[2] + f[3], bias);
                    cr[2 * o + 1] = std::fma(ks, f[4] + f[5] + f[6] + f[7], bias);
                }
}

// Half-integer rates K = KA + 0.5 (formats/trellis_k.h; TRACKER #118) on the int16 kernel above. Lane l's window m
// starts at stream bit 4 P l + E(m) - 16 (P = 2 KA + 1 bits per weight pair, E = k_window_end of lane 0), so within an
// octet lane i sits 4 P i bits after lane 0: a whole byte for even i, half a byte more for odd i. Per even m the
// byte picks (as `pick`, but per lane) and the left shift of each dword are tables; states m and m + 1 (KA + 1 bits
// apart) share m's 32-bit window as in gemv_i16 (shift <= 7, + KA + 1 + 16 <= 27 bits). Activations, sums and the
// int32 flush are gemv_i16's.
// Per KA, the lane tables of gemv_i16f, at compile time: as runtime arrays they were 12 ymm values plus 8 offsets
// next to the 4 constants and the 2 R accumulators, more than AVX2's 16 registers, and the inner loop spilled
// (~1500 stack accesses in gemv_i16f<2, 1> vs ~110 in gemv_i16<3, 1>; one thread 0.68 ms/expert at K2.5 vs 0.55 at
// K3, TRACKER #118). As constexpr data the picks and shifts are memory operands of vpshufb / vpsrlvd / vpsllvd and the
// offsets are immediates.
template <int KA>
struct FracTab {
    alignas(32) int8_t pick[4][32] = {};
    alignas(32) int32_t shr0[4][8] = {}, shl1[4][8] = {};
    int off[4][2] = {};
    constexpr FracTab()
    {
        constexpr int PB = 2 * KA + 1, code = 10 * KA + 5;
        for (int mi = 0; mi < 4; ++mi) {
            const int m = 2 * mi;
            for (int h = 0; h < 2; ++h) {
                off[mi][h] = (4 * PB * (4 * h) + formats::k_window_end(code, m) + 16) >> 3;
                for (int i = 0; i < 4; ++i) {
                    const int bit = 4 * PB * (4 * h + i) + formats::k_window_end(code, m) + 16;   // in col (4 front bytes)
                    const int b = (bit >> 3) - off[mi][h];
                    for (int k = 0; k < 4; ++k) pick[mi][16 * h + 4 * i + k] = (int8_t) (b + 3 - k);
                    shr0[mi][4 * h + i] = 16 - (bit & 7);           // state m: (w << s) >> 16 = w >> (16 - s)
                    shl1[mi][4 * h + i] = (bit & 7) + KA + 1;       // state m + 1 into the high half
                }
            }
        }
    }
};
template <int KA>
inline constexpr FracTab<KA> frac_tab{};

// Half-integer rates K = KA + 0.5 (formats/trellis_k.h; TRACKER #118) on the int16 kernel above. Lane l's window m
// starts at stream bit 4 P l + E(m) - 16 (P = 2 KA + 1 bits per weight pair, E = k_window_end of lane 0), so within an
// octet lane i sits 4 P i bits after lane 0: a whole byte for even i, half a byte more for odd i. Per even m the
// byte picks (as `pick`, but per lane) and the left shift of each dword are tables (FracTab); states m and m + 1
// (KA + 1 bits apart) share m's 32-bit window as in gemv_i16 (shift <= 7, + KA + 1 + 16 <= 27 bits). Activations,
// sums and the int32 flush are gemv_i16's.
template <int KA, int R>
void gemv_i16f(const TrellisMat & W, const float * P, int nt0, int nt1, float * c, int ldc, uint8_t * col)
{
    constexpr int PB = 2 * KA + 1, code = 10 * KA + 5, words = formats::k_tile_u16(code) / 2, stride = 4 * words + 32;
    constexpr const FracTab<KA> & T = frac_tab<KA>;
    const int KT = W.in / 16, NT = W.out / 16, in = W.in, rs = trellis_prep_floats(in, 1);
    const __m256i bswap = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                           3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    const __m256i mlo = _mm256_set1_epi16((short) 0xD12D), mhi = _mm256_set1_epi16((short) 0x83DC);
    const __m256i lo8 = _mm256_set1_epi16(0x00ff), ones8 = _mm256_set1_epi8(1);
    const float kinv = k_inv(), cb = std::fma(1024.f, k_inv(), k_bias());
    float scale[R], suma[R];
    for (int r = 0; r < R; ++r) {
        const float * tail = P + (size_t) r * rs + in;
        scale[r] = tail[0], suma[r] = tail[1];
    }
    constexpr int KB = 8;
    const int ncol = nt1 - nt0;
    alignas(32) float facc[512 / 16][4][2][R][8];
    std::memset(facc, 0, sizeof(float) * 8 * R * 8 * ncol);
    auto prefetch_block = [&](int k0) {
        for (int kt = k0; kt < std::min(KT, k0 + KB); ++kt) {
            const char * t = reinterpret_cast<const char *>(W.tiles + ((size_t) kt * NT + nt0) * words);
            const int n = ncol * words * 4;
            for (int b = 0; b < n; b += 64) _mm_prefetch(t + b, _MM_HINT_T0);
        }
    };
    prefetch_block(0);
    for (int k0 = 0; k0 < KT; k0 += KB) {
        const int k1 = std::min(KT, k0 + KB);
        if (k1 < KT) prefetch_block(k1);
        for (int nt = nt0; nt < nt1; ++nt) {
            for (int kt = k0; kt < k1; ++kt) {
                const uint32_t * t = W.tiles + ((size_t) kt * NT + nt) * words;
                uint8_t * d = col + (size_t) (kt - k0) * stride;
                const uint32_t last = __builtin_bswap32(t[words - 1]);
                std::memcpy(d, &last, 4);
                // words = 20 / 28 (K2.5 / K3.5): the tail is one more vector overlapping the last, not 4 scalar swaps
                static_assert(words >= 8);
                for (int w = 0; w < words; w += 8) {
                    const int at = std::min(w, words - 8);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + 4 + 4 * at),
                                        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(t + at)), bswap));
                }
            }
            for (int o = 0; o < 4; ++o) {
                __m256i iacc[2][R];
                for (int r = 0; r < R; ++r) iacc[0][r] = iacc[1][r] = _mm256_setzero_si256();
                for (int kt = k0; kt < k1; ++kt) {
                    const uint8_t * d = col + (size_t) (kt - k0) * stride + 4 * PB * o;   // lane 8o: 32 P o bits in
                    const int16_t * q = reinterpret_cast<const int16_t *>(P) + (size_t) kt * 32;
                    auto pair = [&](auto mc) {
                        constexpr int mi = decltype(mc)::value, m = 2 * mi;
                        constexpr int o0 = T.off[mi][0], o1 = T.off[mi][1];
                        const __m256i raw = _mm256_loadu2_m128i(reinterpret_cast<const __m128i *>(d + o1),
                                                                reinterpret_cast<const __m128i *>(d + o0));
                        const __m256i w = _mm256_shuffle_epi8(raw, _mm256_load_si256(reinterpret_cast<const __m256i *>(T.pick[mi])));
                        const __m256i s = _mm256_blend_epi16(
                            _mm256_srlv_epi32(w, _mm256_load_si256(reinterpret_cast<const __m256i *>(T.shr0[mi]))),
                            _mm256_sllv_epi32(w, _mm256_load_si256(reinterpret_cast<const __m256i *>(T.shl1[mi]))), 0xAA);
                        const __m256i lo = _mm256_mullo_epi16(s, mlo);
                        const __m256i hi = _mm256_add_epi16(_mm256_mulhi_epu16(s, mlo), _mm256_mullo_epi16(s, mhi));
                        const __m256i bs = _mm256_add_epi16(_mm256_maddubs_epi16(lo, ones8),
                                                            _mm256_add_epi16(_mm256_and_si256(hi, lo8), _mm256_srli_epi16(hi, 8)));
                        constexpr int v = (m >> 1) & 1, g = m >> 2;
                        for (int r = 0; r < R; ++r) {
                            const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + (size_t) r * rs * 2 + v * 16));
                            iacc[g][r] = _mm256_add_epi32(iacc[g][r], _mm256_madd_epi16(bs, av));
                        }
                    };
                    pair(std::integral_constant<int, 0>{}), pair(std::integral_constant<int, 1>{});
                    pair(std::integral_constant<int, 2>{}), pair(std::integral_constant<int, 3>{});
                }
                for (int r = 0; r < R; ++r)
                    for (int g = 0; g < 2; ++g) {
                        float * f = facc[nt - nt0][o][g][r];
                        _mm256_store_ps(f, _mm256_add_ps(_mm256_load_ps(f), _mm256_cvtepi32_ps(iacc[g][r])));
                    }
            }
        }
    }
    for (int nt = nt0; nt < nt1; ++nt)
        for (int o = 0; o < 4; ++o)
            for (int r = 0; r < R; ++r)
                for (int g = 0; g < 2; ++g) {
                    const float * f = facc[nt - nt0][o][g][r];
                    float * cr = c + (size_t) r * ldc + (nt - nt0) * 16 + 8 * g;
                    const float ks = kinv * scale[r], bias = cb * suma[r];
                    cr[2 * o] = std::fma(ks, f[0] + f[1] + f[2] + f[3], bias);
                    cr[2 * o + 1] = std::fma(ks, f[4] + f[5] + f[6] + f[7], bias);
                }
}

template <int KA>
void dispatch_i16f(const TrellisMat & W, const float * P, int R, int nt0, int nt1, float * c, uint8_t * col)
{
    const int rs = trellis_prep_floats(W.in, 1);
    for (int r0 = 0; r0 < R; r0 += 4) {
        const float * Pr = P + (size_t) r0 * rs;
        float * cr = c + (size_t) r0 * 512;
        switch (std::min(4, R - r0)) {
        case 1: gemv_i16f<KA, 1>(W, Pr, nt0, nt1, cr, 512, col); break;
        case 2: gemv_i16f<KA, 2>(W, Pr, nt0, nt1, cr, 512, col); break;
        case 3: gemv_i16f<KA, 3>(W, Pr, nt0, nt1, cr, 512, col); break;
        default: gemv_i16f<KA, 4>(W, Pr, nt0, nt1, cr, 512, col); break;
        }
    }
}

template <int K>
void dispatch_i16(const TrellisMat & W, const float * P, int R, int nt0, int nt1, float * c, uint8_t * col)
{
    const int rs = trellis_prep_floats(W.in, 1);
    for (int r0 = 0; r0 < R; r0 += 4) {   // at most 4 rows per pass (registers)
        const float * Pr = P + (size_t) r0 * rs;
        float * cr = c + (size_t) r0 * 512;
        switch (std::min(4, R - r0)) {
        case 1: gemv_i16<K, 1>(W, Pr, nt0, nt1, cr, 512, col); break;
        case 2: gemv_i16<K, 2>(W, Pr, nt0, nt1, cr, 512, col); break;
        case 3: gemv_i16<K, 3>(W, Pr, nt0, nt1, cr, 512, col); break;
        default: gemv_i16<K, 4>(W, Pr, nt0, nt1, cr, 512, col); break;
        }
    }
}
void dispatch_i16k(const TrellisMat & W, const float * P, int R, int nt0, int nt1, float * c, uint8_t * col)
{
    switch (W.K) {
    case 1: dispatch_i16<1>(W, P, R, nt0, nt1, c, col); break;
    case 2: dispatch_i16<2>(W, P, R, nt0, nt1, c, col); break;
    case 3: dispatch_i16<3>(W, P, R, nt0, nt1, c, col); break;
    default: dispatch_i16<4>(W, P, R, nt0, nt1, c, col); break;
    }
}

template <int K, bool ROUND>
void dispatch4r(const TrellisMat & W, const float * P, int R, int nt0, int nt1, float * c, uint8_t * col)
{
    switch (R) {
    case 1: gemv_tiles4<K, 1, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    case 2: gemv_tiles4<K, 2, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    case 3: gemv_tiles4<K, 3, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    case 4: gemv_tiles4<K, 4, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    case 5: gemv_tiles4<K, 5, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    case 6: gemv_tiles4<K, 6, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    case 7: gemv_tiles4<K, 7, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    default: gemv_tiles4<K, 8, ROUND>(W, P, nt0, nt1, c, 512, col); break;
    }
}
template <bool ROUND>
void dispatch4k(const TrellisMat & W, const float * P, int R, int nt0, int nt1, float * c, uint8_t * col)
{
    switch (W.K) {
    case 1: dispatch4r<1, ROUND>(W, P, R, nt0, nt1, c, col); break;
    case 2: dispatch4r<2, ROUND>(W, P, R, nt0, nt1, c, col); break;
    case 3: dispatch4r<3, ROUND>(W, P, R, nt0, nt1, c, col); break;
    default: dispatch4r<4, ROUND>(W, P, R, nt0, nt1, c, col); break;
    }
}
void dispatch4(const TrellisMat & W, const float * P, int R, bool round, int nt0, int nt1, float * c, uint8_t * col)
{
    if (round) dispatch4k<true>(W, P, R, nt0, nt1, c, col);
    else dispatch4k<false>(W, P, R, nt0, nt1, c, col);
}
template <bool ROUND>
void dispatch6r(const TrellisMat & W, const float * P, int R, int nt0, int nt1, float * c)
{
    switch (R) {
    case 1: gemv_tiles<1, ROUND>(W, P, nt0, nt1, c, 512); break;
    case 2: gemv_tiles<2, ROUND>(W, P, nt0, nt1, c, 512); break;
    case 3: gemv_tiles<3, ROUND>(W, P, nt0, nt1, c, 512); break;
    case 4: gemv_tiles<4, ROUND>(W, P, nt0, nt1, c, 512); break;
    case 5: gemv_tiles<5, ROUND>(W, P, nt0, nt1, c, 512); break;
    case 6: gemv_tiles<6, ROUND>(W, P, nt0, nt1, c, 512); break;
    case 7: gemv_tiles<7, ROUND>(W, P, nt0, nt1, c, 512); break;
    default: gemv_tiles<8, ROUND>(W, P, nt0, nt1, c, 512); break;
    }
}
void dispatch6(const TrellisMat & W, const float * P, int R, bool round, int nt0, int nt1, float * c)
{
    if (round) dispatch6r<true>(W, P, R, nt0, nt1, c);
    else dispatch6r<false>(W, P, R, nt0, nt1, c);
}

}  // namespace

float trellis_weight_ref(const TrellisMat & W, int o, int i, bool r16)
{
    const int K = W.K;   // rate code (formats/trellis_k.h)
    const uint32_t * tile = W.tiles + ((size_t) (i / 16) * (W.out / 16) + o / 16) * (formats::k_tile_u16(K) / 2);
    const int n = o % 16, k = i % 16;
    const int lane = (n % 8) * 4 + (k % 8) / 2;
    const int j = lane * 8 + (k & 1) + (k >= 8 ? 2 : 0) + (n >= 8 ? 4 : 0);
    const int bits = formats::k_tile_bits(K), end = formats::k_window_end(K, j);
    uint32_t state = 0;
    for (int b = 0; b < 16; ++b) {
        const int p = ((end - 16 + b) % bits + bits) % bits;
        state = (state << 1) | ((tile[p / 32] >> (31 - p % 32)) & 1u);
    }
    const uint32_t x = state * 0x83DCD12Du;
    const uint32_t sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    const float v = std::fma((float) (1024 + sum), k_inv(), k_bias());
    return r16 ? round16(v) : v;
}

// P layout: K <= 4 [R][in/16][8 m][8 lanes] = 4 * in floats per row; K 5-6 [R][in/16][4 lane quads][8] = 2 * in
void trellis_prep(const TrellisMat & W, const float * x, int ldx, int R, float * P)
{
    const int in = W.in, KT = in / 16;
    if (in > MAX_IN || in % 128) throw std::invalid_argument("trellis_prep: in must be a multiple of 128, <= 4096");
    alignas(32) float a[MAX_IN];
    for (int r = 0; r < R; ++r) {
        const float * xr = x + (size_t) r * ldx;
        for (int i = 0; i < in; i += 8)
            _mm256_store_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(xr + i),
                                                 _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(W.suh + i)))));
        for (int b = 0; b < in; b += 128) hadamard128(a + b);
        if ((W.K <= 4 && use_i16() && !round_weights()) || formats::k_frac(W.K)) {   // gemv_i16(f): int16 [kt][v][16], then scale, sum
            // lanes 2e + j of block (kt, v) read a[16 kt + 8 v + (2 (e % 4) + j)]: the 8 inputs a[16 kt + 8 v ..] twice
            const __m256 sign = _mm256_set1_ps(-0.f);
            __m256 mx = _mm256_setzero_ps();
            for (int i = 0; i < in; i += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, _mm256_load_ps(a + i)));
            alignas(32) float m8[8];
            _mm256_store_ps(m8, mx);
            const float amax = *std::max_element(m8, m8 + 8);
            const float s = amax / 32767.f, inv = amax > 0 ? 32767.f / amax : 0.f;
            float * pr = P + (size_t) r * in * 4;
            int16_t * q = reinterpret_cast<int16_t *>(pr);
            const __m256 vi = _mm256_set1_ps(inv);
            __m256i acc = _mm256_setzero_si256();
            for (int i = 0; i < in; i += 8) {   // block (kt, v) = i / 8; cvtps rounds to nearest even, as lrint
                const __m256i x = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(a + i), vi));
                acc = _mm256_add_epi32(acc, x);
                const __m128i h = _mm_packs_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(q + 2 * i), _mm256_set_m128i(h, h));
            }
            alignas(32) int32_t s8[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(s8), acc);
            long long sum = 0;
            for (int k = 0; k < 8; ++k) sum += s8[k];
            pr[in] = s, pr[in + 1] = (float) ((double) sum * s);
            continue;
        }
        for (int i = 0; i < in; ++i) a[i] = round16(a[i]);   // the GPU feeds fp16 activations to its MMA
        if (W.K <= 4) {   // gemv_tiles4: [kt][m][lane of the octet]
            float * pr = P + (size_t) r * in * 4;
            for (int kt = 0; kt < KT; ++kt)
                for (int m = 0; m < 8; ++m)
                    for (int l = 0; l < 8; ++l)
                        pr[(kt * 8 + m) * 8 + l] = a[kt * 16 + 2 * (l % 4) + (m & 1) + 8 * ((m >> 1) & 1)];
        } else {          // gemv_tiles: [kt][lane quad][element] (2 * in of the row's 4 * in)
            float * pr = P + (size_t) r * in * 4;
            for (int kt = 0; kt < KT; ++kt)
                for (int q = 0; q < 4; ++q)
                    for (int e = 0; e < 8; ++e) {
                        const int m = ORDER[e];
                        pr[(kt * 4 + q) * 8 + e] = a[kt * 16 + 2 * q + (m & 1) + 8 * ((m >> 1) & 1)];
                    }
        }
    }
}

TrellisMat trellis_inputs(const TrellisMat & W, int i0, int i1)
{
    if (i0 < 0 || i1 > W.in || i0 >= i1 || i0 % 128 || i1 % 128) throw std::invalid_argument("trellis_inputs: range");
    TrellisMat s = W;
    s.tiles = W.tiles + (size_t) (i0 / 16) * (W.out / 16) * (formats::k_tile_u16(W.K) / 2);   // k-slice major: slices i0/16 .. contiguous
    s.in = i1 - i0;
    s.suh = W.suh + i0;
    return s;
}

void trellis_gemv_raw(const TrellisMat & W, const float * P, int R, int c0, int c1, float * c, int ldc)
{
    if (!((W.K >= 1 && W.K <= 6) || W.K == 25 || W.K == 35))
        throw std::invalid_argument("trellis_gemv: K must be 1 .. 6, 25 or 35 (K2.5 / K3.5)");
    if (c0 % 128 || c1 % 128 || c1 > W.out || c0 >= c1 || R < 1 || R > MAXR)
        throw std::invalid_argument("trellis_gemv: column range / rows");
    alignas(32) float buf[MAXR * 512];
    alignas(32) uint8_t col[256 * (32 * 4 + 32)];   // gemv_tiles4's byte-swapped column (in <= 4096)
    // The GPU rounds each decoded weight to fp16 (hfma2). Rounding here costs ~40% of the loop (Zen 2's vector
    // integer pipes are the limit) and changes a weight by at most half an fp16 ulp (~5e-4 relative), far below the
    // codebook's own quantization error, so it is off unless TRUSS_TRELLIS_ROUND=1.
    const bool round = round_weights();
    for (int b0 = c0; b0 < c1; b0 += 512) {   // at most 512 columns per pass of the kernels' buffer
        const int b1 = std::min(c1, b0 + 512), nb = b1 - b0;
        if (W.K == 25) dispatch_i16f<2>(W, P, R, b0 / 16, b1 / 16, buf, col);   // half-integer rates: int16 only
        else if (W.K == 35) dispatch_i16f<3>(W, P, R, b0 / 16, b1 / 16, buf, col);
        else if (W.K <= 4 && use_i16() && !round) dispatch_i16k(W, P, R, b0 / 16, b1 / 16, buf, col);
        else if (W.K <= 4) dispatch4(W, P, R, round, b0 / 16, b1 / 16, buf, col);
        else dispatch6(W, P, R, round, b0 / 16, b1 / 16, buf);
        for (int r = 0; r < R; ++r) std::memcpy(c + (size_t) r * ldc + (b0 - c0), buf + (size_t) r * 512, sizeof(float) * nb);
    }
}

void trellis_out(const TrellisMat & W, float * c, int R, int ldc, int c0, int c1, float * y, int ld_y)
{
    if (c0 % 128 || c1 % 128 || c1 > W.out || c0 >= c1) throw std::invalid_argument("trellis_out: column range");
    const int n = c1 - c0;
    for (int r = 0; r < R; ++r) {
        float * cr = c + (size_t) r * ldc;
        float * yr = y + (size_t) r * ld_y;
        for (int b = 0; b < n; b += 128) hadamard128(cr + b);
        for (int j = 0; j < n; j += 8)
            _mm256_storeu_ps(yr + j, _mm256_mul_ps(_mm256_loadu_ps(cr + j),
                                                   _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(W.svh + c0 + j)))));
    }
}

void trellis_gemv(const TrellisMat & W, const float * P, int R, int c0, int c1, float * y, int ld_y)
{
    if (c1 - c0 > MAX_COLS) throw std::invalid_argument("trellis_gemv: column range / rows");
    alignas(32) float c[MAXR * 512];
    for (int b0 = c0; b0 < c1; b0 += 512) {
        const int b1 = std::min(c1, b0 + 512);
        trellis_gemv_raw(W, P, R, b0, b1, c, 512);
        trellis_out(W, c, R, 512, b0, b1, y + (b0 - c0), ld_y);
    }
}

}  // namespace truss::cpu
