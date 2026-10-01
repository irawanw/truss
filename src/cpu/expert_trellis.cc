#include "cpu/expert_trellis.h"

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
    const int K = W.K, words = 8 * K, KT = W.in / 16, NT = W.out / 16, in = W.in;
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
                        acc[q & 1][r] = _mm256_fmadd_ps(v, _mm256_loadu_ps(pk + (size_t) r * in * 2 + q * 8), acc[q & 1][r]);
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
    const int K = W.K;
    const uint32_t * tile = W.tiles + ((size_t) (i / 16) * (W.out / 16) + o / 16) * 8 * K;
    const int n = o % 16, k = i % 16;
    const int lane = (n % 8) * 4 + (k % 8) / 2;
    const int j = lane * 8 + (k & 1) + (k >= 8 ? 2 : 0) + (n >= 8 ? 4 : 0);
    const int bits = 256 * K;
    uint32_t state = 0;
    for (int b = 0; b < 16; ++b) {
        const int p = (((j + 1) * K - 16 + b) % bits + bits) % bits;
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
        for (int i = 0; i < in; ++i) a[i] = round16(a[i]);   // the GPU feeds fp16 activations to its MMA
        if (W.K <= 4) {   // gemv_tiles4: [kt][m][lane of the octet]
            float * pr = P + (size_t) r * in * 4;
            for (int kt = 0; kt < KT; ++kt)
                for (int m = 0; m < 8; ++m)
                    for (int l = 0; l < 8; ++l)
                        pr[(kt * 8 + m) * 8 + l] = a[kt * 16 + 2 * (l % 4) + (m & 1) + 8 * ((m >> 1) & 1)];
        } else {          // gemv_tiles: [kt][lane quad][element]
            float * pr = P + (size_t) r * in * 2;
            for (int kt = 0; kt < KT; ++kt)
                for (int q = 0; q < 4; ++q)
                    for (int e = 0; e < 8; ++e) {
                        const int m = ORDER[e];
                        pr[(kt * 4 + q) * 8 + e] = a[kt * 16 + 2 * q + (m & 1) + 8 * ((m >> 1) & 1)];
                    }
        }
    }
}

void trellis_gemv(const TrellisMat & W, const float * P, int R, int c0, int c1, float * y, int ld_y)
{
    if (W.K < 1 || W.K > 6) throw std::invalid_argument("trellis_gemv: K must be 1 .. 6");
    if (c0 % 128 || c1 % 128 || c1 > W.out || c1 - c0 > MAX_COLS || R < 1 || R > MAXR)
        throw std::invalid_argument("trellis_gemv: column range / rows");
    alignas(32) float c[MAXR * 512];
    alignas(32) uint8_t col[256 * (32 * 4 + 32)];   // gemv_tiles4's byte-swapped column (in <= 4096)
    // The GPU rounds each decoded weight to fp16 (hfma2). Rounding here costs ~40% of the loop (Zen 2's vector
    // integer pipes are the limit) and changes a weight by at most half an fp16 ulp (~5e-4 relative), far below the
    // codebook's own quantization error, so it is off unless TRUSS_TRELLIS_ROUND=1.
    static const bool round = getenv("TRUSS_TRELLIS_ROUND") && atoi(getenv("TRUSS_TRELLIS_ROUND"));
    // at most 512 columns per pass of the buffer
    for (int b0 = c0; b0 < c1; b0 += 512) {
        const int b1 = std::min(c1, b0 + 512), nb = b1 - b0;
        if (W.K <= 4) dispatch4(W, P, R, round, b0 / 16, b1 / 16, c, col);
        else dispatch6(W, P, R, round, b0 / 16, b1 / 16, c);
        for (int r = 0; r < R; ++r) {
            float * cr = c + (size_t) r * 512;
            for (int b = 0; b < nb; b += 128) hadamard128(cr + b);
            float * yr = y + (size_t) r * ld_y + (b0 - c0);
            for (int j = 0; j < nb; j += 8)
                _mm256_storeu_ps(yr + j, _mm256_mul_ps(_mm256_loadu_ps(cr + j),
                                                       _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(W.svh + b0 + j)))));
        }
    }
}

}  // namespace truss::cpu
