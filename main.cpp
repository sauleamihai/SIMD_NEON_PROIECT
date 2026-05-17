#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>
#include <sstream>
#include <cassert>
#include <functional>
#include <arm_neon.h>

// ----------------------------------------------------------------------------
//  TYPES & CONSTANTS
// ----------------------------------------------------------------------------
typedef std::complex<float> Complex;
static constexpr float PI  = 3.14159265358979323846f;
static constexpr float TAU = 6.28318530717958647692f;

// ----------------------------------------------------------------------------
//  WAV I/O
// ----------------------------------------------------------------------------
#pragma pack(push, 1)
struct WavHeader {
    char     riff[4];
    uint32_t wav_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bit_depth;
    char     data[4];
    uint32_t data_bytes;
};
#pragma pack(pop)

std::vector<float> wav_read(const std::string& path, WavHeader& hdr) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> out;
    if (!f) return out;

    f.read(reinterpret_cast<char*>(&hdr), 36);
    char id[5] = {};
    uint32_t sz;
    while (f.read(id, 4) && f.read(reinterpret_cast<char*>(&sz), 4)) {
        if (!strncmp(id, "data", 4)) {
            memcpy(hdr.data, "data", 4);
            hdr.data_bytes = sz;
            break;
        }
        f.seekg(sz, std::ios::cur);
    }
    if (!hdr.data_bytes) return out;

    int n = hdr.data_bytes / (hdr.bit_depth / 8);
    out.reserve(n);
    int16_t s;
    for (int i = 0; i < n; ++i) {
        if (!f.read(reinterpret_cast<char*>(&s), 2)) break;
        out.push_back(s / 32768.0f);
    }
    hdr.data_bytes = out.size() * 2;
    return out;
}

void wav_write(const std::string& path, const std::vector<float>& sig, WavHeader hdr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    hdr.data_bytes = sig.size() * 2;
    hdr.wav_size   = hdr.data_bytes + 36;
    memcpy(hdr.data, "data", 4);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    for (float v : sig) {
        v = std::max(-1.0f, std::min(1.0f, v));
        int16_t s = static_cast<int16_t>(v * 32767.0f);
        f.write(reinterpret_cast<const char*>(&s), 2);
    }
    std::cout << "[WAV] Saved: " << path << "\n";
}

// ----------------------------------------------------------------------------
//  NEON HELPERS
// ----------------------------------------------------------------------------

// Horizontal sum of float32x4_t
inline float neon_hsum(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s  = vadd_f32(lo, hi);
    return vget_lane_f32(vpadd_f32(s, s), 0);
}

// Deinterleave stereo [L R L R ...] -> two mono arrays, NEON accelerated
void neon_deinterleave(const float* lr, float* L, float* R, int frames) {
    int i = 0;
    for (; i <= frames - 4; i += 4) {
        float32x4x2_t v = vld2q_f32(lr + i * 2);
        vst1q_f32(L + i, v.val[0]);
        vst1q_f32(R + i, v.val[1]);
    }
    for (; i < frames; ++i) { L[i] = lr[i*2]; R[i] = lr[i*2+1]; }
}

// Interleave back, NEON accelerated
void neon_interleave(const float* L, const float* R, float* lr, int frames) {
    int i = 0;
    for (; i <= frames - 4; i += 4) {
        float32x4x2_t v;
        v.val[0] = vld1q_f32(L + i);
        v.val[1] = vld1q_f32(R + i);
        vst2q_f32(lr + i * 2, v);
    }
    for (; i < frames; ++i) { lr[i*2] = L[i]; lr[i*2+1] = R[i]; }
}

// NEON vector multiply-add: out += a * b
inline void neon_vmla(float* __restrict__ out,
                      const float* __restrict__ a,
                      const float* __restrict__ b,
                      int n) {
    int i = 0;
    for (; i <= n - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(a+i   ), b0 = vld1q_f32(b+i   );
        float32x4_t a1 = vld1q_f32(a+i+ 4), b1 = vld1q_f32(b+i+ 4);
        float32x4_t a2 = vld1q_f32(a+i+ 8), b2 = vld1q_f32(b+i+ 8);
        float32x4_t a3 = vld1q_f32(a+i+12), b3 = vld1q_f32(b+i+12);
        float32x4_t o0 = vld1q_f32(out+i   );
        float32x4_t o1 = vld1q_f32(out+i+ 4);
        float32x4_t o2 = vld1q_f32(out+i+ 8);
        float32x4_t o3 = vld1q_f32(out+i+12);
        vst1q_f32(out+i,    vmlaq_f32(o0, a0, b0));
        vst1q_f32(out+i+ 4, vmlaq_f32(o1, a1, b1));
        vst1q_f32(out+i+ 8, vmlaq_f32(o2, a2, b2));
        vst1q_f32(out+i+12, vmlaq_f32(o3, a3, b3));
    }
    for (; i <= n - 4; i += 4) {
        float32x4_t av = vld1q_f32(a+i), bv = vld1q_f32(b+i);
        vst1q_f32(out+i, vmlaq_f32(vld1q_f32(out+i), av, bv));
    }
    for (; i < n; ++i) out[i] += a[i] * b[i];
}

// ----------------------------------------------------------------------------
//  FFT - Cooley-Tukey iterative, precomputed twiddle table
// ----------------------------------------------------------------------------
struct FFTPlan {
    int                  N;
    std::vector<Complex> twiddle;   // twiddle[k] = exp(-2pi i k/N)
    std::vector<int>     rev;       // bit-reversal permutation

    explicit FFTPlan(int n) : N(n), twiddle(n/2), rev(n) {
        assert((n & (n-1)) == 0 && "FFT size must be power of 2");
        for (int k = 0; k < n/2; ++k)
            twiddle[k] = {cosf(TAU * k / n), -sinf(TAU * k / n)};
        int logn = __builtin_ctz(n);
        for (int i = 0; i < n; ++i) {
            rev[i] = 0;
            for (int b = 0; b < logn; ++b)
                rev[i] |= ((i >> b) & 1) << (logn - 1 - b);
        }
    }

    // In-place forward FFT
    void forward(std::vector<Complex>& a) const {
        // Bit-reversal
        for (int i = 0; i < N; ++i)
            if (i < rev[i]) std::swap(a[i], a[rev[i]]);

        for (int len = 2; len <= N; len <<= 1) {
            int half   = len >> 1;
            int stride = N / len;

            for (int i = 0; i < N; i += len) {
                // ---- NEON butterfly: 4 butterflies per iteration ----
                //   a[u] = U + W*V,  a[v] = U - W*V
                //   re(W*V) = re(W)*re(V) - im(W)*im(V)
                //   im(W*V) = re(W)*im(V) + im(W)*re(V)
                // Layout: [re0 im0 re1 im1 ...]; vld2q splits re/im.

                int j = 0;
                for (; j <= half - 4; j += 4) {
                    float32x4x2_t U = vld2q_f32(
                        reinterpret_cast<const float*>(&a[i + j]));
                    float32x4_t U_re = U.val[0];
                    float32x4_t U_im = U.val[1];

                    float32x4x2_t V_raw = vld2q_f32(
                        reinterpret_cast<const float*>(&a[i + j + half]));
                    float32x4_t V_re = V_raw.val[0];
                    float32x4_t V_im = V_raw.val[1];

                    // [FIX 2] The needed twiddles are `stride` apart in
                    // memory, so a single vector load cannot gather them.
                    // (The original code's vld2q twiddle load was dead
                    // code and has been removed.) Scalar gather is correct:
                    float w_re[4], w_im[4];
                    for (int k = 0; k < 4; ++k) {
                        w_re[k] = twiddle[(j + k) * stride].real();
                        w_im[k] = twiddle[(j + k) * stride].imag();
                    }
                    float32x4_t W_re = vld1q_f32(w_re);
                    float32x4_t W_im = vld1q_f32(w_im);

                    // WV = W * V  (complex)
                    float32x4_t WV_re = vmulq_f32(W_re, V_re);
                    WV_re = vmlsq_f32(WV_re, W_im, V_im);   // -= W_im*V_im
                    float32x4_t WV_im = vmulq_f32(W_re, V_im);
                    WV_im = vmlaq_f32(WV_im, W_im, V_re);   // += W_im*V_re

                    float32x4_t S_re = vaddq_f32(U_re, WV_re);
                    float32x4_t S_im = vaddq_f32(U_im, WV_im);
                    float32x4_t D_re = vsubq_f32(U_re, WV_re);
                    float32x4_t D_im = vsubq_f32(U_im, WV_im);

                    float32x4x2_t S_out = {S_re, S_im};
                    float32x4x2_t D_out = {D_re, D_im};
                    vst2q_f32(reinterpret_cast<float*>(&a[i + j]),        S_out);
                    vst2q_f32(reinterpret_cast<float*>(&a[i + j + half]), D_out);
                }

                // Scalar tail for remaining butterflies
                for (; j < half; ++j) {
                    Complex w = twiddle[j * stride];
                    Complex u = a[i + j];
                    Complex v = a[i + j + half] * w;
                    a[i + j]        = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }

    // Inverse FFT (conjugate trick)
    void inverse(std::vector<Complex>& a) const {
        for (auto& c : a) c = std::conj(c);
        forward(a);
        for (auto& c : a) c = std::conj(c) * (1.0f / N);
    }
};

// ----------------------------------------------------------------------------
//  FIR FILTER - NEON optimized direct-form convolution
//  (Retained for completeness; not used by the streaming EQ path.)
// ----------------------------------------------------------------------------
struct FIRFilter {
    std::vector<float> coef;
    std::vector<float> delay;   // circular delay line
    int ptr = 0;

    FIRFilter() = default;
    explicit FIRFilter(std::vector<float> c) : coef(std::move(c)), delay(coef.size(), 0.0f) {}

    float process_sample(float x) {
        delay[ptr] = x;
        ptr = (ptr + 1) % (int)delay.size();
        int M = coef.size();
        float acc = 0.0f;
        float32x4_t sum = vdupq_n_f32(0.0f);
        int idx = ptr;
        int j = 0;
        for (; j <= M - 4; j += 4) {
            float tmp[4];
            for (int k = 0; k < 4; ++k) tmp[k] = delay[(idx + k) % M];
            float32x4_t dv = vld1q_f32(tmp);
            float32x4_t cv = vld1q_f32(&coef[j]);
            sum = vmlaq_f32(sum, dv, cv);
            idx = (idx + 4) % M;
        }
        acc = neon_hsum(sum);
        for (; j < M; ++j) { acc += delay[idx] * coef[j]; idx = (idx+1)%M; }
        return acc;
    }

    void process_block(const std::vector<float>& in, std::vector<float>& out) {
        int n = in.size();
        int M = coef.size();
        out.resize(n, 0.0f);

        for (int i = 0; i < std::min(M - 1, n); ++i) {
            float acc = 0.0f;
            for (int j = 0; j <= i; ++j) acc += in[i - j] * coef[j];
            out[i] = acc;
        }

        for (int i = M - 1; i <= n - 4; i += 4) {
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (int j = 0; j <= M - 4; j += 4) {
                float32x4_t cv = vld1q_f32(&coef[j]);
                acc0 = vmlaq_f32(acc0, vld1q_f32(&in[i  -j]), cv);
                acc1 = vmlaq_f32(acc1, vld1q_f32(&in[i+1-j]), cv);
                acc2 = vmlaq_f32(acc2, vld1q_f32(&in[i+2-j]), cv);
                acc3 = vmlaq_f32(acc3, vld1q_f32(&in[i+3-j]), cv);
            }
            for (int j = M & ~3; j < M; ++j) {
                float c = coef[j];
                out[i  ] += in[i  -j] * c;
                out[i+1] += in[i+1-j] * c;
                out[i+2] += in[i+2-j] * c;
                out[i+3] += in[i+3-j] * c;
            }
            float tmp[4] = { neon_hsum(acc0), neon_hsum(acc1),
                             neon_hsum(acc2), neon_hsum(acc3) };
            float32x4_t partial = vld1q_f32(&out[i]);
            vst1q_f32(&out[i], vaddq_f32(partial, vld1q_f32(tmp)));
        }

        for (int i = std::max(M-1, (n/4)*4); i < n; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < M && i-j >= 0; ++j) acc += in[i-j]*coef[j];
            out[i] = acc;
        }
    }
};

// ----------------------------------------------------------------------------
//  BIQUAD - Transposed Direct Form II (TDF2)
//
//  TDF2 recurrence (2 state vars vs 4 for DF1):
//    y[n]  = b0*x[n] + s1
//    s1'   = b1*x[n] - a1*y[n] + s2
//    s2'   = b2*x[n] - a2*y[n]
//
//  Why scalar (not NEON) for the biquad on Cortex-A53:
//  A53 is in-order; every NEON<->scalar transfer (vget_lane_f32) costs a
//  3-4 cycle cross-domain stall that the in-order pipe cannot hide. A
//  series biquad cascade needs each band's scalar output as the next
//  band's input, forcing such a transfer per sample -> NEON loses. The
//  correct A53 strategy is tight scalar TDF2 with locals in registers and
//  -ffast-math so the compiler can overlap FP-multiply latencies. NEON is
//  used everywhere data-parallel (int16<->float, FFT, magnitude, stereo).
// ----------------------------------------------------------------------------

struct BiquadCoeffs {
    float b0, b1, b2, a1, a2;
    void precompute() {}   // no-op; kept for API compatibility
};

// TDF2 state: only 2 floats (vs 4 for DF1)
struct BiquadState {
    float s1 = 0.f;
    float s2 = 0.f;
};

inline float biquad_tick(const BiquadCoeffs& c, BiquadState& s, float x) {
    float y   = c.b0 * x + s.s1;
    float ns1 = c.b1 * x - c.a1 * y + s.s2;
    float ns2 = c.b2 * x - c.a2 * y;
    s.s1 = ns1;
    s.s2 = ns2;
    return y;
}

// ---- RBJ Audio EQ Cookbook coefficient calculators ----

BiquadCoeffs biquad_peaking(float fc, float gain_db, float Q, float fs) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / fs;
    float cw = cosf(w0), sw = sinf(w0);
    float alpha = sw / (2.0f * Q);
    BiquadCoeffs c;
    float inv = 1.0f / (1.0f + alpha / A);
    c.b0 = (1.0f + alpha * A) * inv;
    c.b1 = (-2.0f * cw)       * inv;
    c.b2 = (1.0f - alpha * A) * inv;
    c.a1 = c.b1;
    c.a2 = (1.0f - alpha / A) * inv;
    c.precompute();
    return c;
}

BiquadCoeffs biquad_low_shelf(float fc, float gain_db, float fs) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / fs;
    float cw = cosf(w0), sw = sinf(w0);
    float alpha = sw / 2.0f * sqrtf((A + 1.0f/A) * (1.0f/0.707f - 1.0f) + 2.0f);
    float inv = 1.0f / ((A+1) + (A-1)*cw + 2*sqrtf(A)*alpha);
    BiquadCoeffs c;
    c.b0 =      A * ((A+1) - (A-1)*cw + 2*sqrtf(A)*alpha) * inv;
    c.b1 =  2.0f*A * ((A-1) - (A+1)*cw)                   * inv;
    c.b2 =      A * ((A+1) - (A-1)*cw - 2*sqrtf(A)*alpha) * inv;
    c.a1 = -2.0f   * ((A-1) + (A+1)*cw)                   * inv;
    c.a2 =           ((A+1) + (A-1)*cw - 2*sqrtf(A)*alpha) * inv;
    c.precompute();
    return c;
}

BiquadCoeffs biquad_high_shelf(float fc, float gain_db, float fs) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / fs;
    float cw = cosf(w0), sw = sinf(w0);
    float alpha = sw / 2.0f * sqrtf((A + 1.0f/A) * (1.0f/0.707f - 1.0f) + 2.0f);
    float inv = 1.0f / ((A+1) - (A-1)*cw + 2*sqrtf(A)*alpha);
    BiquadCoeffs c;
    c.b0 =      A * ((A+1) + (A-1)*cw + 2*sqrtf(A)*alpha) * inv;
    c.b1 = -2.0f*A * ((A-1) + (A+1)*cw)                   * inv;
    c.b2 =      A * ((A+1) + (A-1)*cw - 2*sqrtf(A)*alpha) * inv;
    c.a1 =  2.0f   * ((A-1) - (A+1)*cw)                   * inv;
    c.a2 =           ((A+1) - (A-1)*cw - 2*sqrtf(A)*alpha) * inv;
    c.precompute();
    return c;
}

// ----------------------------------------------------------------------------
//  10-BAND PARAMETRIC EQ  (correct SERIES biquad cascade)
//
//  Signal path:  in -> [low-shelf] -> [peak x8] -> [high-shelf] -> out
//  Each biquad is applied IN SERIES (the previous band's output is the
//  next band's input). 0 dB bands are exact pass-through and are skipped.
// ----------------------------------------------------------------------------

struct EQBandDef {
    const char* name;
    float       fc;       // centre / corner frequency
    float       Q;        // quality factor (bandwidth)
    enum Type { LOW_SHELF, PEAK, HIGH_SHELF } type;
    float       gain_db;
};

static EQBandDef DEFAULT_BANDS[10] = {
    {"Sub-Bass",   60.0f,   0.7f, EQBandDef::LOW_SHELF,  0.f},
    {"Bass",      170.0f,   1.0f, EQBandDef::PEAK,       0.f},
    {"Low-Mid",   350.0f,   1.0f, EQBandDef::PEAK,       0.f},
    {"Mid",       700.0f,   1.4f, EQBandDef::PEAK,       0.f},
    {"Upper-Mid", 1400.0f,  1.4f, EQBandDef::PEAK,       0.f},
    {"Presence",  3000.0f,  1.4f, EQBandDef::PEAK,       0.f},
    {"Brilliance",6000.0f,  1.4f, EQBandDef::PEAK,       0.f},
    {"Air",      10000.0f,  1.4f, EQBandDef::PEAK,       0.f},
    {"Ultra",    14000.0f,  1.4f, EQBandDef::PEAK,       0.f},
    {"High",     18000.0f,  0.7f, EQBandDef::HIGH_SHELF, 0.f},
};

class Equalizer {
public:
    static constexpr int N_BANDS = 10;

    EQBandDef bands[N_BANDS];
    float     sample_rate;

    // Precomputed coefficients (rebuilt when gains change)
    BiquadCoeffs coeffs[N_BANDS];

    // Per-channel TDF2 state: [channel][band] (2 floats each)
    BiquadState state[2][N_BANDS];

    explicit Equalizer(float fs) : sample_rate(fs) {
        memcpy(bands, DEFAULT_BANDS, sizeof(DEFAULT_BANDS));
        memset(state, 0, sizeof(state));
        rebuild_filters();
    }

    void set_gain(int b, float db) {
        bands[b].gain_db = std::max(-24.0f, std::min(24.0f, db));
    }

    BiquadCoeffs make_coeffs(int b) const {
        const auto& bd = bands[b];
        if (fabsf(bd.gain_db) < 0.01f) {
            // Identity: y = x  ->  b0=1, all else 0 (exact pass-through)
            BiquadCoeffs c{1.f, 0.f, 0.f, 0.f, 0.f};
            c.precompute();
            return c;
        }
        switch (bd.type) {
            case EQBandDef::LOW_SHELF:
                return biquad_low_shelf(bd.fc, bd.gain_db, sample_rate);
            case EQBandDef::HIGH_SHELF:
                return biquad_high_shelf(bd.fc, bd.gain_db, sample_rate);
            default:
                return biquad_peaking(bd.fc, bd.gain_db, bd.Q, sample_rate);
        }
    }

    // Recompute coefficients and clear all filter state.
    void rebuild_filters() {
        for (int b = 0; b < N_BANDS; ++b)
            coeffs[b] = make_coeffs(b);
        memset(state, 0, sizeof(state));
    }

    // ---- [FIX 1] Correct SERIES cascade ----
    //
    //  out[i] = ( ... ( in[i] -> band0 ) -> band1 ) ... -> bandK
    //
    //  Bands at 0 dB are identity (b0=1, rest 0) and are skipped, which is
    //  numerically exact and faster. Per-channel state is carried across
    //  calls (chunked streaming continuity). Supports in-place (out == in).
    //
    //  Scalar by design: a series IIR recurrence has no time-parallelism,
    //  and vectorising across the chain on the in-order Cortex-A53 forces
    //  a per-sample NEON<->scalar stall (see header note). This tight
    //  scalar TDF2 loop is the recommended A53 strategy.
    //
    //  Fast MONO path: band-major scalar series cascade -- one tight pass
    //  per active band over the whole block. Band-major is the OPTIMAL
    //  scalar form on the in-order Cortex-A53: each band's recurrence
    //  carries only s1/s2 (kept in registers across the long i-loop),
    //  while consecutive samples' input multiplies are independent, so the
    //  pipeline overlaps them. (A sample-major fused loop instead serialises
    //  K biquads per sample with no overlap and is ~2x slower -- measured.)
    //  0 dB bands are exact pass-through and skipped. In-place safe.
    void process_block_mono(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n, int ch) {
        if (out != in) memcpy(out, in, n * sizeof(float));
        for (int b = 0; b < N_BANDS; ++b) {
            if (fabsf(bands[b].gain_db) < 0.01f) continue;   // exact bypass
            const float b0 = coeffs[b].b0, b1 = coeffs[b].b1, b2 = coeffs[b].b2;
            const float a1 = coeffs[b].a1, a2 = coeffs[b].a2;
            float s1 = state[ch][b].s1, s2 = state[ch][b].s2;   // registers
            for (int i = 0; i < n; ++i) {
                float x  = out[i];
                float y  = b0 * x + s1;
                float n1 = b1 * x - a1 * y + s2;
                float n2 = b2 * x - a2 * y;
                s1 = n1; s2 = n2;
                out[i] = y;                  // feeds the next band's pass
            }
            state[ch][b].s1 = s1; state[ch][b].s2 = s2;
        }
    }

    //  Fast STEREO path: NEON 2-lane (lane0 = L, lane1 = R), band-major.
    //
    //  L and R are FULLY INDEPENDENT signals filtered by the SAME band
    //  coefficients -> both channels are processed by a single float32x2_t
    //  op. This is the legitimate ~2x NEON win for the IIR EQ (real audio
    //  is stereo) and it does NOT contradict the in-order thesis: the
    //  interleaved [L R L R ...] buffer lets vld1_f32 load a (L,R) pair in
    //  one NEON load, coeffs are scalars broadcast once per band, and the
    //  only NEON->scalar transfers are the state write-back ONCE per band
    //  (amortised over the whole block, like neon_hsum). No per-sample
    //  cross-domain stall.  `lr` = interleaved stereo, `frames` = #frames.
    void process_block_stereo(float* __restrict__ lr, int frames) {
        // ---- [PERF] Band-PAIR fusion ----
        // Band-major previously made ONE full pass over the 705 KB
        // interleaved stereo buffer PER band. That buffer >> the 512 KB L2,
        // so every band re-streamed it from RAM (~10x traffic) -> the loop
        // was memory-bound (deeper compute unrolling gave nothing).
        // Fusing TWO bands per pass loads/stores each (L,R) sample once for
        // two biquads -> HALVES the passes and the RAM traffic. The series
        // order (band k feeds band k+1) is preserved exactly.
        int act[N_BANDS], na = 0;
        for (int b = 0; b < N_BANDS; ++b)
            if (fabsf(bands[b].gain_db) >= 0.01f) act[na++] = b;
        if (na == 0) return;                  // all 0 dB -> buffer == input

        int k = 0;
        for (; k + 2 <= na; k += 2) {
            const int ba = act[k], bb = act[k + 1];
            const float32x2_t A0=vdup_n_f32(coeffs[ba].b0), A1=vdup_n_f32(coeffs[ba].b1),
                              A2=vdup_n_f32(coeffs[ba].b2), Aa1=vdup_n_f32(coeffs[ba].a1),
                              Aa2=vdup_n_f32(coeffs[ba].a2);
            const float32x2_t B0=vdup_n_f32(coeffs[bb].b0), B1=vdup_n_f32(coeffs[bb].b1),
                              B2=vdup_n_f32(coeffs[bb].b2), Bb1=vdup_n_f32(coeffs[bb].a1),
                              Bb2=vdup_n_f32(coeffs[bb].a2);
            float32x2_t sa1=vset_lane_f32(state[1][ba].s1, vdup_n_f32(state[0][ba].s1),1);
            float32x2_t sa2=vset_lane_f32(state[1][ba].s2, vdup_n_f32(state[0][ba].s2),1);
            float32x2_t sb1=vset_lane_f32(state[1][bb].s1, vdup_n_f32(state[0][bb].s1),1);
            float32x2_t sb2=vset_lane_f32(state[1][bb].s2, vdup_n_f32(state[0][bb].s2),1);

            for (int i = 0; i < frames; ++i) {
                float32x2_t x  = vld1_f32(lr + 2 * i);          // (L,R) once
                // band ba
                float32x2_t ya = vmla_f32(sa1, A0, x);          // b0*x + s1
                sa1 = vmla_f32(vmls_f32(sa2, Aa1, ya), A1, x);
                sa2 = vmls_f32(vmul_f32(A2, x), Aa2, ya);
                // band bb  (input = ya, fused -> no extra load/store)
                float32x2_t yb = vmla_f32(sb1, B0, ya);
                sb1 = vmla_f32(vmls_f32(sb2, Bb1, yb), B1, ya);
                sb2 = vmls_f32(vmul_f32(B2, ya), Bb2, yb);
                vst1_f32(lr + 2 * i, yb);                        // store once
            }
            state[0][ba].s1=vget_lane_f32(sa1,0); state[1][ba].s1=vget_lane_f32(sa1,1);
            state[0][ba].s2=vget_lane_f32(sa2,0); state[1][ba].s2=vget_lane_f32(sa2,1);
            state[0][bb].s1=vget_lane_f32(sb1,0); state[1][bb].s1=vget_lane_f32(sb1,1);
            state[0][bb].s2=vget_lane_f32(sb2,0); state[1][bb].s2=vget_lane_f32(sb2,1);
        }
        if (k < na) {                          // odd trailing band
            const int b = act[k];
            const float32x2_t b0=vdup_n_f32(coeffs[b].b0), b1=vdup_n_f32(coeffs[b].b1),
                              b2=vdup_n_f32(coeffs[b].b2), a1=vdup_n_f32(coeffs[b].a1),
                              a2=vdup_n_f32(coeffs[b].a2);
            float32x2_t s1=vset_lane_f32(state[1][b].s1, vdup_n_f32(state[0][b].s1),1);
            float32x2_t s2=vset_lane_f32(state[1][b].s2, vdup_n_f32(state[0][b].s2),1);
            for (int i = 0; i < frames; ++i) {
                float32x2_t x  = vld1_f32(lr + 2 * i);
                float32x2_t y  = vmla_f32(s1, b0, x);
                float32x2_t n1 = vmla_f32(vmls_f32(s2, a1, y), b1, x);
                float32x2_t n2 = vmls_f32(vmul_f32(b2, x), a2, y);
                s1 = n1; s2 = n2;
                vst1_f32(lr + 2 * i, y);
            }
            state[0][b].s1=vget_lane_f32(s1,0); state[1][b].s1=vget_lane_f32(s1,1);
            state[0][b].s2=vget_lane_f32(s2,0); state[1][b].s2=vget_lane_f32(s2,1);
        }
    }

    // Legacy interface
    std::vector<float> process(const std::vector<float>& in) {
        std::vector<float> out(in.size());
        process_block_mono(in.data(), out.data(), in.size(), 0);
        return out;
    }
};

// ----------------------------------------------------------------------------
//  SPECTRUM ANALYZER (NEON magnitude)
// ----------------------------------------------------------------------------
struct SpectrumResult {
    std::vector<float> magnitudes;  // N_BINS values
    float              peak_db;
    float              rms_db;
    static constexpr int N_BINS = 64;
};

// NEON magnitude: sqrt(re^2 + im^2). vsqrtq_f32 is a true IEEE hardware
// sqrt on AArch64 (sqrt(0)=0, no NaN). 2x unrolled to hide its ~17-cycle
// latency by keeping both sqrt issues in flight on the in-order pipe.
void neon_magnitude(const Complex* fft_buf, float* mag, int n) {
    int i = 0;
    for (; i <= n - 8; i += 8) {
        float32x4x2_t v0 = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i));
        float32x4x2_t v1 = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i + 4));

        float32x4_t sq0 = vmlaq_f32(vmulq_f32(v0.val[0], v0.val[0]), v0.val[1], v0.val[1]);
        float32x4_t sq1 = vmlaq_f32(vmulq_f32(v1.val[0], v1.val[0]), v1.val[1], v1.val[1]);

        float32x4_t m0 = vsqrtq_f32(sq0);
        float32x4_t m1 = vsqrtq_f32(sq1);

        vst1q_f32(mag + i,     m0);
        vst1q_f32(mag + i + 4, m1);
    }
    for (; i <= n - 4; i += 4) {
        float32x4x2_t v = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i));
        float32x4_t sq = vmlaq_f32(vmulq_f32(v.val[0], v.val[0]), v.val[1], v.val[1]);
        vst1q_f32(mag + i, vsqrtq_f32(sq));
    }
    for (; i < n; ++i)
        mag[i] = std::abs(fft_buf[i]);
}

SpectrumResult analyze_spectrum(const std::vector<float>& signal,
                                 const FFTPlan& plan,
                                 float fs) {
    SpectrumResult res;
    res.magnitudes.resize(SpectrumResult::N_BINS, 0.0f);

    int N = plan.N;
    int start = std::max(0, (int)signal.size() / 2 - N / 2);
    std::vector<Complex> buf(N, {0.f, 0.f});
    int lim = std::min(N, (int)signal.size() - start);

    // Hann window
    for (int i = 0; i < lim; ++i) {
        float w = 0.5f - 0.5f * cosf(TAU * i / (lim - 1));
        buf[i] = {signal[start + i] * w, 0.f};
    }

    plan.forward(buf);

    int half = N / 2;
    std::vector<float> mag(half);
    neon_magnitude(buf.data(), mag.data(), half);

    float log_lo = log10f(20.0f);
    float log_hi = log10f(fs / 2.0f);
    float log_range = log_hi - log_lo;

    for (int i = 0; i < half; ++i) {
        float freq = (float)i * fs / N;
        if (freq < 20.0f) continue;
        float logf_val = log10f(freq);
        int bin = (int)((logf_val - log_lo) / log_range * SpectrumResult::N_BINS);
        bin = std::max(0, std::min(SpectrumResult::N_BINS - 1, bin));
        res.magnitudes[bin] = std::max(res.magnitudes[bin], mag[i]);
    }

    float peak = 1e-10f;
    for (float& m : res.magnitudes) {
        m = (m < 1e-10f) ? -80.0f : 20.0f * log10f(m);
        if (m > peak) peak = m;
    }
    res.peak_db = peak;

    float32x4_t sq_sum = vdupq_n_f32(0.f);
    int i = 0;
    for (; i <= (int)signal.size() - 4; i += 4) {
        float32x4_t v = vld1q_f32(&signal[i]);
        sq_sum = vmlaq_f32(sq_sum, v, v);
    }
    float rms = neon_hsum(sq_sum);
    for (; i < (int)signal.size(); ++i) rms += signal[i] * signal[i];
    rms = sqrtf(rms / signal.size());
    res.rms_db = (rms < 1e-10f) ? -80.0f : 20.0f * log10f(rms);

    return res;
}

// ----------------------------------------------------------------------------
//  JSON OUTPUT (for Flask GUI)
// ----------------------------------------------------------------------------
std::string spectrum_to_json(const SpectrumResult& r,
                              const Equalizer& eq) {
    std::ostringstream ss;
    ss << "{\"spectrum\":[";
    for (int i = 0; i < SpectrumResult::N_BINS; ++i) {
        if (i) ss << ",";
        ss << r.magnitudes[i];
    }
    ss << "],\"peak_db\":" << r.peak_db;
    ss << ",\"rms_db\":"   << r.rms_db;
    ss << ",\"bands\":[";
    for (int b = 0; b < Equalizer::N_BANDS; ++b) {
        if (b) ss << ",";
        ss << "{\"name\":\"" << eq.bands[b].name << "\""
           << ",\"freq_lo\":"  << eq.bands[b].fc * 0.5f
           << ",\"freq_hi\":"  << eq.bands[b].fc * 2.0f
           << ",\"gain_db\":"  << eq.bands[b].gain_db << "}";
    }
    ss << "]}";
    return ss.str();
}

// ----------------------------------------------------------------------------
//  BENCHMARK  (writes benchmark.json for Flask GUI)
//  JSON keys kept as "fir"/"fft"/"magnitude"/"stereo"/"total" for GUI
//  compatibility. The "fir" slot actually measures the biquad EQ.
// ----------------------------------------------------------------------------
struct BenchResult {
    double fir_naive_ms;   double fir_neon_ms;   double fir_speedup;
    int    fir_samples;    int    fir_taps;
    double fft_naive_ms;   double fft_neon_ms;   double fft_speedup;
    int    fft_size;       int    fft_runs;
    double mag_scalar_ms;  double mag_neon_ms;   double mag_speedup;
    int    mag_size;
    double stereo_scalar_ms; double stereo_neon_ms; double stereo_speedup;
    int    stereo_frames;
    double total_naive_ms; double total_neon_ms; double total_speedup;
    float  sample_rate;
};

BenchResult benchmark(const std::vector<float>& signal, float fs) {
    std::cout << "========================================\n";
    std::cout << "          BENCHMARK RESULTS             \n";
    std::cout << "========================================\n";

    BenchResult res{};
    res.sample_rate = fs;

    int N = std::min((int)signal.size(), (int)(fs * 2));
    std::vector<float> test(signal.begin(), signal.begin() + N);

    // -- 1. Biquad EQ (realistic STEREO workload) --
    //    Real audio is stereo. The IIR cascade cannot be SIMD-ed along
    //    time (serial recurrence) -- but L and R are INDEPENDENT signals
    //    sharing the SAME coefficients, so the honest NEON win is doing
    //    both channels per op:
    //      naive     = best scalar form (band-major) on L, then on R
    //      optimized = NEON 2-lane (L||R), one float32x2_t op per pair
    //    Both are the SAME correct series cascade -> outputs must match.
    Equalizer eq_simple(fs), eq_opt(fs);
    for (int b = 0; b < Equalizer::N_BANDS; ++b) {
        float g = (b % 3 == 0) ? 6.f : -3.f;
        eq_simple.set_gain(b, g);
        eq_opt.set_gain(b, g);
    }
    eq_simple.rebuild_filters();
    eq_opt.rebuild_filters();

    res.fir_samples = N;
    res.fir_taps    = Equalizer::N_BANDS;

    // Stereo test: L and R decorrelated so it is a true 2-channel signal.
    std::vector<float> Ln(N), Rn(N);          // naive: separate mono buffers
    std::vector<float> LRo(2 * N);            // optimized: interleaved L R L R
    for (int i = 0; i < N; ++i) {
        float l = test[i];
        float r = test[(i + N / 2) % N];
        Ln[i] = l; Rn[i] = r;
        LRo[2*i] = l; LRo[2*i + 1] = r;
    }

    // Naive: best scalar form, one channel at a time (L then R).
    auto t0 = std::chrono::high_resolution_clock::now();
    eq_simple.process_block_mono(Ln.data(), Ln.data(), N, 0);
    eq_simple.process_block_mono(Rn.data(), Rn.data(), N, 1);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Optimized: NEON 2-lane, both channels per op.
    eq_opt.process_block_stereo(LRo.data(), N);
    auto t2 = std::chrono::high_resolution_clock::now();

    res.fir_naive_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.fir_neon_ms  = std::chrono::duration<double,std::milli>(t2-t1).count();
    res.fir_speedup  = res.fir_naive_ms / std::max(res.fir_neon_ms, 0.001);

    // Correctness: scalar L/R vs NEON-interleaved must match.
    double max_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, (double)fabsf(Ln[i] - LRo[2*i]));
        max_diff = std::max(max_diff, (double)fabsf(Rn[i] - LRo[2*i + 1]));
    }
    std::cout << "Biquad EQ scalar L+R:     " << res.fir_naive_ms << " ms\n";
    std::cout << "Biquad EQ NEON 2-lane:    " << res.fir_neon_ms  << " ms\n";
    std::cout << "Speedup:                  " << res.fir_speedup  << "x\n";
    std::cout << "Max |scalar-NEON|:        " << max_diff
              << (max_diff < 1e-2 ? "  (OK same filter)" : "  (MISMATCH!)") << "\n";
    assert(max_diff < 1e-2 && "stereo scalar vs NEON 2-lane must match");

    double audio_duration_ms = ((double)N / fs) * 1000.0;   // per channel
    double rt_factor = (2.0 * audio_duration_ms) / std::max(res.fir_neon_ms, 0.001);
    std::cout << "Real-time factor:         " << rt_factor << "x  ("
              << (rt_factor >= 1.0 ? "REALTIME OK" : "TOO SLOW") << ")\n\n";

    // -- 2. FFT: naive recursive vs iterative NEON --
    //    [FIX 3] Both sides do identical (re-init + transform) work over
    //    the same run count, so the comparison is apples-to-apples.
    constexpr int FFT_N    = 4096;
    constexpr int FFT_RUNS = 50;
    res.fft_size = FFT_N;
    res.fft_runs = FFT_RUNS;

    std::function<void(std::vector<Complex>&)> fft_naive_rec =
        [&](std::vector<Complex>& a) {
            int n = a.size();
            if (n <= 1) return;
            std::vector<Complex> even(n/2), odd(n/2);
            for (int i = 0; i < n/2; i++) { even[i]=a[2*i]; odd[i]=a[2*i+1]; }
            fft_naive_rec(even); fft_naive_rec(odd);
            float ang = -2.0f*PI/n;
            Complex w(1,0), wn(cosf(ang),sinf(ang));
            for (int i = 0; i < n/2; i++) {
                a[i]     = even[i] + w*odd[i];
                a[i+n/2] = even[i] - w*odd[i];
                w *= wn;
            }
        };

    FFTPlan plan_bench(FFT_N);
    std::vector<Complex> buf(FFT_N), buf2(FFT_N);

    // Naive: (re-init + recursive FFT) x FFT_RUNS, timed as a whole.
    auto tfft_n0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < FFT_RUNS; ++r) {
        for (int i = 0; i < FFT_N; ++i) buf[i] = {test[i % N], 0.f};
        fft_naive_rec(buf);
    }
    auto tfft_n1 = std::chrono::high_resolution_clock::now();

    // NEON: identical structure -> fair comparison.
    auto tfft0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < FFT_RUNS; ++r) {
        for (int i = 0; i < FFT_N; ++i) buf2[i] = {test[i % N], 0.f};
        plan_bench.forward(buf2);
    }
    auto tfft1 = std::chrono::high_resolution_clock::now();

    res.fft_naive_ms = std::chrono::duration<double,std::milli>(tfft_n1-tfft_n0).count() / FFT_RUNS;
    res.fft_neon_ms  = std::chrono::duration<double,std::milli>(tfft1-tfft0).count() / FFT_RUNS;
    res.fft_speedup  = res.fft_naive_ms / std::max(res.fft_neon_ms, 0.001);

    std::cout << "FFT naive recursive (avg of " << FFT_RUNS << "): "
              << res.fft_naive_ms << " ms\n";
    std::cout << "FFT iterative NEON  (avg of " << FFT_RUNS << "): "
              << res.fft_neon_ms  << " ms\n";
    std::cout << "FFT speedup: " << res.fft_speedup << "x\n\n";

    // -- 3. Magnitude: scalar sqrt vs NEON vsqrtq --
    constexpr int MAG_N = 65536;
    res.mag_size = MAG_N;
    std::vector<Complex> mag_buf(MAG_N);
    std::vector<float>   mag_out(MAG_N);
    for (int i = 0; i < MAG_N; ++i) mag_buf[i] = {(float)i*0.001f, (float)i*0.0007f};

    auto ts0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MAG_N; ++i) mag_out[i] = std::abs(mag_buf[i]);
    auto ts1 = std::chrono::high_resolution_clock::now();

    auto tn0 = std::chrono::high_resolution_clock::now();
    neon_magnitude(mag_buf.data(), mag_out.data(), MAG_N);
    auto tn1 = std::chrono::high_resolution_clock::now();

    res.mag_scalar_ms = std::chrono::duration<double,std::milli>(ts1-ts0).count();
    res.mag_neon_ms   = std::chrono::duration<double,std::milli>(tn1-tn0).count();
    res.mag_speedup   = res.mag_scalar_ms / std::max(res.mag_neon_ms, 0.0001);

    std::cout << "Magnitude scalar: " << res.mag_scalar_ms << " ms (" << MAG_N << " bins)\n";
    std::cout << "Magnitude NEON:   " << res.mag_neon_ms   << " ms\n";
    std::cout << "Magnitude speedup: " << res.mag_speedup  << "x\n\n";

    // -- 4. Stereo interleave: scalar vs NEON vld2/vst2 --
    constexpr int ST_FRAMES = 88200; // 2s stereo @ 44.1k
    res.stereo_frames = ST_FRAMES;
    std::vector<float> L(ST_FRAMES), R(ST_FRAMES), LR(ST_FRAMES*2), LR2(ST_FRAMES*2);
    for (int i = 0; i < ST_FRAMES; ++i) { L[i]=test[i%N]; R[i]=test[(i+N/2)%N]; }

    auto tst0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ST_FRAMES; ++i) { LR[i*2]=L[i]; LR[i*2+1]=R[i]; }
    for (int i = 0; i < ST_FRAMES; ++i) { L[i]=LR[i*2]; R[i]=LR[i*2+1]; }
    auto tst1 = std::chrono::high_resolution_clock::now();

    auto tst2 = std::chrono::high_resolution_clock::now();
    neon_interleave(L.data(), R.data(), LR2.data(), ST_FRAMES);
    neon_deinterleave(LR2.data(), L.data(), R.data(), ST_FRAMES);
    auto tst3 = std::chrono::high_resolution_clock::now();

    res.stereo_scalar_ms = std::chrono::duration<double,std::milli>(tst1-tst0).count();
    res.stereo_neon_ms   = std::chrono::duration<double,std::milli>(tst3-tst2).count();
    res.stereo_speedup   = res.stereo_scalar_ms / std::max(res.stereo_neon_ms, 0.0001);

    std::cout << "Stereo I/O scalar: " << res.stereo_scalar_ms << " ms (" << ST_FRAMES << " frames)\n";
    std::cout << "Stereo I/O NEON:   " << res.stereo_neon_ms   << " ms\n";
    std::cout << "Stereo speedup: "    << res.stereo_speedup   << "x\n\n";

    // -- 5. Overall totals --
    res.total_naive_ms = res.fir_naive_ms + res.fft_naive_ms + res.mag_scalar_ms + res.stereo_scalar_ms;
    res.total_neon_ms  = res.fir_neon_ms  + res.fft_neon_ms  + res.mag_neon_ms   + res.stereo_neon_ms;
    res.total_speedup  = res.total_naive_ms / std::max(res.total_neon_ms, 0.001);

    std::cout << "----------------------------------------\n";
    std::cout << "TOTAL naive: " << res.total_naive_ms << " ms\n";
    std::cout << "TOTAL NEON:  " << res.total_neon_ms  << " ms\n";
    std::cout << "OVERALL SPEEDUP: " << res.total_speedup << "x\n";
    std::cout << "----------------------------------------\n";

    // -- 6. Write benchmark.json (keys unchanged for Flask GUI) --
    std::ostringstream js;
    js << std::fixed;
    js.precision(3);
    js << "{"
       << "\"fir\":{"
       <<   "\"naive_ms\":"  << res.fir_naive_ms
       <<  ",\"neon_ms\":"   << res.fir_neon_ms
       <<  ",\"speedup\":"   << res.fir_speedup
       <<  ",\"samples\":"   << res.fir_samples
       <<  ",\"taps\":"      << res.fir_taps
       << "},"
       << "\"fft\":{"
       <<   "\"naive_ms\":"  << res.fft_naive_ms
       <<  ",\"neon_ms\":"   << res.fft_neon_ms
       <<  ",\"speedup\":"   << res.fft_speedup
       <<  ",\"size\":"      << res.fft_size
       <<  ",\"runs\":"      << res.fft_runs
       << "},"
       << "\"magnitude\":{"
       <<   "\"naive_ms\":"  << res.mag_scalar_ms
       <<  ",\"neon_ms\":"   << res.mag_neon_ms
       <<  ",\"speedup\":"   << res.mag_speedup
       <<  ",\"size\":"      << res.mag_size
       << "},"
       << "\"stereo\":{"
       <<   "\"naive_ms\":"  << res.stereo_scalar_ms
       <<  ",\"neon_ms\":"   << res.stereo_neon_ms
       <<  ",\"speedup\":"   << res.stereo_speedup
       <<  ",\"frames\":"    << res.stereo_frames
       << "},"
       << "\"total\":{"
       <<   "\"naive_ms\":"  << res.total_naive_ms
       <<  ",\"neon_ms\":"   << res.total_neon_ms
       <<  ",\"speedup\":"   << res.total_speedup
       << "},"
       << "\"sample_rate\":" << (int)res.sample_rate
       << "}";

    std::ofstream bfile("benchmark.json");
    bfile << js.str();
    std::cout << "[JSON] Saved benchmark.json\n";

    return res;
}

// ----------------------------------------------------------------------------
//  CHUNKED STREAMING PROCESSOR
//  Biquad IIR is stateful -> state carries across chunk boundaries
//  automatically (no overlap-add needed). Read CHUNK_FRAMES at a time,
//  process in-place, write out.
// ----------------------------------------------------------------------------
bool process_streaming(const std::string& in_path,
                       const std::string& out_path,
                       Equalizer& eq,
                       const WavHeader& hdr) {
    static constexpr int CHUNK_FRAMES = 65536;
    const int CH = hdr.channels;
    const int bytes_per_sample = hdr.bit_depth / 8;
    const int total_frames = hdr.data_bytes / (bytes_per_sample * CH);

    std::ifstream fin(in_path, std::ios::binary);
    if (!fin) { std::cerr << "[ERR] Cannot open input\n"; return false; }
    fin.seekg(sizeof(WavHeader), std::ios::beg);

    std::ofstream fout(out_path, std::ios::binary);
    if (!fout) { std::cerr << "[ERR] Cannot open output\n"; return false; }

    WavHeader out_hdr = hdr;
    out_hdr.data_bytes = total_frames * CH * bytes_per_sample;
    out_hdr.wav_size   = out_hdr.data_bytes + 36;
    fout.write(reinterpret_cast<const char*>(&out_hdr), sizeof(out_hdr));

    std::cout << "[STREAM] " << total_frames << " frames, "
              << CH << "ch, chunk=" << CHUNK_FRAMES << "\n";

    std::vector<int16_t> raw(CHUNK_FRAMES * CH);
    std::vector<float>   chL(CHUNK_FRAMES);          // mono path
    std::vector<float>   chLR(CHUNK_FRAMES * 2);     // stereo path (interleaved)
    std::vector<int16_t> out_raw(CHUNK_FRAMES * CH);

    const float32x4_t vSCALE_IN  = vdupq_n_f32(1.0f / 32768.0f);
    const float32x4_t vSCALE_OUT = vdupq_n_f32(32767.0f);
    const float32x4_t vCLAMP_HI  = vdupq_n_f32( 1.0f);
    const float32x4_t vCLAMP_LO  = vdupq_n_f32(-1.0f);

    int frames_done = 0;
    while (frames_done < total_frames) {
        int frames_now = std::min(CHUNK_FRAMES, total_frames - frames_done);

        fin.read(reinterpret_cast<char*>(raw.data()),
                 frames_now * CH * bytes_per_sample);

        if (CH == 1) {
            // -- MONO int16 -> float (NEON, 8 samples/iter) --
            int i = 0;
            for (; i <= frames_now - 8; i += 8) {
                int16x8_t s16 = vld1q_s16(raw.data() + i);
                int32x4_t lo32 = vmovl_s16(vget_low_s16(s16));
                int32x4_t hi32 = vmovl_s16(vget_high_s16(s16));
                float32x4_t flo = vmulq_f32(vcvtq_f32_s32(lo32), vSCALE_IN);
                float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(hi32), vSCALE_IN);
                vst1q_f32(chL.data() + i,     flo);
                vst1q_f32(chL.data() + i + 4, fhi);
            }
            for (; i < frames_now; ++i)
                chL[i] = raw[i] * (1.0f / 32768.0f);

            eq.process_block_mono(chL.data(), chL.data(), frames_now, 0);

            // -- MONO float -> int16 (NEON) --
            i = 0;
            for (; i <= frames_now - 8; i += 8) {
                float32x4_t f0 = vld1q_f32(chL.data() + i);
                float32x4_t f1 = vld1q_f32(chL.data() + i + 4);
                f0 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, f0)), vSCALE_OUT);
                f1 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, f1)), vSCALE_OUT);
                int16x4_t s0 = vmovn_s32(vcvtq_s32_f32(f0));
                int16x4_t s1 = vmovn_s32(vcvtq_s32_f32(f1));
                vst1q_s16(out_raw.data() + i, vcombine_s16(s0, s1));
            }
            for (; i < frames_now; ++i) {
                float v = std::max(-1.f, std::min(1.f, chL[i]));
                out_raw[i] = static_cast<int16_t>(v * 32767.f);
            }

        } else {
            // -- STEREO: keep INTERLEAVED [L R L R ...] (no deinterleave) --
            // Interleaving is just memory order, so int16->float conversion
            // runs on the flat 2*frames stream; the 2-lane EQ then loads
            // (L,R) pairs directly with vld1_f32 (no shuffle, no scalar
            // transfer). This is faster than the old deinterleave + two
            // mono passes AND processes both channels per NEON op.
            const int total = frames_now * 2;          // interleaved samples

            int i = 0;
            for (; i <= total - 8; i += 8) {
                int16x8_t s16  = vld1q_s16(raw.data() + i);
                int32x4_t lo32 = vmovl_s16(vget_low_s16(s16));
                int32x4_t hi32 = vmovl_s16(vget_high_s16(s16));
                vst1q_f32(chLR.data() + i,     vmulq_f32(vcvtq_f32_s32(lo32), vSCALE_IN));
                vst1q_f32(chLR.data() + i + 4, vmulq_f32(vcvtq_f32_s32(hi32), vSCALE_IN));
            }
            for (; i < total; ++i)
                chLR[i] = raw[i] * (1.0f / 32768.0f);

            // NEON 2-lane stereo EQ (L||R); per-channel state preserved.
            eq.process_block_stereo(chLR.data(), frames_now);

            i = 0;
            for (; i <= total - 8; i += 8) {
                float32x4_t f0 = vld1q_f32(chLR.data() + i);
                float32x4_t f1 = vld1q_f32(chLR.data() + i + 4);
                f0 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, f0)), vSCALE_OUT);
                f1 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, f1)), vSCALE_OUT);
                int16x4_t s0 = vmovn_s32(vcvtq_s32_f32(f0));
                int16x4_t s1 = vmovn_s32(vcvtq_s32_f32(f1));
                vst1q_s16(out_raw.data() + i, vcombine_s16(s0, s1));
            }
            for (; i < total; ++i) {
                float v = std::max(-1.f, std::min(1.f, chLR[i]));
                out_raw[i] = static_cast<int16_t>(v * 32767.f);
            }
        }

        fout.write(reinterpret_cast<const char*>(out_raw.data()),
                   frames_now * CH * bytes_per_sample);

        frames_done += frames_now;
        std::cout << "[STREAM] " << frames_done << "/" << total_frames
                  << " (" << (frames_done * 100 / total_frames) << "%)\n";
        std::cout.flush();
    }

    std::cout << "[STREAM] Done.\n";
    return true;
}

// ----------------------------------------------------------------------------
//  MAIN
// ----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::cout << "==========================================\n"
              << "  DSP Audio EQ - ARM NEON (Pi Zero 2W)     \n"
              << "  CORRECTED BUILD (series biquad cascade)   \n"
              << "==========================================\n\n";

    std::string in_path  = (argc > 1) ? argv[1] : "test_audio.wav";
    std::string out_path = (argc > 2) ? argv[2] : "audio_eq.wav";

    float gains[10] = {};
    for (int i = 3; i < argc && i - 3 < 10; ++i)
        gains[i - 3] = std::stof(argv[i]);

    // Read header only (not the whole file)
    WavHeader hdr;
    {
        std::ifstream f(in_path, std::ios::binary);
        if (!f) { std::cerr << "[ERR] Cannot open: " << in_path << "\n"; return 1; }
        f.read(reinterpret_cast<char*>(&hdr), 36);
        char id[5] = {}; uint32_t sz;
        while (f.read(id, 4) && f.read(reinterpret_cast<char*>(&sz), 4)) {
            if (!strncmp(id, "data", 4)) {
                memcpy(hdr.data, "data", 4);
                hdr.data_bytes = sz;
                break;
            }
            f.seekg(sz, std::ios::cur);
        }
    }

    if (!hdr.data_bytes) {
        std::cerr << "[ERR] No data chunk found in WAV\n"; return 1;
    }

    float fs = hdr.sample_rate;
    int total_frames = hdr.data_bytes / (hdr.bit_depth/8) / hdr.channels;

    std::cout << "[INFO] " << in_path << "\n"
              << "       " << fs << " Hz, "
              << hdr.channels << "ch, "
              << hdr.bit_depth << "-bit, "
              << total_frames << " frames ("
              << total_frames / fs << "s)\n"
              << "       File size: "
              << hdr.data_bytes / 1024 / 1024 << " MB\n\n";

    FFTPlan plan(4096);
    Equalizer eq(fs);
    for (int b = 0; b < 10; ++b) eq.set_gain(b, gains[b]);
    eq.rebuild_filters();   // ensure coeffs reflect the CLI gains

    // Stream-process (no full file in RAM)
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!process_streaming(in_path, out_path, eq, hdr)) return 1;
    auto t_end = std::chrono::high_resolution_clock::now();
    double proc_ms = std::chrono::duration<double,std::milli>(t_end-t_start).count();
    std::cout << "[PERF] Processing time: " << proc_ms/1000.0 << "s\n";

    // Spectrum: read only a small window for analysis
    auto read_window = [&](const std::string& path, int start_frame) {
        std::vector<float> buf;
        std::ifstream f(path, std::ios::binary);
        if (!f) return buf;
        f.seekg(sizeof(WavHeader) + start_frame * hdr.channels * 2);
        int n = std::min(4096, total_frames - start_frame);
        buf.resize(n);
        int16_t s;
        for (int i = 0; i < n; ++i) {
            f.read(reinterpret_cast<char*>(&s), 2);
            buf[i] = s / 32768.0f;
            if (hdr.channels == 2) f.seekg(2, std::ios::cur); // skip R
        }
        return buf;
    };

    int mid = total_frames / 2;
    auto mono_in  = read_window(in_path,  mid);
    auto mono_out = read_window(out_path, mid);

    auto spec_before = analyze_spectrum(mono_in,  plan, fs);
    auto spec_after  = analyze_spectrum(mono_out, plan, fs);

    {
        std::ofstream jf("spectrum.json");
        jf << "{\"before\":" << spectrum_to_json(spec_before, eq)
           << ",\"after\":"  << spectrum_to_json(spec_after,  eq) << "}";
        std::cout << "[JSON] Saved spectrum.json\n";
    }

    // Benchmark on a small buffer (won't OOM)
    int bench_frames = std::min(total_frames, (int)(fs * 2));
    std::vector<float> bench_buf(bench_frames);
    {
        std::ifstream f(in_path, std::ios::binary);
        f.seekg(sizeof(WavHeader));
        int16_t s;
        for (int i = 0; i < bench_frames; ++i) {
            f.read(reinterpret_cast<char*>(&s), 2);
            bench_buf[i] = s / 32768.0f;
            if (hdr.channels == 2) f.seekg(2, std::ios::cur);
        }
    }

    auto bench = benchmark(bench_buf, fs);
    (void)bench;

    std::cout << "\n=== Pipeline Finalizat ===\n";
    return 0;
}
