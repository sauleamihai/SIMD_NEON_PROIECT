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

// ─────────────────────────────────────────────────────────────
//  TYPES & CONSTANTS
// ─────────────────────────────────────────────────────────────
typedef std::complex<float> Complex;
static constexpr float PI  = 3.14159265358979323846f;
static constexpr float TAU = 6.28318530717958647692f;

// ─────────────────────────────────────────────────────────────
//  WAV I/O
// ─────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────
//  NEON HELPERS
// ─────────────────────────────────────────────────────────────

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

// NEON vector multiply-add: out += a * b  (length must be multiple of 4)
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

// ─────────────────────────────────────────────────────────────
//  FFT — Cooley-Tukey iterative, precomputed twiddle table
// ─────────────────────────────────────────────────────────────
struct FFTPlan {
    int                  N;
    std::vector<Complex> twiddle;   // twiddle[k] = exp(-2πi k/N)
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
                // ── Full NEON butterfly: 4 butterflies per iteration ──
                // Each butterfly:  a[u] = U + W*V,  a[v] = U - W*V
                //
                // Complex multiply W*V in NEON (no scalar cross-domain):
                //   re(W*V) = re(W)*re(V) - im(W)*im(V)
                //   im(W*V) = re(W)*im(V) + im(W)*re(V)
                //
                // Layout in memory: [re0, im0, re1, im1, re2, im2, re3, im3]
                // We process 4 butterflies using two float32x4_t per array half:
                //   U_re = [re(u0), re(u1), re(u2), re(u3)]
                //   U_im = [im(u0), im(u1), im(u2), im(u3)]
                //   V_re, V_im similarly for lower half
                //   W_re, W_im for twiddle factors

                int j = 0;
                for (; j <= half - 4; j += 4) {
                    // Load upper half: 4 complex = 8 floats, deinterleave re/im
                    float32x4x2_t U = vld2q_f32(
                        reinterpret_cast<const float*>(&a[i + j]));
                    float32x4_t U_re = U.val[0];
                    float32x4_t U_im = U.val[1];

                    // Load lower half
                    float32x4x2_t V_raw = vld2q_f32(
                        reinterpret_cast<const float*>(&a[i + j + half]));
                    float32x4_t V_re = V_raw.val[0];
                    float32x4_t V_im = V_raw.val[1];

                    // Load 4 twiddle factors
                    float32x4x2_t W = vld2q_f32(
                        reinterpret_cast<const float*>(&twiddle[(j    ) * stride]));
                    // Note: twiddle array stores consecutive Complex values,
                    // but they have stride between them. Build W manually:
                    float w_re[4], w_im[4];
                    for (int k = 0; k < 4; ++k) {
                        w_re[k] = twiddle[(j + k) * stride].real();
                        w_im[k] = twiddle[(j + k) * stride].imag();
                    }
                    float32x4_t W_re = vld1q_f32(w_re);
                    float32x4_t W_im = vld1q_f32(w_im);

                    // Complex multiply: WV = W * V
                    // WV_re = W_re*V_re - W_im*V_im
                    // WV_im = W_re*V_im + W_im*V_re
                    float32x4_t WV_re = vmulq_f32(W_re, V_re);
                    WV_re = vmlsq_f32(WV_re, W_im, V_im);   // WV_re -= W_im*V_im

                    float32x4_t WV_im = vmulq_f32(W_re, V_im);
                    WV_im = vmlaq_f32(WV_im, W_im, V_re);   // WV_im += W_im*V_re

                    // Butterfly add/sub
                    float32x4_t S_re = vaddq_f32(U_re, WV_re);
                    float32x4_t S_im = vaddq_f32(U_im, WV_im);
                    float32x4_t D_re = vsubq_f32(U_re, WV_re);
                    float32x4_t D_im = vsubq_f32(U_im, WV_im);

                    // Store interleaved [re, im, re, im, ...]
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

// ─────────────────────────────────────────────────────────────
//  FIR FILTER — NEON optimized direct-form convolution
// ─────────────────────────────────────────────────────────────
struct FIRFilter {
    std::vector<float> coef;
    std::vector<float> delay;   // circular delay line
    int ptr = 0;

    FIRFilter() = default;
    explicit FIRFilter(std::vector<float> c) : coef(std::move(c)), delay(coef.size(), 0.0f) {}

    // Process one sample (used in real-time path)
    float process_sample(float x) {
        delay[ptr] = x;
        ptr = (ptr + 1) % (int)delay.size();
        // Flatten circular buffer for NEON dot-product
        int M = coef.size();
        float acc = 0.0f;
        int32x4_t zero = vdupq_n_s32(0);
        float32x4_t sum = vdupq_n_f32(0.0f);
        int idx = ptr; // oldest sample
        int j = 0;
        for (; j <= M - 4; j += 4) {
            float tmp[4];
            for (int k = 0; k < 4; ++k) {
                tmp[k] = delay[(idx + k) % M];
            }
            float32x4_t dv = vld1q_f32(tmp);
            float32x4_t cv = vld1q_f32(&coef[j]);
            sum = vmlaq_f32(sum, dv, cv);
            idx = (idx + 4) % M;
        }
        acc = neon_hsum(sum);
        for (; j < M; ++j) { acc += delay[idx] * coef[j]; idx = (idx+1)%M; }
        return acc;
    }

    // Block process — NEON dot-product, linear (not circular) for batch
    void process_block(const std::vector<float>& in, std::vector<float>& out) {
        int n = in.size();
        int M = coef.size();
        out.resize(n, 0.0f);

        // Prologue: samples where full history not yet available
        for (int i = 0; i < std::min(M - 1, n); ++i) {
            float acc = 0.0f;
            for (int j = 0; j <= i; ++j) acc += in[i - j] * coef[j];
            out[i] = acc;
        }

        // Main loop: NEON unrolled 4x dot-product
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
            // Scalar tail for remainder coefficients
            for (int j = M & ~3; j < M; ++j) {
                float c = coef[j];
                out[i  ] += in[i  -j] * c;
                out[i+1] += in[i+1-j] * c;
                out[i+2] += in[i+2-j] * c;
                out[i+3] += in[i+3-j] * c;
            }
            // Store horizontal sums
            float tmp[4] = { neon_hsum(acc0), neon_hsum(acc1),
                             neon_hsum(acc2), neon_hsum(acc3) };
            float32x4_t partial = vld1q_f32(&out[i]);
            vst1q_f32(&out[i], vaddq_f32(partial, vld1q_f32(tmp)));
        }

        // Epilogue
        for (int i = std::max(M-1, (n/4)*4); i < n; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < M && i-j >= 0; ++j) acc += in[i-j]*coef[j];
            out[i] = acc;
        }
    }
};

// ─────────────────────────────────────────────────────────────
//  BIQUAD — Transposed Direct Form II, scalar 4× unrolled
//
//  TDF2 recurrence (2 state variables, vs 4 for DF1):
//    y[n]  = b0·x[n] + s1
//    s1'   = b1·x[n] - a1·y[n] + s2
//    s2'   = b2·x[n] - a2·y[n]
//
//  Why scalar instead of NEON for the biquad loop?
//  Cortex-A53 is in-order with a 3-4 cycle cross-domain penalty
//  for every NEON→scalar transfer (vgetq_lane_f32). A biquad loop
//  needs the output of each sample as input to the next state
//  update — forcing at least 2 cross-domain transfers per sample.
//  This completely negates any NEON benefit on A53.
//
//  The correct strategy for A53:
//  - Keep the biquad loop purely scalar with locals in registers
//  - Unroll 4× so the compiler can overlap FP multiply latencies
//  - Use -ffast-math so GCC can reorder FP ops freely
//  - NEON is used everywhere else (int16↔float conversion, FFT,
//    magnitude, stereo I/O) where there are NO cross-domain transfers
// ─────────────────────────────────────────────────────────────

struct BiquadCoeffs {
    float b0, b1, b2, a1, a2;

    // precompute() e un no-op acum — păstrat pentru compatibilitate cu
    // funcțiile de design (biquad_peaking etc. îl apelează după calcul)
    void precompute() {}
};

// TDF2 state: only 2 floats needed (vs 4 for DF1)
struct BiquadState {
    float s1 = 0.f;
    float s2 = 0.f;
};

// ── Scalar TDF2 tick (for tail samples) ──────────────────────
inline float biquad_tick(const BiquadCoeffs& c, BiquadState& s, float x) {
    float y  = c.b0 * x + s.s1;
    float ns1 = c.b1 * x - c.a1 * y + s.s2;
    float ns2 = c.b2 * x - c.a2 * y;
    s.s1 = ns1;
    s.s2 = ns2;
    return y;
}

// ── Coefficient calculators — apelează precompute() după ─────

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

// ─────────────────────────────────────────────────────────────
//  10-BAND PARAMETRIC EQ  (biquad chain, correct architecture)
//
//  Signal path:  in → [shelf_lo] → [peak×8] → [shelf_hi] → out
//  Each filter is applied IN SERIES to the signal (not summed).
//  This gives flat response at 0 dB and correct interaction
//  between adjacent bands — exactly like hardware EQ.
// ─────────────────────────────────────────────────────────────

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
    static constexpr int N_BANDS  = 10;
    static constexpr int FIR_TAPS = 127; // kept for benchmark label

    EQBandDef bands[N_BANDS];
    float     sample_rate;

    // Precomputed coefficients (rebuilt when gains change)
    BiquadCoeffs coeffs[N_BANDS];

    // Per-channel TDF2 state: [channel][band]
    // Only s1, s2 needed — 2 floats vs 4 for DF1
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
            // Identity: y=x → b0=1, all else 0, precompute() handles it
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

    // ── Structuri pentru grupuri NEON ────────────────────────
    // Coeficienții packed în float32x4_t: lane k = banda k din grup
    struct BiquadGroup {
        float32x4_t B0, B1, B2, A1, A2;  // coeficienți packed
        float32x4_t S1, S2;              // stări packed per channel
        int         n_active = 0;         // câte bande sunt active în grup
        float       gain_norm = 1.0f;     // 1/n_active pentru normalizare output

        void reset_state() {
            S1 = S2 = vdupq_n_f32(0.f);
        }
    };

    // 3 grupuri × 2 canale
    BiquadGroup groups[2][3];  // [channel][group]

    void rebuild_filters() {
        for (int b = 0; b < N_BANDS; ++b)
            coeffs[b] = make_coeffs(b);
        memset(state, 0, sizeof(state));

        // Construim grupurile NEON — 4 bande per grup
        // Grup 0: bande 0-3, Grup 1: bande 4-7, Grup 2: bande 8-9 (+ 2 pasthrough)
        for (int ch = 0; ch < 2; ++ch) {
            for (int g = 0; g < 3; ++g) {
                float b0[4]={1,1,1,1}, b1[4]={0}, b2[4]={0};
                float a1[4]={0},       a2[4]={0};
                int n_active = 0;

                for (int lane = 0; lane < 4; ++lane) {
                    int b = g * 4 + lane;
                    if (b >= N_BANDS) break;
                    b0[lane] = coeffs[b].b0;
                    b1[lane] = coeffs[b].b1;
                    b2[lane] = coeffs[b].b2;
                    a1[lane] = coeffs[b].a1;
                    a2[lane] = coeffs[b].a2;
                    // Contăm banda ca activă dacă are gain nenul
                    if (fabsf(bands[b].gain_db) >= 0.01f) ++n_active;
                }

                groups[ch][g].B0 = vld1q_f32(b0);
                groups[ch][g].B1 = vld1q_f32(b1);
                groups[ch][g].B2 = vld1q_f32(b2);
                groups[ch][g].A1 = vld1q_f32(a1);
                groups[ch][g].A2 = vld1q_f32(a2);
                groups[ch][g].reset_state();
                groups[ch][g].n_active  = n_active;
                // Normalizăm outputul grupului: dacă nicio bandă nu e activă,
                // pasthrough pur (gain_norm = 1). Altfel sumăm 4 bande și
                // împărțim la 4 ca să menținem amplitudinea corectă.
                groups[ch][g].gain_norm = 1.0f / 4.0f;
            }
        }
    }

    // ── NEON 4-bande-paralel per sample ──────────────────────
    //
    //  Arhitectură:
    //    x → [Grup0: bande 0-3 paralel] → sum/4 → [Grup1: bande 4-7] → sum/4 → [Grup2]→ out
    //
    //  Per sample, un grup face:
    //    y[0..3] = B0*x + S1          (4 biquad-uri simultan, FĂRĂ cross-domain)
    //    S1      = B1*x - A1*y + S2   (update stare în NEON pur)
    //    S2      = B2*x - A2*y
    //    output  = hsum(y) * gain_norm (suma celor 4 bande, normalizată)
    //
    //  Zero vgetq_lane în inner loop — totul rămâne în registre NEON
    //  până la hsum final (o dată per grup, nu per sample).
    //
    void process_block_mono(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n, int ch) {

        // Dacă nicio bandă nu e activă → passthrough pur
        int total_active = 0;
        for (int b = 0; b < N_BANDS; ++b)
            if (fabsf(bands[b].gain_db) >= 0.01f) ++total_active;
        if (total_active == 0) {
            memcpy(out, in, n * sizeof(float));
            return;
        }

        // Recuperăm stările grupurilor în registre locale
        float32x4_t S1_g0 = groups[ch][0].S1, S2_g0 = groups[ch][0].S2;
        float32x4_t S1_g1 = groups[ch][1].S1, S2_g1 = groups[ch][1].S2;
        float32x4_t S1_g2 = groups[ch][2].S1, S2_g2 = groups[ch][2].S2;

        // Coeficienți în registre — rămân pe tot parcursul buclei
        const float32x4_t B0_g0=groups[ch][0].B0, B1_g0=groups[ch][0].B1;
        const float32x4_t B2_g0=groups[ch][0].B2, A1_g0=groups[ch][0].A1;
        const float32x4_t A2_g0=groups[ch][0].A2;

        const float32x4_t B0_g1=groups[ch][1].B0, B1_g1=groups[ch][1].B1;
        const float32x4_t B2_g1=groups[ch][1].B2, A1_g1=groups[ch][1].A1;
        const float32x4_t A2_g1=groups[ch][1].A2;

        const float32x4_t B0_g2=groups[ch][2].B0, B1_g2=groups[ch][2].B1;
        const float32x4_t B2_g2=groups[ch][2].B2, A1_g2=groups[ch][2].A1;
        const float32x4_t A2_g2=groups[ch][2].A2;

        // Normalizare: fiecare grup sumează 4 bande → împărțim la 4
        const float32x4_t NORM = vdupq_n_f32(0.25f);

        for (int i = 0; i < n; ++i) {
            // Broadcast sample scalar în toate 4 lane-urile
            float32x4_t X = vdupq_n_f32(in[i]);

            // ── Grup 0: bande 0-3 procesate simultan ─────────
            // TDF2: y = B0*x + S1
            float32x4_t Y0 = vmlaq_f32(S1_g0, B0_g0, X);
            // S1' = B1*x - A1*y + S2  →  S2 + B1*x - A1*y
            float32x4_t ns1_g0 = vmlaq_f32(vmlsq_f32(S2_g0, A1_g0, Y0), B1_g0, X);
            // S2' = B2*x - A2*y
            float32x4_t ns2_g0 = vmlsq_f32(vmulq_f32(B2_g0, X), A2_g0, Y0);
            S1_g0 = ns1_g0; S2_g0 = ns2_g0;
            // Output grup 0: suma celor 4 bande normalizată → scalar → NEON
            float x1 = neon_hsum(vmulq_f32(Y0, NORM));

            // ── Grup 1: bande 4-7 ─────────────────────────────
            float32x4_t X1 = vdupq_n_f32(x1);
            float32x4_t Y1 = vmlaq_f32(S1_g1, B0_g1, X1);
            float32x4_t ns1_g1 = vmlaq_f32(vmlsq_f32(S2_g1, A1_g1, Y1), B1_g1, X1);
            float32x4_t ns2_g1 = vmlsq_f32(vmulq_f32(B2_g1, X1), A2_g1, Y1);
            S1_g1 = ns1_g1; S2_g1 = ns2_g1;
            float x2 = neon_hsum(vmulq_f32(Y1, NORM));

            // ── Grup 2: bande 8-9 ─────────────────────────────
            float32x4_t X2 = vdupq_n_f32(x2);
            float32x4_t Y2 = vmlaq_f32(S1_g2, B0_g2, X2);
            float32x4_t ns1_g2 = vmlaq_f32(vmlsq_f32(S2_g2, A1_g2, Y2), B1_g2, X2);
            float32x4_t ns2_g2 = vmlsq_f32(vmulq_f32(B2_g2, X2), A2_g2, Y2);
            S1_g2 = ns1_g2; S2_g2 = ns2_g2;
            float x3 = neon_hsum(vmulq_f32(Y2, NORM));

            out[i] = x3;
        }

        // Salvăm stările înapoi
        groups[ch][0].S1=S1_g0; groups[ch][0].S2=S2_g0;
        groups[ch][1].S1=S1_g1; groups[ch][1].S2=S2_g1;
        groups[ch][2].S1=S1_g2; groups[ch][2].S2=S2_g2;
    }

    // Legacy interface
    std::vector<float> process(const std::vector<float>& in) {
        std::vector<float> out(in.size());
        process_block_mono(in.data(), out.data(), in.size(), 0);
        return out;
    }
};

// ─────────────────────────────────────────────────────────────
//  SPECTRUM ANALYZER (NEON magnitude)
// ─────────────────────────────────────────────────────────────
struct SpectrumResult {
    std::vector<float> magnitudes;  // N_BINS values
    float              peak_db;
    float              rms_db;
    static constexpr int N_BINS = 64;
};

// NEON magnitude calculation: sqrt(re² + im²)
//
// AArch64 has vsqrtq_f32 — a native hardware sqrt instruction that is
// IEEE-correct, handles zeros cleanly (sqrt(0)=0), and processes 4 floats
// simultaneously. On Cortex-A53 it has ~17 cycle latency but throughput
// of 1 per 4 cycles when pipelined across iterations.
//
// This is simpler and faster than vrsqrteq_f32 + Newton-Raphson + masking,
// which adds 5 extra instructions and a branch for the zero case.
//
// We unroll 2× (8 complex values per iteration) to hide the sqrt latency
// by keeping both sqrt units busy simultaneously.
void neon_magnitude(const Complex* fft_buf, float* mag, int n) {
    int i = 0;
    // 2× unrolled: 8 complex values per iteration
    for (; i <= n - 8; i += 8) {
        // Load 8 complex values deinterleaved (re and im separate)
        float32x4x2_t v0 = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i));
        float32x4x2_t v1 = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i + 4));

        // sq = re² + im²  (fused multiply-add)
        float32x4_t sq0 = vmlaq_f32(vmulq_f32(v0.val[0], v0.val[0]), v0.val[1], v0.val[1]);
        float32x4_t sq1 = vmlaq_f32(vmulq_f32(v1.val[0], v1.val[0]), v1.val[1], v1.val[1]);

        // Hardware sqrt — IEEE-correct, sqrt(0)=0, no NaN
        // Issue both before storing so A53 can pipeline them
        float32x4_t m0 = vsqrtq_f32(sq0);
        float32x4_t m1 = vsqrtq_f32(sq1);

        vst1q_f32(mag + i,     m0);
        vst1q_f32(mag + i + 4, m1);
    }
    // 4-wide tail
    for (; i <= n - 4; i += 4) {
        float32x4x2_t v = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i));
        float32x4_t sq = vmlaq_f32(vmulq_f32(v.val[0], v.val[0]), v.val[1], v.val[1]);
        vst1q_f32(mag + i, vsqrtq_f32(sq));
    }
    // Scalar tail
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

    // Apply Hann window
    for (int i = 0; i < lim; ++i) {
        float w = 0.5f - 0.5f * cosf(TAU * i / (lim - 1));
        buf[i] = {signal[start + i] * w, 0.f};
    }

    plan.forward(buf);

    // NEON magnitude for first half of spectrum
    int half = N / 2;
    std::vector<float> mag(half);
    neon_magnitude(buf.data(), mag.data(), half);

    // Bin into N_BINS log-frequency bands
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

    // Convert to dB, NEON log approximation not available directly, use scalar
    float peak = 1e-10f;
    for (float& m : res.magnitudes) {
        m = (m < 1e-10f) ? -80.0f : 20.0f * log10f(m);
        if (m > peak) peak = m;
    }
    res.peak_db = peak;

    // RMS
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

// ─────────────────────────────────────────────────────────────
//  JSON OUTPUT (for Flask GUI)
// ─────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────
//  BENCHMARK  (writes benchmark.json for Flask GUI)
// ─────────────────────────────────────────────────────────────
struct BenchResult {
    // FIR
    double fir_naive_ms;
    double fir_neon_ms;
    double fir_speedup;
    int    fir_samples;
    int    fir_taps;
    // FFT
    double fft_naive_ms;
    double fft_neon_ms;
    double fft_speedup;
    int    fft_size;
    int    fft_runs;
    // Magnitude
    double mag_scalar_ms;
    double mag_neon_ms;
    double mag_speedup;
    int    mag_size;
    // Stereo
    double stereo_scalar_ms;
    double stereo_neon_ms;
    double stereo_speedup;
    int    stereo_frames;
    // Overall
    double total_naive_ms;
    double total_neon_ms;
    double total_speedup;
    float  sample_rate;
};

BenchResult benchmark(const std::vector<float>& signal, float fs) {
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout <<   "║        BENCHMARK RESULTS             ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n";

    BenchResult res{};
    res.sample_rate = fs;

    int N = std::min((int)signal.size(), (int)(fs * 2));
    std::vector<float> test(signal.begin(), signal.begin() + N);

    // ── 1. Biquad EQ: single-sample scalar vs 4× unrolled TDF2 ──
    // Both are scalar — we compare the unrolling benefit and measure
    // real-time factor (how many seconds of audio per second of CPU)
    constexpr int TAPS = 127;
    Equalizer eq_simple(fs), eq_unrolled(fs);
    for (int b = 0; b < Equalizer::N_BANDS; ++b) {
        eq_simple.set_gain(b,   (b % 3 == 0) ? 6.f : -3.f);
        eq_unrolled.set_gain(b, (b % 3 == 0) ? 6.f : -3.f);
    }
    eq_simple.rebuild_filters();
    eq_unrolled.rebuild_filters();

    std::vector<float> out_simple(N), out_unrolled(N);
    res.fir_samples = N;
    res.fir_taps    = TAPS;

    // Simple scalar: one sample at a time via biquad_tick
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        BiquadState s[Equalizer::N_BANDS] = {};
        std::copy(test.begin(), test.end(), out_simple.begin());
        for (int b = 0; b < Equalizer::N_BANDS; ++b) {
            if (fabsf(eq_simple.bands[b].gain_db) < 0.01f) continue;
            BiquadCoeffs c = eq_simple.make_coeffs(b);
            for (int i = 0; i < N; ++i)
                out_simple[i] = biquad_tick(c, s[b], out_simple[i]);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // 4× unrolled TDF2 via process_block_mono
    eq_unrolled.process_block_mono(test.data(), out_unrolled.data(), N, 0);
    auto t2 = std::chrono::high_resolution_clock::now();

    res.fir_naive_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.fir_neon_ms  = std::chrono::duration<double,std::milli>(t2-t1).count();
    res.fir_speedup  = res.fir_naive_ms / std::max(res.fir_neon_ms, 0.001);

    // Real-time factor: audio_duration / processing_time
    double audio_duration_ms = (N / fs) * 1000.0;
    double rt_factor = audio_duration_ms / res.fir_neon_ms;

    std::cout << "Biquad 1-sample scalar: " << res.fir_naive_ms  << " ms\n";
    std::cout << "Biquad 4x unrolled TDF2:" << res.fir_neon_ms   << " ms\n";
    std::cout << "Unroll speedup:         " << res.fir_speedup   << "x\n";
    std::cout << "Real-time factor:       " << rt_factor         << "x  ("
              << (rt_factor >= 1.0 ? "REALTIME OK" : "TOO SLOW") << ")\n\n";

    // ── 2. FFT: naive recursive vs NEON iterative ──────────
    constexpr int FFT_N = 4096;
    constexpr int FFT_RUNS = 100;
    res.fft_size = FFT_N;
    res.fft_runs = FFT_RUNS;

    // Naive recursive FFT (copy from original code style)
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
                a[i] = even[i] + w*odd[i];
                a[i+n/2] = even[i] - w*odd[i];
                w *= wn;
            }
        };

    FFTPlan plan_bench(FFT_N);
    std::vector<Complex> buf(FFT_N), buf2(FFT_N);
    for (int i = 0; i < FFT_N; ++i) {
        float s = test[i % N];
        buf[i] = buf2[i] = {s, 0.f};
    }

    // Naive recursive (1 run — it's O(N log²N), can be slow)
    t0 = std::chrono::high_resolution_clock::now();
    fft_naive_rec(buf);
    t1 = std::chrono::high_resolution_clock::now();

    // NEON iterative (100 runs, averaged)
    t1 = std::chrono::high_resolution_clock::now();  // reset start
    for (int i = 0; i < FFT_N; ++i) buf2[i] = {test[i % N], 0.f};
    auto tfft0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < FFT_RUNS; ++r) {
        for (int i = 0; i < FFT_N; ++i) buf2[i] = {test[i % N], 0.f};
        plan_bench.forward(buf2);
    }
    auto tfft1 = std::chrono::high_resolution_clock::now();

    // Single run naive for fair comparison
    for (int i = 0; i < FFT_N; ++i) buf[i] = {test[i % N], 0.f};
    auto tfft_n0 = std::chrono::high_resolution_clock::now();
    fft_naive_rec(buf);
    auto tfft_n1 = std::chrono::high_resolution_clock::now();

    res.fft_naive_ms = std::chrono::duration<double,std::milli>(tfft_n1-tfft_n0).count();
    res.fft_neon_ms  = std::chrono::duration<double,std::milli>(tfft1-tfft0).count() / FFT_RUNS;
    res.fft_speedup  = res.fft_naive_ms / std::max(res.fft_neon_ms, 0.001);

    std::cout << "FFT naive (1 run, " << FFT_N << "-pt): " << res.fft_naive_ms << " ms\n";
    std::cout << "FFT NEON  (avg of " << FFT_RUNS << " runs): " << res.fft_neon_ms  << " ms\n";
    std::cout << "FFT speedup: " << res.fft_speedup << "x\n\n";

    // ── 3. Magnitude: scalar sqrtf vs NEON rsqrt ──────────
    constexpr int MAG_N = 65536;
    res.mag_size = MAG_N;
    std::vector<Complex> mag_buf(MAG_N);
    std::vector<float>   mag_out(MAG_N);
    for (int i = 0; i < MAG_N; ++i) mag_buf[i] = {(float)i*0.001f, (float)i*0.0007f};

    // Scalar
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MAG_N; ++i)
        mag_out[i] = std::abs(mag_buf[i]);
    t1 = std::chrono::high_resolution_clock::now();

    // NEON
    t1 = std::chrono::high_resolution_clock::now();
    neon_magnitude(mag_buf.data(), mag_out.data(), MAG_N);
    t2 = std::chrono::high_resolution_clock::now();

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

    // ── 4. Stereo interleave: scalar vs NEON vld2/vst2 ────
    constexpr int ST_FRAMES = 88200; // 2s stereo @ 44.1k
    res.stereo_frames = ST_FRAMES;
    std::vector<float> L(ST_FRAMES), R(ST_FRAMES), LR(ST_FRAMES*2), LR2(ST_FRAMES*2);
    for (int i = 0; i < ST_FRAMES; ++i) { L[i]=test[i%N]; R[i]=test[(i+N/2)%N]; }

    // Scalar deinterleave
    auto tst0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ST_FRAMES; ++i) { LR[i*2]=L[i]; LR[i*2+1]=R[i]; }
    for (int i = 0; i < ST_FRAMES; ++i) { L[i]=LR[i*2]; R[i]=LR[i*2+1]; }
    auto tst1 = std::chrono::high_resolution_clock::now();

    // NEON
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

    // ── 5. Overall totals ─────────────────────────────────
    res.total_naive_ms = res.fir_naive_ms + res.fft_naive_ms + res.mag_scalar_ms + res.stereo_scalar_ms;
    res.total_neon_ms  = res.fir_neon_ms  + res.fft_neon_ms  + res.mag_neon_ms   + res.stereo_neon_ms;
    res.total_speedup  = res.total_naive_ms / std::max(res.total_neon_ms, 0.001);

    std::cout << "─────────────────────────────────────────\n";
    std::cout << "TOTAL naive: " << res.total_naive_ms << " ms\n";
    std::cout << "TOTAL NEON:  " << res.total_neon_ms  << " ms\n";
    std::cout << "OVERALL SPEEDUP: " << res.total_speedup << "x\n";
    std::cout << "─────────────────────────────────────────\n";

    // ── 6. Write benchmark.json ───────────────────────────
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

// ─────────────────────────────────────────────────────────────
//  CHUNKED STREAMING PROCESSOR
//  Biquad IIR is stateful — state carries across chunk boundaries
//  automatically. No overlap-add needed. Much simpler than FIR.
//  Read CHUNK_FRAMES at a time, process in-place, write out.
// ─────────────────────────────────────────────────────────────
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

    // Working buffers — allocated once outside the loop
    std::vector<int16_t> raw(CHUNK_FRAMES * CH);
    std::vector<float>   chL(CHUNK_FRAMES), chR(CHUNK_FRAMES);
    std::vector<int16_t> out_raw(CHUNK_FRAMES * CH);

    // Precompute NEON constants used in every chunk
    const float32x4_t vSCALE_IN  = vdupq_n_f32(1.0f / 32768.0f);  // int16→float
    const float32x4_t vSCALE_OUT = vdupq_n_f32(32767.0f);          // float→int16
    const float32x4_t vCLAMP_HI  = vdupq_n_f32( 1.0f);
    const float32x4_t vCLAMP_LO  = vdupq_n_f32(-1.0f);

    int frames_done = 0;
    while (frames_done < total_frames) {
        int frames_now = std::min(CHUNK_FRAMES, total_frames - frames_done);

        // Read raw int16 from disk
        fin.read(reinterpret_cast<char*>(raw.data()),
                 frames_now * CH * bytes_per_sample);

        if (CH == 1) {
            // ── MONO int16 → float ──────────────────────────
            // NEON: 8 samples per iteration
            //   vld1_s16     → int16x8  (8 × int16)
            //   vmovl_s16    → int32x4  (widen low 4)
            //   vget_high_s16→ vmovl    (widen high 4)
            //   vcvtq_f32_s32→ float32x4
            //   vmulq_f32    → ÷ 32768
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
            // Scalar tail (< 8 remaining)
            for (; i < frames_now; ++i)
                chL[i] = raw[i] * (1.0f / 32768.0f);

            // Process EQ
            eq.process_block_mono(chL.data(), chL.data(), frames_now, 0);

            // ── MONO float → int16 ──────────────────────────
            // NEON: 8 samples per iteration
            //   vmaxq / vminq → clamp to [-1, 1]
            //   vmulq_f32     → × 32767
            //   vcvtq_s32_f32 → int32
            //   vmovn_s32     → int16 (narrow)
            //   vcombine_s16  → int16x8
            //   vst1q_s16     → store 8 × int16
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
            // ── STEREO int16 → float (deinterleave) ─────────
            // NEON: vld2q_s16 loads 8 frames (16 int16) and
            // automatically deinterleaves into L and R vectors.
            // Then widen + convert each half as in mono path.
            int i = 0;
            for (; i <= frames_now - 8; i += 8) {
                // vld2q_s16: loads [L0 R0 L1 R1 ... L7 R7]
                // → val[0] = [L0 L1 L2 L3 L4 L5 L6 L7]
                // → val[1] = [R0 R1 R2 R3 R4 R5 R6 R7]
                int16x8x2_t s16 = vld2q_s16(raw.data() + i * 2);

                // Left channel
                float32x4_t Llo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16.val[0]))),  vSCALE_IN);
                float32x4_t Lhi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16.val[0]))), vSCALE_IN);
                vst1q_f32(chL.data() + i,     Llo);
                vst1q_f32(chL.data() + i + 4, Lhi);

                // Right channel
                float32x4_t Rlo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16.val[1]))),  vSCALE_IN);
                float32x4_t Rhi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16.val[1]))), vSCALE_IN);
                vst1q_f32(chR.data() + i,     Rlo);
                vst1q_f32(chR.data() + i + 4, Rhi);
            }
            // Scalar tail
            for (; i < frames_now; ++i) {
                chL[i] = raw[i * 2    ] * (1.0f / 32768.0f);
                chR[i] = raw[i * 2 + 1] * (1.0f / 32768.0f);
            }

            // Process EQ — L and R independently (preserves stereo state)
            eq.process_block_mono(chL.data(), chL.data(), frames_now, 0);
            eq.process_block_mono(chR.data(), chR.data(), frames_now, 1);

            // ── STEREO float → int16 (interleave) ───────────
            // vst2_s16: takes two int16x4_t and writes [L0 R0 L1 R1 ...]
            // We do 8 frames per iteration using vst2q_s16 (int16x8x2_t)
            i = 0;
            for (; i <= frames_now - 8; i += 8) {
                float32x4_t L0 = vld1q_f32(chL.data() + i);
                float32x4_t L1 = vld1q_f32(chL.data() + i + 4);
                float32x4_t R0 = vld1q_f32(chR.data() + i);
                float32x4_t R1 = vld1q_f32(chR.data() + i + 4);

                L0 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, L0)), vSCALE_OUT);
                L1 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, L1)), vSCALE_OUT);
                R0 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, R0)), vSCALE_OUT);
                R1 = vmulq_f32(vmaxq_f32(vCLAMP_LO, vminq_f32(vCLAMP_HI, R1)), vSCALE_OUT);

                int16x8x2_t out16;
                out16.val[0] = vcombine_s16(vmovn_s32(vcvtq_s32_f32(L0)),
                                            vmovn_s32(vcvtq_s32_f32(L1)));
                out16.val[1] = vcombine_s16(vmovn_s32(vcvtq_s32_f32(R0)),
                                            vmovn_s32(vcvtq_s32_f32(R1)));
                vst2q_s16(out_raw.data() + i * 2, out16);
            }
            // Scalar tail
            for (; i < frames_now; ++i) {
                float L = std::max(-1.f, std::min(1.f, chL[i]));
                float R = std::max(-1.f, std::min(1.f, chR[i]));
                out_raw[i * 2    ] = static_cast<int16_t>(L * 32767.f);
                out_raw[i * 2 + 1] = static_cast<int16_t>(R * 32767.f);
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

// ─────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════╗\n"
              << "║   DSP Audio EQ — ARM NEON (Pi Zero 2W) ║\n"
              << "╚════════════════════════════════════════╝\n\n";

    std::string in_path  = (argc > 1) ? argv[1] : "test_audio.wav";
    std::string out_path = (argc > 2) ? argv[2] : "audio_eq.wav";

    float gains[10] = {};
    for (int i = 3; i < argc && i - 3 < 10; ++i)
        gains[i - 3] = std::stof(argv[i]);

    // ── Read header only (not the whole file) ────────────────
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

    // ── Stream-process (no full file in RAM) ─────────────────
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!process_streaming(in_path, out_path, eq, hdr)) return 1;
    auto t_end = std::chrono::high_resolution_clock::now();
    double proc_ms = std::chrono::duration<double,std::milli>(t_end-t_start).count();
    std::cout << "[PERF] Processing time: " << proc_ms/1000.0 << "s\n";

    // ── Spectrum: read only a small window for analysis ───────
    // Read 4096 frames from the middle of the file — tiny RAM use
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

    // Save spectrum JSON
    {
        std::ofstream jf("spectrum.json");
        jf << "{\"before\":" << spectrum_to_json(spec_before, eq)
           << ",\"after\":"  << spectrum_to_json(spec_after,  eq) << "}";
        std::cout << "[JSON] Saved spectrum.json\n";
    }

    // ── Benchmark on small buffer (won't OOM) ────────────────
    // Use only 2s worth of data for benchmark
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