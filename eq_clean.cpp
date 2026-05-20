// DSP Audio EQ — 10-band parametric equalizer + FFT spectrum
// Target: Raspberry Pi Zero 2W (ARM Cortex-A53, AArch64 NEON)
//
// NEON-accelerated regions are marked with banners:
//     // ===== NEON =====  ...  // ===== END NEON =====
//
// Build: g++ -O3 -ffast-math -mcpu=cortex-a53 -funroll-loops -std=c++17 \
//            eq_clean.cpp -o eq

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <functional>
#include <arm_neon.h>

typedef std::complex<float> Complex;
static constexpr float PI  = 3.14159265358979323846f;
static constexpr float TAU = 6.28318530717958647692f;

// ─────────────────────────────────────────────────────────────────────────────
// WAV header
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// NEON helpers
// ─────────────────────────────────────────────────────────────────────────────

// ===== NEON =====
// Horizontal sum of a float32x4_t (one NEON→scalar transfer; use once/block).
inline float neon_hsum(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}

// Stereo deinterleave [L R L R ...] -> L[], R[]   (4 frames/op).
void neon_deinterleave(const float* lr, float* L, float* R, int frames) {
    int i = 0;
    for (; i <= frames - 4; i += 4) {
        float32x4x2_t v = vld2q_f32(lr + i * 2);
        vst1q_f32(L + i, v.val[0]);
        vst1q_f32(R + i, v.val[1]);
    }
    for (; i < frames; ++i) { L[i] = lr[i*2]; R[i] = lr[i*2+1]; }
}

// Stereo interleave L[], R[] -> [L R L R ...]   (4 frames/op).
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
// ===== END NEON =====

// ─────────────────────────────────────────────────────────────────────────────
// FFT — iterative radix-2 Cooley-Tukey, precomputed twiddles + bit-reversal
// ─────────────────────────────────────────────────────────────────────────────
struct FFTPlan {
    int                  N;
    std::vector<Complex> twiddle;   // twiddle[k] = exp(-2pi i k/N)
    std::vector<int>     rev;       // bit-reversal permutation

    explicit FFTPlan(int n) : N(n), twiddle(n / 2), rev(n) {
        assert((n & (n - 1)) == 0 && "FFT size must be a power of 2");
        for (int k = 0; k < n / 2; ++k)
            twiddle[k] = { cosf(TAU * k / n), -sinf(TAU * k / n) };
        int logn = __builtin_ctz(n);
        for (int i = 0; i < n; ++i) {
            rev[i] = 0;
            for (int b = 0; b < logn; ++b)
                rev[i] |= ((i >> b) & 1) << (logn - 1 - b);
        }
    }

    void forward(std::vector<Complex>& a) const {
        for (int i = 0; i < N; ++i)
            if (i < rev[i]) std::swap(a[i], a[rev[i]]);

        for (int len = 2; len <= N; len <<= 1) {
            int half   = len >> 1;
            int stride = N / len;

            for (int i = 0; i < N; i += len) {
                int j = 0;
                // ===== NEON ===== (4 independent butterflies / iteration)
                for (; j <= half - 4; j += 4) {
                    float32x4x2_t U = vld2q_f32(
                        reinterpret_cast<const float*>(&a[i + j]));
                    float32x4_t U_re = U.val[0], U_im = U.val[1];

                    float32x4x2_t V = vld2q_f32(
                        reinterpret_cast<const float*>(&a[i + j + half]));
                    float32x4_t V_re = V.val[0], V_im = V.val[1];

                    float wr[4], wi[4];
                    for (int k = 0; k < 4; ++k) {            // strided gather
                        wr[k] = twiddle[(j + k) * stride].real();
                        wi[k] = twiddle[(j + k) * stride].imag();
                    }
                    float32x4_t W_re = vld1q_f32(wr);
                    float32x4_t W_im = vld1q_f32(wi);

                    float32x4_t T_re = vmlsq_f32(vmulq_f32(W_re, V_re), W_im, V_im);
                    float32x4_t T_im = vmlaq_f32(vmulq_f32(W_re, V_im), W_im, V_re);

                    float32x4x2_t S = { vaddq_f32(U_re, T_re),
                                        vaddq_f32(U_im, T_im) };
                    float32x4x2_t D = { vsubq_f32(U_re, T_re),
                                        vsubq_f32(U_im, T_im) };
                    vst2q_f32(reinterpret_cast<float*>(&a[i + j]),        S);
                    vst2q_f32(reinterpret_cast<float*>(&a[i + j + half]), D);
                }
                // ===== END NEON =====
                for (; j < half; ++j) {                      // scalar tail
                    Complex w = twiddle[j * stride];
                    Complex u = a[i + j];
                    Complex v = a[i + j + half] * w;
                    a[i + j]        = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }

    void inverse(std::vector<Complex>& a) const {
        for (auto& c : a) c = std::conj(c);
        forward(a);
        for (auto& c : a) c = std::conj(c) * (1.0f / N);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Biquad (Transposed Direct Form II) + RBJ cookbook coefficients
// ─────────────────────────────────────────────────────────────────────────────
struct BiquadCoeffs { float b0, b1, b2, a1, a2; };
struct BiquadState  { float s1 = 0.f, s2 = 0.f; };

BiquadCoeffs biquad_peaking(float fc, float gain_db, float Q, float fs) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / fs;
    float cw = cosf(w0), sw = sinf(w0);
    float al = sw / (2.0f * Q);
    float inv = 1.0f / (1.0f + al / A);
    return { (1.0f + al*A)*inv, (-2.0f*cw)*inv, (1.0f - al*A)*inv,
             (-2.0f*cw)*inv,    (1.0f - al/A)*inv };
}

BiquadCoeffs biquad_low_shelf(float fc, float gain_db, float fs) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / fs;
    float cw = cosf(w0), sw = sinf(w0);
    float al = sw / 2.0f * sqrtf((A + 1.0f/A) * (1.0f/0.707f - 1.0f) + 2.0f);
    float inv = 1.0f / ((A+1) + (A-1)*cw + 2*sqrtf(A)*al);
    return { A*((A+1) - (A-1)*cw + 2*sqrtf(A)*al)*inv,
             2.0f*A*((A-1) - (A+1)*cw)*inv,
             A*((A+1) - (A-1)*cw - 2*sqrtf(A)*al)*inv,
             -2.0f*((A-1) + (A+1)*cw)*inv,
             ((A+1) + (A-1)*cw - 2*sqrtf(A)*al)*inv };
}

BiquadCoeffs biquad_high_shelf(float fc, float gain_db, float fs) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / fs;
    float cw = cosf(w0), sw = sinf(w0);
    float al = sw / 2.0f * sqrtf((A + 1.0f/A) * (1.0f/0.707f - 1.0f) + 2.0f);
    float inv = 1.0f / ((A+1) - (A-1)*cw + 2*sqrtf(A)*al);
    return { A*((A+1) + (A-1)*cw + 2*sqrtf(A)*al)*inv,
             -2.0f*A*((A-1) + (A+1)*cw)*inv,
             A*((A+1) + (A-1)*cw - 2*sqrtf(A)*al)*inv,
             2.0f*((A-1) - (A+1)*cw)*inv,
             ((A+1) - (A-1)*cw - 2*sqrtf(A)*al)*inv };
}

// ─────────────────────────────────────────────────────────────────────────────
// 10-band parametric equalizer (series cascade)
//   mono   : band-major scalar (optimal scalar form)
//   stereo : NEON 2-lane (L|R) + band-pair fusion (memory-traffic halving)
// ─────────────────────────────────────────────────────────────────────────────
struct EQBandDef {
    const char* name;
    float       fc;
    float       Q;
    enum Type { LOW_SHELF, PEAK, HIGH_SHELF } type;
    float       gain_db;
};

static EQBandDef DEFAULT_BANDS[10] = {
    {"Sub-Bass",    60.0f, 0.7f, EQBandDef::LOW_SHELF,  0.f},
    {"Bass",       170.0f, 1.0f, EQBandDef::PEAK,       0.f},
    {"Low-Mid",    350.0f, 1.0f, EQBandDef::PEAK,       0.f},
    {"Mid",        700.0f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Upper-Mid", 1400.0f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Presence",  3000.0f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Brilliance",6000.0f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Air",      10000.0f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Ultra",    14000.0f, 1.4f, EQBandDef::PEAK,       0.f},
    {"High",     18000.0f, 0.7f, EQBandDef::HIGH_SHELF, 0.f},
};

class Equalizer {
public:
    static constexpr int N_BANDS = 10;

    EQBandDef    bands[N_BANDS];
    float        sample_rate;
    BiquadCoeffs coeffs[N_BANDS];
    BiquadState  state[2][N_BANDS];          // [channel][band]

    explicit Equalizer(float fs) : sample_rate(fs) {
        memcpy(bands, DEFAULT_BANDS, sizeof(DEFAULT_BANDS));
        memset(state, 0, sizeof(state));
        rebuild_filters();
    }

    void set_gain(int b, float db) {
        bands[b].gain_db = std::max(-24.0f, std::min(24.0f, db));
    }

    BiquadCoeffs make_coeffs(int b) const {
        const EQBandDef& d = bands[b];
        if (fabsf(d.gain_db) < 0.01f) return { 1.f, 0.f, 0.f, 0.f, 0.f };
        switch (d.type) {
            case EQBandDef::LOW_SHELF:  return biquad_low_shelf (d.fc, d.gain_db, sample_rate);
            case EQBandDef::HIGH_SHELF: return biquad_high_shelf(d.fc, d.gain_db, sample_rate);
            default:                    return biquad_peaking   (d.fc, d.gain_db, d.Q, sample_rate);
        }
    }

    void rebuild_filters() {
        for (int b = 0; b < N_BANDS; ++b) coeffs[b] = make_coeffs(b);
        memset(state, 0, sizeof(state));
    }

    // Mono: band-major scalar series cascade, in place. 0 dB bands skipped.
    void process_block_mono(const float* __restrict__ in,
                            float* __restrict__ out, int n, int ch) {
        if (out != in) memcpy(out, in, n * sizeof(float));
        for (int b = 0; b < N_BANDS; ++b) {
            if (fabsf(bands[b].gain_db) < 0.01f) continue;
            const float b0 = coeffs[b].b0, b1 = coeffs[b].b1, b2 = coeffs[b].b2;
            const float a1 = coeffs[b].a1, a2 = coeffs[b].a2;
            float s1 = state[ch][b].s1, s2 = state[ch][b].s2;
            for (int i = 0; i < n; ++i) {
                float x  = out[i];
                float y  = b0 * x + s1;
                float n1 = b1 * x - a1 * y + s2;
                float n2 = b2 * x - a2 * y;
                s1 = n1; s2 = n2;
                out[i] = y;
            }
            state[ch][b].s1 = s1; state[ch][b].s2 = s2;
        }
    }

    // Stereo: interleaved [L R ...], in place. Band-pair fusion = one
    // load/store per (L,R) sample for two biquads (halves buffer passes).
    // ===== NEON ===== (lane0 = L, lane1 = R; both channels per op)
    void process_block_stereo(float* __restrict__ lr, int frames) {
        int act[N_BANDS], na = 0;
        for (int b = 0; b < N_BANDS; ++b)
            if (fabsf(bands[b].gain_db) >= 0.01f) act[na++] = b;
        if (na == 0) return;

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
                float32x2_t x  = vld1_f32(lr + 2 * i);
                float32x2_t ya = vmla_f32(sa1, A0, x);
                sa1 = vmla_f32(vmls_f32(sa2, Aa1, ya), A1, x);
                sa2 = vmls_f32(vmul_f32(A2, x), Aa2, ya);
                float32x2_t yb = vmla_f32(sb1, B0, ya);
                sb1 = vmla_f32(vmls_f32(sb2, Bb1, yb), B1, ya);
                sb2 = vmls_f32(vmul_f32(B2, ya), Bb2, yb);
                vst1_f32(lr + 2 * i, yb);
            }
            state[0][ba].s1=vget_lane_f32(sa1,0); state[1][ba].s1=vget_lane_f32(sa1,1);
            state[0][ba].s2=vget_lane_f32(sa2,0); state[1][ba].s2=vget_lane_f32(sa2,1);
            state[0][bb].s1=vget_lane_f32(sb1,0); state[1][bb].s1=vget_lane_f32(sb1,1);
            state[0][bb].s2=vget_lane_f32(sb2,0); state[1][bb].s2=vget_lane_f32(sb2,1);
        }
        if (k < na) {                                       // odd trailing band
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
    // ===== END NEON =====
};

// ─────────────────────────────────────────────────────────────────────────────
// Spectrum analysis
// ─────────────────────────────────────────────────────────────────────────────
struct SpectrumResult {
    std::vector<float> magnitudes;
    float              peak_db;
    float              rms_db;
    static constexpr int N_BINS = 64;
};

// ===== NEON ===== (re^2+im^2 via fused mul-add; 4-wide HW sqrt; 2x unroll)
void neon_magnitude(const Complex* fft_buf, float* mag, int n) {
    int i = 0;
    for (; i <= n - 8; i += 8) {
        float32x4x2_t v0 = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i));
        float32x4x2_t v1 = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i + 4));
        float32x4_t sq0 = vmlaq_f32(vmulq_f32(v0.val[0], v0.val[0]), v0.val[1], v0.val[1]);
        float32x4_t sq1 = vmlaq_f32(vmulq_f32(v1.val[0], v1.val[0]), v1.val[1], v1.val[1]);
        vst1q_f32(mag + i,     vsqrtq_f32(sq0));
        vst1q_f32(mag + i + 4, vsqrtq_f32(sq1));
    }
    for (; i <= n - 4; i += 4) {
        float32x4x2_t v = vld2q_f32(reinterpret_cast<const float*>(fft_buf + i));
        float32x4_t sq = vmlaq_f32(vmulq_f32(v.val[0], v.val[0]), v.val[1], v.val[1]);
        vst1q_f32(mag + i, vsqrtq_f32(sq));
    }
    for (; i < n; ++i) mag[i] = std::abs(fft_buf[i]);
}
// ===== END NEON =====

SpectrumResult analyze_spectrum(const std::vector<float>& signal,
                                const FFTPlan& plan, float fs) {
    SpectrumResult res;
    res.magnitudes.resize(SpectrumResult::N_BINS, 0.0f);

    int N = plan.N;
    int start = std::max(0, (int)signal.size() / 2 - N / 2);
    std::vector<Complex> buf(N, {0.f, 0.f});
    int lim = std::min(N, (int)signal.size() - start);

    for (int i = 0; i < lim; ++i) {                         // Hann window
        float w = 0.5f - 0.5f * cosf(TAU * i / (lim - 1));
        buf[i] = { signal[start + i] * w, 0.f };
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
        int bin = (int)((log10f(freq) - log_lo) / log_range * SpectrumResult::N_BINS);
        bin = std::max(0, std::min(SpectrumResult::N_BINS - 1, bin));
        res.magnitudes[bin] = std::max(res.magnitudes[bin], mag[i]);
    }

    float peak = 1e-10f;
    for (float& m : res.magnitudes) {
        m = (m < 1e-10f) ? -80.0f : 20.0f * log10f(m);
        if (m > peak) peak = m;
    }
    res.peak_db = peak;

    // ===== NEON ===== (squared-sum reduction for RMS)
    float32x4_t sq_sum = vdupq_n_f32(0.f);
    int i = 0;
    for (; i <= (int)signal.size() - 4; i += 4) {
        float32x4_t v = vld1q_f32(&signal[i]);
        sq_sum = vmlaq_f32(sq_sum, v, v);
    }
    float rms = neon_hsum(sq_sum);
    // ===== END NEON =====
    for (; i < (int)signal.size(); ++i) rms += signal[i] * signal[i];
    rms = sqrtf(rms / signal.size());
    res.rms_db = (rms < 1e-10f) ? -80.0f : 20.0f * log10f(rms);
    return res;
}

std::string spectrum_to_json(const SpectrumResult& r, const Equalizer& eq) {
    std::ostringstream ss;
    ss << "{\"spectrum\":[";
    for (int i = 0; i < SpectrumResult::N_BINS; ++i)
        ss << (i ? "," : "") << r.magnitudes[i];
    ss << "],\"peak_db\":" << r.peak_db << ",\"rms_db\":" << r.rms_db
       << ",\"bands\":[";
    for (int b = 0; b < Equalizer::N_BANDS; ++b)
        ss << (b ? "," : "")
           << "{\"name\":\"" << eq.bands[b].name << "\""
           << ",\"freq_lo\":" << eq.bands[b].fc * 0.5f
           << ",\"freq_hi\":" << eq.bands[b].fc * 2.0f
           << ",\"gain_db\":" << eq.bands[b].gain_db << "}";
    ss << "]}";
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark — scalar baseline vs NEON, writes benchmark.json
// ─────────────────────────────────────────────────────────────────────────────
struct BenchResult {
    double fir_naive_ms, fir_neon_ms, fir_speedup;     int fir_samples, fir_taps;
    double fft_naive_ms, fft_neon_ms, fft_speedup;     int fft_size, fft_runs;
    double mag_scalar_ms, mag_neon_ms, mag_speedup;    int mag_size;
    double stereo_scalar_ms, stereo_neon_ms, stereo_speedup; int stereo_frames;
    double total_naive_ms, total_neon_ms, total_speedup;
    float  sample_rate;
};

BenchResult benchmark(const std::vector<float>& signal, float fs) {
    std::cout << "======== BENCHMARK ========\n";
    BenchResult r{};
    r.sample_rate = fs;

    int N = std::min((int)signal.size(), (int)(fs * 2));
    std::vector<float> test(signal.begin(), signal.begin() + N);

    // 1) Biquad EQ (stereo): scalar L+R  vs  NEON 2-lane L|R
    Equalizer eq_simple(fs), eq_opt(fs);
    for (int b = 0; b < Equalizer::N_BANDS; ++b) {
        float g = (b % 3 == 0) ? 6.f : -3.f;
        eq_simple.set_gain(b, g);
        eq_opt.set_gain(b, g);
    }
    eq_simple.rebuild_filters();
    eq_opt.rebuild_filters();
    r.fir_samples = N;
    r.fir_taps    = Equalizer::N_BANDS;

    std::vector<float> Ln(N), Rn(N), LRo(2 * N);
    for (int i = 0; i < N; ++i) {
        float l = test[i], rr = test[(i + N / 2) % N];
        Ln[i] = l; Rn[i] = rr;
        LRo[2*i] = l; LRo[2*i + 1] = rr;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    eq_simple.process_block_mono(Ln.data(), Ln.data(), N, 0);
    eq_simple.process_block_mono(Rn.data(), Rn.data(), N, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    eq_opt.process_block_stereo(LRo.data(), N);
    auto t2 = std::chrono::high_resolution_clock::now();

    r.fir_naive_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    r.fir_neon_ms  = std::chrono::duration<double,std::milli>(t2-t1).count();
    r.fir_speedup  = r.fir_naive_ms / std::max(r.fir_neon_ms, 0.001);

    double max_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, (double)fabsf(Ln[i] - LRo[2*i]));
        max_diff = std::max(max_diff, (double)fabsf(Rn[i] - LRo[2*i + 1]));
    }
    std::cout << "Biquad scalar L+R : " << r.fir_naive_ms << " ms\n"
              << "Biquad NEON 2-lane: " << r.fir_neon_ms  << " ms  ("
              << r.fir_speedup << "x)\n"
              << "max|scalar-NEON|  : " << max_diff << "\n";
    assert(max_diff < 1e-2 && "stereo scalar vs NEON must match");

    // 2) FFT: recursive naive vs iterative NEON (equal work, fair average)
    constexpr int FFT_N = 4096, FFT_RUNS = 50;
    r.fft_size = FFT_N; r.fft_runs = FFT_RUNS;
    std::function<void(std::vector<Complex>&)> fft_rec =
        [&](std::vector<Complex>& a) {
            int n = a.size();
            if (n <= 1) return;
            std::vector<Complex> e(n/2), o(n/2);
            for (int i = 0; i < n/2; ++i) { e[i]=a[2*i]; o[i]=a[2*i+1]; }
            fft_rec(e); fft_rec(o);
            float ang = -2.0f*PI/n;
            Complex w(1,0), wn(cosf(ang), sinf(ang));
            for (int i = 0; i < n/2; ++i) {
                Complex t = w*o[i];
                a[i] = e[i]+t; a[i+n/2] = e[i]-t; w *= wn;
            }
        };
    FFTPlan plan_b(FFT_N);
    std::vector<Complex> bufA(FFT_N), bufB(FFT_N);

    auto n0 = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < FFT_RUNS; ++run) {
        for (int i = 0; i < FFT_N; ++i) bufA[i] = { test[i % N], 0.f };
        fft_rec(bufA);
    }
    auto n1 = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < FFT_RUNS; ++run) {
        for (int i = 0; i < FFT_N; ++i) bufB[i] = { test[i % N], 0.f };
        plan_b.forward(bufB);
    }
    auto n2 = std::chrono::high_resolution_clock::now();

    r.fft_naive_ms = std::chrono::duration<double,std::milli>(n1-n0).count() / FFT_RUNS;
    r.fft_neon_ms  = std::chrono::duration<double,std::milli>(n2-n1).count() / FFT_RUNS;
    r.fft_speedup  = r.fft_naive_ms / std::max(r.fft_neon_ms, 0.001);
    std::cout << "FFT naive : " << r.fft_naive_ms << " ms\n"
              << "FFT NEON  : " << r.fft_neon_ms  << " ms  ("
              << r.fft_speedup << "x)\n";

    // 3) Magnitude: scalar vs NEON
    constexpr int MAG_N = 65536;
    r.mag_size = MAG_N;
    std::vector<Complex> mb(MAG_N);
    std::vector<float>   mo(MAG_N);
    for (int i = 0; i < MAG_N; ++i) mb[i] = { i*0.001f, i*0.0007f };

    auto s0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MAG_N; ++i) mo[i] = std::abs(mb[i]);
    auto s1 = std::chrono::high_resolution_clock::now();
    neon_magnitude(mb.data(), mo.data(), MAG_N);
    auto s2 = std::chrono::high_resolution_clock::now();

    r.mag_scalar_ms = std::chrono::duration<double,std::milli>(s1-s0).count();
    r.mag_neon_ms   = std::chrono::duration<double,std::milli>(s2-s1).count();
    r.mag_speedup   = r.mag_scalar_ms / std::max(r.mag_neon_ms, 0.0001);
    std::cout << "Magnitude : " << r.mag_scalar_ms << " -> " << r.mag_neon_ms
              << " ms  (" << r.mag_speedup << "x)\n";

    // 4) Stereo I/O: scalar vs NEON (bandwidth-bound; ~1.0x expected)
    constexpr int ST = 88200;
    r.stereo_frames = ST;
    std::vector<float> L(ST), R(ST), I1(ST*2), I2(ST*2);
    for (int i = 0; i < ST; ++i) { L[i] = test[i%N]; R[i] = test[(i+N/2)%N]; }

    auto p0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ST; ++i) { I1[i*2]=L[i]; I1[i*2+1]=R[i]; }
    for (int i = 0; i < ST; ++i) { L[i]=I1[i*2]; R[i]=I1[i*2+1]; }
    auto p1 = std::chrono::high_resolution_clock::now();
    neon_interleave(L.data(), R.data(), I2.data(), ST);
    neon_deinterleave(I2.data(), L.data(), R.data(), ST);
    auto p2 = std::chrono::high_resolution_clock::now();

    r.stereo_scalar_ms = std::chrono::duration<double,std::milli>(p1-p0).count();
    r.stereo_neon_ms   = std::chrono::duration<double,std::milli>(p2-p1).count();
    r.stereo_speedup   = r.stereo_scalar_ms / std::max(r.stereo_neon_ms, 0.0001);
    std::cout << "Stereo I/O: " << r.stereo_scalar_ms << " -> " << r.stereo_neon_ms
              << " ms  (" << r.stereo_speedup << "x)\n";

    r.total_naive_ms = r.fir_naive_ms + r.fft_naive_ms + r.mag_scalar_ms + r.stereo_scalar_ms;
    r.total_neon_ms  = r.fir_neon_ms  + r.fft_neon_ms  + r.mag_neon_ms   + r.stereo_neon_ms;
    r.total_speedup  = r.total_naive_ms / std::max(r.total_neon_ms, 0.001);
    std::cout << "TOTAL " << r.total_naive_ms << " -> " << r.total_neon_ms
              << " ms  (" << r.total_speedup << "x)\n";

    std::ostringstream js;
    js << std::fixed; js.precision(3);
    js << "{\"fir\":{\"naive_ms\":" << r.fir_naive_ms << ",\"neon_ms\":" << r.fir_neon_ms
       << ",\"speedup\":" << r.fir_speedup << ",\"samples\":" << r.fir_samples
       << ",\"taps\":" << r.fir_taps << "},"
       << "\"fft\":{\"naive_ms\":" << r.fft_naive_ms << ",\"neon_ms\":" << r.fft_neon_ms
       << ",\"speedup\":" << r.fft_speedup << ",\"size\":" << r.fft_size
       << ",\"runs\":" << r.fft_runs << "},"
       << "\"magnitude\":{\"naive_ms\":" << r.mag_scalar_ms << ",\"neon_ms\":" << r.mag_neon_ms
       << ",\"speedup\":" << r.mag_speedup << ",\"size\":" << r.mag_size << "},"
       << "\"stereo\":{\"naive_ms\":" << r.stereo_scalar_ms << ",\"neon_ms\":" << r.stereo_neon_ms
       << ",\"speedup\":" << r.stereo_speedup << ",\"frames\":" << r.stereo_frames << "},"
       << "\"total\":{\"naive_ms\":" << r.total_naive_ms << ",\"neon_ms\":" << r.total_neon_ms
       << ",\"speedup\":" << r.total_speedup << "},"
       << "\"sample_rate\":" << (int)r.sample_rate << "}";
    std::ofstream("benchmark.json") << js.str();
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunked streaming WAV processor (IIR state carries across chunks)
// ─────────────────────────────────────────────────────────────────────────────
bool process_streaming(const std::string& in_path, const std::string& out_path,
                       Equalizer& eq, const WavHeader& hdr) {
    static constexpr int CHUNK = 65536;
    const int CH  = hdr.channels;
    const int bps = hdr.bit_depth / 8;
    const int total_frames = hdr.data_bytes / (bps * CH);

    std::ifstream fin(in_path, std::ios::binary);
    std::ofstream fout(out_path, std::ios::binary);
    if (!fin || !fout) { std::cerr << "[ERR] file open\n"; return false; }
    fin.seekg(sizeof(WavHeader), std::ios::beg);

    WavHeader oh = hdr;
    oh.data_bytes = total_frames * CH * bps;
    oh.wav_size   = oh.data_bytes + 36;
    fout.write(reinterpret_cast<const char*>(&oh), sizeof(oh));

    std::vector<int16_t> raw(CHUNK * CH);
    std::vector<float>   chL(CHUNK);
    std::vector<float>   chLR(CHUNK * 2);
    std::vector<int16_t> out_raw(CHUNK * CH);

    // ===== NEON ===== (int16<->float convert/clamp constants)
    const float32x4_t SC_IN  = vdupq_n_f32(1.0f / 32768.0f);
    const float32x4_t SC_OUT = vdupq_n_f32(32767.0f);
    const float32x4_t CL_HI  = vdupq_n_f32( 1.0f);
    const float32x4_t CL_LO  = vdupq_n_f32(-1.0f);
    // ===== END NEON =====

    int done = 0;
    while (done < total_frames) {
        int now = std::min(CHUNK, total_frames - done);
        fin.read(reinterpret_cast<char*>(raw.data()), now * CH * bps);

        if (CH == 1) {
            int i = 0;
            // ===== NEON ===== int16 -> float (8/iter)
            for (; i <= now - 8; i += 8) {
                int16x8_t s = vld1q_s16(raw.data() + i);
                vst1q_f32(chL.data()+i,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))),  SC_IN));
                vst1q_f32(chL.data()+i+4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))), SC_IN));
            }
            // ===== END NEON =====
            for (; i < now; ++i) chL[i] = raw[i] * (1.0f / 32768.0f);

            eq.process_block_mono(chL.data(), chL.data(), now, 0);

            i = 0;
            // ===== NEON ===== float -> int16 (8/iter, clamped)
            for (; i <= now - 8; i += 8) {
                float32x4_t f0 = vld1q_f32(chL.data() + i);
                float32x4_t f1 = vld1q_f32(chL.data() + i + 4);
                f0 = vmulq_f32(vmaxq_f32(CL_LO, vminq_f32(CL_HI, f0)), SC_OUT);
                f1 = vmulq_f32(vmaxq_f32(CL_LO, vminq_f32(CL_HI, f1)), SC_OUT);
                vst1q_s16(out_raw.data() + i,
                          vcombine_s16(vmovn_s32(vcvtq_s32_f32(f0)),
                                       vmovn_s32(vcvtq_s32_f32(f1))));
            }
            // ===== END NEON =====
            for (; i < now; ++i) {
                float v = std::max(-1.f, std::min(1.f, chL[i]));
                out_raw[i] = (int16_t)(v * 32767.f);
            }
        } else {
            const int tot = now * 2;                        // interleaved
            int i = 0;
            // ===== NEON ===== int16 -> float, flat (8/iter)
            for (; i <= tot - 8; i += 8) {
                int16x8_t s = vld1q_s16(raw.data() + i);
                vst1q_f32(chLR.data()+i,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))),  SC_IN));
                vst1q_f32(chLR.data()+i+4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))), SC_IN));
            }
            // ===== END NEON =====
            for (; i < tot; ++i) chLR[i] = raw[i] * (1.0f / 32768.0f);

            eq.process_block_stereo(chLR.data(), now);      // NEON 2-lane

            i = 0;
            // ===== NEON ===== float -> int16, flat (8/iter, clamped)
            for (; i <= tot - 8; i += 8) {
                float32x4_t f0 = vld1q_f32(chLR.data() + i);
                float32x4_t f1 = vld1q_f32(chLR.data() + i + 4);
                f0 = vmulq_f32(vmaxq_f32(CL_LO, vminq_f32(CL_HI, f0)), SC_OUT);
                f1 = vmulq_f32(vmaxq_f32(CL_LO, vminq_f32(CL_HI, f1)), SC_OUT);
                vst1q_s16(out_raw.data() + i,
                          vcombine_s16(vmovn_s32(vcvtq_s32_f32(f0)),
                                       vmovn_s32(vcvtq_s32_f32(f1))));
            }
            // ===== END NEON =====
            for (; i < tot; ++i) {
                float v = std::max(-1.f, std::min(1.f, chLR[i]));
                out_raw[i] = (int16_t)(v * 32767.f);
            }
        }

        fout.write(reinterpret_cast<const char*>(out_raw.data()), now * CH * bps);
        done += now;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string in_path  = (argc > 1) ? argv[1] : "test_audio.wav";
    std::string out_path = (argc > 2) ? argv[2] : "audio_eq.wav";

    float gains[10] = {};
    for (int i = 3; i < argc && i - 3 < 10; ++i)
        gains[i - 3] = std::stof(argv[i]);

    WavHeader hdr;
    {
        std::ifstream f(in_path, std::ios::binary);
        if (!f) { std::cerr << "[ERR] cannot open " << in_path << "\n"; return 1; }
        f.read(reinterpret_cast<char*>(&hdr), 36);
        char id[5] = {}; uint32_t sz;
        while (f.read(id, 4) && f.read(reinterpret_cast<char*>(&sz), 4)) {
            if (!strncmp(id, "data", 4)) { memcpy(hdr.data, "data", 4); hdr.data_bytes = sz; break; }
            f.seekg(sz, std::ios::cur);
        }
    }
    if (!hdr.data_bytes) { std::cerr << "[ERR] no data chunk\n"; return 1; }

    float fs = hdr.sample_rate;
    int total_frames = hdr.data_bytes / (hdr.bit_depth/8) / hdr.channels;
    std::cout << "[INFO] " << in_path << "  " << fs << " Hz, "
              << hdr.channels << "ch, " << total_frames << " frames\n";

    FFTPlan plan(4096);
    Equalizer eq(fs);
    for (int b = 0; b < 10; ++b) eq.set_gain(b, gains[b]);
    eq.rebuild_filters();

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!process_streaming(in_path, out_path, eq, hdr)) return 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[PERF] processing "
              << std::chrono::duration<double,std::milli>(t1-t0).count()/1000.0
              << " s\n";

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
            if (hdr.channels == 2) f.seekg(2, std::ios::cur);
        }
        return buf;
    };

    int mid = total_frames / 2;
    auto spec_before = analyze_spectrum(read_window(in_path,  mid), plan, fs);
    auto spec_after  = analyze_spectrum(read_window(out_path, mid), plan, fs);
    {
        std::ofstream jf("spectrum.json");
        jf << "{\"before\":" << spectrum_to_json(spec_before, eq)
           << ",\"after\":"  << spectrum_to_json(spec_after,  eq) << "}";
    }

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
    benchmark(bench_buf, fs);

    std::cout << "[DONE]\n";
    return 0;
}
