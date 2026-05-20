// DSP Audio EQ — OPTIMIZED (ARM NEON) build
// 10-band parametric equalizer + FFT spectrum, Cortex-A53 / RPi Zero 2W.
// Complete standalone program (the scalar counterpart is eq_naive.cpp).
//
// NEON regions are marked:  // ===== NEON =====  ...  // ===== END NEON =====
//
// Build: g++ -O3 -ffast-math -mcpu=cortex-a53 -funroll-loops -std=c++17 \
//            eq_neon.cpp -o eq_neon
// Run  : ./eq_neon in.wav out.wav [g0 g1 ... g9]   (gains in dB)

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
#include <arm_neon.h>

typedef std::complex<float> Complex;
static constexpr float PI  = 3.14159265358979323846f;
static constexpr float TAU = 6.28318530717958647692f;

// ─── WAV header ──────────────────────────────────────────────────────────────
#pragma pack(push, 1)
struct WavHeader {
    char riff[4]; uint32_t wav_size; char wave[4]; char fmt[4];
    uint32_t fmt_size; uint16_t audio_format; uint16_t channels;
    uint32_t sample_rate; uint32_t byte_rate; uint16_t block_align;
    uint16_t bit_depth; char data[4]; uint32_t data_bytes;
};
#pragma pack(pop)

// ===== NEON =====
inline float neon_hsum(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}
// ===== END NEON =====

// ─── FFT — iterative radix-2, precomputed twiddles + bit-reversal, NEON ──────
struct FFTPlan {
    int                  N;
    std::vector<Complex> twiddle;
    std::vector<int>     rev;

    explicit FFTPlan(int n) : N(n), twiddle(n/2), rev(n) {
        assert((n & (n-1)) == 0 && "FFT size must be a power of 2");
        for (int k = 0; k < n/2; ++k)
            twiddle[k] = { cosf(TAU*k/n), -sinf(TAU*k/n) };
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
            int half = len >> 1, stride = N / len;
            for (int i = 0; i < N; i += len) {
                int j = 0;
                // ===== NEON ===== (4 independent butterflies / iteration)
                for (; j <= half - 4; j += 4) {
                    float32x4x2_t U = vld2q_f32(reinterpret_cast<const float*>(&a[i+j]));
                    float32x4_t U_re = U.val[0], U_im = U.val[1];
                    float32x4x2_t V = vld2q_f32(reinterpret_cast<const float*>(&a[i+j+half]));
                    float32x4_t V_re = V.val[0], V_im = V.val[1];

                    float wr[4], wi[4];
                    for (int k = 0; k < 4; ++k) {
                        wr[k] = twiddle[(j+k)*stride].real();
                        wi[k] = twiddle[(j+k)*stride].imag();
                    }
                    float32x4_t W_re = vld1q_f32(wr), W_im = vld1q_f32(wi);

                    float32x4_t T_re = vmlsq_f32(vmulq_f32(W_re, V_re), W_im, V_im);
                    float32x4_t T_im = vmlaq_f32(vmulq_f32(W_re, V_im), W_im, V_re);

                    float32x4x2_t S = { vaddq_f32(U_re, T_re), vaddq_f32(U_im, T_im) };
                    float32x4x2_t D = { vsubq_f32(U_re, T_re), vsubq_f32(U_im, T_im) };
                    vst2q_f32(reinterpret_cast<float*>(&a[i+j]),        S);
                    vst2q_f32(reinterpret_cast<float*>(&a[i+j+half]),   D);
                }
                // ===== END NEON =====
                for (; j < half; ++j) {
                    Complex w = twiddle[j*stride];
                    Complex u = a[i+j];
                    Complex v = a[i+j+half] * w;
                    a[i+j]      = u + v;
                    a[i+j+half] = u - v;
                }
            }
        }
    }
};

// ─── Biquad (TDF2) + RBJ cookbook coefficients ───────────────────────────────
struct BiquadCoeffs { float b0, b1, b2, a1, a2; };
struct BiquadState  { float s1 = 0.f, s2 = 0.f; };

BiquadCoeffs biquad_peaking(float fc, float g, float Q, float fs) {
    float A = powf(10.f, g/40.f), w0 = TAU*fc/fs, cw = cosf(w0), sw = sinf(w0);
    float al = sw / (2.f*Q), inv = 1.f / (1.f + al/A);
    return { (1.f+al*A)*inv, (-2.f*cw)*inv, (1.f-al*A)*inv,
             (-2.f*cw)*inv,  (1.f-al/A)*inv };
}
BiquadCoeffs biquad_low_shelf(float fc, float g, float fs) {
    float A = powf(10.f, g/40.f), w0 = TAU*fc/fs, cw = cosf(w0), sw = sinf(w0);
    float al = sw/2.f * sqrtf((A+1.f/A)*(1.f/0.707f-1.f)+2.f);
    float inv = 1.f / ((A+1)+(A-1)*cw+2*sqrtf(A)*al);
    return { A*((A+1)-(A-1)*cw+2*sqrtf(A)*al)*inv,
             2.f*A*((A-1)-(A+1)*cw)*inv,
             A*((A+1)-(A-1)*cw-2*sqrtf(A)*al)*inv,
             -2.f*((A-1)+(A+1)*cw)*inv,
             ((A+1)+(A-1)*cw-2*sqrtf(A)*al)*inv };
}
BiquadCoeffs biquad_high_shelf(float fc, float g, float fs) {
    float A = powf(10.f, g/40.f), w0 = TAU*fc/fs, cw = cosf(w0), sw = sinf(w0);
    float al = sw/2.f * sqrtf((A+1.f/A)*(1.f/0.707f-1.f)+2.f);
    float inv = 1.f / ((A+1)-(A-1)*cw+2*sqrtf(A)*al);
    return { A*((A+1)+(A-1)*cw+2*sqrtf(A)*al)*inv,
             -2.f*A*((A-1)+(A+1)*cw)*inv,
             A*((A+1)+(A-1)*cw-2*sqrtf(A)*al)*inv,
             2.f*((A-1)-(A+1)*cw)*inv,
             ((A+1)-(A-1)*cw-2*sqrtf(A)*al)*inv };
}

// ─── 10-band parametric EQ ───────────────────────────────────────────────────
//   mono   : band-major scalar (optimal scalar form for a serial IIR)
//   stereo : NEON 2-lane (L|R) + band-pair fusion (halves buffer passes)
struct EQBandDef {
    const char* name; float fc; float Q;
    enum Type { LOW_SHELF, PEAK, HIGH_SHELF } type; float gain_db;
};
static EQBandDef DEFAULT_BANDS[10] = {
    {"Sub-Bass",    60.f, 0.7f, EQBandDef::LOW_SHELF,  0.f},
    {"Bass",       170.f, 1.0f, EQBandDef::PEAK,       0.f},
    {"Low-Mid",    350.f, 1.0f, EQBandDef::PEAK,       0.f},
    {"Mid",        700.f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Upper-Mid", 1400.f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Presence",  3000.f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Brilliance",6000.f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Air",      10000.f, 1.4f, EQBandDef::PEAK,       0.f},
    {"Ultra",    14000.f, 1.4f, EQBandDef::PEAK,       0.f},
    {"High",     18000.f, 0.7f, EQBandDef::HIGH_SHELF, 0.f},
};

class Equalizer {
public:
    static constexpr int N_BANDS = 10;
    EQBandDef    bands[N_BANDS];
    float        sample_rate;
    BiquadCoeffs coeffs[N_BANDS];
    BiquadState  state[2][N_BANDS];

    explicit Equalizer(float fs) : sample_rate(fs) {
        memcpy(bands, DEFAULT_BANDS, sizeof(DEFAULT_BANDS));
        memset(state, 0, sizeof(state));
        rebuild_filters();
    }
    void set_gain(int b, float db) {
        bands[b].gain_db = std::max(-24.f, std::min(24.f, db));
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

    void process_mono(float* buf, int n, int ch) {
        for (int b = 0; b < N_BANDS; ++b) {
            if (fabsf(bands[b].gain_db) < 0.01f) continue;
            const float b0 = coeffs[b].b0, b1 = coeffs[b].b1, b2 = coeffs[b].b2;
            const float a1 = coeffs[b].a1, a2 = coeffs[b].a2;
            float s1 = state[ch][b].s1, s2 = state[ch][b].s2;
            for (int i = 0; i < n; ++i) {
                float x  = buf[i];
                float y  = b0 * x + s1;
                float n1 = b1 * x - a1 * y + s2;
                float n2 = b2 * x - a2 * y;
                s1 = n1; s2 = n2;
                buf[i] = y;
            }
            state[ch][b].s1 = s1; state[ch][b].s2 = s2;
        }
    }

    // ===== NEON ===== (lane0 = L, lane1 = R; band-pair fused; interleaved lr)
    void process_stereo(float* __restrict__ lr, int frames) {
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
        if (k < na) {
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

// ─── Spectrum analysis ───────────────────────────────────────────────────────
struct SpectrumResult {
    std::vector<float> magnitudes; float peak_db; float rms_db;
    static constexpr int N_BINS = 64;
};

// ===== NEON ===== (re^2+im^2 fused mul-add; 4-wide HW sqrt; 2x unroll)
void neon_magnitude(const Complex* fb, float* mag, int n) {
    int i = 0;
    for (; i <= n - 8; i += 8) {
        float32x4x2_t v0 = vld2q_f32(reinterpret_cast<const float*>(fb + i));
        float32x4x2_t v1 = vld2q_f32(reinterpret_cast<const float*>(fb + i + 4));
        float32x4_t s0 = vmlaq_f32(vmulq_f32(v0.val[0], v0.val[0]), v0.val[1], v0.val[1]);
        float32x4_t s1 = vmlaq_f32(vmulq_f32(v1.val[0], v1.val[0]), v1.val[1], v1.val[1]);
        vst1q_f32(mag + i,     vsqrtq_f32(s0));
        vst1q_f32(mag + i + 4, vsqrtq_f32(s1));
    }
    for (; i <= n - 4; i += 4) {
        float32x4x2_t v = vld2q_f32(reinterpret_cast<const float*>(fb + i));
        float32x4_t sq = vmlaq_f32(vmulq_f32(v.val[0], v.val[0]), v.val[1], v.val[1]);
        vst1q_f32(mag + i, vsqrtq_f32(sq));
    }
    for (; i < n; ++i) mag[i] = std::abs(fb[i]);
}
// ===== END NEON =====

SpectrumResult analyze_spectrum(const std::vector<float>& signal,
                                const FFTPlan& plan, float fs) {
    SpectrumResult res;
    res.magnitudes.resize(SpectrumResult::N_BINS, 0.f);

    int N = plan.N;
    int start = std::max(0, (int)signal.size() / 2 - N / 2);
    std::vector<Complex> buf(N, {0.f, 0.f});
    int lim = std::min(N, (int)signal.size() - start);
    for (int i = 0; i < lim; ++i) {
        float w = 0.5f - 0.5f * cosf(TAU * i / (lim - 1));
        buf[i] = { signal[start + i] * w, 0.f };
    }

    plan.forward(buf);

    int half = N / 2;
    std::vector<float> mag(half);
    neon_magnitude(buf.data(), mag.data(), half);

    float log_lo = log10f(20.f), log_hi = log10f(fs / 2.f), lr = log_hi - log_lo;
    for (int i = 0; i < half; ++i) {
        float freq = (float)i * fs / N;
        if (freq < 20.f) continue;
        int bin = (int)((log10f(freq) - log_lo) / lr * SpectrumResult::N_BINS);
        bin = std::max(0, std::min(SpectrumResult::N_BINS - 1, bin));
        res.magnitudes[bin] = std::max(res.magnitudes[bin], mag[i]);
    }
    float peak = 1e-10f;
    for (float& m : res.magnitudes) {
        m = (m < 1e-10f) ? -80.f : 20.f * log10f(m);
        if (m > peak) peak = m;
    }
    res.peak_db = peak;

    // ===== NEON ===== (squared-sum reduction for RMS)
    float32x4_t acc = vdupq_n_f32(0.f);
    int i = 0;
    for (; i <= (int)signal.size() - 4; i += 4) {
        float32x4_t v = vld1q_f32(&signal[i]);
        acc = vmlaq_f32(acc, v, v);
    }
    float rms = neon_hsum(acc);
    // ===== END NEON =====
    for (; i < (int)signal.size(); ++i) rms += signal[i] * signal[i];
    rms = sqrtf(rms / signal.size());
    res.rms_db = (rms < 1e-10f) ? -80.f : 20.f * log10f(rms);
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

// ─── Chunked streaming WAV processor (NEON) ──────────────────────────────────
bool process_streaming(const std::string& in_path, const std::string& out_path,
                       Equalizer& eq, const WavHeader& hdr) {
    // Cache-blocking: working set touched on the stereo path is
    // raw + chLR + out_raw ≈ CHUNK·16 bytes. At 65536 that was 1 MB
    // (≫ 512 KB L2 → every band pass hit RAM). 8192 → 128 KB, fits L2
    // with headroom: measured −10% DSP time on RPi Zero 2W (statistically
    // flat 8192–16384; 8192 chosen for lowest run-to-run variance).
    static constexpr int CHUNK = 8192;
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

    // ===== NEON ===== (convert/clamp constants)
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
            // ===== NEON ===== int16 -> float
            for (; i <= now - 8; i += 8) {
                int16x8_t s = vld1q_s16(raw.data() + i);
                vst1q_f32(chL.data()+i,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))),  SC_IN));
                vst1q_f32(chL.data()+i+4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))), SC_IN));
            }
            // ===== END NEON =====
            for (; i < now; ++i) chL[i] = raw[i] * (1.0f / 32768.0f);

            eq.process_mono(chL.data(), now, 0);

            i = 0;
            // ===== NEON ===== float -> int16 (clamped)
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
            const int tot = now * 2;
            int i = 0;
            // ===== NEON ===== int16 -> float (flat, interleaved)
            for (; i <= tot - 8; i += 8) {
                int16x8_t s = vld1q_s16(raw.data() + i);
                vst1q_f32(chLR.data()+i,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))),  SC_IN));
                vst1q_f32(chLR.data()+i+4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))), SC_IN));
            }
            // ===== END NEON =====
            for (; i < tot; ++i) chLR[i] = raw[i] * (1.0f / 32768.0f);

            eq.process_stereo(chLR.data(), now);          // NEON 2-lane

            i = 0;
            // ===== NEON ===== float -> int16 (flat, clamped)
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

// ─── main ────────────────────────────────────────────────────────────────────
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
    std::cout << "[NEON] " << in_path << "  " << fs << " Hz, "
              << hdr.channels << "ch, " << total_frames << " frames\n";

    FFTPlan plan(4096);
    Equalizer eq(fs);
    for (int b = 0; b < 10; ++b) eq.set_gain(b, gains[b]);
    eq.rebuild_filters();

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!process_streaming(in_path, out_path, eq, hdr)) return 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[PERF] " << std::chrono::duration<double,std::milli>(t1-t0).count()/1000.0
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
    auto sb = analyze_spectrum(read_window(in_path,  mid), plan, fs);
    auto sa = analyze_spectrum(read_window(out_path, mid), plan, fs);
    std::ofstream("spectrum.json")
        << "{\"before\":" << spectrum_to_json(sb, eq)
        << ",\"after\":"  << spectrum_to_json(sa, eq) << "}";

    std::cout << "[DONE]\n";
    return 0;
}
