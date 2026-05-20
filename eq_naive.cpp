// DSP Audio EQ — NAIVE (pure scalar) build
// 10-band parametric equalizer + FFT spectrum. No NEON, no SIMD.
// Complete standalone program (the optimized counterpart is eq_neon.cpp).
//
// Build: g++ -O3 -ffast-math -std=c++17 eq_naive.cpp -o eq_naive
// Run  : ./eq_naive in.wav out.wav [g0 g1 ... g9]   (gains in dB)

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

// ─── FFT — naive recursive radix-2 Cooley-Tukey ──────────────────────────────
void fft_naive(std::vector<Complex>& a) {
    int n = (int)a.size();
    if (n <= 1) return;
    std::vector<Complex> even(n/2), odd(n/2);
    for (int i = 0; i < n/2; ++i) { even[i] = a[2*i]; odd[i] = a[2*i+1]; }
    fft_naive(even);
    fft_naive(odd);
    float ang = -2.0f * PI / n;
    Complex w(1.0f, 0.0f), wn(cosf(ang), sinf(ang));
    for (int i = 0; i < n/2; ++i) {
        Complex t = w * odd[i];
        a[i]       = even[i] + t;
        a[i + n/2] = even[i] - t;
        w *= wn;
    }
}

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

// ─── 10-band parametric EQ — scalar series cascade, band-major ───────────────
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
    // One channel, scalar series cascade, in place. 0 dB bands skipped.
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
};

// ─── Spectrum analysis (scalar) ──────────────────────────────────────────────
struct SpectrumResult {
    std::vector<float> magnitudes; float peak_db; float rms_db;
    static constexpr int N_BINS = 64;
};

SpectrumResult analyze_spectrum(const std::vector<float>& signal, int N, float fs) {
    SpectrumResult res;
    res.magnitudes.resize(SpectrumResult::N_BINS, 0.f);

    int start = std::max(0, (int)signal.size() / 2 - N / 2);
    std::vector<Complex> buf(N, {0.f, 0.f});
    int lim = std::min(N, (int)signal.size() - start);
    for (int i = 0; i < lim; ++i) {
        float w = 0.5f - 0.5f * cosf(TAU * i / (lim - 1));    // Hann window
        buf[i] = { signal[start + i] * w, 0.f };
    }

    fft_naive(buf);

    int half = N / 2;
    std::vector<float> mag(half);
    for (int i = 0; i < half; ++i) mag[i] = std::abs(buf[i]);  // scalar sqrt

    float log_lo = log10f(20.f), log_hi = log10f(fs / 2.f);
    float lr = log_hi - log_lo;
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

    float rms = 0.f;                                          // scalar RMS
    for (float v : signal) rms += v * v;
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

// ─── Chunked streaming WAV processor (scalar) ────────────────────────────────
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
    std::vector<float>   chL(CHUNK), chR(CHUNK);
    std::vector<int16_t> out_raw(CHUNK * CH);

    int done = 0;
    while (done < total_frames) {
        int now = std::min(CHUNK, total_frames - done);
        fin.read(reinterpret_cast<char*>(raw.data()), now * CH * bps);

        if (CH == 1) {
            for (int i = 0; i < now; ++i)                     // int16 -> float
                chL[i] = raw[i] * (1.0f / 32768.0f);
            eq.process_mono(chL.data(), now, 0);
            for (int i = 0; i < now; ++i) {                   // float -> int16
                float v = std::max(-1.f, std::min(1.f, chL[i]));
                out_raw[i] = (int16_t)(v * 32767.f);
            }
        } else {
            for (int i = 0; i < now; ++i) {                   // deinterleave
                chL[i] = raw[i*2]     * (1.0f / 32768.0f);
                chR[i] = raw[i*2 + 1] * (1.0f / 32768.0f);
            }
            eq.process_mono(chL.data(), now, 0);              // L then R
            eq.process_mono(chR.data(), now, 1);
            for (int i = 0; i < now; ++i) {                   // interleave + cvt
                float l = std::max(-1.f, std::min(1.f, chL[i]));
                float rr = std::max(-1.f, std::min(1.f, chR[i]));
                out_raw[i*2]     = (int16_t)(l  * 32767.f);
                out_raw[i*2 + 1] = (int16_t)(rr * 32767.f);
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
    std::cout << "[NAIVE] " << in_path << "  " << fs << " Hz, "
              << hdr.channels << "ch, " << total_frames << " frames\n";

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
    auto sb = analyze_spectrum(read_window(in_path,  mid), 4096, fs);
    auto sa = analyze_spectrum(read_window(out_path, mid), 4096, fs);
    std::ofstream("spectrum.json")
        << "{\"before\":" << spectrum_to_json(sb, eq)
        << ",\"after\":"  << spectrum_to_json(sa, eq) << "}";

    std::cout << "[DONE]\n";
    return 0;
}
