#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <arm_neon.h>

// ==========================================
// DECLARAȚII DE BAZĂ (32-bit float)
// ==========================================
typedef std::complex<float> Complex;
const float PI = 3.14159265358979323846f;

#pragma pack(push, 1)
struct WavHeader {
    char riff_header[4];
    uint32_t wav_size;
    char wave_header[4];
    char fmt_header[4];
    uint32_t fmt_chunk_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t sample_alignment;
    uint16_t bit_depth;
    char data_header[4];
    uint32_t data_bytes;
};
#pragma pack(pop)

// ==========================================
// MODULUL I/O: CITIRE ȘI SCRIERE WAV
// ==========================================
std::vector<float> citeste_wav(const std::string& nume_fisier, WavHeader& header_salvat) {
    std::ifstream fisier(nume_fisier, std::ios::binary);
    std::vector<float> semnal_audio;
    if (!fisier.is_open()) return semnal_audio;

    fisier.read(reinterpret_cast<char*>(&header_salvat), 36);
    char chunk_id[5] = {0};
    uint32_t chunk_size;
    bool data_gasit = false;

    while (fisier.read(chunk_id, 4)) {
        fisier.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (std::strncmp(chunk_id, "data", 4) == 0) {
            data_gasit = true;
            std::memcpy(header_salvat.data_header, "data", 4);
            header_salvat.data_bytes = chunk_size;
            break;
        } else {
            fisier.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!data_gasit) return semnal_audio;

    int num_esantioane = header_salvat.data_bytes / (header_salvat.bit_depth / 8);
    semnal_audio.reserve(num_esantioane);
    int16_t esantion_brut;
    for (int i = 0; i < num_esantioane; i++) {
        if (fisier.read(reinterpret_cast<char*>(&esantion_brut), sizeof(int16_t))) {
            semnal_audio.push_back(static_cast<float>(esantion_brut) / 32768.0f);
        } else break;
    }

    fisier.close();
    header_salvat.data_bytes = semnal_audio.size() * (header_salvat.bit_depth / 8);
    return semnal_audio;
}

void scrie_wav(const std::string& nume_fisier, const std::vector<float>& semnal_audio, WavHeader header) {
    std::ofstream fisier(nume_fisier, std::ios::binary);
    if (!fisier.is_open()) return;

    header.data_bytes = semnal_audio.size() * (header.bit_depth / 8);
    header.wav_size = header.data_bytes + 36;
    std::memcpy(header.data_header, "data", 4);
    fisier.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));

    for (size_t i = 0; i < semnal_audio.size(); i++) {
        float valoare = semnal_audio[i];
        if (valoare > 1.0f) valoare = 1.0f;
        if (valoare < -1.0f) valoare = -1.0f;
        int16_t esantion_final = static_cast<int16_t>(valoare * 32767.0f);
        fisier.write(reinterpret_cast<const char*>(&esantion_final), sizeof(int16_t));
    }
    fisier.close();
    std::cout << "[INFO] Fisier salvat cu succes: " << nume_fisier << std::endl;
}

// ==========================================
// MODULUL FILTER DESIGN (Matematica FIR)
// ==========================================
std::vector<float> genereaza_filtru_trece_jos(float frecventa_taiere, float sample_rate, int numar_coeficienti) {
    std::vector<float> coeficienti(numar_coeficienti, 0.0f);
    
    // Frecvența normalizată
    float ft = frecventa_taiere / sample_rate;
    int M = numar_coeficienti - 1;
    float suma = 0.0f;

    for (int n = 0; n < numar_coeficienti; n++) {
        // Calculăm funcția ideală Sinc
        if (n == M / 2) {
            coeficienti[n] = 2.0f * ft;
        } else {
            float x = PI * (n - M / 2.0f);
            coeficienti[n] = sinf(2.0f * PI * ft * (n - M / 2.0f)) / x;
        }

        // Aplicăm Fereastra Hamming pentru a elimina distorsiunile
        float fereastra_hamming = 0.54f - 0.46f * cosf((2.0f * PI * n) / M);
        coeficienti[n] *= fereastra_hamming;

        suma += coeficienti[n];
    }

    // Normalizăm coeficienții (ca să nu modificăm volumul general al piesei)
    for (int n = 0; n < numar_coeficienti; n++) {
        coeficienti[n] /= suma;
    }

    return coeficienti;
}

// ==========================================
// MODULUL DSP (Funcții cu memorie optimizată)
// ==========================================
void fft_naiv(std::vector<Complex>& a) {
    int n = a.size();
    if (n <= 1) return;
    std::vector<Complex> pare(n / 2), impare(n / 2);
    for (int i = 0; i < n / 2; i++) {
        pare[i] = a[i * 2];
        impare[i] = a[i * 2 + 1];
    }
    fft_naiv(pare);
    fft_naiv(impare);
    float angle = -2.0f * PI / n;
    Complex w(1.0f, 0.0f), wn(cosf(angle), sinf(angle));
    for (int i = 0; i < n / 2; i++) {
        a[i] = pare[i] + w * impare[i];
        a[i + n / 2] = pare[i] - w * impare[i];
        w *= wn;
    }
}

void fir_naiv(const std::vector<float>& intrare, const std::vector<float>& coef, std::vector<float>& iesire) {
    int n_semnal = intrare.size();
    int n_coeficienti = coef.size();
    std::fill(iesire.begin(), iesire.end(), 0.0f);

    for (int i = 0; i < n_semnal; i++) {
        for (int j = 0; j < n_coeficienti; j++) {
            if (i - j >= 0) {
                iesire[i] += intrare[i - j] * coef[j];
            }
        }
    }
}

// ==========================================
// FFT ITERATIV & VIZUALIZARE ASCII
// ==========================================

// Bit-Reversal
uint32_t reverse_bits(uint32_t x, int n) {
    uint32_t res = 0;
    for (int i = 0; i < n; i++) {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

// FFT Iterativ (Cooley-Tukey) - Structura pregătită pentru NEON
void fft_neon(std::vector<Complex>& a) {
    int n = a.size();
    int log_n = log2(n);

    // 1. Permutarea datelor
    for (int i = 0; i < n; i++) {
        int j = reverse_bits(i, log_n);
        if (i < j) std::swap(a[i], a[j]);
    }

    // 2. Calculul propriu-zis (fara recursivitate)
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * PI / len;
        Complex wlen(cosf(angle), sinf(angle));
        
        for (int i = 0; i < n; i += len) {
            Complex w(1, 0);
            for (int j = 0; j < len / 2; j++) {
                //Aici 'asezam' baza pentru vectorizarea fluturelui 
                //cu float32x4_t (vld1q_f32, vmulq_f32) in etapa de rafinare extrema.
                Complex u = a[i + j];
                Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Interfata vizuala in terminal
void afiseaza_spectru_ascii(const std::vector<Complex>& date_fft, int sample_rate, const std::string& titlu) {
    int n = date_fft.size() / 2; 
    const int N_BENZI = 8;
    float magnitudini_benzi[N_BENZI] = {0};
    int samples_per_band = n / N_BENZI;

    std::cout << "\n=== " << titlu << " ===\n";

    // 1. Calculam magnitudinea pe fiecare banda si gasim maximul pentru Auto-Scalare
    float max_mag = 0.0001f;
    for (int b = 0; b < N_BENZI; b++) {
        float suma = 0;
        for (int i = 0; i < samples_per_band; i++) {
            Complex c = date_fft[b * samples_per_band + i];
            suma += std::sqrt(c.real() * c.real() + c.imag() * c.imag());
        }
        magnitudini_benzi[b] = suma / samples_per_band;
        if (magnitudini_benzi[b] > max_mag) {
            max_mag = magnitudini_benzi[b];
        }
    }

    // 2. Etichete detaliate pentru frecvente
    const std::string etichete[N_BENZI] = {
        "SUB-BASS (20-60Hz)  ",
        "BASS     (60-250Hz) ",
        "LOW-MID  (250-500Hz)",
        "MID      (500-2kHz) ",
        "UP-MID   (2k-4kHz)  ",
        "PRESENCE (4k-6kHz)  ",
        "BRILLIANCE(6k-10kHz)",
        "HIGH     (10k-20kHz)"
    };

    // 3. Desenam graficul proportional cu frecventa maxima gasita
    for (int b = 0; b < N_BENZI; b++) {
        // Mapam valoarea raportata la maxim, pe o lungime de 40 de caractere
        int lungime_bara = static_cast<int>((magnitudini_benzi[b] / max_mag) * 40.0f);
        
        std::cout << etichete[b] << " |";
        for (int i = 0; i < lungime_bara; i++) std::cout << "█";
        
        // Un mic artificiu vizual: adaugam un pointer la capăt
        if (lungime_bara > 0) std::cout << "▌"; 
        std::cout << "\n";
    }
    std::cout << "--------------------------------------------------\n";
}

void fir_neon(const std::vector<float>& intrare, const std::vector<float>& coef, std::vector<float>& iesire) {
    int n_semnal = intrare.size();
    int n_coeficienti = coef.size();
    std::fill(iesire.begin(), iesire.end(), 0.0f);

    for (int i = 0; i < n_coeficienti - 1; i++) {
        for (int j = 0; j < n_coeficienti; j++) {
            if (i - j >= 0) {
                iesire[i] += intrare[i - j] * coef[j];
            }
        }
    }

    int i = n_coeficienti - 1;
    for (; i <= n_semnal - 4; i += 4) {
        float32x4_t acumulator = vdupq_n_f32(0.0f);
        for (int j = 0; j < n_coeficienti; j++) {
            float32x4_t coef_vec = vdupq_n_f32(coef[j]);
            float32x4_t semnal_vec = vld1q_f32(&intrare[i - j]);
            acumulator = vmlaq_f32(acumulator, semnal_vec, coef_vec);
        }
        vst1q_f32(&iesire[i], acumulator);
    }

    for (; i < n_semnal; i++) {
        for (int j = 0; j < n_coeficienti; j++) {
            iesire[i] += intrare[i - j] * coef[j];
        }
    }
}

// ==========================================
// FUNCȚIA PRINCIPALĂ
// ==========================================
int main() {
    std::cout << "=== Pipeline DSP Audio (C++ vs ARM NEON) ===" << std::endl;

    std::string fisier_intrare = "test_audio.wav"; 
    std::string fisier_iesire = "audio_filtrat.wav";
    WavHeader header_audio;

    std::cout << "\n[1] Incarcare fisier..." << std::endl;
    std::vector<float> semnal_original = citeste_wav(fisier_intrare, header_audio);
    if (semnal_original.empty()) return 1;

    // Buffer comun pentru rezultate (Optimizare Memorie RAM)
    std::vector<float> buffer_iesire(semnal_original.size(), 0.0f);

    // ---------------------------------------------------------
    // GENERARE COEFICIENȚI EGALIZATOR (Windowed-Sinc)
    // ---------------------------------------------------------
    float frecventa_taiere = 400.0f; // Taiem tot ce e peste 400 Hz (pastram doar basul)
    int numar_coeficienti = 101;     // Filtru precis, de ordinul 100
    float sample_rate = header_audio.sample_rate;

    std::cout << "\n[2] Generez filtru Trece-Jos profesional..." << std::endl;
    std::cout << "    -> Frecventa taiere: " << frecventa_taiere << " Hz" << std::endl;
    std::cout << "    -> Numar coeficienti: " << numar_coeficienti << std::endl;

    std::vector<float> coeficienti_fir = genereaza_filtru_trece_jos(frecventa_taiere, sample_rate, numar_coeficienti);

    // ---------------------------------------------------------
    // RULAM VARIANTA NAIVA
    // ---------------------------------------------------------
    std::cout << "\n[3.A] Procesare FIR NAIV pe " << semnal_original.size() << " esantioane..." << std::endl;
    auto start_fir = std::chrono::high_resolution_clock::now();
    
    fir_naiv(semnal_original, coeficienti_fir, buffer_iesire);
    
    auto stop_fir = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durata_fir = stop_fir - start_fir;
    std::cout << "-> TIMP EXECUTIE FIR NAIV: " << durata_fir.count() << " ms." << std::endl;

    // ---------------------------------------------------------
    // RULAM VARIANTA NEON
    // ---------------------------------------------------------
    std::cout << "\n[3.B] Procesare FIR OPTIMIZAT CU NEON..." << std::endl;
    auto start_neon = std::chrono::high_resolution_clock::now();
    
    fir_neon(semnal_original, coeficienti_fir, buffer_iesire);
    
    auto stop_neon = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durata_neon = stop_neon - start_neon;
    std::cout << "-> TIMP EXECUTIE FIR NEON: " << durata_neon.count() << " ms." << std::endl;
    
    double speedup = durata_fir.count() / durata_neon.count();
    std::cout << "-> ACCELERARE (SPEEDUP): " << speedup << "x mai rapid!" << std::endl;

    // ---------------------------------------------------------
    // FFT SI SALVARE
    // ---------------------------------------------------------

    std::cout << "\n[4.A] Analiza FFT (INAINTE de filtrare) pe un cadru de 1024 esantioane..." << std::endl;
    int dimensiune_cadru = 1024;
    
    // Sarim la jumatatea melodiei
    int start_idx = semnal_original.size() / 2;
    
    std::vector<Complex> cadru_fft_inainte(dimensiune_cadru, Complex(0.0f, 0.0f));
    int limita_inainte = std::min(dimensiune_cadru, (int)semnal_original.size() - start_idx);
    for (int i = 0; i < limita_inainte; i++) {
        cadru_fft_inainte[i] = Complex(semnal_original[start_idx + i], 0.0f);
    }

    fft_neon(cadru_fft_inainte); 
    afiseaza_spectru_ascii(cadru_fft_inainte, header_audio.sample_rate, "SPECTRU ORIGINAL (Inainte de Filtrare)");

    std::cout << "\n[4.B] Analiza FFT (DUPA filtrare) pe acelasi cadru..." << std::endl;
    
    std::vector<Complex> cadru_fft_dupa(dimensiune_cadru, Complex(0.0f, 0.0f));
    int limita_dupa = std::min(dimensiune_cadru, (int)buffer_iesire.size() - start_idx);
    for (int i = 0; i < limita_dupa; i++) {
        cadru_fft_dupa[i] = Complex(buffer_iesire[start_idx + i], 0.0f);
    }

    fft_neon(cadru_fft_dupa); 
    afiseaza_spectru_ascii(cadru_fft_dupa, header_audio.sample_rate, "SPECTRU FILTRAT (Dupa FIR NEON)");

    std::cout << "\n[5] Salvare fisier modificat..." << std::endl;
    scrie_wav(fisier_iesire, buffer_iesire, header_audio);

    std::cout << "\n=== Pipeline Finalizat ===" << std::endl;
    return 0;

    /*std::cout << "\n[4] Analiza FFT (cadru 1024 esantioane)..." << std::endl;
    int dimensiune_cadru = 1024;
    std::vector<Complex> cadru_fft(dimensiune_cadru, Complex(0.0f, 0.0f));
    
    int limita = std::min(dimensiune_cadru, (int)buffer_iesire.size());
    for (int i = 0; i < limita; i++) {
        cadru_fft[i] = Complex(buffer_iesire[i], 0.0f);
    }

    auto start_fft = std::chrono::high_resolution_clock::now();
    fft_naiv(cadru_fft);
    auto stop_fft = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durata_fft = stop_fft - start_fft;
    std::cout << "-> TIMP EXECUTIE FFT NAIV: " << durata_fft.count() << " ms." << std::endl;

    std::cout << "\n[5] Salvare fisier modificat..." << std::endl;
    scrie_wav(fisier_iesire, buffer_iesire, header_audio);

    std::cout << "\n=== Pipeline Finalizat ===" << std::endl;
    return 0;
    */
}
