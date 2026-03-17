#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <algorithm>

// ==========================================
// DECLARAȚII DE BAZĂ (Acum pe 32-bit float)
// ==========================================
typedef std::complex<float> Complex; // Modificat la float
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
            // Impartim la 32768.0f pentru float
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
// MODULUL DSP: FFT ȘI FIR (VARIANTE NAIVE FLOAT)
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
    Complex w(1.0f, 0.0f), wn(cos(angle), sin(angle));
    
    for (int i = 0; i < n / 2; i++) {
        a[i] = pare[i] + w * impare[i];
        a[i + n / 2] = pare[i] - w * impare[i];
        w *= wn;
    }
}

std::vector<float> fir_naiv(const std::vector<float>& semnal_intrare, const std::vector<float>& coeficienti) {
    int n_semnal = semnal_intrare.size();
    int n_coeficienti = coeficienti.size();
    std::vector<float> semnal_iesire(n_semnal, 0.0f);

    for (int i = 0; i < n_semnal; i++) {
        for (int j = 0; j < n_coeficienti; j++) {
            if (i - j >= 0) {
                semnal_iesire[i] += semnal_intrare[i - j] * coeficienti[j];
            }
        }
    }
    return semnal_iesire;
}

// ==========================================
// FUNCȚIA PRINCIPALĂ
// ==========================================
int main() {
    std::cout << "=== Pipeline DSP Audio (Varianta FLOAT Naiva) ===" << std::endl;

    std::string fisier_intrare = "test_audio.wav"; 
    std::string fisier_iesire = "audio_filtrat.wav";
    WavHeader header_audio;

    std::cout << "\n[1] Incarcare fisier..." << std::endl;
    std::vector<float> semnal_original = citeste_wav(fisier_intrare, header_audio);
    if (semnal_original.empty()) return 1;

    std::vector<float> coeficienti_fir(11, 1.0f / 11.0f); 

    std::cout << "\n[2] Procesare filtru FIR pe " << semnal_original.size() << " esantioane..." << std::endl;
    auto start_fir = std::chrono::high_resolution_clock::now();
    
    std::vector<float> semnal_filtrat = fir_naiv(semnal_original, coeficienti_fir);
    
    auto stop_fir = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durata_fir = stop_fir - start_fir;
    std::cout << "-> TIMP EXECUTIE FIR NAIV: " << durata_fir.count() << " milisecunde." << std::endl;

    std::cout << "\n[3] Analiza FFT (cadru 1024 esantioane)..." << std::endl;
    int dimensiune_cadru = 1024;
    std::vector<Complex> cadru_fft(dimensiune_cadru, Complex(0.0f, 0.0f));
    
    int limita = std::min(dimensiune_cadru, (int)semnal_filtrat.size());
    for (int i = 0; i < limita; i++) {
        cadru_fft[i] = Complex(semnal_filtrat[i], 0.0f);
    }

    auto start_fft = std::chrono::high_resolution_clock::now();
    fft_naiv(cadru_fft);
    auto stop_fft = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durata_fft = stop_fft - start_fft;
    std::cout << "-> TIMP EXECUTIE FFT NAIV: " << durata_fft.count() << " milisecunde." << std::endl;

    std::cout << "\n[4] Salvare fisier modificat..." << std::endl;
    scrie_wav(fisier_iesire, semnal_filtrat, header_audio);

    std::cout << "\n=== Pipeline Finalizat ===" << std::endl;
    return 0;
}
