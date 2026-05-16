# Egalizator Audio Digital Multibandă cu Optimizare ARM NEON

Egalizator parametric pe 10 benzi implementat în C++, cu accelerare SIMD ARM NEON, rulând pe Raspberry Pi Zero 2W (Cortex-A53). Include interfață web pentru control și vizualizare spectrum.

## Algoritmul central

Proiectul implementează filtrare **IIR Biquad (Infinite Impulse Response)** în arhitectura **Transposed Direct Form II**, aplicată în serie pe 10 benzi de frecvență. Fiecare bandă este un filtru biquad de ordinul 2 cu ecuația de diferențe:

```
y[n]  = b0·x[n] + s1
s1'   = b1·x[n] - a1·y[n] + s2
s2'   = b2·x[n] - a2·y[n]
```

Coeficienții `b0, b1, b2, a1, a2` sunt calculați prin formulele **Audio EQ Cookbook (RBJ)** pentru filtre peaking, low-shelf și high-shelf.

> **De ce IIR Biquad și nu FIR?** Filtrele FIR sumate pe benzi produc distorsiuni de fază și goluri de amplitudine la frecvențele de crossover. Biquad-ul în serie modifică doar banda țintă, lăsând restul spectrului la exact 0 dB — exact cum funcționează un EQ hardware real.

## Optimizările NEON implementate

Cortex-A53 este un procesor **in-order** cu penalitate de 3-4 cicluri pentru orice transfer NEON↔scalar (`vgetq_lane_f32`). Toate optimizările au fost alese să evite aceste transferuri.

| Modul | Intrinsic cheie | Tehnică | Speedup |
|---|---|---|---|
| **Biquad EQ** | `vmlaq_f32`, `vmlsq_f32` | 4 bande paralel per sample, TDF2 complet în NEON | 1.77× |
| **FFT butterfly** | `vld2q_f32`, `vst2q_f32` | Complex multiply + butterfly pe 4 perechi simultan | 8.03× |
| **Magnitude spectrum** | `vsqrtq_f32` | Hardware sqrt IEEE-correct, 2× unrolled pentru pipeline | 2.76× |
| **int16 → float** | `vld1q_s16`, `vmovl_s16`, `vcvtq_f32_s32` | 8 sample-uri per iterație | eliminat bottleneck |
| **Stereo deinterleave** | `vld2q_s16` | Deinterleave L/R automat în load | zero loop scalar |
| **Stereo interleave** | `vst2q_s16` | Interleave L/R automat în store | zero loop scalar |

### Arhitectura biquad 4-bande-paralel

Bucla scalară procesează 10 bande serial (50 operații per sample). Soluția: grupăm 4 bande per registru `float32x4_t` și procesăm un sample prin toate 4 simultan:

```cpp
float32x4_t X  = vdupq_n_f32(in[i]);              // broadcast sample în 4 lane-uri
float32x4_t Y0 = vmlaq_f32(S1_g0, B0_g0, X);      // y[b0..b3] = B0*x + s1
S1_g0 = vmlaq_f32(vmlsq_f32(S2_g0, A1_g0, Y0), B1_g0, X); // s1' = s2 - A1*y + B1*x
S2_g0 = vmlsq_f32(vmulq_f32(B2_g0, X), A2_g0, Y0);         // s2' = B2*x - A2*y
float x1 = neon_hsum(vmulq_f32(Y0, NORM));         // suma 4 bande → input grup următor
```

3 grupuri × 6 instrucțiuni NEON = **18 instrucțiuni** față de 50 scalare seriale.

## Rezultate benchmark (Pi Zero 2W, semnal 44.1kHz, 2s)

```
Biquad scalar:    11.87ms  →  NEON 4-bande:  6.71ms   (1.77×)
FFT naiv:          3.50ms  →  NEON butterfly: 0.44ms   (8.03×)
Magnitude scalar:  0.87ms  →  vsqrtq 2×:     0.34ms   (2.76×)
─────────────────────────────────────────────────────────────
Total scalar:     17.49ms  →  Total NEON:    8.57ms   (2.04×)
Real-time factor: 298× (procesează 298 secunde audio per secundă CPU)
```

## Structura proiectului

```
dsp_eq/
├── dsp_eq.cpp      # Motor DSP C++ cu optimizări NEON
├── app.py          # Server Flask (backend web GUI)
├── Makefile        # Build system
└── web/
    └── index.html  # GUI web — spectrum, EQ curve, benchmark live
```

## Build și rulare

```bash
# Pe Pi Zero 2W
sudo apt install g++ make python3-pip
pip3 install flask

# Compilare
g++ -std=c++17 -O3 -march=armv8-a+simd -mtune=cortex-a53 \
    -ffast-math -funroll-loops -o dsp_eq dsp_eq.cpp -lm

# Pornire server web (http://<ip-pi>:5001)
python3 app.py
```

## Utilizare CLI

```bash
./dsp_eq input.wav output.wav [b0 b1 b2 b3 b4 b5 b6 b7 b8 b9]
# Gainuri în dB (−24 la +24), câte unul per bandă:
# b0=Sub-Bass(60Hz)  b1=Bass(170Hz)   b2=Low-Mid(350Hz)  b3=Mid(700Hz)
# b4=Upper-Mid(1.4k) b5=Presence(3k)  b6=Brilliance(6k)  b7=Air(10k)
# b8=Ultra(14k)      b9=High(18k)

# Exemplu: boost bass +6dB, tăiere mid −3dB
./dsp_eq track.wav out.wav 0 6 0 -3 0 0 0 0 0 0
```
