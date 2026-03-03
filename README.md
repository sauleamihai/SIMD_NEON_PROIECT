# Egalizator Audio Digital Multibandă (FIR cu Optimizare NEON)

Proiectul vizează implementarea unui egalizator audio digital care procesează un fișier de intrare (`.wav` conținând voce sau muzică) și îi ajustează spectrul de frecvențe. Algoritmul central studiat este **Filtrarea FIR (Finite Impulse Response)**.

## Cum funcționează și ce aduce proiectului

### 1. Procesarea semnalului
Semnalul audio va fi trecut prin mai multe filtre FIR rulate în paralel (ex: filtru trece-jos pentru bași, trece-bandă pentru medii, trece-sus pentru înalte). Fiecare filtru calculează convoluția dintre eșantioanele audio și un set de coeficienți pre-calculați.

Din punct de vedere matematic, algoritmul implementează ecuația de diferențe a filtrului FIR:

$$y[n]=\sum_{k=0}^{N-1}h[k]\cdot x[n-k]$$

unde:
* $x$ este semnalul audio de intrare.
* $h$ sunt coeficienții filtrului.
* $y$ este ieșirea filtrată.

### 2. Optimizarea NEON (Cerința de accelerare)
În implementarea standard (C++ scalar), ecuația de mai sus presupune înmulțirea și adunarea fiecărui eșantion individual. 

Prin utilizarea intrinsecilor NEON (ex: instrucțiunea `vmlaq_f32` - Vector Multiply-Accumulate), procesorul va încărca **4 eșantioane audio** și **4 coeficienți** în registrele sale de 128 de biți și va efectua calculele simultan, într-un singur ciclu de ceas.

### 3. Analiza științifică (Cerința de documentație)
Proiectul va include o analiză riguroasă a performanței. Se va măsura și compara timpul de execuție (în milisecunde) între:
- Procesarea fișierului audio folosind C++ standard (Scalar)
- Procesarea folosind extensiile SIMD/NEON

Acest lucru va demonstra direct eficiența paralelizării datelor (**Data-Level Parallelism**).
