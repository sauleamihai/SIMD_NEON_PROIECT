#!/usr/bin/env python3
"""
DSP Audio Pipeline — Flask Web Server
Rulează pe Raspberry Pi Zero 2W
Expune: POST /run-pipeline, GET /audio, GET /spectrum-data
"""

import os
import struct
import subprocess
from flask import Flask, jsonify, send_file, request, Response
from flask_cors import CORS

# numpy este disponibil pe Pi OS: sudo apt install python3-numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import math
    HAS_NUMPY = False

app = Flask(__name__)
CORS(app)

BINARY_PATH = "./dsp_pipeline"
INPUT_WAV   = "test_audio.wav"
OUTPUT_WAV  = "audio_filtrat.wav"
LOG_FILE    = "pipeline_log.txt"

# Timeout generos: Pi Zero 2W e lent pe fișiere mari.
# FIR NEON pe 64MB (~3.2M eșantioane, 101 coef) ≈ 30-90s pe Pi Zero 2W.
# FIR naiv pe același fișier poate dura 10-20 min → vedem nota din main.cpp.
PIPELINE_TIMEOUT = 600  # 10 minute, suficient și pentru varianta naivă dacă e activată

# ──────────────────────────────────────────────
# Helpers WAV
# ──────────────────────────────────────────────

def read_wav_header(path):
    """Returnează (sample_rate, n_channels, bit_depth, n_samples, duration_s) sau None."""
    try:
        with open(path, "rb") as f:
            raw = f.read(256)  # header e mic
        idx = raw.find(b'fmt ')
        if idx == -1:
            return None
        audio_fmt, channels, sr, _, _, bits = struct.unpack_from('<HHIIHH', raw, idx + 8)
        didx = raw.find(b'data')
        data_bytes = struct.unpack_from('<I', raw, didx + 4)[0] if didx != -1 else 0
        n_samples = data_bytes // (bits // 8)
        duration  = n_samples / sr if sr > 0 else 0
        return {"sample_rate": sr, "channels": channels, "bit_depth": bits,
                "n_samples": n_samples, "duration_s": round(duration, 2)}
    except Exception as e:
        print(f"[ERR] read_wav_header: {e}")
        return None


def read_wav_samples_numpy(path, max_samples=16384):
    """Citește eșantioane PCM16 folosind numpy (rapid)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        idx = raw.find(b'data')
        if idx == -1:
            return None
        chunk_size = struct.unpack_from('<I', raw, idx + 4)[0]
        offset = idx + 8
        n = min(chunk_size // 2, max_samples)
        arr = np.frombuffer(raw, dtype='<i2', count=n, offset=offset).astype(np.float32)
        return arr / 32768.0
    except Exception as e:
        print(f"[ERR] read_wav_samples_numpy: {e}")
        return None


def read_wav_samples_pure(path, max_samples=16384):
    """Fallback fără numpy."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        idx = raw.find(b'data')
        if idx == -1:
            return []
        chunk_size = struct.unpack_from('<I', raw, idx + 4)[0]
        offset = idx + 8
        n = min(chunk_size // 2, max_samples)
        return [struct.unpack_from('<h', raw, offset + i * 2)[0] / 32768.0 for i in range(n)]
    except Exception as e:
        print(f"[ERR] read_wav_samples_pure: {e}")
        return []


def compute_fft_magnitudes_numpy(samples_np, n_bands=64, sample_rate=44100, n_channels=1):
    """
    FFT cu numpy, fereastră Hann, scala dB, benzi logaritmice corecte.

    Fix-uri față de versiunea anterioară:
      - Stereo → downmix la mono înainte de FFT (evită artefacte L/R interleaved)
      - Benzi logaritmice mapate pe frecvențe reale (20 Hz – Nyquist), nu pe indici brut
      - Scala dB (nu liniară) — face vizibil și un LPF agresiv la 400 Hz
      - FFT_SIZE fix la 8192 pentru rezoluție frecvențială bună (~5 Hz/bin la 44100 Hz)
    """
    N = len(samples_np)
    if N == 0:
        return [0.0] * n_bands

    # ── 1. Downmix stereo → mono ──────────────────────────────────
    if n_channels == 2 and N % 2 == 0:
        left  = samples_np[0::2]
        right = samples_np[1::2]
        mono  = (left + right) * 0.5
    else:
        mono = samples_np

    # ── 2. FFT cu fereastră Hann ──────────────────────────────────
    FFT_SIZE = 8192
    frame    = mono[:FFT_SIZE] if len(mono) >= FFT_SIZE else mono
    window   = np.hanning(len(frame))
    windowed = frame * window
    fft_out  = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    # fft_out[k] corespunde frecvenței k * sample_rate / FFT_SIZE
    freqs    = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate)   # array de frecvențe în Hz
    half     = len(fft_out)

    # ── 3. Benzi logaritmice pe frecvențe reale ───────────────────
    # De la 20 Hz la Nyquist, împărțire log uniformă
    f_min = 20.0
    f_max = sample_rate / 2.0
    band_mags = []

    for b in range(n_bands):
        # Limite Hz pentru banda b (spațiere log)
        f_lo = f_min * (f_max / f_min) ** (b       / n_bands)
        f_hi = f_min * (f_max / f_min) ** ((b + 1) / n_bands)

        # Indici bin corespunzători
        bin_lo = int(f_lo * FFT_SIZE / sample_rate)
        bin_hi = int(f_hi * FFT_SIZE / sample_rate) + 1
        bin_lo = max(0,    min(bin_lo, half - 1))
        bin_hi = max(bin_lo + 1, min(bin_hi, half))

        mag = float(np.max(fft_out[bin_lo:bin_hi]))
        band_mags.append(mag)

    # ── 4. Scala dB ───────────────────────────────────────────────
    # Convertim la dB, clipăm la -80 dB (floor de zgomot), normalizăm 0–1
    DB_FLOOR = -80.0
    eps = 1e-10
    band_db = [20.0 * np.log10(max(v, eps)) for v in band_mags]
    db_max  = max(band_db)
    db_min  = db_max + DB_FLOOR   # totul sub db_min e 0
    result  = [max(0.0, (v - db_min) / (db_max - db_min + eps)) for v in band_db]

    return result


def compute_fft_magnitudes_pure(samples, n_bands=64):
    """DFT pe 512 puncte (fără numpy). Suficient de rapid pe Pi."""
    N = min(len(samples), 512)
    if N == 0:
        return [0.0] * n_bands
    samples = samples[:N]
    windowed = [samples[i] * (0.5 - 0.5 * math.cos(2 * math.pi * i / (N - 1))) for i in range(N)]
    half = N // 2
    mags = []
    for k in range(half):
        re = sum(windowed[n] * math.cos(2 * math.pi * k * n / N) for n in range(N))
        im = sum(windowed[n] * math.sin(2 * math.pi * k * n / N) for n in range(N))
        mags.append(math.sqrt(re * re + im * im) / N)
    band_mags = []
    for b in range(n_bands):
        lo = int(half * (b / n_bands) ** 1.5)
        hi = int(half * ((b + 1) / n_bands) ** 1.5)
        hi = max(hi, lo + 1)
        hi = min(hi, half)
        band_mags.append(max(mags[lo:hi]) if lo < hi else 0.0)
    peak = max(band_mags) or 1.0
    return [v / peak for v in band_mags]


def compute_waveform(samples, points=512):
    if HAS_NUMPY and hasattr(samples, '__len__') and hasattr(samples, 'reshape'):
        n = len(samples)
        if n == 0:
            return [0.0] * points
        step = max(1, n // points)
        return samples[::step][:points].tolist()
    else:
        n = len(samples)
        if n == 0:
            return [0.0] * points
        step = max(1, n // points)
        return [samples[i * step] for i in range(min(points, n // step))]



# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    """Rulează binariul C++ și returnează log + status."""
    if not os.path.exists(BINARY_PATH):
        return jsonify({
            "success": False,
            "log": f"[ERR] Binariul '{BINARY_PATH}' nu a fost găsit.\n"
                   f"Compilează cu: g++ -O2 -march=armv8-a+simd -o dsp_pipeline main.cpp"
        }), 500

    if not os.path.exists(INPUT_WAV):
        return jsonify({
            "success": False,
            "log": f"[ERR] Fișierul de intrare '{INPUT_WAV}' lipsește."
        }), 400

    # Estimează și loghează timpul de așteptare înainte de a porni
    hdr = read_wav_header(INPUT_WAV)
    size_mb = os.path.getsize(INPUT_WAV) / (1024 * 1024)
    pre_log = f"[INFO] Fișier: {size_mb:.1f} MB"
    if hdr:
        pre_log += f", {hdr['duration_s']}s @ {hdr['sample_rate']} Hz, {hdr['channels']}ch\n"
        if hdr['n_samples'] > 500_000:
            pre_log += f"[WARN] Fișier mare ({hdr['n_samples']:,} eșantioane). FIR naiv poate dura minute.\n"
            pre_log += f"[WARN] Dacă e prea lent, activează modul FAST_ONLY din main.cpp.\n"
    pre_log += f"[INFO] Timeout setat la {PIPELINE_TIMEOUT}s.\n"

    try:
        result = subprocess.run(
            [BINARY_PATH],
            capture_output=True, text=True, timeout=PIPELINE_TIMEOUT
        )
        log = pre_log + result.stdout + result.stderr
        with open(LOG_FILE, "w") as lf:
            lf.write(log)

        success = result.returncode == 0 and os.path.exists(OUTPUT_WAV)
        return jsonify({"success": success, "log": log})

    except subprocess.TimeoutExpired:
        msg = (f"[ERR] Timeout după {PIPELINE_TIMEOUT}s.\n"
               f"[TIP] Setează FAST_ONLY=true în main.cpp și recompilează "
               f"pentru a sări varianta naivă pe fișiere mari.")
        return jsonify({"success": False, "log": pre_log + msg}), 500
    except Exception as e:
        return jsonify({"success": False, "log": pre_log + str(e)}), 500


@app.route("/wav-info")
def wav_info():
    """Returnează metadatele WAV (sample rate, durată etc.) fără a procesa semnalul."""
    target   = request.args.get("file", "input")
    wav_path = INPUT_WAV if target == "input" else OUTPUT_WAV
    if not os.path.exists(wav_path):
        return jsonify({"error": "Fișier inexistent"}), 404
    hdr = read_wav_header(wav_path)
    size_mb = os.path.getsize(wav_path) / (1024 * 1024)
    if hdr:
        hdr["size_mb"] = round(size_mb, 1)
    return jsonify(hdr or {"error": "Header invalid"})


@app.route("/spectrum-data")
def spectrum_data():
    """Returnează magnitudinile FFT + waveform. Folosește numpy dacă disponibil."""
    target   = request.args.get("file", "filtered")
    wav_path = OUTPUT_WAV if target == "filtered" else INPUT_WAV

    if not os.path.exists(wav_path):
        return jsonify({"error": "Fișier inexistent", "bands": [], "waveform": []}), 404

    hdr = read_wav_header(wav_path)
    sr  = hdr["sample_rate"] if hdr else 44100
    nch = hdr["channels"]    if hdr else 1

    if HAS_NUMPY:
        samples = read_wav_samples_numpy(wav_path, max_samples=16384)
        if samples is None:
            return jsonify({"error": "Citire WAV eșuată", "bands": [], "waveform": []}), 500
        bands = compute_fft_magnitudes_numpy(samples, n_bands=64, sample_rate=sr, n_channels=nch)
        wave  = compute_waveform(samples, points=512)
        n     = len(samples)
    else:
        samples = read_wav_samples_pure(wav_path, max_samples=4096)
        bands   = compute_fft_magnitudes_pure(samples, n_bands=64)
        wave    = compute_waveform(samples, points=512)
        n       = len(samples)

    payload = {
        "bands": bands,
        "waveform": wave,
        "sample_count": n,
        "numpy": HAS_NUMPY,
    }
    if hdr:
        payload.update({
            "sample_rate": hdr["sample_rate"],
            "duration_s":  hdr["duration_s"],
            "channels":    hdr["channels"],
        })
    return jsonify(payload)


@app.route("/audio")
def serve_audio():
    """Servește fișierul WAV pentru playback în browser."""
    target   = request.args.get("file", "filtered")
    wav_path = OUTPUT_WAV if target == "filtered" else INPUT_WAV

    if not os.path.exists(wav_path):
        return Response("Fișier inexistent", status=404)

    return send_file(wav_path, mimetype="audio/wav", as_attachment=False)


@app.route("/upload", methods=["POST"])
def upload_audio():
    """Permite încărcarea unui fișier WAV custom ca sursă de intrare."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Niciun fișier"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".wav"):
        return jsonify({"success": False, "error": "Doar WAV acceptat"}), 400
    f.save(INPUT_WAV)
    hdr     = read_wav_header(INPUT_WAV)
    size_mb = os.path.getsize(INPUT_WAV) / (1024 * 1024)
    info    = f"{size_mb:.1f} MB"
    if hdr:
        info += f" · {hdr['duration_s']}s · {hdr['sample_rate']} Hz"
    return jsonify({"success": True, "message": f"'{INPUT_WAV}' salvat. ({info})", "info": hdr})


if __name__ == "__main__":
    # threaded=False: Pi Zero 2W are 4 core-uri dar Flask threaded poate cauza
    # probleme cu subprocess-ul lung; single-thread e mai sigur.
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=False)
