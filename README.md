# MalevolentSlice

[![PyPI Version](https://img.shields.io/pypi/v/malevolentslice.svg?color=blue)]()
[![Python Versions](https://img.shields.io/pypi/pyversions/malevolentslice.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/drrri-py/malevolentslice/blob/main/LICENSE)
[![RAM Footprint](https://img.shields.io/badge/RAM_Footprint-O(1)_%E2%89%A4_400MB-brightgreen.svg)]()

> **DESIGN AND IMPLEMENTATION OF A VOICE ACTIVITY DETECTION-BASED AUDIO DATASET SEGMENTATION SYSTEM FOR MEMORY-EFFICIENT TEXT-TO-SPEECH**  


> *RANCANG BANGUN SISTEM SEGMENTASI DATASET AUDIO BERBASIS VOICE ACTIVITY DETECTION UNTUK PENGEMBANGAN TEXT-TO-SPEECH EFISIEN MEMORI RAM*

---

## 🌐 English Quick Overview

**MalevolentSlice** (CLI alias: `malevolentslice`) is a lightweight, high-performance Python library and CLI tool designed to slice continuous, raw audio recordings (such as podcasts, audiobooks, and long interviews) into clean, high-quality audio segments formatted for training state-of-the-art Text-to-Speech (TTS) models (e.g., LJSpeech format for VITS, FastSpeech, or Tacotron).

### Quick Installation:
```bash
pip install malevolentslice
```

### Quick Run:
```bash
# Slices audio and automatically transcribes with faster-whisper tiny into metadata.csv
malevolentslice run

# Or specify custom input, output, and language
malevolentslice run -i ./raw_audio/ -o ./dataset/ -l id

# Slice only without transcription
malevolentslice run -i ./raw_audio/ -o ./dataset/ --no-transcribe

# Transcribe an existing sliced dataset directory without re-slicing
malevolentslice transcribe -d ./dataset/ -l id
```


## 📖 Dokumentasi Bahasa Indonesia

`malevolentslice` adalah pustaka (*library*) Python mandiri sekaligus perkakas baris perintah (*Command Line Interface* / CLI) berkinerja tinggi yang dirancang khusus untuk memotong rekaman audio panjang (seperti siniar/podcast, buku audio/*audiobook*, dan rekaman wawancara) menjadi segmen-segmen audio pendek, bersih, dan terstandarisasi yang siap digunakan untuk melatih model AI *Text-to-Speech* (TTS), lengkap dengan transkripsi teks otomatis ke dalam format LJSpeech `metadata.csv`.

## 🌟 Fitur Utama

- **Transkripsi Wicara Otomatis Bawaan (*Native Two-Stage ASR*):** Menghasilkan transkrip wicara riil langsung pada `metadata.csv` menggunakan model `faster-whisper` dengan kuantisasi INT8 di CPU. Menggunakan arsitektur dua tahap bergantian (VAD selesai $\rightarrow$ memori dibebaskan $\rightarrow$ Whisper dimuat) sehingga konsumsi RAM tetap sangat rendah ($\le 200 - 300\text{ MB}$).
- **Efisiensi Memori Konstan $O(1)$:** Menggunakan generator aliran audio (*streaming generator* via `soundfile.blocks()`) yang membaca data per blok dari penyimpanan sekunder langsung ke penyangga memori, mencegah risiko *Out-of-Memory* (OOM) meskipun memproses audio berdurasi puluhan jam.
- **Pencarian Audio Otomatis & Terbatas (`--max-depth`):** Cukup jalankan `malevolentslice run` tanpa argumen, sistem akan otomatis mendeteksi berkas audio di direktori kerja hingga batas kedalaman bawaan 3 level subfolder (`max_depth = 3`), sambil mengecualikan folder sistem/virtual environment (`.venv`, `.git`, `dataset`, dsb.).
- **Perintah Transkripsi Mandiri (`malevolentslice transcribe`):** Mentranskripsi ulang folder dataset yang sudah pernah dipotong sebelumnya tanpa harus mengiris ulang audio, lengkap dengan dukungan *resume*.
- **Hierarki Folder Fleksibel (`--preserve-structure` / `-p`):** Mendukung format standar korpus LJSpeech (semua segmen di `wavs/`), mode datar (`--flat`), maupun pencerminan hierarki subfolder masukan (*mirrored hierarchy*) ke folder luaran.
- **NumPy Vectorized Noise Gate:** Penapisan awal derau latar (*background noise*) secara instan menggunakan komputasi vektor NumPy tanpa *overhead* komputasi tinggi.
- **Mesin Deteksi Suara Silero-VAD ONNX:** Inferensi keberadaan wicara presisi tingkat bingkai (*frame-level*) berbasis model Silero-VAD teroptimasi via ONNX Runtime CPU (ringan dan tanpa ketergantungan PyTorch yang berat).
- **Proteksi Hang-over & Pemotongan Cerdas:** Mencegah pemotongan konsonan akhir kata (*word clipping*) dengan penyangga jeda wicara (*hangover buffer*), serta membatasi durasi segmen ideal TTS ($1.0\text{s} \le t \le 12.0\text{s}$).
- **Antarmuka Terminal Modern & Pythonic API:** Tampilan CLI interaktif berbasis Rich dengan visualisasi progres real-time per tahap, pemantau alokasi RAM aktif `(STABLE)`, serta API Python yang elegan.

---

## 💻 Instalasi

Pastikan Anda menggunakan Python versi 3.8 atau lebih baru:

### Pengguna Akhir (via PyPI)
```bash
pip install malevolentslice
```

### Mode Pengembangan (Development)
Jika Anda mengkloning repositori ini secara lokal:
```bash
git clone https://github.com/drrri-py/malevolentslice.git
cd malevolentslice

# Pasang dalam mode editable beserta dependensi dev (pengujian & profiler)
pip install -e ".[dev]"
```

---

## 🚀 Panduan Penggunaan Cepat

### 1. Antarmuka Baris Perintah (CLI)

#### A. Eksekusi Otomatis (Zero-Configuration)
Jika Anda memiliki berkas audio di direktori kerja atau di dalam subfolder (misalnya di folder `raw/`, folder proyek, atau subfolder sesi), cukup jalankan:

```bash
malevolentslice run
```
> **Catatan:** Perintah di atas akan secara otomatis memindai seluruh berkas audio hingga 3 tingkat subfolder, memprosesnya dengan alokasi RAM yang terjaga stabil, dan menyimpannya ke direktori `./dataset/`.

#### B. Menentukan Sumber Masukan & Folder Luaran
```bash
# Memproses folder sumber tertentu ke folder tujuan tertentu
malevolentslice run -i ./rekaman_mentah/ -o ./dataset_tts/

# Memproses satu berkas audio tunggal
malevolentslice run -i ./rekaman_mentah/audio_wawancara.wav -o ./dataset/
```

#### C. Mempertahankan Struktur Hierarki Subfolder (`-p` / `--preserve-structure`)
Jika audio masukan Anda dikelompokkan ke dalam subfolder (misal: `pembicara_1/`, `pembicara_2/`) dan Anda ingin hasil potongannya terpisah dalam subfolder yang sama:

```bash
malevolentslice run -p
# atau:
malevolentslice run --preserve-structure
```

#### D. Mode Luaran Datar (`--flat`)
Mengekspor seluruh berkas `.wav` langsung ke dalam folder luaran utama tanpa membuat subfolder `wavs/`:

```bash
malevolentslice run --flat
```

#### E. Mengatur Batas Kedalaman Pencarian Folder (`--max-depth`)
```bash
# Menelusuri subfolder hingga kedalaman maksimal 5 tingkat
malevolentslice run --max-depth 5
```

#### F. Mengatur Bahasa & Model Transkripsi (`-l` / `--language`, `-m` / `--model`)
Secara *default*, sistem memotong audio sekaligus mentranskripsikannya secara otomatis ke dalam teks asli menggunakan Whisper model `tiny` (INT8 CPU). Anda dapat menyesuaikan bahasa dan model:

```bash
# Menggunakan Bahasa Indonesia ('id') dengan model Whisper 'base'
malevolentslice run -i ./rekaman/ -o ./dataset/ -l id -m base

# Hanya memotong audio tanpa transkripsi (mengisi template dummy)
malevolentslice run -i ./rekaman/ -o ./dataset/ --no-transcribe
```

#### G. Mentranskripsi Dataset yang Sudah Ada (`malevolentslice transcribe`)
Jika Anda sudah memiliki folder dataset yang berisi berkas audio hasil potongan dan ingin membuat atau memperbarui `metadata.csv` tanpa memotong ulang audio:

```bash
# Transkripsi dataset yang ada dengan Bahasa Indonesia (mode resume otomatis)
malevolentslice transcribe -d ./dataset/ -l id

# Mode overwrite (mentranskripsi ulang seluruh segmen dari awal)
malevolentslice transcribe -d ./dataset/ -l id --overwrite
```

---

### Daftar Lengkap Opsi CLI (`malevolentslice run --help`)

| Opsi | Pilihan Singkat | Nilai Baku (*Default*) | Deskripsi |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | `None` (Otomatis) | Berkas audio atau direktori sumber masukan. |
| `--output` | `-o` | `./dataset/` | Direktori luaran untuk dataset standar LJSpeech. |
| `--threshold` | `-t` | `-40.0` | Ambang batas Noise Gate dalam satuan desibel (dB). |
| `--speech-threshold` | - | `0.5` | Ambang batas probabilitas wicara model VAD ($0.0 - 1.0$). |
| `--silence` | `-s` | `400.0` | Batas jeda hening minimum pemisah segmen (milidetik). |
| `--buffer` | `-b` | `5.0` | Ukuran penyangga blok streaming data audio (MB). |
| `--min-speech` | - | `1000.0` | Durasi segmen wicara minimum yang diekspor (milidetik). |
| `--max-speech` | - | `12000.0` | Durasi segmen wicara maksimum yang diekspor (milidetik). |
| `--sr` | - | `16000` | Frekuensi sampel target audio luaran (Hz). |
| `--wav-dir` | - | `wavs` | Nama subfolder audio WAV (kosongkan `''` jika `--flat`). |
| `--flat` | - | `False` | Simpan berkas WAV langsung di *root* folder tanpa subfolder `wavs/`. |
| `--preserve-structure`| `-p` | `False` | Cerminkan susunan hierarki subfolder masukan ke folder luaran. |
| `--max-depth` | - | `3` | Batas kedalaman penelusuran subfolder audio. |
| `--transcribe / --no-transcribe` | - | `True` | Otomatis menjalankan tahap 2 Speech-to-Text (`faster-whisper` INT8 CPU) untuk mengisi teks asli di `metadata.csv`. |
| `--model` | `-m` | `tiny` | Ukuran model Whisper (`tiny`, `base`, `small`, `medium`). |
| `--language` | `-l` | `None` (Otomatis) | Kode bahasa untuk transkripsi (contoh: `id`, `en`). |
| `--quiet` | `-q` | `False` | Jalankan proses secara senyap tanpa animasi terminal/banner. |

---

### 2. Antarmuka Pemrograman Python (Pythonic API)

Pustaka `malevolentslice` dapat diintegrasikan dengan mudah ke dalam skrip Python atau alur kerja (*pipeline*) pemrosesan data AI:

```python
from malevolentslice import Pipeline

# 1. Inisialisasi Pipeline dengan segmentasi & transkripsi otomatis
pipeline = Pipeline(
    transcribe=True,              # Otomatis transkripsi teks wicara riil
    whisper_model="tiny",         # Model ASR: 'tiny', 'base', 'small', 'medium'
    language="id",                # Kode bahasa: 'id', 'en', atau None (auto-detect)
    noise_threshold_db=-40.0,
    speech_threshold=0.5,
    min_silence_duration_ms=400.0,
    min_speech_duration_ms=1000.0,
    max_speech_duration_ms=12000.0,
    chunk_buffer_mb=5.0,
    target_sr=16000,
    max_depth=3,                  # Batas kedalaman pencarian subfolder
    preserve_structure=False,     # Ubah ke True jika ingin memisahkan per subfolder
    wav_subfolder="wavs"
)

# 2. Jalankan pemotongan dan transkripsi
summary = pipeline.process(
    source="./rekaman_mentah/",
    output_dir="./dataset_tts/"
)

# 3. Akses rekapitulasi hasil eksekusi
print(f"Total berkas diproses : {summary.total_files}")
print(f"Segmen dihasilkan     : {summary.total_segments}")
print(f"Segmen ditranskripsi  : {summary.transcribed_segments}")
print(f"Total durasi wicara   : {summary.total_processed_duration:.2f} detik")
print(f"Waktu eksekusi        : {summary.elapsed_time:.2f} detik")
print(f"Konsumsi Puncak RAM   : {summary.peak_memory_mb:.2f} MB")
```

#### Penggunaan Transcriber Mandiri (`AudioTranscriber`)
Jika Anda hanya ingin mentranskripsi dataset atau file audio yang sudah ada:

```python
from malevolentslice.core.transcriber import AudioTranscriber

transcriber = AudioTranscriber(model_size="tiny", language="id")

# Transkripsi 1 berkas audio
teks = transcriber.transcribe_file("dataset/wavs/segment_000001.wav")
print("Transkrip:", teks)

# Atau transkripsi daftar berkas langsung ke metadata.csv (LJSpeech standard)
# transcriber.transcribe_dataset(wav_items, "dataset/metadata.csv", resume=True)
transcriber.unload_model()
```

#### Fitur Pencarian Mandiri (*Standalone Audio Search*)
Anda juga dapat memanfaatkan fungsi pencarian audio bawaan untuk keperluan skrip kustom:

```python
from malevolentslice import find_audio_files

# Mencari seluruh file audio hingga kedalaman 3 tingkat
berkas_audio = find_audio_files("./proyek_audio", max_depth=3)
print(f"Ditemukan {len(berkas_audio)} berkas audio.")
```

---

## 📁 Struktur Direktori Luaran Dataset

### 1. Struktur Standar LJSpeech (Bawaan)
```text
dataset/
├── wavs/
│   ├── segment_000001.wav
│   ├── segment_000002.wav
│   └── ...
├── metadata.csv
└── processing_audit.log
```

Format berkas `metadata.csv` (pemisah `|` standar korpus LJSpeech):
```csv
segment_000001|Halo ini adalah contoh wicara asli hasil transkripsi.|Halo ini adalah contoh wicara asli hasil transkripsi.
segment_000002|Sistem secara otomatis mengisi teks wicara riil.|Sistem secara otomatis mengisi teks wicara riil.
```

> 💡 **Transkripsi Wicara Otomatis Terintegrasi:**  
> `malevolentslice` secara *default* langsung menyertakan transkripsi wicara asli (Speech-to-Text) menggunakan engine `faster-whisper` (INT8 CPU). Format luaran langsung siap pakai untuk pelatihan model TTS (*ready-to-train*) tanpa memerlukan skrip transkripsi pihak ketiga tambahan. Jika Anda hanya menginginkan teks placeholder dummy, jalankan dengan flag `--no-transcribe`.

### 2. Struktur Hierarki Bercermin (`--preserve-structure` / `-p`)
```text
dataset/
├── wavs/
│   ├── sesi_01/
│   │   ├── segment_000001.wav
│   │   └── segment_000002.wav
│   └── sesi_02/
│       └── segment_000003.wav
├── metadata.csv
└── processing_audit.log
```

---

## 🔄 Alur Kerja Pembuatan Dataset TTS (*Pipeline Workflow*)

Untuk melatih model Text-to-Speech (seperti VITS, FastSpeech 2, Piper TTS, atau Coqui TTS), `malevolentslice` menangani seluruh alur kurasi secara otomatis dalam satu eksekusi:

```text
[ Rekaman Audio Mentah ]
         │
         ▼
[ MalevolentSlice (Two-Stage Pipeline) ]
 ├── Tahap 1: Slicing Akustik VAD (Silero-VAD ONNX) ──▶ Ekspor WAV 16kHz PCM
 ├── Pembersihan Memori (Garbage Collection)       ──▶ RAM kembali bersih (0 MB VAD overhead)
 └── Tahap 2: Transkripsi ASR (faster-whisper INT8) ──▶ Tulis teks riil ke metadata.csv
         │
         ▼
[ Dataset Korpus LJSpeech Siap Latih ]
 (Langsung kompatibel dengan VITS, FastSpeech 2, Piper, Coqui TTS, Tacotron 2)
```

---

## 📊 Indikator Pemantau Memori RAM

Sistem secara aktif memantau konsumsi *Resident Set Size* (RSS) memori fisik komputer melalui pustaka `psutil`:
- **`STABLE` ($\le 100\text{ MB}$)**: Alokasi memori fisik berada dalam batas amplop ideal sistem. Memori dibersihkan secara agresif oleh `gc.collect()` setelah tiap blok selesai diproses, membuktikan alokasi memori konstan $O(1)$.
- **`ELEVATED` ($> 100\text{ MB}$)**: Penggunaan RAM meningkat melebihi ambang batas 100 MB (misal saat parameter buffer diperbesar).

---

## 🧪 Pengujian Unit & Validasi

Jalankan pengujian menggunakan `pytest` untuk memverifikasi fungsionalitas:

```bash
pytest -v
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah lisensi [MIT](https://github.com/drrri-py/malevolentslice/blob/main/LICENSE).
