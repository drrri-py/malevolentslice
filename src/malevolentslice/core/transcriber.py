import os
import csv
import gc
from typing import Optional, List, Dict, Any, Callable
import soundfile as sf
import numpy as np

import logging
import warnings

# Suppress Hugging Face Hub unauthenticated request warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

from malevolentslice.utils.memory import force_garbage_collection, get_memory_usage_mb


class AudioTranscriber:
    """
    Sequential, memory-bounded ASR transcriber using faster-whisper.
    Loads model only during the transcription phase with INT8 CPU quantization
    to keep memory usage bounded (< 150MB RAM for 'tiny', < 300MB for 'base').
    """

    def __init__(
        self,
        model_size: str = "tiny",
        language: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 2
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.model = None

    def load_model(self) -> None:
        """Loads the Whisper model into memory if not already loaded."""
        if self.model is None:
            # Strictly limit BLAS / OpenMP / Apple Accelerate thread pools before model load
            threads_str = str(self.cpu_threads)
            os.environ.setdefault("OMP_NUM_THREADS", threads_str)
            os.environ.setdefault("MKL_NUM_THREADS", threads_str)
            os.environ.setdefault("OPENBLAS_NUM_THREADS", threads_str)
            os.environ.setdefault("VECLIB_MAXIMUM_THREADS", threads_str)

            try:
                from faster_whisper import WhisperModel
            except ImportError as err:
                raise ImportError(
                    "Pustaka 'faster-whisper' belum terinstal. Silakan jalankan 'pip install faster-whisper' "
                    "atau instal ulang paket dengan 'pip install malevolentslice --upgrade'."
                ) from err

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=1
            )

    def unload_model(self) -> None:
        """Unloads model and reclaims memory."""
        if self.model is not None:
            del self.model
            self.model = None
            force_garbage_collection()

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribes a single audio file on disk.
        Returns the transcription string.
        """
        if self.model is None:
            self.load_model()

        if not os.path.exists(audio_path):
            return ""

        # beam_size=1 (greedy search) & condition_on_previous_text=False for bounded memory & speed
        segments, _ = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False
        )

        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(texts).strip()

    def transcribe_dataset(
        self,
        wav_items: List[Dict[str, str]],
        output_metadata_path: str,
        progress_callback: Optional[Callable[[int, int, str, str, float], None]] = None,
        resume: bool = False
    ) -> int:
        """
        Transcribes a list of audio items and writes to LJSpeech metadata.csv.
        wav_items is a list of dicts with keys:
          - 'id': relative metadata identifier (e.g. 'segment_000001')
          - 'path': absolute path to WAV file
        
        Writes line-by-line with immediate flushing to maintain O(1) memory bound.
        """
        self.load_model()

        # Check existing transcripts if resume is requested
        existing_transcripts: Dict[str, str] = {}
        if resume and os.path.exists(output_metadata_path):
            try:
                with open(output_metadata_path, mode="r", encoding="utf-8", errors="replace") as f_in:
                    reader = csv.reader(f_in, delimiter="|")
                    for row in reader:
                        if len(row) >= 2:
                            seg_id = row[0].strip()
                            text = row[1].strip()
                            # If it's not a dummy placeholder, keep it
                            if text and not text.startswith("audio segment "):
                                existing_transcripts[seg_id] = text
            except Exception:
                existing_transcripts = {}

        total = len(wav_items)
        transcribed_count = 0

        # We will write/rewrite metadata.csv
        # Using a temporary file first if not in resume-append mode, or direct file
        os.makedirs(os.path.dirname(os.path.abspath(output_metadata_path)), exist_ok=True)
        temp_meta_path = output_metadata_path + ".tmp"

        with open(temp_meta_path, mode="w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out, delimiter="|")

            for idx, item in enumerate(wav_items, start=1):
                seg_id = item["id"]
                wav_path = item["path"]

                if seg_id in existing_transcripts:
                    text = existing_transcripts[seg_id]
                else:
                    try:
                        text = self.transcribe_file(wav_path)
                    except Exception:
                        text = ""
                    transcribed_count += 1

                # LJSpeech standard: id|transcription|normalized_transcription
                writer.writerow([seg_id, text, text])
                f_out.flush()

                current_ram = get_memory_usage_mb()
                if progress_callback:
                    progress_callback(idx, total, seg_id, text, current_ram)

                force_garbage_collection()

        # Safely replace original metadata.csv with newly transcribed file
        if os.path.exists(output_metadata_path):
            try:
                os.remove(output_metadata_path)
            except Exception:
                pass
        os.replace(temp_meta_path, output_metadata_path)

        # Unload model after finishing to reclaim memory
        self.unload_model()
        return transcribed_count
