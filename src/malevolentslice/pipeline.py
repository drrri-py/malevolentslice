import os
import time
import glob
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Union
import numpy as np

from malevolentslice.core.streamer import AudioStreamer
from malevolentslice.core.noise_gate import NoiseGate
from malevolentslice.core.vad_engine import SileroVAD
from malevolentslice.core.segmenter import AudioSegmenter
from malevolentslice.core.exporter import DatasetExporter
from malevolentslice.utils.memory import MemoryTracker, force_garbage_collection
from malevolentslice.utils.logger import get_logger

VALID_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

def find_audio_files(
    source_dir: str,
    max_depth: int = 3,
    exclude_dirs: Optional[Union[List[str], set]] = None
) -> List[str]:
    """
    Scans directory tree for audio files up to max_depth subfolder levels deep.
    Prunes deeper directories to protect against symlink cycles or deep nested traversals.
    Excludes hidden folders (.git, .venv, etc.) and specified output directories.
    """
    source_dir = os.path.abspath(source_dir)
    audio_files: List[str] = []
    base_depth = source_dir.rstrip(os.sep).count(os.sep)

    ignored_names = {".venv", "venv", ".git", ".gemini", "__pycache__", "build", "dist", "node_modules", "dataset"}
    if exclude_dirs:
        ignored_names.update(d.lower() for d in exclude_dirs)

    for root, dirs, files in os.walk(source_dir, followlinks=False):
        # Exclude hidden directories and system/output folders so os.walk does not enter them
        dirs[:] = [
            d for d in dirs
            if not d.startswith('.') and d.lower() not in ignored_names
        ]

        current_depth = root.count(os.sep) - base_depth
        if current_depth >= max_depth:
            dirs.clear()  # Do not descend deeper than max_depth

        for file in files:
            if file.lower().endswith(VALID_AUDIO_EXTENSIONS):
                audio_files.append(os.path.join(root, file))

    return sorted(audio_files)

@dataclass
class PipelineSummary:
    total_files: int
    total_segments: int
    total_processed_duration: float
    elapsed_time: float
    peak_memory_mb: float
    memory_growth_mb: float

class Pipeline:
    """
    High-level MalevolentSlice Pipeline. Orchestrates disk-to-buffer block streaming,
    NumPy noise gating, Silero-VAD frame inferencing, segment framing, and dataset exporting.
    """
    def __init__(
        self,
        noise_threshold_db: float = -40.0,
        speech_threshold: float = 0.5,
        min_silence_duration_ms: float = 400.0,
        min_speech_duration_ms: float = 1000.0,
        max_speech_duration_ms: float = 12000.0,
        chunk_buffer_mb: float = 5.0,
        export_format: str = "wav",
        target_sr: int = 16000,
        max_depth: int = 3,
        wav_subfolder: Optional[str] = "wavs",
        preserve_structure: bool = False,
        model_path: Optional[str] = None,
        log_file: Optional[str] = None,
        log_to_console: bool = True
    ):
        self.noise_threshold_db = noise_threshold_db
        self.speech_threshold = speech_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.chunk_buffer_mb = chunk_buffer_mb
        self.export_format = export_format
        self.target_sr = target_sr
        self.max_depth = max_depth
        self.wav_subfolder = wav_subfolder
        self.preserve_structure = preserve_structure
        self.model_path = model_path
        self.log_to_console = log_to_console
        self.logger = get_logger("malevolentslice", log_file=log_file, console_output=log_to_console)

    def process_file(
        self,
        file_path: str,
        exporter: DatasetExporter,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        memory_tracker: Optional[MemoryTracker] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single audio file end-to-end with bounded memory footprint.
        Returns list of exported segment metadata dicts.
        """
        streamer = AudioStreamer(
            file_path=file_path,
            chunk_buffer_mb=self.chunk_buffer_mb,
            target_sr=self.target_sr
        )
        noise_gate = NoiseGate(threshold_db=self.noise_threshold_db)
        vad = SileroVAD(model_path=self.model_path, sample_rate=self.target_sr)
        segmenter = AudioSegmenter(
            speech_threshold=self.speech_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_ms=self.max_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            sample_rate=self.target_sr
        )

        exported_segments: List[Dict[str, Any]] = []
        processed_frames_count = 0
        
        try:
            for _, audio_chunk, _, _ in streamer.stream_chunks():
                # Apply fast noise gate
                gated_chunk = noise_gate.process(audio_chunk)
                
                # Process in 512-sample VAD frames
                frame_size = vad.frame_size
                num_frames = len(gated_chunk) // frame_size

                for i in range(num_frames):
                    frame = gated_chunk[i * frame_size : (i + 1) * frame_size]
                    prob = vad.predict_frame(frame)
                    segment_arr = segmenter.process_frame(frame, prob)
                    
                    if segment_arr is not None:
                        seg_meta = exporter.export_segment(segment_arr, source_filename=file_path)
                        exported_segments.append(seg_meta)
                        del segment_arr
                        
                    if memory_tracker:
                        memory_tracker.update()

                # Process remaining samples in chunk if any
                rem = len(gated_chunk) % frame_size
                if rem > 0:
                    pad_frame = np.zeros(frame_size, dtype=np.float32)
                    pad_frame[:rem] = gated_chunk[-rem:]
                    prob = vad.predict_frame(pad_frame)
                    segment_arr = segmenter.process_frame(pad_frame, prob)
                    if segment_arr is not None:
                        seg_meta = exporter.export_segment(segment_arr, source_filename=file_path)
                        exported_segments.append(seg_meta)
                        del segment_arr

                processed_frames_count += len(audio_chunk)
                if progress_callback and streamer.duration_seconds > 0:
                    progress_callback(processed_frames_count / (streamer.duration_seconds * streamer.source_sr), streamer.duration_seconds)
                
                del gated_chunk
                del audio_chunk
                force_garbage_collection()

            # Flush any remaining speech segment
            final_segment = segmenter.flush()
            if final_segment is not None:
                seg_meta = exporter.export_segment(final_segment, source_filename=file_path)
                exported_segments.append(seg_meta)
                del final_segment

        finally:
            force_garbage_collection()

        return exported_segments

    def process(
        self,
        source: Optional[Union[str, List[str]]] = None,
        output_dir: str = "./dataset/",
        max_depth: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float, float, float], None]] = None
    ) -> PipelineSummary:
        """
        Process a list of audio files or a directory of raw audio recordings up to max_depth.
        If source is None, automatically resolves from './raw/' or searches current directory.
        Returns a PipelineSummary containing execution stats and peak RAM usage.
        """
        start_time = time.time()
        mem_tracker = MemoryTracker()
        source_root_path: Optional[str] = None
        effective_max_depth = self.max_depth if max_depth is None else max_depth
        out_name = os.path.basename(os.path.abspath(output_dir)).lower()

        if source is None:
            if os.path.exists("./raw") and os.path.isdir("./raw"):
                source = "./raw"
            else:
                source = "."

        if isinstance(source, str):
            if os.path.isdir(source):
                source_root_path = os.path.abspath(source)
                files = find_audio_files(source, max_depth=effective_max_depth, exclude_dirs=[out_name])
            elif os.path.isfile(source):
                source_root_path = os.path.dirname(os.path.abspath(source))
                files = [source]
            else:
                # If path doesn't exist but is default "./raw", fallback to current directory
                if source in ("./raw", "./raw/", "raw") and os.path.isdir("."):
                    source_root_path = os.path.abspath(".")
                    files = find_audio_files(".", max_depth=effective_max_depth, exclude_dirs=[out_name])
                else:
                    self.logger.warning(f"Audio source does not exist: {source}")
                    files = []
        else:
            files = [f for f in source if os.path.isfile(f)]
            if files:
                source_root_path = os.path.dirname(os.path.commonpath([os.path.abspath(f) for f in files]))

        if not files:
            self.logger.warning(f"No audio files found at source: {source}")
            return PipelineSummary(
                total_files=0,
                total_segments=0,
                total_processed_duration=0.0,
                elapsed_time=time.time() - start_time,
                peak_memory_mb=mem_tracker.get_peak_mb(),
                memory_growth_mb=mem_tracker.get_growth_mb()
            )

        exporter = DatasetExporter(
            output_dir=output_dir,
            sample_rate=self.target_sr,
            wav_subfolder=self.wav_subfolder,
            preserve_structure=self.preserve_structure,
            source_root=source_root_path
        )
        total_segments_count = 0
        total_duration_sec = 0.0

        for file_idx, file_path in enumerate(files):
            self.logger.info(f"Processing [{file_idx + 1}/{len(files)}]: {os.path.basename(file_path)}")
            
            def file_progress(frac: float, file_dur: float):
                if progress_callback:
                    try:
                        progress_callback(
                            os.path.basename(file_path),
                            frac,
                            file_dur,
                            mem_tracker.update(),
                            file_idx + 1,
                            len(files)
                        )
                    except TypeError:
                        progress_callback(os.path.basename(file_path), frac, file_dur, mem_tracker.update())

            segments = self.process_file(
                file_path=file_path,
                exporter=exporter,
                progress_callback=file_progress,
                memory_tracker=mem_tracker
            )
            total_segments_count += len(segments)
            if segments:
                total_duration_sec += sum(s["duration_sec"] for s in segments)
                
            force_garbage_collection()

        elapsed = time.time() - start_time
        peak_ram = mem_tracker.get_peak_mb()
        growth_ram = mem_tracker.get_growth_mb()

        self.logger.info(
            f"Pipeline complete: {total_segments_count} segments generated from {len(files)} files "
            f"in {elapsed:.2f}s (Peak RAM: {peak_ram:.2f} MB, Growth: +{growth_ram:.2f} MB)"
        )

        return PipelineSummary(
            total_files=len(files),
            total_segments=total_segments_count,
            total_processed_duration=total_duration_sec,
            elapsed_time=elapsed,
            peak_memory_mb=peak_ram,
            memory_growth_mb=growth_ram
        )
