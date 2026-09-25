"""
MalevolentSlice: Voice Activity Detection (VAD)-Based Audio Dataset Segmentation System
Memory-efficient TTS Corpus Generation with O(1) Bounded Memory Footprint.
"""

from malevolentslice.pipeline import Pipeline, PipelineSummary, find_audio_files
from malevolentslice.core import (
    AudioStreamer,
    stream_audio_chunks,
    NoiseGate,
    apply_noise_gate,
    SileroVAD,
    AudioSegmenter,
    DatasetExporter,
)
from malevolentslice.utils import force_garbage_collection, get_memory_usage_mb

__version__ = "0.1.2"

__all__ = [
    "Pipeline",
    "PipelineSummary",
    "find_audio_files",
    "AudioStreamer",
    "stream_audio_chunks",
    "NoiseGate",
    "apply_noise_gate",
    "SileroVAD",
    "AudioSegmenter",
    "DatasetExporter",
    "force_garbage_collection",
    "get_memory_usage_mb",
    "__version__",
]
