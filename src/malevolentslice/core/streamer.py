import os
import gc
from typing import Generator, Tuple
import numpy as np
import soundfile as sf

class AudioStreamer:
    """
    Lazy loading audio generator using soundfile.SoundFile.blocks().
    Maintains O(1) memory bound by yielding chunks of audio arrays and freeing references.
    """
    def __init__(self, file_path: str, chunk_buffer_mb: float = 5.0, target_sr: int = 16000):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        self.file_path = file_path
        self.chunk_buffer_mb = chunk_buffer_mb
        self.target_sr = target_sr
        
        # Read metadata without reading file content into memory
        with sf.SoundFile(self.file_path) as f:
            self.source_sr = f.samplerate
            self.channels = f.channels
            self.total_frames = len(f)
            self.duration_seconds = self.total_frames / float(self.source_sr)

        # Calculate chunk size in frames based on MB (assuming float32: 4 bytes per sample per channel)
        bytes_per_frame = 4 * self.channels
        buffer_bytes = int(self.chunk_buffer_mb * 1024 * 1024)
        self.chunk_frames = max(self.source_sr, buffer_bytes // bytes_per_frame)

    def stream_chunks(self) -> Generator[Tuple[int, np.ndarray, int, bool], None, None]:
        """
        Yields (chunk_index, audio_array, sample_rate, is_last).
        audio_array is 1D float32 normalized mono audio at target_sr.
        """
        with sf.SoundFile(self.file_path) as f:
            chunk_index = 0
            # Read blocks sequentially from disk
            for block in f.blocks(blocksize=self.chunk_frames, dtype='float32', always_2d=True):
                chunk_index += 1
                is_last = f.tell() >= self.total_frames
                
                # Mono downmixing (mean across channels)
                if block.shape[1] > 1:
                    mono_audio = np.mean(block, axis=1, dtype=np.float32)
                else:
                    mono_audio = block[:, 0].astype(np.float32)
                
                # Release block reference immediately
                del block

                # Resample to target_sr if necessary
                if self.source_sr != self.target_sr:
                    num_input_samples = len(mono_audio)
                    num_output_samples = int(round(num_input_samples * self.target_sr / self.source_sr))
                    if num_output_samples > 0:
                        x_old = np.arange(num_input_samples, dtype=np.float32)
                        x_new = np.linspace(0, num_input_samples - 1, num_output_samples, dtype=np.float32)
                        resampled_audio = np.interp(x_new, x_old, mono_audio).astype(np.float32)
                        del mono_audio
                        audio_out = resampled_audio
                    else:
                        audio_out = mono_audio
                else:
                    audio_out = mono_audio

                yield (chunk_index, audio_out, self.target_sr, is_last)
                
                # Explicit cleanup after yield
                del audio_out
                gc.collect()

def stream_audio_chunks(
    file_path: str,
    chunk_buffer_mb: float = 5.0,
    target_sr: int = 16000
) -> Generator[Tuple[int, np.ndarray, int, bool], None, None]:
    """Convenience generator function for streaming audio chunks."""
    streamer = AudioStreamer(file_path=file_path, chunk_buffer_mb=chunk_buffer_mb, target_sr=target_sr)
    return streamer.stream_chunks()
