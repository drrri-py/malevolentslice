import gc
from typing import List, Generator, Optional, Tuple
import numpy as np

class AudioSegmenter:
    """
    Stateful segmenter with Hangover Protection and TTS duration constraints (1.0s to 12.0s).
    Accumulates speech frames, applies hangover protection to prevent acoustic clipping,
    and yields slice arrays when silence thresholds or max duration limits are met.
    """
    def __init__(
        self,
        speech_threshold: float = 0.5,
        min_speech_duration_ms: float = 1000.0,
        max_speech_duration_ms: float = 12000.0,
        min_silence_duration_ms: float = 400.0,
        hangover_frames: int = 8,
        sample_rate: int = 16000,
        frame_size: int = 512
    ):
        self.speech_threshold = speech_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.hangover_frames = hangover_frames
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        self.frame_duration_ms = (frame_size / float(sample_rate)) * 1000.0
        self.min_silence_frames = int(np.ceil(min_silence_duration_ms / self.frame_duration_ms))
        self.min_speech_samples = int(round((min_speech_duration_ms / 1000.0) * sample_rate))
        self.max_speech_samples = int(round((max_speech_duration_ms / 1000.0) * sample_rate))

        self.reset()

    def reset(self) -> None:
        """Reset internal segmenter buffers and state flags."""
        self.active_frames: List[np.ndarray] = []
        self.frame_probs: List[float] = []
        self.hangover_counter = 0
        self.consecutive_silence_frames = 0
        self.accumulated_samples = 0

    def process_frame(self, frame: np.ndarray, speech_prob: float) -> Optional[np.ndarray]:
        """
        Process a single 512-sample frame and VAD speech probability.
        Yields a full audio segment array if a valid slice point is reached, else None.
        """
        is_raw_speech = speech_prob >= self.speech_threshold

        if is_raw_speech:
            self.hangover_counter = self.hangover_frames
            effective_speech = True
        else:
            if self.hangover_counter > 0:
                self.hangover_counter -= 1
                effective_speech = True
            else:
                effective_speech = False

        if effective_speech:
            self.consecutive_silence_frames = 0
            self.active_frames.append(frame.copy())
            self.frame_probs.append(speech_prob)
            self.accumulated_samples += len(frame)

            # Check if max speech duration limit is reached (force-split required)
            if self.accumulated_samples >= self.max_speech_samples:
                return self._force_split()
        else:
            self.consecutive_silence_frames += 1
            if len(self.active_frames) > 0:
                # Include hangover / silence tail frames up to min_silence_frames
                self.active_frames.append(frame.copy())
                self.frame_probs.append(speech_prob)
                self.accumulated_samples += len(frame)

            if self.consecutive_silence_frames >= self.min_silence_frames:
                if self.accumulated_samples >= self.min_speech_samples:
                    return self._slice_and_reset()
                else:
                    # Silence exceeded but segment too short; discard buffer
                    self.reset()
                    gc.collect()

        return None

    def _slice_and_reset(self) -> np.ndarray:
        """Concatenates accumulated frames into a single segment array and resets buffer."""
        segment_array = np.concatenate(self.active_frames, axis=0).astype(np.float32)
        self.reset()
        gc.collect()
        return segment_array

    def _force_split(self) -> np.ndarray:
        """
        Force-splits buffer exceeding max duration at the frame with minimum VAD speech probability.
        Returns the first portion and keeps remainder in active_frames buffer.
        """
        # Find frame index with minimum speech probability in second half of active frames
        search_start = len(self.frame_probs) // 2
        min_idx = search_start + int(np.argmin(self.frame_probs[search_start:]))

        split_frames = self.active_frames[:min_idx]
        remainder_frames = self.active_frames[min_idx:]

        segment_array = np.concatenate(split_frames, axis=0).astype(np.float32)

        self.active_frames = remainder_frames
        self.frame_probs = self.frame_probs[min_idx:]
        self.accumulated_samples = sum(len(f) for f in self.active_frames)
        self.consecutive_silence_frames = 0
        self.hangover_counter = 0

        gc.collect()
        return segment_array

    def flush(self) -> Optional[np.ndarray]:
        """Flushes remaining frames at end of audio stream."""
        if self.accumulated_samples >= self.min_speech_samples:
            return self._slice_and_reset()
        else:
            self.reset()
            gc.collect()
            return None
