import os
import urllib.request
import numpy as np
import onnxruntime as ort
from typing import Optional, Dict, Any

from malevolentslice.utils.path_resolver import resolve_or_download_vad_model, resolve_asset_path

def ensure_model_exists(model_path: Optional[str] = None) -> str:
    """
    Ensures that the Silero VAD ONNX model file exists using the robust path resolver.
    Checks PyInstaller bundle, package assets, repo assets, and user cache directory.
    Downloads automatically if not found locally.
    """
    return resolve_or_download_vad_model(model_path)

class SileroVAD:
    """
    Silero VAD ONNX Runtime Wrapper.
    Operates frame-by-frame on 512-sample chunks (at 16,000 Hz) with stateful RNN context.
    Strictly single-threaded CPU execution to prevent thread pool overhead.
    """
    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000):
        self.model_path = ensure_model_exists(model_path)
        self.sample_rate = sample_rate
        self.frame_size = 512 if sample_rate == 16000 else 256  # 512 samples for 16kHz (32ms)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_v5 = 'state' in self.input_names
        self.reset_states()

    def reset_states(self) -> None:
        """Reset internal recurrent state tensors for a new audio stream."""
        if self.is_v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            context_size = 64 if self.sample_rate == 16000 else 32
            self._context = np.zeros((1, context_size), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def predict_frame(self, frame: np.ndarray) -> float:
        """
        Predict speech probability for a single 512-sample float32 audio frame.
        Returns probability float in range [0.0, 1.0].
        """
        if frame.ndim == 1:
            tensor_frame = np.expand_dims(frame, axis=0).astype(np.float32)
        else:
            tensor_frame = frame.astype(np.float32)

        sr_tensor = np.array(self.sample_rate, dtype=np.int64)

        if self.is_v5:
            # Silero VAD v5 requires a 64-sample rolling temporal context buffer (576 samples total)
            model_input = np.concatenate([self._context, tensor_frame], axis=1).astype(np.float32)
            feed: Dict[str, Any] = {
                'input': model_input,
                'state': self._state,
                'sr': sr_tensor
            }
            outs = self.session.run(None, feed)
            prob = float(outs[0].flat[0])
            self._state = outs[1]
            context_size = 64 if self.sample_rate == 16000 else 32
            self._context = model_input[:, -context_size:]
        else:
            feed = {
                'input': tensor_frame,
                'sr': sr_tensor,
                'h': self._h,
                'c': self._c
            }
            outs = self.session.run(None, feed)
            prob = float(outs[0].flat[0])
            self._h = outs[1]
            self._c = outs[2]

        return prob

    def close(self) -> None:
        """Explicitly release ONNX Runtime session and state buffers to reclaim memory."""
        if hasattr(self, "session") and self.session is not None:
            del self.session
            self.session = None
        if hasattr(self, "_state"):
            del self._state
        if hasattr(self, "_h"):
            del self._h
        if hasattr(self, "_c"):
            del self._c
