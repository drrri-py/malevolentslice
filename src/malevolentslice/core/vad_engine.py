import os
import urllib.request
import numpy as np
import onnxruntime as ort
from typing import Optional, Dict, Any

SILERO_VAD_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v4.0/files/silero_vad.onnx"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "silero_vad.onnx")

def ensure_model_exists(model_path: Optional[str] = None) -> str:
    """
    Ensures that the Silero VAD ONNX model file exists.
    Checks user-specified path, local project assets, package assets, and ~/.cache/malevolentslice.
    Downloads automatically to user cache if not found locally.
    """
    if model_path:
        abs_path = os.path.abspath(model_path)
        if os.path.exists(abs_path):
            return abs_path

    # 1. Check local repository assets directory
    local_assets = os.path.abspath(DEFAULT_MODEL_PATH)
    if os.path.exists(local_assets):
        return local_assets

    # 2. Check package-internal assets directory
    pkg_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "silero_vad.onnx"))
    if os.path.exists(pkg_assets):
        return pkg_assets

    # 3. Check user cache directory (~/.cache/malevolentslice/silero_vad.onnx)
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "malevolentslice")
    cache_path = os.path.join(cache_dir, "silero_vad.onnx")
    if os.path.exists(cache_path):
        return cache_path

    # 4. Download into user cache directory
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[malevolentslice] Downloading Silero VAD ONNX model to {cache_path}...")
    urllib.request.urlretrieve(SILERO_VAD_URL, cache_path)
    return cache_path

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
            feed: Dict[str, Any] = {
                'input': tensor_frame,
                'state': self._state,
                'sr': sr_tensor
            }
            outs = self.session.run(None, feed)
            prob = float(outs[0][0][0])
            self._state = outs[1]
        else:
            feed = {
                'input': tensor_frame,
                'sr': sr_tensor,
                'h': self._h,
                'c': self._c
            }
            outs = self.session.run(None, feed)
            prob = float(outs[0][0][0])
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
