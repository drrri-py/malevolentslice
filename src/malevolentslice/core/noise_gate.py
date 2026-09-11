import numpy as np

class NoiseGate:
    """
    NumPy-vectorized noise gate for filtering ambient background noise.
    Applies in-place clamping below a linear amplitude threshold calculated from dB.
    """
    def __init__(self, threshold_db: float = -40.0):
        self.threshold_db = threshold_db
        self.threshold_linear = 10.0 ** (threshold_db / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies noise gating in-place or zero-copy on input 1D numpy array.
        Values with absolute amplitude below threshold_linear are set to 0.0.
        """
        if audio.size == 0:
            return audio
        mask = np.abs(audio) < self.threshold_linear
        audio[mask] = 0.0
        return audio

def apply_noise_gate(audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
    """Convenience function to apply noise gate to an audio array."""
    gate = NoiseGate(threshold_db=threshold_db)
    return gate.process(audio)
