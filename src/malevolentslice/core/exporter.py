import os
import csv
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf

class DatasetExporter:
    """
    Exports audio slices into configurable dataset layouts:
    - Default LJSpeech standard: output_dir/wavs/segment_XXXXXX.wav
    - Flat mode (wav_subfolder=""): output_dir/segment_XXXXXX.wav
    - Preserve structure mode (preserve_structure=True): output_dir/wavs/subfolder/segment_XXXXXX.wav
    """
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        wav_subfolder: Optional[str] = "wavs",
        preserve_structure: bool = False,
        source_root: Optional[str] = None
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.sample_rate = sample_rate
        self.preserve_structure = preserve_structure
        self.source_root = os.path.abspath(source_root) if source_root else None
        
        if wav_subfolder:
            self.wavs_dir = os.path.join(self.output_dir, wav_subfolder)
        else:
            self.wavs_dir = self.output_dir

        self.metadata_path = os.path.join(self.output_dir, "metadata.csv")
        self.audit_log_path = os.path.join(self.output_dir, "processing_audit.log")
        
        os.makedirs(self.wavs_dir, exist_ok=True)
        self.segment_counter = 0

    def export_segment(
        self,
        segment_audio: np.ndarray,
        source_filename: str,
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Writes segment array to PCM_16 WAV file, appends LJSpeech metadata, and logs audit statistics.
        Returns dictionary with segment metadata.
        """
        self.segment_counter += 1
        if custom_id:
            segment_id = custom_id
        else:
            segment_id = f"segment_{self.segment_counter:06d}"

        wav_filename = f"{segment_id}.wav"

        # Handle preserve_structure directory hierarchy
        if self.preserve_structure and self.source_root and os.path.exists(source_filename):
            abs_source = os.path.abspath(source_filename)
            rel_dir = os.path.dirname(os.path.relpath(abs_source, self.source_root))
            target_dir = os.path.join(self.wavs_dir, rel_dir)
            os.makedirs(target_dir, exist_ok=True)
            wav_path = os.path.join(target_dir, wav_filename)
            rel_metadata_id = os.path.normpath(os.path.join(rel_dir, segment_id))
        else:
            wav_path = os.path.join(self.wavs_dir, wav_filename)
            rel_metadata_id = segment_id

        # Write float32 audio as 16-bit PCM WAV
        sf.write(wav_path, segment_audio, self.sample_rate, subtype='PCM_16')

        duration_sec = len(segment_audio) / float(self.sample_rate)
        rms = float(np.sqrt(np.mean(segment_audio ** 2))) if len(segment_audio) > 0 else 0.0
        snr_db = 20.0 * np.log10(rms + 1e-9)

        # Append to LJSpeech metadata.csv (delimiter '|')
        transcript_placeholder = f"audio segment {self.segment_counter:06d}"
        with open(self.metadata_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow([rel_metadata_id, transcript_placeholder, transcript_placeholder])

        # Append to processing_audit.log
        audit_msg = (
            f"[SEGMENT EXPORT] id={rel_metadata_id} source={os.path.basename(source_filename)} "
            f"duration={duration_sec:.2f}s samples={len(segment_audio)} "
            f"rms={rms:.4f} approx_snr={snr_db:.2f}dB\n"
        )
        with open(self.audit_log_path, mode="a", encoding="utf-8") as f:
            f.write(audit_msg)

        return {
            "segment_id": rel_metadata_id,
            "filename": wav_filename,
            "filepath": wav_path,
            "duration_sec": duration_sec,
            "rms": rms,
            "snr_db": snr_db,
            "source_file": source_filename
        }
