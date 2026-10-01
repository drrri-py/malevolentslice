"""
MalevolentSlice Studio - Desktop GUI
A modern, dark-mode desktop interface for memory-efficient TTS dataset segmentation & transcription.
Built with CustomTkinter and optimized for standalone PyInstaller packaging.
"""

import os
import sys
import time
import queue
import logging
import threading
from typing import Optional, List, Dict, Any

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
except ImportError:
    ctk = None  # Handled gracefully in main()

from malevolentslice.pipeline import Pipeline, PipelineSummary
from malevolentslice.utils.path_resolver import (
    get_available_vad_models,
    resolve_or_download_vad_model,
    get_models_cache_dir,
    is_frozen_bundle
)
from malevolentslice.utils.memory import get_memory_usage_mb, force_garbage_collection
from malevolentslice.core.transcriber import is_asr_model_cached, download_asr_model

# Supported Whisper model presets
WHISPER_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
]

# Supported language codes
LANGUAGE_OPTIONS = [
    ("Auto (Detect)", None),
    ("Indonesian (id)", "id"),
    ("English (en)", "en"),
    ("Japanese (ja)", "ja"),
    ("Chinese (zh)", "zh"),
    ("Spanish (es)", "es"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Korean (ko)", "ko"),
    ("Russian (ru)", "ru"),
    ("Portuguese (pt)", "pt"),
    ("Italian (it)", "it"),
    ("Arabic (ar)", "ar"),
]


class TextboxLogHandler(logging.Handler):
    """Logging handler that routes formatted log records to a thread-safe Queue."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)


class MalevolentSliceApp(ctk.CTk if ctk else object):
    """
    MalevolentSlice Desktop GUI Application.
    Provides responsive controls, live RAM profiling, real-time logging,
    and background worker threading for dataset slicing and ASR transcription.
    """

    def __init__(self):
        if ctk is None:
            raise RuntimeError("CustomTkinter is not installed. Run 'pip install customtkinter>=5.2.0'.")

        super().__init__()

        # Appearance & Theme
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("MalevolentSlice Studio — Audio Dataset Slicer & Transcriber")
        self.geometry("1140x840")
        self.minsize(980, 720)

        # Threading & Control Flags
        self.worker_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.is_processing = False
        self.log_queue: queue.Queue = queue.Queue()

        # Session RAM tracking
        self.session_start_ram = get_memory_usage_mb()
        self.session_peak_ram = self.session_start_ram

        # Setup Logging redirection
        self._setup_logging()

        # Initialize GUI Layout
        self._build_ui()

        # Start periodic polling (RAM, Log Queue)
        self._poll_log_queue()
        self._poll_ram()

        # Window close interception
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_logging(self) -> None:
        """Captures malevolentslice log records into our GUI queue."""
        self.log_handler = TextboxLogHandler(self.log_queue)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        self.log_handler.setFormatter(formatter)
        self.log_handler.setLevel(logging.INFO)

        target_logger = logging.getLogger("malevolentslice")
        target_logger.setLevel(logging.INFO)
        target_logger.addHandler(self.log_handler)

    # =========================================================================
    # UI Layout Construction
    # =========================================================================
    def _build_ui(self) -> None:
        # Root grid configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Header Bar
        self._build_header(row=0)

        # 2. Main Content Split View (Left: Settings, Right: Monitoring & Logs)
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.main_container.grid_columnconfigure(0, weight=6, minsize=480)
        self.main_container.grid_columnconfigure(1, weight=5, minsize=420)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Left Column: Scrollable Settings
        self._build_settings_column()

        # Right Column: Monitoring, Progress & Live Logs
        self._build_monitoring_column()

    def _build_header(self, row: int) -> None:
        header_frame = ctk.CTkFrame(self, height=64, corner_radius=10, fg_color=("#1f242d", "#151922"))
        header_frame.grid(row=row, column=0, sticky="ew", padx=16, pady=12)
        header_frame.grid_columnconfigure(0, weight=1)

        # Title & Subtitle in left stack
        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w", padx=16, pady=8)

        title_lbl = ctk.CTkLabel(
            title_box,
            text="MalevolentSlice Studio",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            title_box,
            text="Silero VAD ONNX Audio Slicing & Memory-Bounded Faster-Whisper ASR",
            font=ctk.CTkFont(size=12),
            text_color=("#8c92a4", "#9aa0b4")
        )
        sub_lbl.pack(anchor="w")

        # Status badge on right
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="STATUS: READY",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#22c55e", "#15803d"),
            text_color="#ffffff",
            corner_radius=6,
            padx=10,
            pady=4
        )
        self.status_badge.grid(row=0, column=1, sticky="e", padx=16, pady=8)

    def _build_settings_column(self) -> None:
        self.settings_scroll = ctk.CTkScrollableFrame(
            self.main_container,
            label_text="Configuration & Parameters",
            label_font=ctk.CTkFont(size=14, weight="bold")
        )
        self.settings_scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        self.settings_scroll.grid_columnconfigure(0, weight=1)

        # Section 1: File & Directory Selection Card
        self._build_io_card()

        # Section 2: VAD Engine Parameters Card
        self._build_vad_card()

        # Section 3: Pre-processing (Noise Gate) Card
        self._build_noisegate_card()

        # Section 4: ASR Transcription Card
        self._build_asr_card()

    def _build_io_card(self) -> None:
        card = ctk.CTkFrame(self.settings_scroll, corner_radius=8)
        card.pack(fill="x", padx=6, pady=6)
        card.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(card, text="1. Input & Output Targets", font=ctk.CTkFont(size=13, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 6))

        # Input mode switch (Single File vs Batch Directory)
        ctk.CTkLabel(card, text="Input Mode:").grid(row=1, column=0, sticky="w", padx=12, pady=4)
        self.input_mode_var = ctk.StringVar(value="Batch Directory")
        self.input_mode_seg = ctk.CTkSegmentedButton(
            card,
            values=["Single File", "Batch Directory"],
            variable=self.input_mode_var,
            command=self._on_input_mode_changed
        )
        self.input_mode_seg.grid(row=1, column=1, columnspan=2, sticky="ew", padx=12, pady=4)

        # Input Path
        self.input_lbl = ctk.CTkLabel(card, text="Audio Source:")
        self.input_lbl.grid(row=2, column=0, sticky="w", padx=12, pady=4)

        self.input_path_var = ctk.StringVar(value=os.path.abspath("./raw") if os.path.exists("./raw") else "")
        self.input_entry = ctk.CTkEntry(card, textvariable=self.input_path_var, placeholder_text="Select audio file or folder...")
        self.input_entry.grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=4)

        self.input_browse_btn = ctk.CTkButton(card, text="Browse", width=75, command=self._browse_input)
        self.input_browse_btn.grid(row=2, column=2, padx=(0, 12), pady=4)

        # Output Path
        ctk.CTkLabel(card, text="Output Folder:").grid(row=3, column=0, sticky="w", padx=12, pady=4)
        self.output_path_var = ctk.StringVar(value=os.path.abspath("./dataset"))
        self.output_entry = ctk.CTkEntry(card, textvariable=self.output_path_var, placeholder_text="Target dataset directory...")
        self.output_entry.grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=4)

        self.output_browse_btn = ctk.CTkButton(card, text="Browse", width=75, command=self._browse_output)
        self.output_browse_btn.grid(row=3, column=2, padx=(0, 12), pady=4)

        # Output Options (Structure preservation)
        self.preserve_struct_var = ctk.BooleanVar(value=False)
        self.preserve_struct_chk = ctk.CTkCheckBox(
            card,
            text="Preserve subfolder structure (mirror input hierarchy in dataset/wavs)",
            variable=self.preserve_struct_var
        )
        self.preserve_struct_chk.grid(row=4, column=0, columnspan=3, sticky="w", padx=12, pady=(4, 10))

    def _build_vad_card(self) -> None:
        card = ctk.CTkFrame(self.settings_scroll, corner_radius=8)
        card.pack(fill="x", padx=6, pady=6)
        card.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(card, text="2. Silero VAD (ONNX) Parameters", font=ctk.CTkFont(size=13, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 6))

        # Model Version Dropdown
        ctk.CTkLabel(card, text="VAD Model:").grid(row=1, column=0, sticky="w", padx=12, pady=4)
        vad_options = get_available_vad_models()
        self.vad_model_map = {m["name"]: m["path"] for m in vad_options}
        model_names = list(self.vad_model_map.keys())
        default_vad = model_names[0] if model_names else "Silero VAD v5 (Latest, Recommended)"

        self.vad_model_var = ctk.StringVar(value=default_vad)
        self.vad_dropdown = ctk.CTkOptionMenu(card, values=model_names, variable=self.vad_model_var)
        self.vad_dropdown.grid(row=1, column=1, columnspan=2, sticky="ew", padx=12, pady=4)

        # Speech Threshold Slider (0.1 - 0.9, default 0.5)
        self.threshold_val_lbl = ctk.CTkLabel(card, text="0.50", width=45)
        ctk.CTkLabel(card, text="Speech Threshold:").grid(row=2, column=0, sticky="w", padx=12, pady=4)
        self.threshold_slider = ctk.CTkSlider(
            card, from_=0.10, to=0.90, number_of_steps=80,
            command=lambda v: self.threshold_val_lbl.configure(text=f"{v:.2f}")
        )
        self.threshold_slider.set(0.50)
        self.threshold_slider.grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=4)
        self.threshold_val_lbl.grid(row=2, column=2, padx=(0, 12), pady=4)

        # Min Speech Duration Slider (200 - 3000 ms, default 1000)
        self.min_speech_val_lbl = ctk.CTkLabel(card, text="1000 ms", width=55)
        ctk.CTkLabel(card, text="Min Speech (ms):").grid(row=3, column=0, sticky="w", padx=12, pady=4)
        self.min_speech_slider = ctk.CTkSlider(
            card, from_=200, to=3000, number_of_steps=56,
            command=lambda v: self.min_speech_val_lbl.configure(text=f"{int(v)} ms")
        )
        self.min_speech_slider.set(1000)
        self.min_speech_slider.grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=4)
        self.min_speech_val_lbl.grid(row=3, column=2, padx=(0, 12), pady=4)

        # Min Silence Duration Slider (100 - 2000 ms, default 400)
        self.min_silence_val_lbl = ctk.CTkLabel(card, text="400 ms", width=55)
        ctk.CTkLabel(card, text="Min Silence (ms):").grid(row=4, column=0, sticky="w", padx=12, pady=4)
        self.min_silence_slider = ctk.CTkSlider(
            card, from_=100, to=2000, number_of_steps=38,
            command=lambda v: self.min_silence_val_lbl.configure(text=f"{int(v)} ms")
        )
        self.min_silence_slider.set(400)
        self.min_silence_slider.grid(row=4, column=1, sticky="ew", padx=(0, 6), pady=4)
        self.min_silence_val_lbl.grid(row=4, column=2, padx=(0, 12), pady=4)

        # Speech Pad (Hangover) Slider (0 - 600 ms, default 250)
        self.speech_pad_val_lbl = ctk.CTkLabel(card, text="250 ms", width=55)
        ctk.CTkLabel(card, text="Speech Pad (ms):").grid(row=5, column=0, sticky="w", padx=12, pady=4)
        self.speech_pad_slider = ctk.CTkSlider(
            card, from_=0, to=600, number_of_steps=24,
            command=lambda v: self.speech_pad_val_lbl.configure(text=f"{int(v)} ms")
        )
        self.speech_pad_slider.set(250)
        self.speech_pad_slider.grid(row=5, column=1, sticky="ew", padx=(0, 6), pady=4)
        self.speech_pad_val_lbl.grid(row=5, column=2, padx=(0, 12), pady=4)

        # Max Speech Duration Slider (2000 - 30000 ms, default 12000)
        self.max_speech_val_lbl = ctk.CTkLabel(card, text="12.0 s", width=55)
        ctk.CTkLabel(card, text="Max Speech:").grid(row=6, column=0, sticky="w", padx=12, pady=(4, 10))
        self.max_speech_slider = ctk.CTkSlider(
            card, from_=2000, to=30000, number_of_steps=56,
            command=lambda v: self.max_speech_val_lbl.configure(text=f"{v/1000.0:.1f} s")
        )
        self.max_speech_slider.set(12000)
        self.max_speech_slider.grid(row=6, column=1, sticky="ew", padx=(0, 6), pady=(4, 10))
        self.max_speech_val_lbl.grid(row=6, column=2, padx=(0, 12), pady=(4, 10))

    def _build_noisegate_card(self) -> None:
        card = ctk.CTkFrame(self.settings_scroll, corner_radius=8)
        card.pack(fill="x", padx=6, pady=6)
        card.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(card, text="3. Noise Gate Pre-processing", font=ctk.CTkFont(size=13, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 6))

        # Toggle Switch
        self.enable_gate_var = ctk.BooleanVar(value=True)
        self.gate_switch = ctk.CTkSwitch(
            card,
            text="Enable In-Memory Noise Gate",
            variable=self.enable_gate_var,
            command=self._on_noise_gate_toggled
        )
        self.gate_switch.grid(row=1, column=0, columnspan=3, sticky="w", padx=12, pady=4)

        # Noise Gate dB Slider (-80 dB to -10 dB, default -40 dB)
        self.gate_db_val_lbl = ctk.CTkLabel(card, text="-40 dB", width=55)
        self.gate_lbl = ctk.CTkLabel(card, text="Gate Threshold:")
        self.gate_lbl.grid(row=2, column=0, sticky="w", padx=12, pady=(4, 10))

        self.gate_db_slider = ctk.CTkSlider(
            card, from_=-80, to=-10, number_of_steps=70,
            command=lambda v: self.gate_db_val_lbl.configure(text=f"{int(v)} dB")
        )
        self.gate_db_slider.set(-40)
        self.gate_db_slider.grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=(4, 10))
        self.gate_db_val_lbl.grid(row=2, column=2, padx=(0, 12), pady=(4, 10))

    def _build_asr_card(self) -> None:
        card = ctk.CTkFrame(self.settings_scroll, corner_radius=8)
        card.pack(fill="x", padx=6, pady=6)
        card.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(card, text="4. ASR Transcription (faster-whisper)", font=ctk.CTkFont(size=13, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 6))

        # Toggle Transcription
        self.enable_trans_var = ctk.BooleanVar(value=True)
        self.trans_switch = ctk.CTkSwitch(
            card,
            text="Enable Automatic Transcription (LJSpeech metadata.csv)",
            variable=self.enable_trans_var,
            command=self._on_transcription_toggled
        )
        self.trans_switch.grid(row=1, column=0, columnspan=3, sticky="w", padx=12, pady=4)

        # Model Preset Dropdown
        self.model_lbl = ctk.CTkLabel(card, text="Whisper Model:")
        self.model_lbl.grid(row=2, column=0, sticky="w", padx=12, pady=4)

        # Build dropdown items with cache indicator
        model_display_names = self._get_whisper_display_options()
        self.whisper_model_var = ctk.StringVar(value=model_display_names[0])
        self.whisper_dropdown = ctk.CTkOptionMenu(
            card,
            values=model_display_names,
            variable=self.whisper_model_var,
            command=self._on_whisper_model_changed
        )
        self.whisper_dropdown.grid(row=2, column=1, columnspan=2, sticky="ew", padx=12, pady=4)

        # Language Dropdown
        self.lang_lbl = ctk.CTkLabel(card, text="Spoken Language:")
        self.lang_lbl.grid(row=3, column=0, sticky="w", padx=12, pady=4)

        self.lang_display_names = [opt[0] for opt in LANGUAGE_OPTIONS]
        self.lang_map = {opt[0]: opt[1] for opt in LANGUAGE_OPTIONS}
        self.lang_var = ctk.StringVar(value=self.lang_display_names[0])
        self.lang_dropdown = ctk.CTkOptionMenu(card, values=self.lang_display_names, variable=self.lang_var)
        self.lang_dropdown.grid(row=3, column=1, columnspan=2, sticky="ew", padx=12, pady=4)

        # Compute Device & Quantization
        ctk.CTkLabel(card, text="Compute Device:").grid(row=4, column=0, sticky="w", padx=12, pady=4)
        self.device_var = ctk.StringVar(value="cpu")
        self.device_seg = ctk.CTkSegmentedButton(card, values=["cpu", "cuda"], variable=self.device_var)
        self.device_seg.grid(row=4, column=1, columnspan=2, sticky="ew", padx=12, pady=4)

        ctk.CTkLabel(card, text="Quantization:").grid(row=5, column=0, sticky="w", padx=12, pady=4)
        self.compute_type_var = ctk.StringVar(value="int8")
        self.compute_type_seg = ctk.CTkSegmentedButton(
            card,
            values=["int8", "float16", "float32"],
            variable=self.compute_type_var
        )
        self.compute_type_seg.grid(row=5, column=1, columnspan=2, sticky="ew", padx=12, pady=4)

        # CPU Threads Slider (1 - 8, default 2)
        self.cpu_threads_val_lbl = ctk.CTkLabel(card, text="2 threads", width=65)
        self.cpu_threads_lbl = ctk.CTkLabel(card, text="CPU Threads:")
        self.cpu_threads_lbl.grid(row=6, column=0, sticky="w", padx=12, pady=(4, 10))

        self.cpu_threads_slider = ctk.CTkSlider(
            card, from_=1, to=8, number_of_steps=7,
            command=lambda v: self.cpu_threads_val_lbl.configure(text=f"{int(v)} threads")
        )
        self.cpu_threads_slider.set(2)
        self.cpu_threads_slider.grid(row=6, column=1, sticky="ew", padx=(0, 6), pady=(4, 10))
        self.cpu_threads_val_lbl.grid(row=6, column=2, padx=(0, 12), pady=(4, 10))

    def _get_whisper_display_options(self) -> List[str]:
        """Generates dropdown entries with dynamic cache indicators."""
        options = []
        for m in WHISPER_MODELS:
            cached = is_asr_model_cached(m)
            badge = "[Cached]" if cached else "[Download]"
            options.append(f"{m} {badge}")
        return options

    # =========================================================================
    # Right Column: Monitoring, Progress & Live Logs
    # =========================================================================
    def _build_monitoring_column(self) -> None:
        self.monitor_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.monitor_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        self.monitor_frame.grid_columnconfigure(0, weight=1)
        self.monitor_frame.grid_rowconfigure(3, weight=1)

        # 1. Action Buttons Card
        self._build_action_card()

        # 2. Dual Progress Card
        self._build_progress_card()

        # 3. Live RAM Profile Card
        self._build_ram_card()

        # 4. Terminal Log Console
        self._build_console_card()

    def _build_action_card(self) -> None:
        action_card = ctk.CTkFrame(self.monitor_frame, corner_radius=8)
        action_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        action_card.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(
            action_card,
            text="▶ Start Processing",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            fg_color="#16a34a",
            hover_color="#15803d",
            command=self._start_pipeline
        )
        self.start_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.cancel_btn = ctk.CTkButton(
            action_card,
            text="■ Cancel / Stop",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            fg_color="#dc2626",
            hover_color="#b91c1c",
            state="disabled",
            command=self._cancel_pipeline
        )
        self.cancel_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    def _build_progress_card(self) -> None:
        prog_card = ctk.CTkFrame(self.monitor_frame, corner_radius=8)
        prog_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        prog_card.grid_columnconfigure(0, weight=1)

        # Overall / Batch progress
        self.overall_lbl = ctk.CTkLabel(
            prog_card,
            text="Overall Progress: Idle",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        self.overall_lbl.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))

        self.overall_bar = ctk.CTkProgressBar(prog_card, height=12)
        self.overall_bar.set(0.0)
        self.overall_bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        # Current Subtask progress
        self.subtask_lbl = ctk.CTkLabel(
            prog_card,
            text="Task: Waiting for user to start...",
            font=ctk.CTkFont(size=11),
            text_color=("#8c92a4", "#9aa0b4"),
            anchor="w"
        )
        self.subtask_lbl.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 2))

        self.subtask_bar = ctk.CTkProgressBar(prog_card, height=8, progress_color="#3b82f6")
        self.subtask_bar.set(0.0)
        self.subtask_bar.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))

    def _build_ram_card(self) -> None:
        ram_card = ctk.CTkFrame(self.monitor_frame, corner_radius=8)
        ram_card.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ram_card.grid_columnconfigure((0, 1, 2), weight=1)

        # Metric 1: Current RSS / phys_footprint
        m1_frame = ctk.CTkFrame(ram_card, fg_color="transparent")
        m1_frame.grid(row=0, column=0, padx=8, pady=8)
        ctk.CTkLabel(m1_frame, text="Current RAM", font=ctk.CTkFont(size=10)).pack()
        self.cur_ram_lbl = ctk.CTkLabel(
            m1_frame,
            text=f"{self.session_start_ram:.1f} MB",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#38bdf8"
        )
        self.cur_ram_lbl.pack()

        # Metric 2: Peak Session RAM
        m2_frame = ctk.CTkFrame(ram_card, fg_color="transparent")
        m2_frame.grid(row=0, column=1, padx=8, pady=8)
        ctk.CTkLabel(m2_frame, text="Peak RAM", font=ctk.CTkFont(size=10)).pack()
        self.peak_ram_lbl = ctk.CTkLabel(
            m2_frame,
            text=f"{self.session_peak_ram:.1f} MB",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#f59e0b"
        )
        self.peak_ram_lbl.pack()

        # Metric 3: Memory Growth
        m3_frame = ctk.CTkFrame(ram_card, fg_color="transparent")
        m3_frame.grid(row=0, column=2, padx=8, pady=8)
        ctk.CTkLabel(m3_frame, text="Net Growth", font=ctk.CTkFont(size=10)).pack()
        self.growth_ram_lbl = ctk.CTkLabel(
            m3_frame,
            text="+0.0 MB",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#22c55e"
        )
        self.growth_ram_lbl.pack()

    def _build_console_card(self) -> None:
        console_frame = ctk.CTkFrame(self.monitor_frame, corner_radius=8)
        console_frame.grid(row=3, column=0, sticky="nsew", pady=0)
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_rowconfigure(1, weight=1)

        # Header with action buttons
        header = ctk.CTkFrame(console_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(header, text="Console & Pipeline Logs", font=ctk.CTkFont(size=12, weight="bold"))
        title.grid(row=0, column=0, sticky="w")

        btn_box = ctk.CTkFrame(header, fg_color="transparent")
        btn_box.grid(row=0, column=1, sticky="e")

        copy_btn = ctk.CTkButton(btn_box, text="Copy", width=55, height=24, command=self._copy_logs)
        copy_btn.pack(side="left", padx=3)

        clear_btn = ctk.CTkButton(btn_box, text="Clear", width=55, height=24, fg_color="#4b5563", hover_color="#374151", command=self._clear_logs)
        clear_btn.pack(side="left", padx=3)

        # Read-only Dark Terminal Textbox
        self.log_textbox = ctk.CTkTextbox(
            console_frame,
            font=ctk.CTkFont(family="Courier" if sys.platform == "darwin" else "Consolas", size=11),
            fg_color="#0d1117",
            text_color="#c9d1d9",
            wrap="none"
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    # =========================================================================
    # User Interaction Handlers
    # =========================================================================
    def _on_input_mode_changed(self, mode: str) -> None:
        if mode == "Single File":
            self.input_lbl.configure(text="Audio File:")
            self.input_entry.configure(placeholder_text="Select audio file (.wav, .mp3, .flac)...")
        else:
            self.input_lbl.configure(text="Audio Folder:")
            self.input_entry.configure(placeholder_text="Select audio directory...")

    def _browse_input(self) -> None:
        mode = self.input_mode_var.get()
        if mode == "Single File":
            path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All Files", "*.*")]
            )
        else:
            path = filedialog.askdirectory(title="Select Audio Directory")

        if path:
            self.input_path_var.set(os.path.abspath(path))

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_path_var.set(os.path.abspath(path))

    def _on_noise_gate_toggled(self) -> None:
        is_on = self.enable_gate_var.get()
        state = "normal" if is_on else "disabled"
        self.gate_db_slider.configure(state=state)

    def _on_transcription_toggled(self) -> None:
        is_on = self.enable_trans_var.get()
        state = "normal" if is_on else "disabled"
        self.whisper_dropdown.configure(state=state)
        self.lang_dropdown.configure(state=state)
        self.device_seg.configure(state=state)
        self.compute_type_seg.configure(state=state)
        self.cpu_threads_slider.configure(state=state)

    def _on_whisper_model_changed(self, val: str) -> None:
        # Refresh options cache tags if needed
        pass

    def _clear_logs(self) -> None:
        self.log_textbox.delete("1.0", "end")

    def _copy_logs(self) -> None:
        logs = self.log_textbox.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(logs)
        messagebox.showinfo("Copied", "Logs copied to clipboard!")

    # =========================================================================
    # Pipeline Execution & Background Worker
    # =========================================================================
    def _start_pipeline(self) -> None:
        source_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()

        if not source_path:
            messagebox.showerror("Error", "Please select an audio file or directory.")
            return

        if not os.path.exists(source_path):
            messagebox.showerror("Error", f"Input path does not exist:\n{source_path}")
            return

        if not output_path:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        # Prepare UI state
        self.is_processing = True
        self.cancel_event.clear()

        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_badge.configure(text="STATUS: RUNNING", fg_color=("#3b82f6", "#1d4ed8"))
        self.overall_bar.set(0.0)
        self.subtask_bar.set(0.0)
        self.overall_lbl.configure(text="Overall Progress: Initializing...")
        self.subtask_lbl.configure(text="Preparing pipeline components...")

        # Parse selected configurations
        vad_choice_name = self.vad_model_var.get()
        vad_model_path = self.vad_model_map.get(vad_choice_name, vad_choice_name)

        whisper_raw = self.whisper_model_var.get().split()[0]
        selected_lang = self.lang_map.get(self.lang_var.get())

        config = {
            "source": source_path,
            "output_dir": output_path,
            "speech_threshold": float(self.threshold_slider.get()),
            "min_speech_duration_ms": float(self.min_speech_slider.get()),
            "min_silence_duration_ms": float(self.min_silence_slider.get()),
            "speech_pad_ms": float(self.speech_pad_slider.get()),
            "max_speech_duration_ms": float(self.max_speech_slider.get()),
            "enable_noise_gate": self.enable_gate_var.get(),
            "noise_threshold_db": float(self.gate_db_slider.get()),
            "model_path": vad_model_path,
            "transcribe": self.enable_trans_var.get(),
            "whisper_model": whisper_raw,
            "language": selected_lang,
            "device": self.device_var.get(),
            "compute_type": self.compute_type_var.get(),
            "cpu_threads": int(self.cpu_threads_slider.get()),
            "preserve_structure": self.preserve_struct_var.get(),
        }

        # Spawn non-blocking background thread
        self.worker_thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(config,),
            daemon=True
        )
        self.worker_thread.start()

    def _cancel_pipeline(self) -> None:
        if self.is_processing:
            self.cancel_event.set()
            self.cancel_btn.configure(state="disabled")
            self.status_badge.configure(text="STATUS: CANCELLING", fg_color=("#ea580c", "#c2410c"))
            self.subtask_lbl.configure(text="Cancellation signal sent. Halting workers...")

    def _run_pipeline_worker(self, cfg: Dict[str, Any]) -> None:
        """Executed inside background thread."""
        logger = logging.getLogger("malevolentslice")
        logger.info("Starting pipeline execution worker...")

        try:
            # 1. Check & Download VAD model if needed
            vad_path = cfg["model_path"]
            if not os.path.exists(vad_path):
                self._update_subtask_ui("Resolving / Downloading Silero VAD model...", 0.3)
                cfg["model_path"] = resolve_or_download_vad_model(vad_path)

            if self.cancel_event.is_set():
                self._finish_pipeline(cancelled=True)
                return

            # 2. Check & Lazy-Download Whisper model if transcription enabled
            if cfg["transcribe"]:
                model_name = cfg["whisper_model"]
                if not is_asr_model_cached(model_name):
                    logger.info(f"Whisper model '{model_name}' not cached. Starting download...")
                    self._update_overall_ui("Preparing ASR Model", 0.05)
                    self._update_subtask_ui(f"Downloading Whisper '{model_name}' (this happens only once)...", 0.5)

                    download_asr_model(model_name)
                    logger.info(f"Whisper model '{model_name}' downloaded successfully.")

                    # Refresh dropdown display tags
                    self.after(0, self._refresh_whisper_cache_tags)

            if self.cancel_event.is_set():
                self._finish_pipeline(cancelled=True)
                return

            # 3. Instantiate high-level Pipeline
            pipeline = Pipeline(
                noise_threshold_db=cfg["noise_threshold_db"],
                speech_threshold=cfg["speech_threshold"],
                min_silence_duration_ms=cfg["min_silence_duration_ms"],
                min_speech_duration_ms=cfg["min_speech_duration_ms"],
                max_speech_duration_ms=cfg["max_speech_duration_ms"],
                speech_pad_ms=cfg["speech_pad_ms"],
                enable_noise_gate=cfg["enable_noise_gate"],
                preserve_structure=cfg["preserve_structure"],
                model_path=cfg["model_path"],
                transcribe=cfg["transcribe"],
                whisper_model=cfg["whisper_model"],
                language=cfg["language"],
                device=cfg["device"],
                compute_type=cfg["compute_type"],
                cpu_threads=cfg["cpu_threads"],
                log_to_console=False
            )

            # Progress callbacks
            def on_file_progress(filename: str, frac: float, dur: float, ram: float, cur_file: int = 1, total_files: int = 1):
                overall_frac = (cur_file - 1 + frac) / max(1, total_files)
                # Slicing is stage 1 (0 to 50% if transcription enabled, else 0 to 100%)
                scale = 0.5 if cfg["transcribe"] else 1.0
                display_overall = overall_frac * scale
                self._update_overall_ui(f"Stage 1 (Slicing) - File {cur_file}/{total_files} ({int(overall_frac*100)}%)", display_overall)
                self._update_subtask_ui(f"Slicing: {filename} ({int(frac*100)}%)", frac)

            def on_trans_progress(cur_item: int, total_items: int, seg_id: str, text: str, ram: float):
                trans_frac = cur_item / max(1, total_items)
                display_overall = 0.5 + (trans_frac * 0.5)
                self._update_overall_ui(f"Stage 2 (Transcription) - {cur_item}/{total_items} ({int(trans_frac*100)}%)", display_overall)
                snippet = (text[:28] + "..") if len(text) > 30 else text
                self._update_subtask_ui(f"Transcribing [{cur_item}/{total_items}] {seg_id}: \"{snippet}\"", trans_frac)

            summary: PipelineSummary = pipeline.process(
                source=cfg["source"],
                output_dir=cfg["output_dir"],
                progress_callback=on_file_progress,
                transcription_callback=on_trans_progress,
                is_cancelled=lambda: self.cancel_event.is_set()
            )

            if self.cancel_event.is_set():
                self._finish_pipeline(cancelled=True)
            else:
                self._finish_pipeline(cancelled=False, summary=summary)

        except Exception as e:
            logger.exception(f"Unhandled error during pipeline processing: {e}")
            self._finish_pipeline(cancelled=False, error=str(e))

    def _update_overall_ui(self, text: str, frac: float) -> None:
        self.after(0, lambda: (self.overall_lbl.configure(text=text), self.overall_bar.set(max(0.0, min(1.0, frac)))))

    def _update_subtask_ui(self, text: str, frac: float) -> None:
        self.after(0, lambda: (self.subtask_lbl.configure(text=text), self.subtask_bar.set(max(0.0, min(1.0, frac)))))

    def _refresh_whisper_cache_tags(self) -> None:
        current_selection = self.whisper_model_var.get().split()[0]
        options = self._get_whisper_display_options()
        self.whisper_dropdown.configure(values=options)
        for opt in options:
            if opt.startswith(current_selection + " "):
                self.whisper_model_var.set(opt)
                break

    def _finish_pipeline(self, cancelled: bool = False, summary: Optional[PipelineSummary] = None, error: Optional[str] = None) -> None:
        def callback():
            self.is_processing = False
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

            if cancelled:
                self.status_badge.configure(text="STATUS: CANCELLED", fg_color=("#ea580c", "#c2410c"))
                self.overall_lbl.configure(text="Overall Progress: Cancelled")
                self.subtask_lbl.configure(text="Processing was aborted by user.")
                messagebox.showwarning("Cancelled", "Processing was stopped by user.")
            elif error:
                self.status_badge.configure(text="STATUS: ERROR", fg_color=("#dc2626", "#b91c1c"))
                self.overall_lbl.configure(text="Overall Progress: Failed")
                self.subtask_lbl.configure(text=f"Error: {error}")
                messagebox.showerror("Execution Error", f"An error occurred during processing:\n\n{error}")
            else:
                self.status_badge.configure(text="STATUS: COMPLETED", fg_color=("#16a34a", "#15803d"))
                self.overall_bar.set(1.0)
                self.subtask_bar.set(1.0)
                if summary:
                    msg = (
                        f"Processed {summary.total_files} file(s) into {summary.total_segments} audio slices "
                        f"in {summary.elapsed_time:.1f}s.\nPeak RAM: {summary.peak_memory_mb:.1f} MB "
                        f"(Growth: +{summary.memory_growth_mb:.1f} MB).\n"
                        f"Transcribed: {summary.transcribed_segments} segment(s)."
                    )
                    self.overall_lbl.configure(text=f"Completed: {summary.total_segments} segments generated.")
                    self.subtask_lbl.configure(text=f"Done in {summary.elapsed_time:.2f}s | Peak RAM: {summary.peak_memory_mb:.1f} MB")
                    messagebox.showinfo("Success", f"Processing completed successfully!\n\n{msg}")
                else:
                    self.overall_lbl.configure(text="Completed.")
                    self.subtask_lbl.configure(text="All tasks finished.")

            force_garbage_collection()

        self.after(0, callback)

    # =========================================================================
    # Periodic GUI Pollers (RAM Profiling & Queue Logs)
    # =========================================================================
    def _poll_log_queue(self) -> None:
        """Polls log queue and appends new messages into CTkTextbox."""
        try:
            records = []
            while not self.log_queue.empty():
                records.append(self.log_queue.get_nowait())

            if records:
                self.log_textbox.configure(state="normal")
                for r in records:
                    self.log_textbox.insert("end", r + "\n")
                self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")
        except Exception:
            pass
        finally:
            self.after(100, self._poll_log_queue)

    def _poll_ram(self) -> None:
        """Polls live RAM consumption and updates the monitoring card."""
        try:
            cur_ram = get_memory_usage_mb()
            if cur_ram > self.session_peak_ram:
                self.session_peak_ram = cur_ram

            growth = max(0.0, self.session_peak_ram - self.session_start_ram)

            self.cur_ram_lbl.configure(text=f"{cur_ram:.1f} MB")
            self.peak_ram_lbl.configure(text=f"{self.session_peak_ram:.1f} MB")
            self.growth_ram_lbl.configure(text=f"+{growth:.1f} MB")

            # Health color coding
            if cur_ram < 350.0:
                self.cur_ram_lbl.configure(text_color="#38bdf8")
            elif cur_ram < 800.0:
                self.cur_ram_lbl.configure(text_color="#facc15")
            else:
                self.cur_ram_lbl.configure(text_color="#f87171")
        except Exception:
            pass
        finally:
            self.after(500, self._poll_ram)

    def _on_close(self) -> None:
        """Handles application shutdown gracefully."""
        if self.is_processing:
            if messagebox.askyesno("Exit Confirmation", "Processing is currently running. Do you want to cancel and exit?"):
                self.cancel_event.set()
                self.destroy()
        else:
            self.destroy()


def main():
    """Desktop GUI Entrypoint."""
    if ctk is None:
        print("[malevolentslice] Error: CustomTkinter is not installed.")
        print("Please install GUI dependencies via: pip install malevolentslice[gui] or pip install customtkinter")
        sys.exit(1)

    app = MalevolentSliceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
