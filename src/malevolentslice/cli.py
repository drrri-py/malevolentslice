import os
import sys
import time
import logging
import warnings
from typing import Optional

# Suppress Hugging Face unauthenticated request warnings & disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn
)

from malevolentslice.pipeline import Pipeline, find_audio_files, VALID_AUDIO_EXTENSIONS
from malevolentslice import __version__

# Ensure UTF-8 compatibility on Windows terminal without encoding crashes
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

console = Console(highlight=False)


def format_duration(seconds: float) -> str:
    """Format duration in seconds into a clean, human-readable string."""
    if seconds <= 0:
        return "0.00s"
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs}h {mins}m {secs:.1f}s"
    elif mins > 0:
        return f"{mins}m {secs:.1f}s"
    else:
        return f"{secs:.2f}s"


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="malevolentslice")
@click.pass_context
def main(ctx):
    """MalevolentSlice: Memory-efficient VAD-based audio dataset segmentation system for TTS corpus generation."""
    if ctx.invoked_subcommand is None:
        banner = Text()
        banner.append("MalevolentSlice ", style="bold cyan")
        banner.append(f"v{__version__}\n", style="dim cyan")
        banner.append("Memory-Efficient VAD Audio Dataset Segmenter for TTS Corpus Generation\n\n", style="dim")
        banner.append("Use ", style="dim")
        banner.append("malevolentslice run --help", style="bold white")
        banner.append(" to view all available options and parameters.", style="dim")
        console.print(Panel(banner, box=box.ROUNDED, border_style="cyan", padding=(0, 2)))


@main.command(name="run")
@click.option("--input", "-i", "input_path", default=None, help="Input file or directory (default: auto-detect from './raw/' or current directory up to max-depth).")
@click.option("--output", "-o", "output_dir", default="./dataset/", help="Output directory for LJSpeech dataset layout.")
@click.option("--threshold", "-t", "threshold_db", type=float, default=-40.0, help="Noise gate threshold in dB (default: -40.0).")
@click.option("--speech-threshold", type=float, default=0.5, help="VAD speech probability threshold (default: 0.5).")
@click.option("--silence", "-s", "silence_ms", type=float, default=400.0, help="Minimum silence duration in ms (default: 400.0).")
@click.option("--buffer", "-b", "buffer_mb", type=float, default=5.0, help="Chunk buffer size in MB (default: 5.0).")
@click.option("--min-speech", "min_speech_ms", type=float, default=1000.0, help="Minimum speech segment duration in ms (default: 1000.0).")
@click.option("--max-speech", "max_speech_ms", type=float, default=12000.0, help="Maximum speech segment duration in ms (default: 12000.0).")
@click.option("--sr", "target_sr", type=int, default=16000, help="Target sample rate in Hz (default: 16000).")
@click.option("--wav-dir", "wav_subfolder", default="wavs", help="Subfolder name for output WAV files (default: 'wavs'). Set empty '' for flat layout.")
@click.option("--flat", is_flag=True, help="Export WAV files directly into output root without subfolder.")
@click.option("--preserve-structure", "-p", "preserve_structure", is_flag=True, help="Mirror input folder directory hierarchy inside output directory.")
@click.option("--max-depth", type=int, default=3, help="Maximum directory depth limit for scanning input files (default: 3).")
@click.option("--quiet", "-q", is_flag=True, help="Run quietly without interactive progress or banner.")
@click.option("--transcribe/--no-transcribe", default=True, help="Enable/disable automatic Stage 2 speech-to-text transcription for metadata.csv (default: True).")
@click.option("--model", "-m", "whisper_model", default="tiny", type=click.Choice(["tiny", "base", "small", "medium"]), help="Whisper model size for transcription (default: 'tiny').")
@click.option("--language", "-l", default=None, help="Language code for transcription (e.g. 'id', 'en'). Auto-detected if omitted.")
def run_command(
    input_path: Optional[str],
    output_dir: str,
    threshold_db: float,
    speech_threshold: float,
    silence_ms: float,
    buffer_mb: float,
    min_speech_ms: float,
    max_speech_ms: float,
    target_sr: int,
    wav_subfolder: str,
    flat: bool,
    preserve_structure: bool,
    max_depth: int,
    quiet: bool,
    transcribe: bool,
    whisper_model: str,
    language: Optional[str]
):
    """Run VAD segmentation pipeline to produce LJSpeech corpus."""
    chosen_wav_subfolder = "" if flat else wav_subfolder
    out_dirname = os.path.basename(os.path.abspath(output_dir)).lower()

    # 1. Resolve Input Path & Auto-discovery
    auto_discovered = False
    if input_path is None or input_path in ("./raw/", "./raw", "raw"):
        if os.path.isdir("./raw"):
            input_path = "./raw"
        elif os.path.isdir("raw"):
            input_path = "raw"
        else:
            # Auto-detect audio files recursively from current working directory up to max_depth
            input_path = "."
            auto_discovered = True

    if not os.path.exists(input_path):
        if not quiet:
            console.print(Panel(
                f"[yellow]Input path does not exist:[/yellow]\n"
                f"[bold white]{os.path.abspath(input_path)}[/bold white]\n\n"
                f"[dim]Please specify a valid audio file or directory with '--input <path>'.[/dim]",
                title="[bold yellow]⚠ Input Not Found[/bold yellow]",
                title_align="left",
                box=box.ROUNDED,
                border_style="yellow",
                padding=(0, 2)
            ))
        else:
            print(f"Error: Input path '{input_path}' does not exist.")
        return

    # Check for audio files before displaying progress
    if os.path.isdir(input_path):
        discovered_files = find_audio_files(input_path, max_depth=max_depth, exclude_dirs=[out_dirname])
        total_discovered = len(discovered_files)
        if total_discovered == 0:
            if not quiet:
                auto_hint = " (auto-scanned current directory)" if auto_discovered else ""
                console.print(Panel(
                    f"[yellow]No supported audio files found in source directory{auto_hint}:[/yellow]\n"
                    f"[bold white]{os.path.abspath(input_path)}[/bold white]\n\n"
                    f"[dim]Supported formats: {', '.join(VALID_AUDIO_EXTENSIONS)}\n"
                    f"Scanned directory up to {max_depth} subfolder level(s) deep.\n"
                    f"Tip: Verify audio extensions or increase '--max-depth' (current: {max_depth}) if files are nested in deeper subfolders.[/dim]",
                    title="[bold yellow]⚠ No Audio Files Found[/bold yellow]",
                    title_align="left",
                    box=box.ROUNDED,
                    border_style="yellow",
                    padding=(0, 2)
                ))
            else:
                print(f"Warning: No audio files found in '{input_path}'")
            return
    elif os.path.isfile(input_path):
        if not input_path.lower().endswith(VALID_AUDIO_EXTENSIONS):
            if not quiet:
                console.print(Panel(
                    f"[yellow]The specified file is not a supported audio format:[/yellow]\n"
                    f"[bold white]{os.path.abspath(input_path)}[/bold white]\n\n"
                    f"[dim]Supported formats: {', '.join(VALID_AUDIO_EXTENSIONS)}[/dim]",
                    title="[bold yellow]⚠ Unsupported Format[/bold yellow]",
                    title_align="left",
                    box=box.ROUNDED,
                    border_style="yellow",
                    padding=(0, 2)
                ))
            else:
                print(f"Error: Unsupported audio format '{input_path}'")
            return
        total_discovered = 1
    else:
        total_discovered = 0

    # 2. Header Banner & Config Panel (skipped in quiet mode)
    if not quiet:
        banner = Text()
        banner.append("MalevolentSlice ", style="bold cyan")
        banner.append(f"v{__version__}\n", style="dim cyan")
        banner.append("Memory-Efficient VAD Audio Dataset Segmenter for TTS Corpus Generation", style="dim")
        console.print(Panel(banner, box=box.ROUNDED, border_style="cyan", padding=(0, 2)))

        layout_desc = f"Subfolder ('{chosen_wav_subfolder}')" if chosen_wav_subfolder else "Flat root directory"
        structure_desc = f"Mirrored hierarchy (max depth: {max_depth})" if preserve_structure else f"Single directory (max depth: {max_depth})"

        config_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2), expand=True)
        config_table.add_column("Key", style="dim cyan", width=22)
        config_table.add_column("Value", style="bold white")

        source_label = f"{os.path.normpath(input_path)} [dim]({total_discovered} file(s) found)[/dim]"
        if auto_discovered:
            source_label += f" [dim cyan](auto-detected, depth \u2264 {max_depth})[/dim cyan]"
        config_table.add_row("Input Source", source_label)
        config_table.add_row("Output Directory", f"{os.path.normpath(output_dir)} [dim]({layout_desc})[/dim]")
        config_table.add_row("Structure / Depth", structure_desc)
        config_table.add_row("Audio Specs", f"{target_sr:,} Hz · 16-bit PCM WAV")
        config_table.add_row(
            "VAD & Duration",
            f"Gate: {threshold_db:.1f} dB · Silence: {silence_ms:.0f} ms · Range: {min_speech_ms/1000.0:.1f}s – {max_speech_ms/1000.0:.1f}s"
        )
        if transcribe:
            lang_label = language if language else "Auto-detect"
            config_table.add_row(
                "Transcription",
                f"Enabled (Stage 2) · Model: '{whisper_model}' (INT8 CPU) · Lang: {lang_label}"
            )

        config_panel = Panel(
            config_table,
            title="[bold cyan]Pipeline Configuration[/bold cyan]",
            title_align="left",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1)
        )
        console.print(config_panel)
        console.print()

    # 3. Pipeline Initialization
    audit_log_path = os.path.join(output_dir, "processing_audit.log")
    pipeline = Pipeline(
        noise_threshold_db=threshold_db,
        speech_threshold=speech_threshold,
        min_silence_duration_ms=silence_ms,
        min_speech_duration_ms=min_speech_ms,
        max_speech_duration_ms=max_speech_ms,
        chunk_buffer_mb=buffer_mb,
        target_sr=target_sr,
        max_depth=max_depth,
        wav_subfolder=chosen_wav_subfolder,
        preserve_structure=preserve_structure,
        log_file=audit_log_path,
        log_to_console=False,
        transcribe=transcribe,
        whisper_model=whisper_model,
        language=language
    )

    # 4. Execution with Interactive Progress Display
    if not quiet:
        with Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold white]{task.description}[/bold white]"),
            BarColumn(bar_width=25, style="dim white", complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[dim]•[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]•[/dim]"),
            TextColumn("{task.fields[extra]}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[bold cyan][Stage 1: 0/{total_discovered}][/bold cyan] Initializing...",
                total=100,
                extra="[dim]Monitoring RAM...[/dim]"
            )

            def progress_cb(
                file_name: str,
                fraction: float,
                file_dur: float,
                current_ram: float,
                current_file_idx: int = 1,
                total_files: int = 1
            ):
                overall_frac = ((current_file_idx - 1) + fraction) / max(1, total_files)
                overall_pct = min(99, int(overall_frac * 100))

                display_name = file_name if len(file_name) <= 24 else f"{file_name[:21]}..."
                ram_badge = (
                    f"[cyan]RAM: {current_ram:.1f} MB[/cyan] [green](STABLE)[/green]"
                    if current_ram <= 100.0
                    else f"[yellow]RAM: {current_ram:.1f} MB[/yellow] [bold yellow](ELEVATED)[/bold yellow]"
                )
                progress.update(
                    task,
                    completed=overall_pct,
                    description=f"[bold cyan][Stage 1: {current_file_idx}/{total_files}][/bold cyan] {display_name}",
                    extra=ram_badge
                )

            def trans_cb(
                current_idx: int,
                total_segs: int,
                seg_id: str,
                text: str,
                current_ram: float
            ):
                pct = int((current_idx / max(1, total_segs)) * 100)
                disp_text = text if len(text) <= 28 else f"{text[:25]}..."
                ram_badge = (
                    f"[cyan]RAM: {current_ram:.1f} MB[/cyan]"
                    if current_ram <= 400.0
                    else f"[yellow]RAM: {current_ram:.1f} MB[/yellow]"
                )
                progress.update(
                    task,
                    completed=pct,
                    description=f"[bold magenta][Stage 2: {current_idx}/{total_segs}][/bold magenta] {seg_id} [dim]\"{disp_text}\"[/dim]",
                    extra=ram_badge
                )

            summary = pipeline.process(
                source=input_path,
                output_dir=output_dir,
                progress_callback=progress_cb,
                transcription_callback=trans_cb
            )

            ram_badge_final = (
                f"[cyan]Peak RAM: {summary.peak_memory_mb:.1f} MB[/cyan] [green](STABLE)[/green]"
                if summary.peak_memory_mb <= 400.0
                else f"[yellow]Peak RAM: {summary.peak_memory_mb:.1f} MB[/yellow] [yellow](ELEVATED)[/yellow]"
            )
            progress.update(
                task,
                completed=100,
                description=f"[bold green]✔ Sliced {summary.total_files} file(s)[/bold green]",
                extra=ram_badge_final
            )
        console.print()
    else:
        summary = pipeline.process(
            source=input_path,
            output_dir=output_dir,
            progress_callback=None,
            transcription_callback=None
        )

    # 5. Handle Zero Speech Segments Edge Case
    if summary.total_segments == 0:
        if not quiet:
            console.print(Panel(
                f"[yellow]Processed {summary.total_files} audio file(s), but 0 speech segments met the duration criteria.[/yellow]\n\n"
                f"[dim]• Noise Gate Threshold: {threshold_db:.1f} dB\n"
                f"• Speech Probability Threshold: {speech_threshold:.2f}\n"
                f"• Duration Bounds: {min_speech_ms/1000.0:.1f}s – {max_speech_ms/1000.0:.1f}s\n\n"
                f"Tip: Try adjusting '--threshold' (e.g. -45.0 dB) or '--speech-threshold' (e.g. 0.3) if speech volume is low.[/dim]",
                title="[bold yellow]⚠ No Speech Segments Generated[/bold yellow]",
                title_align="left",
                box=box.ROUNDED,
                border_style="yellow",
                padding=(0, 2)
            ))
        else:
            print(f"Notice: 0 speech segments generated from {summary.total_files} file(s).")
        return

    # 6. Clean Execution Summary Card
    if not quiet:
        summary_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2), expand=True)
        summary_table.add_column("Key", style="dim green", width=22)
        summary_table.add_column("Value", style="bold white")

        avg_seg_dur = (summary.total_processed_duration / summary.total_segments) if summary.total_segments > 0 else 0.0
        rtf = (summary.elapsed_time / summary.total_processed_duration) if summary.total_processed_duration > 0 else 0.0
        speed_text = f"{1.0 / rtf:.1f}x real-time [dim](RTF: {rtf:.4f})[/dim]" if rtf > 0 else "N/A"

        if summary.peak_memory_mb <= 400.0:
            ram_status = f"[bold green]{summary.peak_memory_mb:.2f} MB[/bold green] [green](STABLE)[/green] [dim]envelope <= 400 MB[/dim]"
        else:
            ram_status = f"[bold yellow]{summary.peak_memory_mb:.2f} MB[/bold yellow] [bold yellow](ELEVATED)[/bold yellow] [dim]> 400 MB threshold[/dim]"

        summary_table.add_row("Processed Files", f"{summary.total_files} audio file(s)")
        summary_table.add_row(
            "Generated Segments",
            f"[bold white]{summary.total_segments:,}[/bold white] segments [dim](avg {avg_seg_dur:.1f}s / segment)[/dim]"
        )
        if summary.transcribed_segments > 0:
            summary_table.add_row(
                "Transcribed Segments",
                f"[bold green]{summary.transcribed_segments:,}[/bold green] segments [dim](Real Speech-to-Text)[/dim]"
            )
        summary_table.add_row(
            "Total Speech Time",
            f"{format_duration(summary.total_processed_duration)} [dim]({summary.total_processed_duration:.1f}s)[/dim]"
        )
        summary_table.add_row("Execution Time", f"{format_duration(summary.elapsed_time)} [dim]· Speed: {speed_text}[/dim]")
        summary_table.add_row("Peak RAM Usage", ram_status)
        summary_table.add_row("Dataset Location", f"[bold cyan]{os.path.abspath(output_dir)}[/bold cyan]")

        metadata_rel = os.path.join(output_dir, "metadata.csv")
        wav_rel = os.path.join(output_dir, chosen_wav_subfolder) if chosen_wav_subfolder else output_dir
        summary_table.add_row("Layout & Artifacts", f"[dim]{os.path.normpath(wav_rel)} · {os.path.normpath(metadata_rel)}[/dim]")

        summary_panel = Panel(
            summary_table,
            title="[bold green]✔ Processing Finished Successfully[/bold green]",
            title_align="left",
            box=box.ROUNDED,
            border_style="green",
            padding=(0, 1)
        )
        console.print(summary_panel)
    else:
        print(f"Success: {summary.total_segments} segments generated from {summary.total_files} file(s) in {summary.elapsed_time:.2f}s -> {output_dir}")


@main.command(name="transcribe")
@click.option("--dataset", "-d", "dataset_dir", default="./dataset/", help="Path to dataset directory containing wavs/ and metadata.csv")
@click.option("--model", "-m", "model_size", default="tiny", type=click.Choice(["tiny", "base", "small", "medium"]), help="Whisper model size (default: 'tiny').")
@click.option("--language", "-l", default=None, help="Language code (e.g. 'id' for Indonesian, 'en' for English). Auto-detected if omitted.")
@click.option("--resume/--overwrite", default=True, help="Skip segments that already have real transcripts in metadata.csv (default: resume).")
@click.option("--quiet", "-q", is_flag=True, help="Run without interactive rich progress display.")
def transcribe_command(
    dataset_dir: str,
    model_size: str,
    language: Optional[str],
    resume: bool,
    quiet: bool
):
    """
    Transcribes an existing dataset directory to update metadata.csv with real speech-to-text.
    Maintains O(1) bounded memory (< 150MB RAM for 'tiny', < 300MB for 'base').
    """
    dataset_dir = os.path.abspath(dataset_dir)
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    wavs_dir = os.path.join(dataset_dir, "wavs")
    if not os.path.isdir(wavs_dir):
        wavs_dir = dataset_dir

    if not os.path.exists(wavs_dir):
        if not quiet:
            console.print(Panel(
                f"[yellow]WAV directory does not exist:[/yellow]\n[bold white]{wavs_dir}[/bold white]",
                title="[bold yellow]⚠ Dataset Not Found[/bold yellow]",
                box=box.ROUNDED,
                border_style="yellow"
            ))
        else:
            print(f"Error: Directory '{wavs_dir}' not found.")
        return

    # Find WAV files
    wav_items = []
    for root, _, files in os.walk(wavs_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                full_path = os.path.join(root, f)
                rel_id = os.path.normpath(os.path.relpath(full_path, wavs_dir))
                if rel_id.lower().endswith(".wav"):
                    rel_id = rel_id[:-4]
                wav_items.append({"id": rel_id, "path": full_path})

    wav_items.sort(key=lambda x: x["id"])
    total_files = len(wav_items)

    if total_files == 0:
        if not quiet:
            console.print(Panel(
                f"[yellow]No .wav files found in:[/yellow]\n[bold white]{wavs_dir}[/bold white]",
                title="[bold yellow]⚠ No WAV Files Found[/bold yellow]",
                box=box.ROUNDED,
                border_style="yellow"
            ))
        else:
            print("Error: No WAV files found.")
        return

    start_time = time.time()
    from malevolentslice.core.transcriber import AudioTranscriber
    from malevolentslice.utils.memory import MemoryTracker

    mem_tracker = MemoryTracker()
    transcriber = AudioTranscriber(
        model_size=model_size,
        language=language,
        device="cpu",
        compute_type="int8"
    )

    if not quiet:
        banner = Text()
        banner.append("MalevolentSlice Transcriber ", style="bold magenta")
        banner.append(f"v{__version__}\n", style="dim magenta")
        banner.append("Sequential Memory-Bounded ASR Dataset Transcriber", style="dim")
        console.print(Panel(banner, box=box.ROUNDED, border_style="magenta", padding=(0, 2)))

        lang_label = language if language else "Auto-detect"
        mode_label = "Resume existing" if resume else "Overwrite all"
        cfg_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2), expand=True)
        cfg_table.add_column("Key", style="dim magenta", width=22)
        cfg_table.add_column("Value", style="bold white")
        cfg_table.add_row("Dataset Directory", f"{os.path.normpath(dataset_dir)} [dim]({total_files} WAVs)[/dim]")
        cfg_table.add_row("ASR Model", f"Whisper '{model_size}' (INT8 CPU Quantization)")
        cfg_table.add_row("Language", lang_label)
        cfg_table.add_row("Mode", mode_label)
        console.print(Panel(cfg_table, title="[bold magenta]Transcriber Configuration[/bold magenta]", box=box.ROUNDED, border_style="magenta", padding=(0, 1)))
        console.print(f"[dim cyan]Loading Whisper '{model_size}' model into memory...[/dim cyan]")
        transcriber.load_model()
        console.print()

        with Progress(
            SpinnerColumn(spinner_name="dots", style="magenta"),
            TextColumn("[bold white]{task.description}[/bold white]"),
            BarColumn(bar_width=25, style="dim white", complete_style="magenta", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[dim]•[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]•[/dim]"),
            TextColumn("{task.fields[extra]}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[bold magenta][0/{total_files}][/bold magenta] Starting transcription...",
                total=100,
                extra="[dim]Monitoring RAM...[/dim]"
            )

            def trans_cb(idx: int, total: int, seg_id: str, text: str, ram: float):
                pct = int((idx / max(1, total)) * 100)
                disp_text = text if len(text) <= 28 else f"{text[:25]}..."
                ram_badge = (
                    f"[cyan]RAM: {ram:.1f} MB[/cyan]"
                    if ram <= 400.0
                    else f"[yellow]RAM: {ram:.1f} MB[/yellow]"
                )
                progress.update(
                    task,
                    completed=pct,
                    description=f"[bold magenta][{idx}/{total}][/bold magenta] {seg_id} [dim]\"{disp_text}\"[/dim]",
                    extra=ram_badge
                )

            transcribed_count = transcriber.transcribe_dataset(
                wav_items=wav_items,
                output_metadata_path=metadata_path,
                progress_callback=trans_cb,
                resume=resume
            )

            progress.update(
                task,
                completed=100,
                description=f"[bold green]✔ Finished transcribing {total_files} segment(s)[/bold green]",
                extra=f"[cyan]Peak RAM: {mem_tracker.get_peak_mb():.1f} MB[/cyan]"
            )
        console.print()
    else:
        transcribed_count = transcriber.transcribe_dataset(
            wav_items=wav_items,
            output_metadata_path=metadata_path,
            progress_callback=None,
            resume=resume
        )

    elapsed = time.time() - start_time
    if not quiet:
        summary_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2), expand=True)
        summary_table.add_column("Key", style="dim green", width=22)
        summary_table.add_column("Value", style="bold white")
        summary_table.add_row("Total Segments", f"{total_files:,} segments")
        summary_table.add_row("Newly Transcribed", f"{transcribed_count:,} segments")
        summary_table.add_row("Execution Time", f"{format_duration(elapsed)}")
        summary_table.add_row("Peak RAM Usage", f"[bold green]{mem_tracker.get_peak_mb():.2f} MB[/bold green] [green](STABLE)[/green]")
        summary_table.add_row("Metadata Saved", f"[bold cyan]{metadata_path}[/bold cyan]")
        console.print(Panel(summary_table, title="[bold green]✔ Transcription Finished Successfully[/bold green]", box=box.ROUNDED, border_style="green", padding=(0, 1)))
    else:
        print(f"Success: Transcribed {transcribed_count}/{total_files} segments in {elapsed:.2f}s -> {metadata_path}")


if __name__ == "__main__":
    main()
