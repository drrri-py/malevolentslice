#!/usr/bin/env python3
"""
MalevolentSlice Studio - Standalone Executable Builder
Automates PyInstaller packaging for Windows (.exe), macOS (.app), and Linux.

Usage:
    python build_standalone.py [--onefile] [--clean]
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional

def create_macos_dmg(dist_dir: Path, app_name: str) -> Optional[Path]:
    """Packages the .app bundle into a drag-and-drop .dmg disk image with /Applications symlink."""
    app_path = dist_dir / f"{app_name}.app"
    if not app_path.exists():
        print(f"[DMG Warning] Cannot create DMG, {app_path} not found.")
        return None

    dmg_path = dist_dir / f"{app_name}.dmg"
    staging_dir = dist_dir / "dmg_staging"

    print("\n[*] Packaging macOS Drag-and-Drop .dmg Installer...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy .app into staging directory
        dest_app = staging_dir / f"{app_name}.app"
        shutil.copytree(app_path, dest_app, symlinks=True)

        # Create symlink to /Applications for drag-and-drop installer UX
        apps_link = staging_dir / "Applications"
        os.symlink("/Applications", apps_link)

        # Remove existing dmg if any
        if dmg_path.exists():
            dmg_path.unlink()

        # Run hdiutil to create compressed DMG
        cmd = [
            "hdiutil", "create",
            "-volname", app_name,
            "-srcfolder", str(staging_dir),
            "-ov",
            "-format", "UDZO",
            str(dmg_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[✔] DMG Installer created: {dmg_path.resolve()}")
            return dmg_path
        else:
            print(f"[!] hdiutil error: {res.stderr}")
            return None
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Build MalevolentSlice Desktop Standalone Executable & Installers")
    parser.add_argument("--onefile", action="store_true", help="Package into a single standalone binary file instead of a folder")
    parser.add_argument("--clean", action="store_true", help="Clean build and dist directories before packaging")
    parser.add_argument("--name", type=str, default="MalevolentSliceStudio", help="Executable output name")
    parser.add_argument("--dmg", action="store_true", default=(sys.platform == "darwin"), help="Build a macOS .dmg installer image (macOS only)")
    parser.add_argument("--no-dmg", dest="dmg", action="store_false", help="Do not create a .dmg installer image")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    entrypoint = project_root / "src" / "malevolentslice" / "gui" / "app.py"
    assets_dir = project_root / "src" / "malevolentslice" / "assets"

    if not entrypoint.exists():
        print(f"[Error] GUI Entrypoint not found at: {entrypoint}")
        sys.exit(1)

    if not assets_dir.exists():
        print(f"[Error] Assets directory not found at: {assets_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f" MalevolentSlice Standalone Executable Builder")
    print(f" Platform : {sys.platform} ({'64-bit' if sys.maxsize > 2**32 else '32-bit'})")
    print(f" Python   : {sys.version.split()[0]} ({sys.executable})")
    print(f" Target   : {args.name}")
    print(f" Mode     : {'Single Executable (--onefile)' if args.onefile else 'Application Bundle / Directory (--onedir)'}")
    print("=" * 70)

    # Clean previous build artifacts if requested
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"
    if args.clean:
        print("[1/4] Cleaning previous build artifacts...")
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        if dist_dir.exists():
            shutil.rmtree(dist_dir, ignore_errors=True)

    # Path separator for PyInstaller --add-data (';' on Windows, ':' on Unix)
    sep = ";" if sys.platform.startswith("win") else ":"

    # Base PyInstaller arguments
    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", args.name,
        "--noconsole",
        "--windowed",
        "--noconfirm",
    ]

    if args.onefile:
        pyinstaller_cmd.append("--onefile")
    else:
        pyinstaller_cmd.append("--onedir")

    # Assets mapping: map src/malevolentslice/assets into malevolentslice/assets in the bundle
    assets_target = f"{assets_dir}{sep}malevolentslice/assets"
    assets_root_target = f"{assets_dir}{sep}assets"
    pyinstaller_cmd.extend(["--add-data", assets_target])
    pyinstaller_cmd.extend(["--add-data", assets_root_target])

    # Collect all customtkinter assets (themes, json configs, fonts)
    pyinstaller_cmd.extend(["--collect-all", "customtkinter"])

    # Hidden imports for core modules, ONNX Runtime, and ASR audio libraries
    hidden_imports = [
        "malevolentslice",
        "malevolentslice.gui",
        "malevolentslice.gui.app",
        "malevolentslice.pipeline",
        "malevolentslice.core.vad_engine",
        "malevolentslice.core.streamer",
        "malevolentslice.core.segmenter",
        "malevolentslice.core.exporter",
        "malevolentslice.core.noise_gate",
        "malevolentslice.core.transcriber",
        "malevolentslice.utils.path_resolver",
        "malevolentslice.utils.memory",
        "malevolentslice.utils.logger",
        "onnxruntime",
        "soundfile",
        "scipy",
        "scipy.signal",
        "scipy.io.wavfile",
        "faster_whisper",
        "ctranslate2",
        "tokenizers",
        "huggingface_hub",
        "psutil",
        "click",
        "rich",
        "darkdetect",
        "PIL",
        "PIL.Image",
    ]

    for hi in hidden_imports:
        pyinstaller_cmd.extend(["--hidden-import", hi])

    # Target entrypoint script
    pyinstaller_cmd.append(str(entrypoint))

    print("\n[2/4] Executing PyInstaller with options:")
    for arg in pyinstaller_cmd:
        print(f"   {arg}")

    print("\n[3/4] Building standalone application (this may take 1-3 minutes)...")
    res = subprocess.run(pyinstaller_cmd, cwd=str(project_root))

    if res.returncode != 0:
        print(f"\n[Build Failed] PyInstaller exited with code {res.returncode}")
        sys.exit(res.returncode)

    print("\n[4/4] Build Complete!")
    print(f"Output directory: {dist_dir.resolve()}")
    if sys.platform == "darwin":
        app_bundle = dist_dir / f"{args.name}.app"
        if app_bundle.exists():
            print(f"[✔] macOS Application Bundle: {app_bundle}")
            if args.dmg and not args.onefile:
                create_macos_dmg(dist_dir, args.name)
    elif sys.platform.startswith("win"):
        exe_file = dist_dir / f"{args.name}.exe" if args.onefile else dist_dir / args.name / f"{args.name}.exe"
        print(f"[✔] Windows Executable: {exe_file}")
    else:
        bin_file = dist_dir / args.name / args.name if not args.onefile else dist_dir / args.name
        print(f"[✔] Linux Binary: {bin_file}")

    print("\nTip: Standalone distribution does not require Python or command line to run.")

if __name__ == "__main__":
    main()
