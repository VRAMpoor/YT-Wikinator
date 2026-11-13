#!/usr/bin/env python3
"""
YT-Wikinator setup check.

Finds out which parts needed to run the app are missing on this PC and can
fetch them. These parts are NOT in the GitHub repository because they are
large or belong to someone else:

  - the llama.cpp engine (PrismML build) and NVIDIA's CUDA runtime DLLs
  - the base model Ternary-Bonsai-2-27B (5.95 GB, Hugging Face)
  - Python packages (yt-dlp, faster-whisper, youtube-transcript-api, ...)
  - the Whisper speech model (fetched by faster-whisper)
  - ytc.exe (the ready-made program from this project's GitHub release, or
    built from the Rust source in this repository)

Everything lives inside the app folder (the folder of this file): the engine
in bin/llama, the model in models/, the database in data/. No other folder on
the PC is searched, so the app folder can be copied to another PC as a whole.

Used by YTC_ControlCenter.pyw, and usable on its own:

    python ytc_setup.py              show what is present / missing
    python ytc_setup.py --install    download and install everything missing
    python ytc_setup.py --self-test  built-in checks (no network, nothing installed)

Every download is verified against a pinned sha256 before it is unpacked or run.
"""

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = ROOT / "downloads"
LLAMA_DIR = ROOT / "bin" / "llama"
MODELS_DIR = ROOT / "models"

NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
USER_AGENT = "YT-Wikinator-Setup"

# A real base model is gigabytes. A small .gguf file in models/ (a test file,
# a vocabulary-only file) must never be mistaken for the base model.
MIN_BASE_MODEL_BYTES = 500 * 1024 * 1024

# --- What gets downloaded, and from where -----------------------------------
# The PTQ1_0 packing of Bonsai only runs on PrismML's llama.cpp build; stock
# llama.cpp rejects it. The tag is pinned to the build this app was tested with.
LLAMA_TAG = "prism-b12128-d7b2896"
LLAMA_BASE_URL = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{LLAMA_TAG}/"
# CUDA 12.4 build: its runtime zip contains cublas64_12.dll, which the Whisper
# transcriber (ctranslate2) needs as well, so one download serves both.
# (file name, size in bytes, sha256). The release is not marked immutable on
# GitHub, so the size alone proves nothing: every download is checked against
# the hash before it is unpacked or run. The hashes are the 'digest' fields of
# the release assets (GitHub API, releases/tags/<LLAMA_TAG>).
LLAMA_ZIP = (f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip", 355_011_546,
             "14969ae6831abfc836295e8f4b8e09e686a063b559a8c974412001d9f5ce4f7f")
CUDART_ZIP = ("cudart-llama-bin-win-cuda-12.4-x64.zip", 391_443_627,
              "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6")
MIN_DRIVER_FOR_CUDA_12_4 = 551.61

MODEL_NAME = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
# Pinned to one commit of the Hugging Face repository instead of 'main': if the
# file is ever replaced upstream, this URL still returns the bytes that match
# MODEL_SIZE and MODEL_SHA256 (the file's Git LFS object id).
MODEL_REVISION = "839bd57023c1558945afea70a74fa83c054f35f9"
MODEL_URL = f"https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/{MODEL_REVISION}/{MODEL_NAME}"
MODEL_SIZE = 5_946_648_928
MODEL_SHA256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"

GITHUB_REPO = "VRAMpoor/YT-Wikinator"
RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"

# The ready-made ytc.exe is an asset of this project's GitHub release.
# YTC_EXE_SIZE (bytes) and YTC_EXE_SHA256 are filled in when the release
# binary has been built and uploaded. While they are empty (0 and ""), no
# ready-made program is published for this version: nothing is downloaded,
# because an unverified exe must never be fetched and run, and ytc.exe is
# built from source instead.
YTC_RELEASE_TAG = "v0.1.0"
# What 'ytc.exe --version' must print for this app: the launcher and ytc.exe
# belong to one release and work as a pair.
YTC_VERSION = YTC_RELEASE_TAG.lstrip("v")
YTC_EXE_NAME = "ytc.exe"
YTC_EXE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{YTC_RELEASE_TAG}/{YTC_EXE_NAME}"
YTC_EXE_SIZE = 12434432
YTC_EXE_SHA256 = "8607f50c0cb5155d31c4153048df9544b8df6f11a60685f006e1834fe2582404"

# (import name, pip requirement). A '>=' in the requirement is a real minimum:
# the check compares the installed version with it (see missing_pip_packages).
PIP_PACKAGES = [
    ("youtube_transcript_api", "youtube-transcript-api>=1.0"),
    ("yt_dlp", "yt-dlp>=2024.10.0"),
    ("faster_whisper", "faster-whisper>=1.0.0"),
    ("requests", "requests>=2.31.0"),
]
# Whisper (ctranslate2) on the GPU needs NVIDIA's cuBLAS 12 and cuDNN 9 DLLs.
# Without them transcription still works, on the CPU, only slower.
GPU_WHISPER_PACKAGES = [
    ("nvidia.cublas", "nvidia-cublas-cu12"),
    ("nvidia.cudnn", "nvidia-cudnn-cu12"),
]
# fetch_browser_transcript.py reads a transcript from the YouTube page in a
# real browser: the first way tried for every video. It drives the installed
# Microsoft Edge, or Playwright's own Chromium where Edge is missing.
BROWSER_PACKAGES = [
    ("playwright", "playwright>=1.40"),
]
# The multilingual model (the same name is in src/extractor.rs).
WHISPER_MODEL = "base"
WHISPER_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-base"

# Written into the engine folder before the first file is put there and removed
# after the finished engine has answered '--version'. While it exists, the
# folder is the leftover of an interrupted install.
LLAMA_INCOMPLETE_MARKER = ".ytc_setup_incomplete"

NVIDIA_DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"
RUSTUP_URL = "https://rustup.rs"

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]


class SetupCancelled(Exception):
    pass


class SetupError(Exception):
    pass


class DownloadHttpError(SetupError):
    """The server answered a download request with an HTTP error status."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# --- Locating things ----------------------------------------------------------
def first_existing_file(candidates) -> Optional[Path]:
    for c in candidates:
        try:
            if c and Path(c).is_file():
                return Path(c).resolve()
        except OSError:
            continue
    return None


def _from_config(text: str) -> Path:
    """A path from config.json: absolute as is, relative means relative to the
    project folder (never to whatever the current directory happens to be)."""
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def find_llama_server(cfg: dict) -> Tuple[Optional[Path], list]:
    """Locate llama-server.exe. Returns (path or None, places searched)."""
    configured = str(cfg.get("llm", {}).get("server_binary") or "").strip()
    candidates = []
    if configured:
        candidates += [_from_config(configured)]
    candidates += [LLAMA_DIR / "llama-server.exe"]
    return first_existing_file(candidates), candidates


def find_ytc_binary(cfg: dict) -> Tuple[Optional[Path], list]:
    """Locate ytc.exe. Returns (path or None, places searched)."""
    configured = str(cfg.get("ytc", {}).get("binary") or "").strip()
    candidates = []
    if configured:
        candidates += [_from_config(configured)]
    candidates += [
        ROOT / "target" / "release" / YTC_EXE_NAME,
        ROOT / YTC_EXE_NAME,
    ]
    return first_existing_file(candidates), candidates


def find_model(configured_text: str) -> Optional[Path]:
    """Locate the base GGUF model. An existing configured file always wins.
    Otherwise look for the same file name, then any large GGUF, in the models
    folder of the app. Small .gguf files are never returned as a fallback."""
    configured_text = (configured_text or "").strip()
    model_dirs = [MODELS_DIR]
    if configured_text:
        hit = first_existing_file([_from_config(configured_text)])
        if hit:
            return hit
        name = Path(configured_text).name
        hit = first_existing_file([d / name for d in model_dirs])
        if hit:
            return hit
    for d in model_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.gguf")):
            try:
                if f.stat().st_size >= MIN_BASE_MODEL_BYTES:
                    return f.resolve()
            except OSError:
                continue
    return None


# --- The one database -----------------------------------------------------------
def app_db_path(cfg: dict) -> Path:
    """The database of the app: ytc.db_path from config.json (relative means
    inside the app folder), else <app folder>/data/ytc.db. Nothing else is
    searched; if the file does not exist, ytc.exe creates it there."""
    text = str(cfg.get("ytc", {}).get("db_path") or "").strip()
    return _from_config(text) if text else ROOT / "data" / "ytc.db"


_python_exe_cache: Optional[Path] = None


def python_exe() -> Path:
    """python.exe of the interpreter running this (pythonw.exe has no console)."""
    global _python_exe_cache
    if _python_exe_cache is None:
        exe = Path(sys.executable)
        sibling = exe.with_name("python.exe")
        same = sibling.name.lower() == exe.name.lower()
        _python_exe_cache = sibling if same or _is_working_python(sibling) else exe
    return _python_exe_cache


def _run(cmd, timeout=30) -> Tuple[int, str]:
    try:
        res = subprocess.run(
            cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout,
            encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
        )
        return res.returncode, (res.stdout or "") + (res.stderr or "")
    except Exception as e:  # missing program, timeout, ...
        return -1, str(e)


def _is_working_python(exe: Path) -> bool:
    """True when this file really starts a Python 3 interpreter. Decided by
    running it, never by its path: a Microsoft Store Python lives under
    WindowsApps and works, while the App Execution Alias stub in the same
    folder (a zero-byte entry that only opens the Store) does not."""
    code, out = _run([str(exe), "-c", "import sys; print(sys.version_info[0])"], timeout=30)
    return code == 0 and out.strip() == "3"


def human_size(n: int) -> str:
    """Decimal units (1 GB = 10^9 bytes), the same as README, GitHub and
    Hugging Face use: the model is shown as 5.95 GB everywhere."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    return f"{n / 1e6:.0f} MB"


# --- The checks ---------------------------------------------------------------
@dataclass
class Item:
    key: str
    label: str
    ok: bool
    detail: str
    required: bool = True          # the app cannot do its job without it
    can_install: bool = False      # install_item() knows how to get it
    download_bytes: int = 0
    help_url: str = ""


def check_gpu() -> Item:
    code, out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader,nounits"])
    line = out.strip().splitlines()[0] if code == 0 and out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return Item("gpu", "NVIDIA graphics card", False,
                    "No NVIDIA card or driver found (nvidia-smi did not answer). "
                    "The LLM engine needs an NVIDIA card with 8 GB VRAM or more and a current driver.",
                    required=True, help_url=NVIDIA_DRIVER_URL)
    name, mem, driver = parts[0], parts[1], parts[2]
    notes = []
    ok = True
    try:
        if int(float(mem)) < 7500:
            ok = False
            notes.append(f"only {int(float(mem))} MiB VRAM; 8 GB is needed for the 27B model")
    except ValueError:
        pass
    try:
        if float(".".join(driver.split(".")[:2])) < MIN_DRIVER_FOR_CUDA_12_4:
            ok = False
            notes.append(f"driver {driver} is below the minimum for CUDA 12.4; "
                         f"install {MIN_DRIVER_FOR_CUDA_12_4} or newer")
    except ValueError:
        pass
    detail = f"{name}, {mem} MiB, driver {driver}" + ("; " + "; ".join(notes) if notes else "")
    return Item("gpu", "NVIDIA graphics card", ok, detail, required=True,
                help_url="" if ok else NVIDIA_DRIVER_URL)


def _version_tuple(text: str) -> Tuple[int, ...]:
    """Leading number of each dot-separated part: '1.2.5' -> (1, 2, 5),
    '2024.10.07' -> (2024, 10, 7), '1.0rc1' -> (1, 0)."""
    parts = []
    for piece in (text or "").strip().split("."):
        digits = re.match(r"\d+", piece)
        if not digits:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


def version_at_least(installed: str, minimum: str) -> bool:
    """True when the installed version is the minimum or newer. A version that
    cannot be read counts as below the minimum, so pip installs the package again."""
    have, need = _version_tuple(installed), _version_tuple(minimum)
    if not have:
        return False
    width = max(len(have), len(need))
    return have + (0,) * (width - len(have)) >= need + (0,) * (width - len(need))


def _split_requirement(requirement: str) -> Tuple[str, str]:
    """'yt-dlp>=2024.10.0' -> ('yt-dlp', '2024.10.0'); no '>=' -> (name, '')."""
    name, _, minimum = requirement.partition(">=")
    return name.strip(), minimum.strip()


def _unmet_requirements(packages, installed: dict) -> List[str]:
    """Requirements that are not met. `installed` maps the import name to the
    installed version, to '' when the module is missing, or to '?' when the
    module is there but its version is unknown."""
    unmet = []
    for module, requirement in packages:
        version = installed.get(module, "")
        _, minimum = _split_requirement(requirement)
        if not version or (minimum and not version_at_least(version, minimum)):
            unmet.append(requirement)
    return unmet


def _run_python_probe(code_text: str, args: List[str], timeout: int = 60) -> Optional[str]:
    """Run a snippet in python_exe(); its stdout, or None when it failed."""
    code, out = _run([str(python_exe()), "-c", code_text] + list(args), timeout=timeout)
    return out if code == 0 else None


def missing_pip_packages(packages=None) -> List[str]:
    """pip requirements that python_exe() does not meet: the module cannot be
    imported, or its version is below the minimum in the requirement. Presence
    alone is not enough: fetch_builtin_transcript.py needs the 1.x API of
    youtube-transcript-api, which an import test cannot tell."""
    packages = PIP_PACKAGES if packages is None else packages
    probe = (
        "import importlib.util, importlib.metadata, sys\n"
        "for arg in sys.argv[1:]:\n"
        "    module, dist = arg.split('|')\n"
        "    try:\n"
        "        found = importlib.util.find_spec(module) is not None\n"
        "    except (ImportError, ValueError):\n"
        "        found = False\n"
        "    version = ''\n"
        "    if found:\n"
        "        try:\n"
        "            version = importlib.metadata.version(dist)\n"
        "        except Exception:\n"
        "            version = '?'\n"
        "    print(module + '=' + version)\n"
    )
    args = [f"{module}|{_split_requirement(req)[0]}" for module, req in packages]
    out = _run_python_probe(probe, args)
    if out is None:
        return [req for _, req in packages]
    installed = {}
    for line in out.splitlines():
        module, sep, version = line.strip().partition("=")
        if sep:
            installed[module] = version
    return _unmet_requirements(packages, installed)


def whisper_dll_folders() -> List[Path]:
    """The folders transcribe_cuda.py adds to the DLL search path, asked from
    the helper itself (its module-level `candidates` list) so that this check
    cannot drift away from what the helper really does. If the helper cannot
    be asked, the list it is known to use is rebuilt from that interpreter's
    sys.prefix."""
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(root))\n"
        "folders = None\n"
        "try:\n"
        "    import transcribe_cuda\n"
        "    folders = getattr(transcribe_cuda, 'candidates', None)\n"
        "except Exception:\n"
        "    folders = None\n"
        "if not folders:\n"
        "    nvidia = Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'\n"
        "    folders = [root, root / 'bin' / 'llama',\n"
        "               nvidia / 'cublas' / 'bin', nvidia / 'cudnn' / 'bin']\n"
        "for f in folders:\n"
        "    print(f)\n"
    )
    out = _run_python_probe(probe, [str(ROOT)])
    if out is None:
        return []
    return [Path(line.strip()) for line in out.splitlines() if line.strip()]


def whisper_gpu_dlls_missing(folders) -> List[str]:
    """Which of the two NVIDIA libraries ctranslate2 needs for CUDA are in none
    of the given folders: cuBLAS 12 (cublas64_12.dll) and cuDNN (cudnn*.dll)."""
    have_cublas = have_cudnn = False
    for folder in folders:
        try:
            if not Path(folder).is_dir():
                continue
            have_cublas = have_cublas or (Path(folder) / "cublas64_12.dll").is_file()
            have_cudnn = have_cudnn or any(Path(folder).glob("cudnn*.dll"))
        except OSError:
            continue
    missing = []
    if not have_cublas:
        missing.append("cublas64_12.dll")
    if not have_cudnn:
        missing.append("cuDNN (cudnn*.dll)")
    return missing


def check_gpu_whisper() -> Item:
    label = "GPU speed-up for Whisper (cuBLAS + cuDNN)"
    missing = missing_pip_packages(GPU_WHISPER_PACKAGES)
    if missing:
        return Item("pip_gpu", label, False,
                    "missing: " + ", ".join(missing) + ". Without it Whisper still works, on the CPU, only slower.",
                    required=False, can_install=True, download_bytes=1200 * 1000 * 1000)
    # The packages being installed is not enough: pip may have put them into
    # the per-user site, where transcribe_cuda.py does not look.
    dlls_missing = whisper_gpu_dlls_missing(whisper_dll_folders())
    if dlls_missing:
        return Item("pip_gpu", label, False,
                    "the pip packages are installed, but transcribe_cuda.py does not find "
                    + " and ".join(dlls_missing) + " in the folders it searches (this Python's "
                    "Lib\\site-packages\\nvidia\\...\\bin, the project folder, bin\\llama). Whisper runs on "
                    "the CPU. This happens when pip installed into the per-user folder; install the two "
                    "packages into the Python that runs the app, or copy the DLLs into bin\\llama.",
                    required=False)
    return Item("pip_gpu", label, True,
                "installed, and transcribe_cuda.py finds cublas64_12.dll and cuDNN; "
                "videos without captions can be transcribed on the graphics card", required=False)


def check_python_packages() -> Item:
    missing = missing_pip_packages()
    if not missing:
        return Item("pip", "Python packages", True, "yt-dlp, faster-whisper, youtube-transcript-api, requests")
    return Item("pip", "Python packages", False, "missing or below the minimum version: " + ", ".join(missing),
                can_install=True, download_bytes=250 * 1000 * 1000)


def check_llama(cfg: dict) -> Item:
    path, _ = find_llama_server(cfg)
    size = LLAMA_ZIP[1] + CUDART_ZIP[1]
    if path is None:
        return Item("llama", "LLM engine (llama.cpp, PrismML build) + CUDA runtime", False,
                    f"llama-server.exe not found. Will be placed in {LLAMA_DIR}",
                    can_install=True, download_bytes=size)
    if (path.parent / LLAMA_INCOMPLETE_MARKER).exists():
        return Item("llama", "LLM engine (llama.cpp, PrismML build) + CUDA runtime", False,
                    f"{path.parent} is incomplete: an install into it was interrupted. "
                    "The install will be repeated.", can_install=True, download_bytes=size)
    if not (path.parent / "ggml-cuda.dll").exists():
        return Item("llama", "LLM engine (llama.cpp, PrismML build) + CUDA runtime", False,
                    f"{path} has no CUDA backend (ggml-cuda.dll); it would run on the CPU only",
                    can_install=True, download_bytes=size)
    # Files copied into bin/llama by hand carry no marker: llama-server.exe
    # cannot start without llama.dll, so a missing llama.dll means half a copy.
    if path.parent == LLAMA_DIR.resolve() and not (path.parent / "llama.dll").exists():
        return Item("llama", "LLM engine (llama.cpp, PrismML build) + CUDA runtime", False,
                    f"{path.parent} is incomplete (llama.dll is missing). The install will be repeated.",
                    can_install=True, download_bytes=size)
    has_runtime = any(path.parent.glob("cublas64_*.dll"))
    detail = str(path) if has_runtime else f"{path} (CUDA runtime DLLs come from elsewhere on this PC)"
    return Item("llama", "LLM engine (llama.cpp, PrismML build) + CUDA runtime", True, detail)


def check_model(cfg: dict) -> Item:
    path = find_model(str(cfg.get("llm", {}).get("model_path") or ""))
    if path is not None:
        return Item("model", "Base model (Ternary-Bonsai-2-27B)", True, f"{path} ({human_size(path.stat().st_size)})")
    return Item("model", "Base model (Ternary-Bonsai-2-27B)", False,
                f"{MODEL_NAME} not found. Will be saved to {MODELS_DIR}",
                can_install=True, download_bytes=MODEL_SIZE)


def check_database(cfg: dict) -> Item:
    """Where the one database is."""
    path = app_db_path(cfg)
    try:
        exists = path.is_file()
    except OSError:
        exists = False
    detail = str(path) if exists else f"{path} (does not exist yet; it is created on first use)"
    return Item("db", "Library database", True, detail)


def ytc_version(path: Path) -> Tuple[Optional[str], str]:
    """The version 'ytc.exe --version' prints (run hidden; the answer is
    'ytc <version>'), or (None, reason) when the program could not be asked.
    A call that failed says nothing about the version."""
    code, out = _run([str(path), "--version"], timeout=20)
    if code != 0:
        return None, out.strip()[:300] or f"exit code {code}"
    words = out.split()
    if not words:
        return None, "no answer to '--version'"
    return words[-1], ""


def ytc_exe_published() -> bool:
    """True when size and checksum of the release asset are filled in. Empty
    values mean: no ready-made ytc.exe is published for this version."""
    return YTC_EXE_SIZE > 0 and bool(re.fullmatch(r"[0-9a-fA-F]{64}", YTC_EXE_SHA256.strip()))


def _is_build_output(path: Optional[Path]) -> bool:
    """True for a ytc.exe inside this folder's target/ directory (built here)."""
    if path is None:
        return False
    try:
        return (ROOT / "target").resolve() in path.resolve().parents
    except OSError:
        return False


def ytc_install_order(published: bool, has_cargo: bool, prefer_build: bool = False) -> List[str]:
    """How a missing or not matching ytc.exe is obtained, in the order tried:
    'download' (the verified release asset; preferred) and/or 'build' (cargo;
    the fallback). An empty list means neither is possible on this PC.
    prefer_build: the ytc.exe in use was built here, so it is rebuilt first."""
    order = (["download"] if published else []) + (["build"] if has_cargo else [])
    if prefer_build and "build" in order:
        order.sort(key=lambda step: step != "build")
    return order


def _ytc_how_text(order: List[str]) -> str:
    """What the install button will do, or what the user can do by hand."""
    download = (f"the ready-made {YTC_EXE_NAME} of release {YTC_RELEASE_TAG} ({human_size(YTC_EXE_SIZE)}) is "
                f"downloaded from github.com/{GITHUB_REPO} and checked against its sha256")
    build = "it is built here from the source with cargo (takes a few minutes)"
    if order == ["download", "build"]:
        return f"{download[0].upper()}{download[1:]}; if the download is not possible, {build}."
    if order == ["build", "download"]:
        return f"{build[0].upper()}{build[1:]}; if the build fails, {download}."
    if order == ["download"]:
        return f"{download[0].upper()}{download[1:]}."
    if order == ["build"]:
        why = ("" if ytc_exe_published()
               else f"No ready-made {YTC_EXE_NAME} is published for this version ({YTC_RELEASE_TAG}). ")
        return f"{why}Rust is installed, so {build}."
    return (f"No ready-made {YTC_EXE_NAME} is published for this version ({YTC_RELEASE_TAG}), and Rust is not "
            f"installed. Two ways: (1) install Rust from {RUSTUP_URL} (with the Visual Studio C++ build tools) "
            f"and run this check again, it then builds {YTC_EXE_NAME}; or (2) if {RELEASES_URL} offers a "
            f"{YTC_EXE_NAME} for {YTC_RELEASE_TAG}, save it as {ROOT / YTC_EXE_NAME}.")


def ytc_mismatch_message(path: Path, found: str) -> str:
    """One text for the setup window and the Control Center."""
    return (f"{path} does not match this app (found {found}, need {YTC_VERSION}). "
            f"To replace it: stop the web server, the auto-crawl and any running crawl (a running ytc.exe "
            f"cannot be replaced), then press SETUP CHECK and use the install button there, or run "
            f"'python ytc_setup.py --install' in {ROOT}. By hand: 'cargo build --release' in {ROOT}.")


def check_ytc(cfg: dict) -> Item:
    """Is the ytc.exe of this app's version there? If not, the detail says
    which way it will be obtained: download of the release asset, build with
    cargo, or by hand."""
    label = "YTC engine (ytc.exe)"
    path, _ = find_ytc_binary(cfg)
    has_cargo = bool(shutil.which("cargo"))
    order = ytc_install_order(ytc_exe_published(), has_cargo, prefer_build=_is_build_output(path))
    how = _ytc_how_text(order)
    fixable = dict(can_install=bool(order), help_url="" if order else RUSTUP_URL,
                   download_bytes=YTC_EXE_SIZE if "download" in order else 0)
    if path is not None:
        found, problem = ytc_version(path)
        if found is None:
            return Item("ytc", label, False, f"{path} does not start ({problem}). {how}", **fixable)
        if found != YTC_VERSION:
            return Item("ytc", label, False, f"{ytc_mismatch_message(path, found)} {how}", **fixable)
        return Item("ytc", label, True, str(path))
    return Item("ytc", label, False, f"not found. {how}", **fixable)


def check_whisper() -> Item:
    if WHISPER_CACHE.is_dir():
        return Item("whisper", f"Whisper speech model ({WHISPER_MODEL})", True, str(WHISPER_CACHE), required=False)
    return Item("whisper", f"Whisper speech model ({WHISPER_MODEL})", False,
                "not downloaded yet. Only needed for videos without captions; faster-whisper also fetches it "
                "by itself on first use.", required=False, can_install=True, download_bytes=150 * 1000 * 1000)


def browser_for_transcripts() -> Optional[Path]:
    """The browser fetch_browser_transcript.py can drive: Edge, else a Chromium
    that Playwright downloaded. None when there is neither."""
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            edge = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if edge.is_file():
                return edge
    store = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    if store.is_dir():
        for folder in sorted(store.glob("chromium-*"), reverse=True):
            return folder
    return None


def check_browser_transcripts() -> Item:
    label = "Transcripts through a browser (Playwright)"
    why = ("The first way to get a transcript. Without it the direct request is used, which YouTube "
           "refuses after a number of videos; such videos are then transcribed locally, which takes longer.")
    missing = missing_pip_packages(BROWSER_PACKAGES)
    if missing:
        return Item("pip_browser", label, False, "missing: " + ", ".join(missing) + ". " + why,
                    required=False, can_install=True, download_bytes=45 * 1000 * 1000)
    browser = browser_for_transcripts()
    if browser is None:
        return Item("pip_browser", label, False,
                    "playwright is installed, but neither Microsoft Edge nor a Playwright Chromium was found. "
                    "Install one with:  python -m playwright install chromium", required=False)
    return Item("pip_browser", label, True, f"playwright, browser: {browser}", required=False)


def check_all(cfg: dict) -> List[Item]:
    return [
        check_gpu(),
        check_python_packages(),
        check_llama(cfg),
        check_model(cfg),
        check_ytc(cfg),
        check_database(cfg),
        check_whisper(),
        check_gpu_whisper(),
        check_browser_transcripts(),
    ]


def needs_attention(items: List[Item]) -> bool:
    return any(not i.ok and i.required for i in items)


def total_download_bytes(items: List[Item]) -> int:
    return sum(i.download_bytes for i in items if not i.ok and i.can_install)


# --- Downloading --------------------------------------------------------------
# Download servers sometimes hand out a connection that crawls (measured on
# GitHub's release CDN: 0.1 MB/s on one connection, 6 MB/s on the next, same
# file, same minute). A connection slower than this for a whole window is
# dropped and the download continues on a new one. After a few tries the slow
# speed is accepted, because then the line itself is slow.
SLOW_BYTES_PER_S = 300_000
SLOW_WINDOW_S = 12.0
MAX_SLOW_RECONNECTS = 6
SLOW_CONNECTION = "slow connection"


def _fetch_once(url: str, part: Path, have: int, expected_size: int, name: str, log: LogFn,
                progress: Optional[ProgressFn], cancel: Optional[threading.Event],
                drop_if_slow: bool = False) -> Tuple[bool, str]:
    """One connection: append to the partial file from byte `have`.
    Returns (True, "") at the end of the data, (False, reason) if the
    connection broke while reading or (with drop_if_slow) was too slow.
    Errors that a retry cannot fix raise."""
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
        log(f"Continuing {name} at {human_size(have)}")
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)
    except urllib.error.HTTPError as e:
        raise DownloadHttpError(e.code, f"Download of {name} failed: HTTP {e.code} from {url}") from e
    except (urllib.error.URLError, OSError) as e:
        if have:
            return False, str(e)
        raise SetupError(f"Download of {name} failed: {e}. Check the internet connection.") from e
    with resp:
        if have and resp.status != 206:   # server ignored the Range header: start over
            have = 0
        total = expected_size or (have + int(resp.headers.get("Content-Length") or 0))
        with open(part, "ab" if have else "wb") as f:
            done = have
            window_start = time.monotonic()
            window_bytes = 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise SetupCancelled()
                try:
                    # Small pieces, so cancel and the speed check stay responsive.
                    chunk = resp.read(256 * 1024)
                except (OSError, http.client.HTTPException) as e:
                    return False, str(e)
                if not chunk:
                    return True, ""
                f.write(chunk)
                done += len(chunk)
                window_bytes += len(chunk)
                if progress:
                    progress(done, total)
                elapsed = time.monotonic() - window_start
                if elapsed >= SLOW_WINDOW_S:
                    if drop_if_slow and window_bytes / elapsed < SLOW_BYTES_PER_S:
                        return False, SLOW_CONNECTION
                    window_start = time.monotonic()
                    window_bytes = 0


def sha256_of(path: Path, cancel: Optional[threading.Event] = None) -> str:
    """sha256 of a file as lowercase hex, read in 1 MB pieces."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            _check_cancel(cancel)
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, expected_size: int = 0, log: LogFn = print,
                  progress: Optional[ProgressFn] = None,
                  cancel: Optional[threading.Event] = None, sha256: str = "") -> Path:
    """Download url to dest. Interrupted downloads continue where they stopped
    (the partial file is kept as <name>.part). With `sha256` the finished file
    must have that hash before it gets its final name; a file that does not
    match is deleted, so nothing unverified is ever unpacked or run."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha256 = sha256.strip().lower()
    part = dest.with_name(dest.name + ".part")

    # A verified file that is already there (e.g. unpacking was interrupted
    # after the download) is used instead of being downloaded a second time.
    if sha256 and expected_size and dest.is_file() and dest.stat().st_size == expected_size:
        if sha256_of(dest, cancel) == sha256:
            log(f"{dest.name} is already downloaded and verified.")
            return dest

    have = part.stat().st_size if part.exists() else 0
    if expected_size and have > expected_size:
        part.unlink()
        have = 0

    free = shutil.disk_usage(dest.parent).free
    need = (expected_size - have) if expected_size else 0
    if need and free < need + 200 * 1000 * 1000:
        raise SetupError(f"Not enough disk space in {dest.parent}: {human_size(need)} needed, {human_size(free)} free.")

    # A dropped connection is retried from where it stopped. The retry counter
    # starts again whenever a try made progress, so a long download over a
    # shaky line still finishes.
    stalled_tries = 0
    slow_reconnects = 0
    restarted = False
    while not (expected_size and have == expected_size):
        before = have
        try:
            finished, problem = _fetch_once(url, part, have, expected_size, dest.name, log, progress, cancel,
                                            drop_if_slow=slow_reconnects < MAX_SLOW_RECONNECTS)
        except DownloadHttpError as e:
            # HTTP 416 on a continued download: the partial file is already as
            # long as (or longer than) what the server offers. Without this the
            # same request would fail on every later run.
            if e.code != 416 or not have:
                raise
            if not expected_size and sha256:
                # Size not known beforehand: the partial file may be the whole
                # file. The hash check below decides (and deletes it if not).
                break
            if restarted:
                part.unlink(missing_ok=True)
                raise SetupError(_shorter_upstream_message(dest.name, url, have, expected_size)) from e
            # Cannot be verified as it is: start from zero, once.
            log(f"The server has nothing after {human_size(have)} for {dest.name}; "
                f"the partial file is discarded and the download starts again.")
            part.unlink(missing_ok=True)
            have = 0
            restarted = True
            continue
        have = part.stat().st_size if part.exists() else 0
        if finished and expected_size and have < expected_size:
            # The server ended the data early without an error. Ask for the
            # rest like after a dropped connection (bounded by stalled_tries
            # and by the single restart above) instead of giving up at once.
            finished, problem = False, "the server ended the data early"
        if finished:
            break
        if problem == SLOW_CONNECTION:
            slow_reconnects += 1
            log(f"This connection is slow; trying a new one for {dest.name} "
                f"({slow_reconnects} of {MAX_SLOW_RECONNECTS}) ...")
            continue
        stalled_tries = 0 if have > before else stalled_tries + 1
        if stalled_tries >= 5:
            raise SetupError(f"Connection lost while downloading {dest.name}: {problem}. "
                             f"Run the setup again; it continues where it stopped.")
        log(f"Connection dropped at {human_size(have)}; continuing {dest.name} ...")
        if cancel is not None and cancel.wait(2):
            raise SetupCancelled()

    got = part.stat().st_size if part.exists() else 0
    if expected_size and got != expected_size:
        raise SetupError(f"{dest.name} is incomplete ({got} of {expected_size} bytes). "
                         f"Run the setup again; it continues where it stopped.")
    if sha256:
        # The byte count alone would also accept a file that was replaced
        # upstream, damaged on disk, or stitched together by a resume across
        # two different upstream versions.
        log(f"Verifying {dest.name} (sha256) ...")
        actual = sha256_of(part, cancel)
        if actual != sha256:
            part.unlink()
            raise SetupError(f"{dest.name} failed the integrity check: sha256 is {actual}, expected {sha256}. "
                             f"The file was deleted. Run the setup again; if this repeats, the file at {url} "
                             f"is not the one this app was tested with, and it must not be used.")
    os.replace(part, dest)
    return dest


def _shorter_upstream_message(name: str, url: str, got: int, expected_size: int) -> str:
    """Why a download cannot be completed by running the setup once more."""
    wanted = f" instead of the expected {expected_size} bytes" if expected_size else ""
    return (f"Download of {name} failed: the server offers only {got} bytes{wanted}, also after "
            f"starting again from zero. The partial file was deleted. The file at {url} is not the "
            f"one this app was tested with, and it must not be used.")


def _check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise SetupCancelled()


def _zip_unpacked_bytes(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        return sum(member.file_size for member in zf.infolist())


def _extract_zip(zip_path: Path, target: Path, cancel: Optional[threading.Event] = None) -> None:
    """Extract without trusting member paths (no writing outside target).
    Members are written in 1 MB pieces so that a cancel takes effect inside a
    large file too (ggml-cuda.dll alone is about 780 MB)."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for member in members:
            out = (root / member.filename).resolve()
            if root not in out.parents and out != root:
                raise SetupError(f"{zip_path.name} contains an unsafe path: {member.filename}")
        for member in members:
            out = (root / member.filename).resolve()
            if member.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(out, "wb") as sink:
                while True:
                    _check_cancel(cancel)
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    sink.write(chunk)


def _place_engine_files(files, target_dir: Path, cancel: Optional[threading.Event] = None) -> Path:
    """Move unpacked files into the engine folder. Returns the marker file,
    which is written before the first move: from then until the caller deletes
    it, the folder counts as incomplete, whatever stops the install (cancel,
    full disk, closed window, power loss)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / LLAMA_INCOMPLETE_MARKER
    marker.write_text("An install of the LLM engine into this folder did not finish.\n"
                      "Run the setup check again; it repeats the install.\n", encoding="utf-8")
    for f in files:
        _check_cancel(cancel)
        os.replace(f, target_dir / Path(f).name)
    return marker


def install_llama(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zips = []
    for name, size, sha256 in (LLAMA_ZIP, CUDART_ZIP):
        log(f"Downloading {name} ({human_size(size)}) from github.com/PrismML-Eng/llama.cpp ...")
        zips.append(download_file(LLAMA_BASE_URL + name, DOWNLOAD_DIR / name, size, log, progress, cancel, sha256))

    # Unpack next to bin/llama (same drive), then move the files in: a move on
    # one drive needs no second copy of the 1.4 GB.
    LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    work = LLAMA_DIR.parent / (LLAMA_DIR.name + ".unpack")
    shutil.rmtree(work, ignore_errors=True)
    need = sum(_zip_unpacked_bytes(z) for z in zips)
    free = shutil.disk_usage(LLAMA_DIR.parent).free
    if free < need + 200 * 1000 * 1000:
        raise SetupError(f"Not enough disk space in {LLAMA_DIR.parent} to unpack the LLM engine: "
                         f"{human_size(need)} needed, {human_size(free)} free. "
                         f"The downloaded zip files are kept for the next try.")
    try:
        log("Unpacking the LLM engine ...")
        _extract_zip(zips[0], work / "llama", cancel)
        _extract_zip(zips[1], work / "cudart", cancel)
        servers = list((work / "llama").rglob("llama-server.exe"))
        if not servers:
            raise SetupError(f"{zips[0].name} does not contain llama-server.exe")
        files = [f for f in servers[0].parent.iterdir() if f.is_file()]
        files += list((work / "cudart").rglob("*.dll"))
        marker = _place_engine_files(files, LLAMA_DIR, cancel)

        code, out = _run([str(LLAMA_DIR / "llama-server.exe"), "--version"], timeout=60)
        if code != 0:
            raise SetupError(f"The downloaded llama-server.exe does not start: {out.strip()[:300]}")
        marker.unlink()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    for z in zips:
        z.unlink(missing_ok=True)
    log(f"LLM engine installed in {LLAMA_DIR}")


def install_model(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    log(f"Downloading {MODEL_NAME} ({human_size(MODEL_SIZE)}) from huggingface.co/prism-ml ...")
    download_file(MODEL_URL, MODELS_DIR / MODEL_NAME, MODEL_SIZE, log, progress, cancel, MODEL_SHA256)
    log(f"Base model saved to {MODELS_DIR / MODEL_NAME}")


def _stream(cmd, log: LogFn, cancel: Optional[threading.Event], cwd: Optional[Path] = None) -> int:
    """Run a command hidden and pass its output lines to log()."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=NO_WINDOW, env=env,
        )
    except OSError as e:
        raise SetupError(f"Could not start {cmd[0]}: {e}") from e
    for line in proc.stdout:
        if cancel is not None and cancel.is_set():
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
            raise SetupCancelled()
        if line.strip():
            log(line.rstrip())
    return proc.wait()


def install_pip(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    missing = missing_pip_packages()
    if not missing:
        return
    log("Installing Python packages: " + ", ".join(missing))
    # --upgrade: a package that is installed but below its minimum version is in the list too.
    code = _stream([str(python_exe()), "-m", "pip", "install", "--upgrade", "--disable-pip-version-check"] + missing,
                   log, cancel)
    if code != 0:
        raise SetupError("pip could not install the Python packages (see the lines above).")


def install_pip_gpu(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    missing = missing_pip_packages(GPU_WHISPER_PACKAGES)
    if not missing:
        return
    log("Installing NVIDIA libraries for GPU transcription: " + ", ".join(missing))
    code = _stream([str(python_exe()), "-m", "pip", "install", "--disable-pip-version-check"] + missing, log, cancel)
    if code != 0:
        raise SetupError("pip could not install the NVIDIA libraries. Whisper will use the CPU.")


def install_pip_browser(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    missing = missing_pip_packages(BROWSER_PACKAGES)
    if not missing:
        return
    log("Installing the browser automation package: " + ", ".join(missing))
    code = _stream([str(python_exe()), "-m", "pip", "install", "--disable-pip-version-check"] + missing, log, cancel)
    if code != 0:
        raise SetupError("pip could not install playwright. Transcripts are then asked for directly.")


def install_whisper(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    log(f"Downloading the Whisper speech model '{WHISPER_MODEL}' ...")
    snippet = f"from faster_whisper import WhisperModel; WhisperModel('{WHISPER_MODEL}', device='cpu', compute_type='int8')"
    code = _stream([str(python_exe()), "-c", snippet], log, cancel)
    if code != 0:
        raise SetupError("The Whisper model could not be downloaded. faster-whisper will try again on first use.")


def _download_ytc(dest: Path, log: LogFn, progress: Optional[ProgressFn],
                  cancel: Optional[threading.Event]) -> None:
    """Fetch the release asset to dest. download_file gives the file its final
    name only after size and sha256 match."""
    log(f"Downloading {YTC_EXE_NAME} {YTC_RELEASE_TAG} ({human_size(YTC_EXE_SIZE)}) from "
        f"github.com/{GITHUB_REPO} ...")
    try:
        download_file(YTC_EXE_URL, dest, YTC_EXE_SIZE, log, progress, cancel, YTC_EXE_SHA256)
    except DownloadHttpError as e:
        if e.code == 404:
            raise SetupError(
                f"The release is not reachable (HTTP 404 from {YTC_EXE_URL}). For example the repository "
                f"is private, or release {YTC_RELEASE_TAG} has not been published yet.") from e
        raise
    except PermissionError as e:
        raise SetupError(f"{dest} could not be replaced: it is in use. Stop the web server, the auto-crawl "
                         f"and any running crawl, then try again.") from e
    log(f"{YTC_EXE_NAME} saved to {dest} (sha256 verified).")


def _build_ytc(log: LogFn, cancel: Optional[threading.Event]) -> None:
    """Build ytc.exe from the source in this folder."""
    log("Building ytc.exe from source (cargo build --release); this takes a few minutes ...")
    code = _stream(["cargo", "build", "--release"], log, cancel, cwd=ROOT)
    built = ROOT / "target" / "release" / YTC_EXE_NAME
    if code != 0 or not built.exists():
        raise SetupError("The build failed (see the lines above). If ytc.exe is running (web server or a "
                         "crawl), stop it first: a running program cannot be replaced.")
    log("ytc.exe built.")


def install_ytc(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    """Get the ytc.exe of this app's version: the verified release asset when
    one is published (preferred), else or after a failed download a build with
    cargo. A missing ytc.exe is downloaded to <app folder>/ytc.exe; one that
    does not match is replaced where it is, so that the app really uses the
    installed file."""
    cfg = _load_cfg()
    used, _ = find_ytc_binary(cfg)
    order = ytc_install_order(ytc_exe_published(), bool(shutil.which("cargo")),
                              prefer_build=_is_build_output(used))
    if not order:
        raise SetupError(_ytc_how_text(order))
    problems = []
    for step in order:
        try:
            if step == "download":
                _download_ytc(used if used is not None else ROOT / YTC_EXE_NAME, log, progress, cancel)
            else:
                _build_ytc(log, cancel)
            break
        except SetupError as e:
            problems.append(str(e))
            if step != order[-1]:
                log(f"{e}")
                log("Trying the other way to get ytc.exe ...")
    else:
        raise SetupError(" | ".join(problems))

    # What the app uses may be another ytc.exe than the one just installed
    # (named in config.json, or found before it): say so instead of reporting success.
    now_used, _ = find_ytc_binary(cfg)
    if now_used is None:
        raise SetupError("ytc.exe is still not found after the install (see the lines above).")
    found, problem = ytc_version(now_used)
    if found is None:
        raise SetupError(f"The app uses {now_used}, which does not start ({problem}).")
    if found != YTC_VERSION:
        raise SetupError(f"ytc.exe was installed, but the app uses {now_used}, which does not match this app "
                         f"(found {found}, need {YTC_VERSION}). Remove that file, or let ytc.binary in "
                         f"config.json point to the installed one (target/release/ytc.exe or ytc.exe).")


INSTALLERS = {
    "pip": install_pip,
    "llama": install_llama,
    "model": install_model,
    "ytc": install_ytc,
    "whisper": install_whisper,
    "pip_gpu": install_pip_gpu,
    "pip_browser": install_pip_browser,
}


def install_missing(items: List[Item], log: LogFn = print, progress: Optional[ProgressFn] = None,
                    cancel: Optional[threading.Event] = None) -> List[str]:
    """Install every missing item that can be installed. One failure does not
    stop the others. Returns the list of error messages (empty = all fine)."""
    errors = []
    for item in items:
        if item.ok or not item.can_install or item.key not in INSTALLERS:
            continue
        try:
            INSTALLERS[item.key](log, progress, cancel)
        except SetupCancelled:
            raise
        except SetupError as e:
            errors.append(f"{item.label}: {e}")
            log(f"FAILED - {item.label}: {e}")
        except Exception as e:  # keep going with the next item
            errors.append(f"{item.label}: {type(e).__name__}: {e}")
            log(f"FAILED - {item.label}: {type(e).__name__}: {e}")
    return errors


# --- Command line ---------------------------------------------------------------
def _load_cfg() -> dict:
    try:
        with open(ROOT / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _print_report(items: List[Item]) -> None:
    for i in items:
        mark = "OK     " if i.ok else ("MISSING" if i.required else "OPTIONAL")
        print(f"[{mark}] {i.label}")
        print(f"          {i.detail}")
        if not i.ok and i.help_url:
            print(f"          -> {i.help_url}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cfg = _load_cfg()
    items = check_all(cfg)
    _print_report(items)
    if "--install" not in sys.argv[1:]:
        if needs_attention(items):
            print(f"\nRun 'python ytc_setup.py --install' to fetch what is missing "
                  f"(about {human_size(total_download_bytes(items))}).")
        return 1 if needs_attention(items) else 0

    last = [0]

    def progress(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct != last[0]:
            last[0] = pct
            print(f"\r   {pct:3d}%  {human_size(done)} of {human_size(total)}", end="", flush=True)
            if pct == 100:
                print()

    errors = install_missing(items, print, progress)
    items = check_all(cfg)
    print()
    _print_report(items)
    return 1 if errors or needs_attention(items) else 0


def self_test() -> int:
    """Checks of the logic in this file. Uses temporary folders only: no
    network, no real download, nothing installed."""
    import tempfile
    import unittest
    from unittest import mock

    this = sys.modules[__name__]

    class Tests(unittest.TestCase):
        def setUp(self):
            self._tmp = tempfile.TemporaryDirectory()
            self.tmp = Path(self._tmp.name)
            self.addCleanup(self._tmp.cleanup)

        def test_sizes_use_decimal_units(self):
            self.assertEqual(human_size(MODEL_SIZE), "5.95 GB")
            self.assertEqual(human_size(LLAMA_ZIP[1]), "355 MB")
            self.assertEqual(human_size(CUDART_ZIP[1]), "391 MB")

        def test_pinned_downloads_carry_a_hash_and_a_revision(self):
            for entry in (LLAMA_ZIP, CUDART_ZIP):
                self.assertRegex(entry[2], r"^[0-9a-f]{64}$")
            self.assertRegex(MODEL_SHA256, r"^[0-9a-f]{64}$")
            self.assertIn(f"/resolve/{MODEL_REVISION}/", MODEL_URL)
            self.assertNotIn("/resolve/main/", MODEL_URL)

        def _finished_part(self, data: bytes) -> Path:
            dest = self.tmp / "file.bin"
            dest.with_name("file.bin.part").write_bytes(data)
            return dest

        def test_download_with_matching_hash_is_accepted(self):
            data = b"x" * 3000
            dest = self._finished_part(data)
            out = download_file("http://unused.invalid/f", dest, len(data), lambda m: None, None, None,
                                hashlib.sha256(data).hexdigest().upper())
            self.assertEqual(out.read_bytes(), data)
            self.assertFalse(dest.with_name("file.bin.part").exists())

        def test_download_with_wrong_hash_is_deleted_and_refused(self):
            data = b"x" * 3000
            dest = self._finished_part(data)
            with self.assertRaises(SetupError) as ctx:
                download_file("http://unused.invalid/f", dest, len(data), lambda m: None, sha256="0" * 64)
            self.assertIn("integrity check", str(ctx.exception))
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name("file.bin.part").exists())

        def _server(self, content: bytes, requests: list):
            """A stand-in for _fetch_once that behaves like a server holding
            `content`: appends from byte `have`, answers 416 at or past the end."""
            def fetch(url, part, have, expected_size, name, log, progress, cancel, drop_if_slow=False):
                requests.append(have)
                if have and have >= len(content):
                    raise DownloadHttpError(416, f"Download of {name} failed: HTTP 416 from {url}")
                with open(part, "ab" if have else "wb") as f:
                    f.write(content[have:])
                return True, ""
            return fetch

        def test_http_416_restarts_once_and_never_gets_stuck(self):
            dest = self.tmp / "file.bin"
            part = self.tmp / "file.bin.part"
            data = b"the whole file"
            good = hashlib.sha256(data).hexdigest()
            quiet = threading.Event()        # never set: only keeps the waits between tries short
            # A partial file as long as the server's file, pinned size larger:
            # discarded and fetched again from zero; still short, so a clear end.
            part.write_bytes(b"x" * len(data))
            requests = []
            with mock.patch.object(this, "_fetch_once", side_effect=self._server(data, requests)), \
                    mock.patch.object(quiet, "wait", return_value=False):
                with self.assertRaises(SetupError) as ctx:
                    download_file("http://unused.invalid/f", dest, len(data) + 5, lambda m: None,
                                  None, quiet, good)
            # 416, restart from zero, short again, 416 again: an end, not a loop.
            self.assertEqual(requests, [len(data), 0, len(data)])
            self.assertIn("offers only", str(ctx.exception))
            self.assertNotIn("Run the setup again", str(ctx.exception))
            self.assertFalse(part.exists())          # the next run does not meet the same 416
            self.assertFalse(dest.exists())
            # Size not known, hash known: a partial file that is the whole file is verified and used.
            part.write_bytes(data)
            requests = []
            with mock.patch.object(this, "_fetch_once", side_effect=self._server(data, requests)):
                out = download_file("http://unused.invalid/f", dest, 0, lambda m: None, sha256=good)
            self.assertEqual(out.read_bytes(), data)
            self.assertEqual(requests, [len(data)])
            dest.unlink()
            # Size not known, hash known, wrong content: deleted and refused, nothing kept.
            part.write_bytes(b"y" * len(data))
            with mock.patch.object(this, "_fetch_once", side_effect=self._server(data, [])):
                with self.assertRaises(SetupError) as ctx:
                    download_file("http://unused.invalid/f", dest, 0, lambda m: None, sha256=good)
            self.assertIn("integrity check", str(ctx.exception))
            self.assertFalse(part.exists())
            # Neither size nor hash known: nothing to verify, so start from zero once.
            part.write_bytes(b"z" * (len(data) + 3))
            requests = []
            with mock.patch.object(this, "_fetch_once", side_effect=self._server(data, requests)):
                out = download_file("http://unused.invalid/f", dest, 0, lambda m: None)
            self.assertEqual(out.read_bytes(), data)
            self.assertEqual(requests, [len(data) + 3, 0])
            dest.unlink()

            # A 416 on a download that starts at zero is passed on unchanged.
            def refuse(*args, **kwargs):
                raise DownloadHttpError(416, "Download of file.bin failed: HTTP 416 from x")
            with mock.patch.object(this, "_fetch_once", side_effect=refuse):
                with self.assertRaises(DownloadHttpError):
                    download_file("http://unused.invalid/f", dest, len(data), lambda m: None, sha256=good)

        def test_data_that_ends_early_is_continued(self):
            dest = self.tmp / "file.bin"
            data = b"0123456789"
            calls = []
            quiet = threading.Event()

            def halves(url, part, have, expected_size, name, log, progress, cancel, drop_if_slow=False):
                calls.append(have)
                with open(part, "ab" if have else "wb") as f:
                    f.write(data[have:have + 5])
                return True, ""          # a clean end of the data after 5 bytes
            with mock.patch.object(this, "_fetch_once", side_effect=halves), \
                    mock.patch.object(quiet, "wait", return_value=False):
                out = download_file("http://unused.invalid/f", dest, len(data), lambda m: None, None,
                                    quiet, hashlib.sha256(data).hexdigest())
            self.assertEqual(out.read_bytes(), data)
            self.assertEqual(calls, [0, 5])

        def test_verified_file_is_not_downloaded_again(self):
            data = b"y" * 2000
            dest = self.tmp / "file.bin"
            dest.write_bytes(data)
            lines = []
            with mock.patch.object(this, "_fetch_once", side_effect=AssertionError("must not fetch")):
                download_file("http://unused.invalid/f", dest, len(data), lines.append,
                              sha256=hashlib.sha256(data).hexdigest())
            self.assertTrue(any("already downloaded" in ln for ln in lines))

        def test_hashing_honours_cancel(self):
            f = self.tmp / "big.bin"
            f.write_bytes(b"z" * 10)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(SetupCancelled):
                sha256_of(f, cancel)

        def _zip(self, members: dict) -> Path:
            path = self.tmp / "a.zip"
            with zipfile.ZipFile(path, "w") as zf:
                for name, data in members.items():
                    zf.writestr(name, data)
            return path

        def test_unpack_writes_files_and_reports_the_space_needed(self):
            z = self._zip({"build/llama-server.exe": b"MZ", "build/llama.dll": b"12345"})
            self.assertEqual(_zip_unpacked_bytes(z), 7)
            _extract_zip(z, self.tmp / "out")
            self.assertEqual((self.tmp / "out" / "build" / "llama.dll").read_bytes(), b"12345")

        def test_unpack_refuses_paths_outside_the_target(self):
            z = self._zip({"../evil.dll": b"1"})
            with self.assertRaises(SetupError):
                _extract_zip(z, self.tmp / "out")
            self.assertFalse((self.tmp / "evil.dll").exists())

        def test_unpack_honours_cancel(self):
            z = self._zip({"a.dll": b"1", "b.dll": b"2"})
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(SetupCancelled):
                _extract_zip(z, self.tmp / "out", cancel)

        def test_interrupted_engine_install_is_not_reported_as_complete(self):
            src = self.tmp / "unpacked"
            src.mkdir()
            files = []
            for name in ("ggml-cuda.dll", "llama-server.exe", "llama.dll"):
                (src / name).write_bytes(b"1")
                files.append(src / name)

            class CancelAfter:  # lets two files through, then cancels
                def __init__(self):
                    self.calls = 0

                def is_set(self):
                    self.calls += 1
                    return self.calls > 2

            folder = self.tmp / "engine"
            with self.assertRaises(SetupCancelled):
                _place_engine_files(files, folder, CancelAfter())
            # llama-server.exe and ggml-cuda.dll are there, llama.dll is not.
            self.assertTrue((folder / "llama-server.exe").exists() and (folder / "ggml-cuda.dll").exists())
            cfg = {"llm": {"server_binary": str(folder / "llama-server.exe")}}
            item = check_llama(cfg)
            self.assertFalse(item.ok)
            self.assertTrue(item.can_install)
            self.assertIn("interrupted", item.detail)

            marker = _place_engine_files([src / "llama.dll"], folder)
            marker.unlink()   # what install_llama does once the engine has answered
            self.assertTrue(check_llama(cfg).ok)

        def test_engine_without_cuda_backend_is_not_ok(self):
            folder = self.tmp / "engine"
            folder.mkdir()
            (folder / "llama-server.exe").write_bytes(b"1")
            self.assertFalse(check_llama({"llm": {"server_binary": str(folder / "llama-server.exe")}}).ok)

        def test_version_comparison(self):
            self.assertTrue(version_at_least("1.2.5", "1.0"))
            self.assertTrue(version_at_least("1.0", "1.0.0"))
            self.assertTrue(version_at_least("2025.1.15", "2024.10.0"))
            self.assertTrue(version_at_least("1.10.0", "1.9"))
            self.assertFalse(version_at_least("0.1.9", "1.0"))
            self.assertFalse(version_at_least("2024.8.6", "2024.10.0"))
            self.assertFalse(version_at_least("?", "1.0"))
            self.assertFalse(version_at_least("", "1.0"))

        def test_packages_below_the_minimum_or_missing_are_reported(self):
            packages = [("youtube_transcript_api", "youtube-transcript-api>=1.0"),
                        ("yt_dlp", "yt-dlp>=2024.10.0"), ("nvidia.cudnn", "nvidia-cudnn-cu12")]
            self.assertEqual(_unmet_requirements(packages, {
                "youtube_transcript_api": "0.1.2", "yt_dlp": "2025.3.31", "nvidia.cudnn": "?"}),
                ["youtube-transcript-api>=1.0"])
            self.assertEqual(_unmet_requirements(packages, {
                "youtube_transcript_api": "1.2.5", "yt_dlp": "", "nvidia.cudnn": ""}),
                ["yt-dlp>=2024.10.0", "nvidia-cudnn-cu12"])
            self.assertEqual(_unmet_requirements(packages, {
                "youtube_transcript_api": "1.0.0", "yt_dlp": "2024.10.0", "nvidia.cudnn": "9.1"}), [])

        def test_whisper_gpu_needs_both_dlls_where_the_helper_looks(self):
            a, b = self.tmp / "a", self.tmp / "b"
            a.mkdir()
            b.mkdir()
            self.assertEqual(len(whisper_gpu_dlls_missing([a, b, self.tmp / "missing"])), 2)
            (a / "cublas64_12.dll").write_bytes(b"1")
            self.assertEqual(whisper_gpu_dlls_missing([a, b]), ["cuDNN (cudnn*.dll)"])
            (b / "cudnn_ops64_9.dll").write_bytes(b"1")
            self.assertEqual(whisper_gpu_dlls_missing([a, b]), [])
            self.assertEqual(whisper_gpu_dlls_missing([a]), ["cuDNN (cudnn*.dll)"])

        def test_browser_transcripts_are_optional_and_need_package_and_browser(self):
            with mock.patch.object(this, "missing_pip_packages", return_value=["playwright>=1.40"]):
                item = check_browser_transcripts()
                self.assertEqual((item.ok, item.required, item.can_install), (False, False, True))
            with mock.patch.object(this, "missing_pip_packages", return_value=[]), \
                    mock.patch.object(this, "browser_for_transcripts", return_value=None):
                item = check_browser_transcripts()
                self.assertEqual((item.ok, item.required, item.can_install), (False, False, False))
                self.assertIn("playwright install chromium", item.detail)
            with mock.patch.object(this, "missing_pip_packages", return_value=[]), \
                    mock.patch.object(this, "browser_for_transcripts", return_value=self.tmp / "msedge.exe"):
                self.assertTrue(check_browser_transcripts().ok)
            self.assertIn("pip_browser", INSTALLERS)

        def test_gpu_whisper_item_follows_the_dll_check(self):
            with mock.patch.object(this, "missing_pip_packages", return_value=[]), \
                    mock.patch.object(this, "whisper_dll_folders", return_value=[self.tmp]):
                item = check_gpu_whisper()
                self.assertFalse(item.ok)
                self.assertFalse(item.can_install)
                self.assertIn("CPU", item.detail)
                (self.tmp / "cublas64_12.dll").write_bytes(b"1")
                (self.tmp / "cudnn64_9.dll").write_bytes(b"1")
                self.assertTrue(check_gpu_whisper().ok)

        # --- ytc.exe: version check and the download / build decision ---
        VERSION_OK = f"ytc {YTC_VERSION}\n"
        VERSION_OTHER = "ytc 9.9.9\n"

        def _published(self, data: bytes = b"MZ fake program"):
            return (mock.patch.object(this, "YTC_EXE_SIZE", len(data)),
                    mock.patch.object(this, "YTC_EXE_SHA256", hashlib.sha256(data).hexdigest()))

        def _app_folder(self):
            """An empty app folder: nothing of the real project is touched."""
            return mock.patch.object(this, "ROOT", self.tmp)

        def test_version_check_compares_with_the_version_of_this_app(self):
            self.assertEqual(YTC_RELEASE_TAG, f"v{YTC_VERSION}")
            exe = self.tmp / "ytc.exe"
            exe.write_bytes(b"MZ")
            cfg = {"ytc": {"binary": str(exe)}}
            with mock.patch.object(this, "_run", return_value=(0, self.VERSION_OK)) as asked:
                self.assertEqual(ytc_version(exe), (YTC_VERSION, ""))
                self.assertEqual(asked.call_args[0][0], [str(exe), "--version"])
                self.assertTrue(check_ytc(cfg).ok)
            # No ready-made binary published and no Rust: nothing can be installed automatically.
            with mock.patch.object(this, "_run", return_value=(0, self.VERSION_OTHER)), \
                    mock.patch.object(this, "YTC_EXE_SIZE", 0), \
                    mock.patch.object(this, "YTC_EXE_SHA256", ""), \
                    mock.patch.object(shutil, "which", return_value=None):
                self.assertEqual(ytc_version(exe), ("9.9.9", ""))
                item = check_ytc(cfg)
                self.assertFalse(item.ok)
                self.assertFalse(item.can_install)
                self.assertIn(f"does not match this app (found 9.9.9, need {YTC_VERSION})", item.detail)
                self.assertIn("ytc_setup.py --install", item.detail)
                self.assertIn("cargo build --release", item.detail)
                self.assertEqual(item.help_url, RUSTUP_URL)
            # A call that failed says nothing about the version.
            for answer in ((-1, "not a valid application"), (0, "  \n")):
                with mock.patch.object(this, "_run", return_value=answer):
                    self.assertIsNone(ytc_version(exe)[0])
                    item = check_ytc(cfg)
                    self.assertFalse(item.ok)
                    self.assertIn("does not start", item.detail)
                    self.assertNotIn("does not match", item.detail)

        def test_empty_values_mean_nothing_is_published(self):
            # Guards the meaning of the empty values, whatever values the constants carry.
            with mock.patch.object(this, "YTC_EXE_SIZE", 0), mock.patch.object(this, "YTC_EXE_SHA256", ""):
                self.assertFalse(ytc_exe_published())
            with mock.patch.object(this, "YTC_EXE_SIZE", 5_000_000), mock.patch.object(this, "YTC_EXE_SHA256", ""):
                self.assertFalse(ytc_exe_published())
            with mock.patch.object(this, "YTC_EXE_SIZE", 0), mock.patch.object(this, "YTC_EXE_SHA256", "a" * 64):
                self.assertFalse(ytc_exe_published())
            with mock.patch.object(this, "YTC_EXE_SIZE", 5_000_000), \
                    mock.patch.object(this, "YTC_EXE_SHA256", "not a hash"):
                self.assertFalse(ytc_exe_published())
            with mock.patch.object(this, "YTC_EXE_SIZE", 5_000_000), \
                    mock.patch.object(this, "YTC_EXE_SHA256", "A" * 64):
                self.assertTrue(ytc_exe_published())
            self.assertEqual(YTC_EXE_URL,
                             f"https://github.com/{GITHUB_REPO}/releases/download/{YTC_RELEASE_TAG}/ytc.exe")
            # Either both values are filled in or neither.
            self.assertEqual(YTC_EXE_SIZE > 0, bool(YTC_EXE_SHA256))

        def test_order_download_first_then_build(self):
            self.assertEqual(ytc_install_order(True, True), ["download", "build"])
            self.assertEqual(ytc_install_order(True, False), ["download"])
            self.assertEqual(ytc_install_order(False, True), ["build"])
            self.assertEqual(ytc_install_order(False, False), [])
            # A ytc.exe that was built here is rebuilt first.
            self.assertEqual(ytc_install_order(True, True, prefer_build=True), ["build", "download"])
            self.assertEqual(ytc_install_order(True, False, prefer_build=True), ["download"])

        def test_check_reports_which_way_applies(self):
            size, sha = self._published()
            with mock.patch.object(this, "find_ytc_binary", return_value=(None, [])):
                with size, sha, mock.patch.object(shutil, "which", return_value=None):
                    item = check_ytc({})
                    self.assertTrue(item.can_install)
                    self.assertEqual(item.download_bytes, YTC_EXE_SIZE)
                    self.assertIn("downloaded from github.com", item.detail)
                    self.assertIn("sha256", item.detail)
                    self.assertNotIn("cargo", item.detail)
                size, sha = self._published()
                with size, sha, mock.patch.object(shutil, "which", return_value="cargo"):
                    item = check_ytc({})
                    self.assertIn("if the download is not possible", item.detail)
                    self.assertIn("cargo", item.detail)
                with mock.patch.object(this, "YTC_EXE_SIZE", 0), mock.patch.object(this, "YTC_EXE_SHA256", ""):
                    with mock.patch.object(shutil, "which", return_value="cargo"):
                        item = check_ytc({})
                        self.assertTrue(item.can_install)
                        self.assertEqual(item.download_bytes, 0)
                        self.assertIn("No ready-made ytc.exe is published", item.detail)
                        self.assertIn("built here", item.detail)
                    with mock.patch.object(shutil, "which", return_value=None):
                        item = check_ytc({})
                        self.assertFalse(item.ok)
                        self.assertFalse(item.can_install)
                        self.assertEqual(item.help_url, RUSTUP_URL)
                        self.assertIn(RUSTUP_URL, item.detail)      # option 1: build
                        self.assertIn(RELEASES_URL, item.detail)    # option 2: fetch by hand
                        with self.assertRaises(SetupError):
                            install_ytc(lambda m: None, None, None)

        def _run_install(self, download, build, cargo=True, published=True, version_text=None):
            """install_ytc in an empty app folder with fakes for the network,
            cargo and the finished program. Returns (steps taken, log lines)."""
            steps, lines = [], []

            def fake_download(dest, log, progress, cancel):
                steps.append("download")
                download(dest)

            def fake_build(log, cancel):
                steps.append("build")
                build()

            data = b"MZ fake program"
            patches = [
                self._app_folder(),
                mock.patch.object(this, "_load_cfg", return_value={}),
                mock.patch.object(this, "_download_ytc", side_effect=fake_download),
                mock.patch.object(this, "_build_ytc", side_effect=fake_build),
                mock.patch.object(shutil, "which", return_value="cargo" if cargo else None),
                mock.patch.object(this, "_run", return_value=(0, version_text or self.VERSION_OK)),
                mock.patch.object(this, "YTC_EXE_SIZE", len(data) if published else 0),
                mock.patch.object(this, "YTC_EXE_SHA256", hashlib.sha256(data).hexdigest() if published else ""),
            ]
            for patch in patches:
                patch.start()
                self.addCleanup(patch.stop)
            self.steps, self.lines = steps, lines
            install_ytc(lines.append, None, None)
            return steps, lines

        def _write_built(self):
            out = self.tmp / "target" / "release"
            out.mkdir(parents=True, exist_ok=True)
            (out / "ytc.exe").write_bytes(b"MZ built")

        def test_install_downloads_and_does_not_build_when_published(self):
            dests = []

            def download(dest):
                dests.append(dest)
                dest.write_bytes(b"MZ fake program")

            steps, _ = self._run_install(download, self._write_built)
            self.assertEqual(steps, ["download"])
            self.assertEqual(dests, [self.tmp / "ytc.exe"])

        def test_install_builds_when_nothing_is_published(self):
            def never(dest):
                raise AssertionError("must not download")

            steps, _ = self._run_install(never, self._write_built, published=False)
            self.assertEqual(steps, ["build"])

        def test_failed_download_falls_through_to_the_build(self):
            def unreachable(dest):
                raise SetupError("The release is not reachable (HTTP 404 from x).")

            steps, lines = self._run_install(unreachable, self._write_built)
            self.assertEqual(steps, ["download", "build"])
            self.assertTrue(any("not reachable" in ln for ln in lines))

        def test_failed_download_without_rust_reports_the_reason(self):
            def unreachable(dest):
                raise SetupError("The release is not reachable (HTTP 404 from x).")

            with self.assertRaises(SetupError) as ctx:
                self._run_install(unreachable, self._write_built, cargo=False)
            self.assertEqual(self.steps, ["download"])
            self.assertIn("not reachable", str(ctx.exception))

        def test_cancel_is_not_swallowed_by_the_fallback(self):
            def cancelled(dest):
                raise SetupCancelled()

            with self.assertRaises(SetupCancelled):
                self._run_install(cancelled, self._write_built)
            self.assertEqual(self.steps, ["download"])

        def test_program_in_use_is_replaced_where_it_is(self):
            used = self.tmp / "ytc.exe"
            used.write_bytes(b"MZ other")
            dests = []

            def download(dest):
                dests.append(dest)
                dest.write_bytes(b"MZ fake program")

            steps, _ = self._run_install(download, self._write_built)
            self.assertEqual((steps, dests), (["download"], [used.resolve()]))

        def test_install_fails_when_the_app_uses_a_program_that_does_not_match(self):
            def download(dest):
                dest.write_bytes(b"MZ fake program")

            with self.assertRaises(SetupError) as ctx:
                self._run_install(download, self._write_built, cargo=False, version_text=self.VERSION_OTHER)
            self.assertIn("does not match this app (found 9.9.9", str(ctx.exception))

        def test_http_404_is_explained(self):
            def not_found(*args, **kwargs):
                raise DownloadHttpError(404, "Download of ytc.exe failed: HTTP 404 from x")

            def server_error(*args, **kwargs):
                raise DownloadHttpError(500, "Download of ytc.exe failed: HTTP 500 from x")

            with mock.patch.object(this, "download_file", side_effect=not_found):
                with self.assertRaises(SetupError) as ctx:
                    _download_ytc(self.tmp / "ytc.exe", lambda m: None, None, None)
            text = str(ctx.exception)
            self.assertIn("not reachable", text)
            self.assertIn("private", text)
            self.assertIn("not been published", text)
            with mock.patch.object(this, "download_file", side_effect=server_error):
                with self.assertRaises(DownloadHttpError) as ctx:
                    _download_ytc(self.tmp / "ytc.exe", lambda m: None, None, None)
            self.assertEqual(ctx.exception.code, 500)

        def test_download_is_verified_before_it_gets_its_name(self):
            data = b"MZ fake program"
            size, sha = self._published(data)
            dest = self.tmp / "ytc.exe"
            with size, sha:
                dest.with_name("ytc.exe.part").write_bytes(data)   # a finished download: no network needed
                _download_ytc(dest, lambda m: None, None, None)
                self.assertEqual(dest.read_bytes(), data)
                dest.unlink()
                dest.with_name("ytc.exe.part").write_bytes(b"MZ tampered pro")   # same size, other content
                with self.assertRaises(SetupError) as ctx:
                    _download_ytc(dest, lambda m: None, None, None)
                self.assertIn("integrity check", str(ctx.exception))
                self.assertFalse(dest.exists())

        # --- everything is looked up inside the app folder only ---
        def test_nothing_outside_the_app_folder_is_searched(self):
            with self._app_folder(), mock.patch.object(this, "LLAMA_DIR", self.tmp / "bin" / "llama"), \
                    mock.patch.object(this, "MODELS_DIR", self.tmp / "models"):
                _, engine_places = find_llama_server({})
                self.assertEqual(engine_places, [self.tmp / "bin" / "llama" / "llama-server.exe"])
                _, ytc_places = find_ytc_binary({})
                self.assertEqual(ytc_places, [self.tmp / "target" / "release" / "ytc.exe", self.tmp / "ytc.exe"])
                _, configured = find_ytc_binary({"ytc": {"binary": "tools/ytc.exe"}})
                self.assertEqual(configured[0], self.tmp / "tools" / "ytc.exe")
                self.assertIsNone(find_model(""))
                (self.tmp / "models").mkdir()
                small = self.tmp / "models" / "small.gguf"
                small.write_bytes(b"x" * 10)
                self.assertIsNone(find_model(""))                       # too small to be a base model
                self.assertEqual(find_model("models/small.gguf"), small.resolve())   # configured wins

        def test_one_database(self):
            with self._app_folder():
                self.assertEqual(app_db_path({}), self.tmp / "data" / "ytc.db")
                self.assertEqual(app_db_path({"ytc": {"db_path": "  "}}), self.tmp / "data" / "ytc.db")
                self.assertEqual(app_db_path({"ytc": {"db_path": "lib/my.db"}}), self.tmp / "lib" / "my.db")
                elsewhere = self.tmp / "elsewhere" / "x.db"
                self.assertEqual(app_db_path({"ytc": {"db_path": str(elsewhere)}}), elsewhere)
                item = check_database({})
                self.assertTrue(item.ok)
                self.assertIn("created on first use", item.detail)
                self.assertFalse((self.tmp / "data").exists())       # nothing created
                (self.tmp / "data").mkdir()
                (self.tmp / "data" / "ytc.db").write_bytes(b"")
                self.assertEqual(check_database({}).detail, str(self.tmp / "data" / "ytc.db"))

        def test_python_is_accepted_by_running_it_not_by_its_path(self):
            store = Path(r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe")
            with mock.patch.object(this, "_run", return_value=(0, "3\n")):
                self.assertTrue(_is_working_python(store))
            # The App Execution Alias stub starts nothing usable.
            with mock.patch.object(this, "_run", return_value=(9009, "")):
                self.assertFalse(_is_working_python(store))

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main())
