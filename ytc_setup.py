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
import queue
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
# Two builds of that release, one pair of zips each: (file name, size in bytes,
# sha256). The release is not marked immutable on GitHub, so the size alone
# proves nothing: every download is checked against the hash before it is
# unpacked or run. The hashes are the 'digest' fields of the release assets
# (GitHub API, releases/tags/<LLAMA_TAG>).
# CUDA 12.4 build: for cards before RTX 50. Its runtime zip contains
# cublas64_12.dll, which the Whisper transcriber (ctranslate2) needs as well,
# so one download serves both.
LLAMA_ZIP_12_4 = (f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip", 355_011_546,
                  "14969ae6831abfc836295e8f4b8e09e686a063b559a8c974412001d9f5ce4f7f")
CUDART_ZIP_12_4 = ("cudart-llama-bin-win-cuda-12.4-x64.zip", 391_443_627,
                   "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6")
MIN_DRIVER_FOR_CUDA_12_4 = 551.61
# CUDA 13.3 build: for RTX 50 cards (compute capability 12.0 and up), where the
# 12.4 build crashed ("GPU is lost", 3 of 7 runs on an RTX 5090 Laptop) and
# this one passed 8 of 8. It needs a driver whose CUDA version is 13.3 or newer.
LLAMA_ZIP_13_3 = (f"llama-{LLAMA_TAG}-bin-win-cuda-13.3-x64.zip", 198_836_745,
                  "86c6546685ffe2f0954bac062d0bdb347694a2b5eee6f5ddeb1bbfe82d2be1b6")
CUDART_ZIP_13_3 = ("cudart-llama-bin-win-cuda-13.3-x64.zip", 390_970_417,
                   "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e")
MIN_DRIVER_CUDA_FOR_13_3 = (13, 3)
# Cards from this compute capability on (RTX 50, Blackwell) get the 13.3 build.
RTX50_COMPUTE_CAP = (12, 0)


@dataclass(frozen=True)
class EngineBuild:
    """One build of the engine: the llama.cpp zip and its CUDA runtime zip."""
    name: str          # written to ENGINE_BUILD_MARKER, e.g. "cuda-13.3"
    label: str         # for the user, e.g. "CUDA 13.3"
    llama_zip: tuple
    cudart_zip: tuple

    @property
    def download_bytes(self) -> int:
        return self.llama_zip[1] + self.cudart_zip[1]


BUILD_CUDA_12_4 = EngineBuild("cuda-12.4", "CUDA 12.4", LLAMA_ZIP_12_4, CUDART_ZIP_12_4)
BUILD_CUDA_13_3 = EngineBuild("cuda-13.3", "CUDA 13.3", LLAMA_ZIP_13_3, CUDART_ZIP_13_3)
ENGINE_BUILDS = {b.name: b for b in (BUILD_CUDA_12_4, BUILD_CUDA_13_3)}
# Written into bin/llama by the install: which build is in the folder.
ENGINE_BUILD_MARKER = ".ytc_engine_build"

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
YTC_RELEASE_TAG = "v0.1.3"
# What 'ytc.exe --version' must print for this app: the launcher and ytc.exe
# belong to one release and work as a pair.
YTC_VERSION = YTC_RELEASE_TAG.lstrip("v")
YTC_EXE_NAME = "ytc.exe"
YTC_EXE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{YTC_RELEASE_TAG}/{YTC_EXE_NAME}"
YTC_EXE_SIZE = 12431360
YTC_EXE_SHA256 = "bcae988112abec8e226d4323ed64266210ec439e9e60ca608f8cd7e3a69b9839"

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
# Lines in that marker. A marker without the "build:" line was written by
# version 0.1.2 or older.
INCOMPLETE_BUILD_PREFIX = "build: "
INCOMPLETE_ALL_PLACED = "all files placed; the start test (--version) has not passed yet"
# How long 'llama-server.exe --version' may take (install and setup check).
ENGINE_VERSION_TIMEOUT_S = 60

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


def _run(cmd, timeout=30, env: Optional[dict] = None) -> Tuple[int, str]:
    try:
        res = subprocess.run(
            cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout,
            encoding="utf-8", errors="replace", creationflags=NO_WINDOW, env=env,
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
    # Only for the LLM engine: the build the install fetches (ENGINE_BUILDS key),
    # and a build offered only as the user's explicit choice, with a warning
    # (see choose_alternative_build).
    engine_build: str = ""
    alternative_build: str = ""


# --- Graphics card and driver (read with nvidia-smi) ----------------------------
# The query used since 0.1.3; the older one (without compute_cap) is the
# fallback for an nvidia-smi that does not know that field.
GPU_QUERY_FIELDS = ("name", "compute_cap", "driver_version", "memory.total")
GPU_QUERY_FIELDS_OLD = ("name", "memory.total", "driver_version")


@dataclass
class GpuInfo:
    name: str
    memory_mib: str                         # digits only, e.g. "24463"; "" when not readable
    driver: str                             # e.g. "621.24"
    compute_cap: Optional[Tuple[int, int]]  # e.g. (12, 0); None when not known


@dataclass
class GpuFacts:
    """What nvidia-smi says. Empty gpus: no NVIDIA card or no answer."""
    gpus: List[GpuInfo]
    driver_cuda: Optional[Tuple[int, int]] = None   # the driver's CUDA version, e.g. (13, 5)

    def highest_compute_cap(self) -> Optional[Tuple[int, int]]:
        """llama-server uses every CUDA card it sees (the app passes no device
        option), so the rule follows the newest card: one RTX 50 card among
        several is enough to need the 13.3 build. None when no card's compute
        capability is known."""
        known = [g.compute_cap for g in self.gpus if g.compute_cap is not None]
        return max(known) if known else None


def _major_minor(text: str) -> Optional[Tuple[int, int]]:
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_gpu_query(text: str, fields=GPU_QUERY_FIELDS) -> List[GpuInfo]:
    """Lines of 'nvidia-smi --query-gpu=<fields> --format=csv,noheader[,nounits]',
    one per card. Lines that do not have the expected fields (an error text,
    garbage) are skipped. Units such as ' MiB' are removed. A driver text that
    is not a number alone does not drop the card: nvidia-smi marks
    driver_version as deprecated, and a deprecated field answers with a text
    such as '[Requested functionality has been deprecated]'; the card and its
    compute capability are still needed."""
    gpus = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(fields):
            continue
        value = dict(zip(fields, parts))
        memory = re.match(r"\d+", value.get("memory.total", ""))
        driver = value.get("driver_version", "")
        compute_cap = _major_minor(value.get("compute_cap", ""))
        if not value.get("name") or not (re.match(r"\d+\.\d+", driver) or compute_cap or memory):
            continue
        gpus.append(GpuInfo(value["name"], memory.group(0) if memory else "", driver, compute_cap))
    return gpus


def parse_driver_cuda_version(text: str) -> Optional[Tuple[int, int]]:
    """The driver's CUDA version from 'nvidia-smi -q': the line 'CUDA UMD
    Version' (newer drivers mark 'CUDA Version' as deprecated), else 'CUDA
    Version' (older drivers). None when neither line carries a number."""
    for label in ("CUDA UMD Version", "CUDA Version"):
        m = re.search(rf"^\s*{label}\s*:\s*(\d+)\.(\d+)", text or "", re.MULTILINE)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def read_gpu_facts() -> GpuFacts:
    """Ask nvidia-smi. Never raises: a missing nvidia-smi, a timeout or text
    that cannot be read give empty or partial facts. 'nvidia-smi -q' is only
    run for an RTX 50 card, the only case where its answer matters."""
    code, out = _run(["nvidia-smi", f"--query-gpu={','.join(GPU_QUERY_FIELDS)}", "--format=csv,noheader"])
    gpus = parse_gpu_query(out, GPU_QUERY_FIELDS) if code == 0 else []
    if not gpus and code not in (0, -1):
        # nvidia-smi answered with an error, e.g. an older one that does not
        # know compute_cap. (-1: it is missing or timed out; asking again would not help.)
        code, out = _run(["nvidia-smi", f"--query-gpu={','.join(GPU_QUERY_FIELDS_OLD)}",
                          "--format=csv,noheader,nounits"])
        gpus = parse_gpu_query(out, GPU_QUERY_FIELDS_OLD) if code == 0 else []
    facts = GpuFacts(gpus)
    cap = facts.highest_compute_cap()
    if cap is not None and cap >= RTX50_COMPUTE_CAP:
        code, out = _run(["nvidia-smi", "-q"])
        facts.driver_cuda = parse_driver_cuda_version(out) if code == 0 else None
    return facts


def _cuda_text(version: Optional[Tuple[int, int]]) -> str:
    return f"{version[0]}.{version[1]}" if version else "unknown"


RTX50_CUDA_12_4_WARNING = ("The CUDA 12.4 engine is known to crash on RTX 50 cards (\"GPU is lost\"). "
                           "The CUDA 13.3 engine does not, but it needs an NVIDIA driver with CUDA 13.3 or newer.")


@dataclass(frozen=True)
class EngineChoice:
    """Which engine build fits this PC (see choose_engine_build)."""
    build: EngineBuild
    rtx50: bool = False
    driver_problem: str = ""     # not empty: the build cannot be installed with this driver


def choose_engine_build(compute_cap: Optional[Tuple[int, int]],
                        driver_cuda: Optional[Tuple[int, int]]) -> EngineChoice:
    """The rule of 0.1.3:
    - card before RTX 50, or compute capability not known: CUDA 12.4 (as before);
    - RTX 50 and driver CUDA >= 13.3: CUDA 13.3;
    - RTX 50 and an older or unreadable driver CUDA version: CUDA 13.3 is the
      right build, but the driver must be updated first. The 12.4 build is only
      an explicit choice of the user, with RTX50_CUDA_12_4_WARNING."""
    if compute_cap is None or compute_cap < RTX50_COMPUTE_CAP:
        return EngineChoice(BUILD_CUDA_12_4)
    if driver_cuda is not None and driver_cuda >= MIN_DRIVER_CUDA_FOR_13_3:
        return EngineChoice(BUILD_CUDA_13_3, rtx50=True)
    found = (f"this driver supports CUDA {_cuda_text(driver_cuda)}" if driver_cuda
             else "the driver's CUDA version could not be read (nvidia-smi -q)")
    problem = (f"Update the NVIDIA driver (CUDA {_cuda_text(MIN_DRIVER_CUDA_FOR_13_3)} or newer needed; "
               f"{found}). The CUDA 12.4 engine is known to crash on RTX 50 cards.")
    return EngineChoice(BUILD_CUDA_13_3, rtx50=True, driver_problem=problem)


def engine_choice_for(facts: GpuFacts) -> EngineChoice:
    return choose_engine_build(facts.highest_compute_cap(), facts.driver_cuda)


def check_gpu(facts: Optional[GpuFacts] = None) -> Item:
    facts = read_gpu_facts() if facts is None else facts
    if not facts.gpus:
        return Item("gpu", "NVIDIA graphics card", False,
                    "No NVIDIA card or driver found (nvidia-smi did not answer). "
                    "The LLM engine needs an NVIDIA card with 8 GB VRAM or more and a current driver.",
                    required=True, help_url=NVIDIA_DRIVER_URL)
    first = facts.gpus[0]
    name, mem, driver = first.name, first.memory_mib, first.driver
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
    """Run a snippet in python_exe(); its stdout, or None when it failed. The
    child writes UTF-8 (as _run reads it): otherwise Python 3.10-3.14 writes a
    pipe in the ANSI code page, and a path like C:\\Users\\José comes back
    mangled, or the print fails for a character outside that code page."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    code, out = _run([str(python_exe()), "-c", code_text] + list(args), timeout=timeout, env=env)
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
    # The packages being installed is not enough: check that the DLLs are in
    # one of the folders transcribe_cuda.py really searches.
    dlls_missing = whisper_gpu_dlls_missing(whisper_dll_folders())
    if dlls_missing:
        return Item("pip_gpu", label, False,
                    "the pip packages are installed, but transcribe_cuda.py does not find "
                    + " and ".join(dlls_missing) + " in the folders it searches (this Python's and the "
                    "per-user site-packages\\nvidia\\...\\bin, the project folder, bin\\llama). Whisper runs on "
                    "the CPU. Reinstall the two packages into the Python that runs the app, or copy the "
                    "DLLs into bin\\llama.",
                    required=False)
    return Item("pip_gpu", label, True,
                "installed, and transcribe_cuda.py finds cublas64_12.dll and cuDNN; "
                "videos without captions can be transcribed on the graphics card", required=False)


# The newest Python every package above has a Windows build for (checked on
# PyPI 2026-11-16). faster-whisper needs onnxruntime, and onnxruntime 1.32.0
# has Windows wheels for Python 3.11-3.14 only and no source package, so pip
# cannot install faster-whisper on Python 3.15. Newer onnxruntime releases may
# fix that, so a newer Python is still tried; only the message explains it.
NEWEST_TESTED_PYTHON = (3, 14)
PYTHON_DOWNLOAD_URL = "https://www.python.org/downloads/windows/"


def python_version() -> Tuple[int, int]:
    """(major, minor) of python_exe(), the Python the packages go into."""
    code, out = _run([str(python_exe()), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], timeout=30)
    try:
        major, minor = out.strip().split(".")
        if code == 0:
            return int(major), int(minor)
    except ValueError:
        pass
    return sys.version_info[0], sys.version_info[1]


def too_new_python_hint(missing: List[str]) -> str:
    """An explanation when faster-whisper is missing on a Python newer than
    NEWEST_TESTED_PYTHON (empty otherwise)."""
    version = python_version()
    if version <= NEWEST_TESTED_PYTHON or not any(r.startswith("faster-whisper") for r in missing):
        return ""
    newest = "%d.%d" % NEWEST_TESTED_PYTHON
    return (f" This is Python {version[0]}.{version[1]}: faster-whisper needs onnxruntime, which had no Windows "
            f"build for it yet (checked 2026-11-16). Install Python {newest} from python.org (tick \"Add "
            f"python.exe to PATH\"), then run Create_Desktop_Shortcut.bat again: the launchers prefer a Python up "
            f"to {newest}. Without faster-whisper the app still runs; only videos that must be transcribed "
            f"locally (no captions, or YouTube refuses the transcript) fail.")


def check_python_packages(missing: Optional[List[str]] = None) -> Item:
    missing = missing_pip_packages() if missing is None else missing
    if not missing:
        return Item("pip", "Python packages", True, "yt-dlp, faster-whisper, youtube-transcript-api, requests")
    detail = "missing or below the minimum version: " + ", ".join(missing) + "."
    hint = too_new_python_hint(missing)
    if hint and all(r.startswith("faster-whisper") for r in missing):
        # Only faster-whisper is left, and pip probably cannot get it on this
        # Python. Not required: opening the window at every start would not
        # help, and the app runs without it. The install stays on offer for
        # when onnxruntime ships a build for this Python.
        return Item("pip", "Python packages", False, detail + hint, required=False,
                    can_install=True, download_bytes=250 * 1000 * 1000, help_url=PYTHON_DOWNLOAD_URL)
    return Item("pip", "Python packages", False, detail + hint,
                can_install=True, download_bytes=250 * 1000 * 1000,
                help_url=PYTHON_DOWNLOAD_URL if hint else "")


# --- Windows Smart App Control ----------------------------------------------------
SAC_KEY = r"SYSTEM\CurrentControlSet\Control\CI\Policy"
SAC_VALUE = "VerifiedAndReputablePolicyState"
SAC_ON_MESSAGE = ("Windows Smart App Control is on. It blocks the LLM engine, which is not code-signed. The engine "
                  "can only run with Smart App Control off. That is your security decision; Windows may not let "
                  "you switch it back on without resetting Windows.")
SAC_EVALUATION_MESSAGE = ("Windows Smart App Control is in evaluation mode. Windows may switch it on later by "
                          "itself; it then blocks the LLM engine, which is not code-signed.")


def smart_app_control_state(reg=None) -> int:
    """VerifiedAndReputablePolicyState, read-only: 1 = on, 2 = evaluation mode,
    0 = off. 0 as well when the value is missing or cannot be read, and on
    other systems than Windows. Never raises. reg: a stand-in for winreg (tests)."""
    try:
        if reg is None:
            if sys.platform != "win32":
                return 0
            import winreg as reg
        with reg.OpenKey(reg.HKEY_LOCAL_MACHINE, SAC_KEY, 0, reg.KEY_READ) as key:
            value, _kind = reg.QueryValueEx(key, SAC_VALUE)
        return int(value) if int(value) in (0, 1, 2) else 0
    except Exception:  # missing key or value, no access, unexpected type
        return 0


def check_smart_app_control(state: int) -> Optional[Item]:
    """A row for the setup window, or None when Smart App Control is off."""
    label = "Windows Smart App Control"
    if state == 1:
        return Item("sac", label, False, SAC_ON_MESSAGE, required=True)
    if state == 2:
        return Item("sac", label, True, "Warning: " + SAC_EVALUATION_MESSAGE, required=False)
    return None


# --- Microsoft Visual C++ runtime --------------------------------------------------
# llama-server.exe and its DLLs (and ytc.exe) are built against the Visual C++
# 2015-2022 runtime and do not carry it; a fresh Windows does not have all of it.
# Without it the engine fails its start test with 0xC0000135 ('a DLL is
# missing'), and installing the engine again cannot change that.
VC_RUNTIME_DLLS = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll")
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
VC_RUNTIME_MESSAGE = (f"The LLM engine and ytc.exe need the Microsoft Visual C++ 2015-2022 runtime (x64). "
                      f"Install it with vc_redist.x64.exe from {VC_REDIST_URL}, then run the setup check again.")


def _vc_runtime_folders() -> List[Path]:
    """Where Windows looks for these DLLs when it starts the engine: its own
    folder, the 64-bit system folder, then the folders on PATH."""
    windir = Path(os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows")
    # A 32-bit Python sees SysWOW64 as System32; Sysnative is the real one then.
    system = windir / "Sysnative" if (windir / "Sysnative").is_dir() else windir / "System32"
    path = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p.strip()]
    return [LLAMA_DIR, system] + path


def vc_runtime_missing(folders=None) -> List[str]:
    """The Visual C++ runtime DLLs (VC_RUNTIME_DLLS) that are in none of the
    folders (default: _vc_runtime_folders()). Always [] on other systems than
    Windows. Never raises."""
    if folders is None:
        if sys.platform != "win32":
            return []
        folders = _vc_runtime_folders()
    missing = []
    for name in VC_RUNTIME_DLLS:
        found = False
        for folder in folders:
            try:
                if (Path(folder) / name).is_file():
                    found = True
                    break
            except (OSError, ValueError):
                continue
        if not found:
            missing.append(name)
    return missing


def check_vc_runtime(missing: List[str]) -> Optional[Item]:
    """A row for the setup window, or None when the runtime is there."""
    if not missing:
        return None
    return Item("vcredist", "Microsoft Visual C++ runtime (2015-2022, x64)", False,
                f"{', '.join(missing)} not found. {VC_RUNTIME_MESSAGE}", required=True, help_url=VC_REDIST_URL)


# --- Why the engine does not start ------------------------------------------------
# {who}: the program that stopped ('the engine', 'the web server', ...).
KNOWN_EXIT_CODES = {
    0xC0E90002: "Windows (Smart App Control / Code Integrity) blocked {who}",
    0xC0000135: "a DLL is missing",
    0xC0000409: "{who} stopped itself after an error",
}
STATUS_DLL_NOT_FOUND = 0xC0000135
# CreateProcess error: 'An Application Control policy has blocked this file.'
WINERROR_APP_CONTROL_BLOCKED = 4551


def describe_exit_code(code: int, who: str = "the engine") -> str:
    """An exit code for people: decimal and hex, plus what a known Windows code
    means; `who` names the program that stopped. Negative values (a signed
    reading of the same 32 bits) are shown as Windows shows them:
    describe_exit_code(3236495362) == describe_exit_code(-1058471934)
    == 'exit code 3236495362 (0xC0E90002): Windows (Smart App Control / Code
    Integrity) blocked the engine'."""
    unsigned = int(code) & 0xFFFFFFFF
    text = f"exit code {unsigned} (0x{unsigned:08X})"
    meaning = KNOWN_EXIT_CODES.get(unsigned)
    return f"{text}: {meaning.format(who=who)}" if meaning else text


def dll_missing_hint(code: int) -> str:
    """For 0xC0000135 (a DLL is missing): which DLLs are usually missing and
    where to get them. Empty for every other code."""
    if int(code) & 0xFFFFFFFF != STATUS_DLL_NOT_FOUND:
        return ""
    return (" (usually the Microsoft Visual C++ 2015-2022 runtime: install vc_redist.x64.exe from "
            f"{VC_REDIST_URL})")


def last_log_lines(path: Path, count: int = 15) -> str:
    """The last `count` lines of a log file, for an error message. When the file
    is empty or does not exist: '<name> is empty: the engine stopped before it
    wrote anything'. Never raises."""
    empty = f"{Path(path).name} is empty: the engine stopped before it wrote anything"
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return empty
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-count:]) if lines else empty


def engine_failure_text(code: int, output: str) -> str:
    """Why '--version' failed, from its exit code and what it printed."""
    out = (output or "").strip()
    return (describe_exit_code(code) + dll_missing_hint(code)
            + (f". It printed: {out[:300]}" if out else ". It printed nothing."))


def engine_timeout_text(timeout: int) -> str:
    return (f"it did not answer '--version' within {timeout} s. That usually means Windows stopped it with an "
            f"error window (for example 'Bad Image', when Smart App Control or an antivirus blocks one of its "
            f"DLLs) that waits for a click, or an antivirus is still scanning the new files. Look for such a "
            f"window, check Smart App Control (Windows Security > App & browser control), then run the setup "
            f"check again.")


SEM_FAILCRITICALERRORS = 0x0001
SEM_NOOPENFILEERRORBOX = 0x8000


class _NoErrorBoxes:
    """While active, Windows shows no error window ('Bad Image', 'DLL not
    found') for this process and the programs it starts (they inherit the
    error mode), so a broken engine fails at once with its exit code instead of
    waiting for a click. Process-wide, restored afterwards. Does nothing on
    other systems, and any failure here is ignored.

    The error mode belongs to the whole process, and several threads may use
    this at once (the engine and the web server start in parallel, each with a
    '--version' check). So the first one in sets the mode, the last one out
    restores it, and a thread leaving early never undoes it for the others."""

    _lock = threading.Lock()
    _users = 0
    _kernel32 = None
    _previous = None

    def __enter__(self):
        if sys.platform != "win32":
            return self
        cls = type(self)
        with cls._lock:
            if cls._users == 0:
                try:
                    import ctypes
                    kernel32 = ctypes.WinDLL("kernel32")
                    kernel32.SetErrorMode.argtypes = [ctypes.c_uint]
                    kernel32.SetErrorMode.restype = ctypes.c_uint
                    wanted = SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX
                    previous = kernel32.SetErrorMode(wanted)
                    kernel32.SetErrorMode(previous | wanted)     # keep flags that were set before
                    cls._kernel32, cls._previous = kernel32, previous
                except Exception:
                    cls._kernel32, cls._previous = None, None
            cls._users += 1
        return self

    def __exit__(self, *exc):
        if sys.platform != "win32":
            return False
        cls = type(self)
        with cls._lock:
            cls._users = max(0, cls._users - 1)
            if cls._users == 0 and cls._kernel32 is not None:
                try:
                    cls._kernel32.SetErrorMode(cls._previous)
                except Exception:
                    pass
                cls._kernel32, cls._previous = None, None
        return False


def run_engine_version(exe: Path, timeout: int = ENGINE_VERSION_TIMEOUT_S) -> Tuple[bool, str]:
    """Run 'llama-server.exe --version' hidden, without error windows. It loads
    no model. Returns (True, "") or (False, the reason for the user)."""
    try:
        with _NoErrorBoxes():
            res = subprocess.run([str(exe), "--version"], capture_output=True, stdin=subprocess.DEVNULL,
                                 timeout=timeout, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        return False, engine_timeout_text(timeout)
    except OSError as e:
        if getattr(e, "winerror", None) == WINERROR_APP_CONTROL_BLOCKED:
            return False, f"Windows (Smart App Control / Code Integrity) blocked the engine: {e}"
        return False, f"Windows could not start it: {e}"
    if res.returncode == 0:
        return True, ""
    return False, engine_failure_text(res.returncode, (res.stdout or "") + (res.stderr or ""))


# --- The LLM engine ------------------------------------------------------------------
def installed_engine_build(folder: Path) -> Optional[str]:
    """Which build is in the engine folder: the marker written by the install
    ('cuda-12.4', 'cuda-13.3') wins. Without it, the CUDA runtime DLLs tell the
    CUDA generation: cublas64_13.dll -> 'cuda-13.x', cublas64_12.dll ->
    'cuda-12.4'. Both together (a switch from 12.4 to 13.3 leaves the _12 DLLs
    behind on purpose, for Whisper) count as 13. None when nothing tells."""
    try:
        text = (folder / ENGINE_BUILD_MARKER).read_text(encoding="utf-8", errors="replace").strip()
        first = text.splitlines()[0].strip() if text else ""
        if re.fullmatch(r"cuda-\d+\.(\d+|x)", first):
            return first
    except OSError:
        pass
    try:
        if any(folder.glob("cublas64_13*.dll")):
            return "cuda-13.x"
        if any(folder.glob("cublas64_12*.dll")):
            return BUILD_CUDA_12_4.name
    except OSError:
        pass
    return None


def is_cuda_12_build(name: Optional[str]) -> bool:
    return bool(name) and name.startswith("cuda-12.")


def _engine_files_placed(folder: Path) -> bool:
    """For a folder with the incomplete marker: were all files placed, so that
    only the start test is missing? Markers of this version say so in a line.
    An older marker (no 'build:' line) has no such line, so the files that
    llama-server.exe cannot start without are looked at instead."""
    try:
        text = (folder / LLAMA_INCOMPLETE_MARKER).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if INCOMPLETE_BUILD_PREFIX in text:
        return INCOMPLETE_ALL_PLACED in text
    # 0.1.2 moved the llama zip files first and the CUDA runtime DLLs of the
    # cudart zip last: those are looked for too.
    try:
        return (all((folder / name).exists() for name in ("llama-server.exe", "llama.dll", "ggml-cuda.dll"))
                and all(any(folder.glob(pattern))
                        for pattern in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")))
    except OSError:
        return False


def choose_alternative_build(item: Item) -> None:
    """The user chose the item's alternative build (the CUDA 12.4 engine on an
    RTX 50 card whose driver is too old for 13.3) after seeing
    RTX50_CUDA_12_4_WARNING: the install button then fetches that build."""
    build = ENGINE_BUILDS.get(item.alternative_build)
    if build is None:
        return
    item.engine_build = build.name
    item.alternative_build = ""
    item.can_install = True
    item.download_bytes = build.download_bytes


def check_llama(cfg: dict, gpu: Optional[GpuFacts] = None, sac: Optional[int] = None,
                vc_missing: Optional[List[str]] = None) -> Item:
    """gpu, sac, vc_missing: what check_all has read already (read here when None)."""
    label = "LLM engine (llama.cpp, PrismML build) + CUDA runtime"
    facts = read_gpu_facts() if gpu is None else gpu
    choice = engine_choice_for(facts)
    sac = smart_app_control_state() if sac is None else sac
    vc_missing = vc_runtime_missing() if vc_missing is None else vc_missing
    want = choice.build

    def offer(detail: str) -> Item:
        """Not OK; the install of the right build is offered when it can run."""
        if sac == 1:
            return Item("llama", label, False, f"{detail} The download waits until Smart App Control is off "
                        f"(see 'Windows Smart App Control').", can_install=False)
        if vc_missing:
            # The engine cannot start without it: installing it again would fail the same way.
            return Item("llama", label, False, f"{detail} The download waits until the Microsoft Visual C++ "
                        f"runtime is installed (see 'Microsoft Visual C++ runtime').", can_install=False,
                        help_url=VC_REDIST_URL)
        if choice.driver_problem:
            return Item("llama", label, False, f"{detail} {choice.driver_problem}", can_install=False,
                        help_url=NVIDIA_DRIVER_URL, alternative_build=BUILD_CUDA_12_4.name)
        return Item("llama", label, False, detail, can_install=True, download_bytes=want.download_bytes,
                    engine_build=want.name)

    path, _ = find_llama_server(cfg)
    if path is None:
        return offer(f"llama-server.exe not found. Will be placed in {LLAMA_DIR}.")
    folder = path.parent
    incomplete = folder / LLAMA_INCOMPLETE_MARKER
    if incomplete.exists():
        if not _engine_files_placed(folder):
            return offer(f"{folder} is incomplete: an install into it was interrupted. "
                         "The install will be repeated.")
        # Everything was placed; only the start test failed (for example
        # Windows blocked a DLL). The cause may be gone: test again, once.
        started, reason = run_engine_version(path)
        if not started:
            return offer(f"The engine in {folder} was installed but does not start: {reason.rstrip('.')}. "
                         "The install can be repeated.")
        try:
            incomplete.unlink()
        except OSError:
            pass
    if not (folder / "ggml-cuda.dll").exists():
        return offer(f"{path} has no CUDA backend (ggml-cuda.dll); it would run on the CPU only.")
    # Files copied into bin/llama by hand carry no marker: llama-server.exe
    # cannot start without llama.dll, so a missing llama.dll means half a copy.
    if folder == LLAMA_DIR.resolve() and not (folder / "llama.dll").exists():
        return offer(f"{folder} is incomplete (llama.dll is missing). The install will be repeated.")
    has_runtime = any(folder.glob("cublas64_*.dll"))
    detail = str(path) if has_runtime else f"{path} (CUDA runtime DLLs come from elsewhere on this PC)"
    if not choice.rtx50:
        return Item("llama", label, True, detail)

    # An RTX 50 card: the CUDA 12.4 build can crash there.
    installed = installed_engine_build(folder)
    if is_cuda_12_build(installed):
        if folder != LLAMA_DIR.resolve():
            # The engine named by llm.server_binary in config.json. The install
            # only writes bin/llama and the app would keep using this one, so
            # offering it would never end: a warning that says what to do.
            return Item("llama", label, False,
                        f"{path} (llm.server_binary in config.json) is the CUDA 12.4 build. On RTX 50 cards it "
                        f"can crash (\"GPU is lost\"). Replace it with the {want.label} build, or remove "
                        f"llm.server_binary from config.json so that the setup check offers the {want.label} "
                        f"build for {LLAMA_DIR}." + (f" {choice.driver_problem}" if choice.driver_problem else ""),
                        required=False, help_url=NVIDIA_DRIVER_URL if choice.driver_problem else "")
        if choice.driver_problem:
            # Nothing to install until the driver is updated, and the 12.4
            # engine still runs most of the time: a warning, not a blocker
            # (also when the user chose 12.4 on purpose).
            return Item("llama", label, False,
                        f"{path}: the CUDA 12.4 build is installed. On RTX 50 cards it can crash (\"GPU is lost\"). "
                        f"{choice.driver_problem} After the update, the setup check offers the "
                        f"{want.label} build.", required=False, help_url=NVIDIA_DRIVER_URL)
        return offer(f"The CUDA 12.4 build is installed. On RTX 50 cards it can crash (\"GPU is lost\"). "
                     f"Recommended: switch to the {want.label} build ({human_size(want.download_bytes)} download).")
    if installed is None:
        return Item("llama", label, True, f"{detail} (build not known; on RTX 50 cards use the "
                                          f"{want.label} build)")
    if choice.driver_problem:
        if facts.driver_cuda is None:
            # Not read (nvidia-smi -q failed or timed out) is not "too old": the
            # installed 13.x engine stays as it is; no 12.4 build is suggested.
            return Item("llama", label, True, f"{detail} (the driver's CUDA version could not be read "
                                              f"with nvidia-smi -q; it needs CUDA 13.3 or newer)")
        return offer(f"{path}: the {installed.replace('cuda-', 'CUDA ')} build is installed.")
    return Item("llama", label, True, detail)


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
    with _NoErrorBoxes():   # a missing DLL fails at once instead of waiting behind an error window
        code, out = _run([str(path), "--version"], timeout=20)
    if code != 0:
        return None, out.strip()[:300] or describe_exit_code(code, "ytc.exe") + dll_missing_hint(code)
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
    missing = missing_pip_packages()
    gpu = read_gpu_facts()
    sac = smart_app_control_state()
    sac_item = check_smart_app_control(sac)
    vc_missing = vc_runtime_missing()
    vc_item = check_vc_runtime(vc_missing)
    items = [
        check_gpu(gpu),
        *([sac_item] if sac_item else []),
        *([vc_item] if vc_item else []),
        check_python_packages(missing),
        check_llama(cfg, gpu, sac, vc_missing),
        check_model(cfg),
        check_ytc(cfg),
        check_database(cfg),
        check_whisper(),
        check_gpu_whisper(),
        check_browser_transcripts(),
    ]
    if too_new_python_hint(missing):
        # faster-whisper cannot be installed on this Python, so the Whisper
        # model and the NVIDIA libraries (about 1.2 GB, used only by Whisper)
        # would be downloaded for nothing.
        # The rows' own text ("Whisper still works on the CPU", "faster-whisper
        # fetches it by itself") is not true here, so it is replaced.
        for item in items:
            if item.key in ("whisper", "pip_gpu") and not item.ok:
                item.can_install = False
                item.detail = ("not installed: only faster-whisper uses it, and faster-whisper is not "
                               "installed and probably cannot be on this Python (see 'Python packages'). "
                               "Without it, videos that must be transcribed locally fail.")
    return items


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


def _file_in_use(path: Path) -> bool:
    """True when an existing file cannot be opened for writing. On Windows that
    is an .exe or .dll a running program has loaded (llama-server.exe started
    from this folder): such a file cannot be replaced. Opening it changes nothing."""
    try:
        if not path.is_file():
            return False
        with open(path, "r+b"):
            return False
    except PermissionError:
        return True
    except OSError:
        return False


def engine_in_use_text(folder: Path, name: str) -> str:
    return (f"{name} in {folder} is in use: the LLM engine (or another program using its files) is running, "
            f"so the engine cannot be replaced. Press 'Stop Engine' in the Control Center (and stop any "
            f"running crawl), then install again. Nothing in {folder} was changed.")


def _place_engine_files(files, target_dir: Path, cancel: Optional[threading.Event] = None,
                        build: str = "") -> Path:
    """Move unpacked files into the engine folder. Returns the marker file,
    which is written before the first move: from then until the caller deletes
    it, the folder counts as incomplete, whatever stops the install (cancel,
    full disk, closed window, power loss). When every file is in place, the
    marker says so, and the build name is written to ENGINE_BUILD_MARKER (the
    old one is removed before the first move, so it never names a half-replaced
    folder). A file to be replaced that is in use (the engine is running) stops
    it before anything is changed."""
    target_dir.mkdir(parents=True, exist_ok=True)
    files = list(files)
    for f in files:
        if _file_in_use(target_dir / Path(f).name):
            raise SetupError(engine_in_use_text(target_dir, Path(f).name))
    marker = target_dir / LLAMA_INCOMPLETE_MARKER
    marker.write_text("An install of the LLM engine into this folder did not finish.\n"
                      "Run the setup check again; it repeats the install.\n"
                      f"{INCOMPLETE_BUILD_PREFIX}{build or 'unknown'}\n", encoding="utf-8")
    (target_dir / ENGINE_BUILD_MARKER).unlink(missing_ok=True)
    for f in files:
        _check_cancel(cancel)
        os.replace(f, target_dir / Path(f).name)
    if build:
        (target_dir / ENGINE_BUILD_MARKER).write_text(build + "\n", encoding="utf-8")
    with open(marker, "a", encoding="utf-8") as f:
        f.write(INCOMPLETE_ALL_PLACED + "\n")
    return marker


def install_llama(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event],
                  build: str = "") -> None:
    """Install the engine build `build` (an ENGINE_BUILDS key; the user's
    explicit choice), or, when empty, the build that choose_engine_build picks
    for this PC's card and driver."""
    if smart_app_control_state() == 1:
        # The engine could not run: do not download 600-750 MB for nothing.
        raise SetupError(SAC_ON_MESSAGE)
    vc_missing = vc_runtime_missing()
    if vc_missing:
        # Same: the start test would fail with 'a DLL is missing' after the download.
        raise SetupError(f"{', '.join(vc_missing)} not found. {VC_RUNTIME_MESSAGE}")
    if build:
        chosen = ENGINE_BUILDS.get(build)
        if chosen is None:
            raise SetupError(f"Unknown engine build '{build}'.")
    else:
        choice = engine_choice_for(read_gpu_facts())
        if choice.driver_problem:
            raise SetupError(choice.driver_problem)
        chosen = choice.build
    if _file_in_use(LLAMA_DIR / "llama-server.exe"):
        # The engine is running from bin/llama: do not download 600-750 MB that
        # could not be put in place (_place_engine_files checks every file again).
        raise SetupError(engine_in_use_text(LLAMA_DIR, "llama-server.exe"))
    log(f"Engine build: {chosen.label}.")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zips = []
    for name, size, sha256 in (chosen.llama_zip, chosen.cudart_zip):
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
        marker = _place_engine_files(files, LLAMA_DIR, cancel, chosen.name)

        started, reason = run_engine_version(LLAMA_DIR / "llama-server.exe")
        if not started:
            raise SetupError(f"The downloaded llama-server.exe does not start: {reason}")
        marker.unlink()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    for z in zips:
        z.unlink(missing_ok=True)
    log(f"LLM engine ({chosen.label} build) installed in {LLAMA_DIR}")


def install_model(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    log(f"Downloading {MODEL_NAME} ({human_size(MODEL_SIZE)}) from huggingface.co/prism-ml ...")
    download_file(MODEL_URL, MODELS_DIR / MODEL_NAME, MODEL_SIZE, log, progress, cancel, MODEL_SHA256)
    log(f"Base model saved to {MODELS_DIR / MODEL_NAME}")


def _stream(cmd, log: LogFn, cancel: Optional[threading.Event], cwd: Optional[Path] = None,
            extra_env: Optional[dict] = None, hide: Optional[Callable[[str], bool]] = None) -> int:
    """Run a command hidden and pass its output lines to log() (except lines
    for which hide(line) is true)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **(extra_env or {}))
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=NO_WINDOW, env=env,
        )
    except OSError as e:
        raise SetupError(f"Could not start {cmd[0]}: {e}") from e
    # The lines are read in a helper thread: pip prints nothing while it
    # downloads a large wheel into a pipe, and Cancel must work meanwhile.
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def pump() -> None:
        try:
            for text in proc.stdout:
                lines.put(text)
        except (OSError, ValueError):   # pipe closed after taskkill
            pass
        finally:
            lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    while True:
        if cancel is not None and cancel.is_set():
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
            raise SetupCancelled()
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            break
        if line.strip() and not (hide is not None and hide(line)):
            log(line.rstrip())
    return proc.wait()


def install_pip(log: LogFn, progress: Optional[ProgressFn], cancel: Optional[threading.Event]) -> None:
    missing = missing_pip_packages()
    if not missing:
        return
    log("Installing Python packages: " + ", ".join(missing))
    # One pip call per package: pip installs nothing when one requirement of a
    # call fails, and faster-whisper fails on a Python without onnxruntime
    # builds, which would also take yt-dlp and the others down with it.
    failed = []
    for requirement in missing:
        # --upgrade: a package that is installed but below its minimum version is in the list too.
        code = _stream([str(python_exe()), "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
                        requirement], log, cancel)
        if code != 0:
            failed.append(requirement)
    if failed:
        raise SetupError("pip could not install " + ", ".join(failed) + " (see the lines above)."
                         + too_new_python_hint(failed))


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
    code = _stream([str(python_exe()), "-c", snippet], log, cancel,
                   extra_env={"HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}, hide=is_harmless_hf_warning)
    if code != 0:
        raise SetupError("The Whisper model could not be downloaded (see the lines above). Once faster-whisper "
                         "is installed, it fetches the model by itself on first use.")
    log(f"Whisper model downloaded ({WHISPER_CACHE}).")


def is_harmless_hf_warning(line: str) -> bool:
    """huggingface_hub's notes while the Whisper model downloads that need no
    action: the optional 'hf_xet' package is not installed (it falls back to
    plain HTTP), and Windows gives no symlinks (the cache still works). The
    symlink note is also switched off with HF_HUB_DISABLE_SYMLINKS_WARNING;
    these patterns catch it should a version ignore that. Any other line that
    names hf_xet (an import error, an hf_xet exception) is shown."""
    if "Xet Storage is enabled for this repo" in line and "package is not installed" in line:
        return True
    return any(p in line for p in ("cache-system uses symlinks", "To support symlinks on Windows"))


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
            # Only the engine item carries a build (the one the check chose, or the user's explicit choice).
            extra = {"build": item.engine_build} if item.engine_build else {}
            INSTALLERS[item.key](log, progress, cancel, **extra)
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
    """config.json, or {} (the defaults) when it is missing or cannot be read.
    utf-8-sig: Notepad may save the file with a BOM. A broken file is named."""
    path = ROOT / "config.json"
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"{path.name} could not be read ({e}); the default settings are used.")
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
    # The real functions, for the tests of them; every test runs with stand-ins
    # (see setUp), so no test depends on this PC's card, driver or registry,
    # and none can start the engine.
    real_read_gpu_facts = read_gpu_facts
    real_smart_app_control_state = smart_app_control_state
    real_run_engine_version = run_engine_version
    real_vc_runtime_missing = vc_runtime_missing

    RTX5090 = GpuInfo("NVIDIA GeForce RTX 5090 Laptop GPU", "24463", "621.24", (12, 0))
    RTX4070 = GpuInfo("NVIDIA GeForce RTX 4070 Laptop GPU", "8188", "566.36", (8, 9))

    class Tests(unittest.TestCase):
        def setUp(self):
            self._tmp = tempfile.TemporaryDirectory()
            self.tmp = Path(self._tmp.name)
            self.addCleanup(self._tmp.cleanup)
            for patch in (mock.patch.object(this, "LLAMA_DIR", self.tmp / "app" / "bin" / "llama"),
                          mock.patch.object(this, "read_gpu_facts", return_value=GpuFacts([])),
                          mock.patch.object(this, "smart_app_control_state", return_value=0),
                          mock.patch.object(this, "vc_runtime_missing", return_value=[]),
                          mock.patch.object(this, "run_engine_version",
                                            side_effect=AssertionError("the self-test must not start the engine"))):
                patch.start()
                self.addCleanup(patch.stop)

        def test_sizes_use_decimal_units(self):
            self.assertEqual(human_size(MODEL_SIZE), "5.95 GB")
            self.assertEqual(human_size(LLAMA_ZIP_12_4[1]), "355 MB")
            self.assertEqual(human_size(CUDART_ZIP_12_4[1]), "391 MB")
            self.assertEqual(human_size(LLAMA_ZIP_13_3[1]), "199 MB")
            self.assertEqual(human_size(CUDART_ZIP_13_3[1]), "391 MB")
            self.assertEqual(BUILD_CUDA_12_4.download_bytes, 746_455_173)
            self.assertEqual(BUILD_CUDA_13_3.download_bytes, 589_807_162)
            self.assertEqual(human_size(BUILD_CUDA_13_3.download_bytes), "590 MB")

        def test_pinned_downloads_carry_a_hash_and_a_revision(self):
            for entry in (LLAMA_ZIP_12_4, CUDART_ZIP_12_4, LLAMA_ZIP_13_3, CUDART_ZIP_13_3):
                self.assertRegex(entry[2], r"^[0-9a-f]{64}$")
            # Each pair belongs to its build, all from the pinned release.
            for build in ENGINE_BUILDS.values():
                version = build.name.split("-")[1]
                self.assertIn(f"-{version}-", build.llama_zip[0])
                self.assertIn(f"-{version}-", build.cudart_zip[0])
                self.assertIn(LLAMA_TAG, build.llama_zip[0])
            self.assertEqual(LLAMA_ZIP_13_3[2], "86c6546685ffe2f0954bac062d0bdb347694a2b5eee6f5ddeb1bbfe82d2be1b6")
            self.assertEqual(CUDART_ZIP_13_3[2], "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e")
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

        # --- 0.1.3: card, driver and engine build ---
        def test_nvidia_smi_query_is_parsed(self):
            # A real line from an RTX 5090 Laptop GPU (fields as GPU_QUERY_FIELDS, with units).
            gpus = parse_gpu_query("NVIDIA GeForce RTX 5090 Laptop GPU, 12.0, 621.24, 24463 MiB\n")
            self.assertEqual(gpus, [RTX5090])
            self.assertEqual(GPU_QUERY_FIELDS, ("name", "compute_cap", "driver_version", "memory.total"))
            # Two cards, one compute capability not available.
            gpus = parse_gpu_query("NVIDIA GeForce RTX 4070 Laptop GPU, 8.9, 566.36, 8188 MiB\r\n"
                                   "NVIDIA GeForce GTX 1080, [N/A], 566.36, 8192 MiB\r\n")
            self.assertEqual([g.compute_cap for g in gpus], [(8, 9), None])
            self.assertEqual(gpus[1].memory_mib, "8192")
            # The older query (no compute_cap, no units).
            gpus = parse_gpu_query("NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 566.36", GPU_QUERY_FIELDS_OLD)
            self.assertEqual(gpus, [GpuInfo("NVIDIA GeForce RTX 4070 Laptop GPU", "8188", "566.36", None)])
            # driver_version is deprecated (to be removed in CUDA 14.0); a deprecated field answers
            # with a text. The card and its compute capability are kept.
            gpus = parse_gpu_query("NVIDIA GeForce RTX 5090 Laptop GPU, 12.0, "
                                   "[Requested functionality has been deprecated], 24463 MiB\n")
            self.assertEqual([(g.compute_cap, g.memory_mib) for g in gpus], [((12, 0), "24463")])
            self.assertEqual(GpuFacts(gpus).highest_compute_cap(), (12, 0))
            check_gpu(GpuFacts(gpus))      # a driver text that is not a number does not raise
            # Garbage, error texts, a header, empty.
            for text in ("", "\n\n", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
                         'Field "compute_cap" is not a valid field to query.',
                         "name, compute_cap, driver_version, memory.total [MiB]", "a, b, c, d", None):
                self.assertEqual(parse_gpu_query(text), [], text)

        def test_driver_cuda_version_is_parsed(self):
            self.assertEqual(parse_driver_cuda_version("    CUDA UMD Version                      : 13.5\n"), (13, 5))
            # Newer drivers: the UMD line wins over the deprecated one, wherever it is.
            both = ("Driver Version                            : 621.24\n"
                    "CUDA Version                              : 13.0 (Deprecated. Use CUDA UMD Version instead)\n"
                    "CUDA UMD Version                          : 13.5\n")
            self.assertEqual(parse_driver_cuda_version(both), (13, 5))
            deprecated_text = ("CUDA Version : Deprecated. Use CUDA UMD Version instead\n"
                               "CUDA UMD Version : 13.5\n")
            self.assertEqual(parse_driver_cuda_version(deprecated_text), (13, 5))
            # Older drivers: only 'CUDA Version'.
            self.assertEqual(parse_driver_cuda_version("Driver Version : 566.36\r\nCUDA Version : 12.7\r\n"), (12, 7))
            for text in ("", None, "garbage\n", "CUDA Version : N/A", "CUDA UMD Version : Deprecated",
                         "Failed to initialize NVML: Driver/library version mismatch"):
                self.assertIsNone(parse_driver_cuda_version(text), text)

        def test_engine_build_rule_table(self):
            table = [
                # (compute capability, driver CUDA) -> (build, RTX 50, driver problem)
                ((None, None), ("cuda-12.4", False, False)),
                ((None, (13, 5)), ("cuda-12.4", False, False)),
                (((8, 9), None), ("cuda-12.4", False, False)),
                (((8, 9), (13, 5)), ("cuda-12.4", False, False)),      # decision A
                (((11, 9), (13, 5)), ("cuda-12.4", False, False)),
                (((12, 0), (13, 5)), ("cuda-13.3", True, False)),
                (((12, 0), (13, 3)), ("cuda-13.3", True, False)),
                (((12, 1), (14, 0)), ("cuda-13.3", True, False)),
                (((12, 0), (13, 2)), ("cuda-13.3", True, True)),
                (((12, 0), (12, 8)), ("cuda-13.3", True, True)),
                (((12, 0), None), ("cuda-13.3", True, True)),
            ]
            for (cap, cuda), (build, rtx50, problem) in table:
                choice = choose_engine_build(cap, cuda)
                self.assertEqual((choice.build.name, choice.rtx50, bool(choice.driver_problem)),
                                 (build, rtx50, problem), (cap, cuda))
            text = choose_engine_build((12, 0), (12, 8)).driver_problem
            self.assertIn("Update the NVIDIA driver (CUDA 13.3 or newer needed", text)
            self.assertIn("CUDA 12.8", text)
            self.assertIn("The CUDA 12.4 engine is known to crash on RTX 50 cards.", text)
            self.assertIn("could not be read", choose_engine_build((12, 0), None).driver_problem)
            # Several cards: the newest one decides; unknown ones are ignored.
            self.assertEqual(GpuFacts([RTX4070, RTX5090]).highest_compute_cap(), (12, 0))
            unknown = GpuInfo("x", "", "566.36", None)
            self.assertEqual(GpuFacts([unknown, RTX4070]).highest_compute_cap(), (8, 9))
            self.assertIsNone(GpuFacts([unknown]).highest_compute_cap())
            self.assertIsNone(GpuFacts([]).highest_compute_cap())
            self.assertEqual(engine_choice_for(GpuFacts([RTX4070, RTX5090], (13, 5))).build, BUILD_CUDA_13_3)

        def test_reading_the_card_degrades_safely(self):
            sample_line = "NVIDIA GeForce RTX 5090 Laptop GPU, 12.0, 621.24, 24463 MiB\n"

            def run_with(answers):
                calls = []

                def fake(cmd, timeout=30):
                    calls.append(cmd[1])
                    return answers.get(cmd[1], (-1, "[WinError 2] The system cannot find the file specified"))
                with mock.patch.object(this, "_run", side_effect=fake):
                    facts = real_read_gpu_facts()
                return facts, calls

            new_query = f"--query-gpu={','.join(GPU_QUERY_FIELDS)}"
            old_query = f"--query-gpu={','.join(GPU_QUERY_FIELDS_OLD)}"
            # No nvidia-smi (or a timeout): asked once, nothing more.
            facts, calls = run_with({})
            self.assertEqual((facts.gpus, facts.driver_cuda, calls), ([], None, [new_query]))
            # An nvidia-smi without compute_cap: the old query, and no 'nvidia-smi -q'.
            facts, calls = run_with({new_query: (2, 'Field "compute_cap" is not a valid field to query.'),
                                     old_query: (0, "NVIDIA GeForce GTX 1080, 8192, 551.86\n")})
            self.assertEqual(calls, [new_query, old_query])
            self.assertEqual((facts.gpus[0].compute_cap, facts.highest_compute_cap()), (None, None))
            self.assertEqual(engine_choice_for(facts).build, BUILD_CUDA_12_4)
            # RTX 4070: no 'nvidia-smi -q'.
            facts, calls = run_with({new_query: (0, "NVIDIA GeForce RTX 4070 Laptop GPU, 8.9, 566.36, 8188 MiB\n")})
            self.assertEqual(calls, [new_query])
            # RTX 5090: the driver's CUDA version is read.
            facts, calls = run_with({new_query: (0, sample_line), "-q": (0, "CUDA UMD Version : 13.5\n")})
            self.assertEqual((calls, facts.driver_cuda), ([new_query, "-q"], (13, 5)))
            self.assertEqual(engine_choice_for(facts).build, BUILD_CUDA_13_3)
            # 'nvidia-smi -q' times out or prints garbage: unknown, so the driver must be checked.
            for answer in ((-1, "timed out after 30 seconds"), (0, "garbage")):
                facts, _ = run_with({new_query: (0, sample_line), "-q": answer})
                self.assertIsNone(facts.driver_cuda)
                self.assertTrue(engine_choice_for(facts).driver_problem)
            # check_gpu reads the first card as before.
            item = check_gpu(GpuFacts([RTX4070]))
            self.assertTrue(item.ok)
            self.assertEqual(item.detail, "NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, driver 566.36")
            self.assertFalse(check_gpu(GpuFacts([])).ok)

        def _engine(self, dlls=("cublas64_12.dll", "cublasLt64_12.dll"), build_marker=None, folder=None) -> dict:
            folder = self.tmp / "engine" if folder is None else folder
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("llama-server.exe", "llama.dll", "ggml-cuda.dll") + tuple(dlls):
                (folder / name).write_bytes(b"1")
            if build_marker is not None:
                (folder / ENGINE_BUILD_MARKER).write_text(build_marker, encoding="utf-8")
            return {"llm": {"server_binary": str(folder / "llama-server.exe")}}

        def test_installed_build_from_marker_or_dll_names(self):
            folder = self.tmp / "engine"
            folder.mkdir()
            self.assertIsNone(installed_engine_build(folder))
            self.assertIsNone(installed_engine_build(self.tmp / "missing"))
            (folder / "cublas64_12.dll").write_bytes(b"1")
            (folder / "cublasLt64_12.dll").write_bytes(b"1")
            self.assertEqual(installed_engine_build(folder), "cuda-12.4")
            (folder / "cublas64_13.dll").write_bytes(b"1")          # both generations, no marker: 13
            self.assertEqual(installed_engine_build(folder), "cuda-13.x")
            (folder / ENGINE_BUILD_MARKER).write_text("cuda-12.4\n", encoding="utf-8")   # the marker wins
            self.assertEqual(installed_engine_build(folder), "cuda-12.4")
            (folder / ENGINE_BUILD_MARKER).write_text("cuda-13.3\n", encoding="utf-8")
            self.assertEqual(installed_engine_build(folder), "cuda-13.3")
            for garbage in ("", "\n", "hello", "cuda-13.3; rm -rf", "CUDA 13.3"):
                (folder / ENGINE_BUILD_MARKER).write_text(garbage, encoding="utf-8")
                self.assertEqual(installed_engine_build(folder), "cuda-13.x", garbage)   # back to the DLL names
            self.assertTrue(is_cuda_12_build("cuda-12.4"))
            self.assertFalse(is_cuda_12_build("cuda-13.x") or is_cuda_12_build(None))

        def test_install_writes_the_build_marker_only_when_all_files_are_placed(self):
            src = self.tmp / "unpacked"
            src.mkdir()
            names = ("llama-server.exe", "llama.dll", "ggml-cuda.dll", "cublas64_13.dll")
            for name in names:
                (src / name).write_bytes(b"1")
            folder = self.tmp / "engine"
            folder.mkdir()
            (folder / ENGINE_BUILD_MARKER).write_text("cuda-12.4\n", encoding="utf-8")   # an older install
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(SetupCancelled):
                _place_engine_files([src / n for n in names], folder, cancel, "cuda-13.3")
            self.assertFalse((folder / ENGINE_BUILD_MARKER).exists())       # never names a half-replaced folder
            text = (folder / LLAMA_INCOMPLETE_MARKER).read_text(encoding="utf-8")
            self.assertIn("build: cuda-13.3", text)
            self.assertNotIn(INCOMPLETE_ALL_PLACED, text)
            marker = _place_engine_files([src / n for n in names], folder, None, "cuda-13.3")
            self.assertEqual((folder / ENGINE_BUILD_MARKER).read_text(encoding="utf-8").strip(), "cuda-13.3")
            self.assertIn(INCOMPLETE_ALL_PLACED, marker.read_text(encoding="utf-8"))
            self.assertEqual(installed_engine_build(folder), "cuda-13.3")

        def test_running_engine_is_not_replaced(self):
            src = self.tmp / "unpacked"
            src.mkdir()
            names = ("llama-server.exe", "llama.dll", "ggml-base.dll", "cublas64_13.dll")
            for name in names:
                (src / name).write_bytes(b"new")
            folder = self.tmp / "engine"
            folder.mkdir()
            for name in names[:3]:
                (folder / name).write_bytes(b"old")
            (folder / ENGINE_BUILD_MARKER).write_text("cuda-12.4\n", encoding="utf-8")
            # A DLL the running engine has loaded: refused before anything is changed.
            with mock.patch.object(this, "_file_in_use", side_effect=lambda p: p.name == "ggml-base.dll"):
                with self.assertRaises(SetupError) as ctx:
                    _place_engine_files([src / n for n in names], folder, None, "cuda-13.3")
            self.assertIn("ggml-base.dll", str(ctx.exception))
            self.assertIn("Stop Engine", str(ctx.exception))
            self.assertFalse((folder / LLAMA_INCOMPLETE_MARKER).exists())
            self.assertEqual(installed_engine_build(folder), "cuda-12.4")
            self.assertEqual([(folder / n).read_bytes() for n in names[:3]], [b"old"] * 3)
            self.assertFalse((folder / "cublas64_13.dll").exists())
            self.assertTrue(all((src / n).exists() for n in names))
            # The install refuses before downloading while llama-server.exe in bin/llama is in use.
            fetched = []
            with mock.patch.object(this, "_file_in_use", side_effect=lambda p: p.name == "llama-server.exe"), \
                    mock.patch.object(this, "download_file", side_effect=lambda *a, **k: fetched.append(a)), \
                    mock.patch.object(this, "DOWNLOAD_DIR", self.tmp / "downloads"):
                with self.assertRaises(SetupError) as ctx:
                    install_llama(lambda m: None, None, None, "cuda-13.3")
            self.assertEqual(fetched, [])
            self.assertIn("llama-server.exe", str(ctx.exception))
            # install_missing reports it as a SetupError, not as a raw PermissionError.
            with mock.patch.object(this, "_file_in_use", return_value=True):
                errors = install_missing([Item("llama", "engine", False, "", can_install=True,
                                               engine_build="cuda-13.3")], lambda m: None)
            self.assertEqual(len(errors), 1)
            self.assertNotIn("PermissionError", errors[0])
            # The real test: a missing file, a folder and a closed file are not in use.
            self.assertFalse(_file_in_use(self.tmp / "missing.dll"))
            self.assertFalse(_file_in_use(folder))
            self.assertFalse(_file_in_use(folder / "llama.dll"))
            self.assertEqual((folder / "llama.dll").read_bytes(), b"old")
            if sys.platform == "win32":
                # The running interpreter's own .exe is loaded, like a running llama-server.exe.
                self.assertTrue(_file_in_use(Path(sys.executable)))

        def test_rtx50_with_the_cuda_12_4_build_is_offered_the_switch(self):
            cfg = self._engine(folder=this.LLAMA_DIR)          # bin/llama, where the install writes
            good = GpuFacts([RTX5090], (13, 5))
            item = check_llama(cfg, good, 0)
            self.assertEqual((item.ok, item.required, item.can_install), (False, True, True))
            self.assertEqual((item.engine_build, item.download_bytes), ("cuda-13.3", 589_807_162))
            self.assertIn("The CUDA 12.4 build is installed. On RTX 50 cards it can crash (\"GPU is lost\")",
                          item.detail)
            self.assertIn("switch to the CUDA 13.3 build (590 MB download)", item.detail)
            self.assertEqual(total_download_bytes([item]), 589_807_162)
            # Driver too old: nothing to install, a warning only (also after an explicit 12.4 choice).
            item = check_llama(cfg, GpuFacts([RTX5090], (12, 8)), 0)
            self.assertEqual((item.ok, item.required, item.can_install, item.alternative_build),
                             (False, False, False, ""))
            self.assertIn("Update the NVIDIA driver", item.detail)
            self.assertEqual(item.help_url, NVIDIA_DRIVER_URL)
            # A 12.4 engine named in config.json outside bin/llama: the install would not
            # replace it, so no (endless) offer; a warning that says what to do.
            outside = self._engine()
            for facts in (good, GpuFacts([RTX5090], (12, 8))):
                item = check_llama(outside, facts, 0)
                self.assertEqual((item.ok, item.required, item.can_install, item.alternative_build),
                                 (False, False, False, ""), facts)
                self.assertIn("remove llm.server_binary from config.json", item.detail)
            # The 13.3 build (marker; or the DLL left behind by a switch): fine.
            cfg = self._engine(dlls=("cublas64_13.dll",), build_marker="cuda-13.3\n")
            self.assertTrue(check_llama(cfg, good, 0).ok)
            # A 13.3 engine and a driver CUDA version that could not be read (nvidia-smi -q
            # failed once): still OK, no 12.4 suggested. A driver read as too old still offers.
            item = check_llama(cfg, GpuFacts([RTX5090], None), 0)
            self.assertEqual((item.ok, item.alternative_build), (True, ""))
            self.assertIn("could not be read", item.detail)
            self.assertFalse(check_llama(cfg, GpuFacts([RTX5090], (12, 8)), 0).ok)

        def test_cards_before_rtx50_are_unchanged(self):
            for marker, dlls in ((None, ("cublas64_12.dll",)), ("cuda-13.3\n", ("cublas64_13.dll",)), (None, ())):
                folder = self.tmp / "engine"
                shutil.rmtree(folder, ignore_errors=True)
                cfg = self._engine(dlls=dlls, build_marker=marker)
                for facts in (GpuFacts([RTX4070]), GpuFacts([RTX4070], (13, 5)), GpuFacts([])):
                    self.assertTrue(check_llama(cfg, facts, 0).ok, (marker, dlls, facts))
            item = check_llama({"llm": {"server_binary": str(self.tmp / "none" / "llama-server.exe")}},
                               GpuFacts([RTX4070]), 0)
            self.assertEqual((item.ok, item.can_install, item.engine_build, item.download_bytes),
                             (False, True, "cuda-12.4", 746_455_173))

        def test_rtx50_with_an_old_driver_offers_12_4_only_as_an_explicit_choice(self):
            missing = {"llm": {"server_binary": str(self.tmp / "none" / "llama-server.exe")}}
            old = GpuFacts([RTX5090], (12, 8))
            item = check_llama(missing, old, 0)
            self.assertEqual((item.ok, item.required, item.can_install), (False, True, False))
            self.assertEqual((item.alternative_build, item.engine_build, item.download_bytes), ("cuda-12.4", "", 0))
            self.assertIn("Update the NVIDIA driver", item.detail)
            self.assertIn("known to crash", RTX50_CUDA_12_4_WARNING)
            choose_alternative_build(item)
            self.assertEqual((item.can_install, item.engine_build, item.alternative_build, item.download_bytes),
                             (True, "cuda-12.4", "", 746_455_173))
            # A 13.x engine with a driver that cannot run it: the same offer.
            item = check_llama(self._engine(dlls=("cublas64_13.dll",)), old, 0)
            self.assertEqual((item.ok, item.can_install, item.alternative_build), (False, False, "cuda-12.4"))
            self.assertIn("CUDA 13.x build is installed", item.detail)
            # With a good driver, no choice is needed.
            item = check_llama(missing, GpuFacts([RTX5090], (13, 5)), 0)
            self.assertEqual((item.can_install, item.engine_build, item.alternative_build), (True, "cuda-13.3", ""))
            # Build not known (no runtime DLLs in the folder, no marker): not blocked, a note.
            shutil.rmtree(self.tmp / "engine")
            item = check_llama(self._engine(dlls=()), GpuFacts([RTX5090], (13, 5)), 0)
            self.assertTrue(item.ok)
            self.assertIn("build not known", item.detail)

        def test_install_fetches_the_chosen_pair(self):
            fetched = []

            def fake_download(url, dest, size, log, progress, cancel, sha256):
                fetched.append((dest.name, size, sha256))
                raise SetupError("stop here")
            with mock.patch.object(this, "download_file", side_effect=fake_download), \
                    mock.patch.object(this, "DOWNLOAD_DIR", self.tmp / "downloads"):
                for build, facts, expected in (
                        ("cuda-13.3", GpuFacts([]), LLAMA_ZIP_13_3),
                        ("cuda-12.4", GpuFacts([RTX5090], (12, 8)), LLAMA_ZIP_12_4),   # the explicit choice
                        ("", GpuFacts([RTX4070]), LLAMA_ZIP_12_4),
                        ("", GpuFacts([RTX5090], (13, 5)), LLAMA_ZIP_13_3)):
                    fetched.clear()
                    with mock.patch.object(this, "read_gpu_facts", return_value=facts):
                        with self.assertRaises(SetupError):
                            install_llama(lambda m: None, None, None, build)
                    self.assertEqual(fetched, [expected], build)
                # RTX 50 with an old driver and no explicit choice: refused before any download.
                fetched.clear()
                with mock.patch.object(this, "read_gpu_facts", return_value=GpuFacts([RTX5090], (12, 8))):
                    with self.assertRaises(SetupError) as ctx:
                        install_llama(lambda m: None, None, None)
                self.assertEqual(fetched, [])
                self.assertIn("Update the NVIDIA driver", str(ctx.exception))
                # Smart App Control on: refused before any download.
                with mock.patch.object(this, "smart_app_control_state", return_value=1):
                    with self.assertRaises(SetupError) as ctx:
                        install_llama(lambda m: None, None, None, "cuda-13.3")
                self.assertEqual(fetched, [])
                self.assertIn("Smart App Control is on", str(ctx.exception))
            # install_missing hands the item's build to the installer.
            got = []
            with mock.patch.dict(INSTALLERS, {"llama": lambda log, progress, cancel, build="": got.append(build)}):
                item = Item("llama", "engine", False, "", can_install=True, engine_build="cuda-12.4")
                self.assertEqual(install_missing([item], lambda m: None), [])
            self.assertEqual(got, ["cuda-12.4"])

        def test_incomplete_marker_is_retested_once_and_cleared(self):
            src = self.tmp / "unpacked"
            src.mkdir()
            names = ("llama-server.exe", "llama.dll", "ggml-cuda.dll",
                     "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
            for name in names:
                (src / name).write_bytes(b"1")
            folder = self.tmp / "engine"
            _place_engine_files([src / n for n in names], folder, None, "cuda-12.4")
            cfg = {"llm": {"server_binary": str(folder / "llama-server.exe")}}
            marker = folder / LLAMA_INCOMPLETE_MARKER
            # Still failing: the reason is shown, the install is offered, the marker stays.
            reason = engine_failure_text(3236495362, "")
            with mock.patch.object(this, "run_engine_version", return_value=(False, reason)) as ran:
                item = check_llama(cfg, GpuFacts([RTX4070]), 0)
            self.assertEqual(ran.call_count, 1)
            self.assertEqual((item.ok, item.can_install), (False, True))
            self.assertIn("0xC0E90002", item.detail)
            self.assertIn("Smart App Control", item.detail)
            self.assertTrue(marker.exists())
            # The cause is gone: OK, marker removed, no further run.
            with mock.patch.object(this, "run_engine_version", return_value=(True, "")) as ran:
                self.assertTrue(check_llama(cfg, GpuFacts([RTX4070]), 0).ok)
                self.assertEqual(ran.call_count, 1)
                self.assertFalse(marker.exists())
                self.assertTrue(check_llama(cfg, GpuFacts([RTX4070]), 0).ok)
                self.assertEqual(ran.call_count, 1)
            # A marker of 0.1.2 (no build line) with the files there: retested as well.
            marker.write_text("An install of the LLM engine into this folder did not finish.\n"
                              "Run the setup check again; it repeats the install.\n", encoding="utf-8")
            with mock.patch.object(this, "run_engine_version", return_value=(True, "")) as ran:
                self.assertTrue(check_llama(cfg, GpuFacts([RTX4070]), 0).ok)
                self.assertEqual(ran.call_count, 1)
            # A marker of 0.1.2 where the CUDA runtime DLLs (moved last by 0.1.2) are missing:
            # interrupted, never run, the marker stays.
            old_text = ("An install of the LLM engine into this folder did not finish.\n"
                        "Run the setup check again; it repeats the install.\n")
            for missing in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
                marker.write_text(old_text, encoding="utf-8")
                (folder / missing).unlink()
                item = check_llama(cfg, GpuFacts([RTX4070]), 0)
                self.assertEqual((item.ok, item.can_install), (False, True), missing)
                self.assertIn("interrupted", item.detail)
                self.assertTrue(marker.exists())
                (folder / missing).write_bytes(b"1")
            # An interrupted install of this version (not all files placed) is never run
            # (setUp's stand-in fails the test if it is).
            marker.write_text("An install ...\nbuild: cuda-12.4\n", encoding="utf-8")
            item = check_llama(cfg, GpuFacts([RTX4070]), 0)
            self.assertEqual((item.ok, item.can_install), (False, True))
            self.assertIn("interrupted", item.detail)

        def test_exit_codes_are_explained(self):
            self.assertEqual(describe_exit_code(3236495362),
                             "exit code 3236495362 (0xC0E90002): Windows (Smart App Control / Code Integrity) "
                             "blocked the engine")
            self.assertEqual(describe_exit_code(-1058471934), describe_exit_code(3236495362))
            self.assertEqual(describe_exit_code(0xC0000135), "exit code 3221225781 (0xC0000135): a DLL is missing")
            self.assertEqual(describe_exit_code(-1073741515), describe_exit_code(0xC0000135))
            self.assertEqual(describe_exit_code(0xC0000409),
                             "exit code 3221226505 (0xC0000409): the engine stopped itself after an error")
            self.assertEqual(describe_exit_code(1), "exit code 1 (0x00000001)")
            self.assertEqual(describe_exit_code(-1), "exit code 4294967295 (0xFFFFFFFF)")
            self.assertTrue(engine_failure_text(1, "").endswith("It printed nothing."))
            self.assertIn("It printed: error: x", engine_failure_text(1, "  error: x\n"))
            self.assertLessEqual(len(engine_failure_text(1, "y" * 5000)), 400)
            self.assertIn("did not answer '--version' within 60 s", engine_timeout_text(60))
            self.assertIn("Smart App Control", engine_timeout_text(60))
            # The program that stopped is named (default: the engine).
            self.assertEqual(describe_exit_code(0xC0000409, "the web server"),
                             "exit code 3221226505 (0xC0000409): the web server stopped itself after an error")
            self.assertTrue(describe_exit_code(-1058471934, "ytc.exe").endswith("blocked ytc.exe"))
            self.assertNotIn("engine", describe_exit_code(0xC0000409, "the web server"))
            # A missing DLL names the Visual C++ runtime; other codes do not.
            self.assertIn(VC_REDIST_URL, engine_failure_text(-1073741515, ""))
            self.assertIn("a DLL is missing (usually the Microsoft Visual C++", engine_failure_text(0xC0000135, ""))
            self.assertNotIn("Visual C++", engine_failure_text(0xC0E90002, ""))
            self.assertEqual(dll_missing_hint(1), "")

        def test_missing_vc_runtime_is_found_and_blocks_the_engine_install(self):
            names = list(VC_RUNTIME_DLLS)
            a, b = self.tmp / "a", self.tmp / "b"
            a.mkdir()
            b.mkdir()
            self.assertEqual(real_vc_runtime_missing([a, b, self.tmp / "gone"]), names)
            for name in names[:2]:
                (a / name).write_bytes(b"1")
            for name in names[2:]:
                (b / name).write_bytes(b"1")
            self.assertEqual(real_vc_runtime_missing([a]), names[2:])
            self.assertEqual(real_vc_runtime_missing([self.tmp / "gone", a, b]), [])
            self.assertIsNone(check_vc_runtime([]))
            item = check_vc_runtime(["vcomp140.dll"])
            self.assertEqual((item.key, item.ok, item.required, item.can_install), ("vcredist", False, True, False))
            self.assertIn("vcomp140.dll", item.detail)
            self.assertEqual(item.help_url, VC_REDIST_URL)
            # The default folders: the engine folder and System32 come first.
            folders = _vc_runtime_folders()
            self.assertEqual(folders[0], LLAMA_DIR)
            self.assertIn(folders[1].name.lower(), ("system32", "sysnative"))
            # No engine yet / failing start test: the install is not offered while the runtime
            # is missing (it would fail with 'a DLL is missing' again and again).
            cfg = {}
            item = check_llama(cfg, GpuFacts([RTX4070]), 0, ["vcomp140.dll"])
            self.assertEqual((item.ok, item.can_install), (False, False))
            self.assertIn("Visual C++", item.detail)
            self.assertEqual(item.help_url, VC_REDIST_URL)
            self.assertTrue(check_llama(cfg, GpuFacts([RTX4070]), 0, []).can_install)
            with mock.patch.object(this, "vc_runtime_missing", return_value=["msvcp140.dll"]), \
                    mock.patch.object(this, "download_file",
                                      side_effect=AssertionError("must not download")):
                with self.assertRaises(SetupError) as raised:
                    install_llama(lambda _line: None, None, None)
                self.assertIn("msvcp140.dll", str(raised.exception))
                self.assertIn(VC_REDIST_URL, str(raised.exception))

        def test_ytc_version_runs_without_error_windows_and_explains_the_code(self):
            exe = self.tmp / "ytc.exe"
            entered = []

            class Boxes:
                def __enter__(self):
                    entered.append(True)

                def __exit__(self, *exc):
                    return False
            with mock.patch.object(this, "_NoErrorBoxes", Boxes), \
                    mock.patch.object(this, "_run", return_value=(-1073741515, "")):
                found, problem = ytc_version(exe)
            self.assertEqual(entered, [True])
            self.assertIsNone(found)
            self.assertIn("(0xC0000135): a DLL is missing", problem)
            self.assertIn(VC_REDIST_URL, problem)

        def test_python_probe_child_writes_utf8(self):
            done = subprocess.CompletedProcess([], 0, "C:\\Users\\José\\x\n", "")
            with mock.patch.object(subprocess, "run", return_value=done) as run, \
                    mock.patch.object(this, "python_exe", return_value=Path("python.exe")):
                self.assertEqual(_run_python_probe("print(1)", []), "C:\\Users\\José\\x\n")
            env = run.call_args[1]["env"]
            self.assertEqual((env["PYTHONIOENCODING"], env["PYTHONUTF8"]), ("utf-8", "1"))
            self.assertEqual(run.call_args[1]["encoding"], "utf-8")
            self.assertIn("PATH", {k.upper() for k in env})              # the rest of the environment is kept

        def test_stream_cancel_works_while_the_child_is_silent(self):
            release = threading.Event()
            self.addCleanup(release.set)

            class SilentOut:
                def __iter__(self):
                    yield "Downloading nvidia_cudnn_cu12 (700 MB)\n"
                    release.wait(30)          # a long download: no further line

            class FakeProc:
                pid = 4242
                stdout = SilentOut()

                def wait(self):
                    return 0
            cancel = threading.Event()
            logged = []

            def log(line):
                logged.append(line)
                cancel.set()                  # the user presses Cancel after the first line
            with mock.patch.object(subprocess, "Popen", return_value=FakeProc()), \
                    mock.patch.object(subprocess, "run") as run:
                started = time.monotonic()
                with self.assertRaises(SetupCancelled):
                    _stream(["pip"], log, cancel)
                self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(logged, ["Downloading nvidia_cudnn_cu12 (700 MB)"])
            self.assertEqual(run.call_args[0][0], ["taskkill", "/PID", "4242", "/T", "/F"])

        def test_stream_passes_lines_and_returns_the_exit_code(self):
            class FakeProc:
                pid = 1
                stdout = iter(["one\n", "  \n", "secret\n", "two\r\n"])

                def wait(self):
                    return 3
            logged = []
            with mock.patch.object(subprocess, "Popen", return_value=FakeProc()), \
                    mock.patch.object(subprocess, "run", side_effect=AssertionError("no taskkill")):
                code = _stream(["x"], logged.append, threading.Event(), hide=lambda ln: "secret" in ln)
            self.assertEqual((code, logged), (3, ["one", "two"]))

        def test_engine_version_run_reports_the_reason(self):
            exe = self.tmp / "llama-server.exe"
            done = subprocess.CompletedProcess

            def answers(*items):
                return mock.patch.object(subprocess, "run", side_effect=list(items))
            with answers(done([], 0, "version: 1 (d7b2896)", "")):
                self.assertEqual(real_run_engine_version(exe), (True, ""))
            with answers(done([], 3236495362, "", "")):
                ok, reason = real_run_engine_version(exe)
                self.assertFalse(ok)
                self.assertIn("0xC0E90002", reason)
                self.assertIn("printed nothing", reason)
            with answers(subprocess.TimeoutExpired([str(exe)], 60)) as run:
                ok, reason = real_run_engine_version(exe)
                self.assertFalse(ok)
                self.assertIn("within 60 s", reason)
                self.assertEqual(run.call_args[0][0], [str(exe), "--version"])
                self.assertEqual(run.call_args[1]["timeout"], ENGINE_VERSION_TIMEOUT_S)
            blocked = OSError("An Application Control policy has blocked this file")
            blocked.winerror = WINERROR_APP_CONTROL_BLOCKED
            with answers(blocked):
                self.assertIn("Smart App Control", real_run_engine_version(exe)[1])
            with answers(FileNotFoundError(2, "not found")):
                self.assertIn("could not start", real_run_engine_version(exe)[1])

        @unittest.skipUnless(sys.platform == "win32", "Windows only")
        def test_error_mode_is_restored(self):
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32")
            before = kernel32.GetErrorMode()
            wanted = SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX
            with _NoErrorBoxes():
                inside = kernel32.GetErrorMode()
            self.assertEqual(inside & wanted, wanted)
            self.assertEqual(kernel32.GetErrorMode(), before)
            # Overlapping users (two threads starting programs): the first one
            # leaving must not restore the mode while the second is still inside.
            first, second = _NoErrorBoxes(), _NoErrorBoxes()
            first.__enter__()
            second.__enter__()
            first.__exit__(None, None, None)
            self.assertEqual(kernel32.GetErrorMode() & wanted, wanted)
            second.__exit__(None, None, None)
            self.assertEqual(kernel32.GetErrorMode(), before)

        def test_last_log_lines_or_the_empty_sentence(self):
            log = self.tmp / "llama-server.log"
            empty = "llama-server.log is empty: the engine stopped before it wrote anything"
            self.assertEqual(last_log_lines(log), empty)               # missing
            log.write_text("", encoding="utf-8")
            self.assertEqual(last_log_lines(log), empty)
            log.write_text("\n  \r\n", encoding="utf-8")
            self.assertEqual(last_log_lines(log), empty)
            log.write_text("".join(f"line {n}\n" for n in range(1, 31)), encoding="utf-8")
            self.assertEqual(last_log_lines(log).splitlines(), [f"line {n}" for n in range(16, 31)])
            self.assertEqual(last_log_lines(log, 2), "line 29\nline 30")
            log.write_bytes(b"ok \xff\xfe broken utf-8\n")
            self.assertIn("broken utf-8", last_log_lines(log))

        def test_smart_app_control_value_mapping(self):
            class FakeKey:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            def fake_winreg(value=None, open_error=None, query_error=None):
                reg = mock.Mock()
                reg.HKEY_LOCAL_MACHINE, reg.KEY_READ = "HKLM", 0x20019
                reg.OpenKey.side_effect = open_error if open_error else (lambda *a: FakeKey())
                reg.QueryValueEx.side_effect = query_error if query_error else (lambda key, name: (value, 4))
                return reg
            reg = fake_winreg(1)
            self.assertEqual(real_smart_app_control_state(reg), 1)
            self.assertEqual(reg.OpenKey.call_args[0][:2], ("HKLM", SAC_KEY))
            self.assertEqual(reg.OpenKey.call_args[0][3], reg.KEY_READ)          # read-only
            self.assertEqual(reg.QueryValueEx.call_args[0][1], "VerifiedAndReputablePolicyState")
            self.assertEqual(real_smart_app_control_state(fake_winreg(2)), 2)
            self.assertEqual(real_smart_app_control_state(fake_winreg(0)), 0)
            for reg in (fake_winreg(query_error=FileNotFoundError(2, "missing")),
                        fake_winreg(open_error=PermissionError(5, "denied")),
                        fake_winreg("garbage"), fake_winreg(7), fake_winreg(None)):
                self.assertEqual(real_smart_app_control_state(reg), 0)
            item = check_smart_app_control(1)
            self.assertEqual((item.ok, item.required, item.can_install), (False, True, False))
            self.assertEqual(item.detail, SAC_ON_MESSAGE)
            self.assertIn("not code-signed", item.detail)
            item = check_smart_app_control(2)
            self.assertEqual((item.ok, item.required), (True, False))
            self.assertIn("evaluation mode", item.detail)
            self.assertIsNone(check_smart_app_control(0))
            # While it is on, the engine download is not offered.
            item = check_llama({"llm": {"server_binary": str(self.tmp / "none" / "llama-server.exe")}},
                               GpuFacts([RTX5090], (12, 8)), 1)
            self.assertEqual((item.can_install, item.alternative_build), (False, ""))
            self.assertIn("waits until Smart App Control is off", item.detail)

        def test_harmless_huggingface_warnings_are_hidden(self):
            for line in ("Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. "
                         "Falling back to regular HTTP download.",
                         "C:\\py\\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks "
                         "by default to efficiently store duplicated files but your machine does not support them",
                         "To support symlinks on Windows, you either need to activate Developer Mode"):
                self.assertTrue(is_harmless_hf_warning(line), line)
            for line in ("Error while downloading from https://huggingface.co/x: Read timed out.",
                         "Trying to resume download...", "OSError: disk full",
                         # Real failures of the Xet transport name hf_xet too: shown.
                         "ImportError: DLL load failed while importing hf_xet: The specified module could not be found.",
                         "    from hf_xet import PyXetDownloadInfo, download_files",
                         "hf_xet.XetObjectNotFoundError: object not found",
                         "ValueError: To use optimized download using Xet storage, you need to install the "
                         "hf_xet package. Try `pip install \"huggingface_hub[hf_xet]\"` or `pip install hf_xet`."):
                self.assertFalse(is_harmless_hf_warning(line), line)

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
