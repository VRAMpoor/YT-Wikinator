#!/usr/bin/env python3
"""
YTC Control Center - desktop window for YT-Wikinator.

Starts and stops the LLM engine and the web server, manages the channel list
and runs crawls, single videos, disqualify / restore and library re-analysis
through ytc.exe.

Everything lives inside the app folder (the folder of this file): the engine
in bin/llama, the model in models/, ytc.exe in target/release or next to this
file, the database in data/ytc.db. No other folder is searched, so the whole
folder can be copied to another PC.
"""

import os
import sys
import json
import time
import queue
import socket
import sqlite3
import threading
import traceback
import subprocess
import webbrowser
from pathlib import Path
import urllib.request
import urllib.error

# --- Paths & Directory Discovery ---
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
LOG_DIR = SCRIPT_DIR / "logs"

import ytc_setup  # noqa: E402  (lives next to this file)
from ytc_setup import find_llama_server, find_ytc_binary, find_model  # noqa: E402

APP_TITLE = "YTC Mission Control Center (CUDA Accelerated)"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
# A second launch with --autostart leaves this file for the running window.
AUTOSTART_REQUEST = LOG_DIR / "autostart.request"
LLM_READY_TIMEOUT_S = 300
YTC_READY_TIMEOUT_S = 30


def setup_stdio():
    """Under pythonw.exe there is no console (sys.stdout/stderr are None).
    Send prints and tracebacks to logs/control_center.log so failures are visible."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None or sys.stderr is None:
        f = open(LOG_DIR / "control_center.log", "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f


_log_file_lock = threading.Lock()


def append_log_file(text):
    """Append to logs/control_center.log; never raises."""
    try:
        with _log_file_lock:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_DIR / "control_center.log", "a", encoding="utf-8") as f:
                f.write(text)
    except OSError:
        pass


try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    # Under pythonw there is no console, so say it in a native message box.
    append_log_file(traceback.format_exc())
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None,
            "This Python has no Tk support (tkinter).\n"
            "Install Python 3.10+ from python.org with the 'tcl/tk and IDLE' option.",
            "YTC Control Center", 0x10)
    sys.exit(1)


def child_env(context_tokens=None, base=None):
    """Environment for every child process: project root + this Python first on
    PATH, UTF-8 output, and the same interpreter for ytc.exe's Python helpers.

    context_tokens is llm.context_size from config.json, the context window
    the LLM engine is started with. ytc.exe gets it as YTC_LLM_CONTEXT_TOKENS
    and cuts long transcripts into sections that fit that window. It is left
    alone when YTC_LLM_URL points ytc.exe at another engine, whose context
    window config.json does not describe."""
    env = dict(os.environ if base is None else base)
    py_dir = Path(sys.executable).resolve().parent
    env["PATH"] = os.pathsep.join(
        [str(SCRIPT_DIR), str(py_dir), str(py_dir / "Scripts"), env.get("PATH", "")]
    )
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"
    python_exe = py_dir / "python.exe"
    if python_exe.exists() and "YTC_PYTHON" not in env:
        env["YTC_PYTHON"] = str(python_exe)
    if context_tokens and not env.get("YTC_LLM_URL"):
        env["YTC_LLM_CONTEXT_TOKENS"] = str(int(context_tokens))
    return env

DEFAULT_CONFIG = {
    "llm": {
        "server_binary": "bin/llama/llama-server.exe",
        "model_path": "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "port": 8080,
        "gpu_layers": 99,
        "context_size": 8192,
        "reasoning": False
    },
    "ytc": {
        "binary": "target/release/ytc.exe",
        "port": 8088,
        "db_path": "data/ytc.db",
        "wiki_dir": "wiki"
    },
    "crawler": {
        "search_depth": 3,
        "delay_seconds": 10,
        "max_slop_score": 45,
        "watch_interval_minutes": 60
    }
}


def _merge_defaults(defaults, loaded):
    out = {}
    for key, dval in defaults.items():
        lval = loaded.get(key) if isinstance(loaded, dict) else None
        if isinstance(dval, dict):
            out[key] = _merge_defaults(dval, lval if isinstance(lval, dict) else {})
        else:
            out[key] = dval if lval is None else lval
    if isinstance(loaded, dict):
        for key, lval in loaded.items():
            if key not in out:
                out[key] = lval
    return out


def load_config():
    """Read config.json; any missing section or key falls back to the default."""
    loaded = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as e:
            print(f"config.json could not be read ({e}); using defaults")
    return _merge_defaults(DEFAULT_CONFIG, loaded)


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def resolve_db_path(text, root=None):
    """The one database rule of the Control Center: the given path (the
    database box / ytc.db_path in config.json; relative means inside the app
    folder), else <app folder>/data/ytc.db. No other place is tried; a file
    that does not exist yet is created there by ytc.exe."""
    root = SCRIPT_DIR if root is None else Path(root)
    text = (text or "").strip().strip('"')
    if not text:
        return root / "data" / "ytc.db"
    path = Path(text)
    return path if path.is_absolute() else root / path


def config_path_text(path, root=None):
    """How a path is written into config.json: relative with forward slashes
    when it lies inside the app folder (the folder can then be moved or
    copied to another PC), otherwise absolute."""
    root = SCRIPT_DIR if root is None else Path(root)
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


# The keep decision is stored per video when it is analysed; ytc does not
# re-decide stored videos when the cutoff changes.
SLOP_CHANGE_NOTE = ("A changed Max Slop Score applies to videos analysed from now on. Videos already in "
                    "the library keep their decision until they are re-analysed (ytc reanalyze --all).")


def channel_overview(conn):
    """One row per channel for the channel list:
    (id, title, status, rolling slop, kept, below cutoff, off-topic, not
    enough information, disqualified). Every analysed video is in exactly one
    of the five groups, with the same precedence ytc uses: disqualified by
    hand, off-topic, not enough information, then the quality cutoff. A
    video without a recorded topic verdict or information status is rejected
    by neither."""
    query = """
        SELECT c.id, c.title, c.status, c.rolling_slop_score,
               COALESCE(SUM(j.cat = 'kept'), 0),
               COALESCE(SUM(j.cat = 'below'), 0),
               COALESCE(SUM(j.cat = 'off_topic'), 0),
               COALESCE(SUM(j.cat = 'insufficient'), 0),
               COALESCE(SUM(j.cat = 'excluded'), 0)
        FROM channels c
        LEFT JOIN (
            SELECT v.channel_id AS channel_id,
                   CASE WHEN i.excluded_manual = 1 THEN 'excluded'
                        WHEN i.topic_verdict <> 'ok' THEN 'off_topic'
                        WHEN i.info_status = 'insufficient' THEN 'insufficient'
                        WHEN i.is_worth_keeping = 1 THEN 'kept'
                        ELSE 'below' END AS cat
            FROM insights i JOIN videos v ON v.id = i.video_id
        ) j ON j.channel_id = c.id
        GROUP BY c.id
        ORDER BY c.status ASC, c.title ASC
    """
    return conn.execute(query).fetchall()


def channel_row_texts(row):
    """Display texts of one channel_overview row: (title, status, slop,
    'kept / quality-judged', 'below · off-topic · no info · removed', id)."""
    chan_id, title, status, slop, kept, below, off_topic, insufficient, excluded = row
    slop_display = f"{slop:.0f}/100" if slop is not None else "Pending"
    status_display = {"active": "🟢 Active", "high_signal": "⭐ High Signal",
                      "muted": "⚪ Muted"}.get(status, str(status))
    return (title, status_display, slop_display, f"{kept} / {kept + below}",
            f"{below} · {off_topic} · {insufficient} · {excluded}", chan_id)


# What 'Re-analyse library...' can run: (selection of 'ytc reanalyze', what it selects).
REANALYZE_SELECTIONS = (
    ("--analyzer heuristic", "Videos that only have the heuristic estimate, which ytc stores while no LLM "
                             "answers. Use it once the LLM engine is set up."),
    ("--all", "Every analysed video, for example after changing the categories or the model."),
)

REANALYZE_EXPLANATION = (
    "This runs 'ytc reanalyze' over the stored transcripts. No request goes to YouTube.\n\n"
    "For every selected video the LLM reads the WHOLE transcript again and judges the topic (anything "
    "is fine except fringe, nonsense and politics) and, separately, the quality (slop score). A stored "
    "result is replaced only by a valid new answer.\n\n"
    "How long it takes: one LLM request for a short video; a long video is read in sections, one "
    "request per section plus a final one. That can be several minutes per long video and many hours "
    "for a large library. ytc prints the number of videos and LLM requests before it starts.\n\n"
    "You can stop it at any time with the same button; videos finished until then keep their new "
    "result. The videos are taken in the order of their last analysis, the one analysed longest ago "
    "first, so a later run continues with the rest. Afterwards use 'Sync Wiki Files'."
)


def reanalyze_args(selection, limit=0):
    """Arguments of 'ytc reanalyze' for one of REANALYZE_SELECTIONS; limit 0 = all."""
    if selection not in [option for option, _ in REANALYZE_SELECTIONS]:
        raise ValueError(f"unknown selection: {selection}")
    args = ["reanalyze"] + selection.split()
    if int(limit) > 0:
        args += ["--limit", str(int(limit))]
    return args


def exclude_args(video, reason=""):
    """Arguments of 'ytc exclude'. The video comes after '--' so that an ID
    with a leading hyphen is not read as an option."""
    args = ["exclude"]
    if reason.strip():
        args += ["--reason", reason.strip()]
    return args + ["--", video.strip()]


def is_port_open(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def check_http_alive(url, timeout=2.0):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "YTC-ControlCenter"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_quiet(cmd, timeout=20):
    """Run a short helper command with no console window; returns stdout text."""
    try:
        res = subprocess.run(
            cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout,
            encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
        )
        return res.stdout or ""
    except Exception:
        return ""


def pids_listening_on(port):
    """PIDs that own a listening TCP socket on this port. The state column of
    netstat is translated on non-English Windows, so a listener is recognised
    by its remote address (0.0.0.0:0 or [::]:0) instead."""
    pids = set()
    for line in run_quiet(["netstat", "-ano", "-p", "TCP"]).splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" \
                and parts[1].endswith(f":{port}") \
                and parts[2] in ("0.0.0.0:0", "[::]:0", "*:*") \
                and parts[-1].isdigit():
            pids.add(int(parts[-1]))
    pids.discard(0)
    return pids


def process_image_name(pid):
    out = run_quiet(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
    for line in out.splitlines():
        if line.startswith('"'):
            return line.split('","')[0].strip('"').lower()
    return ""


def kill_process_tree(pid):
    """Stop one process and its children (e.g. python/yt-dlp helpers of ytc.exe)."""
    run_quiet(["taskkill", "/PID", str(pid), "/T", "/F"])


def tail_text(path, max_lines=12):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]
        return lines[-max_lines:]
    except OSError:
        return []


def acquire_single_instance():
    """True if this is the only Control Center. Otherwise the existing window
    is brought to the front and False is returned."""
    if sys.platform != "win32":
        return True
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # The handle is kept for the life of the process on purpose.
    acquire_single_instance.handle = kernel32.CreateMutexW(None, False, "YTC_ControlCenter_SingleInstance")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, APP_TITLE)
        if hwnd:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
        return False
    return True


def build_ytc_cmd(ytc_bin, db_path, args, llm_port, max_slop, wiki_dir, env=None):
    """Command line for ytc.exe. Global options go before the subcommand.

    The Control Center's own settings win over the environment: --db,
    --max-slop and --wiki-dir are always passed, so YTC_DB, YTC_MAX_SLOP and
    YTC_WIKI_DIR have no effect on what is started from this window (README
    says the same). YTC_LLM_URL is the one exception and is left to ytc.exe."""
    env = os.environ if env is None else env
    cmd = [str(ytc_bin), "--db", str(db_path)]
    if llm_port != 8080 and not env.get("YTC_LLM_URL"):
        cmd += ["--llm-url", f"http://127.0.0.1:{llm_port}/v1"]
    cmd += ["--max-slop", str(max_slop), "--wiki-dir", str(wiki_dir)]
    return cmd + [str(a) for a in args]


class YTCControlApp:
    def __init__(self, root, autostart=False):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1080x820")
        self.root.minsize(960, 740)

        self.cfg = load_config()

        # Process Handles
        self.llama_proc = None
        self.ytc_proc = None
        self.is_monitoring = True
        self.llm_state = "checking"   # checking | stopped | loading | online
        self.ytc_state = "checking"   # checking | stopped | blocked | online
        self._llm_start_lock = threading.Lock()
        self._ytc_start_lock = threading.Lock()
        # Bumped by every Stop; a start that was overtaken by a Stop cancels itself.
        self._llm_stop_gen = 0
        self._ytc_stop_gen = 0
        # Running crawl / ingest child processes (see _run_stream).
        self._jobs = set()
        self._jobs_lock = threading.Lock()
        # Plain copy of the Max Slop spinbox: worker threads must not read Tk variables.
        try:
            self._max_slop = max(0, min(100, int(self.cfg["crawler"].get("max_slop_score", 45))))
        except (TypeError, ValueError):
            self._max_slop = 45
        self.watch_proc = None
        # True from the click on 'Start Auto-Crawl' until 'ytc watch' has ended.
        # Main thread only. Without it a second click before the process exists
        # would start a second loop that the button could not stop any more.
        self._watch_busy = False
        # The same pair for 'Re-analyse library...' ('ytc reanalyze').
        self.reanalyze_proc = None
        self._reanalyze_busy = False
        self.reanalyze_dialog = None
        # ytc.exe path -> (file time, version it printed). Only answers of a
        # ytc.exe that printed its version are stored here.
        self._ytc_version_cache = {}
        self._ytc_version_lock = threading.Lock()
        self._llm_owner_checked = 0.0
        self._llm_owner_is_llama = False
        try:
            AUTOSTART_REQUEST.unlink()
        except OSError:
            pass

        # Worker threads never touch Tk directly. They put callables on this
        # queue and the main thread runs them (see _pump_ui_queue).
        self._ui_queue = queue.Queue()

        # Styles & Theme
        self.setup_theme()

        # Build UI Structure
        self.build_ui()

        # Errors inside button handlers and worker threads go to the log pane
        # and logs/control_center.log instead of vanishing under pythonw.
        self.root.report_callback_exception = self._on_tk_error
        threading.excepthook = self._on_thread_error
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._pump_ui_queue()

        # Start periodic status checker thread
        self.status_thread = threading.Thread(target=self.status_loop, daemon=True)
        self.status_thread.start()

        # Load channels into listview
        self.refresh_channels_list()

        # First thing after the window is up: is everything needed installed?
        self.setup_dialog = None
        self.root.after(400, lambda: self.run_setup_check(then_start=autostart))

    # --- Thread-safe UI plumbing ---
    def ui(self, func, *args):
        """Schedule func(*args) on the Tk main thread. Safe from any thread."""
        self._ui_queue.put((func, args))

    def _pump_ui_queue(self):
        try:
            for _ in range(200):
                func, args = self._ui_queue.get_nowait()
                try:
                    func(*args)
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        if not self.is_monitoring:
            return
        if AUTOSTART_REQUEST.exists():
            try:
                AUTOSTART_REQUEST.unlink()
            except OSError:
                pass
            self.start_all_services()
        self.root.after(80, self._pump_ui_queue)

    def _on_tk_error(self, exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        self.log(f"❌ Internal error: {exc_type.__name__}: {exc} (details in logs\\control_center.log)")

    def _on_thread_error(self, args):
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
        self.log(f"❌ Background task failed: {args.exc_type.__name__}: {args.exc_value}")

    def _int(self, var, default, lo, hi):
        """Read a spinbox value; an empty or mistyped field falls back to default."""
        try:
            value = int(var.get())
        except (tk.TclError, ValueError):
            value = default
            var.set(default)
        return max(lo, min(hi, value))

    def setup_theme(self):
        self.bg_dark = "#0d1117"
        self.bg_card = "#161b22"
        self.bg_card_active = "#21262d"
        self.border = "#30363d"
        self.fg_bright = "#f0f6fc"
        self.fg_text = "#c9d1d9"
        self.fg_dim = "#8b949e"
        self.accent_blue = "#58a6ff"
        self.accent_green = "#3fb950"
        self.accent_red = "#f85149"
        self.accent_orange = "#d29922"
        self.accent_purple = "#bc8cff"

        self.root.configure(bg=self.bg_dark)

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=self.bg_dark, foreground=self.fg_text, font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=self.bg_card, relief="solid", borderwidth=1)
        style.configure("Treeview",
                        background=self.bg_card,
                        foreground=self.fg_bright,
                        fieldbackground=self.bg_card,
                        rowheight=26,
                        bordercolor=self.border)
        style.map("Treeview", background=[("selected", "#1f6feb")], foreground=[("selected", "#ffffff")])
        style.configure("Treeview.Heading",
                        background=self.bg_card_active,
                        foreground=self.fg_bright,
                        font=("Segoe UI", 9, "bold"),
                        relief="flat")
        style.map("Treeview.Heading", background=[("active", "#30363d")])

    def build_ui(self):
        # Master Container with 2 primary panes: Top (Dashboard & Services), Bottom (Channel Manager & Logs)
        main_frame = tk.Frame(self.root, bg=self.bg_dark, padx=12, pady=10)
        main_frame.pack(fill="both", expand=True)

        # Header bar
        header = tk.Frame(main_frame, bg=self.bg_dark)
        header.pack(fill="x", pady=(0, 10))

        title_lbl = tk.Label(header, text="🌌 YTC MISSION CONTROL CENTER",
                             font=("Segoe UI", 14, "bold"), fg=self.fg_bright, bg=self.bg_dark)
        title_lbl.pack(side="left")

        sub_lbl = tk.Label(header, text="Autonomous YouTube Knowledge Synthesizer • RTX GPU Accelerated",
                           font=("Segoe UI", 9), fg=self.fg_dim, bg=self.bg_dark)
        sub_lbl.pack(side="left", padx=(12, 0), pady=(3, 0))

        # Top Right Master Actions: [START ALL] [SHUTDOWN ALL] [OPEN BROWSER]
        btn_box = tk.Frame(header, bg=self.bg_dark)
        btn_box.pack(side="right")

        self.btn_start_all = tk.Button(btn_box, text="⚡ START ALL SERVICES", font=("Segoe UI", 9, "bold"),
                                       bg="#238636", fg="#ffffff", activebackground="#2ea043", activeforeground="#ffffff",
                                       relief="flat", padx=12, pady=5, cursor="hand2", command=self.start_all_services)
        self.btn_start_all.pack(side="left", padx=4)

        self.btn_stop_all = tk.Button(btn_box, text="🛑 SHUT DOWN ALL", font=("Segoe UI", 9, "bold"),
                                      bg="#b62324", fg="#ffffff", activebackground="#da3633", activeforeground="#ffffff",
                                      relief="flat", padx=12, pady=5, cursor="hand2", command=self.stop_all_services)
        self.btn_stop_all.pack(side="left", padx=4)

        self.btn_open_web = tk.Button(btn_box, text="🌐 OPEN WEB UI", font=("Segoe UI", 9, "bold"),
                                      bg="#1f6feb", fg="#ffffff", activebackground="#388bfd", activeforeground="#ffffff",
                                      relief="flat", padx=12, pady=5, cursor="hand2", command=self.open_web_ui)
        self.btn_open_web.pack(side="left", padx=4)

        self.btn_setup = tk.Button(btn_box, text="🧰 SETUP CHECK", font=("Segoe UI", 9, "bold"),
                                   bg="#21262d", fg=self.fg_bright, activebackground="#30363d", activeforeground="#ffffff",
                                   relief="flat", padx=12, pady=5, cursor="hand2",
                                   command=lambda: self.run_setup_check(always_show=True))
        self.btn_setup.pack(side="left", padx=4)

        # ---------------------------------------------------------------------
        # TOP ROW: Service Control & Parameters
        # ---------------------------------------------------------------------
        top_grid = tk.Frame(main_frame, bg=self.bg_dark)
        top_grid.pack(fill="x", pady=(0, 10))

        # Card 1: LLM Engine (llama-server). Port numbers are the configured ones.
        llm_port, ytc_port = self._ports()
        llm_card = tk.LabelFrame(top_grid, text=f" 🧠 LLM Inference Engine (Port {llm_port}) ",
                                 font=("Segoe UI", 9, "bold"), fg=self.accent_blue, bg=self.bg_card,
                                 padx=10, pady=8, bd=1, relief="solid")
        llm_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

        llm_status_row = tk.Frame(llm_card, bg=self.bg_card)
        llm_status_row.pack(fill="x", pady=(0, 4))
        tk.Label(llm_status_row, text="Status:", font=("Segoe UI", 9, "bold"), fg=self.fg_dim, bg=self.bg_card).pack(side="left")
        self.llm_status_lbl = tk.Label(llm_status_row, text="CHECKING...", font=("Segoe UI", 9, "bold"),
                                       fg=self.accent_orange, bg=self.bg_card)
        self.llm_status_lbl.pack(side="left", padx=(6, 0))

        llm_btn_row = tk.Frame(llm_card, bg=self.bg_card)
        llm_btn_row.pack(fill="x", pady=4)
        self.btn_llm_start = tk.Button(llm_btn_row, text="Start Engine", font=("Segoe UI", 8, "bold"),
                                       bg="#21262d", fg=self.accent_green, activebackground="#30363d",
                                       relief="groove", padx=8, pady=2, command=self.start_llm_server)
        self.btn_llm_start.pack(side="left", padx=(0, 4))

        self.btn_llm_stop = tk.Button(llm_btn_row, text="Stop Engine", font=("Segoe UI", 8, "bold"),
                                      bg="#21262d", fg=self.accent_red, activebackground="#30363d",
                                      relief="groove", padx=8, pady=2, command=self.stop_llm_server)
        self.btn_llm_stop.pack(side="left")

        # Model path selector
        model_row = tk.Frame(llm_card, bg=self.bg_card)
        model_row.pack(fill="x", pady=(6, 0))
        tk.Label(model_row, text="Model GGUF:", font=("Segoe UI", 8), fg=self.fg_dim, bg=self.bg_card).pack(side="left")
        self.model_entry = tk.Entry(model_row, font=("Consolas", 8), bg=self.bg_dark, fg=self.fg_bright,
                                    insertbackground=self.fg_bright, relief="flat")
        self.model_entry.insert(0, str(self.cfg["llm"].get("model_path", "")))
        self.model_entry.pack(side="left", fill="x", expand=True, padx=4)
        tk.Button(model_row, text="...", font=("Segoe UI", 8), bg="#21262d", fg=self.fg_bright,
                  relief="groove", padx=6, command=self.browse_model_file).pack(side="right")

        # Card 2: Web Server Engine (ytc serve)
        ytc_card = tk.LabelFrame(top_grid, text=f" 🌐 Web Wiki Engine (Port {ytc_port}) ",
                                 font=("Segoe UI", 9, "bold"), fg=self.accent_blue, bg=self.bg_card,
                                 padx=10, pady=8, bd=1, relief="solid")
        ytc_card.pack(side="left", fill="both", expand=True, padx=3)

        ytc_status_row = tk.Frame(ytc_card, bg=self.bg_card)
        ytc_status_row.pack(fill="x", pady=(0, 4))
        tk.Label(ytc_status_row, text="Status:", font=("Segoe UI", 9, "bold"), fg=self.fg_dim, bg=self.bg_card).pack(side="left")
        self.ytc_status_lbl = tk.Label(ytc_status_row, text="CHECKING...", font=("Segoe UI", 9, "bold"),
                                       fg=self.accent_orange, bg=self.bg_card)
        self.ytc_status_lbl.pack(side="left", padx=(6, 0))

        ytc_btn_row = tk.Frame(ytc_card, bg=self.bg_card)
        ytc_btn_row.pack(fill="x", pady=4)
        self.btn_ytc_start = tk.Button(ytc_btn_row, text="Start Server", font=("Segoe UI", 8, "bold"),
                                       bg="#21262d", fg=self.accent_green, activebackground="#30363d",
                                       relief="groove", padx=8, pady=2, command=self.start_ytc_server)
        self.btn_ytc_start.pack(side="left", padx=(0, 4))

        self.btn_ytc_stop = tk.Button(ytc_btn_row, text="Stop Server", font=("Segoe UI", 8, "bold"),
                                      bg="#21262d", fg=self.accent_red, activebackground="#30363d",
                                      relief="groove", padx=8, pady=2, command=self.stop_ytc_server)
        self.btn_ytc_stop.pack(side="left", padx=(0, 4))

        tk.Button(ytc_btn_row, text="Sync Wiki Files", font=("Segoe UI", 8),
                  bg="#21262d", fg=self.fg_bright, activebackground="#30363d",
                  relief="groove", padx=8, pady=2, command=self.sync_wiki_action).pack(side="left")

        # Database path row
        db_row = tk.Frame(ytc_card, bg=self.bg_card)
        db_row.pack(fill="x", pady=(6, 0))
        tk.Label(db_row, text="Database DB:", font=("Segoe UI", 8), fg=self.fg_dim, bg=self.bg_card).pack(side="left")
        self.db_entry = tk.Entry(db_row, font=("Consolas", 8), bg=self.bg_dark, fg=self.fg_bright,
                                 insertbackground=self.fg_bright, relief="flat")
        # The box shows the real file: config.json may contain a relative path.
        self.db_entry.insert(0, str(resolve_db_path(str(self.cfg["ytc"].get("db_path") or ""))))
        self.db_entry.pack(side="left", fill="x", expand=True, padx=4)

        # Card 3: Crawl Parameters & Depth Setting
        param_card = tk.LabelFrame(top_grid, text=" ⚙️ Search Depth & Ingestion Rules ",
                                   font=("Segoe UI", 9, "bold"), fg=self.accent_blue, bg=self.bg_card,
                                   padx=10, pady=8, bd=1, relief="solid")
        param_card.pack(side="left", fill="both", expand=True, padx=(6, 0))

        # Depth Slider & Spinbox
        depth_row = tk.Frame(param_card, bg=self.bg_card)
        depth_row.pack(fill="x", pady=(0, 3))
        tk.Label(depth_row, text="Search Depth (vids/channel):", font=("Segoe UI", 8, "bold"),
                 fg=self.fg_bright, bg=self.bg_card).pack(side="left")
        self.depth_var = tk.IntVar(value=int(self.cfg["crawler"].get("search_depth", 3)))
        self.depth_spin = tk.Spinbox(depth_row, from_=1, to=25, textvariable=self.depth_var,
                                     width=4, font=("Segoe UI", 8, "bold"), bg=self.bg_dark, fg=self.fg_bright,
                                     buttonbackground="#21262d")
        self.depth_spin.pack(side="right")

        # Delay
        delay_row = tk.Frame(param_card, bg=self.bg_card)
        delay_row.pack(fill="x", pady=2)
        tk.Label(delay_row, text="Pacing Delay (seconds):", font=("Segoe UI", 8),
                 fg=self.fg_dim, bg=self.bg_card).pack(side="left")
        # ytc never pauses less than 10 seconds in a crawl, so the box starts there.
        self.delay_var = tk.IntVar(value=max(10, int(self.cfg["crawler"].get("delay_seconds", 10))))
        self.delay_spin = tk.Spinbox(delay_row, from_=10, to=30, textvariable=self.delay_var,
                                     width=4, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.fg_bright,
                                     buttonbackground="#21262d")
        self.delay_spin.pack(side="right")

        # Keep cutoff: passed to ytc.exe as --max-slop (videos scoring above it
        # are not kept).
        slop_row = tk.Frame(param_card, bg=self.bg_card)
        slop_row.pack(fill="x", pady=2)
        tk.Label(slop_row, text="Max Slop Score (quality cutoff):", font=("Segoe UI", 8),
                 fg=self.fg_dim, bg=self.bg_card).pack(side="left")
        crawler_cfg = self.cfg["crawler"]
        self.slop_var = tk.IntVar(value=int(crawler_cfg.get("max_slop_score", 45)))
        self.slop_spin = tk.Spinbox(slop_row, from_=0, to=100, textvariable=self.slop_var,
                                    width=4, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.fg_bright,
                                    buttonbackground="#21262d")
        self.slop_spin.pack(side="right")

        # Automatic crawling: runs 'ytc watch' every N minutes until stopped.
        watch_row = tk.Frame(param_card, bg=self.bg_card)
        watch_row.pack(fill="x", pady=2)
        tk.Label(watch_row, text="Auto-crawl every (minutes):", font=("Segoe UI", 8),
                 fg=self.fg_dim, bg=self.bg_card).pack(side="left")
        self.watch_var = tk.IntVar(value=int(crawler_cfg.get("watch_interval_minutes", 60)))
        self.watch_spin = tk.Spinbox(watch_row, from_=5, to=1440, textvariable=self.watch_var,
                                     width=5, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.fg_bright,
                                     buttonbackground="#21262d")
        self.watch_spin.pack(side="right")

        # Save Settings Button
        save_btn_row = tk.Frame(param_card, bg=self.bg_card)
        save_btn_row.pack(fill="x", pady=(4, 0))
        tk.Button(save_btn_row, text="💾 Save Preferences", font=("Segoe UI", 8, "bold"),
                  bg="#21262d", fg=self.accent_blue, activebackground="#30363d",
                  relief="groove", padx=8, pady=2, command=self.save_preferences_action).pack(side="right")
        self.btn_watch = tk.Button(save_btn_row, text="▶ Start Auto-Crawl", font=("Segoe UI", 8, "bold"),
                                   bg="#21262d", fg=self.accent_green, activebackground="#30363d",
                                   relief="groove", padx=8, pady=2, command=self.toggle_watch_action)
        self.btn_watch.pack(side="left")

        # ---------------------------------------------------------------------
        # MIDDLE: Channel Management Table & Direct Ingest
        # ---------------------------------------------------------------------
        chan_frame = tk.LabelFrame(main_frame, text=" 🎙️ Tracked YouTube Channels & Feed Controller ",
                                   font=("Segoe UI", 9, "bold"), fg=self.accent_blue, bg=self.bg_card,
                                   padx=10, pady=8, bd=1, relief="solid")
        chan_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Channel Actions Toolbar: [Add Channel Input] [+ Add] [Toggle Mute] [Remove] [Crawl Selected] [Crawl All]
        toolbar = tk.Frame(chan_frame, bg=self.bg_card)
        toolbar.pack(fill="x", pady=(0, 6))

        tk.Label(toolbar, text="Add Channel (@handle, URL, or ID):", font=("Segoe UI", 8, "bold"),
                 fg=self.fg_dim, bg=self.bg_card).pack(side="left", padx=(0, 6))

        self.add_chan_entry = tk.Entry(toolbar, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.fg_bright,
                                       insertbackground=self.fg_bright, relief="solid", bd=1, width=32)
        self.add_chan_entry.pack(side="left", padx=(0, 6))
        self.add_chan_entry.bind("<Return>", lambda e: self.add_channel_action())

        tk.Button(toolbar, text="+ Add Channel", font=("Segoe UI", 8, "bold"),
                  bg="#1f6feb", fg="#ffffff", activebackground="#388bfd",
                  relief="flat", padx=10, pady=3, command=self.add_channel_action).pack(side="left", padx=(0, 8))

        tk.Button(toolbar, text="🔄 Toggle Mute", font=("Segoe UI", 8),
                  bg="#21262d", fg=self.accent_orange, activebackground="#30363d",
                  relief="groove", padx=8, pady=3, command=self.toggle_mute_action).pack(side="left", padx=2)

        tk.Button(toolbar, text="🗑️ Remove", font=("Segoe UI", 8),
                  bg="#21262d", fg=self.accent_red, activebackground="#30363d",
                  relief="groove", padx=8, pady=3, command=self.remove_channel_action).pack(side="left", padx=2)

        tk.Button(toolbar, text="⚡ Crawl Selected", font=("Segoe UI", 8, "bold"),
                  bg="#21262d", fg=self.accent_green, activebackground="#30363d",
                  relief="groove", padx=8, pady=3, command=self.crawl_selected_action).pack(side="left", padx=(8, 2))

        tk.Button(toolbar, text="⚡ Crawl All Channels", font=("Segoe UI", 8, "bold"),
                  bg="#21262d", fg=self.accent_purple, activebackground="#30363d",
                  relief="groove", padx=8, pady=3, command=self.crawl_all_action).pack(side="left", padx=2)

        tk.Button(toolbar, text="🔄 Refresh List", font=("Segoe UI", 8),
                  bg="#21262d", fg=self.fg_dim, activebackground="#30363d",
                  relief="groove", padx=6, pady=3, command=self.refresh_channels_list).pack(side="right")

        # Legend for the two count columns and the average.
        tk.Label(chan_frame, font=("Segoe UI", 8), fg=self.fg_dim, bg=self.bg_card, anchor="w", justify="left",
                 wraplength=1000,
                 text="Rolling slop = quality only (0 dense, 100 slop), average of the last 5 videos the LLM "
                      "judged for quality.  Not kept = below the quality cutoff · off-topic (fringe / nonsense / politics) · "
                      "not enough information · disqualified by hand; only the first of these counts in the average."
                 ).pack(fill="x", pady=(0, 4))

        # Treeview for channels
        columns = ("title", "status", "slop", "kept", "not_kept", "channel_id")
        self.chan_tree = ttk.Treeview(chan_frame, columns=columns, show="headings", height=8, selectmode="browse")

        self.chan_tree.heading("title", text="Creator / Channel Title", anchor="w")
        self.chan_tree.heading("status", text="Status", anchor="center")
        self.chan_tree.heading("slop", text="Rolling Slop (last 5)", anchor="center")
        self.chan_tree.heading("kept", text="Kept / Quality-judged", anchor="center")
        self.chan_tree.heading("not_kept", text="Not kept: cutoff · off-topic · no info · removed", anchor="center")
        self.chan_tree.heading("channel_id", text="Channel ID", anchor="w")

        self.chan_tree.column("title", width=250, anchor="w")
        self.chan_tree.column("status", width=90, anchor="center")
        self.chan_tree.column("slop", width=120, anchor="center")
        self.chan_tree.column("kept", width=130, anchor="center")
        self.chan_tree.column("not_kept", width=260, anchor="center")
        self.chan_tree.column("channel_id", width=190, anchor="w")

        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(chan_frame, orient="vertical", command=self.chan_tree.yview)
        self.chan_tree.configure(yscrollcommand=tree_scroll.set)

        self.chan_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        # ---------------------------------------------------------------------
        # BOTTOM ROW: Direct Single Ingest + Live Monospace Console Log
        # ---------------------------------------------------------------------
        bottom_frame = tk.Frame(main_frame, bg=self.bg_dark)
        bottom_frame.pack(fill="both", expand=True)

        # Single Video Ingest Bar
        ingest_bar = tk.Frame(bottom_frame, bg=self.bg_card, padx=8, pady=6, bd=1, relief="solid")
        ingest_bar.pack(fill="x", pady=(0, 6))

        tk.Label(ingest_bar, text="🎯 Direct Video Ingest (URL or ID):", font=("Segoe UI", 8, "bold"),
                 fg=self.accent_green, bg=self.bg_card).pack(side="left", padx=(0, 6))

        self.ingest_entry = tk.Entry(ingest_bar, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.fg_bright,
                                     insertbackground=self.fg_bright, relief="solid", bd=1, width=42)
        self.ingest_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.ingest_entry.bind("<Return>", lambda e: self.direct_ingest_action())

        tk.Button(ingest_bar, text="🚀 Analyze & Ingest Video", font=("Segoe UI", 8, "bold"),
                  bg="#238636", fg="#ffffff", activebackground="#2ea043",
                  relief="flat", padx=10, pady=2, command=self.direct_ingest_action).pack(side="right")

        # Library bar: take a video out of the wiki by hand (and undo it),
        # and re-analyse stored videos with the current rules.
        library_bar = tk.Frame(bottom_frame, bg=self.bg_card, padx=8, pady=6, bd=1, relief="solid")
        library_bar.pack(fill="x", pady=(0, 6))

        tk.Label(library_bar, text="🚫 Disqualify video (URL or ID):", font=("Segoe UI", 8, "bold"),
                 fg=self.accent_orange, bg=self.bg_card).pack(side="left", padx=(0, 6))
        self.exclude_entry = tk.Entry(library_bar, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.fg_bright,
                                      insertbackground=self.fg_bright, relief="solid", bd=1, width=26)
        self.exclude_entry.pack(side="left", padx=(0, 8))
        tk.Label(library_bar, text="Reason:", font=("Segoe UI", 8), fg=self.fg_dim,
                 bg=self.bg_card).pack(side="left", padx=(0, 4))
        self.exclude_reason_entry = tk.Entry(library_bar, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.fg_bright,
                                             insertbackground=self.fg_bright, relief="solid", bd=1, width=22)
        self.exclude_reason_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.exclude_reason_entry.bind("<Return>", lambda e: self.disqualify_video_action())

        tk.Button(library_bar, text="Disqualify", font=("Segoe UI", 8, "bold"),
                  bg="#21262d", fg=self.accent_red, activebackground="#30363d",
                  relief="groove", padx=8, pady=2, command=self.disqualify_video_action).pack(side="left", padx=2)
        tk.Button(library_bar, text="Restore", font=("Segoe UI", 8),
                  bg="#21262d", fg=self.accent_green, activebackground="#30363d",
                  relief="groove", padx=8, pady=2, command=self.restore_video_action).pack(side="left", padx=2)
        self.btn_reanalyze = tk.Button(library_bar, text="♻ Re-analyse library...", font=("Segoe UI", 8, "bold"),
                                       bg="#21262d", fg=self.accent_purple, activebackground="#30363d",
                                       relief="groove", padx=8, pady=2, command=self.reanalyze_library_action)
        self.btn_reanalyze.pack(side="right", padx=(8, 0))

        # Live Console Text
        console_box = tk.LabelFrame(bottom_frame, text=" 📜 Live Activity & System Output Console ",
                                    font=("Segoe UI", 8, "bold"), fg=self.fg_dim, bg=self.bg_card,
                                    padx=6, pady=4, bd=1, relief="solid")
        console_box.pack(fill="both", expand=True)

        self.log_text = tk.Text(console_box, bg="#05070a", fg="#7ee787", insertbackground="#7ee787",
                                font=("Consolas", 9), height=7, wrap="word", relief="flat")
        log_scroll = ttk.Scrollbar(console_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self.log("🚀 YTC Mission Control GUI initialized.")
        self.log(f"📁 Root directory: {SCRIPT_DIR}")

    def log(self, message):
        """Write one line to the log pane and to logs/control_center.log.
        Safe to call from any thread."""
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        append_log_file(formatted)
        self.ui(self._append_log, formatted)

    def _append_log(self, text):
        self.log_text.insert("end", text)
        # Keep the pane bounded so a long crawl cannot grow it without limit.
        if int(self.log_text.index("end-1c").split(".")[0]) > 3000:
            self.log_text.delete("1.0", "500.0")
        self.log_text.see("end")

    # --- Background Status Checker ---
    def _ports(self):
        return int(self.cfg["llm"].get("port", 8080)), int(self.cfg["ytc"].get("port", 8088))

    def status_loop(self):
        while self.is_monitoring:
            llm_port, ytc_port = self._ports()

            # llama-server opens its port at once but answers /health with 503
            # until the model is loaded, so "port open" is not "ready".
            if not is_port_open(llm_port):
                llm_state = "stopped"
            elif check_http_alive(f"http://127.0.0.1:{llm_port}/health"):
                llm_state = "online"
            elif self._llm_port_is_ours(llm_port):
                llm_state = "loading"
            else:
                llm_state = "blocked"

            if not is_port_open(ytc_port):
                ytc_state = "stopped"
            elif check_http_alive(f"http://127.0.0.1:{ytc_port}/api/crawl/progress"):
                ytc_state = "online"
            else:
                ytc_state = "blocked"

            self.llm_state, self.ytc_state = llm_state, ytc_state
            self.ui(self._update_status_labels, llm_state, ytc_state)
            time.sleep(2.0)

    def _llm_port_is_ours(self, llm_port):
        """Port open but not healthy: is that our llama-server still loading,
        or some other program sitting on the port? (Lookup cached for 10 s.)"""
        proc = self.llama_proc
        if (proc is not None and proc.poll() is None) or self._llm_start_lock.locked():
            return True
        if time.time() - self._llm_owner_checked > 10:
            names = {process_image_name(p) for p in pids_listening_on(llm_port)}
            self._llm_owner_is_llama = "llama-server.exe" in names
            self._llm_owner_checked = time.time()
        return self._llm_owner_is_llama

    def _update_status_labels(self, llm_state, ytc_state):
        llm_port, ytc_port = self._ports()
        if llm_state == "online":
            self.llm_status_lbl.config(text=f"🟢 ONLINE (PORT {llm_port})", fg=self.accent_green)
        elif llm_state == "loading":
            self.llm_status_lbl.config(text="🟡 LOADING MODEL...", fg=self.accent_orange)
        elif llm_state == "blocked":
            self.llm_status_lbl.config(text=f"🔴 PORT {llm_port} USED BY OTHER APP", fg=self.accent_red)
        else:
            self.llm_status_lbl.config(text="⚪ STOPPED", fg=self.fg_dim)
        self.btn_llm_start.config(state="normal" if llm_state == "stopped" else "disabled")
        self.btn_llm_stop.config(state="normal" if llm_state in ("online", "loading") else "disabled")

        if ytc_state == "online":
            self.ytc_status_lbl.config(text=f"🟢 ONLINE (PORT {ytc_port})", fg=self.accent_green)
        elif ytc_state == "blocked":
            self.ytc_status_lbl.config(text=f"🔴 PORT {ytc_port} USED BY OTHER APP", fg=self.accent_red)
        else:
            self.ytc_status_lbl.config(text="⚪ STOPPED", fg=self.fg_dim)
        self.btn_ytc_start.config(state="normal" if ytc_state == "stopped" else "disabled")
        self.btn_ytc_stop.config(state="normal" if ytc_state == "online" else "disabled")

    # --- Setup check (missing downloads) ---
    def run_setup_check(self, then_start=False, always_show=False):
        """Check in the background which parts are missing. Opens the setup
        window when something required is missing (or always_show)."""
        cfg = self.cfg

        def _task():
            items = ytc_setup.check_all(cfg)
            self.ui(self._setup_checked, items, then_start, always_show)

        threading.Thread(target=_task, daemon=True).start()

    def _setup_checked(self, items, then_start, always_show):
        missing = ytc_setup.needs_attention(items)
        if missing:
            names = ", ".join(i.label for i in items if not i.ok and i.required)
            self.log(f"🧰 Setup check: missing or not usable: {names}")
        else:
            self.log("✓ Setup check passed: everything needed is installed.")
        if missing or always_show:
            if self.setup_dialog is not None:
                self.setup_dialog.show(items, then_start)
            else:
                self.setup_dialog = SetupDialog(self, items, then_start)
        elif then_start:
            self.start_all_services()

    # --- Service Actions ---
    def _port_owners(self, port):
        names = {process_image_name(p) or f"PID {p}" for p in pids_listening_on(port)}
        return ", ".join(sorted(names)) or "an unknown program"

    def start_llm_server(self):
        model_text = self.model_entry.get().strip()
        threading.Thread(target=self._launch_llm, args=(model_text,), daemon=True).start()

    def _launch_llm(self, model_text):
        """Start llama-server and wait until the model is really loaded.
        Runs in a worker thread. Returns True when /health answers 200."""
        if not self._llm_start_lock.acquire(blocking=False):
            self.log("ℹ️ LLM engine start is already in progress.")
            return False
        try:
            return self._launch_llm_locked(model_text)
        finally:
            self._llm_start_lock.release()

    def _launch_llm_locked(self, model_text):
        llm_port, _ = self._ports()
        health = f"http://127.0.0.1:{llm_port}/health"
        log_path = LOG_DIR / "llama-server.log"
        proc = None
        stop_gen = self._llm_stop_gen

        if is_port_open(llm_port):
            if check_http_alive(health):
                self.log(f"✓ LLM engine is already running on port {llm_port}.")
                return True
            owners = self._port_owners(llm_port)
            if "llama-server.exe" not in owners:
                self.log(f"❌ Port {llm_port} is in use by {owners}. The LLM engine cannot start there. "
                         f"Close that program or change llm.port in config.json.")
                return False
            self.log(f"⏳ An LLM engine is already loading on port {llm_port}; waiting for it...")
        else:
            llm_bin, searched = find_llama_server(self.cfg)
            if llm_bin is None:
                self.log("❌ llama-server.exe was not found. Looked in:")
                for c in searched:
                    self.log(f"   | {c}")
                return False

            model_p = find_model(model_text)
            if model_p is None:
                self.log(f"❌ No base model found. Configured path does not exist: {model_text or '(empty)'}")
                self.log(f"   | Also searched {SCRIPT_DIR / 'models'} for a .gguf larger than 500 MB.")
                self.log("   | Use the '...' button next to 'Model GGUF' to pick the model file.")
                return False
            if not model_text or Path(model_text).resolve() != model_p:
                self.log(f"💡 Model found at {model_p}")
                self.ui(self._remember_model_path, str(model_p))

            cmd = [
                str(llm_bin),
                "-m", str(model_p),
                "-ngl", str(int(self.cfg["llm"].get("gpu_layers", 99))),
                "-c", str(self._context_tokens()),
                "--port", str(llm_port),
                "--host", "127.0.0.1",
            ]
            if not self.cfg["llm"].get("reasoning", False):
                cmd += ["--reasoning", "off"]

            self.log(f"🚀 Launching LLM engine: {llm_bin}")
            self.log(f"   | model: {model_p.name}  port: {llm_port}  log: {log_path}")
            try:
                with open(log_path, "wb") as logf:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(llm_bin.parent),
                        stdin=subprocess.DEVNULL,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        creationflags=NO_WINDOW,
                        env=self._child_env(),
                    )
            except Exception as e:
                self.log(f"❌ Failed to launch llama-server: {e}")
                return False
            if not self.is_monitoring or stop_gen != self._llm_stop_gen:
                kill_process_tree(proc.pid)
                self.log("ℹ️ LLM engine start cancelled.")
                return False
            self.llama_proc = proc
            self.log(f"⏳ LLM engine started (PID {proc.pid}); loading the model into the GPU...")

        started = time.time()
        while self.is_monitoring and time.time() - started < LLM_READY_TIMEOUT_S:
            if check_http_alive(health):
                self.log(f"✓ LLM engine is ready after {time.time() - started:.0f}s.")
                return True
            if stop_gen != self._llm_stop_gen:
                self.log("ℹ️ LLM engine start cancelled.")
                return False
            if proc is not None and proc.poll() is not None:
                self.log(f"❌ LLM engine stopped during start-up (exit code {proc.returncode}). "
                         f"Last lines of {log_path.name}:")
                for ln in tail_text(log_path):
                    self.log(f"   | {ln}")
                if self.llama_proc is proc:
                    self.llama_proc = None
                return False
            if proc is None and not is_port_open(llm_port):
                self.log("❌ The LLM engine that was loading has disappeared.")
                return False
            time.sleep(1.0)
        if self.is_monitoring:
            self.log(f"❌ LLM engine did not become ready within {LLM_READY_TIMEOUT_S}s. See {log_path}")
        return False

    def _remember_model_path(self, path_text):
        self.model_entry.delete(0, "end")
        self.model_entry.insert(0, path_text)

    def stop_llm_server(self):
        self._llm_stop_gen += 1
        proc, self.llama_proc = self.llama_proc, None
        llm_port, _ = self._ports()
        threading.Thread(target=self._stop_service,
                         args=("LLM engine", proc, llm_port, "llama-server.exe"), daemon=True).start()

    def _stop_service(self, label, proc, port, image):
        """Stop the process this GUI started, plus whatever instance of the same
        program is listening on our port. Other programs are never touched."""
        self.log(f"🛑 Stopping {label}...")
        if proc is not None and proc.poll() is None:
            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        for pid in pids_listening_on(port):
            name = process_image_name(pid)
            if name == image:
                kill_process_tree(pid)
            else:
                self.log(f"⚠️ Port {port} is held by {name or f'PID {pid}'}, which is not {image}; left alone.")
        if is_port_open(port):
            self.log(f"⚠️ {label}: port {port} is still in use.")
        else:
            self.log(f"✓ {label} stopped.")

    def _context_tokens(self):
        """llm.context_size from config.json: the engine's -c value, and what
        ytc.exe receives as YTC_LLM_CONTEXT_TOKENS. Any thread."""
        try:
            return max(1024, int(self.cfg["llm"].get("context_size", 8192)))
        except (TypeError, ValueError):
            return 8192

    def _child_env(self):
        return child_env(self._context_tokens())

    def ytc_cmd(self, ytc_bin, db_path, *args):
        """Build a ytc.exe command line (see build_ytc_cmd). Any thread."""
        llm_port, _ = self._ports()
        return build_ytc_cmd(ytc_bin, db_path, args, llm_port, self._max_slop, self._wiki_dir())

    def _ytc_version_problem(self, ytc_bin):
        """None when this ytc.exe may be used, else the text that says why it
        does not match this app. Asked once per file ('ytc.exe --version',
        hidden) and again only after the file changed. Any thread; touches no
        Tk object. A '--version' that failed (antivirus scan of a fresh file,
        PC under load) says nothing about the version: it is logged, not
        stored, and the action goes on; the command itself then reports what
        is wrong with the program."""
        try:
            stamp = Path(ytc_bin).stat().st_mtime_ns
        except OSError:
            stamp = 0
        key = str(ytc_bin)
        problem = ""
        with self._ytc_version_lock:
            cached = self._ytc_version_cache.get(key)
            if cached is not None and cached[0] == stamp:
                found = cached[1]
            else:
                found, problem = ytc_setup.ytc_version(Path(ytc_bin))
                if found is None:
                    self._ytc_version_cache.pop(key, None)
                else:
                    self._ytc_version_cache[key] = (stamp, found)
        if found is None:
            self.log(f"⚠️ {ytc_bin} did not answer '--version' ({problem}). This says nothing about its version.")
            return None
        if found != ytc_setup.YTC_VERSION:
            return ytc_setup.ytc_mismatch_message(Path(ytc_bin), found)
        return None

    def _wiki_dir(self):
        """Wiki output folder from config.json; relative means inside the project."""
        text = str(self.cfg["ytc"].get("wiki_dir") or "wiki").strip() or "wiki"
        path = Path(text)
        return path if path.is_absolute() else SCRIPT_DIR / path

    def _require_ytc(self):
        """Main thread only. Returns (ytc_bin, db_path) or None after saying why."""
        ytc_bin, searched = find_ytc_binary(self.cfg)
        if ytc_bin is None:
            self.log("❌ ytc.exe was not found. Press SETUP CHECK to download or build it. Looked in:")
            for c in searched:
                self.log(f"   | {c}")
            return None
        mismatch = self._ytc_version_problem(ytc_bin)
        if mismatch:
            self.log(f"❌ {mismatch}")
            messagebox.showwarning("ytc.exe does not match this app", mismatch)
            return None
        return ytc_bin, self.get_db_path()

    def start_ytc_server(self):
        db_text = self.db_entry.get().strip()
        threading.Thread(target=self._launch_ytc, args=(db_text,), daemon=True).start()

    def _launch_ytc(self, db_text):
        """Start 'ytc serve' and wait until it answers. Worker thread."""
        if not self._ytc_start_lock.acquire(blocking=False):
            self.log("ℹ️ Web server start is already in progress.")
            return False
        try:
            return self._launch_ytc_locked(db_text)
        finally:
            self._ytc_start_lock.release()

    def _launch_ytc_locked(self, db_text):
        _, ytc_port = self._ports()
        probe = f"http://127.0.0.1:{ytc_port}/api/crawl/progress"
        log_path = LOG_DIR / "ytc-server.log"
        stop_gen = self._ytc_stop_gen

        if is_port_open(ytc_port):
            if check_http_alive(probe):
                self.log(f"✓ YTC web server is already running on port {ytc_port}.")
                return True
            self.log(f"❌ Port {ytc_port} is in use by {self._port_owners(ytc_port)}. "
                     f"Close that program or change ytc.port in config.json.")
            return False

        ytc_bin, searched = find_ytc_binary(self.cfg)
        if ytc_bin is None:
            self.log("❌ ytc.exe was not found. Press SETUP CHECK to download or build it. Looked in:")
            for c in searched:
                self.log(f"   | {c}")
            return False
        mismatch = self._ytc_version_problem(ytc_bin)
        if mismatch:
            self.log(f"❌ {mismatch}")
            return False

        db_path = self.get_db_path(db_text)
        cmd = self.ytc_cmd(ytc_bin, db_path, "serve", "--port", ytc_port)
        self.log(f"🚀 Starting YTC web server on port {ytc_port} (DB: {db_path})...")
        try:
            # Output goes to a file, not a pipe: the server then keeps working
            # even if this window is closed while it is left running.
            with open(log_path, "wb") as logf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(SCRIPT_DIR),
                    stdin=subprocess.DEVNULL,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    creationflags=NO_WINDOW,
                    env=self._child_env(),
                )
        except Exception as e:
            self.log(f"❌ Failed to start YTC web server: {e}")
            return False
        if not self.is_monitoring or stop_gen != self._ytc_stop_gen:
            kill_process_tree(proc.pid)
            self.log("ℹ️ YTC web server start cancelled.")
            return False
        self.ytc_proc = proc
        threading.Thread(target=self._tail_log_file, args=(log_path, proc, "[Server]"), daemon=True).start()

        started = time.time()
        while self.is_monitoring and time.time() - started < YTC_READY_TIMEOUT_S:
            if stop_gen != self._ytc_stop_gen:
                self.log("ℹ️ YTC web server start cancelled.")
                return False
            if proc.poll() is not None:
                self.log(f"❌ YTC web server stopped during start-up (exit code {proc.returncode}). "
                         f"See {log_path}")
                if self.ytc_proc is proc:
                    self.ytc_proc = None
                return False
            if check_http_alive(probe):
                self.log(f"✓ YTC web server is ready (PID {proc.pid}).")
                return True
            time.sleep(0.5)
        if self.is_monitoring:
            self.log(f"❌ YTC web server did not answer within {YTC_READY_TIMEOUT_S}s. See {log_path}")
        return False

    def _tail_log_file(self, log_path, proc, tag):
        """Copy new lines of a child's log file into the log pane until it exits."""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                pending = ""
                while self.is_monitoring:
                    chunk = f.readline()
                    if chunk:
                        pending += chunk
                        if pending.endswith("\n"):
                            if pending.strip():
                                self.log(f"{tag} {pending.strip()}")
                            pending = ""
                        continue
                    if proc.poll() is not None:
                        if pending.strip():
                            self.log(f"{tag} {pending.strip()}")
                        break
                    time.sleep(0.25)
        except OSError as e:
            self.log(f"⚠️ Could not read {log_path}: {e}")

    def stop_ytc_server(self):
        self._ytc_stop_gen += 1
        proc, self.ytc_proc = self.ytc_proc, None
        _, ytc_port = self._ports()
        threading.Thread(target=self._stop_service,
                         args=("YTC web server", proc, ytc_port, "ytc.exe"), daemon=True).start()

    def start_all_services(self):
        model_text = self.model_entry.get().strip()
        db_text = self.db_entry.get().strip()
        self.log("⚡ Starting all services...")

        def _web():
            # The wiki is usable without the LLM, so it does not wait for the model.
            if self._launch_ytc(db_text):
                self.ui(self.open_web_ui)
                self.ui(self.refresh_channels_list)

        threading.Thread(target=self._launch_llm, args=(model_text,), daemon=True).start()
        threading.Thread(target=_web, daemon=True).start()

    def stop_all_services(self):
        self.stop_ytc_server()
        self.stop_llm_server()

    def open_web_ui(self):
        _, ytc_port = self._ports()
        url = f"http://localhost:{ytc_port}"
        self.log(f"🌐 Opening web browser at {url}...")
        webbrowser.open(url)

    def on_close(self):
        """Window close: ask whether the background services should stop too."""
        with self._jobs_lock:
            jobs = [p for p in self._jobs if p.poll() is None]
        if jobs and not messagebox.askokcancel(
                "Work still running",
                "A crawl, video ingest or re-analysis started from this window is still running.\n\n"
                "Closing the window aborts it. Close anyway?"):
            return
        owned = [p for p in (self.llama_proc, self.ytc_proc) if p is not None and p.poll() is None]
        running = bool(owned) or self.llm_state in ("online", "loading") or self.ytc_state == "online"
        stop_services = False
        if running:
            answer = messagebox.askyesnocancel(
                "Close YTC Control Center",
                "The LLM engine and/or the web server are still running.\n\n"
                "Yes  = stop them and close (frees the GPU memory)\n"
                "No   = close this window but leave them running\n"
                "Cancel = keep the Control Center open",
            )
            if answer is None:
                return
            stop_services = answer
        self.is_monitoring = False
        self._llm_stop_gen += 1
        self._ytc_stop_gen += 1
        # A crawl's output pipe dies with this window, so it cannot be left running.
        for job in jobs:
            kill_process_tree(job.pid)
        if stop_services:
            try:
                self.root.config(cursor="watch")
                self.root.update_idletasks()
                llm_port, ytc_port = self._ports()
                self._stop_service("YTC web server", self.ytc_proc, ytc_port, "ytc.exe")
                self._stop_service("LLM engine", self.llama_proc, llm_port, "llama-server.exe")
            except Exception:
                traceback.print_exc()
        self.root.destroy()

    # --- Database & Channel Actions ---
    def get_db_path(self, db_text=None):
        """Resolve the database file. Call without arguments only from the main
        thread (it reads the entry box); worker threads pass the text in."""
        if db_text is None:
            db_text = self.db_entry.get().strip()
        return resolve_db_path(db_text)

    def _log_output(self, tag, output):
        """Copy the lines a short ytc command printed into the log pane."""
        for line in (output or "").splitlines():
            if line.strip():
                self.log(f"{tag} {line.strip()}")

    def _run_capture(self, cmd):
        """Run a short ytc command hidden; returns (exit code, combined output)."""
        try:
            res = subprocess.run(
                cmd, cwd=str(SCRIPT_DIR), capture_output=True, stdin=subprocess.DEVNULL,
                encoding="utf-8", errors="replace", creationflags=NO_WINDOW, env=self._child_env(),
            )
            return res.returncode, ((res.stderr or "") + (res.stdout or "")).strip()
        except Exception as e:
            return -1, str(e)

    def _run_stream(self, cmd, tag, on_start=None):
        """Run a long ytc command hidden and copy its output to the log pane."""
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(SCRIPT_DIR), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding="utf-8", errors="replace", bufsize=1,
                creationflags=NO_WINDOW, env=self._child_env(),
            )
        except Exception as e:
            self.log(f"❌ Could not start {Path(cmd[0]).name}: {e}")
            return -1
        with self._jobs_lock:
            self._jobs.add(proc)
        if on_start is not None:
            on_start(proc)
        try:
            for line in proc.stdout:
                if line.strip():
                    self.log(f"{tag} {line.strip()}")
            return proc.wait()
        finally:
            with self._jobs_lock:
                self._jobs.discard(proc)

    def _confirm_llm_ready(self):
        """Main thread. Without the LLM ytc cannot judge a video (it leaves
        the videos pending), so warn before starting anything."""
        if self.llm_state == "online":
            return True
        state = "still loading the model" if self.llm_state == "loading" else "not running"
        if self.llm_state == "blocked":
            state = "not running (its port is used by another program)"
        return messagebox.askyesno(
            "LLM engine not ready",
            f"The LLM engine is {state}.\n\n"
            "Without it no video can be judged: ytc reports the engine as not reachable "
            "and leaves the videos pending. Start the engine and wait for ONLINE first.\n\n"
            "Continue anyway? (Only useful when YTC_LLM_URL points ytc.exe at another engine.)",
        )

    def refresh_channels_list(self):
        db_p = self.get_db_path()

        def _task():
            if not db_p.exists():
                self.log(f"ℹ️ No database yet at {db_p}. It is created there when the first channel "
                         f"or video is added.")
                self.ui(self._clear_channel_list)
                return

            try:
                conn = sqlite3.connect(str(db_p), timeout=5)
                try:
                    rows = channel_overview(conn)
                finally:
                    conn.close()

                def _render():
                    self._clear_channel_list()
                    for r in rows:
                        self.chan_tree.insert("", "end", iid=r[0], values=channel_row_texts(r))
                    self.log(f"✓ Channel list refreshed: {len(rows)} channels tracked.")

                self.ui(_render)
            except Exception as e:
                self.log(f"❌ Error reading database: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def _clear_channel_list(self):
        for item in self.chan_tree.get_children():
            self.chan_tree.delete(item)

    def add_channel_action(self):
        inp = self.add_chan_entry.get().strip()
        if not inp:
            return
        target = self._require_ytc()
        if target is None:
            return
        ytc_bin, db_p = target

        def _task():
            self.log(f"➕ Adding channel: {inp}...")
            code, output = self._run_capture(self.ytc_cmd(ytc_bin, db_p, "add-channel", "--", inp))
            if code == 0:
                self.log("✓ Channel added successfully!")
                self.ui(self.add_chan_entry.delete, 0, "end")
                self.ui(self.refresh_channels_list)
            else:
                self.log(f"❌ Failed to add channel: {output}")

        threading.Thread(target=_task, daemon=True).start()

    def toggle_mute_action(self):
        sel = self.chan_tree.selection()
        if not sel:
            messagebox.showinfo("Select Channel", "Please select a channel in the list first.")
            return
        chan_id = sel[0]
        target = self._require_ytc()
        if target is None:
            return
        ytc_bin, db_p = target

        def _task():
            self.log(f"🔄 Toggling status for channel {chan_id}...")
            code, output = self._run_capture(self.ytc_cmd(ytc_bin, db_p, "toggle-channel", "--", chan_id))
            if code == 0:
                self.log("✓ Channel status toggled.")
                self.ui(self.refresh_channels_list)
            else:
                self.log(f"❌ Error toggling status: {output}")

        threading.Thread(target=_task, daemon=True).start()

    def remove_channel_action(self):
        sel = self.chan_tree.selection()
        if not sel:
            messagebox.showinfo("Select Channel", "Please select a channel in the list first.")
            return
        chan_id = sel[0]

        confirm = messagebox.askyesno("Confirm Removal", f"Are you sure you want to remove channel:\n{chan_id}?")
        if not confirm:
            return
        db_p = self.get_db_path()

        def _task():
            try:
                conn = sqlite3.connect(str(db_p), timeout=5)
                conn.execute("DELETE FROM channels WHERE id = ?", (chan_id,))
                conn.commit()
                conn.close()
                self.log(f"✓ Removed channel {chan_id} from database.")
                self.ui(self.refresh_channels_list)
            except Exception as e:
                self.log(f"❌ Error removing channel: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def crawl_selected_action(self):
        sel = self.chan_tree.selection()
        if not sel:
            messagebox.showinfo("Select Channel", "Please select a channel in the list to crawl.")
            return
        chan_id = sel[0]
        depth = self._int(self.depth_var, 3, 1, 25)
        delay = self._int(self.delay_var, 10, 10, 30)
        target = self._require_ytc()
        if target is None or not self._confirm_llm_ready():
            return
        ytc_bin, db_p = target

        def _task():
            self.log(f"⚡ Starting crawl for channel {chan_id} (Depth: {depth} videos, Delay: {delay}s)...")
            code = self._run_stream(
                self.ytc_cmd(ytc_bin, db_p, "crawl", "--channel", chan_id, "--limit", depth, "--delay", delay),
                "[Crawl]")
            if code == 0:
                self.log(f"✓ Channel crawl complete for {chan_id}.")
            else:
                self.log(f"❌ Channel crawl for {chan_id} ended with exit code {code}.")
            self.ui(self.refresh_channels_list)

        threading.Thread(target=_task, daemon=True).start()

    def crawl_all_action(self):
        depth = self._int(self.depth_var, 3, 1, 25)
        delay = self._int(self.delay_var, 10, 10, 30)
        target = self._require_ytc()
        if target is None or not self._confirm_llm_ready():
            return
        ytc_bin, db_p = target

        def _task():
            self.log(f"⚡ Starting full crawl across all active channels (Depth: {depth} videos, Delay: {delay}s)...")
            code = self._run_stream(
                self.ytc_cmd(ytc_bin, db_p, "crawl", "--limit", depth, "--delay", delay), "[Crawl]")
            if code == 0:
                self.log("✓ Full crawl complete.")
            else:
                self.log(f"❌ Full crawl ended with exit code {code}.")
            self.ui(self.refresh_channels_list)

        threading.Thread(target=_task, daemon=True).start()

    def direct_ingest_action(self):
        val = self.ingest_entry.get().strip()
        if not val:
            return
        target = self._require_ytc()
        if target is None or not self._confirm_llm_ready():
            return
        ytc_bin, db_p = target

        def _task():
            self.log(f"🎯 Ingesting video directly: {val}...")
            code = self._run_stream(self.ytc_cmd(ytc_bin, db_p, "process-video", "--", val), "[Ingest]")
            if code == 0:
                self.log("✓ Direct video ingestion finished.")
                self.ui(self.ingest_entry.delete, 0, "end")
            else:
                self.log(f"❌ Video ingestion ended with exit code {code}.")
            self.ui(self.refresh_channels_list)

        threading.Thread(target=_task, daemon=True).start()

    def disqualify_video_action(self):
        """Take a video out of the wiki by hand ('ytc exclude'). Main thread."""
        video = self.exclude_entry.get().strip()
        reason = self.exclude_reason_entry.get().strip()
        if not video:
            messagebox.showinfo("Disqualify video", "Enter the URL or ID of the video first.")
            return
        target = self._require_ytc()
        if target is None:
            return
        if not messagebox.askyesno(
                "Disqualify video",
                f"Disqualify this video?\n\n{video}\n\n"
                "It leaves the library at once, is never kept again whatever a later analysis says, and "
                "does not count in its channel's average. Its wiki page is removed by the next "
                "'ytc wiki --prune'.\n\nUndo: enter the same video and press 'Restore'."):
            return
        ytc_bin, db_p = target

        def _task():
            self.log(f"🚫 Disqualifying video: {video}...")
            code, output = self._run_capture(self.ytc_cmd(ytc_bin, db_p, *exclude_args(video, reason)))
            self._log_output("[Exclude]", output)
            if code == 0:
                self.ui(self.exclude_reason_entry.delete, 0, "end")
                self.ui(self.refresh_channels_list)
            else:
                self.log(f"❌ The video was not disqualified (exit code {code}).")

        threading.Thread(target=_task, daemon=True).start()

    def restore_video_action(self):
        """Undo a disqualification ('ytc include'). Main thread."""
        video = self.exclude_entry.get().strip()
        if not video:
            messagebox.showinfo("Restore video", "Enter the URL or ID of the video first.")
            return
        target = self._require_ytc()
        if target is None:
            return
        ytc_bin, db_p = target

        def _task():
            self.log(f"↩ Restoring video: {video}...")
            code, output = self._run_capture(self.ytc_cmd(ytc_bin, db_p, "include", "--", video))
            self._log_output("[Include]", output)
            if code == 0:
                self.log("   | Use 'Sync Wiki Files' to bring its page back.")
                self.ui(self.refresh_channels_list)
            else:
                self.log(f"❌ The video was not restored (exit code {code}).")

        threading.Thread(target=_task, daemon=True).start()

    def reanalyze_library_action(self):
        """Open the 'Re-analyse library' window, or stop a running re-analysis. Main thread."""
        proc = self.reanalyze_proc
        if proc is not None and proc.poll() is None:
            self.log("🛑 Stopping re-analysis (videos finished so far keep their new result)...")
            threading.Thread(target=kill_process_tree, args=(proc.pid,), daemon=True).start()
            return
        if self._reanalyze_busy:
            return
        if self.reanalyze_dialog is not None and self.reanalyze_dialog.alive():
            self.reanalyze_dialog.win.lift()
            return
        self.reanalyze_dialog = ReanalyzeDialog(self)

    def start_reanalyze(self, selection, limit):
        """Run 'ytc reanalyze <selection>' and stream its output to the log. Main thread."""
        if self._reanalyze_busy:
            return
        target = self._require_ytc()
        if target is None or not self._confirm_llm_for_reanalysis():
            return
        ytc_bin, db_p = target
        cmd = self.ytc_cmd(ytc_bin, db_p, *reanalyze_args(selection, limit))
        self._reanalyze_busy = True
        self.btn_reanalyze.config(text="⏳ Starting re-analysis...", fg=self.accent_orange, state="disabled")

        def _task():
            code = -1
            try:
                scope = "all selected videos" if int(limit) <= 0 else f"at most {int(limit)} videos"
                self.log(f"♻ Re-analysis started: ytc reanalyze {selection} ({scope}). "
                         f"Long videos are read in sections; this can take a long time.")
                code = self._run_stream(cmd, "[Reanalyse]", on_start=self._set_reanalyze_proc)
            finally:
                self.reanalyze_proc = None
                if code == 0:
                    self.log("✓ Re-analysis finished. Use 'Sync Wiki Files' to update the wiki.")
                else:
                    self.log(f"⏹ Re-analysis ended (exit code {code}). Videos finished before that keep "
                             f"their new result; start it again to continue with the rest.")
                self.ui(self._reanalyze_ended)
                self.ui(self.refresh_channels_list)

        threading.Thread(target=_task, daemon=True).start()

    def _confirm_llm_for_reanalysis(self):
        """Main thread. A re-analysis without the LLM changes nothing: ytc
        replaces a stored result only by a valid LLM answer."""
        if self.llm_state == "online":
            return True
        state = {"loading": "still loading the model",
                 "blocked": "not running (its port is used by another program)"}.get(self.llm_state, "not running")
        return messagebox.askyesno(
            "LLM engine not ready",
            f"The LLM engine is {state}.\n\n"
            "Without it no video can be re-analysed: the stored results stay as they are and every "
            "video is reported as failed. Start the engine and wait for ONLINE first.\n\n"
            "Continue anyway? (Only useful when YTC_LLM_URL points ytc.exe at another engine.)",
            default="no")

    def _set_reanalyze_proc(self, proc):
        """Called in the worker thread as soon as the process exists."""
        self.reanalyze_proc = proc
        self.ui(self._reanalyze_started)

    def _reanalyze_started(self):
        if self.reanalyze_proc is not None:
            self.btn_reanalyze.config(text="⏹ Stop re-analysis", fg=self.accent_red, state="normal")

    def _reanalyze_ended(self):
        self._reanalyze_busy = False
        self.btn_reanalyze.config(text="♻ Re-analyse library...", fg=self.accent_purple, state="normal")

    def sync_wiki_action(self):
        target = self._require_ytc()
        if target is None:
            return
        ytc_bin, db_p = target

        def _task():
            self.log("📚 Synchronizing Markdown Knowledge Wiki...")
            code, output = self._run_capture(self.ytc_cmd(ytc_bin, db_p, "wiki"))
            if code == 0:
                self.log("✓ Wiki synchronized successfully!")
            else:
                self.log(f"❌ Wiki generation error: {output}")

        threading.Thread(target=_task, daemon=True).start()

    def toggle_watch_action(self):
        """Start or stop the automatic crawl loop ('ytc watch'). Main thread."""
        proc = self.watch_proc
        if proc is not None and proc.poll() is None:
            self.log("🛑 Stopping auto-crawl...")
            threading.Thread(target=kill_process_tree, args=(proc.pid,), daemon=True).start()
            return
        if self._watch_busy:
            # Clicked again before 'ytc watch' was up (or while it is ending):
            # starting now would create a second loop.
            return
        interval = self._int(self.watch_var, 60, 5, 1440)
        depth = self._int(self.depth_var, 3, 1, 25)
        target = self._require_ytc()
        if target is None or not self._confirm_llm_ready():
            return
        ytc_bin, db_p = target
        cmd = self.ytc_cmd(ytc_bin, db_p, "watch", "--interval", interval, "--limit", depth)
        self._watch_busy = True
        self.btn_watch.config(text="⏳ Starting Auto-Crawl...", fg=self.accent_orange, state="disabled")

        def _task():
            code = -1
            try:
                self.log(f"🔁 Auto-crawl started: every {interval} minutes, {depth} videos per channel.")
                code = self._run_stream(cmd, "[Watch]", on_start=self._set_watch_proc)
            finally:
                self.watch_proc = None
                self.log(f"⏹ Auto-crawl stopped (exit code {code}).")
                self.ui(self._watch_ended)
                self.ui(self.refresh_channels_list)

        threading.Thread(target=_task, daemon=True).start()

    def _set_watch_proc(self, proc):
        """Called in the worker thread as soon as the process exists."""
        self.watch_proc = proc
        self.ui(self._watch_started)

    def _watch_started(self):
        if self.watch_proc is not None:
            self.btn_watch.config(text="⏹ Stop Auto-Crawl", fg=self.accent_red, state="normal")

    def _watch_ended(self):
        self._watch_busy = False
        self.btn_watch.config(text="▶ Start Auto-Crawl", fg=self.accent_green, state="normal")

    def browse_model_file(self):
        chosen = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF Model", "*.gguf"), ("All Files", "*.*")]
        )
        if chosen:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, chosen)
            # Inside the app folder the path is stored relative, so the folder stays movable.
            self.cfg["llm"]["model_path"] = config_path_text(chosen)
            save_config(self.cfg)
            self.log(f"💾 Updated model path: {chosen}")

    def save_preferences_action(self):
        self.cfg["crawler"]["search_depth"] = self._int(self.depth_var, 3, 1, 25)
        self.cfg["crawler"]["delay_seconds"] = self._int(self.delay_var, 10, 10, 30)
        new_slop = self._int(self.slop_var, 45, 0, 100)
        slop_changed = new_slop != self._max_slop
        self._max_slop = new_slop
        self.cfg["crawler"]["max_slop_score"] = new_slop
        self.cfg["crawler"]["watch_interval_minutes"] = self._int(self.watch_var, 60, 5, 1440)
        # Paths inside the app folder are stored relative, so the folder can be
        # moved or copied to another PC; the boxes keep showing the real path.
        model_text = self.model_entry.get().strip()
        self.cfg["llm"]["model_path"] = config_path_text(model_text) if model_text else ""
        db_path = self.get_db_path()
        self.cfg["ytc"]["db_path"] = config_path_text(db_path)
        self.db_entry.delete(0, "end")
        self.db_entry.insert(0, str(db_path))

        if save_config(self.cfg):
            self.log("✓ Preferences saved to config.json.")
            note = ""
            if slop_changed:
                note = "\n\n" + SLOP_CHANGE_NOTE
            if slop_changed and self.ytc_state == "online":
                note += ("\n\nThe new Max Slop Score applies to crawls started from this window right away. "
                         "The web UI uses it after the web server is restarted (Stop Server, Start Server).")
            messagebox.showinfo("Saved", "Configuration successfully saved!" + note)
        else:
            messagebox.showerror("Error", "Failed to save config.json")


class ReanalyzeDialog:
    """Explains what a library re-analysis does and how long it can take,
    lets the owner pick the selection, and starts it."""

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Re-analyse library")
        self.win.configure(bg=app.bg_dark)
        self.win.geometry("640x520")
        self.win.minsize(560, 460)
        self.win.transient(app.root)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(self.win, text="♻ Re-analyse library", font=("Segoe UI", 13, "bold"),
                 fg=app.fg_bright, bg=app.bg_dark).pack(anchor="w", padx=14, pady=(12, 4))
        tk.Label(self.win, text=REANALYZE_EXPLANATION, font=("Segoe UI", 9), fg=app.fg_text, bg=app.bg_dark,
                 justify="left", wraplength=600, anchor="w").pack(fill="x", padx=14)

        box = tk.Frame(self.win, bg=app.bg_card, bd=1, relief="solid", padx=10, pady=8)
        box.pack(fill="x", padx=14, pady=(10, 0))
        tk.Label(box, text="Which videos:", font=("Segoe UI", 9, "bold"), fg=app.fg_bright,
                 bg=app.bg_card, anchor="w").pack(fill="x")
        self.selection = tk.StringVar(value=REANALYZE_SELECTIONS[0][0])
        for option, description in REANALYZE_SELECTIONS:
            tk.Radiobutton(box, text=f"{description}  [ytc reanalyze {option}]", variable=self.selection,
                           value=option, font=("Segoe UI", 9), fg=app.fg_text, bg=app.bg_card,
                           selectcolor=app.bg_dark, activebackground=app.bg_card, activeforeground=app.fg_bright,
                           anchor="w", justify="left", wraplength=560).pack(fill="x", pady=2)

        limit_row = tk.Frame(box, bg=app.bg_card)
        limit_row.pack(fill="x", pady=(6, 0))
        tk.Label(limit_row, text="At most this many videos in this run (0 = all selected):", font=("Segoe UI", 9),
                 fg=app.fg_text, bg=app.bg_card).pack(side="left")
        self.limit_var = tk.IntVar(value=0)
        tk.Spinbox(limit_row, from_=0, to=100000, textvariable=self.limit_var, width=7, font=("Segoe UI", 9),
                   bg=app.bg_dark, fg=app.fg_bright, buttonbackground="#21262d").pack(side="left", padx=8)

        btns = tk.Frame(self.win, bg=app.bg_dark)
        btns.pack(fill="x", padx=14, pady=12, side="bottom")
        tk.Button(btns, text="Start re-analysis", font=("Segoe UI", 9, "bold"), bg="#238636", fg="#ffffff",
                  activebackground="#2ea043", activeforeground="#ffffff", relief="flat", padx=12, pady=5,
                  command=self.start).pack(side="left")
        tk.Button(btns, text="Close", font=("Segoe UI", 9), bg="#21262d", fg=app.fg_bright, relief="flat",
                  padx=12, pady=5, command=self.close).pack(side="right")

    def alive(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def start(self):
        selection = self.selection.get()
        limit = self.app._int(self.limit_var, 0, 0, 100000)
        self.close()
        self.app.start_reanalyze(selection, limit)

    def close(self):
        self.win.destroy()
        self.app.reanalyze_dialog = None


class SetupDialog:
    """Lists what is installed and what is missing, and downloads the rest."""

    def __init__(self, app, items, then_start=False):
        self.app = app
        self.items = items
        self.then_start = then_start
        self.busy = False
        self.cancel = threading.Event()
        self._last_progress = 0.0

        self.win = tk.Toplevel(app.root)
        self.win.title("YTC Setup Check")
        self.win.configure(bg=app.bg_dark)
        self.win.geometry("820x720")
        self.win.minsize(700, 560)
        self.win.transient(app.root)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(self.win, text="🧰 Setup Check", font=("Segoe UI", 13, "bold"),
                 fg=app.fg_bright, bg=app.bg_dark).pack(anchor="w", padx=14, pady=(12, 0))
        self.summary_lbl = tk.Label(self.win, text="", font=("Segoe UI", 9), fg=app.fg_dim, bg=app.bg_dark,
                                    justify="left", wraplength=780)
        self.summary_lbl.pack(anchor="w", padx=14, pady=(2, 8))

        self.rows = tk.Frame(self.win, bg=app.bg_card, bd=1, relief="solid")
        self.rows.pack(fill="x", padx=14)

        self.progress = ttk.Progressbar(self.win, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", padx=14, pady=(10, 2))
        self.status_lbl = tk.Label(self.win, text="", font=("Segoe UI", 8), fg=app.fg_dim, bg=app.bg_dark, anchor="w")
        self.status_lbl.pack(fill="x", padx=14)

        self.log_text = tk.Text(self.win, bg="#05070a", fg="#7ee787", font=("Consolas", 8), height=8,
                                wrap="word", relief="flat")
        self.log_text.pack(fill="both", expand=True, padx=14, pady=(6, 8))

        btns = tk.Frame(self.win, bg=app.bg_dark)
        btns.pack(fill="x", padx=14, pady=(0, 12))
        self.btn_install = tk.Button(btns, text="", font=("Segoe UI", 9, "bold"), bg="#238636", fg="#ffffff",
                                     activebackground="#2ea043", activeforeground="#ffffff", relief="flat",
                                     padx=12, pady=5, command=self.start_install)
        self.btn_install.pack(side="left")
        self.btn_recheck = tk.Button(btns, text="🔄 Check again", font=("Segoe UI", 9), bg="#21262d",
                                     fg=app.fg_bright, relief="flat", padx=12, pady=5, command=self.recheck)
        self.btn_recheck.pack(side="left", padx=8)
        self.btn_close = tk.Button(btns, text="Close", font=("Segoe UI", 9), bg="#21262d", fg=app.fg_bright,
                                   relief="flat", padx=12, pady=5, command=self.close)
        self.btn_close.pack(side="right")

        self.render()

    def alive(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def show(self, items, then_start):
        """Reuse the open window for a fresh check result."""
        if not self.alive() or self.busy:
            return
        self.items = items
        self.then_start = self.then_start or then_start
        self.render()
        self.win.lift()

    def render(self):
        app = self.app
        for child in self.rows.winfo_children():
            child.destroy()
        for item in self.items:
            row = tk.Frame(self.rows, bg=app.bg_card)
            row.pack(fill="x", padx=8, pady=4)
            if item.ok:
                mark, colour = "✅", app.accent_green
            elif item.required:
                mark, colour = "❌", app.accent_red
            else:
                mark, colour = "➖", app.accent_orange
            tk.Label(row, text=mark, font=("Segoe UI", 10), fg=colour, bg=app.bg_card, width=3).pack(side="left", anchor="n")
            text = tk.Frame(row, bg=app.bg_card)
            text.pack(side="left", fill="x", expand=True)
            title = item.label if item.required else f"{item.label} (optional)"
            tk.Label(text, text=title, font=("Segoe UI", 9, "bold"), fg=app.fg_bright, bg=app.bg_card,
                     anchor="w").pack(fill="x")
            tk.Label(text, text=item.detail, font=("Segoe UI", 8), fg=app.fg_dim, bg=app.bg_card, anchor="w",
                     justify="left", wraplength=700).pack(fill="x")
            if not item.ok and item.help_url:
                link = tk.Label(text, text=item.help_url, font=("Segoe UI", 8, "underline"), fg=app.accent_blue,
                                bg=app.bg_card, anchor="w", cursor="hand2")
                link.pack(fill="x")
                link.bind("<Button-1>", lambda e, url=item.help_url: webbrowser.open(url))

        installable = [i for i in self.items if not i.ok and i.can_install]
        manual = [i for i in self.items if not i.ok and i.required and not i.can_install]
        size = ytc_setup.total_download_bytes(self.items)
        if installable:
            label = f"⬇ Download && install missing parts (about {ytc_setup.human_size(size)})" if size else "⬇ Install missing parts"
            self.btn_install.config(text=label.replace("&&", "&"), state="normal")
        else:
            self.btn_install.config(text="Nothing to download", state="disabled")

        if not ytc_setup.needs_attention(self.items):
            summary = "Everything needed is installed. You can close this window."
        elif manual and not installable:
            summary = "These parts cannot be installed automatically. Follow the link shown for each one, then press 'Check again'."
        elif manual:
            summary = ("Some parts are missing. The button below downloads what can be fetched automatically; "
                       "items with a link need you (for example the graphics driver).")
        else:
            summary = ("Some parts are missing. They are not in the GitHub repository because they are large or "
                       "belong to other projects. The button below downloads them from their official sources.")
        self.summary_lbl.config(text=summary)

    def _log(self, message):
        self.app.log(f"[Setup] {message}")
        self.app.ui(self._append, message)

    def _append(self, message):
        if self.alive():
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")

    def _progress(self, done, total):
        now = time.time()
        if now - self._last_progress < 0.2 and done < total:
            return
        self._last_progress = now
        self.app.ui(self._set_progress, done, total)

    def _set_progress(self, done, total):
        if not self.alive():
            return
        self.progress["value"] = int(done * 1000 / total) if total else 0
        self.status_lbl.config(text=f"{ytc_setup.human_size(done)} of {ytc_setup.human_size(total)}" if total else "")

    def start_install(self):
        if self.busy:
            return
        todo = [i for i in self.items if not i.ok and i.can_install]
        if not todo:
            return
        lines = "\n".join(f"  • {i.label}" + (f"  (~{ytc_setup.human_size(i.download_bytes)})" if i.download_bytes else "")
                          for i in todo)
        if not messagebox.askyesno(
                "Download missing parts",
                f"This will download and install:\n\n{lines}\n\n"
                "Sources: github.com/PrismML-Eng/llama.cpp, huggingface.co/prism-ml, pypi.org and, for a\n"
                f"ready-made ytc.exe, github.com/{ytc_setup.GITHUB_REPO}.\n"
                "The engine, the model and ytc.exe are checked against fixed SHA-256 checksums before use.\n"
                "An interrupted download continues where it stopped.\n\nStart now?",
                parent=self.win):
            return
        self.busy = True
        self.cancel.clear()
        self.btn_install.config(state="disabled")
        self.btn_recheck.config(state="disabled")
        self.btn_close.config(text="Cancel")
        cfg = self.app.cfg
        items = self.items

        def _work():
            errors = []
            try:
                errors = ytc_setup.install_missing(items, self._log, self._progress, self.cancel)
            except ytc_setup.SetupCancelled:
                errors = ["Cancelled. Downloaded parts are kept and continue next time."]
            except Exception as e:
                traceback.print_exc()
                errors = [f"{type(e).__name__}: {e}"]
            fresh = ytc_setup.check_all(cfg)
            self.app.ui(self._finished, fresh, errors)

        threading.Thread(target=_work, daemon=True).start()

    def _finished(self, items, errors):
        self.busy = False
        self.items = items
        if not self.alive():
            self.app.setup_dialog = None
            return
        self.btn_recheck.config(state="normal")
        self.btn_close.config(text="Close")
        self.progress["value"] = 0
        self.status_lbl.config(text="")
        self.render()
        for err in errors:
            self._append(f"⚠ {err}")
        if not ytc_setup.needs_attention(items):
            self._append("✓ Setup complete.")
            if self.then_start:
                self.then_start = False
                self.win.destroy()
                self.app.setup_dialog = None
                self.app.start_all_services()

    def recheck(self):
        if self.busy:
            return
        self.status_lbl.config(text="Checking...")
        cfg = self.app.cfg

        def _work():
            fresh = ytc_setup.check_all(cfg)
            self.app.ui(self._finished, fresh, [])

        threading.Thread(target=_work, daemon=True).start()

    def close(self):
        if self.busy:
            if messagebox.askyesno("Cancel setup", "The setup is still working. Cancel it?\n\n"
                                   "What was downloaded so far is kept.", parent=self.win):
                self.cancel.set()
            return
        start_anyway = self.then_start
        self.then_start = False
        self.win.destroy()
        self.app.setup_dialog = None
        if start_anyway:
            # The user closed the window without fixing everything: try to start
            # anyway; whatever is still missing is reported in the log pane.
            self.app.start_all_services()


def main():
    setup_stdio()
    os.chdir(SCRIPT_DIR)
    autostart = "--autostart" in sys.argv[1:]

    if not acquire_single_instance():
        if autostart:
            try:
                AUTOSTART_REQUEST.write_text("1", encoding="utf-8")
            except OSError:
                pass
        return

    if sys.platform == "win32":
        try:
            import ctypes
            # Own taskbar identity, so Windows shows our icon instead of Python's.
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("YTC.ControlCenter")
        except Exception:
            pass

    root = tk.Tk()
    icon_path = SCRIPT_DIR / "ytc.ico"
    if icon_path.exists():
        try:
            root.iconbitmap(default=str(icon_path))
        except tk.TclError:
            pass
    YTCControlApp(root, autostart=autostart)
    root.mainloop()


def self_test():
    """Checks of the logic that needs no window:  python YTC_ControlCenter.pyw --self-test"""
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_one_database_rule(self):
            root = Path(tempfile.gettempdir()) / "ytc-app"
            self.assertEqual(resolve_db_path("", root), root / "data" / "ytc.db")
            self.assertEqual(resolve_db_path("   ", root), root / "data" / "ytc.db")
            self.assertEqual(resolve_db_path("data/ytc.db", root), root / "data" / "ytc.db")
            self.assertEqual(resolve_db_path("lib/other.db", root), root / "lib" / "other.db")
            elsewhere = Path(tempfile.gettempdir()) / "elsewhere" / "x.db"
            self.assertEqual(resolve_db_path(str(elsewhere), root), elsewhere)
            self.assertEqual(resolve_db_path(f'"{elsewhere}"', root), elsewhere)
            self.assertEqual(resolve_db_path(""), SCRIPT_DIR / "data" / "ytc.db")

        def test_paths_inside_the_app_folder_are_stored_relative(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "app"
                (root / "models").mkdir(parents=True)
                self.assertEqual(config_path_text(root / "data" / "ytc.db", root), "data/ytc.db")
                self.assertEqual(config_path_text(root / "models" / "m.gguf", root), "models/m.gguf")
                outside = Path(tmp) / "other" / "m.gguf"
                self.assertEqual(config_path_text(outside, root), str(outside))
                # Round trip: what is stored resolves to the same file again.
                stored = config_path_text(resolve_db_path("lib/x.db", root), root)
                self.assertEqual(resolve_db_path(stored, root), root / "lib" / "x.db")

        def test_context_size_reaches_ytc(self):
            env = child_env(8192, base={"PATH": "x"})
            self.assertEqual(env["YTC_LLM_CONTEXT_TOKENS"], "8192")
            self.assertEqual(child_env(16384, base={})["YTC_LLM_CONTEXT_TOKENS"], "16384")
            self.assertNotIn("YTC_LLM_CONTEXT_TOKENS", child_env(base={}))
            # Another engine is in use: its context window is not ours to state.
            other = child_env(8192, base={"YTC_LLM_URL": "http://10.0.0.5:1234/v1", "YTC_LLM_CONTEXT_TOKENS": "32768"})
            self.assertEqual(other["YTC_LLM_CONTEXT_TOKENS"], "32768")
            self.assertNotIn("YTC_LLM_CONTEXT_TOKENS", child_env(8192, base={"YTC_LLM_URL": "http://10.0.0.5:1234/v1"}))

        def _library(self):
            conn = sqlite3.connect(":memory:")
            conn.executescript(
                "CREATE TABLE channels(id TEXT PRIMARY KEY, title TEXT, status TEXT, rolling_slop_score REAL);"
                "CREATE TABLE videos(id TEXT PRIMARY KEY, channel_id TEXT, title TEXT);"
                "CREATE TABLE insights(video_id TEXT PRIMARY KEY, slop_score INTEGER, is_worth_keeping INTEGER,"
                " analyzer_type TEXT, topic_verdict TEXT, info_status TEXT, excluded_manual INTEGER DEFAULT 0);")
            conn.execute("INSERT INTO channels VALUES('c1', 'Alpha', 'active', 25.0)")
            conn.execute("INSERT INTO channels VALUES('c2', 'Empty', 'muted', NULL)")
            return conn

        def test_channel_list_counts_every_video_in_one_group(self):
            conn = self._library()
            rows = [
                # id, slop, flag, analyzer, verdict, info, excluded
                ("kept1", 20, 1, "llm:m", "ok", "sufficient", 0),
                ("kept2", 30, 1, "heuristic", None, None, 0),         # no verdict recorded rejects nothing
                ("below", 80, 0, "llm:m", "ok", "sufficient", 0),
                ("heur", 70, 0, "heuristic", None, None, 0),
                ("polit", 20, 1, "llm:m", "politics", "sufficient", 0),   # good quality, wrong topic
                ("fringe", 60, 0, "llm:m", "fringe", "sufficient", 0),
                ("short", None, 0, "info:insufficient", None, "insufficient", 0),
                ("removed", 10, 1, "llm:m", "ok", "sufficient", 1),      # by hand wins over everything
                ("removed2", 20, 0, "llm:m", "politics", "sufficient", 1),
            ]
            for vid, slop, flag, analyzer, verdict, info, excluded in rows:
                conn.execute("INSERT INTO videos VALUES(?, 'c1', ?)", (vid, vid))
                conn.execute("INSERT INTO insights VALUES(?,?,?,?,?,?,?)",
                             (vid, slop, flag, analyzer, verdict, info, excluded))
            conn.execute("INSERT INTO videos VALUES('pending', 'c1', 'no insight yet')")
            overview = {row[0]: row for row in channel_overview(conn)}
            self.assertEqual(overview["c1"][4:], (2, 2, 2, 1, 2))
            self.assertEqual(sum(overview["c1"][4:]), len(rows))
            self.assertEqual(overview["c2"][4:], (0, 0, 0, 0, 0))
            texts = channel_row_texts(overview["c1"])
            self.assertEqual(texts[2:5], ("25/100", "2 / 4", "2 · 2 · 1 · 2"))
            self.assertEqual(channel_row_texts(overview["c2"])[2:5], ("Pending", "0 / 0", "0 · 0 · 0 · 0"))

        def test_reanalyze_and_exclude_commands(self):
            self.assertEqual(reanalyze_args("--all"), ["reanalyze", "--all"])
            self.assertEqual(reanalyze_args("--analyzer heuristic", 25),
                             ["reanalyze", "--analyzer", "heuristic", "--limit", "25"])
            with self.assertRaises(ValueError):
                reanalyze_args("--limit")        # not a selection
            self.assertEqual(exclude_args("-AbCdEfGh12", " duplicate "),
                             ["exclude", "--reason", "duplicate", "--", "-AbCdEfGh12"])
            self.assertEqual(exclude_args("abc"), ["exclude", "--", "abc"])
            cmd = build_ytc_cmd("y", "d", exclude_args("abc", "x"), 8080, 45, "w", env={})
            self.assertLess(cmd.index("--wiki-dir"), cmd.index("exclude"))
            self.assertIn("several minutes", REANALYZE_EXPLANATION)
            self.assertIn("stop", REANALYZE_EXPLANATION)
            self.assertIn("reanalyze --all", SLOP_CHANGE_NOTE)

        def test_version_of_ytc_exe_is_compared_with_this_app(self):
            from unittest import mock

            class Fake:
                def __init__(self):
                    self._ytc_version_cache = {}
                    self._ytc_version_lock = threading.Lock()
                    self.lines = []

                def log(self, text):
                    self.lines.append(text)

            need = ytc_setup.YTC_VERSION
            with tempfile.TemporaryDirectory() as tmp:
                exe = Path(tmp) / "ytc.exe"
                exe.write_bytes(b"MZ")
                fake = Fake()
                answers = [(None, "timed out after 20s"), (need, ""), AssertionError("asked a third time")]
                with mock.patch.object(ytc_setup, "ytc_version", side_effect=answers) as asked:
                    # A call that failed is reported and not taken for a mismatch.
                    self.assertIsNone(YTCControlApp._ytc_version_problem(fake, exe))
                    self.assertEqual(fake._ytc_version_cache, {})             # nothing stored
                    self.assertIn("timed out after 20s", fake.lines[-1])
                    # The next action asks again and gets the real answer, which is then kept.
                    self.assertIsNone(YTCControlApp._ytc_version_problem(fake, exe))
                    self.assertIsNone(YTCControlApp._ytc_version_problem(fake, exe))
                    self.assertEqual(asked.call_count, 2)
                    self.assertEqual(len(fake.lines), 1)
                fake = Fake()
                with mock.patch.object(ytc_setup, "ytc_version", return_value=("9.9.9", "")):
                    text = YTCControlApp._ytc_version_problem(fake, exe)
                self.assertIn(f"does not match this app (found 9.9.9, need {need})", text)

        def test_settings_of_the_control_center_are_always_passed(self):
            env = {"YTC_MAX_SLOP": "60", "YTC_WIKI_DIR": "elsewhere", "YTC_DB": "other.db"}
            cmd = build_ytc_cmd("ytc.exe", "data/ytc.db", ("crawl", "--limit", 3), 8080, 45, "wiki", env=env)
            self.assertEqual(cmd, ["ytc.exe", "--db", "data/ytc.db", "--max-slop", "45", "--wiki-dir", "wiki",
                                   "crawl", "--limit", "3"])

        def test_llm_url_follows_the_port_unless_the_environment_sets_it(self):
            self.assertIn("http://127.0.0.1:8090/v1", build_ytc_cmd("y", "d", ("wiki",), 8090, 45, "w", env={}))
            self.assertNotIn("--llm-url", build_ytc_cmd("y", "d", ("wiki",), 8080, 45, "w", env={}))
            self.assertNotIn("--llm-url", build_ytc_cmd("y", "d", ("wiki",), 8090, 45, "w",
                                                        env={"YTC_LLM_URL": "http://10.0.0.5:1234/v1"}))

        def test_options_come_before_the_subcommand(self):
            cmd = build_ytc_cmd("y", "d", ("add-channel", "--", "@x"), 8081, 30, "w", env={})
            self.assertLess(cmd.index("--wiki-dir"), cmd.index("add-channel"))
            self.assertEqual(cmd[-2:], ["--", "@x"])

        def test_missing_keys_fall_back_to_defaults(self):
            cfg = _merge_defaults(DEFAULT_CONFIG, {"ytc": {"port": 9000}, "extra": 1})
            self.assertEqual(cfg["ytc"]["port"], 9000)
            self.assertEqual(cfg["ytc"]["wiki_dir"], "wiki")
            self.assertEqual(cfg["llm"]["port"], 8080)
            self.assertEqual(cfg["extra"], 1)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    try:
        main()
    except Exception:
        details = traceback.format_exc()
        try:
            append_log_file(details)
        except Exception:
            pass
        try:
            messagebox.showerror(
                "YTC Control Center could not start",
                f"{details[-1500:]}\n\nFull details: {LOG_DIR / 'control_center.log'}",
            )
        except Exception:
            pass
        sys.exit(1)
