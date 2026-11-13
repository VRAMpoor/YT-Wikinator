"""Where the scripts in this folder find the database and the ytc binary.

One rule, the same as ytc.exe itself uses:

    database:  --db argument (handled by the caller), else the environment
               variable YTC_DB, else <app folder>/data/ytc.db
    binary:    the environment variable YTC_BIN, else
               <app folder>/target/release/ytc(.exe), else <app folder>/ytc(.exe)

The app folder is the folder that contains scripts/. No other place is searched.

Also here: the one time limit the scripts give a ytc run (PER_VIDEO_TIMEOUT_S,
ytc_timeout_s), so that no script stops ytc in the middle of a long video.

Built-in checks:  python scripts/_ytc_paths.py --self-test
"""

import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Mapping, Optional

REPO = Path(__file__).resolve().parent.parent
EXE_NAME = "ytc.exe" if sys.platform == "win32" else "ytc"

# How long ONE video may take before a script stops ytc. The LLM reads the
# whole transcript, one request per section (measured: 312 s for a
# 208,000-character transcript, 136 s for 71,000 characters), and a video
# without captions is transcribed by Whisper first, possibly on the CPU. The
# limit is a guard against a hung process, not a budget, hence generous.
PER_VIDEO_TIMEOUT_S = 30 * 60
# Feed and search requests, start-up, writing the wiki.
RUN_OVERHEAD_S = 120


def ytc_timeout_s(videos: int = 1, pause_s: float = 0) -> int:
    """Time limit in seconds for one ytc run that may analyse up to `videos`
    videos with `pause_s` seconds between them (crawl/discover --delay). Every
    limit a script passes for such a run comes from here, so none can be
    shorter than what a single long video needs."""
    videos = max(1, int(videos))
    return int(videos * (PER_VIDEO_TIMEOUT_S + max(0.0, float(pause_s))) + RUN_OVERHEAD_S)


def default_db(env: Optional[Mapping[str, str]] = None) -> Path:
    """YTC_DB if set and not blank, else <app folder>/data/ytc.db. A relative
    YTC_DB is made absolute here, so that this script and the ytc process it
    starts (which runs in the app folder) mean the same file."""
    env = os.environ if env is None else env
    text = (env.get("YTC_DB") or "").strip()
    return Path(text).absolute() if text else REPO / "data" / "ytc.db"


def resolve_db(arg: Optional[object] = None, env: Optional[Mapping[str, str]] = None) -> Path:
    """The --db value when one was given, else default_db()."""
    text = str(arg).strip() if arg is not None else ""
    return Path(text).absolute() if text else default_db(env)


def ytc_binary_candidates(env: Optional[Mapping[str, str]] = None) -> List[Path]:
    env = os.environ if env is None else env
    text = (env.get("YTC_BIN") or "").strip()
    return ([Path(text)] if text else []) + [REPO / "target" / "release" / EXE_NAME, REPO / EXE_NAME]


def find_ytc_binary(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """The first candidate that exists, or None."""
    for candidate in ytc_binary_candidates(env):
        if candidate.is_file():
            return candidate.resolve()
    return None


def binary_missing_message(env: Optional[Mapping[str, str]] = None) -> str:
    looked = ", ".join(str(c) for c in ytc_binary_candidates(env))
    return (f"ytc binary not found. Looked in: {looked}\n"
            "Run 'python ytc_setup.py --install' (downloads or builds it), "
            "build it with 'cargo build --release', or set YTC_BIN.")


def open_readonly(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open the database read-only (default: default_db()). Stops with a clear
    message when the file does not exist; never creates one."""
    path = Path(db_path) if db_path is not None else default_db()
    if not path.is_file():
        raise SystemExit(f"Database not found: {path}\n"
                         "Pass --db or set YTC_DB if your database is somewhere else.")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def self_test() -> int:
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_database_rule(self):
            self.assertEqual(default_db({}), REPO / "data" / "ytc.db")
            self.assertEqual(default_db({"YTC_DB": "   "}), REPO / "data" / "ytc.db")
            with tempfile.TemporaryDirectory() as tmp:
                other = Path(tmp) / "other.db"
                self.assertEqual(default_db({"YTC_DB": str(other)}), other)
                self.assertEqual(resolve_db(str(other), {"YTC_DB": "ignored.db"}), other)
            self.assertEqual(resolve_db(None, {}), REPO / "data" / "ytc.db")
            self.assertTrue(default_db({"YTC_DB": "rel.db"}).is_absolute())

        def test_binary_rule(self):
            names = [str(c.relative_to(REPO)).replace("\\", "/") for c in ytc_binary_candidates({})]
            self.assertEqual(names, [f"target/release/{EXE_NAME}", EXE_NAME])
            with tempfile.TemporaryDirectory() as tmp:
                exe = Path(tmp) / EXE_NAME
                exe.write_bytes(b"x")
                self.assertEqual(find_ytc_binary({"YTC_BIN": str(exe)}), exe.resolve())
                self.assertEqual(ytc_binary_candidates({"YTC_BIN": str(exe)})[0], exe)
            self.assertIn("ytc_setup.py --install", binary_missing_message({}))

        def test_time_limits_fit_whole_transcript_analysis(self):
            self.assertGreaterEqual(PER_VIDEO_TIMEOUT_S, 30 * 60)
            self.assertEqual(ytc_timeout_s(), PER_VIDEO_TIMEOUT_S + RUN_OVERHEAD_S)
            self.assertEqual(ytc_timeout_s(2, 20), 2 * (PER_VIDEO_TIMEOUT_S + 20) + RUN_OVERHEAD_S)
            # Never below one video, whatever is passed.
            self.assertEqual(ytc_timeout_s(0), ytc_timeout_s(1))
            self.assertEqual(ytc_timeout_s(-3, -5), ytc_timeout_s(1))
            for n in (1, 2, 10):
                self.assertGreaterEqual(ytc_timeout_s(n), n * PER_VIDEO_TIMEOUT_S)

        def test_read_only_open(self):
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "t.db"
                conn = sqlite3.connect(db)
                conn.execute("CREATE TABLE t(x)")
                conn.commit()
                conn.close()
                conn = open_readonly(db)
                with self.assertRaises(sqlite3.OperationalError):
                    conn.execute("INSERT INTO t VALUES(1)")
                conn.close()
                missing = Path(tmp) / "missing.db"
                with self.assertRaises(SystemExit):
                    open_readonly(missing)
                self.assertFalse(missing.exists())

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    print(f"database: {default_db()}")
    print(f"binary:   {find_ytc_binary() or 'not found'}")
