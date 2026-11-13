#!/usr/bin/env python3
"""
Re-run the real pipeline for ONE video and show what was stored.

This is a thin wrapper around the binary:

    ytc --db <db> process-video <video id>

process-video fetches the transcript again, removes SponsorBlock segments and
sends the text to the configured analyzer; the result it stores is whatever the
analyzer returned. Afterwards this script reads the stored row (read-only) and
prints it, including the analyzer_type the binary recorded.

This script never writes analysis results itself: every stored value comes
from the binary.

The printed decision is the effective one: a video is kept only when its topic
is acceptable (not fringe, nonsense or politics), its slop score (quality only)
is at or below the cutoff, the transcript holds enough information and the
video was not disqualified by hand.

Database: --db, else YTC_DB, else <app folder>/data/ytc.db.
Binary:   YTC_BIN, else target/release/ytc(.exe), else ytc(.exe) in the app folder.

Usage:
    python scripts/update_single_video.py <video id or URL> [--db PATH] [--reanalyze]
    python scripts/update_single_video.py --self-test

The helpers below (find_ytc_binary, resolve_db_path, run_ytc, read_insight) are
also used by the other scripts in this folder that call the binary.
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ytc_paths  # noqa: E402

ROOT = _ytc_paths.REPO
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YtcError(Exception):
    """The binary or the database is missing. The message is shown as is."""


def find_ytc_binary() -> Path:
    """YTC_BIN if set, else the release build, else ytc(.exe) in the app folder."""
    found = _ytc_paths.find_ytc_binary()
    if found is None:
        raise YtcError(_ytc_paths.binary_missing_message())
    return found


def resolve_db_path(arg: Optional[str] = None) -> Path:
    """--db value, else YTC_DB, else <app folder>/data/ytc.db (the rule ytc uses)."""
    return _ytc_paths.resolve_db(arg)


def run_ytc(args: Sequence[str], db_path: Path, timeout: float = _ytc_paths.ytc_timeout_s(1),
            binary: Optional[Path] = None) -> Tuple[int, str]:
    """Runs `ytc --db <db> <args...>`. Returns (exit code, combined output).
    A timeout is reported as exit code -1, never as success. The default
    limit fits one video read in full (see _ytc_paths.ytc_timeout_s); a run
    over several videos needs runner_for()."""
    cmd = [str(binary or find_ytc_binary()), "--db", str(db_path), *args]
    try:
        res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout, creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {int(timeout)} s"
    except OSError as e:
        return -1, f"could not start {cmd[0]}: {e}"
    return res.returncode, (res.stdout or "") + (res.stderr or "")


def runner_for(videos: int, pause_s: float = 0):
    """run_ytc with a time limit for a command that may analyse up to `videos`
    videos in one run (e.g. `ytc discover --limit N`)."""
    limit = _ytc_paths.ytc_timeout_s(videos, pause_s)

    def runner(args: Sequence[str], db_path: Path) -> Tuple[int, str]:
        return run_ytc(args, db_path, timeout=limit)

    runner.timeout = limit   # for callers and tests that want to see the limit
    return runner


def open_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise YtcError(f"Database not found: {db_path}")
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def read_insight(db_path: Path, video_id: str) -> Optional[Dict[str, Any]]:
    """The stored analysis of one video, or None. Read-only."""
    if not db_path.is_file():
        return None
    conn = open_readonly(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT i.video_id, v.title, v.status, i.slop_score, i.is_worth_keeping, i.quality_rating, "
            "i.analyzer_type, i.summary, i.key_insights_json, i.created_at, i.updated_at, i.category, "
            "i.topic_verdict, i.topic_reason, i.info_status, i.info_reason, i.excluded_manual, "
            "i.excluded_reason, i.sections_total "
            "FROM insights i LEFT JOIN videos v ON v.id = i.video_id WHERE i.video_id = ?",
            (video_id,)).fetchone()
    except sqlite3.Error as e:
        raise YtcError(f"Could not read {db_path}: {e}") from e
    finally:
        conn.close()
    return dict(row) if row else None


def extract_video_id(text: str) -> str:
    """11-character id from an id, a watch URL or a youtu.be link."""
    text = text.strip()
    if VIDEO_ID_RE.match(text):
        return text
    # Same pattern as extract_video_id in src/main.rs.
    m = re.search(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)", text)
    if not m:
        raise YtcError(f"Not a YouTube video id or URL: {text!r}")
    return m.group(1)


KEPT, BELOW_CUTOFF, OFF_TOPIC, INSUFFICIENT, DISQUALIFIED = (
    "kept", "below_cutoff", "off_topic", "insufficient", "disqualified")


def row_outcome(row: Dict[str, Any]) -> str:
    """What became of a stored video. Same order of checks as ytc
    (pipeline::video_outcome): disqualified by hand, then off-topic, then not
    enough information, then the quality cutoff. A row without a recorded
    topic verdict or information status is rejected by neither."""
    if row.get("excluded_manual"):
        return DISQUALIFIED
    if (row.get("topic_verdict") or "ok") != "ok":
        return OFF_TOPIC
    if row.get("info_status") == "insufficient":
        return INSUFFICIENT
    return KEPT if row.get("is_worth_keeping") else BELOW_CUTOFF


def has_quality_score(row: Dict[str, Any]) -> bool:
    """False for 'not enough information' rows: ytc does not score them."""
    return row.get("info_status") != "insufficient"


def counts_in_average(row: Dict[str, Any]) -> bool:
    """Does ytc count this video in its channel's rolling slop average (see
    SCORED_SQL in src/db.rs)? Only videos whose quality the LLM judged, with
    an acceptable topic, that were not disqualified. A heuristic score is a
    word-count estimate made without the LLM and does not count either."""
    analyzer = str(row.get("analyzer_type") or "")
    return (has_quality_score(row) and not analyzer.startswith("heuristic")
            and row_outcome(row) in (KEPT, BELOW_CUTOFF))


def _with_reason(label: str, reason: Any) -> str:
    return f"{label} ({reason})" if reason else label


def outcome_text(row: Dict[str, Any]) -> str:
    outcome = row_outcome(row)
    if outcome == DISQUALIFIED:
        return _with_reason("not kept: disqualified by hand", row.get("excluded_reason"))
    if outcome == OFF_TOPIC:
        return _with_reason(f"not kept: off-topic, {row['topic_verdict']}", row.get("topic_reason"))
    if outcome == INSUFFICIENT:
        return _with_reason("not kept: not enough information", row.get("info_reason"))
    return "kept" if outcome == KEPT else "not kept: below the quality cutoff"


def stored_at(row: Dict[str, Any]) -> Any:
    """Time of the latest analysis."""
    return row.get("updated_at")


def format_insight(row: Dict[str, Any]) -> str:
    try:
        insights: List[str] = json.loads(row.get("key_insights_json") or "[]")
    except ValueError:
        insights = []
    score = f"{row['slop_score']}/100 (quality only)" if has_quality_score(row) else "not scored"
    lines = [
        f"Video:     {row['video_id']}  {row.get('title') or ''}",
        f"Status:    {row.get('status')}",
        f"Analyzer:  {row['analyzer_type']}   (as recorded by the binary)",
        f"Stored at: {stored_at(row)}",
        f"Slop:      {score}   rating: {row['quality_rating']}",
        f"Topic:     {row.get('topic_verdict') or 'no verdict recorded'}",
        f"Decision:  {outcome_text(row)}",
    ]
    # An analysis reads the whole transcript; NULL when none ran.
    total = row.get("sections_total")
    if total:
        lines.append(f"Read:      whole transcript ({total} section{'' if total == 1 else 's'})")
    lines += [
        f"Category:  {row.get('category') or 'none'}",
        f"Summary:   {row.get('summary')}",
    ]
    lines += [f"  - {i}" for i in insights if isinstance(i, str)]
    return "\n".join(lines)


def update_video(video: str, db_path: Path, reanalyze: bool = False, runner=run_ytc) -> int:
    """Returns the process exit code: 0 only when the binary succeeded AND a row is stored."""
    video_id = extract_video_id(video)
    before = read_insight(db_path, video_id)
    command = ["reanalyze", video_id] if reanalyze else ["process-video", video_id]
    print(f"Running: ytc --db {db_path} {' '.join(command)}")
    code, output = runner(command, db_path)
    print(output.rstrip())
    if code != 0:
        print(f"\nFAILED: ytc exited with code {code}. Nothing is reported as updated.")
        if before:
            print("The row stored before this run is unchanged unless the output above says otherwise.")
        return 1
    after = read_insight(db_path, video_id)
    if after is None:
        print(f"\nFAILED: ytc exited with 0 but no analysis is stored for {video_id}.")
        return 1
    print("\nStored result:")
    print(format_insight(after))
    if before and stored_at(before) == stored_at(after):
        print("\nNOTE: the stored row has the same timestamp as before; the analysis was not replaced.")
    return 0


def self_test() -> int:
    import tempfile
    import unittest

    def make_db(path: Path) -> None:
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE videos(id TEXT PRIMARY KEY, title TEXT, status TEXT);"
            "CREATE TABLE insights(video_id TEXT PRIMARY KEY, slop_score INTEGER, is_worth_keeping INTEGER,"
            " quality_rating TEXT, summary TEXT, key_insights_json TEXT, analyzer_type TEXT, category TEXT,"
            " topic_verdict TEXT, topic_reason TEXT, info_status TEXT, info_reason TEXT,"
            " sections_total INTEGER, excluded_manual INTEGER NOT NULL DEFAULT 0,"
            " excluded_reason TEXT, created_at TEXT, updated_at TEXT);")
        conn.commit()
        conn.close()

    def store(path: Path, vid: str, analysed: str, excluded: int = 0) -> None:
        conn = sqlite3.connect(path)
        conn.execute("INSERT OR REPLACE INTO videos VALUES(?,?,?)", (vid, "A title", "processed"))
        conn.execute(
            "INSERT OR REPLACE INTO insights(video_id, slop_score, is_worth_keeping, quality_rating, summary,"
            " key_insights_json, analyzer_type, category, topic_verdict, info_status,"
            " sections_total, excluded_manual, created_at, updated_at)"
            " VALUES(?, 20, 1, 'High Signal', 'A summary', '[\"one\"]', 'llm:test-model', 'Baking', 'ok',"
            " 'sufficient', 4, ?, ?, ?)", (vid, excluded, analysed, analysed))
        conn.commit()
        conn.close()

    class Tests(unittest.TestCase):
        VID = "AbCdEfGh123"

        def test_video_id_forms(self):
            self.assertEqual(extract_video_id(self.VID), self.VID)
            self.assertEqual(extract_video_id(f"https://www.youtube.com/watch?v={self.VID}&t=3"), self.VID)
            self.assertEqual(extract_video_id(f"https://youtu.be/{self.VID}"), self.VID)
            self.assertEqual(extract_video_id("-AbCdEfGh12"), "-AbCdEfGh12")
            with self.assertRaises(YtcError):
                extract_video_id("not a video")

        def test_success_needs_exit_code_and_row(self):
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "t.db"
                make_db(db)
                calls = []

                def ok_but_nothing_stored(args, db_path):
                    calls.append(list(args))
                    return 0, "KEPT High Signal"   # words alone prove nothing
                self.assertEqual(update_video(self.VID, db, runner=ok_but_nothing_stored), 1)
                self.assertEqual(calls, [["process-video", self.VID]])

                def ok_and_stored(args, db_path):
                    store(db_path, self.VID, "2026-01-02T00:00:00Z")
                    return 0, "done"
                self.assertEqual(update_video(self.VID, db, runner=ok_and_stored), 0)
                self.assertEqual(read_insight(db, self.VID)["analyzer_type"], "llm:test-model")

                def failing(args, db_path):
                    return 3, "LLM unavailable"
                self.assertEqual(update_video(self.VID, db, runner=failing), 1)

                def reanalyze(args, db_path):
                    calls.append(list(args))
                    return 0, ""
                update_video(self.VID, db, reanalyze=True, runner=reanalyze)
                self.assertEqual(calls[-1], ["reanalyze", self.VID])

        def test_database_is_opened_read_only(self):
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "t.db"
                make_db(db)
                conn = open_readonly(db)
                with self.assertRaises(sqlite3.OperationalError):
                    conn.execute("INSERT INTO videos VALUES('x','y','z')")
                conn.close()
                self.assertIsNone(read_insight(Path(tmp) / "missing.db", self.VID))

        def test_outcome_follows_the_judgment_columns(self):
            base = {"video_id": "v", "analyzer_type": "llm:m", "slop_score": 20, "is_worth_keeping": 1,
                    "quality_rating": "High Signal"}
            self.assertEqual(row_outcome(base), KEPT)
            self.assertEqual(row_outcome(dict(base, is_worth_keeping=0, slop_score=80)), BELOW_CUTOFF)
            self.assertEqual(row_outcome(dict(base, topic_verdict="ok")), KEPT)
            politics = dict(base, topic_verdict="politics", topic_reason="election commentary")
            self.assertEqual(row_outcome(politics), OFF_TOPIC)
            self.assertFalse(counts_in_average(politics))
            self.assertIn("politics (election commentary)", outcome_text(politics))
            short = dict(base, analyzer_type="info:insufficient", slop_score=None, is_worth_keeping=0,
                         info_status="insufficient", info_reason="only 40 words")
            self.assertEqual(row_outcome(short), INSUFFICIENT)
            self.assertFalse(has_quality_score(short))
            self.assertFalse(counts_in_average(short))
            self.assertIn("not scored", format_insight(short))
            self.assertNotIn("/100", format_insight(short))
            self.assertIn("no verdict recorded", format_insight(dict(base, analyzer_type="heuristic")))
            # A disqualification wins over everything the model said.
            removed = dict(base, excluded_manual=1, excluded_reason="duplicate")
            self.assertEqual(row_outcome(removed), DISQUALIFIED)
            self.assertFalse(counts_in_average(removed))
            self.assertIn("disqualified by hand (duplicate)", outcome_text(removed))
            self.assertTrue(counts_in_average(base))
            self.assertFalse(counts_in_average(dict(base, analyzer_type="heuristic")))
            self.assertTrue(counts_in_average(dict(base, is_worth_keeping=0)))

        def test_stored_row_is_read_with_its_judgment(self):
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "t.db"
                make_db(db)
                store(db, self.VID, "2026-01-02T00:00:00Z")
                row = read_insight(db, self.VID)
                self.assertEqual(row_outcome(row), KEPT)
                self.assertIn("Read:      whole transcript (4 sections)", format_insight(row))
                self.assertIn("Category:  Baking", format_insight(row))
                self.assertIn("Topic:     ok", format_insight(row))
                store(db, self.VID, "2026-01-03T00:00:00Z", excluded=1)
                row = read_insight(db, self.VID)
                self.assertEqual(row_outcome(row), DISQUALIFIED)
                self.assertEqual(stored_at(row), "2026-01-03T00:00:00Z")

        def test_database_and_binary_follow_the_one_rule(self):
            self.assertEqual(resolve_db_path("x.db"), Path("x.db").absolute())
            self.assertEqual(_ytc_paths.default_db({}), ROOT / "data" / "ytc.db")

        def test_time_limits_fit_whole_transcript_analysis(self):
            import inspect
            from unittest import mock
            default = inspect.signature(run_ytc).parameters["timeout"].default
            self.assertGreaterEqual(default, 30 * 60)                 # one video, read in full
            self.assertGreaterEqual(runner_for(3).timeout, 3 * 30 * 60)
            seen = {}

            def fake_run(cmd, **kwargs):
                seen.update(kwargs)
                return subprocess.CompletedProcess(cmd, 0, "out", "")
            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                self.assertEqual(runner_for(3, 2)(["discover", "q"], Path("x.db"))[0], 0)
            self.assertEqual(seen["timeout"], _ytc_paths.ytc_timeout_s(3, 2))

        def test_missing_binary_is_reported(self):
            code, out = run_ytc(["list"], Path("x.db"), binary=Path("definitely-not-here-ytc"))
            self.assertEqual(code, -1)
            self.assertIn("could not start", out)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    ap = argparse.ArgumentParser(description="Re-run the real ytc pipeline for one video and print the stored result.")
    ap.add_argument("video", nargs="?",
                    help="YouTube video id or URL (for an id that starts with '-', write -- before it)")
    ap.add_argument("--db", help="database file (default: YTC_DB, else <app folder>/data/ytc.db)")
    ap.add_argument("--reanalyze", action="store_true",
                    help="call 'ytc reanalyze <id>' (stored transcript, no YouTube request) instead of 'process-video'")
    ap.add_argument("--self-test", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.video:
        ap.error("a video id or URL is required")
    try:
        return update_video(args.video, resolve_db_path(args.db), reanalyze=args.reanalyze)
    except YtcError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
