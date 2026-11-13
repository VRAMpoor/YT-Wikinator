#!/usr/bin/env python3
"""
Vet candidate YouTube channels and add the ones that pass.

Where candidates come from (this script discovers nothing by itself):
  --file handles.txt     a list YOU wrote: one @handle per line
  --discover "query"     runs the binary's real search, `ytc discover <query>`,
                         which evaluates search results and promotes creators
                         on its own; this script then only reports which
                         channels are new in the database

What the vetting does: for each handle it asks
yt-dlp for the channel's most recent long-form videos, runs
`ytc process-video` on the first two, reads the stored slop scores back from
the database (read-only) and averages them. A two-video average is a rough
screen, not a measurement of the channel.

Only videos whose quality the LLM judged enter the average, the same rule ytc
uses for a channel's rolling average: a video with too little content ("not
enough information"), an off-topic video (fringe, nonsense, politics), one
disqualified by hand or one that only has the heuristic estimate (no LLM
running) gives no score. The slop score measures quality only; the subject of a channel
does not raise or lower it.

Admission: a channel whose average is at or below --max-avg-slop is added with
`ytc add-channel @handle`. The binary resolves the channel id, stores the
channel as 'active' without a score, and computes score and status later from
crawled videos. This script never writes to the database and never sets
'high_signal' itself.

Database: --db, else YTC_DB, else <app folder>/data/ytc.db.
Binary:   YTC_BIN, else target/release/ytc(.exe), else ytc(.exe) in the app folder.

Usage:
    python scripts/channel_discovery.py --file handles.txt [--out result.json]
    python scripts/channel_discovery.py --discover "sourdough baking" [--limit 3]
    python scripts/channel_discovery.py --self-test
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_admitted_channels import read_handles  # noqa: E402
from update_single_video import (  # noqa: E402
    YtcError, counts_in_average, open_readonly, outcome_text, read_insight, resolve_db_path, run_ytc,
    runner_for,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_MAX_AVG_SLOP = 40.0
SAMPLE_VIDEOS = 2

Runner = Callable[[Sequence[str], Path], Tuple[int, str]]


def log(msg: str) -> None:
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def tracked_channel_ids(db_path: Path) -> Dict[str, str]:
    """{channel id: title} of the channels in the database. Read-only."""
    if not db_path.is_file():
        return {}
    conn = open_readonly(db_path)
    try:
        return {row[0]: row[1] for row in conn.execute("SELECT id, title FROM channels")}
    finally:
        conn.close()


def resolve_channel_metadata(handle: str) -> Tuple[Optional[str], Optional[str]]:
    """Channel id and title from yt-dlp (one request to YouTube)."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--playlist-items", "1",
        "--print", "%(channel_id)s\t%(channel)s",
        f"https://www.youtube.com/{handle}"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        out = res.stdout.strip()
        if "\t" in out:
            ch_id, ch_name = out.split("\t", 1)
            ch_id = ch_id.strip()
            ch_name = ch_name.strip()
            if ch_id and ch_id != "NA":
                return ch_id, ch_name
    except Exception as e:
        log(f"Metadata error for {handle}: {e}")
    return None, None


def fetch_channel_top_videos(handle: str, limit: int = 3) -> List[Dict[str, str]]:
    """Ids and titles of the channel's most recent videos (shorts skipped)."""
    url = f"https://www.youtube.com/{handle}/videos"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--playlist-end", str(limit),
        "--print", "%(id)s\t%(title)s",
        url
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=40)
        videos = []
        for line in res.stdout.strip().split("\n"):
            if not line.strip() or "\t" not in line:
                continue
            vid_id, title = (part.strip() for part in line.split("\t", 1))
            if vid_id and "#shorts" not in title.lower():
                videos.append({"id": vid_id, "title": title})
        return videos
    except Exception as e:
        log(f"Videos fetch error for {handle}: {e}")
        return []


def evaluate_video_slop(vid_id: str, db_path: Path, runner: Runner = run_ytc) -> Optional[int]:
    """Runs the real pipeline on one video and returns the quality score the
    binary stored. A score counts only when the binary exited with 0 AND the
    row exists AND ytc itself would count it in a channel average; words in
    the output are not evidence."""
    code, output = runner(["process-video", vid_id], db_path)
    if code != 0:
        tail = next((ln.strip() for ln in reversed(output.splitlines()) if ln.strip()), "")
        log(f"    ytc process-video failed (exit code {code}): {tail}")
        return None
    row = read_insight(db_path, vid_id)
    if row is None:
        return None
    if not counts_in_average(row):
        log(f"    not counted: {outcome_text(row)}")
        return None
    return int(row["slop_score"])


def vet_channel(handle: str, db_path: Path, max_avg_slop: float, pause: float = 15.0,
                runner: Runner = run_ytc,
                top_videos: Callable[[str, int], List[Dict[str, str]]] = fetch_channel_top_videos,
                ) -> Dict[str, Any]:
    """Scores up to SAMPLE_VIDEOS recent videos and decides. Adds the channel
    through `ytc add-channel` when it passes. Returns a result record."""
    record: Dict[str, Any] = {"handle": handle, "sample_scores": [], "sample_videos": [],
                              "avg_slop": None, "decision": "deferred", "reason": ""}
    videos = top_videos(handle, 3)[:SAMPLE_VIDEOS]
    if not videos:
        record["reason"] = "no long-form videos found (or yt-dlp could not reach YouTube)"
        return record
    for n, v in enumerate(videos):
        if n and pause > 0:
            time.sleep(pause)  # polite pacing
        log(f"  Sample: '{v['title'][:55]}' ({v['id']})")
        score = evaluate_video_slop(v["id"], db_path, runner)
        if score is None:
            log("    no quality score for this video")
            continue
        log(f"    stored slop score: {score}/100")
        record["sample_scores"].append(score)
        record["sample_videos"].append(v["id"])
    scores = record["sample_scores"]
    if not scores:
        record["reason"] = ("no video gave a quality score (failed, too little content, off-topic, "
                            "disqualified, or scored without the LLM)")
        return record
    record["avg_slop"] = round(sum(scores) / len(scores), 1)
    if record["avg_slop"] > max_avg_slop:
        record["decision"] = "not added"
        record["reason"] = f"average {record['avg_slop']} of {len(scores)} video(s) is above {max_avg_slop}"
        return record
    code, output = runner(["add-channel", handle], db_path)
    if code == 0:
        record["decision"] = "added"
        record["reason"] = (f"average {record['avg_slop']} of {len(scores)} video(s); added as 'active', "
                            "score and status will be computed by ytc from crawled videos")
    else:
        tail = next((ln.strip() for ln in reversed(output.splitlines()) if ln.strip()), "")
        record["decision"] = "add failed"
        record["reason"] = f"ytc add-channel exited with {code}: {tail}"
    return record


def vet_handles(handles: Sequence[str], db_path: Path, max_avg_slop: float = DEFAULT_MAX_AVG_SLOP,
                pause: float = 15.0, **kwargs: Any) -> List[Dict[str, Any]]:
    results = []
    for n, handle in enumerate(handles):
        if n and pause > 0:
            time.sleep(pause)
        log("=" * 65)
        log(f"Candidate: {handle}")
        rec = vet_channel(handle, db_path, max_avg_slop, pause=pause, **kwargs)
        log(f"  -> {rec['decision'].upper()}: {rec['reason']}")
        results.append(rec)
    return results


# The pause `ytc discover` makes between two candidates unless --delay is given.
DISCOVER_DELAY_S = 2


def run_discover(queries: Sequence[str], db_path: Path, limit: int,
                 runner: Optional[Runner] = None) -> Dict[str, Any]:
    """Runs `ytc discover` per query and reports the channels that are new in
    the database afterwards. All evaluation is done by the binary. One run
    analyses up to `limit` videos, each read in full, so the default runner
    gets a time limit for that many videos instead of the one-video default."""
    if runner is None:
        runner = runner_for(limit, DISCOVER_DELAY_S)
    before = tracked_channel_ids(db_path)
    runs = []
    for q in queries:
        log(f"ytc discover \"{q}\" --limit {limit}")
        code, output = runner(["discover", q, "--limit", str(limit)], db_path)
        print(output.rstrip())
        runs.append({"query": q, "exit_code": code})
        if code != 0:
            log(f"  discover FAILED with exit code {code}")
    after = tracked_channel_ids(db_path)
    new = {cid: title for cid, title in after.items() if cid not in before}
    return {"runs": runs, "new_channels": new}


def write_results(path: Path, source: str, body: Dict[str, Any]) -> None:
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "candidate_source": source, **body}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Results written to {path}")


def self_test() -> int:
    import sqlite3
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.db = Path(self.tmp.name) / "t.db"
            conn = sqlite3.connect(self.db)
            conn.executescript(
                "CREATE TABLE channels(id TEXT PRIMARY KEY, title TEXT);"
                "CREATE TABLE videos(id TEXT PRIMARY KEY, title TEXT, status TEXT);"
                "CREATE TABLE insights(video_id TEXT PRIMARY KEY, slop_score INTEGER, is_worth_keeping INTEGER,"
                " quality_rating TEXT, summary TEXT, key_insights_json TEXT, analyzer_type TEXT, category TEXT,"
                " topic_verdict TEXT, topic_reason TEXT, info_status TEXT, info_reason TEXT,"
                " sections_total INTEGER, excluded_manual INTEGER NOT NULL DEFAULT 0,"
                " excluded_reason TEXT, created_at TEXT, updated_at TEXT);")
            conn.commit()
            conn.close()
            self.calls: List[List[str]] = []

        def tearDown(self):
            self.tmp.cleanup()

        def fake_ytc(self, scores: Dict[str, Optional[int]], fail: Sequence[str] = (),
                     rows: Optional[Dict[str, Tuple[str, Optional[str], Optional[str]]]] = None) -> Runner:
            """Stands in for the binary: stores a score like process-video would.
            `rows` maps a video id to (analyzer_type, topic_verdict, info_status)."""
            def runner(args, db_path):
                self.calls.append(list(args))
                if args[0] == "process-video":
                    if args[1] in fail:
                        return 1, "Error: transcript unavailable\nKEPT High Signal"
                    analyzer, verdict, info = (rows or {}).get(args[1], ("llm:x", "ok", "sufficient"))
                    conn = sqlite3.connect(db_path)
                    conn.execute(
                        "INSERT OR REPLACE INTO insights(video_id, slop_score, is_worth_keeping, quality_rating,"
                        " summary, key_insights_json, analyzer_type, topic_verdict, info_status, created_at,"
                        " updated_at) VALUES(?, ?, 1, 'q', 's', '[]', ?, ?, ?, 't', 't')",
                        (args[1], scores[args[1]], analyzer, verdict, info))
                    conn.commit()
                    conn.close()
                    return 0, "ok"
                if args[0] == "add-channel":
                    conn = sqlite3.connect(db_path)
                    conn.execute("INSERT OR REPLACE INTO channels VALUES(?,?)", ("UC" + args[1], args[1]))
                    conn.commit()
                    conn.close()
                    return 0, "Added channel"
                if args[0] == "discover":
                    conn = sqlite3.connect(db_path)
                    conn.execute("INSERT OR REPLACE INTO channels VALUES('UCnew','Found by ytc')")
                    conn.commit()
                    conn.close()
                    return 0, "Discovery harvest complete"
                return 2, "unknown"
            return runner

        @staticmethod
        def videos(handle, limit):
            return [{"id": handle + "_1", "title": "one"}, {"id": handle + "_2", "title": "two"},
                    {"id": handle + "_3", "title": "three"}]

        def test_good_channel_is_added_through_the_binary(self):
            runner = self.fake_ytc({"@a_1": 10, "@a_2": 30})
            rec = vet_channel("@a", self.db, 40.0, pause=0, runner=runner, top_videos=self.videos)
            self.assertEqual((rec["decision"], rec["avg_slop"], rec["sample_scores"]), ("added", 20.0, [10, 30]))
            self.assertEqual(self.calls[-1], ["add-channel", "@a"])
            self.assertEqual(tracked_channel_ids(self.db), {"UC@a": "@a"})

        def test_sloppy_channel_is_not_added(self):
            runner = self.fake_ytc({"@b_1": 90, "@b_2": 70})
            rec = vet_channel("@b", self.db, 40.0, pause=0, runner=runner, top_videos=self.videos)
            self.assertEqual(rec["decision"], "not added")
            self.assertNotIn(["add-channel", "@b"], self.calls)

        def test_failed_run_gives_no_score_even_if_output_says_kept(self):
            runner = self.fake_ytc({}, fail=["@c_1", "@c_2"])
            rec = vet_channel("@c", self.db, 40.0, pause=0, runner=runner, top_videos=self.videos)
            self.assertEqual((rec["decision"], rec["sample_scores"]), ("deferred", []))
            self.assertEqual(tracked_channel_ids(self.db), {})

        def test_short_and_off_topic_videos_never_enter_the_average(self):
            # A video with too little content is not scored.
            runner = self.fake_ytc({"@e_1": None, "@e_2": 30},
                                   rows={"@e_1": ("info:insufficient", None, "insufficient")})
            rec = vet_channel("@e", self.db, 40.0, pause=0, runner=runner, top_videos=self.videos)
            self.assertEqual((rec["sample_scores"], rec["avg_slop"], rec["decision"]), ([30], 30.0, "added"))
            # An off-topic video has a quality score, which is not counted.
            runner = self.fake_ytc({"@f_1": 35, "@f_2": 20},
                                   rows={"@f_1": ("llm:x", "politics", "sufficient"),
                                         "@f_2": ("llm:x", "fringe", "sufficient")})
            rec = vet_channel("@f", self.db, 40.0, pause=0, runner=runner, top_videos=self.videos)
            self.assertEqual((rec["sample_scores"], rec["decision"]), ([], "deferred"))
            self.assertNotIn(["add-channel", "@f"], self.calls)

        def test_no_videos_defers(self):
            rec = vet_channel("@d", self.db, 40.0, pause=0, runner=self.fake_ytc({}), top_videos=lambda h, n: [])
            self.assertEqual(rec["decision"], "deferred")

        def test_discover_reports_only_new_channels(self):
            conn = sqlite3.connect(self.db)
            conn.execute("INSERT INTO channels VALUES('UCknown','Known')")
            conn.commit()
            conn.close()
            out = run_discover(["sourdough baking"], self.db, 3, runner=self.fake_ytc({}))
            self.assertEqual(out["new_channels"], {"UCnew": "Found by ytc"})
            self.assertEqual(self.calls, [["discover", "sourdough baking", "--limit", "3"]])
            # Without a runner of its own the run gets room for all 3 videos, each read in full.
            limits = []

            def fake_runner_for(videos, pause_s=0):
                limits.append((videos, pause_s))
                return self.fake_ytc({})
            saved = globals()["runner_for"]
            globals()["runner_for"] = fake_runner_for
            try:
                run_discover(["sourdough baking"], self.db, 3)
            finally:
                globals()["runner_for"] = saved
            self.assertEqual(limits, [(3, DISCOVER_DELAY_S)])
            self.assertGreaterEqual(runner_for(3, DISCOVER_DELAY_S).timeout, 3 * 30 * 60)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None, default_out: Optional[Path] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Vet candidate channels (from your own list or from 'ytc discover') and add the ones that pass.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--file", type=Path, help="text file with one @handle per line (your own candidate list)")
    src.add_argument("--discover", action="append", metavar="QUERY",
                     help="run 'ytc discover QUERY' (repeatable); the binary searches and evaluates")
    ap.add_argument("--db", help="database file (default: YTC_DB, else <app folder>/data/ytc.db)")
    ap.add_argument("--limit", type=int, default=3, help="candidate videos per --discover query (default 3)")
    ap.add_argument("--max-avg-slop", type=float, default=DEFAULT_MAX_AVG_SLOP,
                    help=f"add a channel when its sample average is at or below this (default {DEFAULT_MAX_AVG_SLOP:g})")
    ap.add_argument("--pause", type=float, default=15.0, help="seconds between videos and channels (default 15)")
    ap.add_argument("--out", type=Path, default=default_out, help="write the results as JSON to this file")
    ap.add_argument("--self-test", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not (args.file or args.discover):
        ap.error("no candidates: pass --file handles.txt or --discover QUERY")

    try:
        db_path = resolve_db_path(args.db)
        if args.discover:
            log(f"Candidate source: ytc discover ({len(args.discover)} query/queries). Database: {db_path}")
            out = run_discover(args.discover, db_path, args.limit)
            log(f"New channels in the database after discover: {len(out['new_channels'])}")
            for cid, title in out["new_channels"].items():
                log(f"  + {title} ({cid})")
            if args.out:
                write_results(args.out, "ytc discover", out)
            return 0 if all(r["exit_code"] == 0 for r in out["runs"]) else 1

        handles, source = read_handles(args.file), f"handles file {args.file}"
        if not handles:
            raise YtcError("The candidate list is empty.")
        log(f"Candidate source: {source}. {len(handles)} handle(s). Database: {db_path}")
        results = vet_handles(handles, db_path, args.max_avg_slop, pause=args.pause)
    except YtcError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    log("=" * 65)
    for decision in ("added", "not added", "deferred", "add failed"):
        group = [r for r in results if r["decision"] == decision]
        log(f"{decision}: {len(group)}")
        for r in group:
            log(f"  {r['handle']}  avg slop {r['avg_slop']}  samples {r['sample_scores']}")
    log("Averages are over at most two videos per channel: a rough screen, not a channel rating.")
    if args.out:
        write_results(args.out, source, {"max_avg_slop": args.max_avg_slop, "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
