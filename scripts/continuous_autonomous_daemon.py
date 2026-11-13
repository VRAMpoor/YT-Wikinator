#!/usr/bin/env python3
"""
YTC supervisor loop: works through pending videos, then re-crawls channels.

What it does, in a loop until you stop it (Ctrl+C):
  1. takes up to 10 pending videos and runs `ytc process-video --json` on each
  2. when nothing is pending, runs `ytc crawl --channel <id>` for the three
     channels that were crawled longest ago
  3. regenerates the Markdown wiki with `ytc wiki`

What it does not do: it does not search for new channels or topics (use
`ytc discover` or `ytc watch --discover` for that). It also does nothing of
its own when YouTube refuses a transcript request: ytc itself then reads the
transcript from the video page or transcribes the audio. Only when ytc reports
that a video failed because of such a refusal does this script wait longer
between videos.

How results are judged: a video counts as processed only when
`ytc process-video --json` exited with code 0 AND printed the stored result as
a JSON object for that video. A crawl or wiki run counts as successful only
when ytc exited with code 0. Words in the program's messages are used solely
to tell WHY a run failed (rate limit, LLM not reachable).

A processed video is not the same as a kept one. The printed JSON says what
became of it, and the loop counts each group: kept, below the quality cutoff,
off-topic (fringe / nonsense / politics), not enough information (too short or
almost no speech; not counted in the channel's average), disqualified by hand.

The database is only read here. Every change to a video goes through ytc: this
script never writes a video status. A video that failed or ran too long stays
pending. This script (not ytc) then leaves it alone for 30 minutes, after each
further failure twice as long, up to a day; nothing is taken out of the queue
for good.

Database: YTC_DB, else <app folder>/data/ytc.db (the rule ytc itself uses).
Binary:   YTC_BIN, else target/release/ytc(.exe), else ytc(.exe) in the app folder.
Built-in checks:  python scripts/continuous_autonomous_daemon.py --self-test
"""

import json
import sys
import time
import sqlite3
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ytc_paths  # noqa: E402

BASE_DIR = _ytc_paths.REPO
# The one database rule of the app: YTC_DB if set, else <app folder>/data/ytc.db.
DB_PATH = _ytc_paths.default_db()
# None when no binary exists; require_db_and_binary() then stops with the places searched.
YTC_BIN = _ytc_paths.find_ytc_binary()

INITIAL_DELAY = 20
MAX_BACKOFF = 300
CURRENT_DELAY = INITIAL_DELAY

# One video may need the Whisper fallback on the CPU, and the LLM reads the
# whole transcript (one request per section of a long video). Both take far
# longer than five minutes; a short limit would stop it every time.
PROCESS_TIMEOUT_S = _ytc_paths.PER_VIDEO_TIMEOUT_S
# `ytc crawl --limit N` analyses up to N new videos of the channel in one run,
# so its limit must cover N whole videos (see crawl_timeout_s), not one.
CRAWL_VIDEOS_PER_CHANNEL = 2
# Pause before a video that failed or timed out is tried again; doubles each
# time up to the maximum. Kept in memory only.
RETRY_FIRST_S = 30 * 60
RETRY_MAX_S = 24 * 60 * 60

# Outcomes of one `ytc process-video --json` run.
SUCCESS = "success"                  # exit code 0 and the stored result was printed
NO_RESULT = "no_result"              # exit code 0 but no result object on stdout
LLM_UNAVAILABLE = "llm_unavailable"  # ytc left the video pending on purpose
RATE_LIMITED = "rate_limited"        # YouTube refused (HTTP 429 / IP block)
FAILED = "failed"                    # any other non-zero exit code
TIMEOUT = "timeout"                  # stopped after PROCESS_TIMEOUT_S

# What became of a processed video (fields of `ytc process-video --json`).
KEPT = "kept"
BELOW_CUTOFF = "below_cutoff"
OFF_TOPIC = "off_topic"
INSUFFICIENT = "insufficient"
DISQUALIFIED = "disqualified"
CATEGORIES = (KEPT, BELOW_CUTOFF, OFF_TOPIC, INSUFFICIENT, DISQUALIFIED)
CATEGORY_LABELS = {
    KEPT: "kept",
    BELOW_CUTOFF: "below the quality cutoff",
    OFF_TOPIC: "off-topic (fringe / nonsense / politics)",
    INSUFFICIENT: "not enough information",
    DISQUALIFIED: "disqualified by hand",
}


def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def require_db_and_binary():
    """Stops with a clear message instead of idling on a missing DB or binary."""
    if YTC_BIN is None:
        sys.exit(_ytc_paths.binary_missing_message())
    if not DB_PATH.exists():
        sys.exit(
            f"Database not found: {DB_PATH}\n"
            "Add a channel first (ytc add-channel <url>) or set YTC_DB to your database file."
        )


def classify_outcome(returncode: int, output: str, row_stored: bool) -> str:
    """Success is decided by the exit code and the stored row only. The output
    text is consulted just to name the reason for a failure."""
    if returncode == 0:
        return SUCCESS if row_stored else NO_RESULT
    if "LLM engine not available" in output:
        return LLM_UNAVAILABLE
    if "429" in output or "Too Many Requests" in output or "IpBlocked" in output:
        return RATE_LIMITED
    return FAILED


def parse_result(stdout: str, vid_id: str):
    """The object `ytc process-video --json` prints on success (one JSON object
    on stdout and nothing else), or None when stdout is not that."""
    try:
        data = json.loads((stdout or "").strip())
    except ValueError:
        return None
    if isinstance(data, dict) and data.get("video_id") == vid_id:
        return data
    return None


def result_category(result: dict) -> str:
    """Which group a processed video belongs to. `is_worth_keeping` in the
    JSON is the effective decision of ytc, so it is asked first; the other
    fields then say why a video was not kept (same order as ytc: by hand,
    topic, amount of information, quality)."""
    if result.get("is_worth_keeping") is True:
        return KEPT
    if result.get("excluded") is True:
        return DISQUALIFIED
    if result.get("topic_verdict") not in (None, "ok"):
        return OFF_TOPIC
    if result.get("info_status") == "insufficient":
        return INSUFFICIENT
    return BELOW_CUTOFF


def describe_result(result: dict) -> str:
    """One log line about a processed video: group, reason, score, reading."""
    category = result_category(result)
    text = CATEGORY_LABELS[category]
    reason = {
        OFF_TOPIC: result.get("topic_reason"),
        INSUFFICIENT: result.get("info_reason"),
        DISQUALIFIED: result.get("excluded_reason"),
    }.get(category)
    if category == OFF_TOPIC:
        text = f"off-topic: {result['topic_verdict']}"
    if reason:
        text += f" ({reason})"
    score = result.get("slop_score")
    # The score is null for videos that were not scored.
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        text += f", slop score {int(score)}/100"
    # An analysis reads the whole transcript; null when none ran.
    total = result.get("sections_total")
    if isinstance(total, int) and not isinstance(total, bool) and total > 0:
        text += f", read: whole transcript ({total} section{'' if total == 1 else 's'})"
    return text


def new_counts() -> dict:
    return {category: 0 for category in CATEGORIES}


def counts_line(counts: dict) -> str:
    """The same wording and order as ytc's own crawl summary."""
    return ", ".join(f"{counts.get(c, 0)} {CATEGORY_LABELS[c]}" for c in CATEGORIES)


def judge_process_video(returncode: int, stdout: str, stderr: str, vid_id: str) -> str:
    """Outcome of one `ytc process-video --json` run. ytc exits non-zero and
    leaves stdout empty when the video could not be processed, so the exit
    code plus the printed object decide; the database is not consulted."""
    return classify_outcome(returncode, stderr or "", parse_result(stdout, vid_id) is not None)


class RetryLater:
    """Videos that failed or timed out, and when each may be tried again.
    In memory only: the video stays 'pending' in the database, and after a
    restart of this script everything is eligible at once."""

    def __init__(self, first=RETRY_FIRST_S, longest=RETRY_MAX_S, clock=time.time):
        self.first, self.longest, self.clock = first, longest, clock
        self._entries = {}   # video id -> (time of next try, pause used)

    def postpone(self, vid_id: str) -> int:
        """Schedule the next try; returns the pause in seconds."""
        _, last_pause = self._entries.get(vid_id, (0, 0))
        pause = min(self.longest, last_pause * 2) if last_pause else self.first
        self._entries[vid_id] = (self.clock() + pause, pause)
        return pause

    def clear(self, vid_id: str) -> None:
        self._entries.pop(vid_id, None)

    def waiting(self) -> set:
        """Ids that must not be tried yet."""
        now = self.clock()
        return {vid for vid, (due, _) in self._entries.items() if due > now}


def last_line(output: str) -> str:
    return next((ln.strip() for ln in reversed((output or "").splitlines()) if ln.strip()), "")[:180]


def kill_tree(proc) -> None:
    """Stops ytc together with its helpers (python, yt-dlp, Whisper). Killing
    only ytc would leave them running and holding the temporary audio file."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30)
        else:
            proc.kill()
    except (OSError, subprocess.SubprocessError):
        pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def run_ytc_streams(args, timeout):
    """Runs ytc with the daemon's database. Returns (exit code, stdout,
    stderr); a timeout or launch error is exit code -1 with the reason in
    stderr."""
    cmd = [str(YTC_BIN), "--db", str(DB_PATH), *args]
    try:
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        return -1, "", f"could not start ytc: {e}"
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        try:
            proc.communicate(timeout=30)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
        return -1, "", f"timed out after {timeout}s"
    return proc.returncode, out or "", err or ""


def run_ytc(args, timeout):
    """Runs ytc with the daemon's database. Returns (exit code, output);
    a timeout or launch error is exit code -1."""
    code, out, err = run_ytc_streams(args, timeout)
    return code, out + err


def is_timeout(code: int, message: str) -> bool:
    return code == -1 and "timed out" in (message or "")


def is_active_crawl_running():
    """Checks if another ytc crawl process is active."""
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name like '%ytc%' and CommandLine like '%crawl%'\" | Measure-Object | Select-Object -ExpandProperty Count"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        count = int(res.stdout.strip() or "0")
        return count > 0
    except Exception:
        return False


def sync_wiki(runner=run_ytc) -> bool:
    """Regenerates the Markdown wiki in <repo>/wiki. True only on exit code 0."""
    log("📚 Synchronizing Knowledge Wiki...")
    code, out = runner(["wiki", "--output", str(BASE_DIR / "wiki")], 120)
    if code == 0:
        log("✓ Wiki synchronized.")
        return True
    log(f"⚠️ Wiki sync FAILED (exit code {code}): {last_line(out)}")
    return False


def get_pending_videos(skip_ids=()):
    """Up to 10 videos waiting for analysis, newest first."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)
    c = conn.cursor()
    query = """
        SELECT v.id, v.title, v.channel_id, COALESCE(c.title, 'Direct Ingest')
        FROM videos v
        LEFT JOIN channels c ON v.channel_id = c.id
        WHERE v.status = 'pending'
          AND (c.status IN ('active', 'high_signal') OR c.status IS NULL)
          AND LOWER(v.title) NOT LIKE '%#shorts%'
          AND LOWER(v.title) NOT LIKE '%#short%'
          AND LOWER(v.title) NOT LIKE '%shorts%'
        ORDER BY v.published_at DESC
        LIMIT 50
    """
    c.execute(query)
    rows = [r for r in c.fetchall() if r[0] not in skip_ids][:10]
    conn.close()
    return rows


def get_channels_to_crawl(limit=5):
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)
    c = conn.cursor()
    # Prioritize channels never crawled or oldest crawled
    query = """
        SELECT id, title
        FROM channels
        WHERE status IN ('active', 'high_signal')
        ORDER BY last_crawled_at ASC NULLS FIRST
        LIMIT ?
    """
    c.execute(query, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def process_video_item(vid_id, title, channel_name, runner=run_ytc_streams, counts=None) -> str:
    """Runs the pipeline for one video and returns one of the outcome
    constants. Never writes to the database: ytc owns the video status.
    With `counts` (see new_counts) a processed video is added to its group."""
    global CURRENT_DELAY
    target_url = f"https://www.youtube.com/watch?v={vid_id}"
    log(f"🎯 Ingesting video: '{title}' ({vid_id}) [{channel_name}]...")

    start_t = time.time()
    code, out, err = runner(["process-video", "--json", target_url], PROCESS_TIMEOUT_S)
    elapsed = time.time() - start_t
    if is_timeout(code, err):
        log(f"⚠️ Video processing stopped after {PROCESS_TIMEOUT_S}s: {title}")
        return TIMEOUT

    outcome = judge_process_video(code, out, err, vid_id)
    if outcome == SUCCESS:
        result = parse_result(out, vid_id) or {}
        if counts is not None:
            counts[result_category(result)] += 1
        log(f"✓ Processed in {elapsed:.1f}s (exit code 0, result stored): {title}")
        log(f"   Result: {describe_result(result)}")
        if CURRENT_DELAY > INITIAL_DELAY:
            CURRENT_DELAY = max(INITIAL_DELAY, int(CURRENT_DELAY * 0.8))
            log(f"📉 Relaxed request delay to {CURRENT_DELAY}s.")
    elif outcome == NO_RESULT:
        log(f"⚠️ ytc exited with 0 but printed no result for {vid_id}; not counted.")
    elif outcome == LLM_UNAVAILABLE:
        # ytc left the video unprocessed on purpose: keep it pending and wait
        # for the LLM engine instead of marking the queue as failed.
        log("⏸ LLM engine is not reachable. Video stays pending; waiting 60s...")
        time.sleep(60)
    elif outcome == RATE_LIMITED:
        CURRENT_DELAY = min(MAX_BACKOFF, int(CURRENT_DELAY * 1.5))
        log(f"⏳ Rate limited by YouTube. Video stays pending; delay increased to {CURRENT_DELAY}s.")
    else:
        log(f"⚠️ Processing FAILED (exit code {code}): {last_line(err)}")
    return outcome


def needs_retry_later(outcome: str) -> bool:
    """Outcomes after which the same video would most likely fail again right
    away. It is postponed, never dropped."""
    return outcome in (NO_RESULT, FAILED, TIMEOUT)


def crawl_timeout_s(videos: int, delay: float) -> int:
    """Limit for one `ytc crawl` run that may analyse up to `videos` videos.
    Derived from the per-video limit: a fixed few minutes would stop the crawl
    in the middle of an LLM run, throw the work away and leave the channel
    marked as not crawled."""
    return _ytc_paths.ytc_timeout_s(videos, delay)


def crawl_channel(channel_id, channel_title, runner=run_ytc) -> bool:
    """True only when `ytc crawl` exited with code 0."""
    log(f"📡 Crawling RSS feed for: {channel_title} ({channel_id})...")
    limit = crawl_timeout_s(CRAWL_VIDEOS_PER_CHANNEL, CURRENT_DELAY)
    code, out = runner(["crawl", "--channel", channel_id, "--limit", str(CRAWL_VIDEOS_PER_CHANNEL),
                        "--delay", str(CURRENT_DELAY)], limit)
    if code == 0:
        return True
    if is_timeout(code, out):
        # The supervisor's own timer fired; that says nothing about the video.
        # ytc was stopped mid-run: the video it was reading is still pending
        # and is read through process-video in the next cycle.
        log(f"⚠️ Crawl of {channel_title} was stopped by this script after {limit}s (not a failure of "
            f"a video; whatever was being read stays pending and is picked up next cycle).")
        return False
    log(f"⚠️ Crawl FAILED for {channel_title} (exit code {code}): {last_line(out)}")
    return False


def self_test() -> int:
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_words_do_not_make_a_success(self):
            self.assertEqual(classify_outcome(1, "KEPT High Signal TITLE: x", True), FAILED)
            self.assertEqual(classify_outcome(0, "KEPT High Signal", False), NO_RESULT)
            self.assertEqual(classify_outcome(0, "", True), SUCCESS)

        def test_failure_reasons(self):
            self.assertEqual(classify_outcome(1, "Error: LLM engine not available", False), LLM_UNAVAILABLE)
            self.assertEqual(classify_outcome(1, "HTTP 429 Too Many Requests", False), RATE_LIMITED)
            self.assertEqual(classify_outcome(1, "IpBlocked", True), RATE_LIMITED)
            self.assertEqual(classify_outcome(2, "no transcript", False), FAILED)

        def test_success_needs_the_json_result_for_this_video(self):
            good = json.dumps({"video_id": "v1", "slop_score": 20, "is_worth_keeping": True})
            self.assertEqual(judge_process_video(0, good + "\n", "progress text", "v1"), SUCCESS)
            self.assertEqual(judge_process_video(0, "", "", "v1"), NO_RESULT)
            self.assertEqual(judge_process_video(0, "KEPT High Signal", "", "v1"), NO_RESULT)
            self.assertEqual(judge_process_video(0, good, "", "other"), NO_RESULT)
            self.assertEqual(judge_process_video(0, "[1, 2]", "", "v1"), NO_RESULT)
            # A non-zero exit code is never a success, whatever was printed.
            self.assertEqual(judge_process_video(1, good, "Error: no transcript", "v1"), FAILED)
            self.assertEqual(judge_process_video(1, "", "Error: LLM engine not available (x)", "v1"), LLM_UNAVAILABLE)
            self.assertEqual(judge_process_video(1, "", "HTTP Error 429: Too Many Requests", "v1"), RATE_LIMITED)

        def test_processed_videos_are_sorted_into_the_right_group(self):
            base = {"video_id": "v1", "engine": "llm", "not_scored": False, "slop_score": 20,
                    "is_worth_keeping": True, "topic_verdict": "ok", "info_status": "sufficient",
                    "excluded": False, "sections_total": 3}
            self.assertEqual(result_category(base), KEPT)
            self.assertEqual(result_category(dict(base, is_worth_keeping=False, slop_score=80)), BELOW_CUTOFF)
            politics = dict(base, is_worth_keeping=False, topic_verdict="politics", topic_reason="election talk")
            self.assertEqual(result_category(politics), OFF_TOPIC)
            self.assertIn("off-topic: politics (election talk)", describe_result(politics))
            # Not enough information: not scored, the score is null.
            short = dict(base, not_scored=True, slop_score=None, is_worth_keeping=False, topic_verdict=None,
                         info_status="insufficient", info_reason="only 40 words in 31 seconds",
                         sections_total=None)
            self.assertEqual(result_category(short), INSUFFICIENT)
            self.assertEqual(describe_result(short), "not enough information (only 40 words in 31 seconds)")
            # Disqualified by hand wins over a good score and an accepted topic.
            removed = dict(base, is_worth_keeping=False, excluded=True, excluded_reason="duplicate")
            self.assertEqual(result_category(removed), DISQUALIFIED)
            self.assertIn("read: whole transcript (3 sections)", describe_result(base))
            self.assertIn("read: whole transcript (1 section)", describe_result(dict(base, sections_total=1)))
            # A heuristic row records no verdict; it has a (rough) score and is
            # judged by the cutoff like any other.
            heuristic = dict(base, engine="heuristic", topic_verdict=None, info_status=None, slop_score=70,
                             is_worth_keeping=False)
            self.assertEqual(result_category(heuristic), BELOW_CUTOFF)
            self.assertEqual(result_category(dict(heuristic, slop_score=30, is_worth_keeping=True)), KEPT)

        def test_groups_are_counted_per_processed_video(self):
            counts = new_counts()
            answers = iter([
                {"video_id": "v1", "is_worth_keeping": True, "slop_score": 10},
                {"video_id": "v1", "is_worth_keeping": False, "info_status": "insufficient", "not_scored": True},
                {"video_id": "v1", "is_worth_keeping": False, "topic_verdict": "nonsense", "slop_score": 70},
            ])
            for _ in range(3):
                process_video_item("v1", "t", "c", runner=lambda a, t: (0, json.dumps(next(answers)), ""),
                                   counts=counts)
            # A failed run is in no group.
            process_video_item("v1", "t", "c", runner=lambda a, t: (1, "", "Error: boom"), counts=counts)
            self.assertEqual(counts, {KEPT: 1, BELOW_CUTOFF: 0, OFF_TOPIC: 1, INSUFFICIENT: 1, DISQUALIFIED: 0})
            self.assertEqual(counts_line(counts),
                             "1 kept, 0 below the quality cutoff, 1 off-topic (fringe / nonsense / politics), "
                             "1 not enough information, 0 disqualified by hand")

        def test_database_and_binary_follow_the_one_rule(self):
            self.assertEqual(_ytc_paths.default_db({}), BASE_DIR / "data" / "ytc.db")
            self.assertEqual(_ytc_paths.default_db({"YTC_DB": str(BASE_DIR / "x.db")}), BASE_DIR / "x.db")

        def test_failed_videos_are_postponed_not_dropped(self):
            now = [1000.0]
            retry = RetryLater(first=60, longest=200, clock=lambda: now[0])
            self.assertEqual(retry.postpone("v1"), 60)
            self.assertEqual(retry.waiting(), {"v1"})
            now[0] += 61
            self.assertEqual(retry.waiting(), set())        # eligible again
            self.assertEqual(retry.postpone("v1"), 120)     # second failure: longer pause
            self.assertEqual(retry.postpone("v1"), 200)     # capped
            self.assertEqual(retry.postpone("v1"), 200)
            retry.clear("v1")
            self.assertEqual(retry.waiting(), set())
            self.assertEqual(retry.postpone("v1"), 60)
            for outcome in (NO_RESULT, FAILED, TIMEOUT):
                self.assertTrue(needs_retry_later(outcome))
            for outcome in (SUCCESS, LLM_UNAVAILABLE, RATE_LIMITED):
                self.assertFalse(needs_retry_later(outcome))

        def test_queue_holds_pending_videos_only(self):
            global DB_PATH
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "t.db"
                conn = sqlite3.connect(db)
                conn.execute("CREATE TABLE channels(id TEXT PRIMARY KEY, title TEXT, status TEXT)")
                conn.execute("CREATE TABLE videos(id TEXT PRIMARY KEY, channel_id TEXT, title TEXT, "
                             "status TEXT, published_at TEXT)")
                conn.execute("INSERT INTO channels VALUES('c1', 'Chan', 'active')")
                conn.executemany("INSERT INTO videos VALUES(?, 'c1', ?, ?, ?)", [
                    ("p1", "pending one", "pending", "2026-01-03"),
                    ("p2", "pending two", "pending", "2026-01-02"),
                    ("d1", "done", "processed", "2026-01-01"),
                    ("f1", "given up by ytc", "failed", "2026-01-01"),
                ])
                conn.commit()
                conn.close()
                saved, DB_PATH = DB_PATH, db
                try:
                    self.assertEqual([r[0] for r in get_pending_videos()], ["p1", "p2"])
                    self.assertEqual([r[0] for r in get_pending_videos({"p1"})], ["p2"])
                finally:
                    DB_PATH = saved
                # Nothing above wrote to the database.
                conn = sqlite3.connect(db)
                self.assertEqual(conn.execute("SELECT status FROM videos WHERE id='p2'").fetchone()[0], "pending")
                conn.close()

        def test_wiki_and_crawl_report_the_exit_code(self):
            self.assertTrue(sync_wiki(runner=lambda a, t: (0, "")))
            self.assertFalse(sync_wiki(runner=lambda a, t: (1, "Error: disk full")))
            self.assertTrue(crawl_channel("UC1", "X", runner=lambda a, t: (0, "")))
            self.assertFalse(crawl_channel("UC1", "X", runner=lambda a, t: (-1, "timed out after 360s")))

        def test_time_limits_fit_whole_transcript_analysis(self):
            calls = []

            def ok(args, timeout):
                calls.append((args, timeout))
                return 0, ""

            self.assertTrue(crawl_channel("UC1", "X", runner=ok))
            args, timeout = calls[0]
            self.assertEqual(args[args.index("--limit") + 1], str(CRAWL_VIDEOS_PER_CHANNEL))
            # Room for every video the crawl may analyse, each a whole transcript.
            self.assertGreaterEqual(timeout, CRAWL_VIDEOS_PER_CHANNEL * PROCESS_TIMEOUT_S)
            self.assertEqual(timeout, crawl_timeout_s(CRAWL_VIDEOS_PER_CHANNEL, CURRENT_DELAY))
            self.assertGreaterEqual(PROCESS_TIMEOUT_S, 30 * 60)
            self.assertGreater(crawl_timeout_s(3, 20), crawl_timeout_s(2, 20))

        def test_the_scripts_own_timer_never_fails_a_video_for_good(self):
            # A run stopped by this script: the video is only postponed (in
            # memory), and nothing here can write a status to the database.
            outcome = process_video_item("v1", "t", "c", runner=lambda a, t: (-1, "", f"timed out after {t}s"))
            self.assertEqual(outcome, TIMEOUT)
            self.assertTrue(needs_retry_later(outcome))
            source = Path(__file__).read_text(encoding="utf-8")
            for write in ("UPDATE " + "videos", "UPDATE " + "insights", "mode=" + "rw"):
                self.assertNotIn(write, source)

        def test_process_video_uses_json_mode_and_never_touches_the_database(self):
            calls = []

            def ok(args, timeout):
                calls.append((args, timeout))
                return 0, json.dumps({"video_id": "v1"}), "progress"

            self.assertEqual(process_video_item("v1", "t", "c", runner=ok), SUCCESS)
            self.assertEqual(calls[0][0][:2], ["process-video", "--json"])
            self.assertEqual(calls[0][1], PROCESS_TIMEOUT_S)
            self.assertGreaterEqual(PROCESS_TIMEOUT_S, 1800)
            self.assertEqual(process_video_item("v1", "t", "c", runner=lambda a, t: (0, "KEPT", "")), NO_RESULT)
            self.assertEqual(process_video_item("v1", "t", "c", runner=lambda a, t: (1, "", "Error: boom")), FAILED)
            self.assertEqual(process_video_item(
                "v1", "t", "c", runner=lambda a, t: (-1, "", f"timed out after {t}s")), TIMEOUT)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main():
    require_db_and_binary()
    log("======================================================================")
    log("🚀 YTC supervisor loop started (pending videos -> channel crawl -> wiki)")
    log(f"   Database: {DB_PATH}")
    log(f"   Base Pacing Delay: {CURRENT_DELAY}s")
    log("======================================================================")

    processed_count = 0
    counts = new_counts()  # what became of the processed videos
    retry = RetryLater()  # videos that failed or ran too long; tried again later

    while True:
        try:
            # 1. If a batch crawl is in progress, yield and monitor
            if is_active_crawl_running():
                log("⏳ Background batch crawl in progress. Monitoring queue in 30s...")
                time.sleep(30)
                continue

            # 2. Check pending queue
            pending = get_pending_videos(retry.waiting())
            if pending:
                log(f"📋 Found {len(pending)} pending video(s) in queue. Beginning polite ingestion cycle...")
                for vid_id, title, ch_id, ch_name in pending:
                    outcome = process_video_item(vid_id, title, ch_name, counts=counts)
                    if outcome == SUCCESS:
                        retry.clear(vid_id)
                        processed_count += 1
                        if processed_count % 3 == 0:
                            sync_wiki()
                    elif needs_retry_later(outcome):
                        pause = retry.postpone(vid_id)
                        log(f"   Video {vid_id} stays pending; next try in about {pause // 60} minutes.")
                    time.sleep(CURRENT_DELAY)
                continue

            # 3. If queue empty, refresh channels with oldest crawl timestamp
            channels = get_channels_to_crawl(limit=3)
            crawled_ok = 0
            for ch_id, ch_name in channels:
                if crawl_channel(ch_id, ch_name):
                    crawled_ok += 1
                time.sleep(10)
            if channels:
                sync_wiki()

            log(f"💤 Queue empty. Channel crawls this cycle: {crawled_ok} of {len(channels)} succeeded. "
                f"Videos processed since start: {processed_count} ({counts_line(counts)}). "
                f"Next cycle in 60s...")
            time.sleep(60)

        except Exception as e:
            log(f"🚨 Supervisor cycle exception (retrying in 30s): {e}")
            time.sleep(30)


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    main()
