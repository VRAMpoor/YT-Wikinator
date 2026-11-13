#!/usr/bin/env python3
"""
YTC adaptive ingestion worker: one pending video at a time, with a pause that
grows when YouTube rate-limits and shrinks again after clean runs.

A video counts as processed only when `ytc process-video --json` exited with
code 0 AND printed the stored result for it; a crawl or wiki run counts as
successful only on exit code 0. The message text is used solely to tell why
a run failed (rate limit, LLM not reachable). The judging logic is shared with
continuous_autonomous_daemon.py.

The database is only read here: this script never writes a video status. A
video that failed or ran too long stays pending and is tried again later.

A processed video is logged with what became of it (kept, below the quality
cutoff, off-topic, not enough information, disqualified by hand).

It does not search for new channels; when nothing is pending it runs
`ytc crawl` over the tracked channels.

Database: YTC_DB, else <app folder>/data/ytc.db (the rule ytc itself uses).
Binary:   YTC_BIN, else target/release/ytc(.exe), else ytc(.exe) in the app folder.
Built-in checks:  python scripts/adaptive_continuous_worker.py --self-test
"""

import sys
import time
import sqlite3
from pathlib import Path

# Force UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ytc_paths  # noqa: E402
from continuous_autonomous_daemon import (  # noqa: E402
    DB_PATH, FAILED, LLM_UNAVAILABLE, NO_RESULT, PROCESS_TIMEOUT_S, RATE_LIMITED, SUCCESS,
    RetryLater, counts_line, describe_result, is_timeout, judge_process_video, last_line, log,
    new_counts, parse_result, require_db_and_binary, result_category, run_ytc, run_ytc_streams, sync_wiki,
)

INITIAL_DELAY_SECS = 25  # Polite base pacing
MAX_BACKOFF_SECS = 480   # Max 8-minute cooldown on 429
CURRENT_PACING = INITIAL_DELAY_SECS

# `ytc crawl --limit 1` goes over ALL tracked channels and analyses up to one
# new video per channel, each read in full. Its limit therefore grows with the
# number of channels. The cap keeps a guard against a hung process; a crawl
# stopped at the cap loses at most the one video being read, which stays
# pending and is then processed on its own.
FEED_CRAWL_MAX_S = 12 * 60 * 60


def count_crawled_channels() -> int:
    """How many channels `ytc crawl` visits (muted ones are skipped). Read-only;
    0 when it cannot be found out."""
    try:
        conn = sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)
    except (OSError, ValueError, sqlite3.Error):
        return 0
    try:
        row = conn.execute("SELECT COUNT(*) FROM channels WHERE status IN ('active', 'high_signal')").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def feed_crawl_timeout_s(channels: int, pacing: float) -> int:
    """Limit for the crawl over all channels: one whole video per channel plus
    the pause, never less than one video, never more than FEED_CRAWL_MAX_S."""
    return min(FEED_CRAWL_MAX_S, _ytc_paths.ytc_timeout_s(max(1, channels), pacing))


def get_next_unprocessed_video(skip_ids=()):
    """Next pending video from tracked channels, newest first."""
    conn = sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()

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
    cursor.execute(query)
    row = next((r for r in cursor.fetchall() if r[0] not in skip_ids), None)
    conn.close()
    return row


def main():
    global CURRENT_PACING
    require_db_and_binary()
    log("🚀 Starting Adaptive Ingestion Worker...")
    log(f"📁 Target Database: {DB_PATH}")
    log(f"⏱ Initial Request Pacing: {CURRENT_PACING}s")

    consecutive_successes = 0
    counts = new_counts()  # what became of the processed videos
    retry = RetryLater()  # videos that failed or ran too long; tried again later

    while True:
        target = get_next_unprocessed_video(retry.waiting())
        if not target:
            log("No pending videos in queue. Checking for channel feed updates...")
            limit = feed_crawl_timeout_s(count_crawled_channels(), CURRENT_PACING)
            code, out = run_ytc(["crawl", "--limit", "1", "--delay", str(CURRENT_PACING)], limit)
            if code == 0:
                log("✓ Feed crawl finished (exit code 0).")
            elif is_timeout(code, out):
                # This script's own timer, not a verdict on any video: what
                # ytc was reading stays pending and is processed on its own.
                log(f"⌛ Feed crawl was stopped by this script after {limit}s; "
                    f"a video it was reading stays pending.")
            else:
                log(f"⚠ Feed crawl FAILED (exit code {code}): {last_line(out)}")
            time.sleep(60)
            continue

        vid_id, title, channel_id, channel_name = target
        log(f"🎯 Ingesting video: '{title}' ({vid_id}) from {channel_name}...")

        # Pass full URL so IDs with leading dashes (e.g. -AbCdEfGh12) never conflict with CLI flags
        video_target = f"https://www.youtube.com/watch?v={vid_id}"
        start_t = time.time()
        code, stdout, stderr = run_ytc_streams(["process-video", "--json", video_target], PROCESS_TIMEOUT_S)
        elapsed = time.time() - start_t

        if is_timeout(code, stderr):
            pause = retry.postpone(vid_id)
            log(f"⌛ Video {vid_id} was stopped after {PROCESS_TIMEOUT_S}s. It stays pending; "
                f"next try in about {pause // 60} minutes.")
        else:
            outcome = judge_process_video(code, stdout, stderr, vid_id)
            if outcome == SUCCESS:
                result = parse_result(stdout, vid_id) or {}
                counts[result_category(result)] += 1
                log(f"✓ Processed in {elapsed:.1f}s (exit code 0, result stored): {title}")
                log(f"   Result: {describe_result(result)}")
                log(f"   Since start: {counts_line(counts)}")
                retry.clear(vid_id)
                consecutive_successes += 1

                # If running smoothly, gradually ease pacing down to base rate
                if CURRENT_PACING > INITIAL_DELAY_SECS and consecutive_successes >= 2:
                    CURRENT_PACING = max(INITIAL_DELAY_SECS, int(CURRENT_PACING * 0.75))
                    log(f"📉 Relaxing pacing delay to {CURRENT_PACING}s after clean responses.")
                    consecutive_successes = 0

                sync_wiki()
            else:
                consecutive_successes = 0
                if outcome == NO_RESULT:
                    pause = retry.postpone(vid_id)
                    log(f"⚠ ytc exited with 0 but printed no result for {vid_id}; not counted, "
                        f"next try in about {pause // 60} minutes.")
                elif outcome == LLM_UNAVAILABLE:
                    # ytc left the video unprocessed on purpose: keep it pending and
                    # wait for the LLM engine instead of marking the queue as failed.
                    log("⏸ LLM engine is not reachable. Video stays pending; waiting 60s...")
                    time.sleep(60)
                elif outcome == RATE_LIMITED:
                    CURRENT_PACING = min(MAX_BACKOFF_SECS, int(CURRENT_PACING * 2.0))
                    log(f"🛑 Rate limited by YouTube. Video stays pending; pacing increased to {CURRENT_PACING}s...")
                elif outcome == FAILED:
                    # ytc owns the video status; here the video is only postponed so
                    # that the loop does not hammer the same failure.
                    pause = retry.postpone(vid_id)
                    log(f"⚠ Processing FAILED (exit code {code}): {last_line(stderr)}. "
                        f"Video stays pending; next try in about {pause // 60} minutes.")

        # Polite non-RSS pacing delay
        log(f"⏱ Pacing pause ({CURRENT_PACING}s) before next YouTube non-RSS request...")
        time.sleep(CURRENT_PACING)


def self_test() -> int:
    """Checks of the logic that needs no ytc, no network and no real database."""
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_feed_crawl_limit_fits_whole_transcript_analysis(self):
            one = PROCESS_TIMEOUT_S
            self.assertGreaterEqual(one, 30 * 60)
            # Never shorter than one whole video, also when the count is unknown (0).
            self.assertGreaterEqual(feed_crawl_timeout_s(0, 25), one)
            self.assertGreaterEqual(feed_crawl_timeout_s(1, 25), one)
            # One video per channel.
            self.assertGreaterEqual(feed_crawl_timeout_s(5, 25), 5 * one)
            # Bounded, and the bound still holds many videos.
            self.assertEqual(feed_crawl_timeout_s(10_000, 25), FEED_CRAWL_MAX_S)
            self.assertGreaterEqual(FEED_CRAWL_MAX_S, 10 * one)

        def test_channel_count_is_read_only_and_never_raises(self):
            global DB_PATH
            saved = DB_PATH
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    DB_PATH = Path(tmp) / "missing.db"
                    self.assertEqual(count_crawled_channels(), 0)
                    self.assertFalse(DB_PATH.exists())               # not created
                    DB_PATH = Path(tmp) / "t.db"
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("CREATE TABLE channels(id TEXT PRIMARY KEY, title TEXT, status TEXT)")
                    conn.executemany("INSERT INTO channels VALUES(?, 't', ?)",
                                     [("a", "active"), ("b", "high_signal"), ("c", "muted")])
                    conn.commit()
                    conn.close()
                    self.assertEqual(count_crawled_channels(), 2)
                    DB_PATH = Path(tmp) / "broken.db"
                    DB_PATH.write_bytes(b"not a database " * 50)
                    self.assertEqual(count_crawled_channels(), 0)
                finally:
                    DB_PATH = saved

        def test_this_script_never_writes_a_video_status(self):
            source = Path(__file__).read_text(encoding="utf-8")
            for write in ("UPDATE " + "videos", "INSERT " + "INTO videos", "DELETE " + "FROM"):
                self.assertNotIn(write, source)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    main()
