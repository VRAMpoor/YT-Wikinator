#!/usr/bin/env python3
"""
Add channels to the watch list through the binary, one handle per line.

For every handle in the given text file this runs

    ytc --db <db> add-channel @handle

so the channel id and title are resolved by YouTube (rss::resolve_channel), the
channel starts as 'active' with NO slop score, and its score and status are
later computed by the binary from the videos it really analysed.

This script writes nothing to the database itself.

Handles file: one @handle, channel URL or UC... id per line; blank lines and
lines starting with # are ignored.

Database: --db, else YTC_DB, else <app folder>/data/ytc.db.
Binary:   YTC_BIN, else target/release/ytc(.exe), else ytc(.exe) in the app folder.

Usage:
    python scripts/register_admitted_channels.py handles.txt [--db PATH] [--delay 3]
    python scripts/register_admitted_channels.py --self-test
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_single_video import YtcError, resolve_db_path, run_ytc  # noqa: E402

Runner = Callable[[Sequence[str], Path], Tuple[int, str]]


def read_handles(path: Path) -> List[str]:
    """Handles from a text file, in order, without duplicates."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as e:
        raise YtcError(f"Cannot read handles file {path}: {e}") from e
    seen, handles = set(), []
    for line in lines:
        item = line.split("#", 1)[0].strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            handles.append(item)
    return handles


def register(handles: Sequence[str], db_path: Path, delay: float = 3.0,
             runner: Runner = run_ytc) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Returns (handles added, [(handle, reason)] for the ones that failed).
    A channel counts as added only when the binary exits with 0."""
    added: List[str] = []
    failed: List[Tuple[str, str]] = []
    for n, handle in enumerate(handles):
        if n and delay > 0:
            time.sleep(delay)  # each add-channel is one request to YouTube
        code, output = runner(["add-channel", handle], db_path)
        last_line = next((ln.strip() for ln in reversed(output.splitlines()) if ln.strip()), "")
        if code == 0:
            added.append(handle)
            print(f"  + {handle}: {last_line}")
        else:
            failed.append((handle, f"exit code {code}: {last_line}"))
            print(f"  ! {handle}: FAILED (exit code {code}) {last_line}")
    return added, failed


def self_test() -> int:
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_read_handles(self):
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / "h.txt"
                f.write_text("# my list\n@One\n\n@two  # comment\n@one\nhttps://www.youtube.com/@Three\n",
                             encoding="utf-8")
                self.assertEqual(read_handles(f), ["@One", "@two", "https://www.youtube.com/@Three"])
                with self.assertRaises(YtcError):
                    read_handles(Path(tmp) / "missing.txt")

        def test_only_exit_code_zero_counts(self):
            calls = []

            def runner(args, db_path):
                calls.append(list(args))
                return (1, "Error: channel not found") if args[1] == "@bad" else (0, "✓ Added channel: X (UC123)")
            added, failed = register(["@good", "@bad"], Path("x.db"), delay=0, runner=runner)
            self.assertEqual(added, ["@good"])
            self.assertEqual(failed[0][0], "@bad")
            self.assertIn("exit code 1", failed[0][1])
            # nothing but add-channel is ever sent: no scores, no statuses
            self.assertEqual(calls, [["add-channel", "@good"], ["add-channel", "@bad"]])

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    ap = argparse.ArgumentParser(description="Add channels via 'ytc add-channel', handles taken from a text file.")
    ap.add_argument("handles_file", nargs="?", type=Path, help="text file with one @handle / URL / channel id per line")
    ap.add_argument("--db", help="database file (default: YTC_DB, else <app folder>/data/ytc.db)")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between channels (default 3)")
    ap.add_argument("--self-test", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.handles_file:
        ap.error("a handles file is required")
    try:
        handles = read_handles(args.handles_file)
        if not handles:
            raise YtcError(f"{args.handles_file} contains no handles.")
        db_path = resolve_db_path(args.db)
        print(f"Adding {len(handles)} channel(s) to {db_path} through 'ytc add-channel'...")
        added, failed = register(handles, db_path, delay=args.delay)
    except YtcError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"\nAdded: {len(added)}   Failed: {len(failed)}")
    print("Channels start as 'active' without a score; scores come from crawled videos (ytc crawl).")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
