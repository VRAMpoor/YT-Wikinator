"""Read a video's transcript from the YouTube page in a real browser.

The first way src/extractor.rs tries for every video. Opens the watch page in
the installed Microsoft Edge (or Playwright's own Chromium), presses YouTube's
own "Show transcript" button and reads the lines the page then shows. The
video is not played.

Usage: fetch_browser_transcript.py <video_id> [--show] [--check]
  --show   open a visible browser window (default: no window)
  --check  only say whether this way is set up: exit 0 or 4, nothing is opened

Exit codes (read by src/extractor.rs):
  0  success; stdout is JSON: {"duration_sec": <video length, 0 if the page
     names none>, "cues": [...]}. An empty cue list means the page offers no
     transcript, or only one in a non-Latin script.
  1  unexpected failure; reason on stderr.
  3  YouTube asked for a sign-in or a bot check instead of showing the page;
     ytc treats this like a block: the video waits.
  4  this way is not set up (Python package 'playwright' or a browser is
     missing); reason on stderr.
"""
import json
import re
import sys
from pathlib import Path

# The browser keeps its cookies here, so the consent answer is given once.
PROFILE_DIR_NAME = "browser_profile"
PAGE_TIMEOUT_MS = 30_000
PANEL_TIMEOUT_MS = 20_000

# Same script test as fetch_builtin_transcript.py.
FOREIGN_SCRIPT_REGEX = re.compile(
    r'[؀-ۿݐ-ݿࢠ-ࣿऀ-ॿ฀-๿Ѐ-ӿ一-鿿぀-ヿ가-힯]'
)

# Runs in the page. Returns what the page shows; clicks only YouTube's own
# consent "reject" and "Show transcript" buttons.
STATE_JS = r"""
() => {
  const text = (document.body && document.body.innerText || '').slice(0, 4000);
  const player = window.ytInitialPlayerResponse || {};
  const status = (player.playabilityStatus || {});
  const tracks = ((player.captions || {}).playerCaptionsTracklistRenderer || {}).captionTracks || [];
  return {
    host: location.host,
    path: location.pathname,
    status: status.status || '',
    reason: status.reason || '',
    tracks: tracks.length,
    duration: Number((player.videoDetails || {}).lengthSeconds || 0),
    bot: /confirm you.re not a bot|unusual traffic|not a robot/i.test(text + ' ' + (status.reason || '')),
  };
}
"""

REJECT_CONSENT_JS = r"""
() => {
  const wanted = /reject all|reject the use|alle ablehnen|refuse|tout refuser/i;
  for (const b of document.querySelectorAll('button, [role="button"], input[type="submit"]')) {
    const label = (b.getAttribute('aria-label') || '') + ' ' + (b.innerText || b.value || '');
    if (wanted.test(label)) { b.click(); return true; }
  }
  return false;
}
"""

OPEN_PANEL_JS = r"""
() => {
  const v = document.querySelector('video');
  if (v) { try { v.muted = true; v.pause(); } catch (e) {} }
  const direct = document.querySelector(
    'ytd-video-description-transcript-section-renderer button, button[aria-label="Show transcript"]');
  if (direct) { direct.click(); return 'clicked'; }
  const expand = document.querySelector('#description-inline-expander #expand, tp-yt-paper-button#expand');
  if (expand) { expand.click(); return 'expanded'; }
  return 'waiting';
}
"""

# Reads the lines of the open transcript panel: new and old page layout.
READ_SEGMENTS_JS = r"""
() => {
  const stamp = /^\d+:\d{2}(:\d{2})?$/;
  const out = [];
  const nodes = document.querySelectorAll(
    'transcript-segment-view-model, .ytwTranscriptSegmentViewModelHost, ytd-transcript-segment-renderer');
  for (const seg of nodes) {
    if (seg.offsetParent === null) continue;
    let time = '', text = '';
    const t = seg.querySelector('.ytwTranscriptSegmentViewModelTimestamp, .segment-timestamp');
    if (t) time = t.textContent.trim();
    const x = seg.querySelector('.segment-text, .ytAttributedStringHost[role="text"], span[role="text"], .ytAttributedStringHost');
    if (x) text = x.textContent.trim();
    if (!time) {
      for (const c of seg.querySelectorAll('div, span')) {
        const s = c.textContent.trim();
        if (stamp.test(s)) { time = s; break; }
      }
    }
    if (text) out.push([time, text]);
  }
  return out;
}
"""


def is_foreign_script(text: str) -> bool:
    matches = FOREIGN_SCRIPT_REGEX.findall(text or "")
    return len(matches) > 5 and len(matches) / max(len(text), 1) > 0.05


def stamp_to_seconds(stamp: str):
    """'1:02:03' -> 3723.0; None when the text is no time stamp."""
    parts = stamp.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + int(p)
    return float(seconds)


def to_cues(rows, duration_sec: float):
    """Page lines -> cues. The page shows only start times, to the second: a
    line ends where the next one starts. The last line gets a few seconds and
    is never stretched to the end of the video: ytc compares the end of the
    last line with the video length to see whether the captions stop early."""
    starts = []
    last = 0.0
    for stamp, _ in rows:
        start = stamp_to_seconds(stamp)
        if start is None or start < last:
            start = last
        starts.append(start)
        last = start
    cues = []
    for i, (_, text) in enumerate(rows):
        cleaned = " ".join(text.split())
        if not cleaned:
            continue
        if i + 1 < len(starts):
            end = max(starts[i + 1], starts[i])
        else:
            end = starts[i] + 5.0
            if duration_sec > 0:
                end = max(min(end, duration_sec), starts[i])
        cues.append({"start_sec": round(starts[i], 3), "end_sec": round(end, 3), "text": cleaned})
    return cues


def profile_dir() -> Path:
    here = Path(__file__).resolve().parent
    return here / "data" / PROFILE_DIR_NAME


def not_set_up(reason: str) -> None:
    sys.stderr.write(f"Browser transcripts are not set up: {reason}\n")
    sys.exit(4)


def launch(playwright, show: bool):
    """Installed Edge first (a browser every Windows has), then Playwright's Chromium."""
    folder = profile_dir()
    folder.mkdir(parents=True, exist_ok=True)
    options = dict(
        user_data_dir=str(folder),
        headless=not show,
        locale="en-US",
        viewport={"width": 1280, "height": 900},
        # No graphics card use: the card's memory belongs to the LLM.
        args=["--mute-audio", "--autoplay-policy=user-gesture-required", "--disable-gpu"],
    )
    errors = []
    for channel in ("msedge", None):
        try:
            if channel:
                return playwright.chromium.launch_persistent_context(channel=channel, **options)
            return playwright.chromium.launch_persistent_context(**options)
        except Exception as e:  # browser missing or profile in use
            errors.append(f"{channel or 'chromium'}: {str(e).splitlines()[0]}")
    not_set_up("no usable browser (" + "; ".join(errors) + ")")


def read_transcript(page, video_id: str):
    """Returns (rows, duration_sec). Exits with 3 on a bot check."""
    page.goto(f"https://www.youtube.com/watch?v={video_id}&hl=en", wait_until="domcontentloaded",
              timeout=PAGE_TIMEOUT_MS)
    state = {}
    opened = False
    waited = 0
    while waited < PAGE_TIMEOUT_MS + PANEL_TIMEOUT_MS:
        state = page.evaluate(STATE_JS)
        if state["host"].startswith("consent."):
            page.evaluate(REJECT_CONSENT_JS)
        elif state["bot"] or state["status"] == "LOGIN_REQUIRED":
            sys.stderr.write(
                "YouTube asked for a sign-in or a bot check in the browser (treated like 429): "
                f"{state['reason'] or 'no reason given'}\n")
            sys.exit(3)
        elif state["path"] == "/watch" and state["status"]:
            if state["status"] != "OK":
                sys.stderr.write(f"No transcript: the video is not playable ({state['reason'] or state['status']})\n")
                return [], 0.0
            if state["tracks"] == 0:
                sys.stderr.write("No transcript: the page lists no caption track\n")
                return [], float(state["duration"])
            rows = page.evaluate(READ_SEGMENTS_JS)
            if rows:
                # The panel fills in one go; one more look catches a late rest.
                page.wait_for_timeout(700)
                rows = page.evaluate(READ_SEGMENTS_JS)
                return rows, float(state["duration"])
            if not opened:
                opened = page.evaluate(OPEN_PANEL_JS) == "clicked"
        page.wait_for_timeout(500)
        waited += 500
    raise RuntimeError(
        "the transcript did not appear on the page in time "
        f"(page state: {state.get('host')}{state.get('path')}, button pressed: {opened})")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        not_set_up(f"Python package 'playwright' is missing ({type(e).__name__})")
    if "--check" in flags:
        return
    if not args:
        sys.stderr.write("usage: fetch_browser_transcript.py <video_id> [--show] [--check]\n")
        sys.exit(1)
    video_id = args[0]

    try:
        with sync_playwright() as p:
            context = launch(p, show="--show" in flags)
            try:
                page = context.pages[0] if context.pages else context.new_page()
                rows, duration = read_transcript(page, video_id)
            finally:
                context.close()
    except SystemExit:
        raise
    except Exception as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else ""
        sys.stderr.write(f"Browser transcript error: {type(e).__name__}: {first}\n")
        sys.exit(1)

    cues = to_cues(rows, duration)
    if is_foreign_script(" ".join(c["text"] for c in cues[:500])):
        sys.stderr.write(f"Rejected transcript for {video_id}: non-Latin script.\n")
        cues = []
    print(json.dumps({"duration_sec": duration, "cues": cues}))


if __name__ == "__main__":
    main()
