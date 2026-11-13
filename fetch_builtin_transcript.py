"""Fetch a video's built-in YouTube transcript and print it as JSON cues.

Exit codes (read by src/extractor.rs):
  0  success; stdout is a JSON list of cues. An empty list means the video
     has no captions, or only captions in a non-Latin script. English and
     German tracks are preferred; otherwise the track in the video's own
     language is returned untranslated.
  1  unexpected failure (missing module, network error, ...); reason on stderr.
  3  YouTube is blocking or rate-limiting this IP; reason on stderr. The text
     contains "IpBlocked" / "429" so the crawler backs off.
"""
import sys
import json
import re

# Non-Latin foreign script ranges (Arabic, Devanagari/Hindi/Sanskrit, Thai, Cyrillic, CJK, etc.)
FOREIGN_SCRIPT_REGEX = re.compile(
    r'[؀-ۿݐ-ݿࢠ-ࣿऀ-ॿ฀-๿Ѐ-ӿ一-鿿぀-ヿ가-힯]'
)

# Exception class names of youtube-transcript-api, matched by name so this works
# across library versions.
BLOCKED_ERRORS = {"IpBlocked", "RequestBlocked", "TooManyRequests"}
NO_CAPTIONS_ERRORS = {
    "TranscriptsDisabled", "NoTranscriptFound", "NoTranscriptAvailable",
    "VideoUnavailable", "VideoUnplayable", "AgeRestricted", "InvalidVideoId",
    "NotTranslatable", "TranslationLanguageNotAvailable",
}


def is_foreign_script(text: str) -> bool:
    """Returns True if the text contains a significant portion of non-Latin foreign scripts."""
    if not text:
        return False
    matches = FOREIGN_SCRIPT_REGEX.findall(text)
    # If more than 5 foreign characters or > 5% of string, reject
    if len(matches) > 5 and len(matches) / max(len(text), 1) > 0.05:
        return True
    return False


def class_names(exc: BaseException) -> set:
    return {c.__name__ for c in type(exc).__mro__}


def one_line(exc: BaseException) -> str:
    """First line of the error text: the library's messages run to many lines."""
    text = str(exc).strip()
    first = text.splitlines()[0].strip() if text else ""
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


def fail(exc: BaseException) -> None:
    """Report a real failure on stderr and exit non-zero."""
    names = class_names(exc)
    text = str(exc)
    if names & BLOCKED_ERRORS or "429" in text or "Too Many Requests" in text:
        sys.stderr.write(f"IpBlocked / rate limit (429) from YouTube: {one_line(exc)}\n")
        sys.exit(3)
    sys.stderr.write(f"Built-in transcript error: {one_line(exc)}\n")
    sys.exit(1)


def fetch_cues(video_id: str):
    """Returns the fetched transcript, or None when the video has no usable captions."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    try:
        transcript_list = list(api.list(video_id))
    except Exception as e:
        if class_names(e) & NO_CAPTIONS_ERRORS:
            sys.stderr.write(f"No captions: {one_line(e)}\n")
            return None
        raise

    def usable(tr) -> bool:
        code = getattr(tr, 'language_code', '').lower()
        return code.startswith('en') or code.startswith('de')

    # 1. Manually created English/German, 2. auto-generated English/German
    ordered = [t for t in transcript_list if usable(t) and not getattr(t, 'is_generated', True)]
    ordered += [t for t in transcript_list if usable(t) and getattr(t, 'is_generated', True)]
    last_error = None
    for tr in ordered:
        try:
            return tr.fetch()
        except Exception as e:
            if class_names(e) & BLOCKED_ERRORS:
                raise
            last_error = e

    # 3. A track in the video's own, other language, as it is. YouTube is not
    #    asked to translate it: its translation endpoint is limited far more
    #    strictly than the tracks themselves. (main() rejects other scripts.)
    others = [t for t in transcript_list if not usable(t)]
    others.sort(key=lambda t: bool(getattr(t, 'is_generated', True)))
    for tr in others:
        try:
            return tr.fetch()
        except Exception as e:
            if class_names(e) & BLOCKED_ERRORS:
                raise
            last_error = e

    if last_error is not None and not (class_names(last_error) & NO_CAPTIONS_ERRORS):
        raise last_error
    return None


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: fetch_builtin_transcript.py <video_id>\n")
        sys.exit(1)

    video_id = sys.argv[1]
    try:
        fetched = fetch_cues(video_id)
    except Exception as e:
        fail(e)

    if not fetched:
        print(json.dumps([]))
        return

    cues = []
    sample_chars = []
    for s in fetched:
        text = getattr(s, 'text', '')
        start = getattr(s, 'start', 0.0)
        dur = getattr(s, 'duration', 0.0)
        if text and text.strip():
            cleaned = text.replace("\n", " ").strip()
            cues.append({
                "start_sec": round(float(start), 3),
                "end_sec": round(float(start + dur), 3),
                "text": cleaned,
            })
            if len(sample_chars) < 500:
                sample_chars.append(cleaned)

    # Final quality gate: ensure the extracted transcript is NOT in a foreign non-Latin script (Arabic, Sanskrit, Thai, etc.)
    sample_text = " ".join(sample_chars)
    if is_foreign_script(sample_text):
        sys.stderr.write(f"Rejected transcript for {video_id}: contains non-Latin foreign script (Arabic/Sanskrit/Thai). Falling back to Whisper.\n")
        print(json.dumps([]))
        return

    print(json.dumps(cues))


if __name__ == "__main__":
    main()
