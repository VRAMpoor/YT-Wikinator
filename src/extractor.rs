use crate::models::TranscriptCue;
use anyhow::{anyhow, Context, Result};
use colored::*;
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

/// Folders that belong to the program, most specific first: the folder of
/// ytc.exe, then the project root when the exe runs from `target/<profile>/`.
/// For an exe anywhere else the folder two levels up is somebody else's
/// (the user's home, for an app unpacked into Downloads) and is not used.
fn app_dirs(exe: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(exe_dir) = exe.and_then(Path::parent) {
        dirs.push(exe_dir.to_path_buf());
        let target_dir = exe_dir.parent().filter(|p| p.file_name().is_some_and(|n| n == "target"));
        if let Some(root) = target_dir.and_then(Path::parent) {
            dirs.push(root.to_path_buf());
        }
    }
    dirs
}

/// Where a helper script is looked for, in this order: the program's own
/// folders (`app_dirs`), then the current directory. Each folder is tried
/// directly and with `scripts/`.
///
/// The program's folders come first: ytc can be started from any folder, and
/// a file named like a helper that happens to lie there must not be run in
/// place of the helper that came with the program.
fn script_candidates(script_name: &str, exe: Option<&Path>, cwd: Option<&Path>) -> Vec<PathBuf> {
    app_dirs(exe)
        .into_iter()
        .chain(cwd.map(Path::to_path_buf))
        .flat_map(|dir| [dir.join(script_name), dir.join("scripts").join(script_name)])
        .collect()
}

fn resolve_script(script_name: &str) -> PathBuf {
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    script_candidates(script_name, exe.as_deref(), cwd.as_deref())
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(script_name))
}

/// Where a bundled Python is looked for: next to ytc.exe first, then in the
/// current directory (same reason as in `script_candidates`).
fn bundled_python_candidates(exe: Option<&Path>, cwd: Option<&Path>) -> Vec<PathBuf> {
    exe.and_then(Path::parent)
        .into_iter()
        .chain(cwd)
        .map(|dir| dir.join("python").join("python.exe"))
        .collect()
}

/// Builds a Command for the Python interpreter with UTF-8 output forced, so
/// non-ASCII video and channel titles survive a piped stdout on Windows.
pub(crate) fn python_command(python_bin: &str) -> Command {
    let mut cmd = Command::new(python_bin);
    cmd.env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1");
    cmd
}

/// Phrases with which YouTube, yt-dlp and youtube-transcript-api report a rate
/// limit or an IP block. A bare "429" is not in the list: it also occurs in
/// video ids, traceback line numbers and timestamps. "Sign in to confirm
/// you're not a bot" is the same IP block, worded by yt-dlp.
const RATE_LIMIT_MARKERS: &[&str] = &[
    "HTTP Error 429",
    "status 429",
    "Too Many Requests",
    "IpBlocked",
    "RequestBlocked",
    "Sign in to confirm",
    "not a bot",
];

/// Wording of the helpers (browser, requests, yt-dlp) when this computer
/// cannot reach YouTube at all: the name is not resolved or no route exists.
/// Not the video's fault and not a block.
const NO_CONNECTION_MARKERS: &[&str] = &[
    "getaddrinfo failed",
    "Failed to resolve",
    "NameResolutionError",
    "Temporary failure in name resolution",
    "ERR_NAME_NOT_RESOLVED",
    "ERR_INTERNET_DISCONNECTED",
    "ERR_NETWORK_CHANGED",
    "ERR_ADDRESS_UNREACHABLE",
    "Network is unreachable",
];

/// True when helper output says YouTube could not be reached from here.
pub(crate) fn is_no_connection_text(text: &str) -> bool {
    NO_CONNECTION_MARKERS.iter().any(|m| text.contains(m))
}

/// yt-dlp wording for a video that is in the feed but cannot be played yet
/// (scheduled premiere, upcoming live stream).
const NOT_AVAILABLE_YET_MARKERS: &[&str] = &[
    "Premieres in",
    "Premiere will begin",
    "live event will begin",
    "This live event",
];

/// Output of a helper that could not run at all: a Python module (yt_dlp,
/// youtube_transcript_api, faster_whisper) or a helper script is not there,
/// or ffmpeg is missing.
const HELPER_MISSING_MARKERS: &[&str] = &[
    "No module named",
    "can't open file",
    "ffmpeg not found",
    "ffmpeg is not installed",
];

/// True when helper output says YouTube is blocking or rate-limiting this IP.
/// Only meant for helper output (yt-dlp, the transcript script), never for the
/// answer of an LLM.
pub(crate) fn is_rate_limit_text(text: &str) -> bool {
    // "Sign in to confirm your age" is an age-restricted video, not a block:
    // treated as one it would be retried, with a back-off, on every crawl.
    let text = text.replace("Sign in to confirm your age", "");
    RATE_LIMIT_MARKERS.iter().any(|m| text.contains(m))
}

/// Why no transcript could be extracted. A crawl counts a failure against the
/// video only for `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptFailure {
    /// YouTube is rate-limiting or blocking this IP.
    RateLimited,
    /// Scheduled premiere or upcoming live stream: nothing to transcribe yet.
    NotAvailableYet,
    /// Python, a Python module, a helper script or ffmpeg is missing.
    HelperMissing,
    /// This computer could not reach YouTube (no network, no name resolution).
    NoConnection,
    /// Anything else: the video itself could not be transcribed.
    Other,
}

/// Error of `SubtitleExtractor::extract_transcript`, with the reason as a
/// type so callers do not have to search the text.
#[derive(Debug)]
pub struct TranscriptError {
    pub kind: TranscriptFailure,
    pub message: String,
}

impl std::fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TranscriptError {}

/// Classifies the error text of a helper. The rate limit is checked first: a
/// blocked IP also produces "video unavailable" style follow-up errors.
pub(crate) fn classify_failure_text(text: &str) -> TranscriptFailure {
    if is_rate_limit_text(text) {
        TranscriptFailure::RateLimited
    } else if is_no_connection_text(text) {
        TranscriptFailure::NoConnection
    } else if NOT_AVAILABLE_YET_MARKERS.iter().any(|m| text.contains(m)) {
        TranscriptFailure::NotAvailableYet
    } else if HELPER_MISSING_MARKERS.iter().any(|m| text.contains(m)) {
        TranscriptFailure::HelperMissing
    } else {
        TranscriptFailure::Other
    }
}

/// Like `classify_failure_text`, and also recognises a helper program that
/// could not be started because it does not exist (no Python on this machine).
fn classify_failure(err: &anyhow::Error) -> TranscriptFailure {
    let program_missing = err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    });
    match classify_failure_text(&format!("{:#}", err)) {
        TranscriptFailure::Other if program_missing => TranscriptFailure::HelperMissing,
        kind => kind,
    }
}

/// The subtitle tracks asked of yt-dlp when the transcript helper itself
/// failed: English or German. `-orig` is yt-dlp's name for the automatic
/// captions of the spoken language. No pattern: `en.*` also matches every
/// regional and machine-translated variant and fetches them all.
const SUBTITLE_LANGUAGES: &str = "en-orig,en,de-orig,de";

/// What the caller knows about a block by YouTube before a transcript is
/// fetched (from `pacing::PacingState`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptionPlan {
    /// Transcript requests rest until this time (Unix seconds) after a block.
    pub resting_until: Option<i64>,
    /// A video that gets no transcript because of a block is transcribed
    /// locally from its audio instead of waiting for the block to end.
    pub whisper_when_blocked: bool,
}

/// What happened to the transcript requests for one video.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptionReport {
    /// YouTube answered a direct transcript request (with captions or "none").
    pub asked: bool,
    /// YouTube refused such a request (429 / IP block).
    pub blocked: bool,
}

/// One line of a channel's video list (`SubtitleExtractor::list_channel_videos`).
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelVideo {
    pub video_id: String,
    pub title: String,
    /// Upload time as Unix seconds, to the day at best. None when not shown.
    pub uploaded_at: Option<i64>,
}

/// Reads the lines "id<TAB>timestamp<TAB>title" yt-dlp printed. Lines that
/// are not a video (no 11-character id) are left out.
fn parse_channel_video_lines(stdout: &str) -> Vec<ChannelVideo> {
    stdout
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(3, '\t');
            let id = parts.next()?.trim();
            let stamp = parts.next()?.trim();
            let title = parts.next().unwrap_or("").trim();
            let id_ok = id.len() == 11 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !id_ok {
                return None;
            }
            Some(ChannelVideo {
                video_id: id.to_string(),
                title: if title.is_empty() || title == "NA" { "Untitled Video".to_string() } else { title.to_string() },
                uploaded_at: stamp.parse::<i64>().ok().filter(|t| *t > 0),
            })
        })
        .collect()
}

/// What the browser helper (`fetch_browser_transcript.py`) answered.
#[derive(Debug, Clone)]
enum BrowserAnswer {
    /// The lines of the transcript the YouTube page shows, and the video
    /// length the page names.
    Cues(Vec<TranscriptCue>, Option<f64>),
    /// The page was read and offers no usable transcript.
    NoCaptions(Option<f64>),
    /// Playwright or a browser is missing: this way does not exist here.
    NotSetUp,
    /// YouTube showed a sign-in or bot check instead of the page.
    Blocked(String),
    /// Anything else (page changed, timeout, ...).
    Failed(String),
}

/// What the browser helper prints on success.
#[derive(serde::Deserialize)]
struct BrowserPage {
    duration_sec: Option<f64>,
    cues: Vec<TranscriptCue>,
}

/// Reads the helper's exit code and output (see the helper's docstring).
fn browser_answer(exit_code: Option<i32>, stdout: &str, stderr: &str) -> BrowserAnswer {
    let reason = last_line(stderr);
    match exit_code {
        Some(0) => match serde_json::from_str::<BrowserPage>(stdout.trim()) {
            Ok(page) => {
                let duration = page.duration_sec.filter(|d| d.is_finite() && *d > 0.0);
                if page.cues.is_empty() {
                    BrowserAnswer::NoCaptions(duration)
                } else {
                    BrowserAnswer::Cues(page.cues, duration)
                }
            }
            Err(e) => BrowserAnswer::Failed(format!("unreadable answer of the browser helper: {}", e)),
        },
        Some(3) => BrowserAnswer::Blocked(reason),
        Some(4) => BrowserAnswer::NotSetUp,
        _ => BrowserAnswer::Failed(reason),
    }
}

/// The retryable error for a transcript request YouTube refuses.
fn blocked_error(message: String) -> anyhow::Error {
    anyhow::Error::new(TranscriptError { kind: TranscriptFailure::RateLimited, message })
}

/// A helper's stderr without yt-dlp's notice that the Python version in use
/// will stop being supported: it is printed on every call and would be shown
/// as the reason of whatever failed.
fn helper_error_text(stderr: &str) -> String {
    stderr
        .lines()
        .filter(|line| !line.trim_start().starts_with("Deprecated Feature:"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Last non-empty line of a helper's stderr: the actual error, without the traceback.
fn last_line(text: &str) -> String {
    text.lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

pub(crate) fn detect_python() -> String {
    if let Ok(val) = std::env::var("YTC_PYTHON") {
        if !val.trim().is_empty() {
            return val;
        }
    }
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    if let Some(p) = bundled_python_candidates(exe.as_deref(), cwd.as_deref()).into_iter().find(|p| p.exists()) {
        return p.to_string_lossy().to_string();
    }
    // Current Debian/Ubuntu and macOS ship only "python3".
    if cfg!(windows) {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

/// Separator between the fields of the one `yt-dlp --print` line used by
/// `fetch_video_meta`. Printable ASCII, so it survives the Windows command line
/// and a piped stdout; unlikely inside a title or channel name.
const META_SEPARATOR: &str = "|~ytc~|";

/// What yt-dlp reports about a video in one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoMeta {
    pub title: String,
    /// The uploader's UC... id; None when yt-dlp did not report a valid one.
    pub channel_id: Option<String>,
    /// The uploader's display name.
    pub channel: Option<String>,
    /// Upload date as RFC 3339 at midnight UTC (yt-dlp reports the day only).
    pub upload_date: Option<String>,
    /// Length of the video in whole seconds. None for a live stream or when
    /// yt-dlp did not report it.
    pub duration_sec: Option<u32>,
}

/// yt-dlp prints "NA" for a field it does not have.
fn meta_field(raw: &str) -> Option<String> {
    let value = raw.trim();
    (!value.is_empty() && value != "NA").then(|| value.to_string())
}

/// "20240131" (yt-dlp `upload_date`) as "2024-01-31T00:00:00+00:00".
pub fn upload_date_to_rfc3339(yyyymmdd: &str) -> Option<String> {
    let date = chrono::NaiveDate::parse_from_str(yyyymmdd.trim(), "%Y%m%d").ok()?;
    let midnight = date.and_hms_opt(0, 0, 0)?;
    Some(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(midnight, chrono::Utc).to_rfc3339())
}

/// yt-dlp `duration` ("213", "213.0") as whole seconds. None for "NA", zero,
/// a negative or an unreadable value: an invented length would make a
/// complete transcript look incomplete.
pub fn parse_duration_secs(raw: &str) -> Option<u32> {
    let secs = raw.trim().parse::<f64>().ok()?;
    (secs.is_finite() && secs >= 0.5 && secs < u32::MAX as f64).then(|| secs.round() as u32)
}

/// Parses the output of
/// `yt-dlp --print "%(title)s|~ytc~|%(channel_id)s|~ytc~|%(channel)s|~ytc~|%(upload_date)s|~ytc~|%(duration)s"`.
/// None when there is no such line or the title is missing. The last four
/// fields are split off from the right, so a separator inside the title cannot
/// shift them.
pub fn parse_video_meta(output: &str) -> Option<VideoMeta> {
    let line = output.lines().map(str::trim).find(|l| l.contains(META_SEPARATOR))?;
    let mut fields = line.rsplitn(5, META_SEPARATOR);
    let duration = fields.next()?;
    let upload_date = fields.next()?;
    let channel = fields.next()?;
    let channel_id = fields.next()?;
    let title = meta_field(fields.next()?)?;

    Some(VideoMeta {
        title,
        channel_id: meta_field(channel_id).filter(|id| id.starts_with("UC") && id.len() == 24),
        channel: meta_field(channel),
        upload_date: meta_field(upload_date).and_then(|d| upload_date_to_rfc3339(&d)),
        duration_sec: meta_field(duration).and_then(|d| parse_duration_secs(&d)),
    })
}

/// `transcripts.source` values (see `models::TRANSCRIPT_SOURCES`).
pub const SOURCE_BUILTIN: &str = "builtin";
pub const SOURCE_SUBTITLES: &str = "subtitles";
pub const SOURCE_WHISPER: &str = "whisper";

/// The speech model of the Whisper fallback. The multilingual one: it hears
/// which language is spoken, where "base.en" would write English words for
/// German speech. Same size as "base.en"; ytc_setup.py downloads this name.
pub const WHISPER_MODEL: &str = "base";

/// A caption track counts as the transcript of the whole video only when its
/// last cue ends at or after this share of the video's length. Not 1.0:
/// videos end with music, end cards or silence that carry no cue. Below it
/// the track is treated as cut short and the next tier is tried.
pub const MIN_CAPTION_COVERAGE: f64 = 0.90;

/// Absolute counterpart of `MIN_CAPTION_COVERAGE` for short videos: a track
/// that ends at most this many seconds before the video does is complete (a
/// 15-second outro is 12% of a two-minute video).
pub const CAPTION_END_SLACK_SECS: f64 = 20.0;

/// Whether a transcript reaches the end of its video.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completeness {
    /// The video length is not known, so nothing can be compared.
    Unknown,
    Complete,
    /// The last cue ends clearly before the video does.
    Incomplete { covered_pct: u32 },
}

/// End of the last cue in seconds; None without cues.
pub fn covered_until(cues: &[TranscriptCue]) -> Option<f64> {
    cues.iter()
        .map(|c| if c.end_sec.is_finite() { c.end_sec.max(c.start_sec) } else { c.start_sec })
        .filter(|t| t.is_finite())
        .fold(None, |best: Option<f64>, t| Some(best.map_or(t, |b| b.max(t))))
}

/// The completeness rule: compares the end of the last cue with the video
/// length (`MIN_CAPTION_COVERAGE`, `CAPTION_END_SLACK_SECS`).
pub fn transcript_completeness(duration_sec: Option<f64>, covered_until_sec: Option<f64>) -> Completeness {
    let Some(duration) = duration_sec.filter(|d| d.is_finite() && *d > 0.0) else {
        return Completeness::Unknown;
    };
    let Some(covered) = covered_until_sec.filter(|c| c.is_finite()) else {
        return Completeness::Unknown;
    };
    if covered >= duration * MIN_CAPTION_COVERAGE || duration - covered <= CAPTION_END_SLACK_SECS {
        Completeness::Complete
    } else {
        // Never rounds a shortfall up to 100.
        let pct = ((covered.max(0.0) / duration) * 100.0).round() as u32;
        Completeness::Incomplete { covered_pct: pct.min(99) }
    }
}

/// "12:34" or "1:02:03".
fn clock(secs: f64) -> String {
    let total = secs.max(0.0).round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        format!("{}:{:02}", m, s)
    }
}

/// A transcript with the facts needed to say how much of the video it covers.
#[derive(Debug, Clone)]
pub struct ExtractedTranscript {
    pub cues: Vec<TranscriptCue>,
    /// `SOURCE_BUILTIN`, `SOURCE_SUBTITLES` or `SOURCE_WHISPER`.
    pub source: &'static str,
    /// Length of the video; None when yt-dlp did not report it.
    pub video_duration_sec: Option<f64>,
    /// End of the last cue.
    pub covered_until_sec: Option<f64>,
}

impl ExtractedTranscript {
    fn new(source: &'static str, cues: Vec<TranscriptCue>, video_duration_sec: Option<f64>) -> Self {
        let covered_until_sec = covered_until(&cues);
        Self { cues, source, video_duration_sec, covered_until_sec }
    }

    pub fn completeness(&self) -> Completeness {
        transcript_completeness(self.video_duration_sec, self.covered_until_sec)
    }
}

/// Decides what to do with the caption track of one tier. A track that
/// reaches the end of the video (or whose video length is unknown) is
/// returned. A track that stops early is not accepted: it is remembered in
/// `partial` when it reaches further than the one already there, and None
/// tells the caller to try the next tier.
fn accept_track(
    track: ExtractedTranscript,
    partial: &mut Option<ExtractedTranscript>,
) -> Option<ExtractedTranscript> {
    match track.completeness() {
        Completeness::Complete | Completeness::Unknown => Some(track),
        Completeness::Incomplete { covered_pct } => {
            println!(
                "    {} The {} track ends at {} of {} ({}% of the video): incomplete, trying the next source.",
                "⚠".yellow(),
                track.source,
                clock(track.covered_until_sec.unwrap_or(0.0)),
                clock(track.video_duration_sec.unwrap_or(0.0)),
                covered_pct
            );
            let reach = |t: &ExtractedTranscript| t.covered_until_sec.unwrap_or(0.0);
            if partial.as_ref().map_or(true, |p| reach(&track) > reach(p)) {
                *partial = Some(track);
            }
            None
        }
    }
}

/// What `extract_transcript` returns when the audio tier failed with `err`.
///
/// A rate limit or a video that is not available yet is passed on as the
/// retryable error, also when an incomplete caption track is at hand: the
/// full transcript can be had later, the crawl has to back off now, and a
/// track accepted here would be judged and stored as if it were the video.
/// A block already seen in another tier (`rate_limit_note`) decides as well: the
/// audio download then fails for the same reason, whatever it says.
///
/// For every other failure (no Whisper installed, audio not downloadable)
/// the incomplete track is better than a video that fails on every crawl. It
/// is returned with its true coverage, which the caller stores and shows.
fn after_whisper_failure(
    partial: Option<ExtractedTranscript>,
    rate_limit_note: Option<String>,
    err: &anyhow::Error,
) -> Result<ExtractedTranscript> {
    let (kind, message) = match rate_limit_note {
        Some(note) => (
            TranscriptFailure::RateLimited,
            format!("YouTube rate limit / IP block (429): {}; audio fallback also failed: {}", note, err),
        ),
        None => (classify_failure(err), format!("{:#}", err)),
    };
    let retry_later = matches!(
        kind,
        TranscriptFailure::RateLimited | TranscriptFailure::NotAvailableYet | TranscriptFailure::NoConnection
    );
    match partial {
        Some(track) if !retry_later => {
            println!(
                "    {} Whisper could not transcribe the audio ({}). Using the {} track, which ends at {} of {}; the stored coverage says so.",
                "⚠".yellow(),
                last_line(&format!("{:#}", err)),
                track.source,
                clock(track.covered_until_sec.unwrap_or(0.0)),
                clock(track.video_duration_sec.unwrap_or(0.0))
            );
            Ok(track)
        }
        Some(track) => {
            println!(
                "    {} The {} track ends at {} of {} and the full audio could not be fetched now. The incomplete track is not used: the video is tried again later.",
                "⚠".yellow(),
                track.source,
                clock(track.covered_until_sec.unwrap_or(0.0)),
                clock(track.video_duration_sec.unwrap_or(0.0))
            );
            Err(anyhow::Error::new(TranscriptError { kind, message }))
        }
        None => Err(anyhow::Error::new(TranscriptError { kind, message })),
    }
}

#[derive(Clone)]
pub struct SubtitleExtractor {
    python_bin: String,
    /// False: transcripts are never read from the YouTube page in a browser.
    browser_transcripts: bool,
}

impl Default for SubtitleExtractor {
    fn default() -> Self {
        Self {
            python_bin: detect_python(),
            // The unit tests never open a browser.
            browser_transcripts: !cfg!(test),
        }
    }
}

#[cfg(test)]
impl SubtitleExtractor {
    /// For tests: no helper can be started, so nothing is asked of anyone.
    pub(crate) fn without_helpers() -> Self {
        Self {
            python_bin: "no-such-python-for-this-test".to_string(),
            browser_transcripts: true,
        }
    }
}

impl SubtitleExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    /// The newest videos of a channel, read from its "Videos" page by yt-dlp.
    /// Used when the channel's RSS feed does not answer: YouTube's feed
    /// address returns 404 and 500 for whole stretches of time, also for
    /// channels that exist. One request, no video is loaded.
    pub fn list_channel_videos(&self, channel_id: &str, max: usize) -> Result<Vec<ChannelVideo>> {
        let output = python_command(&self.python_bin)
            .args([
                "-m",
                "yt_dlp",
                "--flat-playlist",
                "--playlist-end",
                &max.max(1).to_string(),
                // Dates are not on the list page; this turns "3 days ago" into a day.
                "--extractor-args",
                "youtubetab:approximate_date",
                "--print",
                "%(id)s\t%(timestamp)s\t%(title)s",
                "--socket-timeout",
                "15",
                "--no-warnings",
                &format!("https://www.youtube.com/channel/{}/videos", channel_id),
            ])
            .output()
            .context("Failed to run yt-dlp for the channel's video list")?;
        if !output.status.success() {
            let err = helper_error_text(&String::from_utf8_lossy(&output.stderr));
            return Err(anyhow!("yt-dlp could not list the channel's videos: {}", last_line(&err)));
        }
        Ok(parse_channel_video_lines(&String::from_utf8_lossy(&output.stdout)))
    }

    /// Fetches the real title of a video via yt-dlp
    pub fn fetch_title(&self, video_id: &str) -> Option<String> {
        let output = python_command(&self.python_bin)
            .args([
                "-m",
                "yt_dlp",
                "--print",
                "title",
                "--socket-timeout",
                "15",
                "--no-warnings",
                &format!("https://www.youtube.com/watch?v={}", video_id),
            ])
            .output()
            .ok()?;

        if output.status.success() {
            let title = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !title.is_empty() {
                return Some(title);
            }
        }
        None
    }

    /// Title, channel id, channel name, upload date and length of a video
    /// from ONE yt-dlp call (instead of one call for the title and another to
    /// resolve the channel). None when yt-dlp fails or reports no title.
    pub fn fetch_video_meta(&self, video_id: &str) -> Option<VideoMeta> {
        let template =
            ["%(title)s", "%(channel_id)s", "%(channel)s", "%(upload_date)s", "%(duration)s"].join(META_SEPARATOR);
        let output = python_command(&self.python_bin)
            .args([
                "-m",
                "yt_dlp",
                "--print",
                &template,
                "--socket-timeout",
                "15",
                "--no-warnings",
                &format!("https://www.youtube.com/watch?v={}", video_id),
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }
        parse_video_meta(&String::from_utf8_lossy(&output.stdout))
    }

    /// Checks if transcript cues contain predominantly non-Latin foreign scripts (Arabic, Sanskrit/Devanagari, Thai, etc.)
    fn is_foreign_script_cues(cues: &[TranscriptCue]) -> bool {
        let mut foreign_chars = 0;
        let mut total_chars = 0;
        for cue in cues.iter().take(30) {
            for c in cue.text.chars() {
                if !c.is_whitespace() && !c.is_ascii_punctuation() {
                    total_chars += 1;
                    // Arabic, Devanagari/Hindi/Sanskrit, Thai, Cyrillic, CJK, etc.
                    if matches!(c as u32, 0x0600..=0x08FF | 0x0900..=0x097F | 0x0E00..=0x0E7F | 0x0400..=0x04FF | 0x4E00..=0x9FFF | 0x3040..=0x30FF | 0xAC00..=0xD7AF) {
                        foreign_chars += 1;
                    }
                }
            }
        }
        total_chars > 0 && (foreign_chars as f64 / total_chars as f64) > 0.05
    }

    /// Obtains the transcript of the WHOLE video in this order:
    /// Tier 1: the transcript the YouTube page shows, read in a real browser (no audio download)
    /// Tier 2: the direct transcript request (only when the page could not be read)
    /// Tier 3: yt-dlp VTT subtitle scraping (only when the tier 2 helper itself failed)
    /// Tier 4: Local Whisper transcription of the full audio (GPU if the helper can use it, else CPU)
    ///
    /// A caption track of tier 1 to 3 is accepted only when it reaches the end
    /// of the video (`transcript_completeness`); a track that stops early
    /// falls through to the next tier. `known_duration_sec` is the video
    /// length when the caller already has it; otherwise it is asked from
    /// yt-dlp (one metadata call). Without a known length a track cannot be
    /// checked and is accepted as it is.
    ///
    /// If Whisper cannot run and an incomplete caption track exists, that
    /// track is returned with its true coverage instead of failing the video.
    /// Not so when YouTube is rate-limiting or the video is not available
    /// yet: then the error is returned and the video is tried again later
    /// (`after_whisper_failure`).
    ///
    /// When all tiers fail the error is a `TranscriptError` (use
    /// `downcast_ref`), whose kind says whether the video is to blame.
    ///
    /// `plan` says what is known about a block by YouTube (see `CaptionPlan`).
    /// While direct transcript requests rest after a block, tier 2 is left
    /// out. With `whisper_when_blocked` (what the pipeline asks for) a video
    /// that got no transcript because of a block goes on to Whisper; without
    /// it the retryable `RateLimited` error is returned and the video waits.
    /// The returned `CaptionReport` tells the caller what to remember.
    pub fn extract_transcript(
        &self,
        video_id: &str,
        known_duration_sec: Option<f64>,
        plan: CaptionPlan,
    ) -> (Result<ExtractedTranscript>, CaptionReport) {
        let mut report = CaptionReport::default();
        let result = self.extract_transcript_inner(video_id, known_duration_sec, plan, &mut report);
        (result, report)
    }

    fn extract_transcript_inner(
        &self,
        video_id: &str,
        known_duration_sec: Option<f64>,
        plan: CaptionPlan,
        report: &mut CaptionReport,
    ) -> Result<ExtractedTranscript> {
        let ask_youtube = plan.resting_until.is_none();

        // Tier 1: the transcript YouTube itself shows on the video page, read in
        // a real browser. The video is not loaded for this, and nothing else
        // is asked of YouTube before it: the page also names the video length.
        let browser_started = std::time::Instant::now();
        let page = self.fetch_browser_transcript(video_id);
        let browser_secs = browser_started.elapsed().as_secs_f64();
        let page_duration = match &page {
            BrowserAnswer::Cues(_, d) | BrowserAnswer::NoCaptions(d) => *d,
            _ => None,
        };

        let duration = known_duration_sec
            .filter(|d| d.is_finite() && *d > 0.0)
            .or(page_duration)
            .or_else(|| self.fetch_video_meta(video_id).and_then(|m| m.duration_sec).map(f64::from));
        match duration {
            Some(d) => println!("    {} Video length: {}", "ℹ".dimmed(), clock(d)),
            None => println!(
                "    {} Video length not known: the transcript cannot be checked for completeness.",
                "ℹ".dimmed()
            ),
        }

        // Set when YouTube is blocking / rate-limiting us, so the caller can back off.
        let mut rate_limit_note: Option<String> = None;
        // The caption track that reaches furthest, when none reaches the end.
        let mut partial: Option<ExtractedTranscript> = None;

        // True when the page was read: what it says about captions stands, and
        // no second request is made for the same answer.
        let mut page_answered = false;
        let mut browser_note: Option<String> = None;
        match page {
            BrowserAnswer::Cues(cues, _) if !Self::is_foreign_script_cues(&cues) => {
                println!(
                    "    {} Read the transcript from the YouTube page in a browser in {:.1}s ({} lines)",
                    "⚡".green().bold(),
                    browser_secs,
                    cues.len()
                );
                page_answered = true;
                let track = ExtractedTranscript::new(SOURCE_BUILTIN, cues, duration);
                if let Some(complete) = accept_track(track, &mut partial) {
                    return Ok(complete);
                }
            }
            BrowserAnswer::Cues(..) | BrowserAnswer::NoCaptions(_) => {
                println!("    {} The YouTube page offers no usable transcript.", "ℹ".dimmed());
                page_answered = true;
            }
            BrowserAnswer::NotSetUp => {}
            BrowserAnswer::Blocked(reason) => {
                println!("    {} YouTube refused the browser: {}", "⚠".yellow(), reason);
                browser_note = Some(format!(" The browser was refused as well: {}.", reason));
                rate_limit_note = Some(reason);
            }
            BrowserAnswer::Failed(reason) if is_no_connection_text(&reason) => {
                return Err(anyhow::Error::new(TranscriptError {
                    kind: TranscriptFailure::NoConnection,
                    message: format!("YouTube could not be reached from this computer: {}", reason),
                }));
            }
            BrowserAnswer::Failed(reason) => {
                println!("    {} Reading the transcript in a browser failed: {}", "⚠".yellow(), reason);
                browser_note = Some(format!(" Reading it in a browser failed: {}.", reason));
            }
        }
        let ask_youtube = ask_youtube && !page_answered;
        if !page_answered && plan.resting_until.is_some() && rate_limit_note.is_none() {
            rate_limit_note = Some("direct transcript requests are resting after a block".to_string());
        }

        // Tier 2: the direct request (youtube-transcript-api), where the page
        // could not be read and such requests are not resting after a block.
        let tier1_started = std::time::Instant::now();
        // True when the helper could not ask YouTube properly (not: YouTube
        // answered "no captions", and not: YouTube refused).
        let mut tier1_broken = false;
        let tier1 = if ask_youtube {
            self.fetch_builtin_transcript(video_id)
        } else {
            Ok(vec![])
        };
        match tier1 {
            Ok(cues) => {
                // YouTube answered the direct request (with captions or with
                // "none"): only this counts as a request that got through.
                report.asked = ask_youtube;
                if !cues.is_empty() && !Self::is_foreign_script_cues(&cues) {
                    println!(
                        "    {} Retrieved YouTube built-in transcript in {:.1}s ({} cues)",
                        "⚡".green().bold(),
                        tier1_started.elapsed().as_secs_f64(),
                        cues.len()
                    );
                    let track = ExtractedTranscript::new(SOURCE_BUILTIN, cues, duration);
                    if let Some(complete) = accept_track(track, &mut partial) {
                        return Ok(complete);
                    }
                }
            }
            Err(e) => {
                let msg = e.to_string();
                println!("    {} Built-in transcript fetch failed: {}", "⚠".yellow(), msg);
                if is_rate_limit_text(&msg) {
                    report.blocked = true;
                    rate_limit_note = Some(msg);
                } else if is_no_connection_text(&msg) {
                    return Err(anyhow::Error::new(TranscriptError {
                        kind: TranscriptFailure::NoConnection,
                        message: format!("YouTube could not be reached from this computer: {}", msg),
                    }));
                } else {
                    tier1_broken = true;
                }
            }
        }

        // A refused request is not followed by a second one of the same kind for
        // the same video: every further request keeps the block alive.
        if rate_limit_note.is_some() && !plan.whisper_when_blocked {
            let browser_note = browser_note.unwrap_or_default();
            return Err(blocked_error(match plan.resting_until {
                Some(until) => format!(
                    "YouTube refused transcript requests from this computer earlier (429 / IP block). Not asking again before {}; the video stays pending.{}",
                    crate::pacing::local_clock(until),
                    browser_note
                ),
                None => format!(
                    "YouTube refused the transcript request (429 / IP block): {}. The video stays pending and is tried again later.{}",
                    rate_limit_note.unwrap_or_default(),
                    browser_note
                ),
            }));
        }

        // Tier 3: yt-dlp VTT subtitle scraping. Only when the direct request itself
        // failed: when YouTube answered it, the video's own tracks are known, and
        // all yt-dlp could add is a track machine-translated on request, which
        // YouTube limits far more strictly.
        let temp_dir = tempdir().context("Failed to create temporary directory for subtitles")?;
        let output_template = temp_dir.path().join("sub.%(ext)s");
        let video_url = format!("https://www.youtube.com/watch?v={}", video_id);

        let output = if tier1_broken {
            Some(
                python_command(&self.python_bin)
            .args([
                "-m",
                "yt_dlp",
                "--skip-download",
                "--write-sub",
                "--write-auto-sub",
                "--sub-lang",
                SUBTITLE_LANGUAGES,
                "--sub-format",
                "vtt",
                "--extractor-args",
                "youtube:player_client=android,web",
                "--socket-timeout",
                "5",
                "--retries",
                "1",
                "-o",
                output_template.to_str().unwrap(),
                "--no-warnings",
                &video_url,
            ])
            .output(),
            )
        } else {
            None
        };

        if let Some(Ok(ref out)) = output {
            if out.status.success() {
                let mut vtt_path = None;
                if let Ok(entries) = fs::read_dir(temp_dir.path()) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|ext| ext.to_str()) == Some("vtt") {
                            vtt_path = Some(path);
                            break;
                        }
                    }
                }

                if let Some(vtt_file) = vtt_path {
                    if let Ok(content) = fs::read_to_string(&vtt_file) {
                        if let Ok(cues) = parse_vtt(&content) {
                            if !cues.is_empty() && !Self::is_foreign_script_cues(&cues) {
                                let track = ExtractedTranscript::new(SOURCE_SUBTITLES, cues, duration);
                                if let Some(complete) = accept_track(track, &mut partial) {
                                    return Ok(complete);
                                }
                            }
                        }
                    }
                }
            } else {
                let err = helper_error_text(&String::from_utf8_lossy(&out.stderr));
                println!("    {} Subtitle download failed: {}", "⚠".yellow(), last_line(&err));
                if is_rate_limit_text(&err) {
                    report.blocked = true;
                    if rate_limit_note.is_none() {
                        rate_limit_note = Some(last_line(&err));
                    }
                    if !plan.whisper_when_blocked {
                        return Err(blocked_error(format!(
                            "YouTube refused the subtitle request (429 / IP block): {}. The video stays pending and is tried again later.",
                            last_line(&err)
                        )));
                    }
                } else if classify_failure_text(&err) == TranscriptFailure::NotAvailableYet {
                    // A premiere or an upcoming live stream has no audio either.
                    return Err(anyhow::Error::new(TranscriptError {
                        kind: TranscriptFailure::NotAvailableYet,
                        message: last_line(&err),
                    }));
                }
            }
        }

        // Tier 4: local Whisper on the full audio. Which device it runs on is only
        // known afterwards; transcribe_audio_cuda prints the helper's own device line.
        if partial.is_some() {
            println!(
                "    {} The captions on YouTube stop before the end of the video. Transcribing the full audio locally with Whisper...",
                "⚡".yellow()
            );
        } else if rate_limit_note.is_some() {
            println!("    {} YouTube is limiting caption requests. Transcribing locally with Whisper...", "⚡".yellow());
        } else {
            println!("    {} No captions found on YouTube. Transcribing locally with Whisper...", "⚡".yellow());
        }
        match self.transcribe_audio_cuda(video_id) {
            Ok(cues) => {
                let whisper = ExtractedTranscript::new(SOURCE_WHISPER, cues, duration);
                if let Completeness::Incomplete { covered_pct } = whisper.completeness() {
                    // The full audio was transcribed: what is missing has no speech.
                    println!(
                        "    {} Whisper found speech up to {}% of the video; the rest has none.",
                        "ℹ".dimmed(),
                        covered_pct
                    );
                }
                Ok(whisper)
            }
            Err(e) => after_whisper_failure(partial, rate_limit_note, &e),
        }
    }

    /// Fetches YouTube built-in transcript directly via Innertube API without downloading audio
    pub fn fetch_builtin_transcript(&self, video_id: &str) -> Result<Vec<TranscriptCue>> {
        let script = resolve_script("fetch_builtin_transcript.py");
        let output = python_command(&self.python_bin)
            .args([script.to_str().unwrap_or("fetch_builtin_transcript.py"), video_id])
            .output()
            .context("Failed to execute fetch_builtin_transcript.py")?;

        // The helper exits non-zero only for real failures (IP block, missing
        // module, ...). "This video has no captions" is exit 0 with [].
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("{}", last_line(&err)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let trimmed = stdout.trim();
        if trimmed.is_empty() || trimmed == "[]" {
            return Ok(vec![]);
        }

        let cues: Vec<TranscriptCue> = serde_json::from_str(trimmed)
            .with_context(|| format!("Failed to parse built-in transcript JSON: {}", trimmed))?;
        Ok(cues)
    }

    /// Opens the video page in a real browser (no window) and reads the
    /// transcript YouTube shows there: the first thing tried for every video.
    fn fetch_browser_transcript(&self, video_id: &str) -> BrowserAnswer {
        if !self.browser_transcripts {
            return BrowserAnswer::NotSetUp;
        }
        let script = resolve_script("fetch_browser_transcript.py");
        match python_command(&self.python_bin)
            .args([script.to_str().unwrap_or("fetch_browser_transcript.py"), video_id])
            .output()
        {
            Ok(out) => browser_answer(
                out.status.code(),
                &String::from_utf8_lossy(&out.stdout),
                &String::from_utf8_lossy(&out.stderr),
            ),
            // No Python: this way does not exist here.
            Err(_) => BrowserAnswer::NotSetUp,
        }
    }

    /// Downloads the audio and runs local faster-whisper. The helper tries CUDA
    /// and falls back to CPU, so the name says what is attempted, not what ran;
    /// the device really used and the measured time are printed.
    pub fn transcribe_audio_cuda(&self, video_id: &str) -> Result<Vec<TranscriptCue>> {
        let temp_dir = tempdir().context("Failed to create temporary directory for audio")?;
        let audio_template = temp_dir.path().join("audio.%(ext)s");
        let video_url = format!("https://www.youtube.com/watch?v={}", video_id);

        let mut args = vec![
            "-m".to_string(),
            "yt_dlp".to_string(),
            "-f".to_string(),
            "140/ba/bestaudio/best".to_string(),
            "--fixup".to_string(),
            "never".to_string(),
            "--socket-timeout".to_string(),
            "15".to_string(),
            "--retries".to_string(),
            "2".to_string(),
            "-o".to_string(),
            audio_template.to_str().unwrap().to_string(),
            "--no-warnings".to_string(),
        ];
        if std::process::Command::new("node").arg("--version").output().is_ok() {
            args.push("--js-runtimes".to_string());
            args.push("node".to_string());
        }
        args.push(video_url);

        let dl_out = python_command(&self.python_bin)
            .args(&args)
            .output()
            .context("Failed to download audio for Whisper transcription")?;

        if !dl_out.status.success() {
            let err = helper_error_text(&String::from_utf8_lossy(&dl_out.stderr));
            return Err(anyhow!("yt-dlp failed to download audio: {}", err.trim()));
        }

        let mut audio_file = None;
        for entry in fs::read_dir(temp_dir.path())? {
            let entry = entry?;
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ext == "m4a" || ext == "mp3" || ext == "opus" || ext == "webm" || ext == "mp4" || ext == "mkv" {
                    audio_file = Some(path);
                    break;
                }
            }
        }

        let audio_path = audio_file.ok_or_else(|| anyhow!("Audio file not found for video {}", video_id))?;

        let script = resolve_script("transcribe_cuda.py");
        let whisper_started = std::time::Instant::now();
        let whisper_out = python_command(&self.python_bin)
            .args([
                script.to_str().unwrap_or("transcribe_cuda.py"),
                audio_path.to_str().unwrap(),
                WHISPER_MODEL,
            ])
            .output()
            .context("Failed to execute the Whisper transcription script")?;
        let whisper_secs = whisper_started.elapsed().as_secs_f64();

        if !whisper_out.status.success() {
            let err = String::from_utf8_lossy(&whisper_out.stderr);
            return Err(anyhow!("Whisper script failed: {}", err.trim()));
        }

        // The helper falls back to CPU and still exits 0; its last stderr line is
        // the only place that says which device was really used.
        let whisper_note = last_line(&String::from_utf8_lossy(&whisper_out.stderr));
        if whisper_note.is_empty() {
            println!("    {} Whisper finished in {:.1}s (device not reported)", "ℹ".dimmed(), whisper_secs);
        } else {
            println!("    {} Whisper: {} ({:.1}s)", "ℹ".dimmed(), whisper_note, whisper_secs);
        }

        let stdout = String::from_utf8_lossy(&whisper_out.stdout);
        let cues: Vec<TranscriptCue> = serde_json::from_str(stdout.trim())
            .context("Failed to parse JSON cues from the Whisper output")?;

        if cues.is_empty() {
            return Err(anyhow!("Whisper produced an empty transcript for video {}", video_id));
        }

        Ok(cues)
    }
}

/// Parses WebVTT content and normalizes deduplicated rolling cues
pub fn parse_vtt(content: &str) -> Result<Vec<TranscriptCue>> {
    let tag_re = Regex::new(r"<[^>]+>").unwrap();
    let music_re = Regex::new(r"\[[^\]]+\]|♪[^♪]*♪").unwrap();

    let mut cues = Vec::new();
    let mut lines = content.lines().peekable();

    let mut last_added_text = String::new();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("WEBVTT") || trimmed.starts_with("Kind:") || trimmed.starts_with("Language:") {
            continue;
        }

        // Look for timestamp line: e.g. "00:00:01.360 --> 00:00:03.040"
        if trimmed.contains("-->") {
            let mut parts = trimmed.split("-->");
            let start_str = match parts.next() {
                Some(s) => s.trim(),
                None => continue,
            };
            let end_str = match parts.next() {
                Some(s) => s.trim().split_whitespace().next().unwrap_or(""),
                None => continue,
            };

            let start_sec = match parse_timestamp(start_str) {
                Some(sec) => sec,
                None => continue,
            };
            let end_sec = match parse_timestamp(end_str) {
                Some(sec) => sec,
                None => continue,
            };

            // Collect text lines until blank line or next timestamp
            let mut text_lines = Vec::new();
            while let Some(next_line) = lines.peek() {
                let next_trimmed = next_line.trim();
                if next_trimmed.is_empty() || next_trimmed.contains("-->") {
                    break;
                }
                text_lines.push(lines.next().unwrap());
            }

            let raw_text = text_lines.join(" ");
            let clean_no_tags = tag_re.replace_all(&raw_text, "");
            let clean_no_music = music_re.replace_all(&clean_no_tags, "");
            let clean_text = clean_no_music
                .split_whitespace()
                .collect::<Vec<&str>>()
                .join(" ");

            if clean_text.is_empty() {
                continue;
            }

            // Word-level overlap deduplication for YouTube rolling auto-captions
            let curr_words: Vec<&str> = clean_text.split_whitespace().collect();
            let prev_words: Vec<&str> = last_added_text.split_whitespace().collect();

            // Find longest suffix of prev_words that matches prefix of curr_words
            let max_overlap = prev_words.len().min(curr_words.len());
            let mut overlap_len = 0;
            for k in (1..=max_overlap).rev() {
                let prev_suffix = &prev_words[prev_words.len() - k..];
                let curr_prefix = &curr_words[..k];
                if prev_suffix == curr_prefix {
                    overlap_len = k;
                    break;
                }
            }

            let text_to_store = if overlap_len > 0 {
                curr_words[overlap_len..].join(" ")
            } else {
                clean_text.clone()
            };

            if text_to_store.is_empty() {
                continue;
            }

            last_added_text = clean_text;

            cues.push(TranscriptCue {
                start_sec,
                end_sec,
                text: text_to_store,
            });
        }
    }

    Ok(cues)
}

fn parse_timestamp(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        2 => {
            let m: f64 = parts[0].parse().ok()?;
            let s: f64 = parts[1].parse().ok()?;
            Some(m * 60.0 + s)
        }
        3 => {
            let h: f64 = parts[0].parse().ok()?;
            let m: f64 = parts[1].parse().ok()?;
            let s: f64 = parts[2].parse().ok()?;
            Some(h * 3600.0 + m * 60.0 + s)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resting_block_is_answered_without_asking_youtube_or_any_helper() {
        // A Python that does not exist: any helper call would fail as "helper missing".
        let extractor = SubtitleExtractor::without_helpers();
        let plan = CaptionPlan { resting_until: Some(4_000_000_000), whisper_when_blocked: false };
        let (result, report) = extractor.extract_transcript("abcdefghijk", Some(60.0), plan);
        let err = result.expect_err("the video waits");
        assert_eq!(err.downcast_ref::<TranscriptError>().map(|t| t.kind), Some(TranscriptFailure::RateLimited));
        assert!(err.to_string().contains("Not asking again before"), "{}", err);
        assert_eq!(report, CaptionReport { asked: false, blocked: false });
    }

    #[test]
    fn the_browser_helper_answer_is_read_from_exit_code_and_output() {
        let page = r#"{"duration_sec": 870.0, "cues": [{"start_sec": 0.0, "end_sec": 7.0, "text": "hallo"}]}"#;
        match browser_answer(Some(0), page, "") {
            BrowserAnswer::Cues(c, d) => assert_eq!((c.len(), c[0].text.as_str(), d), (1, "hallo", Some(870.0))),
            other => panic!("{:?}", other),
        }
        // A length of 0 means the page did not name one.
        assert!(matches!(
            browser_answer(Some(0), r#"{"duration_sec": 0, "cues": []}"#, ""),
            BrowserAnswer::NoCaptions(None)
        ));
        assert!(matches!(browser_answer(Some(0), "<html>", ""), BrowserAnswer::Failed(_)));
        match browser_answer(Some(3), "", "x\nYouTube asked for a bot check in the browser (treated like 429): y\n") {
            BrowserAnswer::Blocked(reason) => {
                assert_eq!(reason, "YouTube asked for a bot check in the browser (treated like 429): y");
            }
            other => panic!("{:?}", other),
        }
        assert!(matches!(browser_answer(Some(4), "", "Browser transcripts are not set up: z"), BrowserAnswer::NotSetUp));
        assert!(matches!(browser_answer(Some(1), "", "Browser transcript error: TimeoutError: t"), BrowserAnswer::Failed(_)));
        // Killed, no exit code.
        assert!(matches!(browser_answer(None, "", ""), BrowserAnswer::Failed(_)));
    }

    #[test]
    fn the_channel_video_list_of_yt_dlp_is_read_line_by_line() {
        let out = "listVideo01\t1794182400\tA first title\nlistVideo02\tNA\tTabs\tin a title\n\nWARNING: something\nshort\t1\tx\n";
        let videos = parse_channel_video_lines(out);
        assert_eq!(videos.len(), 2, "{:?}", videos);
        assert_eq!(videos[0], ChannelVideo {
            video_id: "listVideo01".to_string(),
            title: "A first title".to_string(),
            uploaded_at: Some(1794182400),
        });
        assert_eq!((videos[1].title.as_str(), videos[1].uploaded_at), ("Tabs\tin a title", None));
    }

    #[test]
    fn a_lost_connection_is_neither_a_block_nor_the_videos_fault() {
        use TranscriptFailure::*;
        for text in [
            "ERROR: [youtube] abc: Unable to download API page: HTTPSConnection(host='www.youtube.com', port=443): Failed to resolve 'www.youtube.com' ([Errno 11001] getaddrinfo failed)",
            "Browser transcript error: Error: Page.goto: net::ERR_NAME_NOT_RESOLVED at https://www.youtube.com/watch?v=abc&hl=en",
            "Built-in transcript error: ConnectionError: HTTPSConnectionPool(host='www.youtube.com', port=443): Max retries exceeded (Caused by NameResolutionError)",
        ] {
            assert_eq!(classify_failure_text(text), NoConnection, "{}", text);
        }
        // A block that also mentions the network stays a block.
        assert_eq!(classify_failure_text("HTTP Error 429: Too Many Requests; Failed to resolve"), RateLimited);
        assert_eq!(classify_failure_text("ERROR: Video unavailable"), Other);
        // With no connection the incomplete track is not used: tried again later.
        let down = anyhow!("yt-dlp failed to download audio: Failed to resolve 'www.youtube.com'");
        let err = after_whisper_failure(None, None, &down).expect_err("tried again later");
        assert_eq!(err.downcast_ref::<TranscriptError>().map(|t| t.kind), Some(NoConnection));
    }

    #[test]
    fn the_python_version_notice_of_yt_dlp_is_not_shown_as_the_reason() {
        let stderr = "Deprecated Feature: Support for Python version 3.10 has been deprecated. Please update to Python 3.11 or above\nERROR: [youtube] abc: This live event will begin in 25 days.\n";
        let text = helper_error_text(stderr);
        assert_eq!(text, "ERROR: [youtube] abc: This live event will begin in 25 days.");
        assert_eq!(classify_failure_text(&text), TranscriptFailure::NotAvailableYet);
        assert_eq!(helper_error_text("Deprecated Feature: x"), "");
        assert_eq!(helper_error_text("ERROR: HTTP Error 429: Too Many Requests"), "ERROR: HTTP Error 429: Too Many Requests");
    }

    #[test]
    fn only_tracks_in_the_videos_own_language_are_asked_of_yt_dlp() {
        assert!(!SUBTITLE_LANGUAGES.contains('*'), "a pattern also matches translated tracks");
        for lang in SUBTITLE_LANGUAGES.split(',') {
            assert!(matches!(lang, "en" | "en-orig" | "de" | "de-orig"), "{}", lang);
        }
    }

    #[test]
    fn rate_limit_is_recognised_by_phrase_not_by_the_digits_429() {
        assert!(is_rate_limit_text("ERROR: unable to download video data: HTTP Error 429: Too Many Requests"));
        assert!(is_rate_limit_text("request failed with status 429"));
        assert!(is_rate_limit_text("IpBlocked / rate limit (429) from YouTube: blocked"));
        assert!(is_rate_limit_text("youtube_transcript_api._errors.RequestBlocked"));
        assert!(is_rate_limit_text("ERROR: [youtube] abc: Sign in to confirm you\u{2019}re not a bot."));
        assert!(is_rate_limit_text("Sign in to confirm you're not a bot"));

        // The digits alone say nothing: ids, line numbers, timings.
        assert!(!is_rate_limit_text("ERROR: [youtube] a429bcdEFgh: Video unavailable"));
        assert!(!is_rate_limit_text("  File \"helper.py\", line 429, in main"));
        assert!(!is_rate_limit_text("Whisper produced an empty transcript after 429.1 s"));
        // Age restriction is a property of the video.
        assert!(!is_rate_limit_text("ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate"));
    }

    #[test]
    fn helper_failures_are_classified_by_who_is_to_blame() {
        use TranscriptFailure::*;
        assert_eq!(classify_failure_text("ERROR: [youtube] x: Premieres in 3 hours"), NotAvailableYet);
        assert_eq!(classify_failure_text("ERROR: [youtube] x: This live event will begin in 2 days."), NotAvailableYet);
        assert_eq!(
            classify_failure_text("Whisper transcription error: ModuleNotFoundError: No module named 'faster_whisper'"),
            HelperMissing
        );
        assert_eq!(classify_failure_text("python.exe: No module named yt_dlp"), HelperMissing);
        assert_eq!(
            classify_failure_text("python: can't open file 'transcribe_cuda.py': [Errno 2] No such file or directory"),
            HelperMissing
        );
        assert_eq!(classify_failure_text("HTTP Error 429: Too Many Requests; Premieres in 1 hour"), RateLimited);
        assert_eq!(classify_failure_text("ERROR: [youtube] x: Video unavailable"), Other);
        assert_eq!(classify_failure_text("Whisper produced an empty transcript for video x"), Other);
    }

    #[test]
    fn a_program_that_does_not_exist_counts_as_missing_helper() {
        let spawn = std::io::Error::new(std::io::ErrorKind::NotFound, "program not found");
        let err = anyhow::Error::new(spawn).context("Failed to download audio for Whisper transcription");
        assert_eq!(classify_failure(&err), TranscriptFailure::HelperMissing);

        let denied = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        assert_eq!(classify_failure(&anyhow::Error::new(denied)), TranscriptFailure::Other);
        assert_eq!(classify_failure(&anyhow!("Audio file not found for video x")), TranscriptFailure::Other);
    }

    #[test]
    fn transcript_error_survives_as_a_typed_anyhow_error() {
        let err = anyhow::Error::new(TranscriptError {
            kind: TranscriptFailure::NotAvailableYet,
            message: "yt-dlp failed to download audio: Premieres in 3 hours".to_string(),
        })
        .context("outer context");
        let typed = err.downcast_ref::<TranscriptError>().expect("typed through context");
        assert_eq!(typed.kind, TranscriptFailure::NotAvailableYet);
        assert!(format!("{:#}", err).contains("Premieres in 3 hours"));
    }

    #[test]
    fn video_meta_is_parsed_from_one_print_line() {
        let out = "Building Bridges: Part 2|~ytc~|UCxxxxxxxxxxxxxxxxxxxxxx|~ytc~|Fakechannel|~ytc~|20240131|~ytc~|754\n";
        let meta = parse_video_meta(out).expect("complete line");
        assert_eq!(meta.title, "Building Bridges: Part 2");
        assert_eq!(meta.channel_id.as_deref(), Some("UCxxxxxxxxxxxxxxxxxxxxxx"));
        assert_eq!(meta.channel.as_deref(), Some("Fakechannel"));
        assert_eq!(meta.upload_date.as_deref(), Some("2024-01-31T00:00:00+00:00"));
        assert_eq!(meta.duration_sec, Some(754));
    }

    #[test]
    fn missing_fields_stay_unknown_instead_of_being_invented() {
        // yt-dlp prints NA for what it does not know.
        let meta = parse_video_meta("Some title|~ytc~|NA|~ytc~|NA|~ytc~|NA|~ytc~|NA").expect("title is enough");
        assert_eq!(meta.title, "Some title");
        assert_eq!(meta.channel_id, None);
        assert_eq!(meta.channel, None);
        assert_eq!(meta.upload_date, None);
        assert_eq!(meta.duration_sec, None);

        // A value that is not a channel id, a date or a length is not passed on.
        let meta = parse_video_meta("T|~ytc~|@handle|~ytc~|Name|~ytc~|2024-01-31|~ytc~|soon").expect("title");
        assert_eq!(meta.duration_sec, None);
        assert_eq!(meta.channel_id, None);
        assert_eq!(meta.channel.as_deref(), Some("Name"));
        assert_eq!(meta.upload_date, None);

        assert_eq!(parse_video_meta(""), None);
        assert_eq!(parse_video_meta("ERROR: Video unavailable"), None);
        assert_eq!(parse_video_meta("NA|~ytc~|NA|~ytc~|NA|~ytc~|NA|~ytc~|NA"), None);
        // A line with too few fields is not misread.
        assert_eq!(parse_video_meta("T|~ytc~|UCxxxxxxxxxxxxxxxxxxxxxx|~ytc~|Chan|~ytc~|20231201"), None);
        assert_eq!(parse_video_meta("only|~ytc~|two"), None);
    }

    #[test]
    fn separator_inside_the_title_does_not_shift_the_fields() {
        let out = "WARNING: something\nOdd |~ytc~| title|~ytc~|UCxxxxxxxxxxxxxxxxxxxxxx|~ytc~|Chan|~ytc~|20231201|~ytc~|3723.0";
        let meta = parse_video_meta(out).expect("line found after the warning");
        assert_eq!(meta.title, "Odd |~ytc~| title");
        assert_eq!(meta.channel.as_deref(), Some("Chan"));
        assert_eq!(meta.upload_date.as_deref(), Some("2023-12-01T00:00:00+00:00"));
        assert_eq!(meta.duration_sec, Some(3723));
    }

    #[test]
    fn upload_dates_are_validated() {
        assert_eq!(upload_date_to_rfc3339("20240229").as_deref(), Some("2024-02-29T00:00:00+00:00"));
        assert_eq!(upload_date_to_rfc3339("20230229"), None);
        assert_eq!(upload_date_to_rfc3339("NA"), None);
        assert_eq!(upload_date_to_rfc3339(""), None);
    }

    #[test]
    fn durations_are_parsed_from_what_yt_dlp_prints() {
        assert_eq!(parse_duration_secs("213"), Some(213));
        assert_eq!(parse_duration_secs(" 213.0 "), Some(213));
        assert_eq!(parse_duration_secs("59.6"), Some(60));
        // Live streams and errors: no length, never a made-up one.
        for unknown in ["NA", "", "0", "-5", "nan", "inf", "12:34"] {
            assert_eq!(parse_duration_secs(unknown), None, "{:?}", unknown);
        }
    }

    fn cues(ends: &[f64]) -> Vec<TranscriptCue> {
        ends.iter()
            .map(|&end| TranscriptCue { start_sec: (end - 4.0).max(0.0), end_sec: end, text: "words here".to_string() })
            .collect()
    }

    #[test]
    fn the_last_cue_end_is_what_a_transcript_covers() {
        assert_eq!(covered_until(&[]), None);
        assert_eq!(covered_until(&cues(&[10.0, 20.0, 600.5])), Some(600.5));
        // Out-of-order cues and cues without a usable end still give the furthest point.
        let mut odd = cues(&[300.0, 120.0]);
        odd.push(TranscriptCue { start_sec: 410.0, end_sec: 0.0, text: "x".to_string() });
        odd.push(TranscriptCue { start_sec: 1.0, end_sec: f64::NAN, text: "x".to_string() });
        assert_eq!(covered_until(&odd), Some(410.0));
    }

    #[test]
    fn a_caption_track_is_complete_only_when_it_reaches_the_end() {
        use Completeness::*;
        // 1 hour video.
        assert_eq!(transcript_completeness(Some(3600.0), Some(3590.0)), Complete);
        assert_eq!(transcript_completeness(Some(3600.0), Some(3240.0)), Complete); // exactly 90%
        assert_eq!(transcript_completeness(Some(3600.0), Some(3200.0)), Incomplete { covered_pct: 89 });
        assert_eq!(transcript_completeness(Some(3600.0), Some(600.0)), Incomplete { covered_pct: 17 });
        assert_eq!(transcript_completeness(Some(3600.0), Some(0.0)), Incomplete { covered_pct: 0 });
        // Cues that run past the reported length are complete.
        assert_eq!(transcript_completeness(Some(3600.0), Some(3605.0)), Complete);
        // Short video: a 15-second outro is not a missing transcript ...
        assert_eq!(transcript_completeness(Some(120.0), Some(104.0)), Complete);
        // ... half of the video is.
        assert_eq!(transcript_completeness(Some(120.0), Some(60.0)), Incomplete { covered_pct: 50 });
        // Unknown length or no cue end: nothing to compare, nothing is claimed.
        assert_eq!(transcript_completeness(None, Some(60.0)), Unknown);
        assert_eq!(transcript_completeness(Some(0.0), Some(60.0)), Unknown);
        assert_eq!(transcript_completeness(Some(f64::NAN), Some(60.0)), Unknown);
        assert_eq!(transcript_completeness(Some(600.0), None), Unknown);
    }

    #[test]
    fn an_incomplete_track_falls_through_and_the_furthest_one_is_remembered() {
        let duration = Some(1800.0);
        let mut partial = None;

        // Tier 1 stops at a third of the video: not accepted.
        let builtin = ExtractedTranscript::new(SOURCE_BUILTIN, cues(&[100.0, 600.0]), duration);
        assert!(accept_track(builtin, &mut partial).is_none());
        assert_eq!(partial.as_ref().map(|p| (p.source, p.covered_until_sec)), Some((SOURCE_BUILTIN, Some(600.0))));

        // Tier 2 reaches further but still not the end: it replaces the remembered one.
        let subs = ExtractedTranscript::new(SOURCE_SUBTITLES, cues(&[100.0, 900.0]), duration);
        assert!(accept_track(subs, &mut partial).is_none());
        assert_eq!(partial.as_ref().map(|p| p.source), Some(SOURCE_SUBTITLES));

        // A shorter one does not.
        let shorter = ExtractedTranscript::new(SOURCE_BUILTIN, cues(&[300.0]), duration);
        assert!(accept_track(shorter, &mut partial).is_none());
        assert_eq!(partial.as_ref().and_then(|p| p.covered_until_sec), Some(900.0));

        // A track that reaches the end is accepted and leaves the memory alone.
        let full = ExtractedTranscript::new(SOURCE_SUBTITLES, cues(&[100.0, 1795.0]), duration);
        let accepted = accept_track(full, &mut partial).expect("complete");
        assert_eq!((accepted.source, accepted.covered_until_sec), (SOURCE_SUBTITLES, Some(1795.0)));
        assert_eq!(accepted.completeness(), Completeness::Complete);

        // Unknown length: accepted, with the coverage left unknown.
        let mut none = None;
        let unchecked = ExtractedTranscript::new(SOURCE_BUILTIN, cues(&[100.0, 600.0]), None);
        let accepted = accept_track(unchecked, &mut none).expect("cannot be checked");
        assert_eq!(accepted.video_duration_sec, None);
        assert_eq!(accepted.completeness(), Completeness::Unknown);
        assert!(none.is_none());
    }

    #[test]
    fn an_incomplete_track_is_not_accepted_when_the_audio_failed_for_a_reason_that_passes() {
        let track = || Some(ExtractedTranscript::new(SOURCE_BUILTIN, cues(&[100.0, 300.0]), Some(1000.0)));
        let kind_of = |result: Result<ExtractedTranscript>| {
            let err = result.expect_err("no transcript");
            err.downcast_ref::<TranscriptError>().expect("typed").kind
        };

        // HTTP 429 on the audio download: back off and try again, with or without a partial track.
        let blocked = anyhow!("yt-dlp failed to download audio: ERROR: HTTP Error 429: Too Many Requests");
        assert_eq!(kind_of(after_whisper_failure(track(), None, &blocked)), TranscriptFailure::RateLimited);
        assert_eq!(kind_of(after_whisper_failure(None, None, &blocked)), TranscriptFailure::RateLimited);

        // A block already seen in another tier decides, whatever the audio step says.
        let other = anyhow!("yt-dlp failed to download audio: ERROR: unable to download video data");
        let note = Some("IpBlocked / rate limit (429) from YouTube".to_string());
        let err = after_whisper_failure(track(), note.clone(), &other).expect_err("rate limited");
        assert_eq!(err.downcast_ref::<TranscriptError>().map(|t| t.kind), Some(TranscriptFailure::RateLimited));
        assert!(err.to_string().contains("audio fallback also failed"), "{}", err);
        assert_eq!(kind_of(after_whisper_failure(None, note, &other)), TranscriptFailure::RateLimited);

        // A premiere: nothing to judge yet.
        let premiere = anyhow!("yt-dlp failed to download audio: ERROR: [youtube] x: Premieres in 3 hours");
        assert_eq!(kind_of(after_whisper_failure(track(), None, &premiere)), TranscriptFailure::NotAvailableYet);
    }

    #[test]
    fn an_incomplete_track_is_used_with_its_true_coverage_when_whisper_cannot_run() {
        let track = || Some(ExtractedTranscript::new(SOURCE_SUBTITLES, cues(&[100.0, 300.0]), Some(1000.0)));

        // No Whisper on this machine, or the audio of this video cannot be had.
        let no_whisper = anyhow!("Whisper script failed: ModuleNotFoundError: No module named 'faster_whisper'");
        let unavailable = anyhow!("yt-dlp failed to download audio: ERROR: [youtube] x: Video unavailable");
        for err in [&no_whisper, &unavailable] {
            let used = after_whisper_failure(track(), None, err).expect("the partial track is used");
            assert_eq!(used.source, SOURCE_SUBTITLES);
            // The coverage is the track's own, not rounded up to the video.
            assert_eq!((used.video_duration_sec, used.covered_until_sec), (Some(1000.0), Some(300.0)));
            assert_eq!(used.completeness(), Completeness::Incomplete { covered_pct: 30 });
        }

        // Without any track the failure keeps its kind.
        let err = after_whisper_failure(None, None, &no_whisper).expect_err("nothing to use");
        assert_eq!(err.downcast_ref::<TranscriptError>().map(|t| t.kind), Some(TranscriptFailure::HelperMissing));
        let err = after_whisper_failure(None, None, &unavailable).expect_err("nothing to use");
        assert_eq!(err.downcast_ref::<TranscriptError>().map(|t| t.kind), Some(TranscriptFailure::Other));
    }

    #[test]
    fn helpers_are_looked_for_in_the_program_folders_before_the_current_directory() {
        let exe = Path::new("app").join("target").join("release").join("ytc.exe");
        let cwd = Path::new("elsewhere");
        let found = script_candidates("transcribe_cuda.py", Some(&exe), Some(cwd));
        let exe_dir = Path::new("app").join("target").join("release");
        assert_eq!(
            found,
            vec![
                exe_dir.join("transcribe_cuda.py"),
                exe_dir.join("scripts").join("transcribe_cuda.py"),
                Path::new("app").join("transcribe_cuda.py"),
                Path::new("app").join("scripts").join("transcribe_cuda.py"),
                // The current directory comes last.
                cwd.join("transcribe_cuda.py"),
                cwd.join("scripts").join("transcribe_cuda.py"),
            ]
        );
        // An installed exe (not under target/<profile>/): two levels up is not the program's.
        let installed = Path::new("home").join("Downloads").join("yt-wikinator").join("ytc.exe");
        let found = script_candidates("x.py", Some(&installed), None);
        let app = Path::new("home").join("Downloads").join("yt-wikinator");
        assert_eq!(found, vec![app.join("x.py"), app.join("scripts").join("x.py")]);
        // Without a known exe only the current directory is left; without both, nothing.
        assert_eq!(script_candidates("x.py", None, Some(cwd)), vec![cwd.join("x.py"), cwd.join("scripts").join("x.py")]);
        assert!(script_candidates("x.py", None, None).is_empty());

        let python = bundled_python_candidates(Some(&exe), Some(cwd));
        assert_eq!(
            python,
            vec![exe_dir.join("python").join("python.exe"), cwd.join("python").join("python.exe")]
        );
        assert_eq!(bundled_python_candidates(None, None), Vec::<PathBuf>::new());
    }

    #[test]
    fn a_helper_next_to_the_program_wins_over_one_in_the_current_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = dir.path().join("app");
        let foreign = dir.path().join("foreign");
        fs::create_dir_all(app.join("scripts")).expect("app dir");
        fs::create_dir_all(&foreign).expect("foreign dir");
        fs::write(app.join("scripts").join("helper.py"), "# ours").expect("write");
        fs::write(foreign.join("helper.py"), "# not ours").expect("write");

        let exe = app.join("ytc.exe");
        let first_existing = |cwd: &Path| {
            script_candidates("helper.py", Some(&exe), Some(cwd)).into_iter().find(|p| p.exists())
        };
        assert_eq!(first_existing(&foreign), Some(app.join("scripts").join("helper.py")));
        // Only when the program has no such helper is the current directory used.
        fs::remove_file(app.join("scripts").join("helper.py")).expect("remove test file");
        assert_eq!(first_existing(&foreign), Some(foreign.join("helper.py")));
    }

    #[test]
    fn source_names_are_the_ones_the_database_accepts() {
        for source in [SOURCE_BUILTIN, SOURCE_SUBTITLES, SOURCE_WHISPER] {
            assert!(crate::models::TRANSCRIPT_SOURCES.contains(&source), "{}", source);
        }
        assert_eq!(clock(754.0), "12:34");
        assert_eq!(clock(3723.4), "1:02:03");
    }
}
