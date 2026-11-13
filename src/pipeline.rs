use crate::analyzer::{Analysis, LlmUnavailable, VideoAnalyzer, HEURISTIC_ANALYZER};
use crate::db::Database;
use crate::extractor::{
    transcript_completeness, CaptionPlan, Completeness, ExtractedTranscript, SubtitleExtractor, TranscriptError,
    TranscriptFailure, SOURCE_WHISPER,
};
use crate::pacing::{clock_seed, jittered_millis, local_clock, now_unix, PacingStore};
use crate::models::{
    Channel, CleanedTranscript, InsightJudgment, InsightMeta, InsightResult, SponsorSegment, TranscriptCoverage, Video,
    INFO_INSUFFICIENT, INFO_SUFFICIENT, TOPIC_OK,
};
use crate::rss::{self, RelatedChannel};
use crate::sponsorblock::SponsorBlockClient;
use crate::wiki::WikiGenerator;
use anyhow::{Context, Result};
use colored::*;
use reqwest::Client;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

#[derive(Debug, Clone, Serialize, Default)]
pub struct CrawlProgress {
    pub is_running: bool,
    pub current_channel_id: Option<String>,
    pub current_channel_title: Option<String>,
    pub current_video_id: Option<String>,
    pub current_video_title: Option<String>,
    pub stage: String,
    /// Videos finished or in work: the n of "[n/total]".
    pub current_index: usize,
    /// Length of the work list built from the feeds: new videos this crawl
    /// really processes. 0 until all feeds are checked.
    pub total_videos: usize,
    /// Videos judged on quality with an acceptable topic (kept + below the
    /// quality cutoff).
    pub videos_evaluated: usize,
    pub videos_kept: usize,
    /// Below the quality cutoff: slop score above the keep threshold.
    pub videos_filtered: usize,
    /// Videos that could not be processed in this crawl.
    #[serde(default)]
    pub videos_failed: usize,
    /// Topic rejected by the model's topic verdict (fringe / nonsense /
    /// politics). Not part of `videos_evaluated`.
    #[serde(default)]
    pub videos_off_topic: usize,
    /// The whole transcript held too little to judge. Not slop, not scored.
    #[serde(default)]
    pub videos_insufficient: usize,
    /// Analysed again although the owner disqualified them; they stay out.
    #[serde(default)]
    pub videos_disqualified: usize,
    /// Feeds that could not be fetched in this crawl (network, 404, IP block).
    #[serde(default)]
    pub feeds_failed: usize,
    pub percent: u8,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct CrawlSummary {
    /// Feeds that were fetched and read. A feed that could not be fetched is
    /// in `feeds_failed`, not here.
    pub channels_checked: usize,
    #[serde(default)]
    pub feeds_failed: usize,
    /// Videos judged on quality with an acceptable topic (kept + below the
    /// quality cutoff). Every processed video is in exactly one of: kept,
    /// filtered_slop, off_topic, insufficient, disqualified, failed.
    pub videos_evaluated: usize,
    pub videos_kept: usize,
    /// Below the quality cutoff: slop score above the keep threshold.
    pub videos_filtered_slop: usize,
    #[serde(default)]
    pub videos_failed: usize,
    /// Topic rejected by the model's topic verdict (fringe / nonsense / politics).
    #[serde(default)]
    pub videos_off_topic: usize,
    /// The whole transcript held too little to judge. Not slop, not scored.
    #[serde(default)]
    pub videos_insufficient: usize,
    /// Analysed again although the owner disqualified them; they stay out.
    #[serde(default)]
    pub videos_disqualified: usize,
    /// Left pending because YouTube refuses transcript requests for now. Not
    /// failures: the next crawl after the block takes them up.
    #[serde(default)]
    pub videos_waiting: usize,
    /// True when this crawl wrote the wiki (only with a wiki directory).
    #[serde(default)]
    pub wiki_generated: bool,
    /// Why the wiki step failed; None when it worked or was not requested.
    #[serde(default)]
    pub wiki_error: Option<String>,
}

impl CrawlSummary {
    /// Counts one processed video under its outcome.
    fn count(&mut self, outcome: VideoOutcome) {
        match outcome {
            VideoOutcome::Kept => {
                self.videos_evaluated += 1;
                self.videos_kept += 1;
            }
            VideoOutcome::BelowCutoff => {
                self.videos_evaluated += 1;
                self.videos_filtered_slop += 1;
            }
            VideoOutcome::OffTopic => self.videos_off_topic += 1,
            VideoOutcome::Insufficient => self.videos_insufficient += 1,
            VideoOutcome::Disqualified => self.videos_disqualified += 1,
        }
    }

    /// Videos of this run that got as far as a transcript or an LLM answer.
    /// One of them shows that the setup works (see
    /// `failures_count_against_videos`).
    fn videos_judged(&self) -> usize {
        self.videos_evaluated + self.videos_insufficient + self.videos_disqualified + self.videos_off_topic
    }

    /// "3 kept, 1 below the quality cutoff, ..." for the result line.
    pub fn counts_line(&self) -> String {
        let mut line = format!(
            "{} kept, {} below the quality cutoff, {} off-topic (fringe / nonsense / politics), {} not enough information, {} failed",
            self.videos_kept,
            self.videos_filtered_slop,
            self.videos_off_topic,
            self.videos_insufficient,
            self.videos_failed
        );
        if self.videos_disqualified > 0 {
            line.push_str(&format!(", {} disqualified by hand", self.videos_disqualified));
        }
        if self.videos_waiting > 0 {
            line.push_str(&format!(
                ", {} waiting (YouTube refuses transcript requests for now)",
                self.videos_waiting
            ));
        }
        if self.feeds_failed > 0 {
            // Otherwise the result reads as if every channel had been looked at.
            line.push_str(&format!("; {} feed(s) could not be fetched", self.feeds_failed));
        }
        line
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredItem {
    pub video_id: String,
    pub title: String,
    /// Not a measurement when `not_scored` is true.
    pub slop_score: u32,
    /// True when nothing was scored: the transcript held too little to judge.
    #[serde(default)]
    pub not_scored: bool,
    /// Why the video is not in the library ("Off-topic: politics (..)",
    /// "Not enough information (..)", "Below quality cutoff",
    /// "Disqualified (..)"); None for a kept video.
    #[serde(default)]
    pub rejection: Option<String>,
    pub quality_rating: String,
    /// The effective decision: quality, topic, information and the owner's
    /// disqualification together.
    pub is_worth_keeping: bool,
    /// Title of the channel this run added to the watchlist because of this video.
    pub auto_promoted_creator: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoverSummary {
    pub query: String,
    pub candidates_found: usize,
    pub items: Vec<DiscoveredItem>,
}

/// Where a `run_categorize` run stands, as GET /api/categorize/progress
/// reports it.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CategorizeProgress {
    pub is_running: bool,
    /// "starting", "running", "complete" or "error"; empty before the first run.
    pub stage: String,
    /// Videos finished or in work: the n of "[n/total]".
    pub current_index: usize,
    pub total_videos: usize,
    pub current_video_id: Option<String>,
    pub current_video_title: Option<String>,
    /// Videos the model put into one of the categories.
    pub assigned: usize,
    /// Videos for which the model named none of the categories.
    pub none: usize,
    pub failed: usize,
    pub percent: u8,
    pub message: String,
}

/// What a `run_categorize` run did. Every video of the run is in exactly one
/// of `assigned`, `none`, `failed`, `skipped` and `not_attempted`.
#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
pub struct CategorizeSummary {
    /// Videos this run was to work on (after `--limit`).
    pub total: usize,
    /// Selected videos that `--limit` left for a later run.
    pub beyond_limit: usize,
    /// Videos per category the model chose, for every category of the user's
    /// in the order of their names; a category nothing went into has 0.
    pub assigned: Vec<(String, usize)>,
    /// Videos for which the model named none of the categories.
    pub none: usize,
    /// Videos whose request failed or whose answer could not be read. Their
    /// category is unchanged.
    pub failed: usize,
    /// Videos whose category was set by hand while the run was under way.
    pub skipped: usize,
    /// Videos the run did not get to because it stopped.
    pub not_attempted: usize,
    /// Why the run stopped before the last video; None when it did not.
    pub stopped: Option<String>,
}

impl CategorizeSummary {
    /// Videos the model put into a category.
    pub fn assigned_total(&self) -> usize {
        self.assigned.iter().map(|(_, count)| count).sum()
    }

    /// The result in one line, for the console and the web page.
    pub fn counts_line(&self) -> String {
        let per_category: Vec<String> = self
            .assigned
            .iter()
            .filter(|(_, count)| *count > 0)
            .map(|(name, count)| format!("{}: {}", name, count))
            .collect();
        let mut line = format!("{} assigned", self.assigned_total());
        if !per_category.is_empty() {
            line.push_str(&format!(" ({})", per_category.join(", ")));
        }
        line.push_str(&format!(", {} in no category, {} failed", self.none, self.failed));
        if self.skipped > 0 {
            line.push_str(&format!(", {} set by hand meanwhile", self.skipped));
        }
        if self.not_attempted > 0 {
            line.push_str(&format!(", {} not attempted", self.not_attempted));
        }
        line
    }
}

#[derive(Clone)]
pub struct IngestionPipeline {
    pub db_path: PathBuf,
    pub extractor: SubtitleExtractor,
    pub sponsor_client: SponsorBlockClient,
    pub analyzer: VideoAnalyzer,
    pub http_client: Client,
    pub progress: Arc<RwLock<CrawlProgress>>,
    /// Progress of `run_categorize`.
    pub categorize_progress: Arc<RwLock<CategorizeProgress>>,
    /// A video is kept only when its slop score is at or below this value.
    /// Set with `with_max_slop`, which also tells the analyzer.
    pub max_slop: u32,
    /// Where `run_crawl_and_wiki` writes the wiki. Set with `with_wiki_dir`.
    pub wiki_dir: PathBuf,
    /// The pause between videos the user set, in seconds: the learned pause
    /// (`pacing`) never goes below it. A crawl sets it to its own delay.
    pub pause_floor_secs: Arc<AtomicU64>,
}

/// Pause between videos when no crawl has set one (the Control Center's default).
pub const DEFAULT_PAUSE_SECS: u64 = 3;

/// Pause after a video for which YouTube could not be reached at all, so a
/// short loss of the connection does not cost one video after another.
const NO_CONNECTION_PAUSE_SECS: u64 = 60;

/// Pause before the feeds that failed at first are asked a second time.
const FEED_RETRY_PAUSE_SECS: u64 = 30;

/// True when yt-dlp reports YouTube's own answer that a channel is not there.
/// Unlike a 404 of the feed address, this is about the channel itself.
fn channel_is_gone(list_error: &str) -> bool {
    list_error.contains("This channel does not exist")
}

/// How many more videos than the crawl takes per channel are read from a
/// channel's video list: Shorts and known videos are sorted out afterwards.
const CHANNEL_LIST_EXTRA: usize = 5;

/// A video from a channel's video list in the shape of a feed entry. The list
/// shows the upload day at best; without one, now is the nearest truth.
fn feed_item_from_list(v: crate::extractor::ChannelVideo) -> rss::RssVideoItem {
    let published_at = v
        .uploaded_at
        .and_then(|t| chrono::DateTime::from_timestamp(t, 0))
        .unwrap_or_else(chrono::Utc::now)
        .to_rfc3339();
    rss::RssVideoItem {
        url: format!("https://www.youtube.com/watch?v={}", v.video_id),
        video_id: v.video_id,
        title: v.title,
        published_at,
    }
}

/// The shortest pause between two videos of a crawl. A crawl asks YouTube for
/// video after video; at a pause of a few seconds YouTube soon refuses.
pub const MIN_CRAWL_PAUSE_SECS: u64 = 10;

/// Default keep threshold: a video is kept only when its slop score is at or
/// below the pipeline's `max_slop`, which starts at this value.
pub const KEEP_THRESHOLD: u32 = 45;

/// Default wiki directory, relative to the working directory.
pub const DEFAULT_WIKI_DIR: &str = "./wiki";

/// Video status after an LLM analysis or a "not enough information" verdict:
/// never crawled again.
pub const STATUS_PROCESSED: &str = "processed";
/// Video status after a heuristic-only result: a crawl with an LLM configured
/// analyses the stored transcript again.
pub const STATUS_HEURISTIC: &str = "heuristic";

/// A video that failed this often for its own reasons (not because the LLM
/// was down or YouTube rate-limited) is not picked up by a crawl any more.
pub const MAX_VIDEO_FAILURES: u32 = 3;

/// A run in which no video was scored and at least this many failed says
/// something about the setup (network, helper programs, LLM), not about the
/// videos: none of its failures is counted.
const RUN_FAILURES_WITHOUT_SUCCESS: usize = 3;

/// `analyzer_type` of a row stored for a video whose transcript holds too
/// little to judge. No analyzer ran.
pub const INSUFFICIENT_ANALYZER: &str = "info:insufficient";

/// Value in the slop_score column of such a row. Nothing was measured; 0 so
/// that no reader takes the video for slop. Readers show `info_status`.
const INSUFFICIENT_PLACEHOLDER_SCORE: u32 = 0;

/// A transcript with fewer words than this cannot be judged: about 45
/// seconds of normal speech, less than a paragraph of content.
pub const MIN_TRANSCRIPT_WORDS: usize = 120;

/// A video shorter than this many seconds needs at least
/// `MIN_WORDS_SHORT_VIDEO` words to be judged: a clip that short is a teaser
/// or a Short unless it is spoken very densely.
pub const SHORT_VIDEO_SECS: f64 = 60.0;
pub const MIN_WORDS_SHORT_VIDEO: usize = 200;

/// "Almost no speech": fewer words per minute than this over the whole video
/// (normal speech is 120-180), when the transcript is also shorter than
/// `SPARSE_SPEECH_MAX_WORDS`. Music, ambience, silent screen recordings.
pub const MIN_WORDS_PER_MINUTE: f64 = 10.0;
pub const SPARSE_SPEECH_MAX_WORDS: usize = 300;

/// Why a complete transcript holds too little to judge the video, or None
/// when there is enough. `words`: words of the cleaned transcript;
/// `duration_sec`: length of the video when known. This is not a quality
/// verdict: such a video is stored as "not enough information", is not kept
/// and does not count in the channel's rolling average.
pub fn insufficient_reason(words: usize, duration_sec: Option<f64>) -> Option<String> {
    let duration = duration_sec.filter(|d| d.is_finite() && *d > 0.0);
    let measured = match duration {
        Some(d) => format!("only {} words in {} seconds", words, d.round() as u64),
        None => format!("only {} words", words),
    };
    if words < MIN_TRANSCRIPT_WORDS {
        return Some(measured);
    }
    let d = duration?;
    if d < SHORT_VIDEO_SECS && words < MIN_WORDS_SHORT_VIDEO {
        return Some(measured);
    }
    if words < SPARSE_SPEECH_MAX_WORDS && (words as f64) / (d / 60.0) < MIN_WORDS_PER_MINUTE {
        return Some(format!("almost no speech: {}", measured));
    }
    None
}

/// THE keep rule, from its parts: the topic is acceptable (no verdict
/// recorded counts as acceptable), the slop score is at or below the
/// threshold, the transcript held enough information, and the owner did not
/// disqualify the video. `Database::KEPT_SQL` applies the same rule to
/// stored rows.
pub fn keep_decision(
    topic_verdict: Option<&str>,
    slop_score: u32,
    max_slop: u32,
    info_sufficient: bool,
    excluded_manual: bool,
) -> bool {
    topic_verdict.map_or(true, |v| v == TOPIC_OK) && slop_score <= max_slop && info_sufficient && !excluded_manual
}

/// What became of a processed video. Every video has exactly one outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoOutcome {
    Kept,
    /// Judged on quality: slop score above the keep threshold.
    BelowCutoff,
    /// Topic rejected (fringe / nonsense / politics).
    OffTopic,
    /// The transcript held too little to judge.
    Insufficient,
    /// Disqualified by the owner; stays out whatever the analysis says.
    Disqualified,
}

/// The outcome of a stored row. The order is the order of precedence, the
/// same as `InsightMeta::rejection_label`: the owner's decision, then the
/// topic, then the information, then plain quality.
pub fn video_outcome(meta: &InsightMeta) -> VideoOutcome {
    if meta.kept {
        VideoOutcome::Kept
    } else if meta.excluded_manual {
        VideoOutcome::Disqualified
    } else if !meta.topic_ok() {
        VideoOutcome::OffTopic
    } else if meta.is_insufficient() {
        VideoOutcome::Insufficient
    } else {
        VideoOutcome::BelowCutoff
    }
}

/// Outcome and result line of a processed video, for the crawl log.
fn outcome_report(insight: &InsightResult, meta: &InsightMeta, max_slop: u32) -> (VideoOutcome, String) {
    let outcome = video_outcome(meta);
    let rejection = || meta.rejection_label().unwrap_or_default();
    let unreviewed = if insight.analyzer_type == HEURISTIC_ANALYZER {
        " (heuristic score, not reviewed by an LLM)"
    } else {
        ""
    };
    let line = match outcome {
        VideoOutcome::Kept => format!(
            "Slop Score: {}/100 - KEPT ({}){}",
            insight.slop_score, insight.quality_rating, unreviewed
        ),
        VideoOutcome::BelowCutoff => format!(
            "Slop Score: {}/100 - below the quality cutoff (keep threshold {}){}",
            insight.slop_score, max_slop, unreviewed
        ),
        VideoOutcome::OffTopic => format!("{} - not kept (quality score {}/100)", rejection(), insight.slop_score),
        VideoOutcome::Insufficient => format!("{} - not judged, not counted in the channel average", rejection()),
        VideoOutcome::Disqualified => format!(
            "{} - stays out of the library whatever the analysis says (undo: ytc include {})",
            rejection(),
            insight.video_id
        ),
    };
    (outcome, line)
}

/// The result line in the colour of its outcome.
fn paint_outcome(outcome: VideoOutcome, line: &str) -> ColoredString {
    match outcome {
        VideoOutcome::Kept => line.green().bold(),
        VideoOutcome::BelowCutoff => line.red().bold(),
        _ => line.yellow(),
    }
}

/// What is known about a row whose meta could not be read back (should not
/// happen right after it was stored): the insight's own flag decides.
fn meta_fallback(insight: &InsightResult) -> InsightMeta {
    InsightMeta { kept: insight.is_worth_keeping, ..Default::default() }
}

/// Discover adds an untracked creator to the watchlist for a kept video at or
/// below this score.
const AUTO_TRACK_MAX_SLOP: u32 = 35;

// Crawl progress: feeds take the first part of the bar, videos the middle,
// the wiki step the end. 100 is only set when everything is written.
const FEED_PHASE_PERCENT: u8 = 20;
const VIDEO_PHASE_END: u8 = 95;
const WIKI_PHASE_PERCENT: u8 = 97;

/// True when a title carries the Shorts hashtag as its own token: the word
/// "shorts" inside a title ("Sewing shorts for summer") or a longer tag
/// ("#shortcuts") is not the hashtag.
fn has_shorts_hashtag(title: &str) -> bool {
    title.split_whitespace().any(|token| {
        // "(#shorts)" and "#shorts!" still count.
        let token = token
            .trim_start_matches(|c: char| c != '#' && !c.is_alphanumeric())
            .trim_end_matches(|c: char| !c.is_alphanumeric());
        token.eq_ignore_ascii_case("#shorts") || token.eq_ignore_ascii_case("#short")
    })
}

/// Who is to blame when a video could not be processed. Only `Video` counts
/// towards `MAX_VIDEO_FAILURES`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureCause {
    /// The LLM is down, timed out, or rejects every request (key, model name).
    Llm,
    /// YouTube is blocking or rate-limiting this IP.
    RateLimited,
    /// Scheduled premiere or upcoming live stream.
    NotAvailableYet,
    /// Python, a Python module, a helper script or ffmpeg is missing.
    HelperMissing,
    /// This computer could not reach YouTube.
    NoConnection,
    /// The video itself: no usable transcript, or an unusable LLM answer.
    Video,
}

/// Reads the cause from the error's type. The text is not searched here: it
/// can carry a whole LLM answer or a traceback, where "429" means nothing.
/// The rate limit is recognised where the helper output is read
/// (`extractor::is_rate_limit_text`).
fn failure_cause(err: &anyhow::Error) -> FailureCause {
    if err.downcast_ref::<LlmUnavailable>().is_some() {
        return FailureCause::Llm;
    }
    match err.downcast_ref::<TranscriptError>().map(|t| t.kind) {
        Some(TranscriptFailure::RateLimited) => FailureCause::RateLimited,
        Some(TranscriptFailure::NotAvailableYet) => FailureCause::NotAvailableYet,
        Some(TranscriptFailure::HelperMissing) => FailureCause::HelperMissing,
        Some(TranscriptFailure::NoConnection) => FailureCause::NoConnection,
        Some(TranscriptFailure::Other) | None => FailureCause::Video,
    }
}

/// Whether an LLM failure ends the crawl. An engine that is down ends it at
/// once: no further video can be analysed. A failure of one request to a
/// reachable engine (a timeout, a 5xx, a rejected key or model name) ends it
/// when it is the second in a row: then it is not this video's bad luck, and
/// every further video would cost a transcript fetch and the same wait.
fn stops_crawl(err: &LlmUnavailable, llm_failures_in_a_row: usize) -> bool {
    err.engine_down || llm_failures_in_a_row >= 2
}

/// What to do after a crawl was stopped by `err`, as the end of its message.
fn llm_stop_hint(err: &LlmUnavailable) -> String {
    if err.is_timeout() {
        format!(
            "The LLM engine is reachable but did not answer within the request timeout. Raise {} or use a faster engine, then crawl again.",
            crate::analyzer::LLM_TIMEOUT_SECS_ENV
        )
    } else if err.engine_down {
        "Start the LLM engine and crawl again.".to_string()
    } else {
        "Check the LLM engine and its settings (URL, model name, API key), then crawl again.".to_string()
    }
}

/// False for a run in which nothing was judged and
/// `RUN_FAILURES_WITHOUT_SUCCESS` or more videos failed: then the failures
/// are not counted against the videos.
fn failures_count_against_videos(videos_judged: usize, videos_failed: usize) -> bool {
    videos_judged > 0 || videos_failed < RUN_FAILURES_WITHOUT_SUCCESS
}

/// A failure of the current run that is the video's own fault. It is written
/// to the database at the end of the run, when it is known whether anything
/// worked at all.
struct PendingFailure {
    video_id: String,
    /// The video has a heuristic insight and a stored transcript.
    reanalysis: bool,
}

/// Counts the failures of a run against their videos, unless the run as a
/// whole shows that the setup was the problem. Returns how many were counted.
fn record_run_failures(
    db: &Database,
    failures: &[PendingFailure],
    videos_judged: usize,
    videos_failed: usize,
) -> usize {
    if failures.is_empty() {
        return 0;
    }
    if !failures_count_against_videos(videos_judged, videos_failed) {
        println!(
            "\n{} No video was judged and {} failed. That points at the setup (network, helper programs, LLM), not at the videos: no failure was counted, and all of them are tried again in the next crawl.",
            "ℹ".yellow(),
            videos_failed
        );
        for failure in failures.iter().filter(|f| !f.reanalysis) {
            // Shown as failed, without a count. A heuristic row keeps its status.
            let _ = db.update_video_status(&failure.video_id, "failed");
        }
        return 0;
    }

    let mut counted = 0;
    for failure in failures {
        match db.mark_video_failed(&failure.video_id) {
            Ok(count) => {
                counted += 1;
                let given_up = if count >= MAX_VIDEO_FAILURES {
                    format!(
                        " No further automatic attempts (retry with: ytc process-video {}).",
                        failure.video_id
                    )
                } else {
                    String::new()
                };
                println!(
                    "  {} Failure {}/{} recorded for video {}.{}",
                    "ℹ".dimmed(),
                    count,
                    MAX_VIDEO_FAILURES,
                    failure.video_id,
                    given_up
                );
            }
            Err(mark_err) => eprintln!("  [Crawl] Could not record the failure: {}", mark_err),
        }
        if failure.reanalysis {
            // The heuristic insight and the transcript are still there: keep
            // the status that says so, so the retry needs no YouTube request.
            // The count limits the retries.
            let _ = db.update_video_status(&failure.video_id, STATUS_HEURISTIC);
        }
    }
    counted
}

/// The cleaned transcript stored for a video, if it has text.
fn stored_transcript(db: &Database, video_id: &str) -> Option<CleanedTranscript> {
    db.get_transcript(video_id)
        .ok()
        .flatten()
        .filter(|t| !t.text.trim().is_empty())
}

/// Share of the video (0..=99) that a stored caption track reaches when it
/// stops before the end of the video; None when the transcript is complete,
/// when nothing can be compared (length or coverage not recorded), or when it
/// is a Whisper transcript: Whisper heard the full audio, so what is missing
/// there has no speech. The rule is `extractor::transcript_completeness`.
pub fn incomplete_caption_pct(coverage: Option<&TranscriptCoverage>) -> Option<u32> {
    let coverage = coverage?;
    if coverage.source.as_deref() == Some(SOURCE_WHISPER) {
        return None;
    }
    match transcript_completeness(coverage.video_duration_sec, coverage.covered_until_sec) {
        Completeness::Incomplete { covered_pct } => Some(covered_pct),
        Completeness::Complete | Completeness::Unknown => None,
    }
}

/// "stored transcript reaches 30% of the video": what a progress or READ
/// line says instead of "whole transcript" for an incomplete caption track.
pub fn partial_transcript_label(covered_pct: u32) -> String {
    format!("stored transcript reaches {}% of the video", covered_pct)
}

/// What was read for a stored analysis, for a progress or READ line: the
/// partial-transcript label when the stored transcript is an incomplete
/// caption track, otherwise `InsightMeta::whole_transcript_note`. All
/// sections of a transcript that covers 30% of the video are not the whole
/// transcript of the video.
#[cfg(test)]
pub fn transcript_read_label(meta: &InsightMeta, coverage: Option<&TranscriptCoverage>) -> Option<String> {
    match incomplete_caption_pct(coverage) {
        Some(pct) => Some(partial_transcript_label(pct)),
        None => meta.whole_transcript_note(),
    }
}

/// The error for a transcript that is an incomplete caption track AND holds
/// too little to judge (`reason`). Such a video is not stored as "not enough
/// information": the words are missing because the transcript is. It counts
/// as a failure of the video, so the next crawl fetches the transcript again
/// (up to `MAX_VIDEO_FAILURES` times).
fn incomplete_transcript_error(video_id: &str, covered_pct: u32, reason: &str) -> anyhow::Error {
    anyhow::Error::new(TranscriptError {
        kind: TranscriptFailure::Other,
        message: format!(
            "Transcript incomplete: the {} and the audio could not be transcribed; that part holds too little to judge ({}). Nothing was stored as 'not enough information'. Retry with: ytc process-video {}",
            partial_transcript_label(covered_pct),
            reason,
            video_id
        ),
    })
}

/// The stored transcript of a video when it can stand for the video: not
/// empty, and not a caption track that stops before the end
/// (`incomplete_caption_pct`). An incomplete one is fetched again instead.
fn reusable_transcript(db: &Database, video_id: &str) -> Option<CleanedTranscript> {
    let transcript = stored_transcript(db, video_id)?;
    let coverage = db.get_transcript_coverage(video_id).ok().flatten();
    match incomplete_caption_pct(coverage.as_ref()) {
        Some(_) => None,
        None => Some(transcript),
    }
}

/// The crawl log line of a judged video, with the partial-transcript label
/// appended when the judgment rests on an incomplete caption track.
fn with_read_note(line: String, outcome: VideoOutcome, partial_pct: Option<u32>) -> String {
    match partial_pct {
        // Insufficient is never stored for a partial transcript.
        Some(pct) if outcome != VideoOutcome::Insufficient => {
            format!("{} [{}]", line, partial_transcript_label(pct))
        }
        _ => line,
    }
}

/// Result line of the feed phase. A feed that could not be fetched is named
/// as such: it was not checked.
fn feeds_checked_line(feeds_checked: usize, feeds_failed: usize, new_videos: usize) -> String {
    if feeds_failed == 0 {
        format!("{} feed(s) checked, {} new video(s) to process.", feeds_checked, new_videos)
    } else {
        format!(
            "{} feed(s) checked, {} could not be fetched, {} new video(s) to process.",
            feeds_checked, feeds_failed, new_videos
        )
    }
}

/// Channels in the order a crawl visits them. Muted channels are left out.
/// 1. status: `high_signal` before everything else,
/// 2. least recently crawled first (never crawled = first),
/// 3. lower rolling slop score first (no score = last).
/// The order matters because a crawl can end early (LLM down, rate limit):
/// whatever is at the back is then not reached.
pub fn crawl_order(channels: Vec<Channel>) -> Vec<Channel> {
    fn status_rank(status: &str) -> u8 {
        if status == "high_signal" {
            0
        } else {
            1
        }
    }
    // None and unreadable timestamps sort first, like "never crawled".
    fn crawled_at(channel: &Channel) -> i64 {
        channel
            .last_crawled_at
            .as_deref()
            .and_then(|t| chrono::DateTime::parse_from_rfc3339(t.trim()).ok())
            .map(|t| t.timestamp_millis())
            .unwrap_or(i64::MIN)
    }

    let mut ordered: Vec<Channel> = channels.into_iter().filter(|c| c.status != "muted").collect();
    // Stable: channels equal in all three keys keep the order they came in.
    ordered.sort_by(|a, b| {
        status_rank(&a.status)
            .cmp(&status_rank(&b.status))
            .then_with(|| crawled_at(a).cmp(&crawled_at(b)))
            .then_with(|| {
                let slop = |c: &Channel| c.rolling_slop_score.unwrap_or(f64::INFINITY);
                slop(a).total_cmp(&slop(b))
            })
    });
    ordered
}

/// What a crawl does with a video of a feed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoPlan {
    /// Nothing to do.
    Skip,
    /// Fetch the transcript and analyse it.
    Process,
    /// Scored by the heuristic and an LLM is configured now: analyse
    /// the stored transcript again, without a YouTube request.
    Reanalyze,
}

/// `status` is None for a video that is not in the database yet.
fn plan_for_video(status: Option<&str>, fail_count: u32, has_llm: bool) -> VideoPlan {
    match status {
        Some(STATUS_PROCESSED) => VideoPlan::Skip,
        // Given up: it failed for its own reasons MAX_VIDEO_FAILURES times
        // (for a heuristic row: the LLM answer was unusable that often).
        Some("failed") | Some(STATUS_HEURISTIC) if fail_count >= MAX_VIDEO_FAILURES => VideoPlan::Skip,
        Some(STATUS_HEURISTIC) if has_llm => VideoPlan::Reanalyze,
        // Without an LLM a second heuristic pass would give the same result.
        Some(STATUS_HEURISTIC) => VideoPlan::Skip,
        _ => VideoPlan::Process,
    }
}

/// Stored status and failure count of a video; (None, 0) when it is not in the
/// database (or cannot be read, which is handled like a new video).
fn video_state(db: &Database, video_id: &str) -> (Option<String>, u32) {
    match db.get_video(video_id) {
        Ok(Some(video)) => (Some(video.status), db.video_fail_count(video_id).unwrap_or(0)),
        _ => (None, 0),
    }
}

/// Progress while feeds are fetched: 0..=FEED_PHASE_PERCENT, one step per feed.
fn feed_phase_percent(feeds_done: usize, feeds_total: usize) -> u8 {
    if feeds_total == 0 {
        return FEED_PHASE_PERCENT;
    }
    ((feeds_done.min(feeds_total) * FEED_PHASE_PERCENT as usize) / feeds_total) as u8
}

/// Progress while the work list is processed: FEED_PHASE_PERCENT..=VIDEO_PHASE_END.
fn video_phase_percent(videos_done: usize, videos_total: usize) -> u8 {
    if videos_total == 0 {
        return VIDEO_PHASE_END;
    }
    let span = (VIDEO_PHASE_END - FEED_PHASE_PERCENT) as usize;
    FEED_PHASE_PERCENT + ((videos_done.min(videos_total) * span) / videos_total) as u8
}

/// One video of the crawl's work list.
struct WorkItem {
    /// Index into the ordered channel list.
    channel_idx: usize,
    video: Video,
}

/// The creators named in the kept insights of one channel, most mentioned
/// first. A name gets a channel id only when a tracked channel has exactly
/// that title; everything else is left for the caller to resolve on demand.
fn related_from_mentions(
    channel_id: &str,
    kept: &[(Video, Channel, InsightResult)],
    tracked: &[Channel],
) -> Vec<RelatedChannel> {
    let own_title = tracked
        .iter()
        .find(|c| c.id == channel_id)
        .map(|c| c.title.to_lowercase());

    // One count per insight: a name repeated inside one list is one mention.
    let mut names: Vec<String> = Vec::new();
    for (_, _, insight) in kept.iter().filter(|(video, _, _)| video.channel_id == channel_id) {
        for (name, _) in rss::rank_mentions(&insight.mentioned_creators) {
            names.push(name);
        }
    }

    rss::rank_mentions(&names)
        .into_iter()
        .filter(|(name, _)| Some(name.to_lowercase()) != own_title)
        .map(|(name, mentions)| {
            let known = tracked.iter().find(|c| c.title.eq_ignore_ascii_case(&name));
            RelatedChannel {
                channel_id: known.map(|c| c.id.clone()),
                title: known.map(|c| c.title.clone()).unwrap_or(name),
                source: rss::RELATED_SOURCE_MENTIONS.to_string(),
                mentions,
                tracked: known.is_some(),
            }
        })
        .collect()
}

impl IngestionPipeline {
    pub fn new(
        db_path: &Path,
        llm_url: Option<String>,
        llm_key: Option<String>,
        llm_model: Option<String>,
    ) -> Result<Self> {
        let extractor = SubtitleExtractor::new();
        let sponsor_client = SponsorBlockClient::new();
        let analyzer = VideoAnalyzer::new(llm_url, llm_key, llm_model);
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .build()?;

        let progress = Arc::new(RwLock::new(CrawlProgress::default()));

        Ok(Self {
            db_path: db_path.to_path_buf(),
            extractor,
            sponsor_client,
            analyzer,
            http_client,
            progress,
            categorize_progress: Arc::new(RwLock::new(CategorizeProgress::default())),
            max_slop: KEEP_THRESHOLD,
            wiki_dir: PathBuf::from(DEFAULT_WIKI_DIR),
            pause_floor_secs: Arc::new(AtomicU64::new(DEFAULT_PAUSE_SECS)),
        })
    }

    /// Sets the keep threshold (0..=100) for this pipeline and its analyzer:
    /// keep decisions, crawl counters and the number in the LLM prompt.
    /// It applies to videos analysed from now on; stored rows keep their flag.
    pub fn with_max_slop(mut self, max_slop: u32) -> Self {
        self.max_slop = max_slop.min(100);
        self.analyzer = self.analyzer.with_max_slop(self.max_slop);
        self
    }

    /// Sets the directory `run_crawl_with_wiki` callers pass as the wiki target.
    pub fn with_wiki_dir<P: Into<PathBuf>>(mut self, wiki_dir: P) -> Self {
        self.wiki_dir = wiki_dir.into();
        self
    }

    pub fn db(&self) -> Result<Database> {
        Database::open(&self.db_path)
    }

    /// Stores a row for a video whose complete transcript holds too little to
    /// judge. No analyzer ran and nothing is called slop: the row says
    /// info_status "insufficient" with `reason`, is not kept, and does not
    /// count in the channel's rolling average.
    fn store_insufficient(&self, db: &Database, video: &Video, reason: &str) -> Result<InsightResult> {
        let mut insight = InsightResult {
            video_id: video.id.clone(),
            slop_score: INSUFFICIENT_PLACEHOLDER_SCORE,
            is_worth_keeping: false,
            quality_rating: "Not enough information".to_string(),
            summary: format!("Not judged: the transcript holds too little to judge the video ({}).", reason),
            key_insights: vec![],
            takeaways: vec![],
            quotes: vec![],
            topics: vec![],
            mentioned_creators: vec![],
            analyzer_type: INSUFFICIENT_ANALYZER.to_string(),
            category: None,
        };

        let judgment = InsightJudgment {
            info_status: Some(INFO_INSUFFICIENT.to_string()),
            info_reason: Some(reason.to_string()),
            ..Default::default()
        };
        db.save_insight_judged(&insight, &judgment)?;
        db.update_video_status(&video.id, STATUS_PROCESSED)?;
        self.respect_manual_category(db, &mut insight);
        self.respect_disqualification(db, &mut insight);
        Ok(insight)
    }

    /// The "too little content" check on a complete transcript: Some(reason)
    /// when the video cannot be judged (see `insufficient_reason`).
    fn insufficiency(&self, db: &Database, transcript: &CleanedTranscript) -> Option<String> {
        let words = transcript.text.split_whitespace().count();
        // The video length was stored with the transcript, when it was known.
        let duration = db
            .get_transcript_coverage(&transcript.video_id)
            .ok()
            .flatten()
            .and_then(|c| c.video_duration_sec);
        insufficient_reason(words, duration)
    }

    /// The analyzer with the categories the user has defined at this moment,
    /// so a category added or removed during a crawl applies from the next
    /// video on.
    fn analyzer_with_categories(&self, db: &Database) -> Result<VideoAnalyzer> {
        Ok(self.analyzer.clone().with_categories(db.list_categories()?))
    }

    /// Computes the keep flag from the parts and stores an analysis with its
    /// topic verdict and coverage in one step. Returns the insight as stored.
    /// A heuristic result leaves the video in status "heuristic", so a later
    /// run with an LLM analyses it again.
    fn store_analysis(&self, db: &Database, video_id: &str, analysis: Analysis) -> Result<InsightResult> {
        let Analysis { mut insight, topic_verdict, topic_reason, coverage } = analysis;

        // The wiki, the library and harvest-mentions read the stored flag, so it is
        // computed here from verdict, score and this pipeline's threshold, whatever
        // the LLM answered. The owner's disqualification is not part of the stored
        // flag: `ytc include` must be able to bring the video back.
        insight.is_worth_keeping = keep_decision(topic_verdict.as_deref(), insight.slop_score, self.max_slop, true, false);

        let judgment = InsightJudgment {
            topic_verdict,
            topic_reason,
            info_status: Some(INFO_SUFFICIENT.to_string()),
            info_reason: None,
            sections_total: Some(coverage.sections as u32),
            transcript_chars: Some(coverage.transcript_chars as u64),
        };
        db.save_insight_judged(&insight, &judgment)?;

        let status = if insight.analyzer_type == HEURISTIC_ANALYZER {
            STATUS_HEURISTIC
        } else {
            STATUS_PROCESSED
        };
        db.update_video_status(video_id, status)?;
        self.respect_manual_category(db, &mut insight);
        self.respect_disqualification(db, &mut insight);
        Ok(insight)
    }

    /// A category the owner set by hand (`ytc set-category`) outlives every
    /// analysis: the database kept it when the row was stored, and the
    /// returned insight carries it instead of the model's answer.
    fn respect_manual_category(&self, db: &Database, insight: &mut InsightResult) {
        let Ok(Some(meta)) = db.get_insight_meta(&insight.video_id) else {
            return;
        };
        if !meta.category_manual {
            return;
        }
        if let Ok(Some(stored)) = db.get_insight(&insight.video_id) {
            insight.category = stored.category;
        }
    }

    /// A video the owner disqualified (`ytc exclude`) is never kept, whatever
    /// a new analysis says. The stored row keeps the model's verdict (so
    /// `ytc include` restores it); the returned insight says "not kept".
    fn respect_disqualification(&self, db: &Database, insight: &mut InsightResult) {
        let Ok(Some(meta)) = db.get_insight_meta(&insight.video_id) else {
            return;
        };
        if !meta.excluded_manual {
            return;
        }
        let reason = match meta.excluded_reason.as_deref().map(str::trim) {
            Some(r) if !r.is_empty() => format!(" ({})", r),
            _ => String::new(),
        };
        println!(
            "    {} Disqualified by hand{}: the video stays out of the library whatever the analysis says. Undo with: ytc include {}",
            "🚫".yellow(),
            reason,
            insight.video_id
        );
        insight.is_worth_keeping = false;
    }

    /// Removes the SponsorBlock segments from an extracted transcript and
    /// stores the text together with how much of the video it covers.
    fn save_extracted(
        &self,
        db: &Database,
        video_id: &str,
        extracted: &ExtractedTranscript,
        segments: &[SponsorSegment],
    ) -> Result<CleanedTranscript> {
        let cleaned = self.sponsor_client.strip_sponsors(video_id, &extracted.cues, segments);
        if cleaned.stripped_ad_seconds > 0.0 || cleaned.stripped_segments_count > 0 {
            println!(
                "    {} Removed {:.1}s of the transcript in {} SponsorBlock segment(s) ({})",
                "✂".yellow(),
                cleaned.stripped_ad_seconds,
                cleaned.stripped_segments_count,
                crate::sponsorblock::SPONSORBLOCK_CATEGORIES.join(", ")
            );
        }

        db.save_transcript(&cleaned)?;
        // After save_transcript, which does not touch these columns. An
        // unknown video length is stored as "not recorded", never guessed.
        db.set_transcript_coverage(
            video_id,
            extracted.video_duration_sec,
            extracted.covered_until_sec,
            Some(extracted.source),
        )?;
        Ok(cleaned)
    }

    /// Fetches the transcript from YouTube (or transcribes the audio) and
    /// analyses it. A stored transcript is replaced.
    pub async fn process_video(&self, video: &Video) -> Result<InsightResult> {
        self.process_video_with(video, false, None).await
    }

    /// `reuse_stored`: when a cleaned transcript is already stored for the
    /// video (a run got that far and then stopped because the LLM was down or
    /// its answer was unusable), analyse that instead of fetching or
    /// transcribing again.
    /// `known_duration_sec`: the video length when the caller already asked
    /// yt-dlp for it, so the extractor need not ask again.
    async fn process_video_with(
        &self,
        video: &Video,
        reuse_stored: bool,
        known_duration_sec: Option<f64>,
    ) -> Result<InsightResult> {
        let db = self.db()?;

        // A stored caption track that stops before the end of the video is not
        // reused: this is the retry that can fetch the whole transcript.
        let stored = if reuse_stored { reusable_transcript(&db, &video.id) } else { None };
        let cleaned = match stored {
            Some(transcript) => {
                println!(
                    "    {} Using the stored transcript (no YouTube request).",
                    "ℹ".dimmed()
                );
                transcript
            }
            None => {
                // 1. Obtain the transcript of the whole video (YouTube captions that
                // reach the end of the video, or local Whisper on the full audio).
                // The helpers are blocking subprocesses that can run for minutes;
                // on a blocking thread they do not stall the web server. The
                // inner error is passed on unchanged, so its type stays readable.
                let extractor = self.extractor.clone();
                let video_id = video.id.clone();
                // What is remembered about a block by YouTube decides whether
                // YouTube is asked at all (see `pacing`).
                let pacing = PacingStore::for_db(&self.db_path);
                let floor = self.pause_floor_secs.load(Ordering::Relaxed);
                let before = pacing.load();
                let asked_at = now_unix();
                let plan = CaptionPlan {
                    resting_until: before.resting_until(asked_at),
                    // A video is never left waiting for a block to end.
                    whisper_when_blocked: true,
                };
                let (extracted, report) = tokio::task::spawn_blocking(move || {
                    extractor.extract_transcript(&video_id, known_duration_sec, plan)
                })
                .await
                .context("The transcript extraction task ended unexpectedly")?;
                if report.blocked {
                    let state = pacing.note_block(now_unix(), before.pause(asked_at, floor));
                    println!(
                        "    {} YouTube refused the direct transcript request. Remembered: no such requests before {}.",
                        "⏳".yellow().bold(),
                        local_clock(state.blocked_until)
                    );
                } else if report.asked {
                    pacing.note_clean(now_unix(), floor);
                }
                let extracted = extracted?;

                // 2. Fetch SponsorBlock segments
                let segments = self.sponsor_client.fetch_segments(&video.id).await.unwrap_or_default();

                // 3. Remove the segments, save the transcript and its coverage
                self.save_extracted(&db, &video.id, &extracted, &segments)?
            }
        };

        self.judge_transcript(db, video, &cleaned).await
    }

    /// Steps 3.5 to 5 of `process_video_with`: decides whether the stored
    /// transcript holds enough to judge, has it analysed and stores the result.
    async fn judge_transcript(&self, db: Database, video: &Video, cleaned: &CleanedTranscript) -> Result<InsightResult> {
        // How far the stored transcript reaches, when it is a caption track
        // that stops before the end of the video (the audio could not be
        // transcribed either).
        let partial_pct = incomplete_caption_pct(db.get_transcript_coverage(&video.id).ok().flatten().as_ref());

        // 3.5. Too little content to judge: stored as such, no analysis.
        if let Some(reason) = self.insufficiency(&db, cleaned) {
            if let Some(pct) = partial_pct {
                // The words are missing because the transcript is, not
                // because the video is silent: nothing is stored.
                return Err(incomplete_transcript_error(&video.id, pct, &reason));
            }
            println!(
                "    {} Not enough information to judge ({}). Not scored, not counted in the channel average.",
                "ℹ".yellow(),
                reason
            );
            return self.store_insufficient(&db, video, &reason);
        }

        // 4. Analyze the whole stored transcript
        match partial_pct {
            Some(pct) => println!(
                "    {} Analysing with {}: the {} only. The rest of the video has no transcript.",
                "⚠".yellow(),
                self.analyzer.engine_label(),
                partial_transcript_label(pct)
            ),
            None => println!("    {} Analysing with {} ...", "🧠".dimmed(), self.analyzer.engine_label()),
        }
        let analyzer = self.analyzer_with_categories(&db)?;
        let analysis = analyzer.analyze_with_coverage(&video.title, cleaned).await?;

        // 5. Save insight, verdict, coverage and video status
        self.store_analysis(&db, &video.id, analysis)
    }

    /// Analyses an already ingested video again from its stored cleaned
    /// transcript. No YouTube request is made.
    ///
    /// The stored insight is replaced only when the LLM returned a valid
    /// answer (or the transcript turns out to hold too little to judge, which
    /// is stored as "not enough information"; not so when the stored transcript
    /// is a caption track that stops before the end of the video: that is an
    /// error, see `incomplete_transcript_error`). Without a configured LLM,
    /// without a stored transcript, or when the LLM is down or answers
    /// unusably, this returns an error and the stored row stays as it was.
    ///
    /// A video the owner disqualified stays disqualified: the new analysis is
    /// stored, the returned insight is not kept.
    pub async fn reanalyze_video(&self, video_id: &str) -> Result<InsightResult> {
        if !self.analyzer.has_llm() {
            anyhow::bail!(
                "Re-analysis needs an LLM and none is configured. The stored insight of {} was left unchanged.",
                video_id
            );
        }

        // Own scope: the connection is not held while the LLM works.
        let (video, transcript, too_little, partial_pct) = {
            let db = self.db()?;
            let video = db
                .get_video(video_id)?
                .ok_or_else(|| anyhow::anyhow!("Video {} is not in the database", video_id))?;
            let transcript = db.get_transcript(video_id)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "No stored transcript for {}. Run process-video to fetch it from YouTube.",
                    video_id
                )
            })?;
            if transcript.text.trim().is_empty() {
                anyhow::bail!("The stored transcript of {} is empty", video_id);
            }
            let too_little = self.insufficiency(&db, &transcript);
            let partial_pct = incomplete_caption_pct(db.get_transcript_coverage(video_id).ok().flatten().as_ref());
            (video, transcript, too_little, partial_pct)
        };

        if let Some(reason) = too_little {
            if let Some(pct) = partial_pct {
                // Not "not enough information": the transcript is what is
                // missing. The stored row stays as it was.
                return Err(incomplete_transcript_error(video_id, pct, &reason));
            }
            println!(
                "    {} Not enough information to judge ({}). Not scored, not counted in the channel average.",
                "ℹ".yellow(),
                reason
            );
            return self.store_insufficient(&self.db()?, &video, &reason);
        }

        match partial_pct {
            Some(pct) => println!(
                "    {} Re-analysing with {}: the {} only (fetch the whole one with: ytc process-video {}).",
                "⚠".yellow(),
                self.analyzer.engine_label(),
                partial_transcript_label(pct),
                video_id
            ),
            None => println!(
                "    {} Re-analysing the stored transcript with {} ...",
                "🧠".dimmed(),
                self.analyzer.engine_label()
            ),
        }
        // On any error nothing has been written yet, so the stored insight stays.
        let analyzer = self.analyzer_with_categories(&self.db()?)?;
        let analysis = analyzer
            .analyze_with_coverage(&video.title, &transcript)
            .await
            .with_context(|| format!("Stored insight of {} left unchanged", video_id))?;

        let db = self.db()?;
        self.store_analysis(&db, &video.id, analysis)
    }

    /// Gives already analysed videos a category without analysing them
    /// again: one short LLM request per video, built from its stored insight
    /// (`VideoAnalyzer::categorize`). Nothing but `insights.category` is
    /// written.
    ///
    /// Worked on are the videos an LLM judged that have no category and
    /// whose category was not set by hand; with `all` also those the model
    /// already put into a category. `limit` stops after that many videos.
    ///
    /// Fails before the first request when no LLM is configured, when it
    /// does not answer, or when the user has no categories. During the run
    /// the stop rule of the crawl applies (`stops_crawl`): at once when the
    /// engine is down, otherwise after two failures in a row, an unreadable
    /// answer included. The run also ends when the categories are changed
    /// while it works. The videos done until then keep their category;
    /// `CategorizeSummary::stopped` says why the run ended.
    pub async fn run_categorize(&self, all: bool, limit: Option<usize>) -> Result<CategorizeSummary> {
        if !self.analyzer.has_llm() {
            anyhow::bail!("Assigning categories needs an LLM and none is configured (--llm-url / YTC_LLM_URL). Nothing was changed.");
        }
        // Own scope: the connection is not held while the LLM works.
        let (categories, mut targets) = {
            let db = self.db()?;
            (db.list_categories()?, db.list_categorize_targets(all)?)
        };
        if categories.is_empty() {
            anyhow::bail!("No categories are defined, so there is nothing to assign. Add one first (ytc category add <name>). Nothing was changed.");
        }
        let mut summary = CategorizeSummary {
            assigned: categories.iter().map(|c| (c.name.clone(), 0)).collect(),
            ..Default::default()
        };
        if let Some(limit) = limit {
            summary.beyond_limit = targets.len().saturating_sub(limit);
            targets.truncate(limit);
        }
        summary.total = targets.len();
        let total = targets.len();

        println!(
            "{} video(s) ahead: {}; one short LLM request each ({}). The transcripts are not read again.",
            total,
            if all {
                "every LLM-judged video whose category was not set by hand"
            } else {
                "LLM-judged videos without a category"
            },
            self.analyzer.engine_label()
        );
        if summary.beyond_limit > 0 {
            println!("{} further video(s) are left for a later run (--limit).", summary.beyond_limit);
        }
        if total == 0 {
            self.finish_categorize(&summary).await;
            return Ok(summary);
        }
        let (_, _, url) = self.analyzer.describe();
        if !self.analyzer.llm_reachable().await {
            anyhow::bail!("The LLM at {} is not reachable (no answer from /health or /models within 2 s). Nothing was changed.", url);
        }

        {
            let mut p = self.categorize_progress.write().await;
            *p = CategorizeProgress {
                is_running: true,
                stage: "running".to_string(),
                total_videos: total,
                message: format!("Assigning categories to {} video(s)...", total),
                ..Default::default()
            };
        }

        let analyzer = self.analyzer.clone().with_categories(categories.clone());
        let mut llm_failures_in_a_row = 0usize;
        let mut attempted = 0usize;
        for (idx, id) in targets.iter().enumerate() {
            let (title, insight) = {
                let db = self.db()?;
                let title = db.get_video(id)?.map(|v| v.title).unwrap_or_else(|| id.clone());
                (title, db.get_insight(id)?)
            };
            attempted += 1;
            {
                let mut p = self.categorize_progress.write().await;
                p.current_index = idx + 1;
                p.current_video_id = Some(id.clone());
                p.current_video_title = Some(title.clone());
                p.percent = (idx * 100 / total) as u8;
                p.message = format!("[{}/{}] {}", idx + 1, total, title);
            }
            let shown: String = title.chars().take(60).collect();
            println!("[{}/{}] {} {}", idx + 1, total, id.cyan(), shown);

            // Gone since the list was read: nothing to label.
            let Some(insight) = insight else {
                summary.skipped += 1;
                continue;
            };
            match analyzer.categorize(&title, &insight).await {
                Ok(category) => {
                    llm_failures_in_a_row = 0;
                    // The answer was chosen from the list this run started
                    // with. If the owner changed the categories meanwhile it
                    // is not stored, and the rest of the run would be asked
                    // with the wrong list: the run ends here.
                    if self.db()?.list_categories()? != categories {
                        attempted -= 1;
                        summary.stopped = Some("Your categories were changed during the run".to_string());
                        break;
                    }
                    if !self.db()?.set_model_category(id, category.as_deref())? {
                        summary.skipped += 1;
                        println!("    {} category set by hand meanwhile: left as it is", "ℹ".yellow());
                        continue;
                    }
                    match category.as_deref() {
                        Some(name) => {
                            if let Some(entry) = summary.assigned.iter_mut().find(|(n, _)| n == name) {
                                entry.1 += 1;
                            }
                            println!("    {} {}", "✓".green(), name);
                        }
                        None => {
                            summary.none += 1;
                            println!("    {} no category fits", "✓".green());
                        }
                    }
                }
                Err(e) => {
                    summary.failed += 1;
                    println!("    {} category unchanged: {:#}", "✗".red(), e);
                    match e.chain().find_map(|cause| cause.downcast_ref::<LlmUnavailable>()) {
                        Some(unavailable) => {
                            llm_failures_in_a_row += 1;
                            if stops_crawl(unavailable, llm_failures_in_a_row) {
                                summary.stopped = Some(if unavailable.engine_down {
                                    "The LLM stopped answering".to_string()
                                } else {
                                    format!("The LLM failed for {} videos in a row", llm_failures_in_a_row)
                                });
                                break;
                            }
                        }
                        // An answer that cannot be read. The request is the
                        // same short one for every video, so two in a row
                        // mean the model does not manage it: going through
                        // the whole library would change nothing.
                        None => {
                            llm_failures_in_a_row += 1;
                            if llm_failures_in_a_row >= 2 {
                                summary.stopped = Some(format!(
                                    "The LLM's answer could not be read for {} videos in a row",
                                    llm_failures_in_a_row
                                ));
                                break;
                            }
                        }
                    }
                }
            }
            let mut p = self.categorize_progress.write().await;
            p.assigned = summary.assigned_total();
            p.none = summary.none;
            p.failed = summary.failed;
        }
        summary.not_attempted = total - attempted;
        if let Some(why) = &summary.stopped {
            println!(
                "    {} {}; the run stops here and the remaining {} video(s) keep their category.",
                "⚠".yellow(),
                why,
                summary.not_attempted
            );
        }
        self.finish_categorize(&summary).await;
        Ok(summary)
    }

    /// Writes the end state of a `run_categorize` run into its progress.
    async fn finish_categorize(&self, summary: &CategorizeSummary) {
        let mut p = self.categorize_progress.write().await;
        p.is_running = false;
        p.total_videos = summary.total;
        p.assigned = summary.assigned_total();
        p.none = summary.none;
        p.failed = summary.failed;
        p.current_video_id = None;
        p.current_video_title = None;
        match &summary.stopped {
            Some(why) => {
                p.stage = "error".to_string();
                p.message = format!("{}: {}.", why, summary.counts_line());
            }
            None => {
                p.stage = "complete".to_string();
                p.percent = 100;
                p.message = format!("Categories assigned: {}.", summary.counts_line());
            }
        }
    }

    /// Creators named in the kept insights of a channel (local database only).
    pub fn mentioned_creators_of_channel(&self, channel_id: &str) -> Result<Vec<RelatedChannel>> {
        let db = self.db()?;
        let kept = db.get_all_kept_videos_with_details()?;
        let tracked = db.list_channels()?;
        Ok(related_from_mentions(channel_id, &kept, &tracked))
    }

    /// Related-channel candidates of a channel, each labelled with its source.
    /// 1. Creators mentioned in this channel's kept insights (database; names
    ///    are resolved to a channel only when the user picks one).
    /// 2. With `include_page`: channels featured on the creator's YouTube page
    ///    (one request).
    /// A failing page request is an error only when source 1 found nothing.
    pub async fn related_channels(&self, channel_id: &str, include_page: bool) -> Result<Vec<RelatedChannel>> {
        let mut related = self.mentioned_creators_of_channel(channel_id)?;
        if !include_page {
            return Ok(related);
        }

        match rss::fetch_related_channels(&self.http_client, channel_id).await {
            Ok(featured) => {
                let tracked_ids: Vec<String> = self.db()?.list_channels()?.into_iter().map(|c| c.id).collect();
                for (id, title) in featured {
                    if related.iter().any(|r| r.channel_id.as_deref() == Some(id.as_str())) {
                        continue;
                    }
                    related.push(RelatedChannel {
                        tracked: tracked_ids.contains(&id),
                        channel_id: Some(id),
                        title,
                        source: rss::RELATED_SOURCE_PAGE.to_string(),
                        mentions: 0,
                    });
                }
            }
            Err(e) if related.is_empty() => return Err(e),
            Err(e) => eprintln!("  [Related] Channel page not read ({}). Showing mentions only.", e),
        }
        Ok(related)
    }

    /// Crawl without a wiki step (the caller writes the wiki, or does not want one).
    pub async fn run_crawl(
        &self,
        channel_filter: Option<String>,
        limit_per_channel: usize,
        delay_secs: u64,
    ) -> Result<CrawlSummary> {
        self.run_crawl_with_wiki(channel_filter, limit_per_channel, delay_secs, None).await
    }

    /// Crawl, then write the wiki to `self.wiki_dir` as part of the same run
    /// (progress stays "running" until the wiki is written). Callers must not
    /// generate the wiki again afterwards.
    pub async fn run_crawl_and_wiki(
        &self,
        channel_filter: Option<String>,
        limit_per_channel: usize,
        delay_secs: u64,
    ) -> Result<CrawlSummary> {
        let wiki_dir = self.wiki_dir.clone();
        self.run_crawl_with_wiki(channel_filter, limit_per_channel, delay_secs, Some(&wiki_dir)).await
    }

    /// Writes the wiki as the last step of a crawl and records the outcome.
    /// Not async: a `&Database` must not be held across an await.
    fn write_wiki_step(db: &Database, wiki_dir: &Path, summary: &mut CrawlSummary) {
        println!("\n{} Writing the wiki to {} ...", "📚".cyan(), wiki_dir.display());
        match WikiGenerator::generate(db, wiki_dir) {
            Ok(stats) => {
                summary.wiki_generated = true;
                println!(
                    "  Wiki written: {} insights, {} topics, {} creators.",
                    stats.insights_count, stats.topics_count, stats.creators_count
                );
            }
            Err(e) => {
                println!("  {} Wiki could not be written: {}", "✗".red(), e);
                summary.wiki_error = Some(e.to_string());
            }
        }
    }

    /// Crawls the tracked channels (or one channel) in two phases.
    ///
    /// Phase 1 fetches every feed and builds the list of videos that really
    /// need work, so the total is known before the first video is processed.
    /// Phase 2 processes exactly that list.
    ///
    /// With `wiki_dir` the wiki is written as part of the crawl, while
    /// `is_running` is still true; the caller must not generate it again.
    /// A wiki failure does not discard the crawl result: the summary carries
    /// `wiki_error` and the progress ends in stage "error".
    pub async fn run_crawl_with_wiki(
        &self,
        channel_filter: Option<String>,
        limit_per_channel: usize,
        delay_secs: u64,
        wiki_dir: Option<&Path>,
    ) -> Result<CrawlSummary> {
        let db = self.db()?;
        let channels = if let Some(target) = channel_filter {
            let target = target.trim().to_string();
            // The RSS feed only exists for the UC id. A handle or URL used as the id
            // returned an error page that parsed as "0 new videos".
            let (ch_id, ch_title) = if target.starts_with("UC") && target.len() == 24 {
                let title = db
                    .get_channel(&target)
                    .ok()
                    .flatten()
                    .map(|c| c.title)
                    .unwrap_or_else(|| target.clone());
                (target, title)
            } else {
                rss::resolve_channel(&self.http_client, &target)
                    .await
                    .with_context(|| format!("Could not resolve channel '{}' to a YouTube channel ID", target))?
            };
            vec![Channel {
                url: format!("https://www.youtube.com/channel/{}", ch_id),
                id: ch_id,
                title: ch_title,
                status: "active".to_string(),
                rolling_slop_score: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                last_crawled_at: None,
            }]
        } else {
            db.list_channels()?
        };

        // high_signal first, then least recently crawled (see crawl_order).
        let active_channels = crawl_order(channels);
        let has_llm = self.analyzer.has_llm();

        {
            let mut p = self.progress.write().await;
            *p = CrawlProgress {
                is_running: true,
                stage: "starting".to_string(),
                message: format!(
                    "Starting crawl across {} channel(s) (up to {} new video(s) each)...",
                    active_channels.len(),
                    limit_per_channel
                ),
                ..Default::default()
            };
        }

        if active_channels.len() > 1 {
            println!("\nCrawl order ({} channels; high_signal first, then least recently crawled):", active_channels.len());
            for (i, ch) in active_channels.iter().enumerate() {
                println!(
                    "  {:>2}. {} [{}] last crawled: {}",
                    i + 1,
                    ch.title,
                    ch.status,
                    ch.last_crawled_at.as_deref().unwrap_or("never")
                );
            }
        }

        let mut summary = CrawlSummary::default();

        // ---- Phase 1: fetch the feeds, build the work list ----------------------
        let mut work: Vec<WorkItem> = Vec::new();
        // Error of the last feed that could not be fetched, for the message.
        let mut last_feed_error: Option<String> = None;
        // A feed that does not answer properly is asked once more after all the
        // others: YouTube's feeds fail in short bursts.
        let mut feed_queue: std::collections::VecDeque<(usize, bool)> =
            (0..active_channels.len()).map(|i| (i, false)).collect();
        let mut waited_before_second_round = false;
        while let Some((channel_idx, second_try)) = feed_queue.pop_front() {
            let ch = &active_channels[channel_idx];
            if second_try && !waited_before_second_round {
                waited_before_second_round = true;
                println!(
                    "\n  {} {} channel(s) could not be read. Asking them once more in {}s...",
                    "↻".yellow(),
                    feed_queue.len() + 1,
                    FEED_RETRY_PAUSE_SECS
                );
                sleep(Duration::from_secs(FEED_RETRY_PAUSE_SECS)).await;
            }
            {
                let mut p = self.progress.write().await;
                p.current_channel_id = Some(ch.id.clone());
                p.current_channel_title = Some(ch.title.clone());
                p.stage = "checking_feed".to_string();
                p.percent = feed_phase_percent(channel_idx, active_channels.len());
                p.message = format!(
                    "Checking feed [{}/{}]: {}",
                    channel_idx + 1,
                    active_channels.len(),
                    ch.title
                );
            }
            println!("\n▶ Checking RSS feed for channel: {} ({})", ch.title.bold(), ch.id.dimmed());
            let feed = match rss::fetch_channel_feed(&self.http_client, &ch.id).await {
                Ok(f) => f,
                Err(feed_err) => {
                    // YouTube's feed address fails for stretches of time, with 404
                    // or 500, also for channels that exist: the channel's video
                    // list is read with yt-dlp instead.
                    println!("  {} Feed not fetched ({:#}). Reading the channel's video list instead...", "↻".yellow(), feed_err);
                    let extractor = self.extractor.clone();
                    let channel_id = ch.id.clone();
                    let wanted = limit_per_channel.saturating_add(CHANNEL_LIST_EXTRA);
                    let listed = tokio::task::spawn_blocking(move || extractor.list_channel_videos(&channel_id, wanted))
                        .await
                        .unwrap_or_else(|e| Err(anyhow::anyhow!("the video list task ended unexpectedly: {}", e)));
                    match listed {
                        Ok(videos) if !videos.is_empty() => rss::ChannelFeed {
                            channel_id: ch.id.clone(),
                            channel_title: ch.title.clone(),
                            videos: videos.into_iter().map(feed_item_from_list).collect(),
                        },
                        other => {
                            let list_err = match other {
                                Err(e) => format!("{:#}", e),
                                Ok(_) => "the video list was empty".to_string(),
                            };
                            if channel_is_gone(&list_err) {
                                println!(
                                    "  {} YouTube says this channel does not exist: it was removed, or the stored channel ID is wrong. Mute or re-add it ({}).",
                                    "✗".red(),
                                    ch.id
                                );
                                summary.feeds_failed += 1;
                                last_feed_error = Some(format!("{}: {}", ch.id, list_err));
                                continue;
                            }
                            if !second_try {
                                println!("  {} That failed too ({}). The channel is asked once more at the end.", "↻".yellow(), list_err);
                                feed_queue.push_back((channel_idx, true));
                                continue;
                            }
                            println!("  {} Failed to fetch RSS feed: {:#}; video list: {}", "✗".red(), feed_err, list_err);
                            summary.feeds_failed += 1;
                            last_feed_error = Some(format!("{:#}; video list: {}", feed_err, list_err));
                            continue;
                        }
                    }
                }
            };
            // Only now: a feed that could not be fetched was not checked.
            summary.channels_checked += 1;

            let mut queued_for_channel = 0;
            for v_item in feed.videos {
                // A video that fails later still used its slot, so one bad video
                // cannot push the channel's other videos out crawl after crawl.
                if queued_for_channel >= limit_per_channel {
                    break;
                }

                // Skip vertical shorts to conserve non-RSS budget for rich content
                if has_shorts_hashtag(&v_item.title) {
                    println!("    {} Skipping Short/clip to prioritize deep content: {}", "⏭".dimmed(), v_item.title.dimmed());
                    continue;
                }

                let (status, fail_count) = video_state(&db, &v_item.video_id);
                if plan_for_video(status.as_deref(), fail_count, has_llm) == VideoPlan::Skip {
                    if fail_count >= MAX_VIDEO_FAILURES {
                        println!(
                            "    {} Given up after {} failed attempts: {} (retry with: ytc process-video {})",
                            "⏭".dimmed(),
                            fail_count,
                            v_item.title.dimmed(),
                            v_item.video_id
                        );
                    }
                    continue;
                }

                work.push(WorkItem {
                    channel_idx,
                    video: Video {
                        id: v_item.video_id,
                        channel_id: ch.id.clone(),
                        title: v_item.title,
                        url: v_item.url,
                        published_at: v_item.published_at,
                        status: "pending".to_string(),
                        created_at: chrono::Utc::now().to_rfc3339(),
                    },
                });
                queued_for_channel += 1;
            }

            if queued_for_channel == 0 {
                // Nothing to do for this channel: it counts as crawled now.
                let _ = db.update_channel_crawled(&ch.id);
                if let Err(e) = db.recalculate_channel_stats(&ch.id) {
                    eprintln!("  [Crawl] Could not update the rolling slop score of {}: {}", ch.id, e);
                }
            } else {
                println!("  {} new video(s) queued.", queued_for_channel);
            }
        }

        let mut total = work.len();
        let feeds_line = feeds_checked_line(summary.channels_checked, summary.feeds_failed, total);
        {
            let mut p = self.progress.write().await;
            p.percent = FEED_PHASE_PERCENT;
            p.total_videos = total;
            p.feeds_failed = summary.feeds_failed;
            p.message = feeds_line.clone();
        }
        println!("\n{}", feeds_line);

        // Not one feed answered: that is not a crawl with "nothing new".
        if summary.feeds_failed > 0 && summary.channels_checked == 0 {
            let message = format!(
                "None of the {} feed(s) could be fetched, so nothing was checked. YouTube may be blocking this IP, or the network is down. Last error: {}",
                summary.feeds_failed,
                last_feed_error.unwrap_or_default()
            );
            let mut p = self.progress.write().await;
            p.is_running = false;
            p.stage = "error".to_string();
            p.message = message.clone();
            return Err(anyhow::anyhow!(message));
        }

        // ---- Phase 2: process exactly the work list -----------------------------
        // What is remembered about YouTube's limits (see `pacing`): the pause
        // between videos, and whether transcript requests rest after a block.
        let pacing = PacingStore::for_db(&self.db_path);
        // A crawl never goes faster than this, whatever pause is set.
        let delay_secs = delay_secs.max(MIN_CRAWL_PAUSE_SECS);
        self.pause_floor_secs.store(delay_secs, Ordering::Relaxed);
        // Set while YouTube refuses transcript requests: videos that need
        // YouTube are then left pending instead of being asked for one by one.
        let mut youtube_waiting: Option<i64> = None;
        if let Some(until) = pacing.load().resting_until(now_unix()) {
            println!(
                "\n  {} YouTube refused direct transcript requests earlier: none are made before {}. Transcripts are read from the YouTube page in a browser; where that fails, the audio is transcribed locally.",
                "⏳".yellow().bold(),
                local_clock(until)
            );
        }

        // Set when the LLM engine is down: the crawl stops instead of marking every
        // remaining video as failed (and hitting YouTube for nothing).
        let mut llm_unavailable: Option<String> = None;
        // A single timeout or 5xx fails only that video; two in a row stop the crawl.
        let mut llm_failures_in_a_row = 0;
        // Videos finished, whatever the outcome: the n of "[n/total]".
        let mut done = 0usize;
        // Failures that are the video's own fault. Counted after the loop.
        let mut pending_failures: Vec<PendingFailure> = Vec::new();

        for (pos, item) in work.iter().enumerate() {
            let ch = &active_channels[item.channel_idx];
            let video = &item.video;
            let last_of_channel = work.get(pos + 1).map_or(true, |next| next.channel_idx != item.channel_idx);

            // The list was built minutes ago: another run (discover, a second
            // crawl, process-video) may have dealt with the video meanwhile.
            let (status, fail_count) = video_state(&db, &video.id);
            let plan = plan_for_video(status.as_deref(), fail_count, has_llm);

            let waits_for_youtube = plan != VideoPlan::Skip
                && youtube_waiting.is_some()
                && reusable_transcript(&db, &video.id).is_none();

            if plan == VideoPlan::Skip {
                total = total.saturating_sub(1);
                let mut p = self.progress.write().await;
                p.total_videos = total;
                p.percent = video_phase_percent(done, total);
            } else if waits_for_youtube {
                // Known to the library, so it is not lost when it leaves the feed.
                let _ = db.upsert_video(video);
                summary.videos_waiting += 1;
                total = total.saturating_sub(1);
                let mut p = self.progress.write().await;
                p.total_videos = total;
                p.percent = video_phase_percent(done, total);
            } else {
                {
                    let mut p = self.progress.write().await;
                    p.current_channel_id = Some(ch.id.clone());
                    p.current_channel_title = Some(ch.title.clone());
                    p.current_video_id = Some(video.id.clone());
                    p.current_video_title = Some(video.title.clone());
                    p.stage = "evaluating".to_string();
                    p.current_index = done + 1;
                    p.total_videos = total;
                    p.message = format!("Evaluating [{}/{}]: {}", done + 1, total, video.title);
                }
                println!(
                    "\n  • [{}/{}] Processing: {} ({}) — {}",
                    done + 1,
                    total,
                    video.title.bold(),
                    video.id.cyan(),
                    ch.title.dimmed()
                );

                // A stored transcript is analysed without YouTube: for a
                // heuristic row (re-analysis), and for a video of which an
                // attempt got as far as the transcript before the LLM was down
                // or answered unusably. Everything else goes the full way.
                // (A stored caption track that stops before the end of the
                // video is not such a transcript: it is fetched again.)
                let from_stored = reusable_transcript(&db, &video.id).is_some();
                let reanalysis = plan == VideoPlan::Reanalyze && from_stored;
                let result = if reanalysis {
                    self.reanalyze_video(&video.id).await
                } else {
                    let _ = db.upsert_video(video);
                    self.process_video_with(video, true, None).await
                };
                done += 1;

                let mut rate_limited = false;
                // Set when the network was down for this video: a longer pause follows.
                let mut no_connection = false;
                match result {
                    Ok(insight) => {
                        let meta = db
                            .get_insight_meta(&video.id)
                            .ok()
                            .flatten()
                            .unwrap_or_else(|| meta_fallback(&insight));
                        let (outcome, line) = outcome_report(&insight, &meta, self.max_slop);
                        summary.count(outcome);
                        let partial_pct =
                            incomplete_caption_pct(db.get_transcript_coverage(&video.id).ok().flatten().as_ref());
                        let line = with_read_note(line, outcome, partial_pct);
                        println!("    ✓ {}", paint_outcome(outcome, &line));
                        llm_failures_in_a_row = 0;
                    }
                    Err(e) => {
                        let llm_err = e.downcast_ref::<LlmUnavailable>();
                        let cause = failure_cause(&e);
                        rate_limited = cause == FailureCause::RateLimited;
                        if rate_limited {
                            // Not a failure of the video or the program: it waits.
                            println!("    {} Left pending: {:#}", "⏳".yellow(), e);
                            summary.videos_waiting += 1;
                        } else {
                            println!("    {} Failed to process: {:#}", "✗".red(), e);
                            summary.videos_failed += 1;
                        }

                        // A video fails "for its own reasons" only when nothing
                        // else is to blame. Only then is the failure counted
                        // (at the end of the run), and after MAX_VIDEO_FAILURES
                        // the video is given up.
                        match cause {
                            FailureCause::Video => pending_failures.push(PendingFailure {
                                video_id: video.id.clone(),
                                reanalysis,
                            }),
                            FailureCause::NotAvailableYet => {
                                println!(
                                    "    {} Not available yet (premiere or upcoming live stream). Not counted as a failure; the next crawl tries again.",
                                    "ℹ".dimmed()
                                );
                                let _ = db.update_video_status(&video.id, "failed");
                            }
                            FailureCause::HelperMissing => {
                                println!(
                                    "    {} A helper is missing (Python, yt-dlp, faster-whisper or a ytc script). Not counted as a failure of the video; run the setup check, then crawl again.",
                                    "⚠".yellow()
                                );
                                let _ = db.update_video_status(&video.id, "failed");
                            }
                            FailureCause::NoConnection => {
                                println!(
                                    "    {} YouTube could not be reached (network or name resolution). Not counted as a failure of the video; it is tried again. Waiting {}s before the next video.",
                                    "ℹ".dimmed(),
                                    NO_CONNECTION_PAUSE_SECS
                                );
                                no_connection = true;
                            }
                            // The video stays as it is and is tried again.
                            FailureCause::Llm | FailureCause::RateLimited => {}
                        }

                        if let Some(llm_err) = llm_err {
                            llm_failures_in_a_row += 1;
                            if stops_crawl(llm_err, llm_failures_in_a_row) {
                                llm_unavailable = Some(format!("{} {}", e, llm_stop_hint(llm_err)));
                            }
                        }
                    }
                }

                {
                    let mut p = self.progress.write().await;
                    p.videos_evaluated = summary.videos_evaluated;
                    p.videos_kept = summary.videos_kept;
                    p.videos_filtered = summary.videos_filtered_slop;
                    p.videos_failed = summary.videos_failed;
                    p.videos_off_topic = summary.videos_off_topic;
                    p.videos_insufficient = summary.videos_insufficient;
                    p.videos_disqualified = summary.videos_disqualified;
                    p.percent = video_phase_percent(done, total);
                    p.message = format!("Done [{}/{}]: {}", done, total, video.title);
                }

                if llm_unavailable.is_some() {
                    break;
                }

                if rate_limited && youtube_waiting.is_none() {
                    // Asking again for every remaining video would only keep the
                    // block alive: they wait, and so does the rest of this run.
                    let now = now_unix();
                    let until = pacing.load().resting_until(now).unwrap_or(now);
                    youtube_waiting = Some(until);
                    println!(
                        "    {} YouTube is refusing requests. The remaining videos that need YouTube stay pending; a crawl after {} takes them up.",
                        "⏳".yellow().bold(),
                        local_clock(until)
                    );
                }

                // No pause after the last video, after one that made no YouTube
                // request, or when YouTube is not asked any more in this run.
                // The pause is the remembered one, never below the user's, plus
                // an irregular 1 to 2 seconds.
                let active_delay = if no_connection {
                    NO_CONNECTION_PAUSE_SECS
                } else {
                    pacing.load().pause(now_unix(), delay_secs)
                };
                if active_delay > 0 && !from_stored && youtube_waiting.is_none() && pos + 1 < work.len() {
                    let millis = jittered_millis(active_delay, clock_seed());
                    println!(
                        "    {} Pause before the next video: {:.1}s",
                        "⏱".dimmed(),
                        millis as f64 / 1000.0
                    );
                    sleep(Duration::from_millis(millis)).await;
                }
            }

            if last_of_channel {
                let _ = db.update_channel_crawled(&ch.id);
                if let Err(e) = db.recalculate_channel_stats(&ch.id) {
                    eprintln!("  [Crawl] Could not update the rolling slop score of {}: {}", ch.id, e);
                }
            }
        }

        // Now it is known whether anything worked in this run.
        record_run_failures(&db, &pending_failures, summary.videos_judged(), summary.videos_failed);

        // The wiki is written while is_running is still true, also after an early
        // stop: the videos analysed before it are in the database.
        if let Some(dir) = wiki_dir {
            if llm_unavailable.is_none() || summary.videos_kept > 0 {
                {
                    let mut p = self.progress.write().await;
                    p.stage = "updating_wiki".to_string();
                    p.percent = WIKI_PHASE_PERCENT;
                    p.message = format!("Writing the wiki to {} ...", dir.display());
                }
                Self::write_wiki_step(&db, dir, &mut summary);
            }
        }

        let counts = summary.counts_line();

        if let Some(reason) = llm_unavailable {
            let mut p = self.progress.write().await;
            p.is_running = false;
            p.stage = "error".to_string();
            p.message = format!("Crawl stopped at video {} of {}: {}", done, total, reason);
            // `reason` ends with what to do (`llm_stop_hint`).
            return Err(anyhow::anyhow!(
                "Crawl stopped at video {} of {} ({}): {}",
                done,
                total,
                counts,
                reason
            ));
        }

        {
            let mut p = self.progress.write().await;
            p.is_running = false;
            match &summary.wiki_error {
                Some(wiki_err) => {
                    p.stage = "error".to_string();
                    p.message = format!("Crawl finished ({}), but the wiki could not be written: {}", counts, wiki_err);
                }
                None => {
                    p.stage = "complete".to_string();
                    p.percent = 100;
                    p.message = format!("Crawl complete: {}.", counts);
                }
            }
        }

        Ok(summary)
    }

    pub async fn run_discover(
        &self,
        query: &str,
        limit: usize,
        delay_secs: u64,
    ) -> Result<DiscoverSummary> {
        let db = self.db()?;
        let candidates = rss::search_youtube(&self.http_client, query, limit).await?;
        let has_llm = self.analyzer.has_llm();
        let mut items = Vec::new();

        for (idx, item) in candidates.iter().enumerate() {
            let url = format!("https://www.youtube.com/watch?v={}", item.video_id);
            println!("Candidate [{}/{}]: Processing {} ...", idx + 1, candidates.len(), item.video_id.cyan());

            // Already analysed (by a discover run or the RSS crawl): report the
            // stored result. Re-processing cost a transcript fetch plus an LLM pass per
            // cycle and overwrote the stored title, channel and publish date.
            // Discover is an explicit request, so the failure limit of the crawl
            // does not apply here (fail count 0).
            let existing = db.get_video(&item.video_id).ok().flatten();
            let plan = plan_for_video(existing.as_ref().map(|v| v.status.as_str()), 0, has_llm);
            if let (Some(existing), VideoPlan::Skip | VideoPlan::Reanalyze) = (&existing, plan) {
                // A heuristic row gets its LLM analysis now, from the stored
                // transcript. If that is not possible the stored row is reported.
                let fresh = if plan == VideoPlan::Reanalyze {
                    match self.reanalyze_video(&existing.id).await {
                        Ok(insight) => Some(insight),
                        Err(e) => {
                            println!("    {} Re-analysis failed: {:#}", "✗".red(), e);
                            if e.downcast_ref::<LlmUnavailable>().is_some() {
                                return Err(e);
                            }
                            None
                        }
                    }
                } else {
                    None
                };
                let stored = match fresh {
                    Some(insight) => Some(insight),
                    None => db.get_insight(&existing.id).ok().flatten(),
                };
                if let Some(stored) = stored {
                    println!("    {} Already in the library: {}", "⏭".dimmed(), existing.title.dimmed());
                    let meta = db
                        .get_insight_meta(&existing.id)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| meta_fallback(&stored));
                    items.push(discovered_item(existing.title.clone(), stored, &meta, None));
                }
                continue;
            }

            // Without the real title the LLM would judge a placeholder, and the
            // placeholder would be stored and published.
            // One yt-dlp call gives title, channel and upload date.
            // yt-dlp is a blocking subprocess: off the async worker, like the
            // transcript helpers.
            let extractor = self.extractor.clone();
            let candidate_id = item.video_id.clone();
            let meta = tokio::task::spawn_blocking(move || extractor.fetch_video_meta(&candidate_id))
                .await
                .ok()
                .flatten();
            let Some(meta) = meta else {
                println!("    {} Could not fetch the title, candidate skipped.", "⚠".yellow());
                continue;
            };
            let real_title = meta.title.clone();
            let ch_id = meta.channel_id.clone().unwrap_or_else(|| "unknown".to_string());
            let ch_title = meta.channel.clone().unwrap_or_else(|| "Unknown Creator".to_string());
            let now = chrono::Utc::now().to_rfc3339();
            let published_at = match meta.upload_date {
                Some(date) => date,
                None => {
                    // The column cannot be empty; say that this date is not the real one.
                    println!(
                        "    {} Upload date not reported by yt-dlp; today's date is stored instead.",
                        "⚠".yellow()
                    );
                    now.clone()
                }
            };

            let temp_video = Video {
                id: item.video_id.clone(),
                channel_id: ch_id.clone(),
                title: real_title.clone(),
                url: url.clone(),
                published_at,
                status: "discovered".to_string(),
                created_at: now,
            };
            let _ = db.upsert_video(&temp_video);

            let mut auto_promoted = None;
            // The length is already known from the metadata call above.
            let known_duration = meta.duration_sec.map(f64::from);
            match self.process_video_with(&temp_video, false, known_duration).await {
                Ok(insight) => {
                    let meta = db
                        .get_insight_meta(&temp_video.id)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| meta_fallback(&insight));
                    let (outcome, line) = outcome_report(&insight, &meta, self.max_slop);
                    let partial_pct =
                        incomplete_caption_pct(db.get_transcript_coverage(&temp_video.id).ok().flatten().as_ref());
                    let line = with_read_note(line, outcome, partial_pct);
                    println!("    ✓ {}", paint_outcome(outcome, &line));
                    let kept = outcome == VideoOutcome::Kept;

                    // Add the creator to the watchlist if the video is high-signal.
                    // Only a channel that is not tracked yet: adding an existing one
                    // would undo the user's mute. The "unknown" placeholder is never a creator.
                    // A heuristic score is no reason to track a channel.
                    let is_real_channel = ch_id.starts_with("UC")
                        && ch_id.len() == 24
                        && !ch_title.eq_ignore_ascii_case("Unknown Creator");
                    let already_tracked = !matches!(db.get_channel(&ch_id), Ok(None));
                    let llm_scored = insight.analyzer_type != HEURISTIC_ANALYZER;
                    if kept
                        && llm_scored
                        && insight.slop_score <= AUTO_TRACK_MAX_SLOP
                        && is_real_channel
                        && !already_tracked
                    {
                        // "active", not "high_signal": one video is not a track
                        // record. The rolling average promotes the channel once
                        // enough of its videos are scored.
                        let new_channel = Channel {
                            id: ch_id.clone(),
                            title: ch_title.clone(),
                            url: format!("https://www.youtube.com/channel/{}", ch_id),
                            status: "active".to_string(),
                            rolling_slop_score: Some(insight.slop_score as f64),
                            created_at: chrono::Utc::now().to_rfc3339(),
                            // Never crawled: only this one video was analysed.
                            last_crawled_at: None,
                        };
                        let _ = db.upsert_channel(&new_channel);
                        if let Err(e) = db.recalculate_channel_stats(&ch_id) {
                            eprintln!("  [Discover] Could not update the rolling slop score of {}: {}", ch_id, e);
                        }

                        println!(
                            "    {} Added creator to the watchlist (one video at {}/100): {} ({})",
                            "🌟".yellow().bold(),
                            insight.slop_score,
                            ch_title.green().bold(),
                            ch_id.dimmed()
                        );
                        auto_promoted = Some(ch_title);
                    }

                    items.push(discovered_item(real_title, insight, &meta, auto_promoted));
                }
                Err(e) => {
                    println!("    {} Failed to evaluate candidate: {:#}", "✗".red(), e);
                    // No LLM: stop instead of fetching more transcripts for nothing
                    // and reporting an empty "success".
                    if e.downcast_ref::<LlmUnavailable>().is_some() {
                        return Err(e);
                    }
                }
            }

            if delay_secs > 0 && idx + 1 < candidates.len() {
                sleep(Duration::from_secs(delay_secs)).await;
            }
        }

        Ok(DiscoverSummary {
            query: query.to_string(),
            candidates_found: candidates.len(),
            items,
        })
    }
}

/// The discover result row of a stored insight.
fn discovered_item(
    title: String,
    insight: InsightResult,
    meta: &InsightMeta,
    auto_promoted_creator: Option<String>,
) -> DiscoveredItem {
    DiscoveredItem {
        video_id: insight.video_id,
        title,
        slop_score: insight.slop_score,
        not_scored: meta.is_insufficient(),
        rejection: meta.rejection_label(),
        quality_rating: insight.quality_rating,
        is_worth_keeping: meta.kept,
        auto_promoted_creator,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{AnalysisCoverage, ContextBudget, HEURISTIC_QUALITY_RATING};
    use std::io::{Read, Write};

    const CHANNEL_ID: &str = "UC0000000000000000000own";

    fn channel(id: &str, title: &str, status: &str, last_crawled: Option<&str>, slop: Option<f64>) -> Channel {
        Channel {
            id: id.to_string(),
            title: title.to_string(),
            url: format!("https://www.youtube.com/channel/{}", id),
            status: status.to_string(),
            rolling_slop_score: slop,
            created_at: "2026-01-01T00:00:00+00:00".to_string(),
            last_crawled_at: last_crawled.map(str::to_string),
        }
    }

    fn video(id: &str, title: &str, status: &str) -> Video {
        Video {
            id: id.to_string(),
            channel_id: CHANNEL_ID.to_string(),
            title: title.to_string(),
            url: format!("https://www.youtube.com/watch?v={}", id),
            published_at: "2026-01-01T00:00:00+00:00".to_string(),
            status: status.to_string(),
            created_at: "2026-01-01T00:00:00+00:00".to_string(),
        }
    }

    /// Pipeline on a fresh database file inside `dir`.
    fn pipeline(dir: &tempfile::TempDir, llm_url: Option<String>) -> IngestionPipeline {
        IngestionPipeline::new(&dir.path().join("test.db"), llm_url, None, Some("stub-model".to_string()))
            .expect("pipeline")
    }

    fn seed(p: &IngestionPipeline, v: &Video, transcript: Option<&str>) {
        let db = p.db().expect("db");
        db.upsert_channel(&channel(CHANNEL_ID, "Own Channel", "active", None, None)).expect("channel");
        db.upsert_video(v).expect("video");
        if let Some(text) = transcript {
            db.save_transcript(&CleanedTranscript {
                video_id: v.id.clone(),
                raw_char_count: text.len(),
                cleaned_char_count: text.len(),
                stripped_ad_seconds: 0.0,
                stripped_segments_count: 0,
                text: text.to_string(),
            })
            .expect("transcript");
        }
    }

    /// Stores a heuristic insight the way a no-LLM crawl does.
    fn seed_heuristic(dir: &tempfile::TempDir, v: &Video, text: &str) -> InsightResult {
        let no_llm = pipeline(dir, None);
        seed(&no_llm, v, Some(text));
        let db = no_llm.db().expect("db");
        let transcript = db.get_transcript(&v.id).expect("read").expect("stored");
        let insight = no_llm.analyzer.analyze_heuristically(&v.title, &transcript);
        no_llm.store_analysis(&db, &v.id, analysis_of(insight, None, text.len(), 1)).expect("store")
    }

    /// An analysis as the analyzer returns it, read completely in `sections` sections.
    fn analysis_of(insight: InsightResult, topic_verdict: Option<&str>, chars: usize, sections: usize) -> Analysis {
        Analysis {
            insight,
            topic_verdict: topic_verdict.map(str::to_string),
            topic_reason: topic_verdict.map(|v| format!("stub reason for {}", v)),
            coverage: AnalysisCoverage { transcript_chars: chars, sections },
        }
    }

    /// A chat-completions endpoint on a loopback port that answers every
    /// request with `content` as the assistant message. Returns its base URL.
    fn stub_llm(content: &'static str) -> String {
        let body = serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string();
        stub_llm_response("200 OK", body)
    }

    /// Like `stub_llm`, with the finish_reason a server sends ("length" for
    /// an answer cut off at max_tokens).
    fn stub_llm_finish(content: &'static str, finish_reason: &'static str) -> String {
        let body = serde_json::json!({
            "choices": [{ "message": { "content": content }, "finish_reason": finish_reason }]
        })
        .to_string();
        stub_llm_response("200 OK", body)
    }

    /// Answers every request with the given status line and body.
    fn stub_llm_response(status_line: &'static str, body: String) -> String {
        stub_llm_with(status_line, body, Duration::ZERO, None)
    }

    /// The JSON bodies of the requests a stub received, in order.
    type RequestLog = Arc<std::sync::Mutex<Vec<serde_json::Value>>>;

    /// Like `stub_llm`, and records every request body.
    fn stub_llm_logged(content: &'static str) -> (String, RequestLog) {
        let log = RequestLog::default();
        let body = serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string();
        (stub_llm_with("200 OK", body, Duration::ZERO, Some(log.clone())), log)
    }

    /// Like `stub_llm`, but every answer takes `delay`.
    fn stub_llm_slow(content: &'static str, delay: Duration) -> String {
        let body = serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string();
        stub_llm_with("200 OK", body, delay, None)
    }

    /// Answers every request with the given status line and body, after
    /// `delay`, and records the request bodies in `log` when one is given.
    fn stub_llm_with(status_line: &'static str, body: String, delay: Duration, log: Option<RequestLog>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                // Read the headers and then as many body bytes as they announce.
                let mut request = Vec::new();
                let mut chunk = [0u8; 4096];
                let mut body_start = 0;
                loop {
                    let n = match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    request.extend_from_slice(&chunk[..n]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..end]).to_lowercase();
                        let body_len = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        body_start = end + 4;
                        if request.len() >= end + 4 + body_len {
                            break;
                        }
                    }
                }
                if let Some(log) = &log {
                    let sent = serde_json::from_slice(&request[body_start..]).unwrap_or_default();
                    log.lock().expect("log").push(sent);
                }
                std::thread::sleep(delay);
                let response = format!(
                    "HTTP/1.1 {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    status_line,
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}/v1", addr)
    }

    const GOOD_ANSWER: &str = r#"{"slop_score":20,"is_worth_keeping":true,
        "quality_rating":"High Signal","summary":"A real summary.","key_insights":["An insight"],
        "takeaways":["A takeaway"],"quotes":[],"topics":["Bread Baking"],"mentioned_creators":[]}"#;

    fn long_text() -> String {
        "The dough rested in the fridge overnight and the crust came out twice as crisp. ".repeat(60)
    }

    #[test]
    fn high_signal_channels_come_first_then_least_recently_crawled() {
        let channels = vec![
            channel("a", "active, crawled recently", "active", Some("2026-03-10T00:00:00+00:00"), Some(20.0)),
            channel("b", "muted", "muted", None, None),
            channel("c", "high, crawled recently", "high_signal", Some("2026-03-10T00:00:00+00:00"), Some(30.0)),
            channel("d", "active, never crawled", "active", None, None),
            channel("e", "high, crawled long ago", "high_signal", Some("2026-01-01T00:00:00+00:00"), Some(33.0)),
            channel("f", "active, crawled long ago, sloppy", "active", Some("2026-01-01T00:00:00+00:00"), Some(60.0)),
            channel("g", "active, crawled long ago, clean", "active", Some("2026-01-01T00:00:00+00:00"), Some(40.0)),
            channel("h", "active, same instant in another zone, unscored", "active", Some("2026-01-01T01:00:00+01:00"), None),
        ];
        let order: Vec<String> = crawl_order(channels).into_iter().map(|c| c.id).collect();
        assert_eq!(order, vec!["e", "c", "d", "g", "f", "h", "a"]);
    }

    #[test]
    fn crawl_order_keeps_input_order_for_equal_channels() {
        let channels = vec![
            channel("x", "X", "active", None, None),
            channel("y", "Y", "active", None, None),
            channel("z", "Z", "active", Some("not a date"), None),
        ];
        let order: Vec<String> = crawl_order(channels).into_iter().map(|c| c.id).collect();
        assert_eq!(order, vec!["x", "y", "z"]);
        assert!(crawl_order(vec![channel("m", "M", "muted", None, None)]).is_empty());
    }

    #[test]
    fn work_list_plan_per_stored_status() {
        // New video, or one that never got through.
        assert_eq!(plan_for_video(None, 0, true), VideoPlan::Process);
        assert_eq!(plan_for_video(Some("pending"), 0, false), VideoPlan::Process);
        assert_eq!(plan_for_video(Some("discovered"), 0, true), VideoPlan::Process);
        // Done.
        assert_eq!(plan_for_video(Some("processed"), 0, true), VideoPlan::Skip);
        // Failed: retried until the limit, then given up.
        assert_eq!(plan_for_video(Some("failed"), 0, true), VideoPlan::Process);
        assert_eq!(plan_for_video(Some("failed"), MAX_VIDEO_FAILURES - 1, true), VideoPlan::Process);
        assert_eq!(plan_for_video(Some("failed"), MAX_VIDEO_FAILURES, true), VideoPlan::Skip);
        assert_eq!(plan_for_video(Some("failed"), MAX_VIDEO_FAILURES + 4, false), VideoPlan::Skip);
        // Heuristic rows wait for an LLM.
        assert_eq!(plan_for_video(Some("heuristic"), 0, false), VideoPlan::Skip);
        assert_eq!(plan_for_video(Some("heuristic"), 0, true), VideoPlan::Reanalyze);
        assert_eq!(plan_for_video(Some("heuristic"), MAX_VIDEO_FAILURES, true), VideoPlan::Skip);
    }

    #[test]
    fn video_state_reads_status_and_failure_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, None);
        let v = video("vidSTATE001", "Some video", "pending");
        seed(&p, &v, None);
        let db = p.db().expect("db");

        assert_eq!(video_state(&db, "unknownVid1"), (None, 0));
        assert_eq!(video_state(&db, &v.id), (Some("pending".to_string()), 0));
        for expected in 1..=MAX_VIDEO_FAILURES {
            assert_eq!(db.mark_video_failed(&v.id).expect("mark"), expected);
        }
        let (status, fails) = video_state(&db, &v.id);
        assert_eq!((status.as_deref(), fails), (Some("failed"), MAX_VIDEO_FAILURES));
        // This is the state in which a crawl stops picking the video up.
        assert_eq!(plan_for_video(status.as_deref(), fails, true), VideoPlan::Skip);
    }

    #[tokio::test]
    async fn a_remembered_block_with_no_other_way_left_keeps_the_video_pending() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut p = pipeline(&dir, None);
        // No helper can run: no browser, no direct request, no Whisper.
        p.extractor = SubtitleExtractor::without_helpers();
        let v = video("blockedvid1", "Blocked", "pending");
        seed(&p, &v, None);
        let store = PacingStore::for_db(&p.db_path);
        let stored = store.note_block(now_unix(), 3);

        let err = p.process_video(&v).await.expect_err("no transcript");
        // Retryable, so a crawl neither counts it against the video nor gives it up.
        assert_eq!(failure_cause(&err), FailureCause::RateLimited);
        // The direct request was left out, so nothing new was learned: not a second block.
        assert_eq!(store.load(), stored);
        assert!(p.db().expect("db").get_transcript(&v.id).ok().flatten().is_none());
    }

    #[test]
    fn only_youtubes_own_answer_marks_a_channel_as_gone() {
        assert!(channel_is_gone(
            "yt-dlp could not list the channel's videos: ERROR: [youtube:tab] UCx: YouTube said: This channel does not exist."
        ));
        assert!(!channel_is_gone("the video list was empty"));
        assert!(!channel_is_gone("ERROR: Unable to download API page: HTTP Error 404: Not Found"));
    }

    #[test]
    fn a_video_from_the_channel_list_becomes_a_feed_entry() {
        let item = feed_item_from_list(crate::extractor::ChannelVideo {
            video_id: "listVideo01".to_string(),
            title: "A title".to_string(),
            uploaded_at: Some(1794182400),
        });
        assert_eq!(item.url, "https://www.youtube.com/watch?v=listVideo01");
        assert!(item.published_at.starts_with("2026-11-0"), "{}", item.published_at);
        // No day shown: the time of the crawl stands in.
        let undated = feed_item_from_list(crate::extractor::ChannelVideo {
            video_id: "listVideo02".to_string(),
            title: "t".to_string(),
            uploaded_at: None,
        });
        assert!(chrono::DateTime::parse_from_rfc3339(&undated.published_at).is_ok());
    }

    #[test]
    fn the_result_line_names_videos_that_wait_for_youtube() {
        let none = CrawlSummary { videos_kept: 2, ..Default::default() };
        assert!(!none.counts_line().contains("waiting"), "{}", none.counts_line());
        let some = CrawlSummary { videos_evaluated: 2, videos_kept: 2, videos_waiting: 3, ..Default::default() };
        assert!(
            some.counts_line().contains("0 failed, 3 waiting (YouTube refuses transcript requests for now)"),
            "{}",
            some.counts_line()
        );
        // Waiting videos say nothing about whether the setup works.
        assert_eq!(some.videos_judged(), 2);
    }

    #[test]
    fn progress_follows_feeds_and_the_work_list() {
        assert_eq!(feed_phase_percent(0, 31), 0);
        assert_eq!(feed_phase_percent(31, 31), FEED_PHASE_PERCENT);
        assert!(feed_phase_percent(10, 31) > 0 && feed_phase_percent(10, 31) < FEED_PHASE_PERCENT);
        assert_eq!(feed_phase_percent(0, 0), FEED_PHASE_PERCENT);

        // Two new videos out of 31 channels: half-way after the first, not 3%.
        assert_eq!(video_phase_percent(0, 2), FEED_PHASE_PERCENT);
        assert_eq!(video_phase_percent(1, 2), 57);
        assert_eq!(video_phase_percent(2, 2), VIDEO_PHASE_END);
        assert_eq!(video_phase_percent(0, 0), VIDEO_PHASE_END);
        let mut last = 0;
        for done in 0..=93 {
            let pct = video_phase_percent(done, 93);
            assert!(pct >= last && pct <= VIDEO_PHASE_END);
            last = pct;
        }
    }

    fn transcript_error(kind: TranscriptFailure, message: &str) -> anyhow::Error {
        anyhow::Error::new(TranscriptError { kind, message: message.to_string() })
    }

    #[test]
    fn failure_cause_comes_from_the_error_type() {
        let blocked = transcript_error(
            TranscriptFailure::RateLimited,
            "YouTube rate limit / IP block (429): IpBlocked; audio fallback also failed",
        );
        assert_eq!(failure_cause(&blocked), FailureCause::RateLimited);
        // Still readable through added context.
        assert_eq!(failure_cause(&blocked.context("while crawling")), FailureCause::RateLimited);

        let premiere = transcript_error(TranscriptFailure::NotAvailableYet, "Premieres in 3 hours");
        assert_eq!(failure_cause(&premiere), FailureCause::NotAvailableYet);
        let no_whisper = transcript_error(TranscriptFailure::HelperMissing, "No module named 'faster_whisper'");
        assert_eq!(failure_cause(&no_whisper), FailureCause::HelperMissing);
        let empty = transcript_error(TranscriptFailure::Other, "Whisper produced an empty transcript for video abc");
        assert_eq!(failure_cause(&empty), FailureCause::Video);

        let wrong_key = anyhow::Error::new(LlmUnavailable {
            reason: "http://x/v1/chat/completions answered HTTP 401".to_string(),
            engine_down: false,
        });
        assert_eq!(failure_cause(&wrong_key), FailureCause::Llm);
        assert_eq!(failure_cause(&wrong_key.context("Stored insight of v left unchanged")), FailureCause::Llm);
    }

    #[test]
    fn an_llm_answer_that_mentions_429_is_not_a_rate_limit() {
        // The body of an LLM answer is part of these error texts.
        let body = anyhow::anyhow!(
            "Failed to parse ChatCompletionResponse from body: {{\"created\":1742900429,\"error\":\"HTTP Error 429: Too Many Requests\"}}"
        );
        assert_eq!(failure_cause(&body), FailureCause::Video);
        let unusable = anyhow::anyhow!("LLM answer has no summary")
            .context("LLM answer unusable after 2 attempts. Video left unprocessed for a later retry");
        assert_eq!(failure_cause(&unusable), FailureCause::Video);
    }

    #[test]
    fn a_run_without_any_success_does_not_count_its_failures() {
        // One or two failures can be the videos' own fault.
        assert!(failures_count_against_videos(0, 0));
        assert!(failures_count_against_videos(0, 1));
        assert!(failures_count_against_videos(0, 2));
        // Nothing worked and three or more failed: the setup is to blame.
        assert!(!failures_count_against_videos(0, 3));
        assert!(!failures_count_against_videos(0, 40));
        // One scored video shows that the setup works.
        assert!(failures_count_against_videos(1, 3));
        assert!(failures_count_against_videos(5, 40));
    }

    #[test]
    fn failures_are_recorded_at_the_end_of_the_run_only_when_something_worked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, None);
        let plain = video("vidFAIL0001", "Fails on its own", "pending");
        seed(&p, &plain, None);
        let heuristic = video("vidFAIL0002", "Heuristic row", "pending");
        seed_heuristic(&dir, &heuristic, &long_text());
        let db = p.db().expect("db");
        let failures = vec![
            PendingFailure { video_id: plain.id.clone(), reanalysis: false },
            PendingFailure { video_id: heuristic.id.clone(), reanalysis: true },
        ];

        // Nothing scored, three failed: nothing is counted.
        assert_eq!(record_run_failures(&db, &failures, 0, 3), 0);
        assert_eq!(video_state(&db, &plain.id), (Some("failed".to_string()), 0));
        assert_eq!(video_state(&db, &heuristic.id), (Some(STATUS_HEURISTIC.to_string()), 0));
        // Both are picked up again by the next crawl.
        assert_eq!(plan_for_video(Some("failed"), 0, true), VideoPlan::Process);

        // Something was scored: the failures are the videos' own.
        assert_eq!(record_run_failures(&db, &failures, 2, 3), 2);
        assert_eq!(video_state(&db, &plain.id), (Some("failed".to_string()), 1));
        // The heuristic row keeps the status that allows a retry without YouTube.
        assert_eq!(video_state(&db, &heuristic.id), (Some(STATUS_HEURISTIC.to_string()), 1));

        // Two failures without a success are counted as well.
        assert_eq!(record_run_failures(&db, &failures[..1], 0, 2), 1);
        assert_eq!(video_state(&db, &plain.id).1, 2);
        assert_eq!(record_run_failures(&db, &[], 0, 5), 0);
    }

    #[test]
    fn feed_result_line_names_feeds_that_could_not_be_fetched() {
        assert_eq!(feeds_checked_line(31, 0, 2), "31 feed(s) checked, 2 new video(s) to process.");
        assert_eq!(
            feeds_checked_line(28, 3, 2),
            "28 feed(s) checked, 3 could not be fetched, 2 new video(s) to process."
        );
        assert_eq!(
            feeds_checked_line(0, 31, 0),
            "0 feed(s) checked, 31 could not be fetched, 0 new video(s) to process."
        );
    }

    #[test]
    fn threshold_and_wiki_dir_are_configurable_and_have_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, None);
        assert_eq!(p.max_slop, KEEP_THRESHOLD);
        assert_eq!(p.analyzer.max_slop(), KEEP_THRESHOLD);
        assert_eq!(p.wiki_dir, PathBuf::from("./wiki"));

        let p = p.with_max_slop(60).with_wiki_dir(dir.path().join("my-wiki"));
        assert_eq!(p.max_slop, 60);
        assert_eq!(p.analyzer.max_slop(), 60);
        assert_eq!(p.wiki_dir, dir.path().join("my-wiki"));
        assert_eq!(pipeline(&dir, None).with_max_slop(1000).max_slop, 100);
    }

    #[test]
    fn heuristic_result_is_stored_unreviewed_and_left_for_an_llm() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0001", "Sourdough at home", "pending");
        let text = long_text();
        let insight = seed_heuristic(&dir, &v, &text);
        assert_eq!(insight.quality_rating, HEURISTIC_QUALITY_RATING);
        assert_eq!(insight.category, None);

        let db = pipeline(&dir, None).db().expect("db");
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_HEURISTIC);
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert_eq!(meta.transcript_chars, Some(text.len() as u64));
        assert_eq!(meta.sections_total, Some(1));
        // The heuristic judged no topic; enough text was there.
        assert_eq!(meta.topic_verdict, None);
        assert_eq!(meta.info_status.as_deref(), Some(INFO_SUFFICIENT));
    }

    #[tokio::test]
    async fn reanalysis_without_an_llm_changes_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0002", "Sourdough at home", "pending");
        let before = seed_heuristic(&dir, &v, &long_text());

        let p = pipeline(&dir, None);
        let err = p.reanalyze_video(&v.id).await.expect_err("no LLM configured");
        assert!(err.to_string().contains("needs an LLM"), "{}", err);

        let db = p.db().expect("db");
        let after = db.get_insight(&v.id).expect("insight").expect("row");
        assert_eq!(after.analyzer_type, HEURISTIC_ANALYZER);
        assert_eq!(after.summary, before.summary);
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_HEURISTIC);
    }

    #[tokio::test]
    async fn reanalysis_replaces_the_insight_when_the_llm_answers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0003", "Sourdough at home", "pending");
        let text = long_text();
        seed_heuristic(&dir, &v, &text);

        let p = pipeline(&dir, Some(stub_llm(GOOD_ANSWER)));
        let insight = p.reanalyze_video(&v.id).await.expect("stub answers");
        assert_eq!(insight.analyzer_type, "llm:stub-model");
        assert_eq!(insight.slop_score, 20);
        assert!(insight.is_worth_keeping);

        let db = p.db().expect("db");
        let stored = db.get_insight(&v.id).expect("insight").expect("row");
        assert_eq!(stored.summary, "A real summary.");
        assert_eq!(stored.topics, vec!["Bread Baking".to_string()]);
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_PROCESSED);

        // The new analysis read the whole transcript, in one request.
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert_eq!(meta.transcript_chars, Some(text.len() as u64));
        assert_eq!(meta.sections_total, Some(1));
        assert_eq!(meta.whole_transcript_note().as_deref(), Some("whole transcript (1 section)"));
        // An answer without a topic verdict is not rejected for its topic.
        assert_eq!(meta.topic_verdict.as_deref(), Some("ok"));
        assert!(meta.kept);
    }

    #[tokio::test]
    async fn reanalysis_applies_the_configured_threshold() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0004", "Sourdough at home", "pending");
        seed_heuristic(&dir, &v, &long_text());

        // The model says "keep" at 20; a threshold of 10 overrules it.
        let p = pipeline(&dir, Some(stub_llm(GOOD_ANSWER))).with_max_slop(10);
        let insight = p.reanalyze_video(&v.id).await.expect("stub answers");
        assert_eq!(insight.slop_score, 20);
        assert!(!insight.is_worth_keeping);
        let stored = p.db().expect("db").get_insight(&v.id).expect("insight").expect("row");
        assert!(!stored.is_worth_keeping);
    }

    #[tokio::test]
    async fn reanalysis_keeps_the_stored_insight_when_the_llm_answer_is_unusable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0005", "Sourdough at home", "pending");
        let before = seed_heuristic(&dir, &v, &long_text());

        // An object without a score is not an analysis.
        let p = pipeline(&dir, Some(stub_llm("{}")));
        let err = p.reanalyze_video(&v.id).await.expect_err("unusable answer");
        assert!(format!("{:#}", err).contains("left unchanged"), "{:#}", err);

        let db = p.db().expect("db");
        let after = db.get_insight(&v.id).expect("insight").expect("row");
        assert_eq!(after.analyzer_type, HEURISTIC_ANALYZER);
        assert_eq!(after.summary, before.summary);
        assert_eq!(after.key_insights, before.key_insights);
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_HEURISTIC);
    }

    /// What must survive a refused re-analysis.
    fn assert_heuristic_insight_unchanged(p: &IngestionPipeline, v: &Video, before: &InsightResult) {
        let db = p.db().expect("db");
        let after = db.get_insight(&v.id).expect("insight").expect("row");
        assert_eq!(after.analyzer_type, HEURISTIC_ANALYZER);
        assert_eq!(after.slop_score, before.slop_score);
        assert_eq!(after.summary, before.summary);
        assert_eq!(after.key_insights, before.key_insights);
        assert_eq!(after.topics, before.topics);
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_HEURISTIC);
    }

    #[tokio::test]
    async fn reanalysis_keeps_the_stored_insight_when_the_llm_answer_was_cut_off() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0006", "Sourdough at home", "pending");
        let before = seed_heuristic(&dir, &v, &long_text());

        // Cut off inside the summary: the JSON repair makes it parse, with
        // every later field empty.
        let cut = r#"{"slop_score":30,"is_worth_keeping":true,"summary":"The video expl"#;
        let p = pipeline(&dir, Some(stub_llm_finish(cut, "length")));
        let err = p.reanalyze_video(&v.id).await.expect_err("cut-off answer");
        let text = format!("{:#}", err);
        assert!(text.contains("left unchanged"), "{}", text);
        assert!(text.contains("cut off at max_tokens"), "{}", text);
        // The video's own failure, not an LLM outage.
        assert_eq!(failure_cause(&err), FailureCause::Video);
        assert_heuristic_insight_unchanged(&p, &v, &before);

        // A complete answer that the server still marks as cut off is refused too.
        let p = pipeline(&dir, Some(stub_llm_finish(GOOD_ANSWER, "length")));
        p.reanalyze_video(&v.id).await.expect_err("marked as cut off");
        assert_heuristic_insight_unchanged(&p, &v, &before);
    }

    #[tokio::test]
    async fn reanalysis_keeps_the_stored_insight_when_the_llm_answer_has_only_a_score() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0007", "Sourdough at home", "pending");
        let before = seed_heuristic(&dir, &v, &long_text());

        let p = pipeline(&dir, Some(stub_llm(r#"{"slop_score": 20}"#)));
        let err = p.reanalyze_video(&v.id).await.expect_err("score-only answer");
        let text = format!("{:#}", err);
        assert!(text.contains("left unchanged"), "{}", text);
        assert!(text.contains("no summary"), "{}", text);
        assert_heuristic_insight_unchanged(&p, &v, &before);

        // finish_reason "stop" with a full answer is the normal case.
        let p = pipeline(&dir, Some(stub_llm_finish(GOOD_ANSWER, "stop")));
        let insight = p.reanalyze_video(&v.id).await.expect("valid answer");
        assert_eq!(insight.summary, "A real summary.");
    }

    #[tokio::test]
    async fn a_wrong_model_name_or_key_is_not_the_videos_fault() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidHEUR0008", "Sourdough at home", "pending");
        let before = seed_heuristic(&dir, &v, &long_text());

        for status_line in ["404 Not Found", "401 Unauthorized", "403 Forbidden"] {
            let url = stub_llm_response(status_line, r#"{"error":"model 'stub-model' not found"}"#.to_string());
            let p = pipeline(&dir, Some(url));
            let err = p.reanalyze_video(&v.id).await.expect_err("configuration error");
            let unavailable = err.downcast_ref::<LlmUnavailable>().expect("typed as LLM problem");
            // The engine answers: one such error does not stop a crawl, two in a row do.
            assert!(!unavailable.engine_down, "{}", status_line);
            assert_eq!(failure_cause(&err), FailureCause::Llm);
            assert_heuristic_insight_unchanged(&p, &v, &before);
        }

        // 400 can be a context overflow for this one transcript: the video's own failure.
        let url = stub_llm_response("400 Bad Request", r#"{"error":"context size exceeded"}"#.to_string());
        let p = pipeline(&dir, Some(url));
        let err = p.reanalyze_video(&v.id).await.expect_err("bad request");
        assert_eq!(failure_cause(&err), FailureCause::Video);
        assert_heuristic_insight_unchanged(&p, &v, &before);
    }

    #[tokio::test]
    async fn a_stored_transcript_is_analysed_without_fetching_it_again() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidSTOR0001", "Sourdough at home", "pending");
        let text = long_text();
        // State after a crawl that saved the transcript and then found the LLM down.
        let p = pipeline(&dir, Some(stub_llm(GOOD_ANSWER)));
        seed(&p, &v, Some(&text));
        assert!(stored_transcript(&p.db().expect("db"), &v.id).is_some());

        // No helper is started: with `reuse_stored` the stored text is used.
        let insight = p.process_video_with(&v, true, None).await.expect("analysed from the stored transcript");
        assert_eq!(insight.analyzer_type, "llm:stub-model");
        assert_eq!(insight.summary, "A real summary.");

        let db = p.db().expect("db");
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_PROCESSED);
        assert_eq!(db.get_transcript(&v.id).expect("read").expect("row").text, text);

        // An empty stored transcript is not something to reuse.
        let empty = video("vidSTOR0002", "Nothing stored", "pending");
        seed(&p, &empty, Some("   "));
        assert!(stored_transcript(&db, &empty.id).is_none());
        assert!(stored_transcript(&db, "unknownVid1").is_none());
    }

    #[tokio::test]
    async fn reanalysis_needs_a_stored_transcript() {
        let dir = tempfile::tempdir().expect("tempdir");
        // The URL is never contacted: the missing transcript is found first.
        let p = pipeline(&dir, Some("http://127.0.0.1:9/v1".to_string()));
        let v = video("vidNOTR0001", "No transcript yet", "pending");
        seed(&p, &v, None);
        let err = p.reanalyze_video(&v.id).await.expect_err("no transcript");
        assert!(err.to_string().contains("No stored transcript"), "{}", err);
        let err = p.reanalyze_video("unknownVid1").await.expect_err("unknown video");
        assert!(err.to_string().contains("not in the database"), "{}", err);
    }

    #[tokio::test]
    async fn crawl_writes_the_wiki_itself_before_reporting_complete() {
        let dir = tempfile::tempdir().expect("tempdir");
        // No channels: no feed is requested, the wiki step still runs.
        let p = pipeline(&dir, None);
        let wiki = dir.path().join("wiki");
        let summary = p.run_crawl_with_wiki(None, 3, 0, Some(&wiki)).await.expect("crawl");
        assert!(summary.wiki_generated);
        assert_eq!(summary.wiki_error, None);
        assert_eq!(summary.channels_checked, 0);
        assert!(wiki.is_dir());

        let progress = p.progress.read().await.clone();
        assert!(!progress.is_running);
        assert_eq!(progress.stage, "complete");
        assert_eq!(progress.percent, 100);
        assert_eq!(progress.total_videos, 0);

        // Without a directory no wiki is written and nothing claims it was.
        let summary = p.run_crawl(None, 3, 0).await.expect("crawl");
        assert!(!summary.wiki_generated);
    }

    #[tokio::test]
    async fn a_wiki_failure_ends_the_crawl_in_stage_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, None);
        // A file where the wiki directory should be.
        let blocked = dir.path().join("not-a-directory");
        std::fs::write(&blocked, "x").expect("file");

        let summary = p.run_crawl_with_wiki(None, 3, 0, Some(&blocked)).await.expect("crawl result is kept");
        assert!(!summary.wiki_generated);
        assert!(summary.wiki_error.is_some());

        let progress = p.progress.read().await.clone();
        assert!(!progress.is_running);
        assert_eq!(progress.stage, "error");
        assert!(progress.message.contains("wiki could not be written"), "{}", progress.message);
        assert!(progress.percent < 100);
    }

    #[test]
    fn related_channels_come_from_this_channels_kept_insights_with_counts() {
        let own = channel(CHANNEL_ID, "Own Channel", "active", None, None);
        let tracked_other = channel("UC000000000000000000alic", "Alice Example", "active", None, None);
        let other_channel = "UC00000000000000000other";

        let insight = |video_id: &str, creators: &[&str]| InsightResult {
            video_id: video_id.to_string(),
            slop_score: 20,
            is_worth_keeping: true,
            quality_rating: "High Signal".to_string(),
            summary: String::new(),
            key_insights: vec![],
            takeaways: vec![],
            quotes: vec![],
            topics: vec![],
            mentioned_creators: creators.iter().map(|c| c.to_string()).collect(),
            analyzer_type: "llm:stub-model".to_string(),
            category: None,
        };
        let mut foreign = video("v3", "Other channel's video", "processed");
        foreign.channel_id = other_channel.to_string();
        let kept = vec![
            (video("v1", "One", "processed"), own.clone(), insight("v1", &["Alice Example", "Sam Sample", "alice example"])),
            (video("v2", "Two", "processed"), own.clone(), insight("v2", &["ALICE EXAMPLE", "Own Channel", "@thirdperson"])),
            (foreign, own.clone(), insight("v3", &["Somebody Else", "Sam Sample"])),
        ];
        let tracked = vec![own, tracked_other];

        let related = related_from_mentions(CHANNEL_ID, &kept, &tracked);
        let rows: Vec<(&str, Option<&str>, usize, bool)> = related
            .iter()
            .map(|r| (r.title.as_str(), r.channel_id.as_deref(), r.mentions, r.tracked))
            .collect();
        assert_eq!(
            rows,
            vec![
                // Twice in v1 counts once; v2 makes it two insights. Tracked, so the id is known.
                ("Alice Example", Some("UC000000000000000000alic"), 2, true),
                // Not tracked: no id is invented, the caller resolves it on demand.
                ("Sam Sample", None, 1, false),
                ("thirdperson", None, 1, false),
            ]
        );
        assert!(related.iter().all(|r| r.source == rss::RELATED_SOURCE_MENTIONS));
        assert!(related_from_mentions("UC00000000000000000nobody", &kept, &tracked).is_empty());
    }

    #[test]
    fn mentioned_creators_are_read_from_the_database() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, None);
        let v = video("vidMENT0001", "Interview", "pending");
        seed(&p, &v, Some("text"));
        let db = p.db().expect("db");
        let mut insight = p.analyzer.analyze_heuristically(&v.title, &db.get_transcript(&v.id).expect("read").expect("row"));
        insight.is_worth_keeping = true;
        insight.slop_score = 10;
        insight.mentioned_creators = vec!["Jane Doe".to_string()];
        db.save_insight(&insight).expect("save");

        let related = p.mentioned_creators_of_channel(CHANNEL_ID).expect("query");
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].title, "Jane Doe");
        assert_eq!(related[0].mentions, 1);
        assert_eq!(related[0].channel_id, None);
    }

    // ---- keep decision, information, outcomes --------------------------------

    #[test]
    fn keep_decision_table() {
        // (topic verdict, slop, max_slop, enough info, disqualified) -> kept
        let table: [(Option<&str>, u32, u32, bool, bool, bool); 14] = [
            (Some("ok"), 20, 45, true, false, true),
            (Some("ok"), 45, 45, true, false, true),
            (Some("ok"), 46, 45, true, false, false),
            (Some("ok"), 0, 0, true, false, true),
            (Some("ok"), 60, 60, true, false, true),
            // No verdict recorded (heuristic): the topic rejects nothing.
            (None, 20, 45, true, false, true),
            (None, 80, 45, true, false, false),
            // A rejected subject is never kept, however good the score.
            (Some("fringe"), 0, 45, true, false, false),
            (Some("nonsense"), 10, 45, true, false, false),
            (Some("politics"), 10, 100, true, false, false),
            // Too little information is never kept.
            (Some("ok"), 0, 45, false, false, false),
            // The owner's disqualification beats everything.
            (Some("ok"), 0, 45, true, true, false),
            (None, 0, 100, true, true, false),
            (Some("politics"), 99, 45, false, true, false),
        ];
        for (verdict, slop, max, info, excluded, kept) in table {
            assert_eq!(
                keep_decision(verdict, slop, max, info, excluded),
                kept,
                "{:?} slop {} max {} info {} excluded {}",
                verdict,
                slop,
                max,
                info,
                excluded
            );
        }
    }

    #[test]
    fn too_little_content_is_recognised_from_words_and_length() {
        // Fewer than MIN_TRANSCRIPT_WORDS words: never enough, whatever the length.
        assert_eq!(insufficient_reason(0, None).as_deref(), Some("only 0 words"));
        assert_eq!(insufficient_reason(42, Some(30.4)).as_deref(), Some("only 42 words in 30 seconds"));
        assert_eq!(insufficient_reason(119, Some(3600.0)).as_deref(), Some("only 119 words in 3600 seconds"));
        assert_eq!(insufficient_reason(120, None), None);
        // An unusable length is no length.
        assert_eq!(insufficient_reason(119, Some(f64::NAN)).as_deref(), Some("only 119 words"));
        assert_eq!(insufficient_reason(150, Some(0.0)), None);

        // A video under a minute needs to be dense to be judged.
        assert_eq!(insufficient_reason(150, Some(45.0)).as_deref(), Some("only 150 words in 45 seconds"));
        assert_eq!(insufficient_reason(200, Some(45.0)), None);
        assert_eq!(insufficient_reason(150, Some(60.0)), None);

        // Almost no speech in a long video.
        assert_eq!(
            insufficient_reason(150, Some(1800.0)).as_deref(),
            Some("almost no speech: only 150 words in 1800 seconds")
        );
        // The same words in ten minutes are slow speech, not silence.
        assert_eq!(insufficient_reason(150, Some(600.0)), None);
        // A long transcript is enough even when the video is mostly silent.
        assert_eq!(insufficient_reason(300, Some(7200.0)), None);
        // Ordinary videos.
        assert_eq!(insufficient_reason(1_500, Some(600.0)), None);
        assert_eq!(insufficient_reason(9_000, None), None);
    }

    #[test]
    fn every_stored_row_has_exactly_one_outcome() {
        let meta = InsightMeta::default;
        assert_eq!(video_outcome(&InsightMeta { kept: true, ..meta() }), VideoOutcome::Kept);
        assert_eq!(video_outcome(&meta()), VideoOutcome::BelowCutoff);
        let politics = InsightMeta { topic_verdict: Some("politics".to_string()), ..meta() };
        assert_eq!(video_outcome(&politics), VideoOutcome::OffTopic);
        let short = InsightMeta { info_status: Some(INFO_INSUFFICIENT.to_string()), ..meta() };
        assert_eq!(video_outcome(&short), VideoOutcome::Insufficient);
        // The owner's decision comes first.
        let excluded = InsightMeta { excluded_manual: true, ..politics.clone() };
        assert_eq!(video_outcome(&excluded), VideoOutcome::Disqualified);
        // Topic before information.
        let both = InsightMeta { info_status: Some(INFO_INSUFFICIENT.to_string()), ..politics };
        assert_eq!(video_outcome(&both), VideoOutcome::OffTopic);
    }

    #[test]
    fn summary_counts_each_outcome_once_and_names_them_in_the_result_line() {
        let mut summary = CrawlSummary::default();
        for _ in 0..3 {
            summary.count(VideoOutcome::Kept);
        }
        summary.count(VideoOutcome::BelowCutoff);
        summary.count(VideoOutcome::OffTopic);
        summary.count(VideoOutcome::OffTopic);
        summary.count(VideoOutcome::Insufficient);
        summary.count(VideoOutcome::Insufficient);
        summary.videos_failed = 1;

        assert_eq!(summary.videos_kept, 3);
        assert_eq!(summary.videos_filtered_slop, 1);
        assert_eq!(summary.videos_evaluated, 4);
        assert_eq!(summary.videos_off_topic, 2);
        assert_eq!(summary.videos_insufficient, 2);
        assert_eq!(summary.videos_disqualified, 0);
        assert_eq!(summary.videos_judged(), 8);
        assert_eq!(
            summary.counts_line(),
            "3 kept, 1 below the quality cutoff, 2 off-topic (fringe / nonsense / politics), 2 not enough information, 1 failed"
        );

        summary.count(VideoOutcome::Disqualified);
        summary.feeds_failed = 2;
        assert!(summary.counts_line().ends_with("1 failed, 1 disqualified by hand; 2 feed(s) could not be fetched"));

        // The counters are part of the JSON of summary and progress.
        let json = serde_json::to_value(&summary).expect("serialize");
        assert_eq!(json["videos_off_topic"], 2);
        assert_eq!(json["videos_insufficient"], 2);
        assert_eq!(json["videos_disqualified"], 1);
        let progress = serde_json::to_value(CrawlProgress::default()).expect("serialize");
        for field in ["videos_off_topic", "videos_insufficient", "videos_disqualified", "videos_kept", "videos_filtered"] {
            assert_eq!(progress[field], 0, "{}", field);
        }

        // A short video got as far as its transcript; a run without any
        // result did not.
        assert_eq!(CrawlSummary { videos_failed: 4, ..Default::default() }.videos_judged(), 0);
        let only_short = CrawlSummary { videos_insufficient: 1, ..Default::default() };
        assert_eq!(only_short.videos_judged(), 1);
    }

    // ---- topic verdict, information and disqualification in the pipeline -------

    const POLITICS_ANSWER: &str = r#"{"topic_verdict":"politics","topic_reason":"Campaign commentary.",
        "slop_score":15,"is_worth_keeping":true,"quality_rating":"High Signal",
        "summary":"A dense but partisan summary.","key_insights":["An insight"],"takeaways":[],"quotes":[],
        "topics":["Campaigns"],"mentioned_creators":[]}"#;

    const HISTORY_ANSWER: &str = r#"{"topic_verdict":"ok","topic_reason":"A history lecture.",
        "category":"ancient history","slop_score":12,"is_worth_keeping":false,"quality_rating":"High Signal",
        "summary":"How the grain supply shaped the late Republic.","key_insights":["An insight"],"takeaways":[],
        "quotes":[],"topics":["Roman Republic"],"mentioned_creators":[]}"#;

    #[tokio::test]
    async fn a_dense_history_lecture_is_kept_without_a_category_when_none_is_defined() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, Some(stub_llm(HISTORY_ANSWER)));
        let v = video("vidHIST0001", "What historians get wrong about the Roman Empire", "pending");
        let text = "In the ancient Roman Empire the grain fleet from Egypt decided who held the city. ".repeat(60);
        seed(&p, &v, Some(&text));

        let insight = p.process_video_with(&v, true, None).await.expect("analysed");
        assert!(insight.is_worth_keeping, "kept although the model's own flag said false");
        // The model named a category, but the user has defined none.
        assert_eq!(insight.category, None);
        assert_eq!(insight.slop_score, 12);

        let db = p.db().expect("db");
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert!(meta.kept);
        assert_eq!(meta.topic_verdict.as_deref(), Some("ok"));
        assert_eq!(meta.topic_reason.as_deref(), Some("A history lecture."));
        assert_eq!(meta.info_status.as_deref(), Some(INFO_SUFFICIENT));
        assert_eq!(video_outcome(&meta), VideoOutcome::Kept);
        let (_, line) = outcome_report(&insight, &meta, p.max_slop);
        assert_eq!(line, "Slop Score: 12/100 - KEPT (High Signal)");
    }

    #[tokio::test]
    async fn the_category_of_a_video_is_one_the_user_defined_at_the_time_of_the_analysis() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, Some(stub_llm(HISTORY_ANSWER)));
        let v = video("vidHIST0002", "The grain fleet of the late Republic", "pending");
        let text = "In the ancient Roman Empire the grain fleet from Egypt decided who held the city. ".repeat(60);
        seed(&p, &v, Some(&text));
        let db = p.db().expect("db");
        db.add_category("Baking", None).expect("category");

        // The model's "ancient history" is not among the user's categories.
        let insight = p.process_video_with(&v, true, None).await.expect("analysed");
        assert_eq!(insight.category, None);

        // Defined afterwards, it applies to the next analysis without a
        // restart, in the user's spelling, and changes nothing else.
        db.add_category("Ancient History", Some("antiquity")).expect("category");
        let again = p.reanalyze_video(&v.id).await.expect("re-analysed");
        assert_eq!(again.category.as_deref(), Some("Ancient History"));
        assert_eq!((again.slop_score, again.is_worth_keeping), (insight.slop_score, insight.is_worth_keeping));
        assert_eq!(db.get_insight(&v.id).expect("get").expect("row").category.as_deref(), Some("Ancient History"));

        // Removing the category keeps the insight.
        assert_eq!(db.remove_category("Ancient History").expect("remove"), ("Ancient History".to_string(), 1));
        let kept = db.get_insight(&v.id).expect("get").expect("the insight is still there");
        assert_eq!(kept.category, None);
        assert!(db.get_insight_meta(&v.id).expect("meta").expect("row").kept);
    }

    #[tokio::test]
    async fn a_category_set_by_hand_survives_every_analysis_until_it_is_handed_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, Some(stub_llm(HISTORY_ANSWER)));
        let v = video("vidHIST0003", "Roads and milestones of the Republic", "pending");
        let text = "The road from the capital to the coast was paved in stages over forty years. ".repeat(60);
        seed(&p, &v, Some(&text));
        let db = p.db().expect("db");
        db.add_category("Ancient History", None).expect("category");
        db.add_category("Baking", None).expect("category");
        let stored = |db: &Database| {
            let category = db.get_insight(&v.id).expect("get").expect("row").category;
            (category, db.get_insight_meta(&v.id).expect("meta").expect("row").category_manual)
        };

        let first = p.process_video_with(&v, true, None).await.expect("analysed");
        assert_eq!(first.category.as_deref(), Some("Ancient History"));
        assert_eq!(stored(&db), (Some("Ancient History".to_string()), false));

        // The model keeps answering "ancient history"; the owner said Baking.
        db.set_video_category(&v.id, Some("baking")).expect("by hand");
        let again = p.reanalyze_video(&v.id).await.expect("re-analysed");
        assert_eq!(again.category.as_deref(), Some("Baking"));
        assert_eq!((again.slop_score, again.is_worth_keeping), (first.slop_score, first.is_worth_keeping));
        assert_eq!(stored(&db), (Some("Baking".to_string()), true));
        // The same through the full path (stored transcript).
        let full = p.process_video_with(&v, true, None).await.expect("analysed");
        assert_eq!(full.category.as_deref(), Some("Baking"));
        assert_eq!(stored(&db), (Some("Baking".to_string()), true));

        // "No category" by hand sticks in the same way.
        db.set_video_category(&v.id, None).expect("none by hand");
        let none = p.reanalyze_video(&v.id).await.expect("re-analysed");
        assert_eq!(none.category, None);
        assert_eq!(stored(&db), (None, true));

        // Handed back: nothing changes at once, the next analysis decides.
        db.set_video_category_auto(&v.id).expect("auto");
        assert_eq!(stored(&db), (None, false));
        let auto = p.reanalyze_video(&v.id).await.expect("re-analysed");
        assert_eq!(auto.category.as_deref(), Some("Ancient History"));
        assert_eq!(stored(&db), (Some("Ancient History".to_string()), false));
    }

    /// An insight an LLM stored earlier, with its video.
    fn seed_judged(p: &IngestionPipeline, id: &str, analyzer_type: &str) {
        let v = video(id, &format!("Title of {}", id), "processed");
        seed(p, &v, Some("TRANSCRIPT-TEXT that a category request must never carry."));
        let insight = InsightResult {
            video_id: id.to_string(),
            slop_score: 22,
            is_worth_keeping: true,
            quality_rating: "High Signal".to_string(),
            summary: format!("Stored summary of {}.", id),
            key_insights: vec!["A stored insight".to_string()],
            takeaways: vec!["A stored takeaway".to_string()],
            quotes: vec![],
            topics: vec!["Stored Topic".to_string()],
            mentioned_creators: vec![],
            analyzer_type: analyzer_type.to_string(),
            category: None,
        };
        let db = p.db().expect("db");
        p.store_analysis(&db, id, analysis_of(insight, Some("ok"), 500, 1)).expect("store");
    }

    /// Everything of a stored row that the tests can read, the category left out.
    fn row_without_category(db: &Database, id: &str) -> (serde_json::Value, InsightMeta) {
        let mut insight = db.get_insight(id).expect("get").expect("row");
        insight.category = None;
        let meta = db.get_insight_meta(id).expect("meta").expect("row");
        (serde_json::to_value(&insight).expect("json"), meta)
    }

    fn category_and_mark(db: &Database, id: &str) -> (Option<String>, bool) {
        let category = db.get_insight(id).expect("get").expect("row").category;
        (category, db.get_insight_meta(id).expect("meta").expect("row").category_manual)
    }

    const IDS: [&str; 6] = ["catOPEN0001", "catOPEN0002", "catSORT0001", "catHAND0001", "catNONE0001", "catHEUR0001"];

    /// Two videos without a category, one the model sorted, one set by hand,
    /// one put into no category by hand, one with a heuristic estimate.
    fn seed_category_library(p: &IngestionPipeline) -> Database {
        let db = p.db().expect("db");
        db.add_category("Baking", Some("bread and pastry")).expect("category");
        db.add_category("Astronomy", None).expect("category");
        for id in &IDS[..5] {
            seed_judged(p, id, "llm:earlier-model");
        }
        seed_judged(p, IDS[5], HEURISTIC_ANALYZER);
        assert!(db.set_model_category("catSORT0001", Some("Astronomy")).expect("model"));
        db.set_video_category("catHAND0001", Some("Astronomy")).expect("by hand");
        db.set_video_category("catNONE0001", None).expect("none by hand");
        db
    }

    #[tokio::test]
    async fn categorize_changes_only_the_category_and_leaves_choices_made_by_hand() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (url, log) = stub_llm_logged(r#"{"category": "baking"}"#);
        let p = pipeline(&dir, Some(url));
        let db = seed_category_library(&p);
        let before: Vec<_> = IDS.iter().map(|id| row_without_category(&db, id)).collect();

        let summary = p.run_categorize(false, None).await.expect("run");
        assert_eq!(
            summary,
            CategorizeSummary {
                total: 2,
                assigned: vec![("Astronomy".to_string(), 0), ("Baking".to_string(), 2)],
                ..Default::default()
            }
        );
        assert_eq!(summary.counts_line(), "2 assigned (Baking: 2), 0 in no category, 0 failed");
        assert_eq!(category_and_mark(&db, "catOPEN0001"), (Some("Baking".to_string()), false));
        assert_eq!(category_and_mark(&db, "catOPEN0002"), (Some("Baking".to_string()), false));
        // Not asked for: sorted by the model already, set by hand, heuristic.
        assert_eq!(category_and_mark(&db, "catSORT0001"), (Some("Astronomy".to_string()), false));
        assert_eq!(category_and_mark(&db, "catHAND0001"), (Some("Astronomy".to_string()), true));
        assert_eq!(category_and_mark(&db, "catNONE0001"), (None, true));
        assert_eq!(category_and_mark(&db, "catHEUR0001"), (None, false));
        // Score, verdict, keep flag, analyzer type, summary: all as before.
        let after: Vec<_> = IDS.iter().map(|id| row_without_category(&db, id)).collect();
        assert_eq!(after, before);
        assert_eq!(db.get_insight("catOPEN0001").expect("get").expect("row").analyzer_type, "llm:earlier-model");

        // One short request per video, made from the stored insight.
        let chats: Vec<serde_json::Value> =
            log.lock().expect("log").iter().filter(|body| body["messages"].is_array()).cloned().collect();
        assert_eq!(chats.len(), 2);
        for chat in &chats {
            let system = chat["messages"][0]["content"].as_str().expect("system");
            let user = chat["messages"][1]["content"].as_str().expect("user");
            assert!(system.contains("- \"Astronomy\"\n- \"Baking\": bread and pastry\n"), "{}", system);
            assert!(user.contains("Summary: Stored summary of catOPEN000"), "{}", user);
            assert!(user.contains("- A stored insight") && user.contains("Topics: Stored Topic"), "{}", user);
            assert!(!user.contains("TRANSCRIPT-TEXT") && !system.contains("TRANSCRIPT-TEXT"));
        }
        let progress = p.categorize_progress.read().await.clone();
        assert_eq!((progress.is_running, progress.stage.as_str(), progress.percent), (false, "complete", 100));
        assert_eq!((progress.total_videos, progress.assigned, progress.none, progress.failed), (2, 2, 0, 0));

        // Nothing is left without --all; with it the model's own choices are
        // asked again, the owner's never.
        assert_eq!(p.run_categorize(false, None).await.expect("run").total, 0);
        let limited = p.run_categorize(true, Some(1)).await.expect("run");
        assert_eq!((limited.total, limited.beyond_limit), (1, 2));
        let all = p.run_categorize(true, None).await.expect("run");
        assert_eq!((all.total, all.assigned_total(), all.skipped), (3, 3, 0));
        assert_eq!(category_and_mark(&db, "catSORT0001"), (Some("Baking".to_string()), false));
        assert_eq!(category_and_mark(&db, "catHAND0001"), (Some("Astronomy".to_string()), true));
        assert_eq!(category_and_mark(&db, "catNONE0001"), (None, true));
        let after: Vec<_> = IDS.iter().map(|id| row_without_category(&db, id)).collect();
        assert_eq!(after, before);
    }

    #[tokio::test]
    async fn a_category_answer_that_is_none_of_the_users_names_is_stored_as_no_category() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, Some(stub_llm(r#"{"category": "Gardening"}"#)));
        let db = seed_category_library(&p);

        let summary = p.run_categorize(true, None).await.expect("run");
        assert_eq!((summary.total, summary.assigned_total(), summary.none, summary.failed), (3, 0, 3, 0));
        assert_eq!(summary.counts_line(), "0 assigned, 3 in no category, 0 failed");
        // Also the one the model had sorted before.
        for id in ["catOPEN0001", "catOPEN0002", "catSORT0001"] {
            assert_eq!(category_and_mark(&db, id), (None, false), "{}", id);
        }
        assert_eq!(category_and_mark(&db, "catHAND0001"), (Some("Astronomy".to_string()), true));

        // An answer that cannot be read at all fails the video and changes
        // nothing; the second one in a row ends the run.
        let unreadable = pipeline(&dir, Some(stub_llm("I would say it is about bread.")));
        assert!(db.set_model_category("catSORT0001", Some("Astronomy")).expect("model"));
        let summary = unreadable.run_categorize(true, None).await.expect("run");
        assert_eq!((summary.total, summary.failed, summary.none, summary.not_attempted), (3, 2, 0, 1));
        assert_eq!(summary.stopped.as_deref(), Some("The LLM's answer could not be read for 2 videos in a row"));
        assert_eq!(category_and_mark(&db, "catSORT0001"), (Some("Astronomy".to_string()), false));
    }

    #[tokio::test]
    async fn categorize_ends_when_the_categories_change_during_the_run_and_batches_of_all_move_on() {
        let dir = tempfile::tempdir().expect("tempdir");
        // The first request is answered late enough for the owner to rename a
        // category meanwhile.
        let mut p = pipeline(&dir, Some(stub_llm_slow(r#"{"category": "Baking"}"#, Duration::from_millis(400))));
        p.analyzer = p.analyzer.clone().with_request_timeout(Duration::from_secs(5));
        let db = seed_category_library(&p);
        let renamer = {
            let db = p.db().expect("db");
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(150));
                db.rename_category("Baking", "Bread").expect("rename");
            })
        };
        let summary = p.run_categorize(false, None).await.expect("run");
        renamer.await.expect("renamer");
        assert_eq!(summary.stopped.as_deref(), Some("Your categories were changed during the run"));
        assert_eq!((summary.assigned_total(), summary.none, summary.failed, summary.not_attempted), (0, 0, 0, 2));
        // Nothing was stored for the answer given from the list of before,
        // and the video is still to be asked about.
        assert_eq!(category_and_mark(&db, "catOPEN0001"), (None, false));
        assert_eq!(db.list_categorize_targets(false).expect("targets").len(), 2);

        // --all in batches: a video asked about goes to the end of the line,
        // so the next batch takes the next video.
        let fast = pipeline(&dir, Some(stub_llm(r#"{"category": "Bread"}"#)));
        let first = db.list_categorize_targets(true).expect("targets");
        assert_eq!(first.len(), 3);
        assert_eq!(fast.run_categorize(true, Some(1)).await.expect("run").assigned_total(), 1);
        let second = db.list_categorize_targets(true).expect("targets");
        assert_eq!(second.last(), first.first());
        assert_eq!(second[0], first[1]);
    }

    #[tokio::test]
    async fn categorize_needs_categories_and_an_llm_and_stops_when_the_llm_keeps_failing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, Some(stub_llm(r#"{"category": "Baking"}"#)));
        seed_judged(&p, "catOPEN0001", "llm:earlier-model");
        // No categories: nothing to choose from.
        let err = p.run_categorize(false, None).await.expect_err("no categories");
        assert!(err.to_string().contains("No categories are defined"), "{}", err);

        let db = p.db().expect("db");
        db.add_category("Baking", None).expect("category");
        seed_judged(&p, "catOPEN0002", "llm:earlier-model");
        seed_judged(&p, "catOPEN0003", "llm:earlier-model");
        let err = pipeline(&dir, None).run_categorize(false, None).await.expect_err("no LLM");
        assert!(err.to_string().contains("needs an LLM"), "{}", err);
        // Nothing listens there (the port was free a moment ago): refused
        // before the first video.
        let free_port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
            listener.local_addr().expect("local addr").port()
        };
        let err = pipeline(&dir, Some(format!("http://127.0.0.1:{}/v1", free_port)))
            .run_categorize(false, None)
            .await
            .expect_err("unreachable");
        assert!(err.to_string().contains("is not reachable"), "{}", err);

        // Reachable, but every request times out: the second failure in a row
        // ends the run and the third video is not attempted.
        let mut slow = pipeline(&dir, Some(stub_llm_slow(r#"{"category": "Baking"}"#, Duration::from_millis(700))));
        slow.analyzer = slow.analyzer.clone().with_request_timeout(Duration::from_millis(150));
        let summary = slow.run_categorize(false, None).await.expect("run");
        assert_eq!((summary.total, summary.failed, summary.not_attempted), (3, 2, 1));
        assert_eq!(summary.stopped.as_deref(), Some("The LLM failed for 2 videos in a row"));
        assert_eq!(summary.assigned_total() + summary.none, 0);
        for id in ["catOPEN0001", "catOPEN0002", "catOPEN0003"] {
            assert_eq!(category_and_mark(&db, id), (None, false), "{}", id);
        }
        let progress = slow.categorize_progress.read().await.clone();
        assert_eq!((progress.is_running, progress.stage.as_str()), (false, "error"));

        // With an engine that answers the same videos get their category.
        let summary = p.run_categorize(false, None).await.expect("run");
        assert_eq!((summary.total, summary.assigned_total()), (3, 3));
    }

    #[tokio::test]
    async fn a_politics_verdict_rejects_the_video_and_keeps_its_quality_score() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, Some(stub_llm(POLITICS_ANSWER)));
        let v = video("vidPOLI0001", "Who wins the midterms", "pending");
        seed(&p, &v, Some(&long_text()));

        let insight = p.process_video_with(&v, true, None).await.expect("analysed");
        // Dense (15/100), and still not kept: the two judgments are separate.
        assert_eq!(insight.slop_score, 15);
        assert!(!insight.is_worth_keeping);

        let db = p.db().expect("db");
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert!(!meta.kept);
        assert_eq!(meta.topic_verdict.as_deref(), Some("politics"));
        assert_eq!(video_outcome(&meta), VideoOutcome::OffTopic);
        let (_, line) = outcome_report(&insight, &meta, p.max_slop);
        assert_eq!(line, "Off-topic: politics (Campaign commentary.) - not kept (quality score 15/100)");
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_PROCESSED);
        assert!(!db.get_insight(&v.id).expect("insight").expect("row").is_worth_keeping);
    }

    #[tokio::test]
    async fn a_transcript_with_too_little_content_is_stored_as_insufficient_without_an_llm_call() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Nothing listens on this port: an LLM request would fail the test.
        let p = pipeline(&dir, Some("http://127.0.0.1:9/v1".to_string()));
        let v = video("vidSHORT001", "Quick teaser", "pending");
        let text = "Just a few words before the music starts. ".repeat(5); // 40 words
        seed(&p, &v, Some(&text));
        let db = p.db().expect("db");
        db.set_transcript_coverage(&v.id, Some(31.0), Some(29.0), Some("builtin")).expect("coverage");

        let insight = p.process_video_with(&v, true, None).await.expect("not an error, not an LLM call");
        assert_eq!(insight.analyzer_type, INSUFFICIENT_ANALYZER);
        assert!(!insight.is_worth_keeping);
        assert_eq!(insight.quality_rating, "Not enough information");

        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert_eq!(meta.info_status.as_deref(), Some(INFO_INSUFFICIENT));
        assert_eq!(meta.info_reason.as_deref(), Some("only 40 words in 31 seconds"));
        assert!(!meta.kept);
        assert_eq!(video_outcome(&meta), VideoOutcome::Insufficient);
        assert_eq!(
            meta.rejection_label().as_deref(),
            Some("Not enough information (only 40 words in 31 seconds)")
        );
        // Done: the next crawl does not fetch it again.
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_PROCESSED);
        // A short video does not give its channel a slop average.
        db.recalculate_channel_stats(CHANNEL_ID).expect("recalculate");
        let channel = db.get_channel(CHANNEL_ID).expect("channel").expect("row");
        assert_eq!(channel.rolling_slop_score, None);
        assert_eq!(channel.status, "active");

        let item = discovered_item(v.title.clone(), insight, &meta, None);
        assert!(item.not_scored);
        assert!(!item.is_worth_keeping);
        assert_eq!(item.rejection.as_deref(), Some("Not enough information (only 40 words in 31 seconds)"));

        // Re-analysis comes to the same result, again without the LLM.
        let again = p.reanalyze_video(&v.id).await.expect("stored as insufficient");
        assert_eq!(again.analyzer_type, INSUFFICIENT_ANALYZER);
    }

    #[tokio::test]
    async fn a_disqualified_video_stays_out_after_reanalysis_and_include_brings_it_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidEXCL0001", "Sourdough at home", "pending");
        seed_heuristic(&dir, &v, &long_text());
        let p = pipeline(&dir, Some(stub_llm(GOOD_ANSWER)));
        let db = p.db().expect("db");
        db.exclude_video(&v.id, Some("duplicate upload")).expect("exclude");

        // The model says 20/100, keep. The owner said no.
        let insight = p.reanalyze_video(&v.id).await.expect("stub answers");
        assert_eq!(insight.analyzer_type, "llm:stub-model");
        assert_eq!(insight.slop_score, 20);
        assert!(!insight.is_worth_keeping);

        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert!(meta.excluded_manual);
        assert!(!meta.kept);
        assert_eq!(video_outcome(&meta), VideoOutcome::Disqualified);
        let (_, line) = outcome_report(&insight, &meta, p.max_slop);
        assert!(line.starts_with("Disqualified (duplicate upload) - stays out of the library"), "{}", line);
        assert!(line.contains("ytc include vidEXCL0001"), "{}", line);

        // The same through the full path (stored transcript).
        let again = p.process_video_with(&v, true, None).await.expect("stub answers");
        assert!(!again.is_worth_keeping);
        assert!(!db.get_insight_meta(&v.id).expect("meta").expect("row").kept);

        // The model's verdict was stored underneath, so the undo needs no new analysis.
        db.include_video(&v.id).expect("include");
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert!(meta.kept);
        assert_eq!(video_outcome(&meta), VideoOutcome::Kept);
    }

    #[test]
    fn the_stored_keep_flag_follows_the_keep_decision() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, None);
        let db = p.db().expect("db");
        let text = long_text();
        let cases: [(&str, Option<&str>, u32); 6] = [
            ("vidKEEP0001", Some("ok"), 20),
            ("vidKEEP0002", Some("ok"), 46),
            ("vidKEEP0003", Some("fringe"), 5),
            ("vidKEEP0004", Some("nonsense"), 5),
            ("vidKEEP0005", Some("politics"), 45),
            ("vidKEEP0006", None, 45),
        ];
        for (id, verdict, slop) in cases {
            let v = video(id, "Some video", "pending");
            seed(&p, &v, Some(&text));
            let mut insight = p.analyzer.analyze_heuristically(&v.title, &db.get_transcript(id).expect("read").expect("row"));
            insight.analyzer_type = "llm:stub-model".to_string();
            insight.slop_score = slop;
            // Whatever flag comes in, the parts decide.
            insight.is_worth_keeping = !keep_decision(verdict, slop, p.max_slop, true, false);

            let stored = p.store_analysis(&db, id, analysis_of(insight, verdict, text.len(), 3)).expect("store");
            let expected = keep_decision(verdict, slop, p.max_slop, true, false);
            assert_eq!(stored.is_worth_keeping, expected, "{}", id);
            let meta = db.get_insight_meta(id).expect("meta").expect("row");
            // The database's kept rule and the pipeline's agree.
            assert_eq!(meta.kept, expected, "{}", id);
            assert_eq!(meta.topic_verdict.as_deref(), verdict);
            assert_eq!(meta.sections_total, Some(3));
            assert_eq!(meta.whole_transcript_note().as_deref(), Some("whole transcript (3 sections)"));
        }
    }

    #[tokio::test]
    async fn a_long_transcript_is_read_in_sections_and_their_number_is_recorded() {
        let dir = tempfile::tempdir().expect("tempdir");
        // The stub answers every request alike: its JSON serves as section
        // notes and as the judgment.
        let mut p = pipeline(&dir, Some(stub_llm(GOOD_ANSWER)));
        p.analyzer = p.analyzer.clone().with_context_budget(ContextBudget { single_chars: 3_000, section_chars: 1_500 });
        let v = video("vidLONG0001", "Sourdough at home, the long version", "pending");
        let text = long_text(); // about 4,900 characters
        assert!(text.len() > 3_000);
        seed(&p, &v, Some(&text));
        let sections = crate::analyzer::split_into_sections(&text, 1_500).len() as u32;
        assert!(sections >= 4);

        let insight = p.process_video_with(&v, true, None).await.expect("analysed");
        assert_eq!(insight.summary, "A real summary.");

        let meta = p.db().expect("db").get_insight_meta(&v.id).expect("meta").expect("row");
        assert_eq!(meta.sections_total, Some(sections));
        assert_eq!(meta.transcript_chars, Some(text.len() as u64));
        assert_eq!(
            meta.whole_transcript_note(),
            Some(format!("whole transcript ({} sections)", sections))
        );
    }

    #[test]
    fn an_extracted_transcript_is_stored_with_its_source_and_coverage() {
        use crate::models::TranscriptCue;
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, None);
        let v = video("vidCOVR0001", "Some video", "pending");
        seed(&p, &v, None);
        let db = p.db().expect("db");
        let cue = |start: f64, end: f64, text: &str| TranscriptCue { start_sec: start, end_sec: end, text: text.to_string() };

        let extracted = ExtractedTranscript {
            cues: vec![cue(0.0, 5.0, "first words"), cue(5.0, 580.0, "last words")],
            source: crate::extractor::SOURCE_WHISPER,
            video_duration_sec: Some(600.0),
            covered_until_sec: Some(580.0),
        };
        let cleaned = p.save_extracted(&db, &v.id, &extracted, &[]).expect("save");
        assert_eq!(cleaned.text, "first words last words");
        assert_eq!(db.get_transcript(&v.id).expect("read").expect("row").text, "first words last words");
        let coverage = db.get_transcript_coverage(&v.id).expect("read").expect("row");
        assert_eq!(coverage.video_duration_sec, Some(600.0));
        assert_eq!(coverage.covered_until_sec, Some(580.0));
        assert_eq!(coverage.source.as_deref(), Some("whisper"));
        assert_eq!(coverage.video_coverage_pct(), Some(97));

        // Unknown length: stored as not recorded, never guessed.
        let unchecked = ExtractedTranscript {
            source: crate::extractor::SOURCE_BUILTIN,
            video_duration_sec: None,
            ..extracted
        };
        p.save_extracted(&db, &v.id, &unchecked, &[]).expect("save");
        let coverage = db.get_transcript_coverage(&v.id).expect("read").expect("row");
        assert_eq!(coverage.video_duration_sec, None);
        assert_eq!(coverage.video_coverage_pct(), None);
        assert_eq!(coverage.source.as_deref(), Some("builtin"));
    }

    // ---- the judgment request ---------------------------------------------------

    #[tokio::test]
    async fn the_model_gets_the_title_and_the_whole_transcript_and_its_verdict_is_stored() {
        let dir = tempfile::tempdir().expect("tempdir");
        let text = format!("{} The grain fleet sailed every summer, and the count of its ships was kept by the harbour clerks.", long_text());
        let (url, log) = stub_llm_logged(HISTORY_ANSWER);
        let p = pipeline(&dir, Some(url));
        let v = video("vidHIST0002", "History lecture 12: the harbours of the late Republic", "pending");
        seed(&p, &v, Some(&text));

        let insight = p.process_video_with(&v, true, None).await.expect("judged by the model");
        assert_eq!(insight.analyzer_type, "llm:stub-model");
        assert!(insight.is_worth_keeping);
        assert_eq!(insight.slop_score, 12);
        let db = p.db().expect("db");
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert_eq!(meta.topic_verdict.as_deref(), Some("ok"));
        assert_eq!(meta.topic_reason.as_deref(), Some("A history lecture."));
        assert_eq!(video_outcome(&meta), VideoOutcome::Kept);

        // One request, with the title and the whole transcript.
        let requests = log.lock().expect("log").clone();
        assert_eq!(requests.len(), 1);
        let user = requests[0]["messages"][1]["content"].as_str().expect("user message");
        assert!(user.starts_with(&format!("Video Title: {}\nVideo ID: {}\n", v.title, v.id)), "{}", user);
        assert!(user.contains(text.trim()), "whole transcript in the request");
    }

    // ---- incomplete caption tracks ----------------------------------------------

    fn coverage(duration: Option<f64>, until: Option<f64>, source: Option<&str>) -> TranscriptCoverage {
        TranscriptCoverage {
            video_duration_sec: duration,
            covered_until_sec: until,
            source: source.map(str::to_string),
        }
    }

    #[test]
    fn only_a_caption_track_that_stops_early_counts_as_incomplete() {
        let pct = |c: TranscriptCoverage| incomplete_caption_pct(Some(&c));
        assert_eq!(pct(coverage(Some(1000.0), Some(300.0), Some("builtin"))), Some(30));
        assert_eq!(pct(coverage(Some(3600.0), Some(180.0), Some("subtitles"))), Some(5));
        // Source not recorded: the figures decide.
        assert_eq!(pct(coverage(Some(1000.0), Some(300.0), None)), Some(30));
        // Reaches the end (90% or within 20 seconds of it).
        assert_eq!(pct(coverage(Some(1000.0), Some(900.0), Some("builtin"))), None);
        assert_eq!(pct(coverage(Some(31.0), Some(29.0), Some("builtin"))), None);
        // Whisper heard the full audio: the rest has no speech.
        assert_eq!(pct(coverage(Some(1000.0), Some(300.0), Some("whisper"))), None);
        // Nothing to compare: nothing is claimed.
        assert_eq!(pct(coverage(None, Some(300.0), Some("builtin"))), None);
        assert_eq!(pct(coverage(Some(1000.0), None, Some("builtin"))), None);
        assert_eq!(incomplete_caption_pct(None), None);

        assert_eq!(partial_transcript_label(30), "stored transcript reaches 30% of the video");
    }

    #[test]
    fn the_read_label_never_says_whole_transcript_for_an_incomplete_track() {
        let meta = InsightMeta { sections_total: Some(1), transcript_chars: Some(4_000), ..Default::default() };
        // All of the stored text was read ...
        assert_eq!(meta.whole_transcript_note().as_deref(), Some("whole transcript (1 section)"));
        // ... but the stored text is 30% of the video.
        let partial = coverage(Some(1000.0), Some(300.0), Some("builtin"));
        assert_eq!(
            transcript_read_label(&meta, Some(&partial)).as_deref(),
            Some("stored transcript reaches 30% of the video")
        );
        let complete = coverage(Some(1000.0), Some(990.0), Some("builtin"));
        assert_eq!(transcript_read_label(&meta, Some(&complete)).as_deref(), Some("whole transcript (1 section)"));
        assert_eq!(transcript_read_label(&meta, None).as_deref(), Some("whole transcript (1 section)"));
        let whisper = coverage(Some(1000.0), Some(300.0), Some("whisper"));
        assert_eq!(transcript_read_label(&meta, Some(&whisper)).as_deref(), Some("whole transcript (1 section)"));

        // The crawl log line carries the same label.
        assert_eq!(
            with_read_note("Slop Score: 20/100 - KEPT (High Signal)".to_string(), VideoOutcome::Kept, Some(30)),
            "Slop Score: 20/100 - KEPT (High Signal) [stored transcript reaches 30% of the video]"
        );
        assert_eq!(with_read_note("line".to_string(), VideoOutcome::Kept, None), "line");
        assert_eq!(with_read_note("line".to_string(), VideoOutcome::Insufficient, Some(30)), "line");
    }

    #[tokio::test]
    async fn a_short_incomplete_caption_track_is_never_stored_as_not_enough_information() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Nothing listens on this port: an LLM request would fail differently.
        let p = pipeline(&dir, Some("http://127.0.0.1:9/v1".to_string()));
        // One hour of video, captions that stop after three minutes (63 words).
        let v = video("vidPART0001", "A one hour talk", "pending");
        let text = "These are the first words of a long talk. ".repeat(7);
        seed(&p, &v, Some(&text));
        let db = p.db().expect("db");
        db.set_transcript_coverage(&v.id, Some(3600.0), Some(180.0), Some("builtin")).expect("coverage");
        let transcript = db.get_transcript(&v.id).expect("read").expect("row");

        // After a fresh extraction (process-video, crawl).
        let err = p
            .judge_transcript(p.db().expect("db"), &v, &transcript)
            .await
            .expect_err("an incomplete transcript is not a silent video");
        let shown = err.to_string();
        assert!(shown.contains("Transcript incomplete"), "{}", shown);
        assert!(shown.contains("stored transcript reaches 5% of the video"), "{}", shown);
        // The video's own failure: counted, and the transcript is fetched again.
        assert_eq!(failure_cause(&err), FailureCause::Video);
        assert!(db.get_insight(&v.id).expect("read").is_none());
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, "pending");
        // The next crawl does not reuse the incomplete track.
        assert!(stored_transcript(&db, &v.id).is_some());
        assert!(reusable_transcript(&db, &v.id).is_none());

        // The same from the stored transcript (ytc reanalyze): an error, nothing stored.
        let err = p.reanalyze_video(&v.id).await.expect_err("incomplete");
        assert!(err.to_string().contains("Transcript incomplete"), "{}", err);
        assert!(db.get_insight(&v.id).expect("read").is_none());

        // A Whisper transcript of the same length is the whole audio: that
        // video really holds too little.
        db.set_transcript_coverage(&v.id, Some(3600.0), Some(180.0), Some("whisper")).expect("coverage");
        assert!(reusable_transcript(&db, &v.id).is_some());
        let insight = p
            .judge_transcript(p.db().expect("db"), &v, &transcript)
            .await
            .expect("stored as insufficient");
        assert_eq!(insight.analyzer_type, INSUFFICIENT_ANALYZER);
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert_eq!(meta.info_status.as_deref(), Some(INFO_INSUFFICIENT));
    }

    #[tokio::test]
    async fn an_incomplete_caption_track_with_enough_text_is_judged_and_keeps_its_true_coverage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = pipeline(&dir, Some(stub_llm(GOOD_ANSWER)));
        let v = video("vidPART0002", "Sourdough at home, part one of a long talk", "pending");
        let text = long_text();
        seed(&p, &v, Some(&text));
        let db = p.db().expect("db");
        // Whisper could not run (not a rate limit): the track that ends at 30% was used.
        db.set_transcript_coverage(&v.id, Some(1000.0), Some(300.0), Some("subtitles")).expect("coverage");
        let transcript = db.get_transcript(&v.id).expect("read").expect("row");

        let insight = p.judge_transcript(p.db().expect("db"), &v, &transcript).await.expect("judged");
        assert_eq!(insight.analyzer_type, "llm:stub-model");
        assert_eq!(db.get_video(&v.id).expect("video").expect("row").status, STATUS_PROCESSED);

        // The coverage on record is the track's own, and the label says so.
        let stored = db.get_transcript_coverage(&v.id).expect("read").expect("row");
        assert_eq!((stored.video_duration_sec, stored.covered_until_sec), (Some(1000.0), Some(300.0)));
        assert_eq!(stored.video_coverage_pct(), Some(30));
        let meta = db.get_insight_meta(&v.id).expect("meta").expect("row");
        assert_eq!(meta.info_status.as_deref(), Some(INFO_SUFFICIENT));
        assert_eq!(
            transcript_read_label(&meta, Some(&stored)).as_deref(),
            Some("stored transcript reaches 30% of the video")
        );
        let (outcome, line) = outcome_report(&insight, &meta, p.max_slop);
        assert_eq!(
            with_read_note(line, outcome, incomplete_caption_pct(Some(&stored))),
            "Slop Score: 20/100 - KEPT (High Signal) [stored transcript reaches 30% of the video]"
        );
    }

    // ---- request timeout --------------------------------------------------------

    #[tokio::test]
    async fn a_timeout_is_a_failed_request_of_a_reachable_engine_and_two_in_a_row_stop_the_crawl() {
        let dir = tempfile::tempdir().expect("tempdir");
        let v = video("vidSLOW0001", "Sourdough at home", "pending");
        let before = seed_heuristic(&dir, &v, &long_text());

        let mut p = pipeline(&dir, Some(stub_llm_slow(GOOD_ANSWER, Duration::from_millis(1_500))));
        p.analyzer = p.analyzer.clone().with_request_timeout(Duration::from_millis(300));
        let err = p.reanalyze_video(&v.id).await.expect_err("the stub answers too late");

        let unavailable = err.downcast_ref::<LlmUnavailable>().expect("typed as an LLM problem");
        assert!(unavailable.is_timeout());
        assert!(!unavailable.engine_down);
        // Not counted against the video, which stays as it was.
        assert_eq!(failure_cause(&err), FailureCause::Llm);
        assert_heuristic_insight_unchanged(&p, &v, &before);
        // The message names the timeout and the setting that raises it.
        let shown = format!("{:#}", err);
        assert!(shown.contains("timed out after 0.3 s"), "{}", shown);
        assert!(shown.contains("YTC_LLM_TIMEOUT_SECS"), "{}", shown);
        assert!(!shown.contains("engine not available"), "{}", shown);

        // One timeout fails that video only; the second in a row stops the crawl.
        assert!(!stops_crawl(unavailable, 1));
        assert!(stops_crawl(unavailable, 2));
        let hint = llm_stop_hint(unavailable);
        assert!(hint.contains("YTC_LLM_TIMEOUT_SECS"), "{}", hint);
        assert!(!hint.contains("Start the LLM engine"), "{}", hint);
    }

    #[test]
    fn an_engine_that_is_down_stops_the_crawl_at_once() {
        let down = LlmUnavailable { reason: "http://x/v1 - connection refused".to_string(), engine_down: true };
        assert!(stops_crawl(&down, 1));
        assert_eq!(llm_stop_hint(&down), "Start the LLM engine and crawl again.");

        // A wrong key answers for every video alike: the second one stops it.
        let wrong_key = LlmUnavailable { reason: "http://x/v1 answered HTTP 401".to_string(), engine_down: false };
        assert!(!stops_crawl(&wrong_key, 1));
        assert!(stops_crawl(&wrong_key, 2));
        assert!(llm_stop_hint(&wrong_key).contains("settings"));
    }
}
