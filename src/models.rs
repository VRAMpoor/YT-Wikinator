use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub id: String,
    pub title: String,
    pub url: String,
    #[serde(default = "default_status")]
    pub status: String, // "active", "high_signal", "muted"
    pub rolling_slop_score: Option<f64>,
    pub created_at: String,
    pub last_crawled_at: Option<String>,
}

fn default_status() -> String {
    "active".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelHealth {
    pub id: String,
    pub title: String,
    pub status: String,
    /// Analysed videos of the channel: the sum of `kept_count` and the four
    /// counts below.
    pub total_evaluated: usize,
    pub kept_count: usize,
    /// Rolling average of the last scored videos. None (JSON `null`) means no
    /// video was scored yet; `Some(0.0)` is a real score, the best possible
    /// one. Consumers must test for None/null, never for "falsy" or `> 0`.
    pub avg_slop_score: Option<f64>,
    pub last_crawled_at: Option<String>,
    // The four counts below and `kept_count` are disjoint: every insight row
    // of the channel is in exactly one of them.
    /// Judged on quality and found above the slop cutoff.
    pub below_cutoff_count: usize,
    /// Topic rejected by the model's topic verdict (fringe / nonsense / politics).
    pub off_topic_count: usize,
    /// The whole transcript held too little to judge. Not slop, not scored.
    pub insufficient_count: usize,
    /// Disqualified by the owner (`ytc exclude`).
    pub excluded_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    pub id: String,
    pub channel_id: String,
    pub title: String,
    pub url: String,
    pub published_at: String,
    pub status: String, // "pending", "discovered", "processed", "heuristic", "failed"
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptCue {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SponsorSegment {
    pub category: String,
    pub action_type: String,
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanedTranscript {
    pub video_id: String,
    pub raw_char_count: usize,
    pub cleaned_char_count: usize,
    pub stripped_ad_seconds: f64,
    pub stripped_segments_count: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightResult {
    pub video_id: String,
    pub slop_score: u32, // 0 = Pure signal, 100 = Pure slop
    pub is_worth_keeping: bool,
    pub quality_rating: String,
    pub summary: String,
    pub key_insights: Vec<String>,
    pub takeaways: Vec<String>,
    pub quotes: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub mentioned_creators: Vec<String>,
    pub analyzer_type: String, // "heuristic", "llm:<model>" or "info:insufficient"
    /// Name of one of the user's categories (`Category::name`), or None. A
    /// label only: score, verdict and the keep decision are never computed
    /// from it. (The category list is part of the prompt, so defining
    /// categories can still shift individual scores slightly.)
    #[serde(default)]
    pub category: Option<String>,
}

/// Longest category name, in characters.
pub const CATEGORY_NAME_MAX_CHARS: usize = 60;
/// Longest category description, in characters.
pub const CATEGORY_DESCRIPTION_MAX_CHARS: usize = 300;

/// A subject category the user defined (`ytc category add`). There are none
/// until the user adds one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Category {
    pub name: String,
    /// What belongs in the category, in the user's words. The analyzer shows
    /// it to the model next to the name.
    pub description: Option<String>,
    pub created_at: String,
}

/// The category `raw` names, in the user's spelling: case and surrounding
/// whitespace do not matter. None when `raw` names none of `categories`.
pub fn match_category(categories: &[Category], raw: &str) -> Option<String> {
    let wanted = raw.trim().to_lowercase();
    if wanted.is_empty() {
        return None;
    }
    categories.iter().find(|c| c.name.to_lowercase() == wanted).map(|c| c.name.clone())
}

/// Topic verdict of a video whose subject is acceptable.
pub const TOPIC_OK: &str = "ok";
/// Values of `insights.topic_verdict`. Only fringe, nonsense and politics
/// reject a video for its subject; quality is judged separately.
pub const TOPIC_VERDICTS: [&str; 4] = [TOPIC_OK, "fringe", "nonsense", "politics"];

pub const INFO_SUFFICIENT: &str = "sufficient";
pub const INFO_INSUFFICIENT: &str = "insufficient";
/// Values of `insights.info_status`.
pub const INFO_STATUSES: [&str; 2] = [INFO_SUFFICIENT, INFO_INSUFFICIENT];

/// Values of `transcripts.source`: which tier produced the transcript.
pub const TRANSCRIPT_SOURCES: [&str; 3] = ["builtin", "subtitles", "whisper"];

impl InsightResult {
    /// The score as it is shown: "37/100".
    pub fn score_label(&self) -> String {
        format!("{}/100", self.slop_score)
    }
}

/// Facts about an insight row that `InsightResult` does not carry: the
/// verdicts, the owner's decision, and how the transcript was read.
/// Read with `Database::get_insight_meta` / `Database::list_insight_meta`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InsightMeta {
    /// Length of the cleaned transcript the analyzer read, all of it. None =
    /// no analysis ran (a "not enough information" row).
    #[serde(default)]
    pub transcript_chars: Option<u64>,
    /// "ok", "fringe", "nonsense" or "politics". None = no verdict recorded
    /// (the heuristic cannot judge a subject).
    #[serde(default)]
    pub topic_verdict: Option<String>,
    #[serde(default)]
    pub topic_reason: Option<String>,
    /// "sufficient" or "insufficient". None = not recorded.
    #[serde(default)]
    pub info_status: Option<String>,
    #[serde(default)]
    pub info_reason: Option<String>,
    /// True when the owner disqualified the video. Survives re-analysis.
    #[serde(default)]
    pub excluded_manual: bool,
    #[serde(default)]
    pub excluded_reason: Option<String>,
    #[serde(default)]
    pub excluded_at: Option<String>,
    /// True when the owner chose the category (or "no category") by hand:
    /// an analysis then leaves `insights.category` as it is.
    #[serde(default)]
    pub category_manual: bool,
    /// Consecutive sections the whole transcript was read in (1 = one
    /// request). An analysis that cannot read a section fails and stores
    /// nothing, so there is no count of sections left out. None = no
    /// analysis ran.
    #[serde(default)]
    pub sections_total: Option<u32>,
    /// The effective decision, computed by the database with the one kept
    /// rule: worth keeping AND not disqualified AND topic ok AND enough
    /// information. `InsightResult::is_worth_keeping` alone is only the
    /// quality part of it.
    #[serde(default)]
    pub kept: bool,
}

impl InsightMeta {
    /// True unless the topic verdict rejects the subject. A row without a
    /// verdict is not rejected.
    pub fn topic_ok(&self) -> bool {
        self.topic_verdict.as_deref().is_none_or(|v| v == TOPIC_OK)
    }

    /// True when the transcript simply held too little to judge.
    pub fn is_insufficient(&self) -> bool {
        self.info_status.as_deref() == Some(INFO_INSUFFICIENT)
    }

    /// What to show in place of the score of `insight`: "37/100" for a scored
    /// video, "Not enough information (..)" when the transcript held too
    /// little (the stored score of such a row is a placeholder, not a
    /// measurement).
    pub fn score_label(&self, insight: &InsightResult) -> String {
        if self.is_insufficient() {
            return match self.info_reason.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
                Some(reason) => format!("Not enough information ({})", reason),
                None => "Not enough information".to_string(),
            };
        }
        insight.score_label()
    }

    /// "whole transcript (3 sections)": how the analysis read the transcript.
    /// None for a row no analysis wrote.
    pub fn whole_transcript_note(&self) -> Option<String> {
        let total = self.sections_total.filter(|total| *total > 0)?;
        Some(format!(
            "whole transcript ({} section{})",
            total,
            if total == 1 { "" } else { "s" }
        ))
    }

    /// Why the video is not in the library, in the words the UI shows; None
    /// for a kept video. The checks run in order of precedence: the owner's
    /// decision first, plain quality last.
    pub fn rejection_label(&self) -> Option<String> {
        if self.kept {
            return None;
        }
        let with_reason = |label: &str, reason: &Option<String>| match reason.as_deref().map(str::trim) {
            Some(r) if !r.is_empty() => format!("{} ({})", label, r),
            _ => label.to_string(),
        };
        if self.excluded_manual {
            return Some(with_reason("Disqualified", &self.excluded_reason));
        }
        if !self.topic_ok() {
            let verdict = self.topic_verdict.as_deref().unwrap_or("off-topic");
            return Some(with_reason(&format!("Off-topic: {}", verdict), &self.topic_reason));
        }
        if self.is_insufficient() {
            return Some(with_reason("Not enough information", &self.info_reason));
        }
        Some("Below quality cutoff".to_string())
    }
}

/// The parts of a judgment that `InsightResult` does not carry. Passed to
/// `Database::save_insight_judged` so that row, verdicts and what was read are
/// written together and the channel average never sees a half-written row.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InsightJudgment {
    /// One of `TOPIC_VERDICTS`; None leaves the verdict unrecorded.
    #[serde(default)]
    pub topic_verdict: Option<String>,
    #[serde(default)]
    pub topic_reason: Option<String>,
    /// One of `INFO_STATUSES`; None leaves it unrecorded.
    #[serde(default)]
    pub info_status: Option<String>,
    #[serde(default)]
    pub info_reason: Option<String>,
    /// Sections the whole transcript was read in; None when no analysis ran.
    #[serde(default)]
    pub sections_total: Option<u32>,
    /// Length of the cleaned transcript that was read; None when no analysis ran.
    #[serde(default)]
    pub transcript_chars: Option<u64>,
}

/// How much of a video its stored transcript covers, and where it came from.
/// Read with `Database::get_transcript_coverage`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TranscriptCoverage {
    /// Length of the video in seconds. None = not recorded.
    #[serde(default)]
    pub video_duration_sec: Option<f64>,
    /// End of the last transcript cue in seconds. None = not recorded.
    #[serde(default)]
    pub covered_until_sec: Option<f64>,
    /// "builtin", "subtitles" or "whisper". None = not recorded.
    #[serde(default)]
    pub source: Option<String>,
}

impl TranscriptCoverage {
    /// Share of the video the transcript reaches, 0..=100; None when either
    /// figure is missing. A shortfall is never rounded up to 100. Cues that end after the reported duration count
    /// as complete.
    pub fn video_coverage_pct(&self) -> Option<u32> {
        let duration = self.video_duration_sec.filter(|d| d.is_finite() && *d > 0.0)?;
        let covered = self.covered_until_sec.filter(|c| c.is_finite() && *c >= 0.0)?;
        if covered >= duration {
            return Some(100);
        }
        Some((((covered / duration) * 100.0).round() as u32).min(99))
    }

    /// Display name of the transcript source; None when it was not recorded.
    pub fn source_label(&self) -> Option<&'static str> {
        match self.source.as_deref() {
            Some("builtin") => Some("YouTube transcript"),
            Some("subtitles") => Some("YouTube subtitle track"),
            Some("whisper") => Some("local Whisper transcription of the audio"),
            _ => None,
        }
    }
}

/// A video the owner disqualified, as listed by `Database::list_excluded`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedVideo {
    pub video_id: String,
    pub title: String,
    pub channel_id: String,
    /// None when the channel is not tracked.
    pub channel_title: Option<String>,
    pub reason: Option<String>,
    pub excluded_at: Option<String>,
    pub slop_score: u32,
}

/// What `Database::merge_from` did, in exact row counts. For every table the
/// counts add up to the number of rows read from the other database.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeStats {
    pub channels_added: usize,
    /// Existing channels that took over a status set by hand in the other database.
    pub channels_updated: usize,
    pub channels_unchanged: usize,
    pub videos_added: usize,
    /// Existing videos whose processing status followed an imported analysis.
    pub videos_updated: usize,
    pub videos_unchanged: usize,
    pub transcripts_added: usize,
    /// The other database had the longer transcript.
    pub transcripts_replaced: usize,
    pub transcripts_kept_current: usize,
    pub insights_added: usize,
    /// The other database had the better analysis (LLM over heuristic, or the more recent one).
    pub insights_replaced: usize,
    pub insights_kept_current: usize,
    /// Manual disqualifications taken over onto an analysis that was kept.
    pub exclusions_carried: usize,
    /// Transcripts and insights of the other database whose video row is
    /// missing there; they cannot be shown and were not copied.
    pub orphans_skipped: usize,
    /// Categories defined only in the other database. Not part of the
    /// per-table sums above.
    #[serde(default)]
    pub categories_added: usize,
    /// Categories set by hand in the other database, taken over onto an
    /// analysis that was kept.
    #[serde(default)]
    pub manual_categories_carried: usize,
}

impl MergeStats {
    /// True when the merge changed nothing in the current database.
    pub fn is_noop(&self) -> bool {
        self.channels_added
            + self.channels_updated
            + self.videos_added
            + self.videos_updated
            + self.transcripts_added
            + self.transcripts_replaced
            + self.insights_added
            + self.insights_replaced
            + self.exclusions_carried
            + self.categories_added
            + self.manual_categories_carried
            == 0
    }
}

impl std::fmt::Display for MergeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Channels:    {} added, {} updated (manual status), {} unchanged",
            self.channels_added, self.channels_updated, self.channels_unchanged
        )?;
        writeln!(
            f,
            "Videos:      {} added, {} updated, {} unchanged",
            self.videos_added, self.videos_updated, self.videos_unchanged
        )?;
        writeln!(
            f,
            "Transcripts: {} added, {} replaced (longer), {} kept as they were",
            self.transcripts_added, self.transcripts_replaced, self.transcripts_kept_current
        )?;
        write!(
            f,
            "Insights:    {} added, {} replaced (better analysis), {} kept as they were",
            self.insights_added, self.insights_replaced, self.insights_kept_current
        )?;
        if self.exclusions_carried > 0 {
            write!(f, "\nDisqualifications taken over: {}", self.exclusions_carried)?;
        }
        if self.orphans_skipped > 0 {
            write!(f, "\nSkipped (no video row in the other database): {}", self.orphans_skipped)?;
        }
        if self.categories_added > 0 {
            write!(f, "\nCategories taken over: {}", self.categories_added)?;
        }
        if self.manual_categories_carried > 0 {
            write!(f, "\nCategories set by hand taken over: {}", self.manual_categories_carried)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insight(analyzer_type: &str, quality_rating: &str, slop_score: u32) -> InsightResult {
        InsightResult {
            video_id: "vid".to_string(),
            slop_score,
            is_worth_keeping: false,
            quality_rating: quality_rating.to_string(),
            summary: String::new(),
            key_insights: vec![],
            takeaways: vec![],
            quotes: vec![],
            topics: vec![],
            mentioned_creators: vec![],
            analyzer_type: analyzer_type.to_string(),
            category: None,
        }
    }

    #[test]
    fn a_zero_average_is_serialized_as_a_number_and_a_missing_one_as_null() {
        let health = |avg: Option<f64>| ChannelHealth {
            id: "UC1".to_string(),
            title: "T".to_string(),
            status: "high_signal".to_string(),
            total_evaluated: 3,
            kept_count: 3,
            avg_slop_score: avg,
            last_crawled_at: None,
            below_cutoff_count: 0,
            off_topic_count: 0,
            insufficient_count: 0,
            excluded_count: 0,
        };
        let zero = serde_json::to_value(health(Some(0.0))).expect("serialize");
        assert!(zero["avg_slop_score"].is_number());
        assert_eq!(zero["avg_slop_score"].as_f64(), Some(0.0));
        let none = serde_json::to_value(health(None)).expect("serialize");
        assert!(none["avg_slop_score"].is_null());

        let back: ChannelHealth = serde_json::from_value(zero).expect("deserialize");
        assert_eq!(back.avg_slop_score, Some(0.0));
    }

    #[test]
    fn insight_meta_deserializes_from_an_empty_object() {
        let meta: InsightMeta = serde_json::from_str("{}").expect("defaults");
        assert_eq!(meta, InsightMeta::default());
        let judgment: InsightJudgment = serde_json::from_str("{}").expect("defaults");
        assert_eq!(judgment, InsightJudgment::default());
        let coverage: TranscriptCoverage = serde_json::from_str("{}").expect("defaults");
        assert_eq!(coverage, TranscriptCoverage::default());
    }

    #[test]
    fn a_category_is_matched_whatever_its_case_and_returned_in_the_users_spelling() {
        let category = |name: &str| Category {
            name: name.to_string(),
            description: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let categories = [category("Ancient History"), category("Baking")];
        assert_eq!(match_category(&categories, "Baking").as_deref(), Some("Baking"));
        assert_eq!(match_category(&categories, "  ancient history ").as_deref(), Some("Ancient History"));
        assert_eq!(match_category(&categories, "BAKING").as_deref(), Some("Baking"));
        for unknown in ["", "  ", "History", "Baking and more", "null"] {
            assert_eq!(match_category(&categories, unknown), None, "{:?}", unknown);
        }
        assert_eq!(match_category(&[], "Baking"), None);
    }

    #[test]
    fn an_insight_without_a_category_field_deserializes_with_none() {
        let json = r#"{"video_id":"vid","slop_score":20,"is_worth_keeping":true,"quality_rating":"High Signal",
                       "summary":"s","key_insights":[],"takeaways":[],"quotes":[],"analyzer_type":"llm:m"}"#;
        let insight: InsightResult = serde_json::from_str(json).expect("insight JSON");
        assert_eq!(insight.category, None);
    }

    #[test]
    fn only_a_recorded_non_ok_verdict_rejects_a_topic() {
        let verdict = |v: Option<&str>| InsightMeta { topic_verdict: v.map(str::to_string), ..Default::default() };
        assert!(verdict(None).topic_ok());
        assert!(verdict(Some("ok")).topic_ok());
        for rejected in ["fringe", "nonsense", "politics"] {
            assert!(TOPIC_VERDICTS.contains(&rejected));
            assert!(!verdict(Some(rejected)).topic_ok());
        }
        let info = |s: Option<&str>| InsightMeta { info_status: s.map(str::to_string), ..Default::default() };
        assert!(!info(None).is_insufficient());
        assert!(!info(Some(INFO_SUFFICIENT)).is_insufficient());
        assert!(info(Some(INFO_INSUFFICIENT)).is_insufficient());
    }

    #[test]
    fn the_read_line_names_the_sections_and_is_absent_without_an_analysis() {
        let meta = |total: Option<u32>| InsightMeta { sections_total: total, ..Default::default() };
        assert_eq!(meta(Some(3)).whole_transcript_note().as_deref(), Some("whole transcript (3 sections)"));
        assert_eq!(meta(Some(1)).whole_transcript_note().as_deref(), Some("whole transcript (1 section)"));
        assert_eq!(meta(None).whole_transcript_note(), None);
        assert_eq!(meta(Some(0)).whole_transcript_note(), None);
    }

    #[test]
    fn a_rejection_names_its_cause_with_the_owners_decision_first() {
        let kept = InsightMeta { kept: true, ..Default::default() };
        assert_eq!(kept.rejection_label(), None);

        let quality = InsightMeta::default();
        assert_eq!(quality.rejection_label().as_deref(), Some("Below quality cutoff"));

        let short = InsightMeta {
            info_status: Some(INFO_INSUFFICIENT.to_string()),
            info_reason: Some("42 words of speech".to_string()),
            ..Default::default()
        };
        assert_eq!(short.rejection_label().as_deref(), Some("Not enough information (42 words of speech)"));

        let politics = InsightMeta { topic_verdict: Some("politics".to_string()), ..short.clone() };
        assert_eq!(politics.rejection_label().as_deref(), Some("Off-topic: politics"));

        let excluded = InsightMeta {
            excluded_manual: true,
            excluded_reason: Some(" duplicate ".to_string()),
            ..politics.clone()
        };
        assert_eq!(excluded.rejection_label().as_deref(), Some("Disqualified (duplicate)"));
    }

    #[test]
    fn the_score_label_shows_a_score_or_why_there_is_none() {
        let scored = insight("llm:m", "High Signal", 37);
        assert_eq!(InsightMeta::default().score_label(&scored), "37/100");
        let short = InsightMeta {
            info_status: Some(INFO_INSUFFICIENT.to_string()),
            info_reason: Some("42 words".to_string()),
            ..Default::default()
        };
        let placeholder = insight("info:insufficient", "Not scored", 0);
        assert_eq!(short.score_label(&placeholder), "Not enough information (42 words)");
        assert_eq!(
            InsightMeta { info_reason: None, ..short }.score_label(&placeholder),
            "Not enough information"
        );
    }

    #[test]
    fn video_coverage_needs_both_figures_and_never_rounds_up() {
        let cov = |d: Option<f64>, c: Option<f64>| TranscriptCoverage {
            video_duration_sec: d,
            covered_until_sec: c,
            source: None,
        };
        assert_eq!(cov(None, Some(10.0)).video_coverage_pct(), None);
        assert_eq!(cov(Some(600.0), None).video_coverage_pct(), None);
        assert_eq!(cov(Some(0.0), Some(10.0)).video_coverage_pct(), None);
        assert_eq!(cov(Some(f64::NAN), Some(10.0)).video_coverage_pct(), None);
        assert_eq!(cov(Some(600.0), Some(300.0)).video_coverage_pct(), Some(50));
        assert_eq!(cov(Some(600.0), Some(599.9)).video_coverage_pct(), Some(99));
        assert_eq!(cov(Some(600.0), Some(612.0)).video_coverage_pct(), Some(100));

        let source = |s: &str| TranscriptCoverage { source: Some(s.to_string()), ..Default::default() };
        for known in TRANSCRIPT_SOURCES {
            assert!(source(known).source_label().is_some(), "{}", known);
        }
        assert_eq!(source("telepathy").source_label(), None);
        assert_eq!(TranscriptCoverage::default().source_label(), None);
    }

    #[test]
    fn merge_stats_report_every_count_and_know_a_no_op() {
        let stats = MergeStats {
            channels_added: 1,
            videos_added: 2,
            transcripts_replaced: 3,
            insights_replaced: 4,
            insights_kept_current: 5,
            exclusions_carried: 1,
            orphans_skipped: 2,
            ..Default::default()
        };
        assert!(!stats.is_noop());
        let text = stats.to_string();
        assert!(text.contains("Channels:    1 added, 0 updated (manual status), 0 unchanged"), "{}", text);
        assert!(text.contains("Transcripts: 0 added, 3 replaced (longer), 0 kept as they were"), "{}", text);
        assert!(text.contains("Insights:    0 added, 4 replaced (better analysis), 5 kept as they were"), "{}", text);
        assert!(text.contains("Disqualifications taken over: 1"), "{}", text);
        assert!(text.contains("Skipped (no video row in the other database): 2"), "{}", text);
        assert!(!text.contains("Categories"), "{}", text);
        let with_categories = MergeStats { categories_added: 2, ..Default::default() };
        assert!(!with_categories.is_noop());
        assert!(with_categories.to_string().contains("Categories taken over: 2"));

        let idle = MergeStats { insights_kept_current: 9, videos_unchanged: 9, ..Default::default() };
        assert!(idle.is_noop());
        assert!(!idle.to_string().contains("Skipped"));
    }
}
