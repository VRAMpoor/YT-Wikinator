use crate::models::{
    match_category, Category, Channel, ChannelHealth, CleanedTranscript, ExcludedVideo, InsightJudgment, InsightMeta,
    InsightResult, MergeStats, TranscriptCoverage, Video, CATEGORY_DESCRIPTION_MAX_CHARS, CATEGORY_NAME_MAX_CHARS,
    INFO_STATUSES, TOPIC_VERDICTS, TRANSCRIPT_SOURCES,
};
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Not disqualified by hand, topic not rejected, enough information to judge.
/// NULL in the two verdict columns means "not recorded" and rejects nothing.
/// Expects the insights table as alias `i`.
macro_rules! eligible_sql {
    () => {
        "COALESCE(i.excluded_manual, 0) = 0 \
         AND (i.topic_verdict IS NULL OR i.topic_verdict = 'ok') \
         AND (i.info_status IS NULL OR i.info_status = 'sufficient')"
    };
}

/// The kept rule as a literal, so other constants can embed it. Use `KEPT_SQL`.
macro_rules! kept_sql {
    () => {
        concat!("i.is_worth_keeping = 1 AND ", eligible_sql!())
    };
}

/// THE rule for "this video is in the library". Every query that decides or
/// lists kept videos uses this fragment and nothing else, so wiki, topic
/// lists, search and counts cannot disagree. Expects the insights table as
/// alias `i`.
pub const KEPT_SQL: &str = kept_sql!();

/// Rows whose slop score may enter a channel's rolling average: measured by
/// the LLM (no heuristic estimate), and not disqualified, topic-rejected or
/// too short to judge. A channel is therefore never muted or demoted because
/// of a few short videos. Expects alias `i`.
const SCORED_SQL: &str = concat!("i.analyzer_type NOT LIKE 'heuristic%' AND ", eligible_sql!());

/// `PRAGMA user_version` of a database with the layout of `SCHEMA_SQL`.
const SCHEMA_VERSION: i64 = 1;

/// The complete layout of a database.
const SCHEMA_SQL: &str = r#"
    CREATE TABLE channels (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        -- 1 once the user set the status by hand: recalculation then leaves it alone.
        status_manual INTEGER NOT NULL DEFAULT 0,
        rolling_slop_score REAL,
        created_at TEXT NOT NULL,
        last_crawled_at TEXT
    );

    CREATE TABLE videos (
        id TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL,
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        published_at TEXT NOT NULL,
        status TEXT NOT NULL,
        -- How often processing of the video has failed; lets callers stop retrying.
        fail_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );

    CREATE TABLE transcripts (
        video_id TEXT PRIMARY KEY,
        raw_char_count INTEGER NOT NULL,
        cleaned_char_count INTEGER NOT NULL,
        stripped_ad_seconds REAL NOT NULL,
        stripped_segments_count INTEGER NOT NULL,
        text TEXT NOT NULL,
        -- How much of the video the transcript covers, and which tier made it.
        video_duration_sec REAL,
        covered_until_sec REAL,
        source TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(video_id) REFERENCES videos(id)
    );

    CREATE TABLE insights (
        video_id TEXT PRIMARY KEY,
        slop_score INTEGER NOT NULL,
        is_worth_keeping INTEGER NOT NULL,
        quality_rating TEXT NOT NULL,
        summary TEXT NOT NULL,
        key_insights_json TEXT NOT NULL,
        takeaways_json TEXT NOT NULL,
        quotes_json TEXT NOT NULL,
        topics_json TEXT NOT NULL DEFAULT '[]',
        mentioned_creators_json TEXT NOT NULL DEFAULT '[]',
        analyzer_type TEXT NOT NULL,
        category TEXT,
        -- 1 once the owner chose the category (or no category) by hand: an
        -- analysis then leaves category as it is. Never written by save_insight.
        category_manual INTEGER NOT NULL DEFAULT 0,
        -- 1 once 'ytc categorize' asked the model about this video with the
        -- categories as they are now, so a video that fits none is not asked
        -- on every run. Back to 0 whenever the categories change.
        category_asked INTEGER NOT NULL DEFAULT 0,
        -- Topic and quality are judged separately. ok|fringe|nonsense|politics;
        -- NULL = no verdict recorded.
        topic_verdict TEXT,
        topic_reason TEXT,
        -- sufficient|insufficient: whether the whole transcript held enough to judge.
        info_status TEXT,
        info_reason TEXT,
        -- An analysis reads the whole transcript or fails and stores nothing:
        -- the number of consecutive sections it was read in (1 = one request)
        -- and its length in bytes of the cleaned text. NULL = no analysis ran.
        sections_total INTEGER,
        transcript_chars INTEGER,
        -- The owner's disqualification. Never written by save_insight.
        excluded_manual INTEGER NOT NULL DEFAULT 0,
        excluded_reason TEXT,
        excluded_at TEXT,
        -- created_at keeps the time of the first analysis, updated_at is the
        -- time of the latest one.
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(video_id) REFERENCES videos(id)
    );

    CREATE TABLE categories (
        name TEXT PRIMARY KEY,
        description TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE topics (
        name TEXT PRIMARY KEY,
        created_at TEXT NOT NULL
    );

    CREATE TABLE video_topics (
        video_id TEXT NOT NULL,
        topic_name TEXT NOT NULL,
        PRIMARY KEY (video_id, topic_name),
        FOREIGN KEY(video_id) REFERENCES videos(id)
    );

    CREATE INDEX idx_videos_channel_id ON videos(channel_id);
    CREATE INDEX idx_insights_score ON insights(slop_score);
    CREATE INDEX idx_insights_keeping ON insights(is_worth_keeping);
    CREATE INDEX idx_video_topics_topic ON video_topics(topic_name);
"#;

/// What opening a file that does not have this layout fails with.
fn not_a_database(path: &Path) -> anyhow::Error {
    anyhow::anyhow!("{}: This file is not a YT-Wikinator database of this version.", path.display())
}

/// Version mark and number of schema objects of the database behind `conn`.
/// An error means the file is not an SQLite database at all.
fn schema_state(conn: &Connection) -> rusqlite::Result<(i64, i64)> {
    let version = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let objects = conn.query_row("SELECT COUNT(*) FROM sqlite_master", [], |r| r.get(0))?;
    Ok((version, objects))
}

/// File name and folder of the one database, below the app folder.
const DB_DIR_NAME: &str = "data";
const DB_FILE_NAME: &str = "ytc.db";

/// The folder that holds the app: the folder of the executable, or the
/// project root when the executable is a cargo build in `target/release` or
/// `target/debug`. Engine (`bin/llama`), model (`models/`) and database
/// (`data/`) live below it, so the folder can be moved as a whole.
pub fn app_root_for(exe: &Path) -> PathBuf {
    let dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let is_named = |p: &Path, names: &[&str]| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| names.iter().any(|want| n.eq_ignore_ascii_case(want)))
    };
    if is_named(dir, &["release", "debug"]) {
        if let Some(target) = dir.parent().filter(|t| is_named(t, &["target"])) {
            if let Some(root) = target.parent().filter(|r| !r.as_os_str().is_empty()) {
                return root.to_path_buf();
            }
        }
    }
    dir.to_path_buf()
}

/// `app_root_for` the running executable.
pub fn app_root() -> PathBuf {
    match std::env::current_exe() {
        Ok(exe) => app_root_for(&exe),
        // The OS did not tell where the executable is; there is then no app
        // folder to speak of and relative paths are all that is left.
        Err(_) => PathBuf::from("."),
    }
}

/// The one database lookup rule, after an explicit `--db`: the value of
/// `YTC_DB` when set and not blank, else `<app folder>/data/ytc.db`. Nothing
/// else is consulted: not the working directory and not whether a file
/// already exists.
pub fn db_path_for(app_root: &Path, env_value: Option<&str>) -> PathBuf {
    match env_value.map(str::trim).filter(|v| !v.is_empty()) {
        Some(value) => PathBuf::from(value),
        None => app_root.join(DB_DIR_NAME).join(DB_FILE_NAME),
    }
}

/// The one auto-mute rule: a rolling average above this value across at least
/// `AUTO_MUTE_MIN_VIDEOS` on-topic videos mutes a channel.
pub const AUTO_MUTE_SLOP_AVG: f64 = 75.0;
/// Minimum number of scored videos before the rolling average may change a
/// channel's status in either direction (mute or high_signal).
pub const AUTO_MUTE_MIN_VIDEOS: usize = 3;
const HIGH_SIGNAL_SLOP_AVG: f64 = 35.0;

/// Statuses a channel row may hold.
pub const CHANNEL_STATUSES: [&str; 3] = ["active", "high_signal", "muted"];

/// Status the automatic rule assigns for a rolling average over `scored`
/// videos. One or two videos are not evidence: they neither mute nor promote.
fn auto_channel_status(scored: usize, avg: f64) -> &'static str {
    if scored < AUTO_MUTE_MIN_VIDEOS {
        "active"
    } else if avg > AUTO_MUTE_SLOP_AVG {
        "muted"
    } else if avg <= HIGH_SIGNAL_SLOP_AVG {
        "high_signal"
    } else {
        "active"
    }
}

/// The line printed when the automatic rule changes a channel's status, or
/// None when nothing changed. Muted channels are not crawled, so the
/// change must be visible at the moment it happens and name the way back.
fn auto_status_change_notice(
    channel_id: &str,
    title: &str,
    status_before: &str,
    new_status: &str,
    avg: f64,
    scored: usize,
) -> Option<String> {
    if status_before == new_status {
        return None;
    }
    Some(format!(
        "Channel status changed automatically: {} ({}) {} -> {} (average slop {:.1}/100 over the last {} scored video(s)); undo with: ytc set-status {} active",
        title, channel_id, status_before, new_status, avg, scored, channel_id
    ))
}

/// Topic names of a topics_json value, trimmed, without empty entries.
/// Non-string array members are ignored.
fn topic_names(topics_json: &str) -> Vec<String> {
    let parsed: serde_json::Value = serde_json::from_str(topics_json).unwrap_or(serde_json::Value::Null);
    let mut names: Vec<String> = Vec::new();
    if let Some(items) = parsed.as_array() {
        for item in items {
            if let Some(name) = item.as_str().map(str::trim).filter(|n| !n.is_empty()) {
                if !names.iter().any(|n| n == name) {
                    names.push(name.to_string());
                }
            }
        }
    }
    names
}

/// Replaces the topic links of one video. Runs on the caller's transaction.
fn link_topics(conn: &Connection, video_id: &str, names: &[String], now: &str) -> Result<()> {
    conn.execute("DELETE FROM video_topics WHERE video_id = ?1", params![video_id])?;
    for name in names {
        conn.execute(
            "INSERT OR IGNORE INTO topics (name, created_at) VALUES (?1, ?2)",
            params![name, now],
        )?;
        conn.execute(
            "INSERT OR IGNORE INTO video_topics (video_id, topic_name) VALUES (?1, ?2)",
            params![video_id, name],
        )?;
    }
    Ok(())
}

/// Rebuilds every row of video_topics from insights.topics_json. Returns the
/// number of links written. Runs on the caller's transaction.
fn rebuild_topic_links_on(conn: &Connection) -> Result<usize> {
    let rows: Vec<(String, String)> = {
        let mut stmt = conn.prepare("SELECT video_id, COALESCE(topics_json, '[]') FROM insights")?;
        let mapped = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        mapped.collect::<rusqlite::Result<_>>()?
    };
    conn.execute("DELETE FROM video_topics", [])?;
    let now = Utc::now().to_rfc3339();
    let mut links = 0;
    for (video_id, topics_json) in &rows {
        let names = topic_names(topics_json);
        link_topics(conn, video_id, &names, &now)?;
        links += names.len();
    }
    Ok(links)
}

/// A text column as an option; blank counts as not recorded.
fn non_blank(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.trim().is_empty())
}

fn insight_meta_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(String, InsightMeta)> {
    let video_id: String = row.get(0)?;
    let transcript_chars: Option<i64> = row.get(1)?;
    let sections_total: Option<i64> = row.get(9)?;
    Ok((
        video_id,
        InsightMeta {
            transcript_chars: transcript_chars.and_then(|n| u64::try_from(n).ok()),
            topic_verdict: non_blank(row.get(2)?),
            topic_reason: non_blank(row.get(3)?),
            info_status: non_blank(row.get(4)?),
            info_reason: non_blank(row.get(5)?),
            excluded_manual: row.get::<_, i64>(6)? != 0,
            excluded_reason: non_blank(row.get(7)?),
            excluded_at: non_blank(row.get(8)?),
            category_manual: row.get::<_, i64>(11)? != 0,
            sections_total: sections_total.and_then(|n| u32::try_from(n).ok()),
            kept: row.get::<_, i64>(10)? != 0,
        },
    ))
}

/// Select list read by `insight_meta_from_row`; column 10 is the kept rule
/// evaluated by SQLite. Needs `FROM insights i`.
const INSIGHT_META_SELECT: &str = concat!(
    "SELECT i.video_id, i.transcript_chars, i.topic_verdict, i.topic_reason, i.info_status, \
     i.info_reason, i.excluded_manual, i.excluded_reason, i.excluded_at, i.sections_total, (",
    kept_sql!(),
    "), i.category_manual FROM insights i"
);

/// Fails unless `value` is one of `allowed`; `what` names the field in the error.
fn check_one_of(what: &str, value: &str, allowed: &[&str]) -> Result<()> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        anyhow::bail!("Unknown {} '{}' (expected one of: {})", what, value, allowed.join(", "))
    }
}

/// Trimmed text for a nullable column; blank becomes NULL.
fn clean_text(value: Option<&str>) -> Option<String> {
    value.map(str::trim).filter(|v| !v.is_empty()).map(str::to_string)
}

pub struct Database {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub video_id: String,
    pub title: String,
    pub slop_score: u32,
    /// The effective decision (the kept rule), not just the quality flag.
    pub is_worth_keeping: bool,
    pub quality_rating: String,
    pub summary: String,
    pub key_insights: Vec<String>,
    pub topics: Vec<String>,
    /// The user's category of the video, if it has one.
    pub category: Option<String>,
    /// Why a hit is not kept although its score may be low.
    pub topic_verdict: Option<String>,
    pub info_status: Option<String>,
    pub info_reason: Option<String>,
    pub excluded_manual: bool,
}

impl SearchHit {
    /// True when `slop_score` is not a measurement: a video with too little
    /// content to judge (placeholder 0, which must never be shown as "0/100").
    pub fn has_no_score(&self) -> bool {
        self.info_status.as_deref() == Some(crate::models::INFO_INSUFFICIENT)
    }

    /// What to show in place of the score: "37/100" for a scored hit or
    /// "Not enough information (..)".
    pub fn score_label(&self) -> String {
        if self.has_no_score() {
            return match self.info_reason.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
                Some(reason) => format!("Not enough information ({})", reason),
                None => "Not enough information".to_string(),
            };
        }
        format!("{}/100", self.slop_score)
    }
}

impl Database {
    /// Opens the database at `path`. A missing or empty file becomes a new
    /// database; a file that holds anything else than a database with this
    /// layout is refused and left as it is.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() && !p.exists() {
                std::fs::create_dir_all(p)?;
            }
        }

        let conn = Connection::open(path).context("Failed to open SQLite database")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        // Looked at before the first write: a file of another kind must not
        // even have its journal mode changed.
        match schema_state(&conn) {
            Ok((SCHEMA_VERSION, _)) => {}
            Ok((0, 0)) => Self::create_schema(&conn, path)?,
            _ => return Err(not_a_database(path)),
        }
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;",
        )?;
        Ok(Self { conn })
    }

    /// Creates the tables in an empty file. The state is read again inside
    /// an IMMEDIATE transaction, so two processes opening a new file together
    /// cannot both create them.
    fn create_schema(conn: &Connection, path: &Path) -> Result<()> {
        let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        match schema_state(&tx)? {
            (SCHEMA_VERSION, _) => {}
            (0, 0) => {
                tx.execute_batch(SCHEMA_SQL)?;
                tx.execute_batch(&format!("PRAGMA user_version = {}", SCHEMA_VERSION))?;
            }
            _ => return Err(not_a_database(path)),
        }
        tx.commit()?;
        Ok(())
    }

    /// Where the database is when `--db` was not given: `YTC_DB`, else
    /// `<app folder>/data/ytc.db` (see `db_path_for`). The file need not
    /// exist; `open` creates it there.
    pub fn default_db_path() -> PathBuf {
        let env_value = std::env::var("YTC_DB").ok();
        db_path_for(&app_root(), env_value.as_deref())
    }

    /// Status and rolling score of an existing row are owned by toggle and
    /// recalculation; re-adding a channel must not un-mute it.
    pub fn upsert_channel(&self, channel: &Channel) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO channels (id, title, url, status, rolling_slop_score, created_at, last_crawled_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                url = excluded.url;
            "#,
            params![
                channel.id,
                channel.title,
                channel.url,
                channel.status,
                channel.rolling_slop_score,
                channel.created_at,
                channel.last_crawled_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_channel(&self, channel_id: &str) -> Result<Option<Channel>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, url, status, rolling_slop_score, created_at, last_crawled_at FROM channels WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![channel_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Channel {
                id: row.get(0)?,
                title: row.get(1)?,
                url: row.get(2)?,
                status: row.get(3).unwrap_or_else(|_| "active".to_string()),
                rolling_slop_score: row.get(4)?,
                created_at: row.get(5)?,
                last_crawled_at: row.get(6)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_channel_crawled(&self, channel_id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE channels SET last_crawled_at = ?1 WHERE id = ?2",
            params![now, channel_id],
        )?;
        Ok(())
    }

    /// Sets a status chosen by the user. Like the mute toggle it sets
    /// status_manual, otherwise the next recalculation would silently undo it.
    /// `set_channel_status_auto` hands the channel back to the automatic rule.
    pub fn set_channel_status(&self, channel_id: &str, status: &str) -> Result<()> {
        if !CHANNEL_STATUSES.contains(&status) {
            anyhow::bail!(
                "Unknown channel status '{}' (expected one of: {})",
                status,
                CHANNEL_STATUSES.join(", ")
            );
        }
        let updated = self.conn.execute(
            "UPDATE channels SET status = ?1, status_manual = 1 WHERE id = ?2",
            params![status, channel_id],
        )?;
        if updated == 0 {
            anyhow::bail!("Channel {} is not tracked", channel_id);
        }
        Ok(())
    }

    /// Clears the manual flag and applies the automatic rule again, so a
    /// channel that was muted or un-muted by hand follows its scores once more.
    pub fn set_channel_status_auto(&self, channel_id: &str) -> Result<()> {
        let updated = self.conn.execute(
            "UPDATE channels SET status_manual = 0 WHERE id = ?1",
            params![channel_id],
        )?;
        if updated == 0 {
            anyhow::bail!("Channel {} is not tracked", channel_id);
        }
        if self.recalculate_channel_stats(channel_id)?.is_none() {
            // Nothing scored yet, so recalculation left the manual status in place.
            self.conn.execute(
                "UPDATE channels SET status = 'active' WHERE id = ?1",
                params![channel_id],
            )?;
        }
        Ok(())
    }

    /// True when the channel's status was set by hand and is exempt from the
    /// automatic rule. Unknown channels are not manual.
    pub fn is_channel_status_manual(&self, channel_id: &str) -> Result<bool> {
        match self.conn.query_row(
            "SELECT COALESCE(status_manual, 0) FROM channels WHERE id = ?1",
            params![channel_id],
            |r| r.get::<_, i64>(0),
        ) {
            Ok(flag) => Ok(flag == 1),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub fn toggle_channel_status(&self, channel_id: &str) -> Result<String> {
        let current: String = match self.conn.query_row(
            "SELECT status FROM channels WHERE id = ?1",
            params![channel_id],
            |r| r.get::<_, Option<String>>(0),
        ) {
            Ok(status) => status.unwrap_or_else(|| "active".to_string()),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                anyhow::bail!("Channel {} is not tracked", channel_id)
            }
            Err(e) => return Err(e.into()),
        };

        let new_status = if current == "muted" { "active" } else { "muted" };
        // status_manual makes the user's choice stick against recalculation.
        let updated = self.conn.execute(
            "UPDATE channels SET status = ?1, status_manual = 1 WHERE id = ?2",
            params![new_status, channel_id],
        )?;
        if updated == 0 {
            anyhow::bail!("Channel {} is not tracked", channel_id);
        }
        Ok(new_status.to_string())
    }

    pub fn delete_channel(&self, channel_id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM channels WHERE id = ?1", params![channel_id])?;
        Ok(())
    }

    /// Recomputes the rolling slop average of a channel from its last five
    /// rows that count (see `SCORED_SQL`) and applies the automatic status
    /// rule unless the status was set by hand. None = no video counts (yet, or
    /// any more): the stored average is then cleared, because an average over
    /// zero videos is not a score, and an automatic status goes back to
    /// 'active'. A status set by hand stays.
    pub fn recalculate_channel_stats(&self, channel_id: &str) -> Result<Option<f64>> {
        // The condition is part of the query, not applied after LIMIT: rows
        // that do not count (too short to judge, disqualified, off-topic,
        // heuristic rows) cannot push scored videos out of the window.
        let mut stmt = self.conn.prepare(&format!(
            r#"
            SELECT i.slop_score
            FROM insights i
            JOIN videos v ON i.video_id = v.id
            WHERE v.channel_id = ?1 AND {SCORED_SQL}
            ORDER BY v.published_at DESC
            LIMIT 5
            "#,
        ))?;
        // Heuristic rows are word-count estimates made without the LLM; they
        // must not be able to mute or promote a channel.
        let scores: Vec<u32> = stmt.query_map(params![channel_id], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if scores.is_empty() {
            // The videos that made up the stored average may all have left it
            // (too short to judge, topic-rejected, disqualified): the
            // listings must not go on showing a score backed by no video.
            // The condition skips no-op writes, as below.
            self.conn.execute(
                r#"
                UPDATE channels
                SET rolling_slop_score = NULL,
                    status = CASE WHEN COALESCE(status_manual, 0) = 1 THEN status ELSE 'active' END
                WHERE id = ?1
                  AND (rolling_slop_score IS NOT NULL
                       OR (COALESCE(status_manual, 0) = 0 AND COALESCE(status, 'active') != 'active'))
                "#,
                params![channel_id],
            )?;
            return Ok(None);
        }

        let avg = scores.iter().sum::<u32>() as f64 / scores.len() as f64;
        let avg_rounded = (avg * 10.0).round() / 10.0;

        let new_status = auto_channel_status(scores.len(), avg_rounded);

        // Read before the update so an automatic change can be announced.
        // None = the channel is not tracked or its status was set by hand.
        let before: Option<(String, String)> = match self.conn.query_row(
            "SELECT title, COALESCE(status, 'active') FROM channels WHERE id = ?1 AND COALESCE(status_manual, 0) = 0",
            params![channel_id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
        ) {
            Ok(row) => Some(row),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(e.into()),
        };

        // A manual mute/unmute keeps its status; only the score follows the data.
        // The last condition skips no-op writes, since channel listings call this too.
        self.conn.execute(
            r#"
            UPDATE channels
            SET rolling_slop_score = ?1,
                status = CASE WHEN COALESCE(status_manual, 0) = 1 THEN status ELSE ?2 END
            WHERE id = ?3
              AND (rolling_slop_score IS NOT ?1
                   OR (COALESCE(status_manual, 0) = 0 AND status IS NOT ?2))
            "#,
            params![avg_rounded, new_status, channel_id],
        )?;

        if let Some((title, status_before)) = before {
            // stderr: stdout may carry the table or JSON of the running command.
            if let Some(notice) = auto_status_change_notice(
                channel_id,
                &title,
                &status_before,
                new_status,
                avg_rounded,
                scores.len(),
            ) {
                eprintln!("{}", notice);
            }
        }

        Ok(Some(avg_rounded))
    }

    pub fn list_channels(&self) -> Result<Vec<Channel>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, url, status, rolling_slop_score, created_at, last_crawled_at FROM channels ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Channel {
                id: row.get(0)?,
                title: row.get(1)?,
                url: row.get(2)?,
                status: row.get(3).unwrap_or_else(|_| "active".to_string()),
                rolling_slop_score: row.get(4)?,
                created_at: row.get(5)?,
                last_crawled_at: row.get(6)?,
            })
        })?;

        let mut channels = Vec::new();
        for r in rows {
            channels.push(r?);
        }
        Ok(channels)
    }

    pub fn list_channels_with_health(&self) -> Result<Vec<ChannelHealth>> {
        let channels = self.list_channels()?;
        let mut health_list = Vec::new();

        for mut c in channels {
            // Refresh the rolling score, then show what is stored: the crawler
            // filters on the stored status, so the listing must not invent its own.
            match self.recalculate_channel_stats(&c.id) {
                // None too: the stored average was cleared, and with it an
                // automatic status may have gone back to 'active'.
                Ok(avg) => {
                    c.rolling_slop_score = avg;
                    match self.conn.query_row(
                        "SELECT status FROM channels WHERE id = ?1",
                        params![c.id],
                        |r| r.get::<_, Option<String>>(0),
                    ) {
                        Ok(Some(status)) => c.status = status,
                        Ok(None) => {}
                        Err(e) => eprintln!("Warning: could not read status of channel {}: {}", c.id, e),
                    }
                }
                Err(e) => eprintln!("Warning: channel stats recalculation failed for {}: {}", c.id, e),
            }

            // One pass, first match wins, so the five counts are disjoint and
            // add up to the channel's insight rows. The order is the order of
            // precedence: the owner's decision, then topic, then information.
            let (kept_count, excluded_count, off_topic_count, insufficient_count, below_cutoff_count): (usize, usize, usize, usize, usize) =
                match self.conn.query_row(
                    &format!(
                        r#"
                        SELECT COALESCE(SUM(reason = 'kept'), 0), COALESCE(SUM(reason = 'excluded'), 0),
                               COALESCE(SUM(reason = 'off_topic'), 0), COALESCE(SUM(reason = 'insufficient'), 0),
                               COALESCE(SUM(reason = 'below_cutoff'), 0)
                        FROM (
                            SELECT CASE
                                WHEN {KEPT_SQL} THEN 'kept'
                                WHEN COALESCE(i.excluded_manual, 0) != 0 THEN 'excluded'
                                WHEN i.topic_verdict IS NOT NULL AND i.topic_verdict != 'ok' THEN 'off_topic'
                                WHEN i.info_status IS NOT NULL AND i.info_status != 'sufficient' THEN 'insufficient'
                                ELSE 'below_cutoff'
                            END AS reason
                            FROM videos v JOIN insights i ON v.id = i.video_id
                            WHERE v.channel_id = ?1
                        )
                        "#
                    ),
                    params![c.id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                ) {
                    Ok(counts) => counts,
                    Err(e) => {
                        eprintln!("Warning: could not count the videos of channel {}: {}", c.id, e);
                        (0, 0, 0, 0, 0)
                    }
                };

            health_list.push(ChannelHealth {
                id: c.id,
                title: c.title,
                status: c.status,
                total_evaluated: kept_count + excluded_count + off_topic_count + insufficient_count + below_cutoff_count,
                kept_count,
                avg_slop_score: c.rolling_slop_score,
                last_crawled_at: c.last_crawled_at,
                below_cutoff_count,
                off_topic_count,
                insufficient_count,
                excluded_count,
            });
        }

        Ok(health_list)
    }

    pub fn upsert_video(&self, video: &Video) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO videos (id, channel_id, title, url, published_at, status, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(id) DO UPDATE SET
                channel_id = excluded.channel_id,
                title = excluded.title,
                published_at = excluded.published_at,
                status = CASE
                    WHEN videos.status IN ('processed', 'failed') THEN videos.status
                    ELSE excluded.status
                END;
            "#,
            params![
                video.id,
                video.channel_id,
                video.title,
                video.url,
                video.published_at,
                video.status,
                video.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_video(&self, video_id: &str) -> Result<Option<Video>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, channel_id, title, url, published_at, status, created_at FROM videos WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![video_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Video {
                id: row.get(0)?,
                channel_id: row.get(1)?,
                title: row.get(2)?,
                url: row.get(3)?,
                published_at: row.get(4)?,
                status: row.get(5)?,
                created_at: row.get(6)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_video_status(&self, video_id: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE videos SET status = ?1 WHERE id = ?2",
            params![status, video_id],
        )?;
        Ok(())
    }

    /// Marks a video as failed and counts the failure. Returns how often the
    /// video has failed so far, so a caller can give up after N attempts.
    pub fn mark_video_failed(&self, video_id: &str) -> Result<u32> {
        match self.conn.query_row(
            "UPDATE videos SET status = 'failed', fail_count = COALESCE(fail_count, 0) + 1
             WHERE id = ?1 RETURNING fail_count",
            params![video_id],
            |r| r.get::<_, u32>(0),
        ) {
            Ok(count) => Ok(count),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                anyhow::bail!("Video {} is not in the database", video_id)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Number of recorded processing failures; 0 for an unknown video.
    pub fn video_fail_count(&self, video_id: &str) -> Result<u32> {
        match self.conn.query_row(
            "SELECT COALESCE(fail_count, 0) FROM videos WHERE id = ?1",
            params![video_id],
            |r| r.get::<_, u32>(0),
        ) {
            Ok(count) => Ok(count),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(0),
            Err(e) => Err(e.into()),
        }
    }

    /// The analysed videos for a bulk re-analysis, the one analysed longest
    /// ago first (then by id): a run in portions therefore always continues
    /// with the videos it has not reached yet. With `prefix` only the videos
    /// whose analyzer type starts with it ("heuristic", "llm:").
    pub fn list_analysed_video_ids(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        // substr comparison instead of LIKE: '%' and '_' in the prefix stay literal.
        let mut stmt = self.conn.prepare(
            "SELECT video_id FROM insights
             WHERE ?1 IS NULL OR substr(analyzer_type, 1, length(?1)) = ?1
             ORDER BY updated_at, video_id",
        )?;
        let rows = stmt.query_map(params![prefix], |r| r.get::<_, String>(0))?;
        let mut ids = Vec::new();
        for r in rows {
            ids.push(r?);
        }
        Ok(ids)
    }

    pub fn save_transcript(&self, transcript: &CleanedTranscript) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            r#"
            INSERT INTO transcripts (video_id, raw_char_count, cleaned_char_count, stripped_ad_seconds, stripped_segments_count, text, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(video_id) DO UPDATE SET
                raw_char_count = excluded.raw_char_count,
                cleaned_char_count = excluded.cleaned_char_count,
                stripped_ad_seconds = excluded.stripped_ad_seconds,
                stripped_segments_count = excluded.stripped_segments_count,
                text = excluded.text;
            "#,
            params![
                transcript.video_id,
                transcript.raw_char_count as i64,
                transcript.cleaned_char_count as i64,
                transcript.stripped_ad_seconds,
                transcript.stripped_segments_count as i64,
                transcript.text,
                now,
            ],
        )?;
        Ok(())
    }

    pub fn get_transcript(&self, video_id: &str) -> Result<Option<CleanedTranscript>> {
        let mut stmt = self.conn.prepare(
            "SELECT video_id, raw_char_count, cleaned_char_count, stripped_ad_seconds, stripped_segments_count, text FROM transcripts WHERE video_id = ?1",
        )?;
        let mut rows = stmt.query(params![video_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(CleanedTranscript {
                video_id: row.get(0)?,
                raw_char_count: row.get::<_, i64>(1)? as usize,
                cleaned_char_count: row.get::<_, i64>(2)? as usize,
                stripped_ad_seconds: row.get(3)?,
                stripped_segments_count: row.get::<_, i64>(4)? as usize,
                text: row.get(5)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Records how long the video is, where the last transcript cue ends and
    /// which tier produced the transcript ("builtin", "subtitles", "whisper").
    /// Call after `save_transcript`; all three values are replaced, None
    /// stores "not recorded".
    pub fn set_transcript_coverage(
        &self,
        video_id: &str,
        video_duration_sec: Option<f64>,
        covered_until_sec: Option<f64>,
        source: Option<&str>,
    ) -> Result<()> {
        if let Some(source) = source {
            check_one_of("transcript source", source, &TRANSCRIPT_SOURCES)?;
        }
        // NaN and infinity are not durations; SQLite would store NaN as NULL anyway.
        let finite = |v: Option<f64>| v.filter(|x| x.is_finite() && *x >= 0.0);
        let updated = self.conn.execute(
            "UPDATE transcripts SET video_duration_sec = ?1, covered_until_sec = ?2, source = ?3 WHERE video_id = ?4",
            params![finite(video_duration_sec), finite(covered_until_sec), source, video_id],
        )?;
        if updated == 0 {
            anyhow::bail!("No transcript stored for video {}", video_id);
        }
        Ok(())
    }

    /// Coverage facts of one stored transcript; None when no transcript exists.
    pub fn get_transcript_coverage(&self, video_id: &str) -> Result<Option<TranscriptCoverage>> {
        let row = self
            .conn
            .query_row(
                "SELECT video_duration_sec, covered_until_sec, source FROM transcripts WHERE video_id = ?1",
                params![video_id],
                |r| {
                    Ok(TranscriptCoverage {
                        video_duration_sec: r.get(0)?,
                        covered_until_sec: r.get(1)?,
                        source: non_blank(r.get(2)?),
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Coverage facts of every stored transcript, keyed by video id. The
    /// transcript texts are not read.
    pub fn list_transcript_coverage(&self) -> Result<HashMap<String, TranscriptCoverage>> {
        let mut stmt = self
            .conn
            .prepare("SELECT video_id, video_duration_sec, covered_until_sec, source FROM transcripts")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                TranscriptCoverage {
                    video_duration_sec: r.get(1)?,
                    covered_until_sec: r.get(2)?,
                    source: non_blank(r.get(3)?),
                },
            ))
        })?;
        let mut map = HashMap::new();
        for r in rows {
            let (video_id, coverage) = r?;
            map.insert(video_id, coverage);
        }
        Ok(map)
    }

    /// Stores an analysis without topic verdict, information status or
    /// read figures (they are reset to NULL). Prefer
    /// `save_insight_judged`, which writes them in the same transaction.
    #[cfg(test)]
    pub fn save_insight(&self, insight: &InsightResult) -> Result<()> {
        self.save_insight_judged(insight, &InsightJudgment::default())
    }

    /// Stores an analysis together with its topic verdict, information status
    /// and coverage, in one transaction, so that neither the kept rule nor the
    /// channel average ever sees the row without them. A manual
    /// disqualification of the video is left untouched: a re-analysis does
    /// not overrule the owner. The same holds for a category set by hand
    /// (`set_video_category`): the stored category stays and
    /// `insight.category` is not written.
    pub fn save_insight_judged(&self, insight: &InsightResult, judgment: &InsightJudgment) -> Result<()> {
        if let Some(verdict) = judgment.topic_verdict.as_deref() {
            check_one_of("topic verdict", verdict, &TOPIC_VERDICTS)?;
        }
        if let Some(status) = judgment.info_status.as_deref() {
            check_one_of("info status", status, &INFO_STATUSES)?;
        }
        let now = Utc::now().to_rfc3339();
        let key_insights_json = serde_json::to_string(&insight.key_insights)?;
        let takeaways_json = serde_json::to_string(&insight.takeaways)?;
        let quotes_json = serde_json::to_string(&insight.quotes)?;
        let topics_json = serde_json::to_string(&insight.topics)?;
        let mentioned_creators_json = serde_json::to_string(&insight.mentioned_creators)?;

        // Insight row and topic links change together: the topic readers use
        // video_topics, so the two must never disagree.
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        tx.execute(
            r#"
            INSERT INTO insights (
                video_id, slop_score, is_worth_keeping, quality_rating, summary,
                key_insights_json, takeaways_json, quotes_json, topics_json, mentioned_creators_json,
                analyzer_type, category, created_at,
                transcript_chars, topic_verdict, topic_reason, info_status, info_reason,
                sections_total, updated_at
            )
            -- The category is stored only while the user has it: a name that
            -- is not in `categories` becomes NULL.
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, (SELECT name FROM categories WHERE name = ?12),
                    ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?13)
            ON CONFLICT(video_id) DO UPDATE SET
                -- A new analysis replaces verdicts and read figures of the
                -- stored one: what the caller did not pass becomes NULL.
                -- excluded_manual, excluded_reason and excluded_at are absent
                -- on purpose: they belong to the owner, not to the analysis.
                transcript_chars = excluded.transcript_chars,
                topic_verdict = excluded.topic_verdict,
                topic_reason = excluded.topic_reason,
                info_status = excluded.info_status,
                info_reason = excluded.info_reason,
                sections_total = excluded.sections_total,
                updated_at = excluded.updated_at,
                slop_score = excluded.slop_score,
                is_worth_keeping = excluded.is_worth_keeping,
                quality_rating = excluded.quality_rating,
                summary = excluded.summary,
                key_insights_json = excluded.key_insights_json,
                takeaways_json = excluded.takeaways_json,
                quotes_json = excluded.quotes_json,
                topics_json = excluded.topics_json,
                mentioned_creators_json = excluded.mentioned_creators_json,
                analyzer_type = excluded.analyzer_type,
                -- category_manual is absent like the excluded_* columns. While
                -- it is set, the category is the owner's and stays.
                category = CASE WHEN insights.category_manual != 0 THEN insights.category
                                ELSE excluded.category END,
                -- An answer of 'ytc categorize' was about the insight this
                -- analysis replaces.
                category_asked = 0;
            "#,
            params![
                insight.video_id,
                insight.slop_score,
                if insight.is_worth_keeping { 1 } else { 0 },
                insight.quality_rating,
                insight.summary,
                key_insights_json,
                takeaways_json,
                quotes_json,
                topics_json,
                mentioned_creators_json,
                insight.analyzer_type,
                insight.category,
                now,
                judgment.transcript_chars.and_then(|n| i64::try_from(n).ok()),
                judgment.topic_verdict,
                clean_text(judgment.topic_reason.as_deref()),
                judgment.info_status,
                clean_text(judgment.info_reason.as_deref()),
                judgment.sections_total,
            ],
        )?;

        // Replace the video's links: after a re-analysis it is listed under
        // the topics it has now and under no others.
        link_topics(&tx, &insight.video_id, &topic_names(&topics_json), &now)?;
        tx.commit()?;

        // Recalculate channel stats
        if let Ok(Some(video)) = self.get_video(&insight.video_id) {
            if let Err(e) = self.recalculate_channel_stats(&video.channel_id) {
                eprintln!("Warning: channel stats recalculation failed for {}: {}", video.channel_id, e);
            }
        }

        Ok(())
    }

    pub fn get_insight(&self, video_id: &str) -> Result<Option<InsightResult>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT video_id, slop_score, is_worth_keeping, quality_rating, summary,
                   key_insights_json, takeaways_json, quotes_json, topics_json, mentioned_creators_json,
                   analyzer_type, category
            FROM insights WHERE video_id = ?1
            "#,
        )?;
        let mut rows = stmt.query(params![video_id])?;
        if let Some(row) = rows.next()? {
            let key_insights_str: String = row.get(5)?;
            let takeaways_str: String = row.get(6)?;
            let quotes_str: String = row.get(7)?;
            let topics_str: String = row.get(8).unwrap_or_else(|_| "[]".to_string());
            let mentioned_str: String = row.get(9).unwrap_or_else(|_| "[]".to_string());

            let key_insights: Vec<String> = serde_json::from_str(&key_insights_str).unwrap_or_default();
            let takeaways: Vec<String> = serde_json::from_str(&takeaways_str).unwrap_or_default();
            let quotes: Vec<String> = serde_json::from_str(&quotes_str).unwrap_or_default();
            let topics: Vec<String> = serde_json::from_str(&topics_str).unwrap_or_default();
            let mentioned_creators: Vec<String> = serde_json::from_str(&mentioned_str).unwrap_or_default();

            Ok(Some(InsightResult {
                video_id: row.get(0)?,
                slop_score: row.get(1)?,
                is_worth_keeping: row.get::<_, i32>(2)? == 1,
                quality_rating: row.get(3)?,
                summary: row.get(4)?,
                key_insights,
                takeaways,
                quotes,
                topics,
                mentioned_creators,
                analyzer_type: row.get(10)?,
                category: non_blank(row.get(11)?),
            }))
        } else {
            Ok(None)
        }
    }

    /// Recalculates the channel of a video after one of its insight columns
    /// changed what counts. A failure is a warning, as in `save_insight`.
    fn refresh_channel_of(&self, video_id: &str) {
        if let Ok(Some(video)) = self.get_video(video_id) {
            if let Err(e) = self.recalculate_channel_stats(&video.channel_id) {
                eprintln!("Warning: channel stats recalculation failed for {}: {}", video.channel_id, e);
            }
        }
    }

    /// Records the topic verdict ("ok", "fringe", "nonsense", "politics") of
    /// a stored insight. Anything but "ok" takes the video out of the library
    /// and out of the channel average, whatever its score.
    #[cfg(test)]
    pub fn set_insight_verdict(&self, video_id: &str, topic_verdict: &str, topic_reason: Option<&str>) -> Result<()> {
        check_one_of("topic verdict", topic_verdict, &TOPIC_VERDICTS)?;
        let updated = self.conn.execute(
            "UPDATE insights SET topic_verdict = ?1, topic_reason = ?2 WHERE video_id = ?3",
            params![topic_verdict, clean_text(topic_reason), video_id],
        )?;
        if updated == 0 {
            anyhow::bail!("No insight stored for video {}", video_id);
        }
        self.refresh_channel_of(video_id);
        Ok(())
    }

    /// Records whether the whole transcript held enough to judge
    /// ("sufficient" / "insufficient"). An insufficient video is not kept and
    /// does not count in the channel average: it is not slop, just too little.
    #[cfg(test)]
    pub fn set_insight_info_status(&self, video_id: &str, info_status: &str, info_reason: Option<&str>) -> Result<()> {
        check_one_of("info status", info_status, &INFO_STATUSES)?;
        let updated = self.conn.execute(
            "UPDATE insights SET info_status = ?1, info_reason = ?2 WHERE video_id = ?3",
            params![info_status, clean_text(info_reason), video_id],
        )?;
        if updated == 0 {
            anyhow::bail!("No insight stored for video {}", video_id);
        }
        self.refresh_channel_of(video_id);
        Ok(())
    }

    /// Disqualifies a video by the owner's decision. It leaves the library
    /// and the channel average at once and stays out after any re-analysis,
    /// until `include_video`. Fails when the video has no insight: there is
    /// then nothing in the library to remove.
    ///
    /// On a video that is already disqualified this only replaces the reason,
    /// and only when a new (non-blank) one is given: repeating the command
    /// without a reason must not erase the owner's note, and the date stays
    /// the day of the decision.
    pub fn exclude_video(&self, video_id: &str, reason: Option<&str>) -> Result<()> {
        let updated = self.conn.execute(
            r#"
            UPDATE insights
            SET excluded_reason = CASE WHEN COALESCE(excluded_manual, 0) != 0
                                       THEN COALESCE(?1, excluded_reason) ELSE ?1 END,
                excluded_at = CASE WHEN COALESCE(excluded_manual, 0) != 0
                                   THEN COALESCE(excluded_at, ?2) ELSE ?2 END,
                excluded_manual = 1
            WHERE video_id = ?3
            "#,
            params![clean_text(reason), Utc::now().to_rfc3339(), video_id],
        )?;
        if updated == 0 {
            anyhow::bail!("Video {} has no analysis, so there is nothing to disqualify", video_id);
        }
        self.refresh_channel_of(video_id);
        Ok(())
    }

    /// Undoes `exclude_video`: the model's judgment decides again. No error
    /// when the video was not disqualified; an error when it has no insight.
    pub fn include_video(&self, video_id: &str) -> Result<()> {
        let updated = self.conn.execute(
            "UPDATE insights SET excluded_manual = 0, excluded_reason = NULL, excluded_at = NULL WHERE video_id = ?1",
            params![video_id],
        )?;
        if updated == 0 {
            anyhow::bail!("Video {} has no analysis", video_id);
        }
        self.refresh_channel_of(video_id);
        Ok(())
    }

    /// The disqualified videos, most recently disqualified first.
    pub fn list_excluded(&self) -> Result<Vec<ExcludedVideo>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT i.video_id, COALESCE(v.title, i.video_id), COALESCE(v.channel_id, ''), c.title,
                   i.excluded_reason, i.excluded_at, i.slop_score
            FROM insights i
            LEFT JOIN videos v ON v.id = i.video_id
            LEFT JOIN channels c ON c.id = v.channel_id
            WHERE COALESCE(i.excluded_manual, 0) != 0
            ORDER BY i.excluded_at DESC, i.video_id ASC
            "#,
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ExcludedVideo {
                video_id: r.get(0)?,
                title: r.get(1)?,
                channel_id: r.get(2)?,
                channel_title: r.get(3)?,
                reason: non_blank(r.get(4)?),
                excluded_at: non_blank(r.get(5)?),
                slop_score: r.get(6)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    /// Scoring, verdict and coverage facts of one insight, including the
    /// effective `kept` decision; None when no insight exists.
    pub fn get_insight_meta(&self, video_id: &str) -> Result<Option<InsightMeta>> {
        let sql = format!("{} WHERE i.video_id = ?1", INSIGHT_META_SELECT);
        match self.conn.query_row(&sql, params![video_id], insight_meta_from_row) {
            Ok((_, meta)) => Ok(Some(meta)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Scoring and coverage facts of every insight, keyed by video id. One
    /// query for listings, instead of one `get_insight_meta` per row.
    pub fn list_insight_meta(&self) -> Result<HashMap<String, InsightMeta>> {
        let mut stmt = self.conn.prepare(INSIGHT_META_SELECT)?;
        let rows = stmt.query_map([], insight_meta_from_row)?;
        let mut map = HashMap::new();
        for r in rows {
            let (video_id, meta) = r?;
            map.insert(video_id, meta);
        }
        Ok(map)
    }

    pub fn list_library(&self, kept_only: bool, limit: usize) -> Result<Vec<(Video, Option<InsightResult>)>> {
        let query = if kept_only {
            format!(
                r#"
            SELECT v.id, v.channel_id, v.title, v.url, v.published_at, v.status, v.created_at,
                   i.slop_score, i.is_worth_keeping, i.quality_rating, i.summary, i.key_insights_json, i.takeaways_json, i.quotes_json, i.topics_json, i.mentioned_creators_json, i.analyzer_type,
                   i.category
            FROM videos v
            JOIN insights i ON v.id = i.video_id
            WHERE {KEPT_SQL}
            ORDER BY v.published_at DESC
            LIMIT ?1
            "#
            )
        } else {
            // `is_worth_keeping` in these rows is the stored quality decision
            // only; `InsightMeta::kept` has the effective one.
            r#"
            SELECT v.id, v.channel_id, v.title, v.url, v.published_at, v.status, v.created_at,
                   i.slop_score, i.is_worth_keeping, i.quality_rating, i.summary, i.key_insights_json, i.takeaways_json, i.quotes_json, i.topics_json, i.mentioned_creators_json, i.analyzer_type,
                   i.category
            FROM videos v
            LEFT JOIN insights i ON v.id = i.video_id
            ORDER BY v.published_at DESC
            LIMIT ?1
            "#
            .to_string()
        };

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let video = Video {
                id: row.get(0)?,
                channel_id: row.get(1)?,
                title: row.get(2)?,
                url: row.get(3)?,
                published_at: row.get(4)?,
                status: row.get(5)?,
                created_at: row.get(6)?,
            };

            let slop_score: Option<u32> = row.get(7)?;
            let insight = if let Some(score) = slop_score {
                let is_worth_keeping: i32 = row.get(8)?;
                let quality_rating: String = row.get(9)?;
                let summary: String = row.get(10)?;
                let key_insights_str: String = row.get(11)?;
                let takeaways_str: String = row.get(12)?;
                let quotes_str: String = row.get(13)?;
                let topics_str: String = row.get(14).unwrap_or_else(|_| "[]".to_string());
                let mentioned_str: String = row.get(15).unwrap_or_else(|_| "[]".to_string());
                let analyzer_type: String = row.get(16)?;
                let category: Option<String> = non_blank(row.get(17)?);

                Some(InsightResult {
                    video_id: video.id.clone(),
                    slop_score: score,
                    is_worth_keeping: is_worth_keeping == 1,
                    quality_rating,
                    summary,
                    key_insights: serde_json::from_str(&key_insights_str).unwrap_or_default(),
                    takeaways: serde_json::from_str(&takeaways_str).unwrap_or_default(),
                    quotes: serde_json::from_str(&quotes_str).unwrap_or_default(),
                    topics: serde_json::from_str(&topics_str).unwrap_or_default(),
                    mentioned_creators: serde_json::from_str(&mentioned_str).unwrap_or_default(),
                    analyzer_type,
                    category,
                })
            } else {
                None
            };

            Ok((video, insight))
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    /// Topics of kept videos with their video count, read from video_topics.
    /// `save_insight` rewrites a video's links together with its topics_json,
    /// so a count cannot point at videos that do not carry the topic.
    pub fn list_all_topics(&self) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(&format!(
            r#"
            SELECT vt.topic_name AS name, COUNT(DISTINCT vt.video_id) AS cnt
            FROM video_topics vt
            JOIN insights i ON i.video_id = vt.video_id
            WHERE {KEPT_SQL}
            GROUP BY vt.topic_name
            ORDER BY cnt DESC, name ASC
            "#,
        ))?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        let mut topics = Vec::new();
        for r in rows {
            topics.push(r?);
        }
        Ok(topics)
    }

    pub fn get_topic_videos(&self, topic: &str) -> Result<Vec<(Video, InsightResult)>> {
        let mut stmt = self.conn.prepare(&format!(
            r#"
            SELECT v.id, v.channel_id, v.title, v.url, v.published_at, v.status, v.created_at,
                   i.slop_score, i.is_worth_keeping, i.quality_rating, i.summary, i.key_insights_json, i.takeaways_json, i.quotes_json, i.topics_json, i.mentioned_creators_json, i.analyzer_type,
                   i.category
            FROM videos v
            JOIN insights i ON v.id = i.video_id
            JOIN video_topics vt ON vt.video_id = v.id AND vt.topic_name = ?1
            WHERE {KEPT_SQL}
            ORDER BY i.slop_score ASC, v.published_at DESC
            "#,
        ))?;

        let rows = stmt.query_map(params![topic], |row| {
            let video = Video {
                id: row.get(0)?,
                channel_id: row.get(1)?,
                title: row.get(2)?,
                url: row.get(3)?,
                published_at: row.get(4)?,
                status: row.get(5)?,
                created_at: row.get(6)?,
            };

            let key_insights_str: String = row.get(11)?;
            let takeaways_str: String = row.get(12)?;
            let quotes_str: String = row.get(13)?;
            let topics_str: String = row.get(14).unwrap_or_else(|_| "[]".to_string());
            let mentioned_str: String = row.get(15).unwrap_or_else(|_| "[]".to_string());

            let insight = InsightResult {
                video_id: video.id.clone(),
                slop_score: row.get(7)?,
                is_worth_keeping: row.get::<_, i32>(8)? == 1,
                quality_rating: row.get(9)?,
                summary: row.get(10)?,
                key_insights: serde_json::from_str(&key_insights_str).unwrap_or_default(),
                takeaways: serde_json::from_str(&takeaways_str).unwrap_or_default(),
                quotes: serde_json::from_str(&quotes_str).unwrap_or_default(),
                topics: serde_json::from_str(&topics_str).unwrap_or_default(),
                mentioned_creators: serde_json::from_str(&mentioned_str).unwrap_or_default(),
                analyzer_type: row.get(16)?,
                category: non_blank(row.get(17)?),
            };

            Ok((video, insight))
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    /// Every video in the library (the kept rule, `KEPT_SQL`), newest first.
    pub fn get_all_kept_videos_with_details(&self) -> Result<Vec<(Video, Channel, InsightResult)>> {
        let mut stmt = self.conn.prepare(&format!(
            r#"
            SELECT v.id, v.channel_id, v.title, v.url, v.published_at, v.status, v.created_at,
                   COALESCE(c.title, 'Independent Analysis'), COALESCE(c.url, ''), COALESCE(c.status, 'active'), c.rolling_slop_score, COALESCE(c.created_at, v.created_at), c.last_crawled_at,
                   i.slop_score, i.is_worth_keeping, i.quality_rating, i.summary, i.key_insights_json, i.takeaways_json, i.quotes_json, i.topics_json, i.mentioned_creators_json, i.analyzer_type,
                   i.category
            FROM videos v
            LEFT JOIN channels c ON v.channel_id = c.id
            JOIN insights i ON v.id = i.video_id
            WHERE {KEPT_SQL}
            ORDER BY v.published_at DESC
            "#,
        ))?;

        let rows = stmt.query_map([], |row| {
            let video = Video {
                id: row.get(0)?,
                channel_id: row.get(1)?,
                title: row.get(2)?,
                url: row.get(3)?,
                published_at: row.get(4)?,
                status: row.get(5)?,
                created_at: row.get(6)?,
            };

            let channel = Channel {
                id: video.channel_id.clone(),
                title: row.get(7)?,
                url: row.get(8)?,
                status: row.get(9).unwrap_or_else(|_| "active".to_string()),
                rolling_slop_score: row.get(10)?,
                created_at: row.get(11)?,
                last_crawled_at: row.get(12)?,
            };

            let key_insights_str: String = row.get(17)?;
            let takeaways_str: String = row.get(18)?;
            let quotes_str: String = row.get(19)?;
            let topics_str: String = row.get(20).unwrap_or_else(|_| "[]".to_string());
            let mentioned_str: String = row.get(21).unwrap_or_else(|_| "[]".to_string());

            let insight = InsightResult {
                video_id: video.id.clone(),
                slop_score: row.get(13)?,
                is_worth_keeping: row.get::<_, i32>(14)? == 1,
                quality_rating: row.get(15)?,
                summary: row.get(16)?,
                key_insights: serde_json::from_str(&key_insights_str).unwrap_or_default(),
                takeaways: serde_json::from_str(&takeaways_str).unwrap_or_default(),
                quotes: serde_json::from_str(&quotes_str).unwrap_or_default(),
                topics: serde_json::from_str(&topics_str).unwrap_or_default(),
                mentioned_creators: serde_json::from_str(&mentioned_str).unwrap_or_default(),
                analyzer_type: row.get(22)?,
                category: non_blank(row.get(23)?),
            };

            Ok((video, channel, insight))
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn get_all_mentioned_creators(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(&format!(
            r#"
            SELECT v.title, i.mentioned_creators_json
            FROM insights i
            JOIN videos v ON i.video_id = v.id
            WHERE {KEPT_SQL} AND i.mentioned_creators_json != '[]'
            "#,
        ))?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        let mut mentions = Vec::new();
        for r in rows {
            let (title, json_str): (String, String) = r?;
            let list: Vec<String> = serde_json::from_str(&json_str).unwrap_or_default();
            for m in list {
                let trimmed = m.trim().to_string();
                if !trimmed.is_empty() {
                    mentions.push((trimmed, title.clone()));
                }
            }
        }
        Ok(mentions)
    }

    /// At most 20 hits. Kept videos come first, then judged ones that were not
    /// kept, and last the rows whose score column is no measurement ("not
    /// enough information" rows with their placeholder 0):
    /// sorted by the bare score those would rank above the best kept video
    /// and push real hits out of the limit. Within a group: lowest slop
    /// score first, then newest.
    pub fn search(&self, query: &str) -> Result<Vec<SearchHit>> {
        let pattern = format!("%{}%", query);
        // Column 3 is the kept rule, not the stored flag: a hit must not be
        // shown as kept when it was disqualified, topic-rejected or too short.
        let mut stmt = self.conn.prepare(&format!(
            r#"
            SELECT v.id, v.title, i.slop_score, ({KEPT_SQL}), i.quality_rating, i.summary, i.key_insights_json, i.topics_json, i.category,
                   i.topic_verdict, i.info_status, i.excluded_manual, i.info_reason
            FROM videos v
            JOIN insights i ON v.id = i.video_id
            LEFT JOIN transcripts t ON v.id = t.video_id
            WHERE v.title LIKE ?1 
               OR i.summary LIKE ?1 
               OR i.key_insights_json LIKE ?1 
               OR i.takeaways_json LIKE ?1
               OR i.topics_json LIKE ?1
               OR i.category LIKE ?1
               OR t.text LIKE ?1
            ORDER BY ({KEPT_SQL}) DESC,
                     (COALESCE(i.info_status, '') = 'insufficient') ASC,
                     i.slop_score ASC, v.published_at DESC
            LIMIT 20
            "#,
        ))?;

        let rows = stmt.query_map(params![pattern], |row| {
            let key_insights_str: String = row.get(6)?;
            let topics_str: String = row.get(7).unwrap_or_else(|_| "[]".to_string());

            let key_insights: Vec<String> = serde_json::from_str(&key_insights_str).unwrap_or_default();
            let topics: Vec<String> = serde_json::from_str(&topics_str).unwrap_or_default();

            Ok(SearchHit {
                video_id: row.get(0)?,
                title: row.get(1)?,
                slop_score: row.get(2)?,
                is_worth_keeping: row.get::<_, i32>(3)? == 1,
                quality_rating: row.get(4)?,
                summary: row.get(5)?,
                key_insights,
                topics,
                category: non_blank(row.get(8)?),
                topic_verdict: non_blank(row.get(9)?),
                info_status: non_blank(row.get(10)?),
                info_reason: non_blank(row.get(12)?),
                excluded_manual: row.get::<_, i64>(11)? != 0,
            })
        })?;

        let mut hits = Vec::new();
        for r in rows {
            hits.push(r?);
        }
        Ok(hits)
    }
}

// ---------------------------------------------------------------------------
// User-defined categories
// ---------------------------------------------------------------------------

/// A category name as it is stored: whitespace runs collapsed to one space,
/// 1 to `CATEGORY_NAME_MAX_CHARS` characters, no control characters. "null"
/// is refused because that is how the model answers "no category".
fn clean_category_name(name: &str) -> Result<String> {
    let cleaned = name.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty() {
        anyhow::bail!("A category needs a name");
    }
    if cleaned.chars().count() > CATEGORY_NAME_MAX_CHARS {
        anyhow::bail!("A category name may have at most {} characters", CATEGORY_NAME_MAX_CHARS);
    }
    if cleaned.chars().any(char::is_control) {
        anyhow::bail!("A category name must not contain control characters");
    }
    if cleaned.eq_ignore_ascii_case("null") {
        anyhow::bail!("'{}' cannot be used as a category name", cleaned);
    }
    Ok(cleaned)
}

/// A category description as it is stored: one line, whitespace runs
/// collapsed, at most `CATEGORY_DESCRIPTION_MAX_CHARS` characters. Blank is None.
fn clean_category_description(description: Option<&str>) -> Result<Option<String>> {
    let cleaned = description.unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty() {
        return Ok(None);
    }
    if cleaned.chars().count() > CATEGORY_DESCRIPTION_MAX_CHARS {
        anyhow::bail!(
            "A category description may have at most {} characters",
            CATEGORY_DESCRIPTION_MAX_CHARS
        );
    }
    if cleaned.chars().any(char::is_control) {
        anyhow::bail!("A category description must not contain control characters");
    }
    Ok(Some(cleaned))
}

/// The user's categories, ordered by name.
fn categories_on(conn: &Connection) -> Result<Vec<Category>> {
    let mut stmt = conn.prepare("SELECT name, description, created_at FROM categories")?;
    let rows = stmt.query_map([], |row| {
        Ok(Category {
            name: row.get(0)?,
            description: non_blank(row.get(1)?),
            created_at: row.get(2)?,
        })
    })?;
    let mut categories = Vec::new();
    for r in rows {
        categories.push(r?);
    }
    categories.sort_by_key(|c| c.name.to_lowercase());
    Ok(categories)
}

impl Database {
    /// The categories the user defined, ordered by name. Empty until the
    /// user adds one.
    pub fn list_categories(&self) -> Result<Vec<Category>> {
        categories_on(&self.conn)
    }

    /// Every category with the number of kept videos (`KEPT_SQL`) it holds.
    pub fn list_categories_with_counts(&self) -> Result<Vec<(Category, usize)>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT i.category, COUNT(*) FROM insights i WHERE i.category IS NOT NULL AND {KEPT_SQL} GROUP BY i.category"
        ))?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, usize>(1)?)))?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for r in rows {
            let (name, count) = r?;
            counts.insert(name, count);
        }
        Ok(self
            .list_categories()?
            .into_iter()
            .map(|category| {
                let count = counts.get(&category.name).copied().unwrap_or(0);
                (category, count)
            })
            .collect())
    }

    /// Defines a category. The name is unique whatever its case; the
    /// description tells the analyzer what belongs in it.
    pub fn add_category(&self, name: &str, description: Option<&str>) -> Result<Category> {
        let name = clean_category_name(name)?;
        let description = clean_category_description(description)?;
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        if let Some(existing) = match_category(&categories_on(&tx)?, &name) {
            anyhow::bail!("Category '{}' already exists", existing);
        }
        let category = Category { name, description, created_at: Utc::now().to_rfc3339() };
        tx.execute(
            "INSERT INTO categories (name, description, created_at) VALUES (?1, ?2, ?3)",
            params![category.name, category.description, category.created_at],
        )?;
        forget_category_answers(&tx)?;
        tx.commit()?;
        Ok(category)
    }

    /// Removes a category (the name matches whatever its case). Its insights
    /// stay and lose their category, and with it the mark that it was set by
    /// hand: the model chooses for them again. Returns the stored name and
    /// the number of insights that carried it.
    pub fn remove_category(&self, name: &str) -> Result<(String, usize)> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let Some(stored) = match_category(&categories_on(&tx)?, name) else {
            anyhow::bail!("Category '{}' does not exist", name.trim());
        };
        let cleared = tx.execute(
            "UPDATE insights SET category = NULL, category_manual = 0 WHERE category = ?1",
            params![stored],
        )?;
        tx.execute("DELETE FROM categories WHERE name = ?1", params![stored])?;
        forget_category_answers(&tx)?;
        tx.commit()?;
        Ok((stored, cleared))
    }

    /// Gives a category another name (`old` matches whatever its case). Its
    /// videos follow. Refused when `new` is the name of a different category;
    /// changing only the spelling of the same one is allowed. Returns the
    /// name before, the name now and the number of insights that carry it.
    pub fn rename_category(&self, old: &str, new: &str) -> Result<(String, String, usize)> {
        let new = clean_category_name(new)?;
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let categories = categories_on(&tx)?;
        let Some(stored) = match_category(&categories, old) else {
            return Err(unknown_category(&categories, old));
        };
        if let Some(existing) = match_category(&categories, &new).filter(|existing| *existing != stored) {
            anyhow::bail!("Category '{}' already exists", existing);
        }
        tx.execute("UPDATE categories SET name = ?1 WHERE name = ?2", params![new, stored])?;
        let moved = tx.execute("UPDATE insights SET category = ?1 WHERE category = ?2", params![new, stored])?;
        forget_category_answers(&tx)?;
        tx.commit()?;
        Ok((stored, new, moved))
    }

    /// Replaces the description of a category (`name` matches whatever its
    /// case). None or blank text removes it. Returns the category as stored.
    pub fn describe_category(&self, name: &str, description: Option<&str>) -> Result<Category> {
        let description = clean_category_description(description)?;
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let categories = categories_on(&tx)?;
        let stored = match_category(&categories, name);
        let Some(mut category) = categories.into_iter().find(|c| Some(&c.name) == stored.as_ref()) else {
            return Err(unknown_category(&categories_on(&tx)?, name));
        };
        tx.execute("UPDATE categories SET description = ?1 WHERE name = ?2", params![description, category.name])?;
        forget_category_answers(&tx)?;
        tx.commit()?;
        category.description = description;
        Ok(category)
    }

    /// Puts a video into a category by the owner's decision; `name` None
    /// puts it into no category. Either way the choice is marked as made by
    /// hand, so no later analysis changes it (`set_video_category_auto`
    /// hands it back). The name matches whatever its case; returns the
    /// category as stored. Fails when the video has no insight or the name
    /// is not one of the user's categories.
    pub fn set_video_category(&self, video_id: &str, name: Option<&str>) -> Result<Option<String>> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let stored = match name {
            Some(raw) => {
                let categories = categories_on(&tx)?;
                match match_category(&categories, raw) {
                    Some(stored) => Some(stored),
                    None => return Err(unknown_category(&categories, raw)),
                }
            }
            None => None,
        };
        let updated = tx.execute(
            "UPDATE insights SET category = ?1, category_manual = 1 WHERE video_id = ?2",
            params![stored, video_id],
        )?;
        if updated == 0 {
            anyhow::bail!("Video {} has no analysis, so it cannot be put into a category", video_id);
        }
        tx.commit()?;
        Ok(stored)
    }

    /// Hands the choice of the category back to the model: the mark set by
    /// `set_video_category` is cleared. The category stays as it is until
    /// the next analysis or `ytc categorize --all`; a video without a
    /// category is asked about by the next `ytc categorize`. No error when
    /// the mark was not set; an error when the video has no insight.
    pub fn set_video_category_auto(&self, video_id: &str) -> Result<()> {
        let updated =
            self.conn.execute(
                "UPDATE insights SET category_manual = 0, category_asked = 0 WHERE video_id = ?1",
                params![video_id],
            )?;
        if updated == 0 {
            anyhow::bail!("Video {} has no analysis", video_id);
        }
        Ok(())
    }

    /// Stores the category the model chose for an already analysed video and
    /// changes nothing else on the row. A name that is not one of the user's
    /// categories is stored as no category. A video whose category was set
    /// by hand is left alone: returns false for it and for a video without
    /// an insight. The row is marked as asked, so `list_categorize_targets`
    /// leaves out a video that fits no category until the categories change.
    pub fn set_model_category(&self, video_id: &str, category: Option<&str>) -> Result<bool> {
        let updated = self.conn.execute(
            "UPDATE insights SET category = (SELECT name FROM categories WHERE name = ?1), category_asked = 1
             WHERE video_id = ?2 AND category_manual = 0",
            params![category, video_id],
        )?;
        Ok(updated > 0)
    }

    /// The videos `ytc categorize` works on, the one analysed longest ago
    /// first: judged by an LLM, category not set by hand, without a category
    /// and not yet asked about with the categories as they are now. With
    /// `all` every one whose category was not set by hand.
    pub fn list_categorize_targets(&self, all: bool) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT video_id FROM insights
             WHERE substr(analyzer_type, 1, 4) = 'llm:'
               AND category_manual = 0
               AND (?1 OR ((category IS NULL OR trim(category) = '') AND category_asked = 0))
             ORDER BY category_asked, updated_at, video_id",
        )?;
        let rows = stmt.query_map(params![all], |r| r.get::<_, String>(0))?;
        let mut ids = Vec::new();
        for r in rows {
            ids.push(r?);
        }
        Ok(ids)
    }
}

/// Clears the mark that `ytc categorize` asked the model about a video. Called
/// whenever the categories change: a video that fitted none may fit now.
fn forget_category_answers(conn: &Connection) -> Result<()> {
    conn.execute("UPDATE insights SET category_asked = 0 WHERE category_asked != 0", [])?;
    Ok(())
}

/// The error for a name that is none of the user's categories. It lists the
/// names there are.
fn unknown_category(categories: &[Category], name: &str) -> anyhow::Error {
    if categories.is_empty() {
        return anyhow::anyhow!(
            "Category '{}' does not exist: no categories are defined. Add one with: ytc category add <name>",
            name.trim()
        );
    }
    let names: Vec<&str> = categories.iter().map(|c| c.name.as_str()).collect();
    anyhow::anyhow!("Category '{}' does not exist. Your categories: {}", name.trim(), names.join(", "))
}

// ---------------------------------------------------------------------------
// Combining two libraries (`ytc merge-db`)
// ---------------------------------------------------------------------------

/// The columns the merge copies: every column of the table.
type MergeColumns = [&'static str];

const MERGE_CHANNEL_COLUMNS: &MergeColumns =
    &["id", "title", "url", "status", "status_manual", "rolling_slop_score", "created_at", "last_crawled_at"];

const MERGE_VIDEO_COLUMNS: &MergeColumns =
    &["id", "channel_id", "title", "url", "published_at", "status", "fail_count", "created_at"];

const MERGE_TRANSCRIPT_COLUMNS: &MergeColumns = &[
    "video_id",
    "raw_char_count",
    "cleaned_char_count",
    "stripped_ad_seconds",
    "stripped_segments_count",
    "text",
    "video_duration_sec",
    "covered_until_sec",
    "source",
    "created_at",
];

const MERGE_INSIGHT_COLUMNS: &MergeColumns = &[
    "video_id",
    "slop_score",
    "is_worth_keeping",
    "quality_rating",
    "summary",
    "key_insights_json",
    "takeaways_json",
    "quotes_json",
    "topics_json",
    "mentioned_creators_json",
    "analyzer_type",
    "category",
    "category_manual",
    "category_asked",
    "topic_verdict",
    "topic_reason",
    "info_status",
    "info_reason",
    "sections_total",
    "transcript_chars",
    "excluded_manual",
    "excluded_reason",
    "excluded_at",
    "created_at",
    "updated_at",
];

const MERGE_CATEGORY_COLUMNS: &MergeColumns = &["name", "description", "created_at"];

type SqlValue = rusqlite::types::Value;

/// Position of a column in one of the merge column lists. The lists are
/// constants of this file, so a miss is a programming error, reported as one.
fn merge_col(columns: &MergeColumns, name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|col| *col == name)
        .with_context(|| format!("internal error: merge column '{}' is not listed", name))
}

fn value_text(value: &SqlValue) -> Option<String> {
    match value {
        SqlValue::Text(text) => Some(text.clone()),
        _ => None,
    }
}

fn value_flag(value: &SqlValue) -> bool {
    matches!(value, SqlValue::Integer(n) if *n != 0)
}

/// `SELECT` over `table` that yields `columns` in order.
fn merge_select(table: &str, columns: &MergeColumns) -> String {
    format!("SELECT {} FROM {}", columns.join(", "), table)
}

fn merge_insert(table: &str, columns: &MergeColumns) -> String {
    let marks: Vec<String> = (1..=columns.len()).map(|n| format!("?{}", n)).collect();
    format!("INSERT INTO {} ({}) VALUES ({})", table, columns.join(", "), marks.join(", "))
}

/// 0 for a word-count estimate, 1 for a judgment made with the LLM. The
/// higher rank wins a merge whatever the dates.
fn analysis_rank(analyzer_type: &str) -> u8 {
    if analyzer_type.starts_with("heuristic") {
        0
    } else {
        1
    }
}

/// True only when both timestamps can be read and `candidate` is strictly
/// later. Without a usable timestamp on either side the current row stays.
fn is_later(candidate: &str, current: &str) -> bool {
    match (
        chrono::DateTime::parse_from_rfc3339(candidate.trim()),
        chrono::DateTime::parse_from_rfc3339(current.trim()),
    ) {
        (Ok(a), Ok(b)) => a > b,
        _ => false,
    }
}

/// True when both paths name the same existing file.
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

impl Database {
    /// Copies channels, videos, transcripts, insights and categories of
    /// another library into this one, in one transaction, without losing
    /// data:
    ///
    /// - rows that exist only in the other database are added;
    /// - for a video analysed in both, the better analysis stays: an LLM
    ///   result beats a heuristic estimate; among equals the one analysed
    ///   later (`updated_at`) wins, and without a usable timestamp the
    ///   current one stays;
    /// - the longer transcript stays;
    /// - a channel status set by hand and a manual disqualification survive
    ///   from either side (the current database's own manual status wins);
    /// - a category set by hand survives from either side with its mark (the
    ///   current database's own choice wins).
    ///
    /// The other file is opened read-only. Topic links are rebuilt in the
    /// same transaction; channel averages are recalculated afterwards.
    /// Merging the same file a second time changes nothing.
    pub fn merge_from(&self, other: &Path) -> Result<MergeStats> {
        if !other.is_file() {
            anyhow::bail!("{} does not exist or is not a file", other.display());
        }
        if let Some(own) = self.conn.path().filter(|p| !p.is_empty()) {
            if same_file(Path::new(own), other) {
                anyhow::bail!("{} is the database that is already open; there is nothing to merge", other.display());
            }
        }

        // Read-only: the merge must not be able to change the other file.
        let src = Connection::open_with_flags(
            other,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("Failed to open {} read-only", other.display()))?;
        src.busy_timeout(std::time::Duration::from_secs(5))?;
        if !matches!(schema_state(&src), Ok((SCHEMA_VERSION, _))) {
            return Err(not_a_database(other));
        }

        let mut stats = MergeStats::default();
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;

        // --- channels ---------------------------------------------------
        {
            let c_status = merge_col(MERGE_CHANNEL_COLUMNS, "status")?;
            let c_manual = merge_col(MERGE_CHANNEL_COLUMNS, "status_manual")?;
            let insert = merge_insert("channels", MERGE_CHANNEL_COLUMNS);
            let mut stmt = src.prepare(&merge_select("channels", MERGE_CHANNEL_COLUMNS))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let mut values = read_values(row, MERGE_CHANNEL_COLUMNS.len())?;
                let Some(id) = value_text(&values[0]) else { continue };
                let status = value_text(&values[c_status])
                    .filter(|s| CHANNEL_STATUSES.contains(&s.as_str()))
                    .unwrap_or_else(|| "active".to_string());
                let src_manual = value_flag(&values[c_manual]);
                values[c_status] = SqlValue::Text(status.clone());
                values[c_manual] = SqlValue::Integer(i64::from(src_manual));

                let current: Option<i64> = tx
                    .query_row(
                        "SELECT COALESCE(status_manual, 0) FROM channels WHERE id = ?1",
                        params![id],
                        |r| r.get(0),
                    )
                    .optional()?;
                match current {
                    None => {
                        tx.execute(&insert, rusqlite::params_from_iter(values.iter()))?;
                        stats.channels_added += 1;
                    }
                    Some(0) if src_manual => {
                        tx.execute(
                            "UPDATE channels SET status = ?1, status_manual = 1 WHERE id = ?2",
                            params![status, id],
                        )?;
                        stats.channels_updated += 1;
                    }
                    Some(_) => stats.channels_unchanged += 1,
                }
            }
        }

        // --- videos -----------------------------------------------------
        // Status of every video in the other database, and which of them
        // were already here: an imported analysis takes its status along.
        let mut source_status: HashMap<String, String> = HashMap::new();
        let mut preexisting: HashSet<String> = HashSet::new();
        {
            let v_status = merge_col(MERGE_VIDEO_COLUMNS, "status")?;
            let insert = merge_insert("videos", MERGE_VIDEO_COLUMNS);
            let mut stmt = src.prepare(&merge_select("videos", MERGE_VIDEO_COLUMNS))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let values = read_values(row, MERGE_VIDEO_COLUMNS.len())?;
                let Some(id) = value_text(&values[0]) else { continue };
                if let Some(status) = value_text(&values[v_status]) {
                    source_status.insert(id.clone(), status);
                }
                let exists: bool =
                    tx.query_row("SELECT EXISTS(SELECT 1 FROM videos WHERE id = ?1)", params![id], |r| r.get(0))?;
                if exists {
                    stats.videos_unchanged += 1;
                    preexisting.insert(id);
                } else {
                    tx.execute(&insert, rusqlite::params_from_iter(values.iter()))?;
                    stats.videos_added += 1;
                }
            }
        }

        // --- transcripts ------------------------------------------------
        {
            let t_text = merge_col(MERGE_TRANSCRIPT_COLUMNS, "text")?;
            let insert = merge_insert("transcripts", MERGE_TRANSCRIPT_COLUMNS);
            let mut stmt = src.prepare(&merge_select("transcripts", MERGE_TRANSCRIPT_COLUMNS))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let values = read_values(row, MERGE_TRANSCRIPT_COLUMNS.len())?;
                let Some(video_id) = value_text(&values[0]) else { continue };
                let has_video: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM videos WHERE id = ?1)",
                    params![video_id],
                    |r| r.get(0),
                )?;
                if !has_video {
                    stats.orphans_skipped += 1;
                    continue;
                }
                let src_len = value_text(&values[t_text]).map_or(0, |t| t.chars().count());
                let current_len: Option<i64> = tx
                    .query_row(
                        "SELECT length(text) FROM transcripts WHERE video_id = ?1",
                        params![video_id],
                        |r| r.get(0),
                    )
                    .optional()?;
                match current_len {
                    None => {
                        tx.execute(&insert, rusqlite::params_from_iter(values.iter()))?;
                        stats.transcripts_added += 1;
                    }
                    Some(len) if (src_len as i64) > len => {
                        tx.execute("DELETE FROM transcripts WHERE video_id = ?1", params![video_id])?;
                        tx.execute(&insert, rusqlite::params_from_iter(values.iter()))?;
                        stats.transcripts_replaced += 1;
                    }
                    Some(_) => stats.transcripts_kept_current += 1,
                }
            }
        }

        // --- categories -------------------------------------------------
        // Categories the other database defines are added; one that exists
        // here under the same name (whatever its case) stays as it is.
        let mut categories = categories_on(&tx)?;
        {
            let mut stmt = src.prepare(&merge_select("categories", MERGE_CATEGORY_COLUMNS))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let values = read_values(row, MERGE_CATEGORY_COLUMNS.len())?;
                let Some(name) = value_text(&values[0]).and_then(|raw| clean_category_name(&raw).ok()) else {
                    continue;
                };
                if match_category(&categories, &name).is_some() {
                    continue;
                }
                let category = Category {
                    name,
                    description: clean_category_description(value_text(&values[1]).as_deref()).ok().flatten(),
                    created_at: value_text(&values[2])
                        .filter(|t| !t.trim().is_empty())
                        .unwrap_or_else(|| Utc::now().to_rfc3339()),
                };
                tx.execute(
                    "INSERT INTO categories (name, description, created_at) VALUES (?1, ?2, ?3)",
                    params![category.name, category.description, category.created_at],
                )?;
                categories.push(category);
                stats.categories_added += 1;
            }
        }

        // --- insights ---------------------------------------------------
        {
            let i_category = merge_col(MERGE_INSIGHT_COLUMNS, "category")?;
            let i_category_manual = merge_col(MERGE_INSIGHT_COLUMNS, "category_manual")?;
            let i_analyzer = merge_col(MERGE_INSIGHT_COLUMNS, "analyzer_type")?;
            let i_updated = merge_col(MERGE_INSIGHT_COLUMNS, "updated_at")?;
            let i_excl = merge_col(MERGE_INSIGHT_COLUMNS, "excluded_manual")?;
            let i_excl_reason = merge_col(MERGE_INSIGHT_COLUMNS, "excluded_reason")?;
            let i_excl_at = merge_col(MERGE_INSIGHT_COLUMNS, "excluded_at")?;
            let insert = merge_insert("insights", MERGE_INSIGHT_COLUMNS);
            let mut status_followed: HashSet<String> = HashSet::new();

            let mut stmt = src.prepare(&merge_select("insights", MERGE_INSIGHT_COLUMNS))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let mut values = read_values(row, MERGE_INSIGHT_COLUMNS.len())?;
                let Some(video_id) = value_text(&values[0]) else { continue };
                let has_video: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM videos WHERE id = ?1)",
                    params![video_id],
                    |r| r.get(0),
                )?;
                if !has_video {
                    stats.orphans_skipped += 1;
                    continue;
                }

                let src_analyzer = value_text(&values[i_analyzer]).unwrap_or_default();
                let src_time = value_text(&values[i_updated]).unwrap_or_default();
                let src_excluded = value_flag(&values[i_excl]);
                // An imported row keeps its category only under a name this
                // database knows, in the spelling used here.
                let src_category_raw = value_text(&values[i_category]).filter(|raw| !raw.trim().is_empty());
                let src_category = src_category_raw.as_deref().and_then(|raw| match_category(&categories, raw));
                // The mark "set by hand" travels with the category. It is
                // dropped with a name this database does not know: such a row
                // must not arrive as "no category, by hand".
                let src_category_manual =
                    value_flag(&values[i_category_manual]) && (src_category.is_some() || src_category_raw.is_none());
                values[i_category] = src_category.clone().map_or(SqlValue::Null, SqlValue::Text);
                values[i_category_manual] = SqlValue::Integer(i64::from(src_category_manual));

                type CurrentInsight = (String, String, bool, Option<String>, Option<String>, Option<String>, bool);
                let current: Option<CurrentInsight> = tx
                    .query_row(
                        "SELECT analyzer_type, updated_at, excluded_manual, excluded_reason, excluded_at,
                                category, category_manual
                         FROM insights WHERE video_id = ?1",
                        params![video_id],
                        |r| {
                            Ok((
                                r.get(0)?,
                                r.get(1)?,
                                r.get::<_, i64>(2)? != 0,
                                r.get(3)?,
                                r.get(4)?,
                                r.get(5)?,
                                r.get::<_, i64>(6)? != 0,
                            ))
                        },
                    )
                    .optional()?;

                let imported = match current {
                    None => {
                        tx.execute(&insert, rusqlite::params_from_iter(values.iter()))?;
                        stats.insights_added += 1;
                        true
                    }
                    Some((cur_analyzer, cur_time, cur_excluded, cur_reason, cur_at, cur_category, cur_category_manual)) => {
                        let (src_rank, cur_rank) = (analysis_rank(&src_analyzer), analysis_rank(&cur_analyzer));
                        let take_source =
                            src_rank > cur_rank || (src_rank == cur_rank && is_later(&src_time, &cur_time));
                        if take_source {
                            // The owner's disqualification outlives the row it was set on.
                            if cur_excluded {
                                values[i_excl] = SqlValue::Integer(1);
                                values[i_excl_reason] = cur_reason.map_or(SqlValue::Null, SqlValue::Text);
                                values[i_excl_at] = cur_at.map_or(SqlValue::Null, SqlValue::Text);
                            }
                            // So does the category the owner chose here.
                            if cur_category_manual {
                                values[i_category] = non_blank(cur_category).map_or(SqlValue::Null, SqlValue::Text);
                                values[i_category_manual] = SqlValue::Integer(1);
                            }
                            tx.execute("DELETE FROM insights WHERE video_id = ?1", params![video_id])?;
                            tx.execute(&insert, rusqlite::params_from_iter(values.iter()))?;
                            stats.insights_replaced += 1;
                            true
                        } else {
                            stats.insights_kept_current += 1;
                            if src_excluded && !cur_excluded {
                                tx.execute(
                                    "UPDATE insights SET excluded_manual = 1, excluded_reason = ?1, excluded_at = ?2 WHERE video_id = ?3",
                                    params![values[i_excl_reason], values[i_excl_at], video_id],
                                )?;
                                stats.exclusions_carried += 1;
                            }
                            if src_category_manual && !cur_category_manual {
                                tx.execute(
                                    "UPDATE insights SET category = ?1, category_manual = 1 WHERE video_id = ?2",
                                    params![src_category, video_id],
                                )?;
                                stats.manual_categories_carried += 1;
                            }
                            false
                        }
                    }
                };

                // A video that was pending here and analysed there is now analysed here.
                if imported && preexisting.contains(&video_id) && !status_followed.contains(&video_id) {
                    if let Some(status) = source_status.get(&video_id) {
                        let changed = tx.execute(
                            "UPDATE videos SET status = ?1 WHERE id = ?2 AND status != ?1",
                            params![status, video_id],
                        )?;
                        if changed > 0 {
                            stats.videos_updated += 1;
                            stats.videos_unchanged = stats.videos_unchanged.saturating_sub(1);
                            status_followed.insert(video_id);
                        }
                    }
                }
            }
        }

        // The topic readers use video_topics; it must match the merged rows
        // before anyone else can see them.
        rebuild_topic_links_on(&tx)?;
        // The two libraries may have had different categories: no answer of
        // 'ytc categorize' is taken as given for the merged set.
        forget_category_answers(&tx)?;
        tx.commit()?;

        // Averages and automatic statuses follow the merged rows. A failure
        // here loses nothing: the next listing recalculates again.
        for channel in self.list_channels()? {
            if let Err(e) = self.recalculate_channel_stats(&channel.id) {
                eprintln!("Warning: channel stats recalculation failed for {}: {}", channel.id, e);
            }
        }

        Ok(stats)
    }
}

/// The first `count` columns of a row, as they are stored.
fn read_values(row: &rusqlite::Row<'_>, count: usize) -> rusqlite::Result<Vec<SqlValue>> {
    (0..count).map(|idx| row.get::<_, SqlValue>(idx)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A database in its own temp directory; never the user's real one.
    fn test_db() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open test db");
        (dir, db)
    }

    fn add_channel(db: &Database, id: &str) {
        db.upsert_channel(&Channel {
            id: id.to_string(),
            title: format!("Channel {}", id),
            url: format!("https://example.invalid/channel/{}", id),
            status: "active".to_string(),
            rolling_slop_score: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            last_crawled_at: None,
        })
        .expect("upsert channel");
    }

    fn add_video(db: &Database, id: &str, channel_id: &str) {
        db.upsert_video(&Video {
            id: id.to_string(),
            channel_id: channel_id.to_string(),
            title: format!("Video {}", id),
            url: format!("https://example.invalid/watch?v={}", id),
            published_at: format!("2026-02-01T00:00:00Z#{}", id),
            status: "pending".to_string(),
            created_at: "2026-02-01T00:00:00Z".to_string(),
        })
        .expect("upsert video");
    }

    fn insight(video_id: &str, slop_score: u32, keep: bool, topics: &[&str]) -> InsightResult {
        InsightResult {
            video_id: video_id.to_string(),
            slop_score,
            is_worth_keeping: keep,
            quality_rating: "High Signal".to_string(),
            summary: format!("Summary of {}", video_id),
            key_insights: vec!["one".to_string()],
            takeaways: vec![],
            quotes: vec![],
            topics: topics.iter().map(|t| t.to_string()).collect(),
            mentioned_creators: vec![],
            analyzer_type: "llm:test-model".to_string(),
            category: None,
        }
    }

    fn add_scored_video(db: &Database, id: &str, channel_id: &str, slop_score: u32) {
        add_video(db, id, channel_id);
        db.save_insight(&insight(id, slop_score, true, &[])).expect("save insight");
    }

    fn channel_status(db: &Database, id: &str) -> String {
        db.get_channel(id).expect("get channel").expect("channel exists").status
    }

    #[test]
    fn auto_status_needs_the_minimum_number_of_videos_both_ways() {
        assert_eq!(auto_channel_status(1, 20.0), "active");
        assert_eq!(auto_channel_status(2, 95.0), "active");
        assert_eq!(auto_channel_status(AUTO_MUTE_MIN_VIDEOS, 20.0), "high_signal");
        assert_eq!(auto_channel_status(AUTO_MUTE_MIN_VIDEOS, HIGH_SIGNAL_SLOP_AVG), "high_signal");
        assert_eq!(auto_channel_status(AUTO_MUTE_MIN_VIDEOS, 50.0), "active");
        assert_eq!(auto_channel_status(AUTO_MUTE_MIN_VIDEOS, AUTO_MUTE_SLOP_AVG), "active");
        assert_eq!(auto_channel_status(AUTO_MUTE_MIN_VIDEOS, 80.0), "muted");
    }

    #[test]
    fn one_good_video_does_not_promote_a_channel_but_three_do() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");

        add_scored_video(&db, "v1", "UC1", 20);
        assert_eq!(channel_status(&db, "UC1"), "active");
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), Some(20.0));

        add_scored_video(&db, "v2", "UC1", 20);
        assert_eq!(channel_status(&db, "UC1"), "active");

        add_scored_video(&db, "v3", "UC1", 20);
        assert_eq!(channel_status(&db, "UC1"), "high_signal");
    }

    #[test]
    fn manual_status_sticks_until_handed_back_to_auto() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        for id in ["v1", "v2", "v3"] {
            add_scored_video(&db, id, "UC1", 10);
        }
        assert_eq!(channel_status(&db, "UC1"), "high_signal");
        assert!(!db.is_channel_status_manual("UC1").expect("manual flag"));

        db.set_channel_status("UC1", "muted").expect("set status");
        assert!(db.is_channel_status_manual("UC1").expect("manual flag"));
        db.recalculate_channel_stats("UC1").expect("recalc");
        assert_eq!(channel_status(&db, "UC1"), "muted");

        db.set_channel_status_auto("UC1").expect("back to auto");
        assert!(!db.is_channel_status_manual("UC1").expect("manual flag"));
        assert_eq!(channel_status(&db, "UC1"), "high_signal");

        assert!(db.set_channel_status("UC1", "banana").is_err());
        assert!(db.set_channel_status("UC-missing", "muted").is_err());
        assert!(db.set_channel_status_auto("UC-missing").is_err());
    }

    #[test]
    fn auto_on_a_channel_without_scores_ends_the_manual_mute() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.set_channel_status("UC1", "muted").expect("set status");
        db.set_channel_status_auto("UC1").expect("back to auto");
        assert_eq!(channel_status(&db, "UC1"), "active");
    }

    #[test]
    fn an_automatic_status_change_is_announced_with_the_way_back() {
        let notice = auto_status_change_notice("UC1", "Some Channel", "active", "muted", 82.4, 5)
            .expect("a change is announced");
        assert!(notice.contains("Some Channel (UC1) active -> muted"), "{}", notice);
        assert!(notice.contains("82.4/100"), "{}", notice);
        assert!(notice.contains("5 scored video(s)"), "{}", notice);
        assert!(notice.ends_with("undo with: ytc set-status UC1 active"), "{}", notice);

        assert_eq!(auto_status_change_notice("UC1", "Some Channel", "muted", "muted", 82.4, 5), None);
    }

    #[test]
    fn heuristic_scores_do_not_enter_the_rolling_average() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        // Three word-count estimates that would mute the channel on their own.
        for id in ["h1", "h2", "h3"] {
            add_video(&db, id, "UC1");
            let mut row = insight(id, 95, false, &[]);
            row.analyzer_type = "heuristic".to_string();
            db.save_insight(&row).expect("save heuristic");
        }
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), None);
        assert_eq!(channel_status(&db, "UC1"), "active");

        add_scored_video(&db, "v1", "UC1", 40);
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), Some(40.0));
        assert_eq!(channel_status(&db, "UC1"), "active");
    }

    #[test]
    fn an_average_of_exactly_zero_is_a_score_not_a_missing_one() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        add_channel(&db, "UC2");
        for id in ["v1", "v2", "v3"] {
            add_scored_video(&db, id, "UC1", 0);
        }
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), Some(0.0));
        assert_eq!(db.recalculate_channel_stats("UC2").expect("recalc"), None);
        assert_eq!(channel_status(&db, "UC1"), "high_signal");

        let health = db.list_channels_with_health().expect("health");
        let score = |id: &str| health.iter().find(|c| c.id == id).expect("listed").avg_slop_score;
        assert_eq!(score("UC1"), Some(0.0));
        assert_eq!(score("UC2"), None);
        assert_eq!(db.get_channel("UC1").expect("get").expect("exists").rolling_slop_score, Some(0.0));
    }

    #[test]
    fn failures_are_counted_per_video() {
        let (_dir, db) = test_db();
        add_video(&db, "v1", "UC1");
        assert_eq!(db.video_fail_count("v1").expect("count"), 0);
        assert_eq!(db.mark_video_failed("v1").expect("first failure"), 1);
        assert_eq!(db.mark_video_failed("v1").expect("second failure"), 2);
        assert_eq!(db.video_fail_count("v1").expect("count"), 2);
        assert_eq!(db.get_video("v1").expect("get").expect("exists").status, "failed");

        assert!(db.mark_video_failed("missing").is_err());
        assert_eq!(db.video_fail_count("missing").expect("count"), 0);
    }

    #[test]
    fn analysed_videos_are_listed_longest_ago_first_and_by_analyzer_prefix() {
        let (_dir, db) = test_db();
        for id in ["b", "a", "c", "d"] {
            add_video(&db, id, "UC1");
        }
        let mut heuristic = insight("b", 50, false, &[]);
        heuristic.analyzer_type = "heuristic".to_string();
        db.save_insight(&heuristic).expect("save");
        heuristic.video_id = "a".to_string();
        db.save_insight(&heuristic).expect("save");
        db.save_insight(&insight("c", 20, true, &[])).expect("save");
        db.save_insight(&insight("d", 20, true, &[])).expect("save");
        for (id, when) in [("a", "2026-03-04"), ("b", "2026-03-01"), ("c", "2026-03-02"), ("d", "2026-03-02")] {
            set_analysed_at(&db, id, &format!("{}T00:00:00+00:00", when));
        }

        assert_eq!(db.list_analysed_video_ids(None).expect("list"), vec!["b", "c", "d", "a"]);
        assert_eq!(db.list_analysed_video_ids(Some("heuristic")).expect("list"), vec!["b", "a"]);
        assert_eq!(db.list_analysed_video_ids(Some("llm:")).expect("list"), vec!["c", "d"]);
        // LIKE wildcards in the prefix are literal text, not patterns.
        assert!(db.list_analysed_video_ids(Some("%")).expect("list").is_empty());
        assert!(db.list_analysed_video_ids(Some("l_m")).expect("list").is_empty());

        // A video that is analysed again moves to the end.
        db.save_insight(&insight("b", 30, true, &[])).expect("save again");
        assert_eq!(db.list_analysed_video_ids(None).expect("list"), vec!["c", "d", "a", "b"]);
    }

    #[test]
    fn topic_names_are_trimmed_deduplicated_and_text_only() {
        assert_eq!(topic_names(r#"[" Rivers ", "Rivers", "", "  ", 7, null, "Glaciers"]"#), vec!["Rivers", "Glaciers"]);
        assert!(topic_names("not json").is_empty());
        assert!(topic_names(r#"{"a": 1}"#).is_empty());
    }

    #[test]
    fn saving_an_insight_keeps_topic_links_in_step_with_topics_json() {
        let (_dir, db) = test_db();
        add_video(&db, "v1", "UC1");
        add_video(&db, "v2", "UC1");
        db.save_insight(&insight("v1", 20, true, &["Rivers", " Glaciers "])).expect("save");
        db.save_insight(&insight("v2", 20, true, &["Rivers"])).expect("save");
        assert_eq!(
            db.list_all_topics().expect("topics"),
            vec![("Rivers".to_string(), 2), ("Glaciers".to_string(), 1)]
        );

        // Re-analysis drops a topic; a video that is not kept any more leaves the counts.
        db.save_insight(&insight("v1", 20, true, &["Deltas"])).expect("save");
        db.save_insight(&insight("v2", 80, false, &["Rivers"])).expect("save");
        assert_eq!(db.list_all_topics().expect("topics"), vec![("Deltas".to_string(), 1)]);
        assert!(db.get_topic_videos("Rivers").expect("videos").is_empty());
    }

    // ----- one kept rule ---------------------------------------------------

    fn judged(verdict: Option<&str>, info: Option<&str>) -> InsightJudgment {
        InsightJudgment {
            topic_verdict: verdict.map(str::to_string),
            info_status: info.map(str::to_string),
            ..Default::default()
        }
    }

    fn sorted(mut ids: Vec<String>) -> Vec<String> {
        ids.sort();
        ids
    }

    #[test]
    fn one_rule_decides_kept_in_every_listing() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        // (id, quality flag, topic verdict, info status, disqualified, kept?)
        let rows: [(&str, bool, Option<&str>, Option<&str>, bool, bool); 9] = [
            ("unjudged", true, None, None, false, true),
            ("good", true, Some("ok"), Some("sufficient"), false, true),
            ("slop", false, Some("ok"), Some("sufficient"), false, false),
            ("fringe", true, Some("fringe"), Some("sufficient"), false, false),
            ("nonsense", true, Some("nonsense"), None, false, false),
            ("politics", true, Some("politics"), Some("sufficient"), false, false),
            ("short", true, Some("ok"), Some("insufficient"), false, false),
            ("thrown", true, Some("ok"), Some("sufficient"), true, false),
            ("thrownbare", true, None, None, true, false),
        ];
        for (id, keep, verdict, info, excluded, _) in rows {
            add_video(&db, id, "UC1");
            let mut row = insight(id, 20, keep, &["Shared", id]);
            row.mentioned_creators = vec![format!("creator-{}", id)];
            db.save_insight_judged(&row, &judged(verdict, info)).expect("save");
            if excluded {
                db.exclude_video(id, Some("owner said no")).expect("exclude");
            }
        }
        let expected: Vec<String> = sorted(rows.iter().filter(|r| r.5).map(|r| r.0.to_string()).collect());
        assert_eq!(expected, vec!["good", "unjudged"]);

        // The fragment itself, as a plain query.
        let direct: Vec<String> = {
            let mut stmt = db
                .conn
                .prepare(&format!("SELECT i.video_id FROM insights i WHERE {}", KEPT_SQL))
                .expect("prepare");
            let ids = stmt.query_map([], |r| r.get::<_, String>(0)).expect("query");
            ids.collect::<rusqlite::Result<_>>().expect("rows")
        };
        assert_eq!(sorted(direct), expected);

        // Every reader that lists or counts kept videos.
        let meta = db.list_insight_meta().expect("meta");
        for (id, _, _, _, _, kept) in rows {
            assert_eq!(meta[id].kept, kept, "meta.kept of {}", id);
            assert_eq!(db.get_insight_meta(id).expect("meta").expect("row").kept, kept, "{}", id);
        }
        let library = db.list_library(true, 100).expect("library");
        assert_eq!(sorted(library.into_iter().map(|(v, _)| v.id).collect()), expected);
        let kept = db.get_all_kept_videos_with_details().expect("kept");
        assert_eq!(sorted(kept.into_iter().map(|(v, _, _)| v.id).collect()), expected);
        let topic = db.get_topic_videos("Shared").expect("topic videos");
        assert_eq!(sorted(topic.into_iter().map(|(v, _)| v.id).collect()), expected);
        let topics = db.list_all_topics().expect("topics");
        assert_eq!(
            topics,
            vec![("Shared".to_string(), 2), ("good".to_string(), 1), ("unjudged".to_string(), 1)]
        );
        let mentions = db.get_all_mentioned_creators().expect("mentions");
        assert_eq!(
            sorted(mentions.into_iter().map(|(m, _)| m).collect()),
            vec!["creator-good", "creator-unjudged"]
        );
        let hits = db.search("Video").expect("search");
        assert_eq!(hits.len(), rows.len());
        for hit in &hits {
            let want = rows.iter().find(|r| r.0 == hit.video_id).expect("known").5;
            assert_eq!(hit.is_worth_keeping, want, "search hit {}", hit.video_id);
        }
        let thrown = hits.iter().find(|h| h.video_id == "thrown").expect("hit");
        assert!(thrown.excluded_manual);
        let short = hits.iter().find(|h| h.video_id == "short").expect("hit");
        assert_eq!(short.info_status.as_deref(), Some("insufficient"));

        // The unfiltered library still lists everything.
        assert_eq!(db.list_library(false, 100).expect("library").len(), rows.len());

        // Channel counts: disjoint, and they add up.
        let health = db.list_channels_with_health().expect("health");
        let c = &health[0];
        assert_eq!(c.kept_count, 2);
        assert_eq!(c.below_cutoff_count, 1);
        assert_eq!(c.off_topic_count, 3);
        assert_eq!(c.insufficient_count, 1);
        assert_eq!(c.excluded_count, 2);
        assert_eq!(c.total_evaluated, rows.len());
    }

    #[test]
    fn search_lists_kept_videos_before_rows_without_a_real_score() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        // 22 clips too short to judge: placeholder score 0, not kept.
        for n in 0..22 {
            let id = format!("short{:02}", n);
            add_video(&db, &id, "UC1");
            let mut row = insight(&id, 0, false, &[]);
            row.analyzer_type = "info:insufficient".to_string();
            let mut judgment = judged(Some("ok"), Some("insufficient"));
            judgment.info_reason = Some("only 12 words".to_string());
            db.save_insight_judged(&row, &judgment).expect("save");
        }
        add_judged_video(&db, "kept-good", "UC1", 20, true, "ok", "sufficient");
        add_judged_video(&db, "kept-ok", "UC1", 40, true, "ok", "sufficient");
        add_judged_video(&db, "dropped", "UC1", 70, false, "ok", "sufficient");

        // Every title contains "Video".
        let hits = db.search("Video").expect("search");
        assert_eq!(hits.len(), 20);
        let ids: Vec<&str> = hits.iter().map(|h| h.video_id.as_str()).collect();
        assert_eq!(&ids[..3], ["kept-good", "kept-ok", "dropped"], "judged hits first, kept ones on top");
        assert!(hits[..2].iter().all(|h| h.is_worth_keeping));
        assert!(hits[2..].iter().all(|h| !h.is_worth_keeping));
        // The rest of the page is filled with the unjudged rows, lowest
        // placeholder first; none of them is shown as a score.
        assert!(hits[3..].iter().all(|h| h.has_no_score()));
        assert_eq!(hits[3].score_label(), "Not enough information (only 12 words)");
        assert_eq!(hits[0].score_label(), "20/100");
        assert!(!hits[0].has_no_score());
    }

    // ----- disqualify --------------------------------------------------------

    #[test]
    fn a_disqualified_video_stays_out_until_it_is_included_again() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        add_scored_video(&db, "v1", "UC1", 10);
        add_scored_video(&db, "v2", "UC1", 12);
        assert!(db.list_excluded().expect("list").is_empty());

        db.exclude_video("v1", Some("  not my field  ")).expect("exclude");
        let meta = db.get_insight_meta("v1").expect("meta").expect("row");
        assert!(meta.excluded_manual && !meta.kept);
        assert_eq!(meta.excluded_reason.as_deref(), Some("not my field"));
        assert!(meta.excluded_at.is_some());
        // The model's own verdict is not rewritten; only the owner's flag is set.
        assert!(db.get_insight("v1").expect("get").expect("row").is_worth_keeping);

        let listed = db.list_excluded().expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].video_id, "v1");
        assert_eq!(listed[0].title, "Video v1");
        assert_eq!(listed[0].channel_id, "UC1");
        assert_eq!(listed[0].channel_title.as_deref(), Some("Channel UC1"));
        assert_eq!(listed[0].reason.as_deref(), Some("not my field"));
        assert_eq!(listed[0].slop_score, 10);

        // Re-analysis, even a glowing one, does not overrule the owner.
        db.save_insight_judged(&insight("v1", 1, true, &["Rivers"]), &judged(Some("ok"), Some("sufficient")))
            .expect("re-analysis");
        db.save_insight(&insight("v1", 1, true, &["Rivers"])).expect("plain re-analysis");
        let meta = db.get_insight_meta("v1").expect("meta").expect("row");
        assert!(meta.excluded_manual && !meta.kept);
        assert_eq!(meta.excluded_reason.as_deref(), Some("not my field"));
        assert!(db.get_topic_videos("Rivers").expect("topic").is_empty());

        db.include_video("v1").expect("include");
        let meta = db.get_insight_meta("v1").expect("meta").expect("row");
        assert!(!meta.excluded_manual && meta.kept);
        assert_eq!((meta.excluded_reason, meta.excluded_at), (None, None));
        assert!(db.list_excluded().expect("list").is_empty());
        // Including a video that is not excluded is not an error.
        db.include_video("v2").expect("include again");

        // A reason is optional; a video without an analysis cannot be disqualified.
        db.exclude_video("v2", None).expect("exclude without reason");
        assert_eq!(db.list_excluded().expect("list")[0].reason, None);
        add_video(&db, "pending", "UC1");
        assert!(db.exclude_video("pending", None).is_err());
        assert!(db.exclude_video("missing", None).is_err());
        assert!(db.include_video("missing").is_err());
    }

    #[test]
    fn disqualifying_again_keeps_reason_and_date_unless_a_new_reason_is_given() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        add_scored_video(&db, "v1", "UC1", 10);
        db.exclude_video("v1", Some("sponsored fluff")).expect("exclude");
        // A fixed date in the past, so "unchanged" can be told from "rewritten now".
        db.conn
            .execute("UPDATE insights SET excluded_at = '2026-01-02T03:04:05+00:00' WHERE video_id = 'v1'", [])
            .expect("set date");
        let state = || {
            let meta = db.get_insight_meta("v1").expect("meta").expect("row");
            (meta.excluded_manual, meta.excluded_reason, meta.excluded_at)
        };
        let original = (true, Some("sponsored fluff".to_string()), Some("2026-01-02T03:04:05+00:00".to_string()));

        // Repeated without a reason, or with a blank one: nothing changes.
        db.exclude_video("v1", None).expect("exclude again");
        assert_eq!(state(), original);
        db.exclude_video("v1", Some("   ")).expect("exclude again, blank reason");
        assert_eq!(state(), original);

        // A new reason replaces the stored one; the date of the decision stays.
        db.exclude_video("v1", Some(" duplicate ")).expect("new reason");
        assert_eq!(state(), (true, Some("duplicate".to_string()), original.2.clone()));

        // After a restore the next disqualification starts afresh: no reason
        // left over, and a new date.
        db.include_video("v1").expect("include");
        db.exclude_video("v1", None).expect("exclude anew");
        let (excluded, reason, at) = state();
        assert!(excluded);
        assert_eq!(reason, None);
        assert!(at.is_some() && at != original.2);

        // A disqualified row without a date gets one.
        db.conn.execute("UPDATE insights SET excluded_at = NULL WHERE video_id = 'v1'", []).expect("clear date");
        db.exclude_video("v1", None).expect("exclude again");
        assert!(state().2.is_some());
    }

    #[test]
    fn a_disqualified_video_leaves_the_channel_average() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        for id in ["v1", "v2", "v3"] {
            add_scored_video(&db, id, "UC1", 10);
        }
        add_scored_video(&db, "bad", "UC1", 100);
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), Some(32.5));
        db.exclude_video("bad", None).expect("exclude");
        // exclude_video recalculated already: the stored score is the new one.
        assert_eq!(db.get_channel("UC1").expect("get").expect("exists").rolling_slop_score, Some(10.0));
        assert_eq!(channel_status(&db, "UC1"), "high_signal");
        db.include_video("bad").expect("include");
        assert_eq!(db.get_channel("UC1").expect("get").expect("exists").rolling_slop_score, Some(32.5));
    }

    // ----- channel average ---------------------------------------------------

    fn add_judged_video(db: &Database, id: &str, channel_id: &str, slop: u32, keep: bool, verdict: &str, info: &str) {
        add_video(db, id, channel_id);
        db.save_insight_judged(&insight(id, slop, keep, &[]), &judged(Some(verdict), Some(info)))
            .expect("save judged");
    }

    #[test]
    fn short_videos_neither_mute_nor_demote_a_channel() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        // Two good long videos, published first...
        add_judged_video(&db, "a-long1", "UC1", 20, true, "ok", "sufficient");
        add_judged_video(&db, "a-long2", "UC1", 30, true, "ok", "sufficient");
        // ...then five shorts with too little speech. Their stored score is
        // whatever the analyzer wrote; it must not matter.
        for n in 0..5 {
            add_judged_video(&db, &format!("z-short{}", n), "UC1", 100, false, "ok", "insufficient");
        }
        // The average comes from the two good ones only, although the five
        // shorts are the five newest videos of the channel.
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), Some(25.0));
        // Two scored videos are below the minimum: the status stays as it is.
        assert_eq!(channel_status(&db, "UC1"), "active");
        let health = db.list_channels_with_health().expect("health");
        assert_eq!(health[0].avg_slop_score, Some(25.0));
        assert_eq!((health[0].kept_count, health[0].insufficient_count), (2, 5));
        assert_eq!(health[0].status, "active");

        // A third good video makes the channel high_signal; five more shorts
        // on top do not take that away again.
        add_judged_video(&db, "a-long3", "UC1", 10, true, "ok", "sufficient");
        assert_eq!(channel_status(&db, "UC1"), "high_signal");
        for n in 5..10 {
            add_judged_video(&db, &format!("z-short{}", n), "UC1", 100, false, "ok", "insufficient");
        }
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), Some(20.0));
        assert_eq!(channel_status(&db, "UC1"), "high_signal");
    }

    #[test]
    fn a_channel_of_only_short_videos_has_no_average_and_keeps_its_status() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        for n in 0..5 {
            add_judged_video(&db, &format!("s{}", n), "UC1", 100, false, "ok", "insufficient");
        }
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), None);
        assert_eq!(channel_status(&db, "UC1"), "active");
        assert_eq!(db.get_channel("UC1").expect("get").expect("exists").rolling_slop_score, None);
    }

    #[test]
    fn the_average_is_cleared_when_no_video_counts_any_more() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        for id in ["v1", "v2", "v3"] {
            add_scored_video(&db, id, "UC1", 85);
        }
        let stored = || {
            let c = db.get_channel("UC1").expect("get").expect("exists");
            (c.rolling_slop_score, c.status)
        };
        assert_eq!(stored(), (Some(85.0), "muted".to_string()));

        // All three leave the average (here: disqualified one by one).
        for id in ["v1", "v2", "v3"] {
            db.exclude_video(id, None).expect("exclude");
        }
        // No score backed by zero videos, and the automatic mute is over.
        assert_eq!(stored(), (None, "active".to_string()));
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), None);
        let health = db.list_channels_with_health().expect("health");
        assert_eq!(health[0].avg_slop_score, None);
        assert_eq!(health[0].status, "active");
        assert_eq!((health[0].kept_count, health[0].excluded_count), (0, 3));

        // The same when a re-analysis finds too little to judge.
        db.include_video("v1").expect("include");
        assert_eq!(stored().0, Some(85.0));
        db.save_insight_judged(&insight("v1", 0, false, &[]), &judged(Some("ok"), Some("insufficient")))
            .expect("re-analysis");
        assert_eq!(stored(), (None, "active".to_string()));

        // A status set by hand survives; only the number goes.
        db.include_video("v2").expect("include");
        db.set_channel_status("UC1", "muted").expect("mute by hand");
        assert_eq!(stored(), (Some(85.0), "muted".to_string()));
        db.exclude_video("v2", None).expect("exclude");
        assert_eq!(stored(), (None, "muted".to_string()));
        let health = db.list_channels_with_health().expect("health");
        assert_eq!((health[0].avg_slop_score, health[0].status.as_str()), (None, "muted"));
    }

    #[test]
    fn topic_rejected_videos_do_not_enter_the_average() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        for id in ["v1", "v2", "v3"] {
            add_judged_video(&db, id, "UC1", 30, true, "ok", "sufficient");
        }
        add_judged_video(&db, "w-politics", "UC1", 100, false, "politics", "sufficient");
        add_judged_video(&db, "w-fringe", "UC1", 100, false, "fringe", "sufficient");
        add_judged_video(&db, "w-nonsense", "UC1", 100, false, "nonsense", "sufficient");
        assert_eq!(db.recalculate_channel_stats("UC1").expect("recalc"), Some(30.0));
        assert_eq!(channel_status(&db, "UC1"), "high_signal");

        // The setters recalculate too: a verdict recorded after the fact
        // takes a row out of the average at once.
        add_scored_video(&db, "x-late", "UC1", 100);
        assert_eq!(db.get_channel("UC1").expect("get").expect("exists").rolling_slop_score, Some(47.5));
        db.set_insight_verdict("x-late", "politics", Some("campaign video")).expect("verdict");
        assert_eq!(db.get_channel("UC1").expect("get").expect("exists").rolling_slop_score, Some(30.0));
        add_scored_video(&db, "y-late", "UC1", 100);
        db.set_insight_info_status("y-late", "insufficient", None).expect("info");
        assert_eq!(db.get_channel("UC1").expect("get").expect("exists").rolling_slop_score, Some(30.0));
    }

    // ----- judgment columns --------------------------------------------------

    #[test]
    fn a_judgment_is_stored_with_the_insight_and_replaced_by_the_next_one() {
        let (_dir, db) = test_db();
        add_video(&db, "v1", "UC1");
        let full = InsightJudgment {
            topic_verdict: Some("ok".to_string()),
            topic_reason: Some("  history of mathematics ".to_string()),
            info_status: Some("sufficient".to_string()),
            info_reason: Some(" ".to_string()),
            sections_total: Some(4),
            transcript_chars: Some(52_000),
        };
        db.save_insight_judged(&insight("v1", 30, true, &[]), &full).expect("save");
        let meta = db.get_insight_meta("v1").expect("meta").expect("row");
        assert_eq!(meta.topic_verdict.as_deref(), Some("ok"));
        assert_eq!(meta.topic_reason.as_deref(), Some("history of mathematics"));
        assert_eq!(meta.info_status.as_deref(), Some("sufficient"));
        assert_eq!(meta.info_reason, None);
        assert_eq!((meta.sections_total, meta.transcript_chars), (Some(4), Some(52_000)));
        assert_eq!(meta.whole_transcript_note().as_deref(), Some("whole transcript (4 sections)"));
        assert!(meta.kept);

        // A plain save describes a different run: nothing of the run before stays.
        db.save_insight(&insight("v1", 30, true, &[])).expect("save again");
        let meta = db.get_insight_meta("v1").expect("meta").expect("row");
        assert_eq!((meta.topic_verdict, meta.info_status), (None, None));
        assert_eq!((meta.sections_total, meta.transcript_chars), (None, None));

        // Unknown values are refused before anything is written.
        let bad_topic = InsightJudgment { topic_verdict: Some("history".to_string()), ..Default::default() };
        assert!(db.save_insight_judged(&insight("v1", 99, false, &[]), &bad_topic).is_err());
        let bad_info = InsightJudgment { info_status: Some("short".to_string()), ..Default::default() };
        assert!(db.save_insight_judged(&insight("v1", 99, false, &[]), &bad_info).is_err());
        assert_eq!(db.get_insight("v1").expect("get").expect("row").slop_score, 30);
    }

    #[test]
    fn verdict_and_info_setters_validate_and_need_a_row() {
        let (_dir, db) = test_db();
        add_video(&db, "v1", "UC1");
        db.save_insight(&insight("v1", 30, true, &[])).expect("save");

        db.set_insight_verdict("v1", "nonsense", Some(" prank compilation ")).expect("verdict");
        db.set_insight_info_status("v1", "insufficient", Some("12 words")).expect("info");
        let meta = db.get_insight_meta("v1").expect("meta").expect("row");
        assert_eq!(meta.topic_verdict.as_deref(), Some("nonsense"));
        assert_eq!(meta.topic_reason.as_deref(), Some("prank compilation"));
        assert_eq!(meta.info_status.as_deref(), Some("insufficient"));
        assert_eq!(meta.info_reason.as_deref(), Some("12 words"));
        assert!(!meta.kept);

        assert!(db.set_insight_verdict("v1", "history", None).is_err());
        assert!(db.set_insight_info_status("v1", "tiny", None).is_err());
        assert!(db.set_insight_verdict("missing", "ok", None).is_err());
        assert!(db.set_insight_info_status("missing", "sufficient", None).is_err());
    }

    fn transcript(video_id: &str, text: &str) -> CleanedTranscript {
        CleanedTranscript {
            video_id: video_id.to_string(),
            raw_char_count: text.len() + 10,
            cleaned_char_count: text.len(),
            stripped_ad_seconds: 0.0,
            stripped_segments_count: 0,
            text: text.to_string(),
        }
    }

    #[test]
    fn transcript_coverage_is_stored_per_transcript() {
        let (_dir, db) = test_db();
        add_video(&db, "v1", "UC1");
        assert_eq!(db.get_transcript_coverage("v1").expect("coverage"), None);
        assert!(db.set_transcript_coverage("v1", Some(600.0), Some(590.0), Some("builtin")).is_err());

        db.save_transcript(&transcript("v1", "some words")).expect("save transcript");
        assert_eq!(db.get_transcript_coverage("v1").expect("coverage"), Some(TranscriptCoverage::default()));

        db.set_transcript_coverage("v1", Some(600.0), Some(291.5), Some("subtitles")).expect("set");
        let coverage = db.get_transcript_coverage("v1").expect("coverage").expect("row");
        assert_eq!(coverage.video_duration_sec, Some(600.0));
        assert_eq!(coverage.covered_until_sec, Some(291.5));
        assert_eq!(coverage.source.as_deref(), Some("subtitles"));
        assert_eq!(coverage.video_coverage_pct(), Some(49));

        // The next tier replaces all three values; unknown ones are not kept from before.
        db.set_transcript_coverage("v1", None, Some(f64::NAN), Some("whisper")).expect("set");
        let coverage = db.get_transcript_coverage("v1").expect("coverage").expect("row");
        assert_eq!((coverage.video_duration_sec, coverage.covered_until_sec), (None, None));
        assert_eq!(coverage.source.as_deref(), Some("whisper"));
        assert!(db.set_transcript_coverage("v1", None, None, Some("telepathy")).is_err());

        let all = db.list_transcript_coverage().expect("list");
        assert_eq!(all.len(), 1);
        assert_eq!(all["v1"].source.as_deref(), Some("whisper"));
        // The text itself is untouched by the coverage columns.
        assert_eq!(db.get_transcript("v1").expect("get").expect("row").text, "some words");
    }

    // ----- one database ------------------------------------------------------

    #[test]
    fn the_app_folder_is_the_exe_folder_or_the_project_root_of_a_cargo_build() {
        let p = |s: &str| PathBuf::from(s);
        assert_eq!(app_root_for(&p("/apps/YTC/ytc.exe")), p("/apps/YTC"));
        assert_eq!(app_root_for(&p("/code/YTC/target/release/ytc.exe")), p("/code/YTC"));
        assert_eq!(app_root_for(&p("/code/YTC/target/debug/ytc")), p("/code/YTC"));
        assert_eq!(app_root_for(&p("/code/YTC/Target/Release/ytc.exe")), p("/code/YTC"));
        // Only exactly target/release and target/debug count.
        assert_eq!(app_root_for(&p("/apps/release/ytc.exe")), p("/apps/release"));
        assert_eq!(app_root_for(&p("/code/YTC/target/release/deps/ytc.exe")), p("/code/YTC/target/release/deps"));
        assert_eq!(app_root_for(&p("/code/YTC/target/ytc.exe")), p("/code/YTC/target"));
        assert_eq!(app_root_for(&p("ytc.exe")), p(""));
    }

    #[test]
    fn the_database_is_found_by_one_rule_only() {
        let root = Path::new("/apps/YTC");
        let default = root.join("data").join("ytc.db");
        assert_eq!(db_path_for(root, None), default);
        assert_eq!(db_path_for(root, Some("")), default);
        assert_eq!(db_path_for(root, Some("   ")), default);
        assert_eq!(db_path_for(root, Some("D:/elsewhere/my.db")), PathBuf::from("D:/elsewhere/my.db"));
        assert_eq!(db_path_for(root, Some("  D:/elsewhere/my.db ")), PathBuf::from("D:/elsewhere/my.db"));

        // Existing files in the working directory or next to the app folder
        // change nothing: the rule does not look at the file system at all.
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("ytc.db"), b"decoy").expect("write decoy");
        assert_eq!(db_path_for(dir.path(), None), dir.path().join("data").join("ytc.db"));
        assert!(!dir.path().join("data").exists());

        // Opening the chosen path creates folder and file there.
        let chosen = db_path_for(dir.path(), None);
        drop(Database::open(&chosen).expect("open creates the database"));
        assert!(chosen.is_file());
        assert_eq!(std::fs::read(dir.path().join("ytc.db")).expect("decoy"), b"decoy");
    }

    // ----- merge -------------------------------------------------------------

    fn db_in(dir: &tempfile::TempDir, name: &str) -> (PathBuf, Database) {
        let path = dir.path().join(name).join("ytc.db");
        let db = Database::open(&path).expect("open test db");
        (path, db)
    }

    fn set_analysed_at(db: &Database, video_id: &str, when: &str) {
        db.conn
            .execute("UPDATE insights SET updated_at = ?1 WHERE video_id = ?2", params![when, video_id])
            .expect("set updated_at");
    }

    fn heuristic_insight(video_id: &str, slop: u32) -> InsightResult {
        InsightResult {
            analyzer_type: "heuristic".to_string(),
            summary: format!("Heuristic summary of {}", video_id),
            ..insight(video_id, slop, false, &["Heuristic Topic"])
        }
    }

    fn llm_insight(video_id: &str, slop: u32, marker: &str) -> InsightResult {
        InsightResult { summary: marker.to_string(), ..insight(video_id, slop, true, &["LLM Topic"]) }
    }

    fn summary_of(db: &Database, video_id: &str) -> String {
        db.get_insight(video_id).expect("get insight").expect("insight exists").summary
    }

    /// Every row of every table, for "nothing changed" comparisons.
    fn dump(db: &Database) -> Vec<String> {
        let mut out = Vec::new();
        for (table, order) in [
            ("channels", "id"),
            ("videos", "id"),
            ("transcripts", "video_id"),
            ("insights", "video_id"),
            ("categories", "name"),
            ("topics", "name"),
            ("video_topics", "video_id, topic_name"),
        ] {
            let mut stmt = db.conn.prepare(&format!("SELECT * FROM {} ORDER BY {}", table, order)).expect("prepare");
            let count = stmt.column_count();
            let mut rows = stmt.query([]).expect("query");
            while let Some(row) = rows.next().expect("row") {
                let values = read_values(row, count).expect("values");
                // topics.created_at is the time of the rebuild, not data.
                if table == "topics" {
                    out.push(format!("{}: {:?}", table, values[0]));
                } else {
                    out.push(format!("{}: {:?}", table, values));
                }
            }
        }
        out
    }

    #[test]
    fn merge_adds_what_only_the_other_database_has() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_, current) = db_in(&dir, "current");
        let (other_path, other) = db_in(&dir, "other");

        add_channel(&current, "UC1");
        add_video(&current, "v1", "UC1");
        current.save_insight(&llm_insight("v1", 20, "mine")).expect("save");
        current.save_transcript(&transcript("v1", "transcript one")).expect("transcript");

        add_channel(&other, "UC2");
        add_video(&other, "v2", "UC2");
        other.update_video_status("v2", "processed").expect("status");
        other
            .save_insight_judged(&llm_insight("v2", 30, "theirs"), &judged(Some("ok"), Some("sufficient")))
            .expect("save");
        other.save_transcript(&transcript("v2", "transcript two")).expect("transcript");
        other.set_transcript_coverage("v2", Some(100.0), Some(99.0), Some("whisper")).expect("coverage");
        add_video(&other, "v3", "UC2"); // pending, no analysis yet
        drop(other);

        let stats = current.merge_from(&other_path).expect("merge");
        assert_eq!(
            stats,
            MergeStats {
                channels_added: 1,
                videos_added: 2,
                transcripts_added: 1,
                insights_added: 1,
                ..Default::default()
            }
        );
        assert!(!stats.is_noop());

        assert_eq!(current.list_channels().expect("channels").len(), 2);
        assert_eq!(current.get_video("v2").expect("get").expect("exists").status, "processed");
        assert_eq!(current.get_video("v3").expect("get").expect("exists").status, "pending");
        assert_eq!(summary_of(&current, "v1"), "mine");
        assert_eq!(summary_of(&current, "v2"), "theirs");
        assert_eq!(current.get_transcript("v2").expect("get").expect("exists").text, "transcript two");
        assert_eq!(
            current.get_transcript_coverage("v2").expect("coverage").expect("row").source.as_deref(),
            Some("whisper")
        );
        let meta = current.get_insight_meta("v2").expect("meta").expect("row");
        assert_eq!(meta.topic_verdict.as_deref(), Some("ok"));
        assert!(meta.kept);
        // Topic links were rebuilt for the imported row, in the same step.
        assert_eq!(current.list_all_topics().expect("topics"), vec![("LLM Topic".to_string(), 2)]);
        // The imported channel has its average.
        assert_eq!(current.get_channel("UC2").expect("get").expect("exists").rolling_slop_score, Some(30.0));
    }

    #[test]
    fn an_llm_analysis_beats_a_heuristic_one_in_both_directions() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_, current) = db_in(&dir, "current");
        let (other_path, other) = db_in(&dir, "other");
        for db in [&current, &other] {
            add_channel(db, "UC1");
            for id in ["up", "down", "later", "sooner", "same"] {
                add_video(db, id, "UC1");
            }
        }
        // "up": heuristic here, LLM there. The LLM row wins whatever the dates.
        current.save_insight(&heuristic_insight("up", 60)).expect("save");
        set_analysed_at(&current, "up", "2026-06-01T00:00:00+00:00");
        other.save_insight(&llm_insight("up", 15, "theirs")).expect("save");
        set_analysed_at(&other, "up", "2026-01-01T00:00:00+00:00");
        other.update_video_status("up", "processed").expect("status");
        // "down": LLM here, heuristic there. The heuristic row loses whatever the dates.
        current.save_insight(&llm_insight("down", 15, "mine")).expect("save");
        set_analysed_at(&current, "down", "2026-01-01T00:00:00+00:00");
        other.save_insight(&heuristic_insight("down", 60)).expect("save");
        set_analysed_at(&other, "down", "2026-06-01T00:00:00+00:00");
        // Equal rank: the later analysis wins, whichever side it is on.
        current.save_insight(&llm_insight("later", 40, "mine")).expect("save");
        set_analysed_at(&current, "later", "2026-03-01T00:00:00+00:00");
        other.save_insight(&llm_insight("later", 20, "theirs")).expect("save");
        set_analysed_at(&other, "later", "2026-03-02T00:00:00+00:00");
        current.save_insight(&llm_insight("sooner", 40, "mine")).expect("save");
        set_analysed_at(&current, "sooner", "2026-03-02T00:00:00+00:00");
        other.save_insight(&llm_insight("sooner", 20, "theirs")).expect("save");
        // An hour before, written with another offset: instants are compared, not text.
        set_analysed_at(&other, "sooner", "2026-03-02T01:00:00+02:00");
        // Identical timestamps: the current one stays.
        current.save_insight(&llm_insight("same", 40, "mine")).expect("save");
        set_analysed_at(&current, "same", "2026-03-01T00:00:00+00:00");
        other.save_insight(&llm_insight("same", 20, "theirs")).expect("save");
        set_analysed_at(&other, "same", "2026-03-01T00:00:00+00:00");
        drop(other);

        let stats = current.merge_from(&other_path).expect("merge");
        assert_eq!((stats.insights_added, stats.insights_replaced, stats.insights_kept_current), (0, 2, 3));
        assert_eq!((stats.videos_added, stats.videos_updated, stats.videos_unchanged), (0, 1, 4));
        assert_eq!((stats.channels_added, stats.channels_updated, stats.channels_unchanged), (0, 0, 1));

        assert_eq!(summary_of(&current, "up"), "theirs");
        assert_eq!(current.get_insight("up").expect("get").expect("row").analyzer_type, "llm:test-model");
        // The video followed its imported analysis.
        assert_eq!(current.get_video("up").expect("get").expect("exists").status, "processed");
        assert_eq!(summary_of(&current, "down"), "mine");
        assert_eq!(summary_of(&current, "later"), "theirs");
        assert_eq!(summary_of(&current, "sooner"), "mine");
        assert_eq!(summary_of(&current, "same"), "mine");
        // Topic links follow the rows that won.
        let heuristic_topic = current.get_topic_videos("Heuristic Topic").expect("topic");
        assert!(heuristic_topic.is_empty());
        assert_eq!(current.get_topic_videos("LLM Topic").expect("topic").len(), 5);
    }

    #[test]
    fn the_longer_transcript_stays() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_, current) = db_in(&dir, "current");
        let (other_path, other) = db_in(&dir, "other");
        for db in [&current, &other] {
            for id in ["grow", "keep", "equal"] {
                add_video(db, id, "UC1");
            }
        }
        current.save_transcript(&transcript("grow", "short")).expect("save");
        current.set_transcript_coverage("grow", Some(600.0), Some(60.0), Some("subtitles")).expect("coverage");
        other.save_transcript(&transcript("grow", "the whole talk, start to end")).expect("save");
        other.set_transcript_coverage("grow", Some(600.0), Some(598.0), Some("whisper")).expect("coverage");
        current.save_transcript(&transcript("keep", "the whole talk, start to end")).expect("save");
        other.save_transcript(&transcript("keep", "short")).expect("save");
        // Same number of characters, counted as characters and not as bytes.
        current.save_transcript(&transcript("equal", "abcd")).expect("save");
        other.save_transcript(&transcript("equal", "äöüß")).expect("save");
        // A transcript without any video row (possible in files written
        // without foreign key enforcement) cannot be shown and is not copied.
        other.conn.execute_batch("PRAGMA foreign_keys = OFF").expect("pragma");
        other.save_transcript(&transcript("orphan", "nobody owns this")).expect("save");
        drop(other);

        let stats = current.merge_from(&other_path).expect("merge");
        assert_eq!(
            (stats.transcripts_added, stats.transcripts_replaced, stats.transcripts_kept_current, stats.orphans_skipped),
            (0, 1, 2, 1)
        );
        let grown = current.get_transcript("grow").expect("get").expect("row");
        assert_eq!(grown.text, "the whole talk, start to end");
        assert_eq!(grown.cleaned_char_count, "the whole talk, start to end".len());
        // The coverage facts travel with the text they describe.
        let coverage = current.get_transcript_coverage("grow").expect("coverage").expect("row");
        assert_eq!((coverage.covered_until_sec, coverage.source.as_deref()), (Some(598.0), Some("whisper")));
        assert_eq!(current.get_transcript("keep").expect("get").expect("row").text, "the whole talk, start to end");
        assert_eq!(current.get_transcript("equal").expect("get").expect("row").text, "abcd");
        assert!(current.get_transcript("orphan").expect("get").is_none());
    }

    #[test]
    fn manual_mutes_and_disqualifications_survive_a_merge_from_either_side() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_, current) = db_in(&dir, "current");
        let (other_path, other) = db_in(&dir, "other");
        for db in [&current, &other] {
            for ch in ["UCmine", "UCtheirs", "UCauto"] {
                add_channel(db, ch);
            }
            for (id, ch) in [("m1", "UCmine"), ("m2", "UCmine"), ("m3", "UCmine"), ("t1", "UCtheirs"), ("x1", "UCauto"), ("x2", "UCauto")] {
                add_video(db, id, ch);
            }
        }
        // Muted by hand here; the other database knows the channel as a good one.
        current.set_channel_status("UCmine", "muted").expect("mute");
        for id in ["m1", "m2", "m3"] {
            other.save_insight(&llm_insight(id, 5, "theirs")).expect("save");
        }
        // Muted by hand there; automatic and active here.
        other.set_channel_status("UCtheirs", "muted").expect("mute");
        current.save_insight(&llm_insight("t1", 5, "mine")).expect("save");

        // Disqualified here, and the other side has the better analysis.
        current.save_insight(&heuristic_insight("x1", 50)).expect("save");
        current.exclude_video("x1", Some("my reason")).expect("exclude");
        other.save_insight(&llm_insight("x1", 5, "theirs")).expect("save");
        // Disqualified there, and this side has the better analysis.
        current.save_insight(&llm_insight("x2", 5, "mine")).expect("save");
        other.save_insight(&heuristic_insight("x2", 50)).expect("save");
        other.exclude_video("x2", Some("their reason")).expect("exclude");
        drop(other);

        let stats = current.merge_from(&other_path).expect("merge");
        assert_eq!((stats.channels_added, stats.channels_updated, stats.channels_unchanged), (0, 1, 2));
        assert_eq!((stats.insights_added, stats.insights_replaced, stats.insights_kept_current), (3, 1, 1));
        assert_eq!(stats.exclusions_carried, 1);

        // Three imported top scores do not un-mute what the owner muted.
        assert_eq!(channel_status(&current, "UCmine"), "muted");
        assert!(current.is_channel_status_manual("UCmine").expect("flag"));
        assert_eq!(current.get_channel("UCmine").expect("get").expect("exists").rolling_slop_score, Some(5.0));
        // The other side's manual mute arrived, as a manual one.
        assert_eq!(channel_status(&current, "UCtheirs"), "muted");
        assert!(current.is_channel_status_manual("UCtheirs").expect("flag"));
        assert!(!current.is_channel_status_manual("UCauto").expect("flag"));

        let x1 = current.get_insight_meta("x1").expect("meta").expect("row");
        assert_eq!(summary_of(&current, "x1"), "theirs");
        assert!(x1.excluded_manual && !x1.kept);
        assert_eq!(x1.excluded_reason.as_deref(), Some("my reason"));
        assert!(x1.excluded_at.is_some());
        let x2 = current.get_insight_meta("x2").expect("meta").expect("row");
        assert_eq!(summary_of(&current, "x2"), "mine");
        assert!(x2.excluded_manual && !x2.kept);
        assert_eq!(x2.excluded_reason.as_deref(), Some("their reason"));
        let excluded: Vec<String> = current.list_excluded().expect("list").into_iter().map(|e| e.video_id).collect();
        assert_eq!(sorted(excluded), vec!["x1", "x2"]);
    }

    #[test]
    fn merging_the_same_database_twice_changes_nothing_the_second_time() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_, current) = db_in(&dir, "current");
        let (other_path, other) = db_in(&dir, "other");
        add_channel(&current, "UC1");
        add_video(&current, "shared", "UC1");
        current.save_insight(&heuristic_insight("shared", 60)).expect("save");
        current.save_transcript(&transcript("shared", "short")).expect("save");

        add_channel(&other, "UC1");
        add_channel(&other, "UC2");
        other.set_channel_status("UC2", "muted").expect("mute");
        add_video(&other, "shared", "UC1");
        other.save_insight(&llm_insight("shared", 10, "theirs")).expect("save");
        other.save_transcript(&transcript("shared", "a much longer transcript")).expect("save");
        other.exclude_video("shared", Some("their reason")).expect("exclude");
        for id in ["n1", "n2", "n3"] {
            add_video(&other, id, "UC2");
            other.save_insight(&llm_insight(id, 90, "theirs")).expect("save");
            other.save_transcript(&transcript(id, "text")).expect("save");
        }
        drop(other);

        let first = current.merge_from(&other_path).expect("first merge");
        assert!(!first.is_noop());
        assert_eq!((first.insights_added, first.insights_replaced), (3, 1));
        let after_first = dump(&current);

        let second = current.merge_from(&other_path).expect("second merge");
        assert!(second.is_noop(), "{}", second);
        assert_eq!(
            second,
            MergeStats {
                channels_unchanged: 2,
                videos_unchanged: 4,
                transcripts_kept_current: 4,
                insights_kept_current: 4,
                ..Default::default()
            }
        );
        assert_eq!(dump(&current), after_first);
    }

    #[test]
    fn merge_refuses_what_is_not_another_library() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (current_path, current) = db_in(&dir, "current");
        add_video(&current, "v1", "UC1");
        let before = dump(&current);

        let err = current.merge_from(&dir.path().join("nowhere.db")).expect_err("missing file");
        assert!(err.to_string().contains("does not exist"), "{}", err);
        let err = current.merge_from(&current_path).expect_err("the open database itself");
        assert!(err.to_string().contains("already open"), "{}", err);

        let foreign = dir.path().join("foreign.db");
        Connection::open(&foreign)
            .expect("create")
            .execute_batch("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);")
            .expect("schema");
        let empty = dir.path().join("empty.db");
        std::fs::write(&empty, b"").expect("write");
        let garbage = dir.path().join("garbage.db");
        std::fs::write(&garbage, b"this is not sqlite at all, just text in a file").expect("write");
        for other in [&foreign, &empty, &garbage] {
            let bytes = std::fs::read(other).expect("read");
            let err = current.merge_from(other).expect_err("not a library");
            assert!(err.to_string().ends_with(NOT_A_DATABASE), "{}", err);
            assert_eq!(std::fs::read(other).expect("read"), bytes, "the other file is only read");
        }

        assert_eq!(dump(&current), before);
    }

    #[test]
    fn merge_helpers_rank_and_compare_as_documented() {
        assert_eq!(analysis_rank("llm:any-model"), 1);
        assert_eq!(analysis_rank("info:insufficient"), 1);
        assert_eq!(analysis_rank("heuristic"), 0);

        assert!(is_later("2026-03-02T00:00:00+00:00", "2026-03-01T00:00:00+00:00"));
        assert!(!is_later("2026-03-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"));
        assert!(is_later("2026-03-01T00:00:00Z", "2026-03-01T01:00:00+02:00"));
        // No usable timestamp on either side: the current row stays.
        assert!(!is_later("yesterday", "2026-03-01T00:00:00+00:00"));
        assert!(!is_later("2026-03-01T00:00:00+00:00", ""));

        assert_eq!(merge_select("categories", MERGE_CATEGORY_COLUMNS), "SELECT name, description, created_at FROM categories");
        assert!(merge_col(MERGE_INSIGHT_COLUMNS, "excluded_manual").is_ok());
        assert!(merge_col(MERGE_INSIGHT_COLUMNS, "no_such_column").is_err());
    }

    // ----- one layout ----------------------------------------------------------

    const NOT_A_DATABASE: &str = "This file is not a YT-Wikinator database of this version.";

    #[test]
    fn a_new_file_gets_the_layout_and_its_mark_and_opens_again() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (path, db) = db_in(&dir, "fresh");
        assert_eq!(schema_state(&db.conn).expect("state").0, SCHEMA_VERSION);
        add_channel(&db, "UC1");
        drop(db);
        let again = Database::open(&path).expect("open again");
        assert_eq!(again.list_channels().expect("channels").len(), 1);

        // A file of zero bytes is a new database too.
        let blank = dir.path().join("blank.db");
        std::fs::write(&blank, b"").expect("write");
        let db = Database::open(&blank).expect("open blank file");
        assert_eq!(schema_state(&db.conn).expect("state").0, SCHEMA_VERSION);
    }

    #[test]
    fn a_file_with_another_content_is_refused_and_left_as_it_is() {
        let dir = tempfile::tempdir().expect("temp dir");

        let foreign = dir.path().join("foreign.db");
        Connection::open(&foreign)
            .expect("create")
            .execute_batch("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);")
            .expect("schema");
        // The right tables under another mark.
        let marked = dir.path().join("marked.db");
        Connection::open(&marked)
            .expect("create")
            .execute_batch(&format!("{} PRAGMA user_version = {};", SCHEMA_SQL, SCHEMA_VERSION + 1))
            .expect("schema");
        let garbage = dir.path().join("garbage.db");
        std::fs::write(&garbage, b"this is not sqlite at all, just text in a file").expect("write");

        for path in [&foreign, &marked, &garbage] {
            let bytes = std::fs::read(path).expect("read");
            let err = Database::open(path).err().expect("refused");
            assert_eq!(err.to_string(), format!("{}: {}", path.display(), NOT_A_DATABASE));
            assert_eq!(std::fs::read(path).expect("read"), bytes, "{} was changed", path.display());
        }
    }

    fn categorised(video_id: &str, category: &str) -> InsightResult {
        InsightResult { category: Some(category.to_string()), ..insight(video_id, 20, true, &[]) }
    }

    #[test]
    fn a_new_database_has_no_categories() {
        let (_dir, db) = test_db();
        assert!(db.list_categories().expect("list").is_empty());
        assert!(db.list_categories_with_counts().expect("counts").is_empty());
    }

    #[test]
    fn categories_are_added_listed_by_name_and_unique_whatever_their_case() {
        let (_dir, db) = test_db();
        let added = db.add_category("  Medieval   History ", Some(" castles,\n guilds and trade ")).expect("add");
        assert_eq!(added.name, "Medieval History");
        assert_eq!(added.description.as_deref(), Some("castles, guilds and trade"));
        db.add_category("baking", None).expect("add");
        db.add_category("Astronomy", Some("   ")).expect("add");

        let listed = db.list_categories().expect("list");
        let names: Vec<&str> = listed.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Astronomy", "baking", "Medieval History"]);
        assert_eq!(listed[0].description, None);

        let err = db.add_category("BAKING", None).expect_err("duplicate");
        assert!(err.to_string().contains("'baking' already exists"), "{}", err);
        assert_eq!(db.list_categories().expect("list").len(), 3);
    }

    #[test]
    fn unusable_category_names_and_descriptions_are_refused() {
        let (_dir, db) = test_db();
        let too_long = "x".repeat(CATEGORY_NAME_MAX_CHARS + 1);
        for bad in ["", "   ", "null", "NULL", "bell\u{7}", too_long.as_str()] {
            assert!(db.add_category(bad, None).is_err(), "{:?}", bad);
        }
        let long_text = "y".repeat(CATEGORY_DESCRIPTION_MAX_CHARS + 1);
        assert!(db.add_category("Baking", Some(&long_text)).is_err());
        assert!(db.list_categories().expect("list").is_empty());
        // The limits themselves are fine.
        db.add_category(&"x".repeat(CATEGORY_NAME_MAX_CHARS), Some(&"y".repeat(CATEGORY_DESCRIPTION_MAX_CHARS)))
            .expect("at the limit");
    }

    #[test]
    fn an_insight_keeps_only_a_category_the_user_has_defined() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.add_category("Baking", None).expect("add");
        for id in ["known", "unknown", "none"] {
            add_video(&db, id, "UC1");
        }
        db.save_insight(&categorised("known", "Baking")).expect("save");
        db.save_insight(&categorised("unknown", "Gardening")).expect("save");
        db.save_insight(&insight("none", 20, true, &[])).expect("save");

        let category = |id: &str| db.get_insight(id).expect("get").expect("row").category;
        assert_eq!(category("known").as_deref(), Some("Baking"));
        assert_eq!(category("unknown"), None);
        assert_eq!(category("none"), None);

        // Every reader reports the same value.
        let library = db.list_library(true, 10).expect("library");
        let listed = library.iter().find(|(v, _)| v.id == "known").and_then(|(_, i)| i.clone()).expect("listed");
        assert_eq!(listed.category.as_deref(), Some("Baking"));
        let kept = db.get_all_kept_videos_with_details().expect("kept");
        let with_category: Vec<&str> =
            kept.iter().filter(|(_, _, i)| i.category.is_some()).map(|(v, _, _)| v.id.as_str()).collect();
        assert_eq!(with_category, ["known"]);
        let hits = db.search("Baking").expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].video_id.as_str(), hits[0].category.as_deref()), ("known", Some("Baking")));

        // A new analysis replaces the category like every other field.
        db.save_insight(&insight("known", 20, true, &[])).expect("save again");
        assert_eq!(category("known"), None);
    }

    #[test]
    fn removing_a_category_keeps_its_insights_and_clears_their_category() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.add_category("Baking", Some("bread and pastry")).expect("add");
        db.add_category("Astronomy", None).expect("add");
        for (id, category) in [("b1", "Baking"), ("b2", "Baking"), ("a1", "Astronomy")] {
            add_video(&db, id, "UC1");
            db.save_insight(&categorised(id, category)).expect("save");
        }
        // Not kept: counted by the removal, not by the kept count.
        let mut slop = categorised("b2", "Baking");
        slop.is_worth_keeping = false;
        db.save_insight(&slop).expect("save");

        let counts: Vec<(String, usize)> = db
            .list_categories_with_counts()
            .expect("counts")
            .into_iter()
            .map(|(c, n)| (c.name, n))
            .collect();
        assert_eq!(counts, [("Astronomy".to_string(), 1), ("Baking".to_string(), 1)]);

        assert_eq!(db.remove_category(" baking ").expect("remove"), ("Baking".to_string(), 2));
        let names: Vec<String> = db.list_categories().expect("list").into_iter().map(|c| c.name).collect();
        assert_eq!(names, ["Astronomy"]);
        for id in ["b1", "b2"] {
            let row = db.get_insight(id).expect("get").expect("the insight is still there");
            assert_eq!(row.category, None);
            assert_eq!(row.summary, format!("Summary of {}", id));
        }
        assert!(db.get_insight_meta("b1").expect("meta").expect("row").kept);
        assert_eq!(db.get_insight("a1").expect("get").expect("row").category.as_deref(), Some("Astronomy"));

        let err = db.remove_category("Baking").expect_err("already gone");
        assert!(err.to_string().contains("does not exist"), "{}", err);
    }

    #[test]
    fn a_merge_takes_over_categories_and_drops_names_nobody_defined() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_, current) = db_in(&dir, "current");
        let (other_path, other) = db_in(&dir, "other");
        current.add_category("Baking", Some("mine")).expect("add");
        other.add_category("baking", Some("theirs")).expect("add");
        other.add_category("Astronomy", Some("stars")).expect("add");
        add_channel(&other, "UC1");
        for (id, category) in [("m1", "baking"), ("m2", "Astronomy")] {
            add_video(&other, id, "UC1");
            other.save_insight(&categorised(id, category)).expect("save");
        }
        // A name without a categories row, as a hand-edited file could hold it.
        add_video(&other, "m3", "UC1");
        other.save_insight(&insight("m3", 20, true, &[])).expect("save");
        other.conn.execute("UPDATE insights SET category = 'Gardening' WHERE video_id = 'm3'", []).expect("edit");
        drop(other);

        let stats = current.merge_from(&other_path).expect("merge");
        assert_eq!((stats.categories_added, stats.insights_added), (1, 3));
        let listed = current.list_categories().expect("list");
        let seen: Vec<(&str, Option<&str>)> = listed.iter().map(|c| (c.name.as_str(), c.description.as_deref())).collect();
        assert_eq!(seen, [("Astronomy", Some("stars")), ("Baking", Some("mine"))]);

        let category = |id: &str| current.get_insight(id).expect("get").expect("row").category;
        // The spelling of this database is used.
        assert_eq!(category("m1").as_deref(), Some("Baking"));
        assert_eq!(category("m2").as_deref(), Some("Astronomy"));
        assert_eq!(category("m3"), None);

        let second = current.merge_from(&other_path).expect("second merge");
        assert!(second.is_noop(), "{:?}", second);
    }

    /// (category, set by hand) of a stored insight.
    fn category_of(db: &Database, video_id: &str) -> (Option<String>, bool) {
        let category = db.get_insight(video_id).expect("get").expect("row").category;
        let manual = db.get_insight_meta(video_id).expect("meta").expect("row").category_manual;
        (category, manual)
    }

    #[test]
    fn a_category_set_by_hand_survives_a_new_analysis() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.add_category("Baking", None).expect("add");
        db.add_category("Astronomy", None).expect("add");
        add_video(&db, "v1", "UC1");
        db.save_insight(&categorised("v1", "Astronomy")).expect("save");
        assert_eq!(category_of(&db, "v1"), (Some("Astronomy".to_string()), false));

        // The name matches whatever its case and is stored in the user's spelling.
        assert_eq!(db.set_video_category("v1", Some("  bAKING ")).expect("set").as_deref(), Some("Baking"));
        assert_eq!(category_of(&db, "v1"), (Some("Baking".to_string()), true));

        // A new analysis replaces everything else and leaves category and mark.
        let mut again = categorised("v1", "Astronomy");
        again.slop_score = 33;
        again.summary = "A new summary".to_string();
        db.save_insight(&again).expect("save again");
        let stored = db.get_insight("v1").expect("get").expect("row");
        assert_eq!((stored.slop_score, stored.summary.as_str()), (33, "A new summary"));
        assert_eq!(category_of(&db, "v1"), (Some("Baking".to_string()), true));
        // So does one that names no category at all.
        db.save_insight(&insight("v1", 20, true, &[])).expect("save without");
        assert_eq!(category_of(&db, "v1"), (Some("Baking".to_string()), true));
    }

    #[test]
    fn no_category_set_by_hand_sticks_and_auto_hands_the_choice_back() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.add_category("Baking", None).expect("add");
        add_video(&db, "v1", "UC1");
        db.save_insight(&categorised("v1", "Baking")).expect("save");

        assert_eq!(db.set_video_category("v1", None).expect("none"), None);
        assert_eq!(category_of(&db, "v1"), (None, true));
        db.save_insight(&categorised("v1", "Baking")).expect("save again");
        assert_eq!(category_of(&db, "v1"), (None, true));

        // Handing it back clears the mark only; the next analysis decides.
        db.set_video_category("v1", Some("Baking")).expect("by hand");
        db.set_video_category_auto("v1").expect("auto");
        assert_eq!(category_of(&db, "v1"), (Some("Baking".to_string()), false));
        db.save_insight(&insight("v1", 20, true, &[])).expect("save without");
        assert_eq!(category_of(&db, "v1"), (None, false));
        // Not an error on a video that was never set by hand.
        db.set_video_category_auto("v1").expect("auto again");
    }

    #[test]
    fn setting_a_category_by_hand_needs_an_insight_and_a_known_name() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        add_video(&db, "v1", "UC1");
        add_video(&db, "pending", "UC1");
        db.save_insight(&insight("v1", 20, true, &[])).expect("save");

        // Without categories the error says so.
        let err = db.set_video_category("v1", Some("Baking")).expect_err("no categories");
        assert!(err.to_string().contains("no categories are defined"), "{}", err);

        db.add_category("Baking", None).expect("add");
        db.add_category("Astronomy", None).expect("add");
        let err = db.set_video_category("v1", Some("Gardening")).expect_err("unknown name");
        assert_eq!(err.to_string(), "Category 'Gardening' does not exist. Your categories: Astronomy, Baking");
        assert_eq!(category_of(&db, "v1"), (None, false));

        for result in [
            db.set_video_category("pending", Some("Baking")).map(|_| ()),
            db.set_video_category("pending", None).map(|_| ()),
            db.set_video_category_auto("pending"),
        ] {
            assert!(result.expect_err("no insight").to_string().contains("has no analysis"));
        }
        assert!(db.get_insight("pending").expect("get").is_none());
    }

    #[test]
    fn renaming_a_category_takes_its_videos_along() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.add_category("Baking", Some("bread and pastry")).expect("add");
        db.add_category("Astronomy", None).expect("add");
        for (id, category) in [("b1", "Baking"), ("b2", "Baking"), ("a1", "Astronomy")] {
            add_video(&db, id, "UC1");
            db.save_insight(&categorised(id, category)).expect("save");
        }
        db.set_video_category("b2", Some("Baking")).expect("by hand");

        // Another category has the name, whatever its case: refused, nothing changes.
        let err = db.rename_category("baking", "ASTRONOMY").expect_err("taken");
        assert!(err.to_string().contains("'Astronomy' already exists"), "{}", err);
        let err = db.rename_category("Gardening", "Plants").expect_err("unknown");
        assert!(err.to_string().contains("Your categories: Astronomy, Baking"), "{}", err);
        assert!(db.rename_category("Baking", "  ").is_err());
        assert_eq!(category_of(&db, "b1").0.as_deref(), Some("Baking"));

        let renamed = db.rename_category(" baking ", " Bread  and Pastry ").expect("rename");
        assert_eq!(renamed, ("Baking".to_string(), "Bread and Pastry".to_string(), 2));
        let listed = db.list_categories().expect("list");
        let seen: Vec<(&str, Option<&str>)> = listed.iter().map(|c| (c.name.as_str(), c.description.as_deref())).collect();
        assert_eq!(seen, [("Astronomy", None), ("Bread and Pastry", Some("bread and pastry"))]);
        assert_eq!(category_of(&db, "b1"), (Some("Bread and Pastry".to_string()), false));
        // The mark "set by hand" stays with the video.
        assert_eq!(category_of(&db, "b2"), (Some("Bread and Pastry".to_string()), true));
        assert_eq!(category_of(&db, "a1").0.as_deref(), Some("Astronomy"));

        // Only the spelling of the same category: allowed.
        let respelled = db.rename_category("bread and pastry", "Bread And Pastry").expect("respell");
        assert_eq!(respelled.2, 2);
        assert_eq!(category_of(&db, "b1").0.as_deref(), Some("Bread And Pastry"));
    }

    #[test]
    fn a_category_description_is_replaced_or_removed() {
        let (_dir, db) = test_db();
        let added = db.add_category("Baking", Some("bread")).expect("add");
        let described = db.describe_category("BAKING", Some("  bread,\n pastry and ovens ")).expect("describe");
        assert_eq!(described.description.as_deref(), Some("bread, pastry and ovens"));
        assert_eq!((described.name.as_str(), &described.created_at), ("Baking", &added.created_at));
        assert_eq!(db.list_categories().expect("list"), [described]);

        for blank in [None, Some(""), Some("   ")] {
            db.describe_category("Baking", Some("something")).expect("describe");
            assert_eq!(db.describe_category("Baking", blank).expect("remove").description, None);
            assert_eq!(db.list_categories().expect("list")[0].description, None);
        }

        let long_text = "y".repeat(CATEGORY_DESCRIPTION_MAX_CHARS + 1);
        assert!(db.describe_category("Baking", Some(&long_text)).is_err());
        let err = db.describe_category("Gardening", Some("plants")).expect_err("unknown");
        assert!(err.to_string().contains("Your categories: Baking"), "{}", err);
    }

    #[test]
    fn removing_a_category_clears_the_mark_set_by_hand() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.add_category("Baking", None).expect("add");
        db.add_category("Astronomy", None).expect("add");
        for id in ["b1", "a1", "n1"] {
            add_video(&db, id, "UC1");
            db.save_insight(&insight(id, 20, true, &[])).expect("save");
        }
        db.set_video_category("b1", Some("Baking")).expect("by hand");
        db.set_video_category("a1", Some("Astronomy")).expect("by hand");
        db.set_video_category("n1", None).expect("none by hand");

        assert_eq!(db.remove_category("baking").expect("remove"), ("Baking".to_string(), 1));
        // The model chooses again for the video that was in it.
        assert_eq!(category_of(&db, "b1"), (None, false));
        db.add_category("Baking", None).expect("add again");
        db.save_insight(&categorised("b1", "Baking")).expect("save");
        assert_eq!(category_of(&db, "b1"), (Some("Baking".to_string()), false));
        // Other choices made by hand are not touched.
        assert_eq!(category_of(&db, "a1"), (Some("Astronomy".to_string()), true));
        assert_eq!(category_of(&db, "n1"), (None, true));
    }

    #[test]
    fn the_models_category_for_a_stored_video_changes_nothing_else_and_never_a_choice_made_by_hand() {
        let (_dir, db) = test_db();
        add_channel(&db, "UC1");
        db.add_category("Baking", None).expect("add");
        db.add_category("Astronomy", None).expect("add");
        for id in ["open", "sorted", "byhand", "nonebyhand"] {
            add_video(&db, id, "UC1");
            db.save_insight(&insight(id, 20, true, &["A Topic"])).expect("save");
        }
        add_video(&db, "heur", "UC1");
        db.save_insight(&InsightResult { analyzer_type: "heuristic".to_string(), ..insight("heur", 50, true, &[]) })
            .expect("save");
        add_video(&db, "short", "UC1");
        db.save_insight(&InsightResult { analyzer_type: "info:insufficient".to_string(), ..insight("short", 0, false, &[]) })
            .expect("save");
        db.save_insight(&categorised("sorted", "Astronomy")).expect("save");
        db.set_video_category("byhand", Some("Astronomy")).expect("by hand");
        db.set_video_category("nonebyhand", None).expect("none by hand");
        for (n, id) in ["open", "sorted", "byhand", "nonebyhand", "heur", "short"].iter().enumerate() {
            set_analysed_at(&db, id, &format!("2026-03-0{}T00:00:00+00:00", n + 1));
        }

        // Only LLM-judged videos that were not set by hand are worked on.
        assert_eq!(db.list_categorize_targets(false).expect("targets"), ["open"]);
        assert_eq!(db.list_categorize_targets(true).expect("targets"), ["open", "sorted"]);

        let before = dump(&db);
        assert!(db.set_model_category("open", Some("Baking")).expect("set"));
        let after = dump(&db);
        let changed: Vec<(&String, &String)> = before.iter().zip(&after).filter(|(a, b)| a != b).collect();
        assert_eq!(changed.len(), 1, "{:?}", changed);
        assert!(changed[0].0.contains("\"open\"") && changed[0].1.contains("Text(\"Baking\")"), "{:?}", changed);
        assert_eq!(category_of(&db, "open"), (Some("Baking".to_string()), false));

        // A name the user does not have is no category.
        assert!(db.set_model_category("open", Some("Gardening")).expect("set"));
        assert_eq!(category_of(&db, "open"), (None, false));
        assert!(db.set_model_category("sorted", None).expect("set"));
        assert_eq!(category_of(&db, "sorted"), (None, false));

        // A video the model was asked about and that fits no category is not
        // asked on every run: only --all takes it, until the categories change
        // or the owner hands the video back to the model.
        assert!(db.list_categorize_targets(false).expect("targets").is_empty());
        assert_eq!(db.list_categorize_targets(true).expect("targets"), ["open", "sorted"]);
        db.set_video_category_auto("sorted").expect("auto");
        assert_eq!(db.list_categorize_targets(false).expect("targets"), ["sorted"]);
        db.add_category("Gardening", None).expect("add");
        assert_eq!(db.list_categorize_targets(false).expect("targets"), ["open", "sorted"]);
        assert!(db.set_model_category("open", None).expect("set"));
        db.describe_category("Gardening", Some("Plants and soil")).expect("describe");
        assert_eq!(db.list_categorize_targets(false).expect("targets"), ["open", "sorted"]);
        assert!(db.set_model_category("open", None).expect("set"));
        db.rename_category("Gardening", "Garden").expect("rename");
        assert_eq!(db.list_categorize_targets(false).expect("targets"), ["open", "sorted"]);
        assert!(db.set_model_category("open", None).expect("set"));
        db.remove_category("Garden").expect("remove");
        assert_eq!(db.list_categorize_targets(false).expect("targets"), ["open", "sorted"]);

        // A choice made by hand is left alone, in both directions.
        let before = dump(&db);
        assert!(!db.set_model_category("byhand", Some("Baking")).expect("skipped"));
        assert!(!db.set_model_category("nonebyhand", Some("Baking")).expect("skipped"));
        assert!(!db.set_model_category("missing", Some("Baking")).expect("no row"));
        assert_eq!(dump(&db), before);
    }

    #[test]
    fn a_merge_keeps_categories_set_by_hand_on_both_sides() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_, current) = db_in(&dir, "current");
        let (other_path, other) = db_in(&dir, "other");
        for db in [&current, &other] {
            add_channel(db, "UC1");
            db.add_category("Baking", None).expect("add");
            db.add_category("Astronomy", None).expect("add");
        }
        other.add_category("Gardening", None).expect("add");

        // new1, new2: only in the other database, set by hand there.
        for id in ["new1", "new2"] {
            add_video(&other, id, "UC1");
            other.save_insight(&insight(id, 20, true, &[])).expect("save");
        }
        other.set_video_category("new1", Some("Gardening")).expect("by hand");
        other.set_video_category("new2", None).expect("none by hand");

        // both1: set by hand there only, and the analysis here is the later one.
        // both2: set by hand on both sides. both3: set by hand here only, and
        // the analysis there is the later one.
        for id in ["both1", "both2", "both3"] {
            for db in [&current, &other] {
                add_video(db, id, "UC1");
                db.save_insight(&categorised(id, "Baking")).expect("save");
            }
        }
        other.set_video_category("both1", Some("Astronomy")).expect("by hand");
        other.set_video_category("both2", Some("Astronomy")).expect("by hand");
        current.set_video_category("both2", None).expect("none by hand");
        current.set_video_category("both3", Some("Astronomy")).expect("by hand");
        for id in ["both1", "both2"] {
            set_analysed_at(&other, id, "2026-03-01T00:00:00+00:00");
            set_analysed_at(&current, id, "2026-03-02T00:00:00+00:00");
        }
        set_analysed_at(&other, "both3", "2026-03-02T00:00:00+00:00");
        set_analysed_at(&current, "both3", "2026-03-01T00:00:00+00:00");
        // A mark on a name nobody defined, as a hand-edited file could hold it.
        add_video(&other, "odd", "UC1");
        other.save_insight(&insight("odd", 20, true, &[])).expect("save");
        other
            .conn
            .execute("UPDATE insights SET category = 'Knitting', category_manual = 1 WHERE video_id = 'odd'", [])
            .expect("edit");
        drop(other);

        let stats = current.merge_from(&other_path).expect("merge");
        assert_eq!((stats.insights_added, stats.insights_replaced, stats.insights_kept_current), (3, 1, 2));
        assert_eq!(stats.manual_categories_carried, 1);
        assert!(stats.to_string().contains("Categories set by hand taken over: 1"), "{}", stats);

        assert_eq!(category_of(&current, "new1"), (Some("Gardening".to_string()), true));
        assert_eq!(category_of(&current, "new2"), (None, true));
        assert_eq!(category_of(&current, "both1"), (Some("Astronomy".to_string()), true));
        // The choice made in this database wins.
        assert_eq!(category_of(&current, "both2"), (None, true));
        assert_eq!(category_of(&current, "both3"), (Some("Astronomy".to_string()), true));
        // Not "no category, by hand": the model may choose for it.
        assert_eq!(category_of(&current, "odd"), (None, false));

        let second = current.merge_from(&other_path).expect("second merge");
        assert!(second.is_noop(), "{:?}", second);
    }

    #[test]
    fn every_column_is_part_of_the_merge() {
        // A column missing from the merge lists would be dropped by a merge.
        let (_dir, db) = test_db();
        for (table, columns) in [
            ("channels", MERGE_CHANNEL_COLUMNS),
            ("videos", MERGE_VIDEO_COLUMNS),
            ("transcripts", MERGE_TRANSCRIPT_COLUMNS),
            ("insights", MERGE_INSIGHT_COLUMNS),
            ("categories", MERGE_CATEGORY_COLUMNS),
        ] {
            // table_info takes no bound parameter; `table` is one of the literals above.
            let mut stmt = db.conn.prepare(&format!("PRAGMA table_info({})", table)).expect("prepare");
            let schema: Vec<String> =
                stmt.query_map([], |r| r.get(1)).expect("query").collect::<rusqlite::Result<_>>().expect("rows");
            assert_eq!(schema, columns.to_vec(), "columns of {}", table);
        }
    }
}
