//! What this computer has learned about YouTube's limits, kept across restarts.
//!
//! YouTube publishes no request limit and gives no warning: the only signal is
//! a refused transcript request ("429"). This module remembers that signal in a
//! small JSON file next to the library database:
//!
//! - until when transcript requests are left alone after a block, so a blocked
//!   address is not kept flagged by asking again for every video;
//! - the pause between videos, which grows after a block and shrinks again
//!   after clean requests;
//! - the pause that was in use when a block came out of a clean phase, so the
//!   program stays slower than that for a while.
//!
//! The rules are plain functions of the state and a clock value, so they can be
//! tested without waiting.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// File name of the memory, in the folder of the library database.
pub const PACING_FILE: &str = "youtube_pacing.json";

/// Transcript requests rest this long after the first block ...
const FIRST_REST_SECS: i64 = 30 * 60;
/// ... twice as long after every further block in a row, up to this.
const MAX_REST_SECS: i64 = 6 * 60 * 60;
/// How long the pause that led to a block is remembered.
const REMEMBER_BLOCK_PACE_SECS: i64 = 7 * 24 * 60 * 60;
/// Longest pause between videos.
pub const MAX_PAUSE_SECS: u64 = 300;
/// The pause after a block is at least twice this.
const MIN_PAUSE_AFTER_BLOCK_SECS: u64 = 10;

/// The stored memory. Times are Unix seconds; 0 means "never".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PacingState {
    /// Pause between videos learned so far, in seconds (0 = nothing learned).
    pub pause_secs: u64,
    /// No transcript request is made before this time.
    pub blocked_until: i64,
    /// Blocks since the last transcript that came through.
    pub blocks_in_a_row: u32,
    /// When the current run of blocks began.
    pub block_started_at: i64,
    /// When YouTube last refused a transcript request.
    pub last_block_at: i64,
    /// The pause in use when a block last came after clean requests.
    pub blocked_at_pause_secs: u64,
}

impl PacingState {
    /// The time until which transcript requests rest, when that is still ahead.
    pub fn resting_until(&self, now: i64) -> Option<i64> {
        (self.blocked_until > now).then_some(self.blocked_until)
    }

    /// The slowest pace YouTube has refused lately, with a margin: the pause
    /// does not go below this while the block is remembered.
    fn learned_floor(&self, now: i64, floor: u64) -> u64 {
        let remembered = self.last_block_at > 0 && now - self.last_block_at < REMEMBER_BLOCK_PACE_SECS;
        if remembered && self.blocked_at_pause_secs > 0 {
            floor.max((self.blocked_at_pause_secs.saturating_mul(3) / 2).min(MAX_PAUSE_SECS))
        } else {
            floor
        }
    }

    /// The pause to keep between two videos. `floor` is the pause the user set.
    pub fn pause(&self, now: i64, floor: u64) -> u64 {
        self.pause_secs.max(self.learned_floor(now, floor)).min(MAX_PAUSE_SECS.max(floor))
    }

    /// YouTube refused a transcript request while `pause_in_use` was the pause.
    pub fn note_block(&mut self, now: i64, pause_in_use: u64) {
        if self.blocks_in_a_row == 0 {
            // Only a block after clean requests says something about the pace:
            // a refused retry during a block says nothing new.
            self.block_started_at = now;
            self.blocked_at_pause_secs = pause_in_use;
        }
        self.blocks_in_a_row = self.blocks_in_a_row.saturating_add(1);
        self.last_block_at = now;
        let doublings = (self.blocks_in_a_row - 1).min(16);
        let rest = FIRST_REST_SECS.saturating_mul(1i64 << doublings).min(MAX_REST_SECS);
        self.blocked_until = now + rest;
        self.pause_secs = pause_in_use.max(MIN_PAUSE_AFTER_BLOCK_SECS).saturating_mul(2).min(MAX_PAUSE_SECS);
    }

    /// A transcript request came through.
    pub fn note_clean(&mut self, now: i64, floor: u64) {
        self.blocks_in_a_row = 0;
        self.blocked_until = 0;
        self.block_started_at = 0;
        if self.pause_secs > 0 {
            self.pause_secs = (self.pause_secs.saturating_mul(3) / 4).max(self.learned_floor(now, floor));
        }
    }
}

/// Reads and writes the memory file. Every change is written at once, so
/// separate `ytc` processes (crawl, process-video, the server) share it.
#[derive(Debug, Clone)]
pub struct PacingStore {
    path: PathBuf,
}

impl PacingStore {
    /// The memory that belongs to the library at `db_path`.
    pub fn for_db(db_path: &Path) -> Self {
        let dir = db_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
        Self { path: dir.join(PACING_FILE) }
    }

    /// The stored state; an empty one when the file is missing or unreadable.
    pub fn load(&self) -> PacingState {
        fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// Writes the state. A failure is reported and otherwise ignored: the
    /// memory is a help, not something a crawl may fail on.
    pub fn save(&self, state: &PacingState) {
        if let Err(e) = self.try_save(state) {
            eprintln!("  [Pacing] Could not store {}: {}", self.path.display(), e);
        }
    }

    fn try_save(&self, state: &PacingState) -> anyhow::Result<()> {
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    /// Records a refused transcript request and returns the new state.
    pub fn note_block(&self, now: i64, pause_in_use: u64) -> PacingState {
        let mut state = self.load();
        state.note_block(now, pause_in_use);
        self.save(&state);
        state
    }

    /// Records a transcript that came through. Writes only when that changes
    /// something, so a clean crawl does not rewrite the file for every video.
    pub fn note_clean(&self, now: i64, floor: u64) -> PacingState {
        let before = self.load();
        let mut state = before.clone();
        state.note_clean(now, floor);
        if state != before {
            self.save(&state);
        }
        state
    }
}

/// The current time in Unix seconds.
pub fn now_unix() -> i64 {
    chrono::Utc::now().timestamp()
}

/// A time of day for messages, in the computer's own time zone.
pub fn local_clock(unix: i64) -> String {
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "later".to_string())
}

/// `base` plus an irregular 1.0 to 2.0 seconds, in milliseconds: an exact
/// rhythm is the easiest thing to recognise as a script. `seed` is any
/// changing number (the clock's nanoseconds).
pub fn jittered_millis(base_secs: u64, seed: u32) -> u64 {
    base_secs.saturating_mul(1000) + 1000 + u64::from(seed % 1001)
}

/// A changing number for `jittered_millis`.
pub fn clock_seed() -> u32 {
    chrono::Utc::now().timestamp_subsec_nanos() / 1000
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_800_000_000;

    #[test]
    fn nothing_learned_means_the_users_pause_and_no_rest() {
        let state = PacingState::default();
        assert_eq!(state.pause(T0, 3), 3);
        assert_eq!(state.resting_until(T0), None);
        // A clean request with nothing learned leaves nothing to store.
        let mut clean = PacingState::default();
        clean.note_clean(T0, 3);
        assert_eq!(clean, PacingState::default());
    }

    #[test]
    fn a_block_rests_transcript_requests_and_each_further_block_rests_longer() {
        let mut state = PacingState::default();
        state.note_block(T0, 3);
        assert_eq!(state.resting_until(T0), Some(T0 + 30 * 60));
        assert_eq!(state.resting_until(T0 + 30 * 60), None, "the rest is over at its end");
        assert_eq!(state.pause_secs, 20);

        state.note_block(T0 + 1800, 20);
        assert_eq!(state.blocked_until, T0 + 1800 + 60 * 60);
        state.note_block(T0 + 5400, 40);
        assert_eq!(state.blocked_until, T0 + 5400 + 2 * 60 * 60);
        for i in 0..40 {
            state.note_block(T0 + 9000 + i, 300);
        }
        assert_eq!(state.blocked_until - state.last_block_at, MAX_REST_SECS, "the rest is capped");
        assert_eq!(state.pause_secs, MAX_PAUSE_SECS, "and so is the pause");
        assert_eq!(state.block_started_at, T0, "the run of blocks began with the first one");
    }

    #[test]
    fn only_a_block_after_clean_requests_is_remembered_as_a_pace() {
        let mut state = PacingState::default();
        state.note_block(T0, 4);
        state.note_block(T0 + 1800, 20);
        state.note_block(T0 + 5400, 40);
        assert_eq!(state.blocked_at_pause_secs, 4, "refused retries say nothing about the pace");

        // Clean again: the pause comes down, but not below 1.5 x the pace that was blocked.
        let mut now = T0 + 20_000;
        for _ in 0..30 {
            state.note_clean(now, 3);
            now += 60;
        }
        assert_eq!(state.pause_secs, 6);
        assert_eq!(state.pause(now, 3), 6);
        assert_eq!((state.blocks_in_a_row, state.blocked_until, state.block_started_at), (0, 0, 0));

        // After a week the blocked pace is forgotten.
        let later = T0 + 5400 + REMEMBER_BLOCK_PACE_SECS;
        assert_eq!(state.pause(later, 3), 6, "the stored pause itself only shrinks with clean requests");
        state.note_clean(later, 3);
        state.note_clean(later, 3);
        state.note_clean(later, 3);
        assert_eq!(state.pause(later, 3), 3);
    }

    #[test]
    fn the_users_own_pause_is_never_undercut() {
        let mut state = PacingState::default();
        state.note_block(T0, 3);
        assert_eq!(state.pause(T0, 30), 30);
        state.note_clean(T0 + 10, 30);
        assert_eq!(state.pause(T0 + 10, 30), 30);
        // A pause the user set above the cap is kept as it is.
        assert_eq!(PacingState::default().pause(T0, 600), 600);
    }

    #[test]
    fn the_memory_survives_a_restart_and_a_broken_file_counts_as_empty() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = PacingStore::for_db(&dir.path().join("ytc.db"));
        assert_eq!(store.load(), PacingState::default());

        let stored = store.note_block(T0, 3);
        let again = PacingStore::for_db(&dir.path().join("ytc.db"));
        assert_eq!(again.load(), stored);
        assert_eq!(again.load().resting_until(T0 + 60), Some(T0 + 1800));

        let cleared = again.note_clean(T0 + 4000, 3);
        assert_eq!(store.load(), cleared);
        assert_eq!(cleared.resting_until(T0 + 4000), None);

        fs::write(dir.path().join(PACING_FILE), "{ not json").expect("write");
        assert_eq!(store.load(), PacingState::default());
        // A file from a build with fewer fields is read, the rest is empty.
        fs::write(dir.path().join(PACING_FILE), r#"{"pause_secs": 12}"#).expect("write");
        assert_eq!(store.load().pause_secs, 12);
    }

    #[test]
    fn the_pause_between_videos_is_irregular_by_one_to_two_seconds() {
        assert_eq!(jittered_millis(3, 0), 4000);
        assert_eq!(jittered_millis(3, 1000), 5000);
        assert_eq!(jittered_millis(0, 500), 1500);
        for seed in [1u32, 77, 1001, 999_999] {
            let ms = jittered_millis(3, seed);
            assert!((4000..=5000).contains(&ms), "{}", ms);
        }
    }
}
