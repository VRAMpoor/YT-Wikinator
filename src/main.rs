// serde_json::json! expands recursively, once per key: the video objects of
// the web API (src/server.rs) have more keys than the default limit allows.
#![recursion_limit = "256"]

mod analyzer;
mod db;
mod extractor;
mod models;
mod pacing;
mod pipeline;
mod rss;
mod server;
mod sponsorblock;
mod wiki;

use clap::{Parser, Subcommand, ValueEnum};
use colored::*;
use db::Database;
use models::{Channel, InsightMeta, InsightResult, TranscriptCoverage, Video};
use pipeline::{IngestionPipeline, VideoOutcome};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shown instead of the stored rating for rows the word-count heuristic wrote.
const UNREVIEWED_LABEL: &str = "Unreviewed (heuristic)";

/// Folder inside the wiki folder that `wiki --prune` moves stale pages to.
const PRUNED_DIR_NAME: &str = ".ytc-pruned";

/// How often a running watch step looks at the stop request.
const STOP_POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// How long the watch loop gets to react to Ctrl+C before the process is ended
/// from the listener task. Several poll periods: a loop that has not reacted
/// by then is inside a blocking call (yt-dlp, Whisper) and cannot react.
const STOP_GRACE: std::time::Duration = std::time::Duration::from_millis(500);

#[derive(Parser)]
#[command(
    name = "ytc",
    author = "VRAMpoor",
    version = env!("CARGO_PKG_VERSION"),
    about = "YouTube librarian and wiki creator: fetches the complete transcript of each video of the tracked channels, removes SponsorBlock segments, has a local LLM read all of it and judge topic and quality separately, and writes the kept insights to a Markdown wiki",
    long_about = "YouTube librarian and wiki creator: fetches the complete transcript of each video of the tracked channels, removes SponsorBlock segments, has a local LLM read all of it and writes the kept insights to a Markdown wiki.\n\nA video is kept when all of this holds: its topic is acceptable (any subject is, except fringe, nonsense and partisan politics), its slop score (quality only: substance versus padding and bait, 0 = dense, 100 = slop) is at or below --max-slop, its transcript holds enough information to judge, and it was not disqualified by hand with 'ytc exclude'.",
    after_help = "The options --db, --llm-url, --llm-key, --llm-model, --max-slop and --wiki-dir may be given before or after the command, e.g. 'ytc --max-slop 30 crawl' or 'ytc crawl --max-slop 30'."
)]
struct Cli {
    // The long options below are `global`: clap then accepts them after the
    // subcommand as well, which is where users put them.
    #[arg(
        long,
        global = true,
        value_name = "FILE",
        help = "SQLite database file. Without it: the YTC_DB env var, else <app folder>/data/ytc.db, where the app folder is the folder that contains ytc.exe (for a cargo build in target/release or target/debug: the project folder). Nothing else is searched. The file is created if it does not exist; the chosen path is printed at startup"
    )]
    db: Option<PathBuf>,

    // Not global and separate from --db: 'crawl', 'discover' and 'watch' use
    // -d for their delay, so the short form can only exist before the command.
    #[arg(
        short = 'd',
        value_name = "FILE",
        conflicts_with = "db",
        help = "Short form of --db. Only before the command: after 'crawl', 'discover' and 'watch', -d is the delay"
    )]
    db_short: Option<PathBuf>,

    #[arg(long, global = true, env = "YTC_LLM_URL", default_value = "http://127.0.0.1:8080/v1", help = "OpenAI-compatible LLM endpoint (e.g. http://127.0.0.1:8080/v1)")]
    llm_url: Option<String>,

    #[arg(long, global = true, env = "YTC_LLM_KEY", help = "API key for the LLM endpoint (optional)")]
    llm_key: Option<String>,

    #[arg(long, global = true, env = "YTC_LLM_MODEL", default_value = "Ternary-Bonsai-2-27B-PTQ1_0", help = "Model name for the LLM endpoint")]
    llm_model: String,

    #[arg(
        long,
        global = true,
        env = "YTC_MAX_SLOP",
        default_value_t = pipeline::KEEP_THRESHOLD,
        value_parser = clap::value_parser!(u32).range(0..=100),
        value_name = "N",
        help = "Quality cutoff, 0-100: a video can only be kept when its slop score (quality only, independent of the subject) is at or below this value. Topic, amount of information and a disqualification by hand are checked separately. Applies to videos analysed from now on; stored rows keep their flag until they are re-analysed"
    )]
    max_slop: u32,

    #[arg(
        long,
        global = true,
        env = "YTC_WIKI_DIR",
        default_value = pipeline::DEFAULT_WIKI_DIR,
        value_name = "PATH",
        help = "Wiki directory used by 'wiki', 'watch', 'crawl --wiki' and the web UI. 'wiki --output' and 'watch --wiki' override it for that run"
    )]
    wiki_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

/// Target of `set-status`. `high_signal` is not offered: only the automatic
/// rule awards it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum StatusChoice {
    Active,
    Muted,
    Auto,
}

/// What `ytc category` does with the user's own subject categories.
#[derive(Subcommand)]
enum CategoryAction {
    #[command(about = "Define a category. From then on the LLM assigns each video it analyses to one of your categories, or to none")]
    Add {
        #[arg(help = "Name of the category, for example \"Baking\" or \"Astronomy\"")]
        name: String,

        #[arg(short, long, help = "What belongs in the category, in a sentence. The LLM reads it next to the name")]
        description: Option<String>,
    },

    #[command(about = "Remove a category. Its videos stay in the library and are in no category afterwards")]
    Remove {
        #[arg(help = "Name of the category (upper and lower case do not matter)")]
        name: String,
    },

    #[command(about = "Give a category another name. Its videos follow, also those put into it by hand")]
    Rename {
        #[arg(help = "Current name of the category (upper and lower case do not matter)")]
        old: String,

        #[arg(help = "New name. Refused when another category already has it")]
        new: String,
    },

    #[command(about = "Replace the description of a category, the sentence the LLM reads next to its name. Without a text the description is removed")]
    Describe {
        #[arg(help = "Name of the category (upper and lower case do not matter)")]
        name: String,

        #[arg(help = "What belongs in the category, in a sentence. Leave it out (or give \"\") to remove the description")]
        text: Option<String>,
    },

    #[command(about = "List your categories with the number of kept videos in each")]
    List,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Add a YouTube channel to the watchlist (URL, @handle, or channel ID)")]
    AddChannel {
        #[arg(allow_hyphen_values = true, help = "Channel handle (@channel), URL, or UC channel ID")]
        input: String,
    },

    #[command(about = "List tracked channels with status, average slop score and kept / quality-judged counts")]
    ListChannels,

    #[command(about = "Toggle a channel between active and muted. A status set by hand stays until it is changed again or handed back with 'set-status <channel> auto'")]
    ToggleChannel {
        #[arg(allow_hyphen_values = true, help = "Channel ID or exact title of a tracked channel (no request), otherwise a handle or URL that is resolved through YouTube")]
        input: String,
    },

    #[command(about = "Set the status of a tracked channel by hand (active or muted), or hand it back to the automatic rule (auto)")]
    SetStatus {
        #[arg(help = "Channel ID or exact title of a tracked channel (no request), otherwise a handle or URL that is resolved through YouTube")]
        channel: String,

        #[arg(value_enum, help = "active / muted: fixed by hand and exempt from auto-mute and auto-promotion. auto: clear the manual lock and apply the rule from the scores again")]
        status: StatusChoice,
    },

    #[command(about = "Show each channel's status, average slop score and what became of its videos: kept, below the quality cutoff, off-topic (fringe / nonsense / politics), not enough information, disqualified by hand. Only quality-judged videos count in the average")]
    ChannelHealth,

    #[command(about = "List creators related to a channel: names mentioned in its kept insights (local database) and channels featured on its YouTube page (one request). One hop only; nothing is stored")]
    RelatedChannels {
        #[arg(allow_hyphen_values = true, help = "Channel ID or exact title of a tracked channel, otherwise a handle or URL that is resolved through YouTube")]
        input: String,

        #[arg(long, help = "Do not request the channel page: report only the names found in the local database")]
        offline: bool,
    },

    #[command(
        about = "Crawl tracked channels via their RSS feeds (high-signal channels first, muted ones skipped): fetch the complete transcript of each new video (captions that stop early are replaced by the next source, in the end a local Whisper transcription), remove SponsorBlock segments, let the LLM read the whole transcript and store its judgment",
        after_help = failure_limit_help()
    )]
    Crawl {
        #[arg(short, long, help = "Crawl a specific channel ID only")]
        channel: Option<String>,

        #[arg(short, long, default_value = "3", help = "Max new videos to process per channel")]
        limit: usize,

        #[arg(short, long, default_value = "10", help = "Pause in seconds between two videos of a crawl. Never less than 10, also when a smaller value is given. It gets longer for a while, up to 300 s, when YouTube refuses a direct transcript request")]
        delay: u64,

        #[arg(long, help = "Write the wiki (global --wiki-dir) as the last step of the crawl. Without it the wiki is left as it is; run 'ytc wiki' afterwards")]
        wiki: bool,
    },

    #[command(about = "Search YouTube for a query and analyse the top results on any subject (whole transcript; topic and quality judged separately). With an LLM, the channel of a kept video scoring 35 or lower is added to the watchlist as 'active'")]
    Discover {
        #[arg(help = "Search query (e.g. 'sourdough baking', 'how tides work')")]
        query: String,

        #[arg(short, long, default_value = "3", help = "Max candidate videos to evaluate")]
        limit: usize,

        #[arg(short, long, default_value = "2", help = "Delay in seconds between candidates")]
        delay: u64,
    },

    #[command(
        about = "Repeat until Ctrl+C: crawl the tracked channels, optionally run discovery queries, write the wiki, then wait for the interval. A cycle that finds the LLM unreachable makes no YouTube request and only writes the wiki",
        after_help = "Ctrl+C stops the watcher within about a second, also in the middle of a crawl or a discovery query; a wiki that is being written is finished first."
    )]
    Watch {
        #[arg(short, long, default_value = "60", value_parser = clap::value_parser!(u64).range(1..), help = "Minutes to wait between the end of one cycle and the start of the next (at least 1)")]
        interval: u64,

        #[arg(short, long, default_value = "2", help = "Max new videos to process per channel per cycle")]
        limit: usize,

        #[arg(short, long, default_value = "15", help = "Pause in seconds between two videos during the crawl (same rule as 'crawl --delay': never less than 10, longer for a while when YouTube refuses a direct transcript request)")]
        delay: u64,

        #[arg(long, help = "Also run the discovery queries on each cycle (not in a cycle in which the LLM was found unreachable)")]
        discover: bool,

        #[arg(long, help = "Discovery queries separated by comma (e.g. 'first subject,second subject'). Required with --discover: there are no built-in queries")]
        queries: Option<String>,

        #[arg(long, default_value = "2", value_name = "N", help = "Max candidate videos to evaluate per discovery query")]
        discover_limit: usize,

        #[arg(long, default_value = "2", value_name = "SECONDS", help = "Delay in seconds between discovery candidates")]
        discover_delay: u64,

        #[arg(short, long, help = "Output directory for the wiki (default: the global --wiki-dir)")]
        wiki: Option<PathBuf>,
    },

    #[command(
        about = "Ingest and analyse a single video by URL or ID (complete transcript, read in full by the LLM), then print its insight with topic verdict, quality score and how the transcript was read (in how many sections)",
        after_help = process_video_retry_help()
    )]
    ProcessVideo {
        // Video IDs can start with '-' (e.g. -AbCdEfGh12); without this clap reads them as flags.
        #[arg(allow_hyphen_values = true, help = "YouTube Video URL or 11-character video ID")]
        input: String,

        #[arg(long, help = "Print the stored result as one JSON object on stdout and nothing else there (progress goes to stderr). The object carries, next to 'video' and 'insight': slop_score, is_worth_keeping (the effective decision), topic_verdict, topic_reason, info_status, info_reason, excluded, sections_total (the number of consecutive sections the whole transcript was read in; null when no analysis ran), transcript_source, video_duration_sec, covered_until_sec. Exit code is non-zero and stdout stays empty when the video could not be processed")]
        json: bool,
    },

    #[command(
        about = "Re-run the analysis on stored transcripts (no YouTube request), for example after you changed your categories or the model, or after an LLM was set up for videos that only have a heuristic estimate. Give exactly one selection: a video, --analyzer PREFIX or --all. The whole stored transcript is read again; a stored insight is replaced only by a valid LLM answer (or by 'not enough information' when the transcript is too short to judge)",
        after_help = "Before it starts the command prints how many videos and how many LLM requests are ahead. Reading a whole transcript takes one request for a short video and one per section plus a final one for a long video: this can take minutes per long video.\n\nWith --analyzer and --all the videos are taken in the order of their last analysis, the one analysed longest ago first. A run in portions with --limit, or a run started again after an interruption, therefore continues with the videos that were not analysed again yet. A video whose analysis failed keeps its place and is tried first by the next run. The run stops when the LLM is down or fails for two videos in a row; the remaining videos are left unchanged.\n\nOnly videos with a stored transcript can be analysed again. Selected videos without one are counted and listed with their IDs; 'ytc process-video <video id>' fetches the transcript and judges the video.\n\nVideos disqualified with 'ytc exclude' are skipped by --analyzer and --all. A video named directly is analysed and stays disqualified.",
        group(clap::ArgGroup::new("selection").required(true).multiple(false).args(["video_id", "analyzer", "all"]))
    )]
    Reanalyze {
        #[arg(allow_hyphen_values = true, help = "Video ID or YouTube video URL")]
        video_id: Option<String>,

        #[arg(long, value_name = "PREFIX", help = "Every video whose analyzer type starts with this prefix, e.g. 'heuristic' (estimated without an LLM) or 'llm:' followed by a model name")]
        analyzer: Option<String>,

        #[arg(long, help = "Every analysed video")]
        all: bool,

        #[arg(short, long, value_name = "N", help = "With --analyzer or --all: stop after N videos (default: all selected videos). The same command run again continues with the next N")]
        limit: Option<usize>,
    },

    #[command(about = "Disqualify a video by hand: it leaves the library at once, is never kept again whatever a later analysis says, and is left out of its channel's average. The next 'ytc wiki' stops listing it; 'ytc wiki --prune' moves its page out. Undo with 'ytc include'")]
    Exclude {
        #[arg(allow_hyphen_values = true, help = "Video ID or YouTube video URL")]
        video: String,

        #[arg(long, value_name = "TEXT", help = "Why the video is disqualified (shown in lists and on the insight card)")]
        reason: Option<String>,
    },

    #[command(about = "Undo 'ytc exclude': the stored judgment (topic, quality, amount of information) decides again whether the video is kept")]
    Include {
        #[arg(allow_hyphen_values = true, help = "Video ID or YouTube video URL")]
        video: String,
    },

    #[command(about = "List the videos disqualified by hand, most recent first")]
    Excluded,

    #[command(about = "Combine another library with this one, for example the one of a second PC: channels, videos, transcripts, insights and categories of the other database file are copied into the current database. One transaction, nothing is lost: for a video in both, the better analysis (LLM over heuristic, then the more recent one) and the longer transcript are kept; channel statuses set by hand, disqualifications and categories set by hand survive. The other file is only read")]
    MergeDb {
        #[arg(value_name = "OTHER_DB", help = "The database file of the other library. It is opened read-only")]
        other: PathBuf,
    },

    #[command(about = "Manage your own subject categories (add, rename, describe, remove, list). There are none until you add one; they only label videos, and the label is not used when deciding whether a video is kept. Adding or changing categories can shift individual scores slightly")]
    Category {
        #[command(subcommand)]
        action: CategoryAction,
    },

    #[command(
        about = "Set the category of a video by hand (one of your categories, or --none), or hand the choice back to the model (--auto). A category set by hand stays: no later analysis changes it",
        group(clap::ArgGroup::new("choice").required(true).multiple(false).args(["name", "none", "auto"]))
    )]
    SetCategory {
        #[arg(allow_hyphen_values = true, help = "Video ID or YouTube video URL")]
        video: String,

        #[arg(help = "Name of one of your categories (upper and lower case do not matter)")]
        name: Option<String>,

        #[arg(long, help = "Put the video into no category, by hand: the model does not assign one later")]
        none: bool,

        #[arg(long, help = "Clear the mark 'set by hand'. The category stays as it is until the next analysis or 'ytc categorize --all' lets the model choose again")]
        auto: bool,
    },

    #[command(
        about = "Give already analysed videos a category without analysing them again: one short LLM request per video, built from its stored summary, key insights and topics (the transcript is not read). Only the category changes; score, verdict and summary stay. Without --all: LLM-judged videos that have no category. Videos whose category was set by hand are never touched",
        after_help = "Before it starts the command prints how many videos are ahead; at the end it prints how many went into each category, how many fit none and how many failed. An answer that names none of your categories is stored as no category. The run stops when the LLM is down or fails for two videos in a row; the videos done until then keep their category.\n\nA video that fits none of your categories is remembered and not asked about on every run, so runs with --limit move on through the library. It is asked again after you add, rename, describe or remove a category, or with --all. Videos that only have a heuristic estimate or too little information are left out: 'ytc reanalyze' judges them."
    )]
    Categorize {
        #[arg(long, help = "Also ask again for the videos the model already put into a category (never for those set by hand)")]
        all: bool,

        #[arg(short, long, value_name = "N", help = "Stop after N videos (default: all selected videos)")]
        limit: Option<usize>,
    },

    #[command(about = "Write the Markdown wiki from the database (kept videos only, with one page per category of yours that holds kept videos) and report generated pages that match nothing in the library any more")]
    Wiki {
        #[arg(short, long, help = "Output directory for the wiki (default: the global --wiki-dir)")]
        output: Option<PathBuf>,

        #[arg(long, help = "Move stale generated pages (insights, topics, creators, categories that left the library) out of the wiki into the .ytc-pruned folder inside the wiki folder; nothing is deleted, move a page back to restore it. Only pages carrying the ytc marker are affected")]
        prune: bool,
    },

    #[command(about = "Start the local web UI server for the library and the wiki")]
    Serve {
        #[arg(short, long, default_value = "8088", help = "Local HTTP port to listen on")]
        port: u16,
    },

    #[command(about = "Report the creator names the analysis listed in kept insights, by number of videos mentioning them. Offline unless --resolve is given; nothing is tracked unless --add is given")]
    HarvestMentions {
        #[arg(long, default_value_t = 0, value_name = "N", help = "Try to find a channel for the N most-mentioned untracked names by guessing the handle (@NameWithoutSpaces). At most N lookups; each is one yt-dlp call plus one page request if that fails. 0 = no network")]
        resolve: usize,

        #[arg(long, help = "With --resolve: add a resolved channel to the watchlist as 'active' when its title matches the mentioned name. Others are only listed")]
        add: bool,

        #[arg(short, long, default_value_t = 50, value_name = "N", help = "Max names to print (0 = all)")]
        limit: usize,
    },

    #[command(about = "List ingested videos with quality score and rating. Videos that are not kept for another reason than quality say so: 'Off-topic: <fringe|nonsense|politics>', 'Not enough information', 'Disqualified by hand'; heuristic rows show 'Unreviewed (heuristic)'")]
    List {
        #[arg(long, help = "Only the videos in the library: topic acceptable, slop score at or below the cutoff it was judged with, enough information, not disqualified by hand")]
        kept_only: bool,

        #[arg(short, long, default_value = "20", help = "Max videos to list")]
        limit: usize,
    },

    #[command(about = "Show the stored insight of a video: quality score, topic verdict, whether it is kept and why not, in how many sections the whole transcript was read, where the transcript came from and how much of the video it covers")]
    Show {
        #[arg(allow_hyphen_values = true, help = "Video ID or YouTube video URL")]
        video_id: String,
    },

    #[command(about = "Search titles, insights, topics and transcripts of all ingested videos for a text (substring match; at most 20 hits, lowest slop score first). Each hit says whether it is kept and, if not, why")]
    Search {
        #[arg(help = "Search query")]
        query: String,
    },

    #[command(about = "Export the stored insight of a video to a Markdown note, with its quality score, topic verdict, kept status and transcript coverage")]
    Export {
        #[arg(allow_hyphen_values = true, help = "Video ID or YouTube video URL")]
        video_id: String,

        #[arg(short, long, help = "Output markdown file path (defaults to ytc_<video_id>.md in the current directory)")]
        output: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Cli { db, db_short, llm_url, llm_key, llm_model, max_slop, wiki_dir, command } = Cli::parse();

    // One rule: --db, else YTC_DB, else <app folder>/data/ytc.db. Nothing is
    // searched, so the line below names the only file this run reads and writes.
    let db_path = db.or(db_short).unwrap_or_else(Database::default_db_path);
    eprintln!("Using database: {}", absolute_for_display(&db_path).display());

    // Handled before anything else prints: this mode must keep stdout clean.
    if let Commands::ProcessVideo { input, json: true } = &command {
        return process_video_json(&db_path, input);
    }

    // An empty YTC_WIKI_DIR would otherwise make the working directory the wiki.
    let wiki_dir = if wiki_dir.as_os_str().is_empty() {
        PathBuf::from(pipeline::DEFAULT_WIKI_DIR)
    } else {
        wiki_dir
    };

    if command_sends_transcripts(&command) {
        if let Some(host) = llm_url.as_deref().and_then(remote_llm_host) {
            eprintln!(
                "{} The LLM endpoint is on '{}', not on this machine: transcript text is sent there for analysis.",
                "Warning:".yellow().bold(),
                host
            );
        }
    }

    let db = Database::open(&db_path)?;

    let pipeline = Arc::new(
        IngestionPipeline::new(&db_path, llm_url, llm_key, Some(llm_model))?
            .with_max_slop(max_slop)
            .with_wiki_dir(wiki_dir.clone()),
    );

    match command {
        Commands::AddChannel { input } => {
            println!("{}", format!("Resolving channel: {}", input).cyan());
            let (channel_id, channel_title) = rss::resolve_channel(&pipeline.http_client, &input).await?;

            db.upsert_channel(&new_active_channel(&channel_id, &channel_title))?;
            println!(
                "{} Added channel: {} ({})",
                "✓".green().bold(),
                channel_title.bold(),
                channel_id.dimmed()
            );
        }

        Commands::ListChannels => {
            let channels = db.list_channels_with_health()?;
            if channels.is_empty() {
                println!("{}", "No channels tracked yet. Add one with: ytc add-channel <url_or_handle>".yellow());
                return Ok(());
            }

            println!("\n{:<26} {:<16} {:<10} {:<10} {:<32}", "CHANNEL ID".bold(), "STATUS".bold(), "AVG SLOP".bold(), "KEPT".bold(), "TITLE".bold());
            println!("{}", "-".repeat(95).dimmed());
            for c in channels {
                let badge = match c.status.as_str() {
                    "high_signal" => "🟢 High Signal".green(),
                    "muted" => "🔴 Muted".red(),
                    _ => "🔵 Active".cyan(),
                };
                let slop_display = c.avg_slop_score
                    .map(|s| format!("{:.0}/100", s))
                    .unwrap_or_else(|| "N/A".to_string());
                // Kept / judged on quality. Off-topic, too short and disqualified
                // videos were not decided by their score and are not in the total.
                let kept_display = format!("{}/{}", c.kept_count, c.kept_count + c.below_cutoff_count);
                let manual = if db.is_channel_status_manual(&c.id)? { " [set by hand]" } else { "" };

                println!(
                    "{:<26} {:<16} {:<10} {:<10} {}{}",
                    c.id.dimmed(),
                    badge,
                    slop_display,
                    kept_display,
                    c.title.bold(),
                    manual.dimmed()
                );
            }
            println!("\nKEPT = kept / videos decided by their quality score. Off-topic videos (fringe / nonsense / politics), videos with not enough information and videos disqualified by hand are not in these numbers; see 'ytc channel-health'.\n");
        }

        Commands::ToggleChannel { input } => {
            let (channel_id, channel_title) = tracked_or_resolved(&db, &pipeline, &input).await?;
            let new_status = db.toggle_channel_status(&channel_id)?;
            let badge = if new_status == "muted" {
                "🔴 Muted".red()
            } else {
                "🔵 Active".cyan()
            };
            println!("Channel '{}' ({}) is now: {} (set by hand)", channel_title.bold(), channel_id.dimmed(), badge);
        }

        Commands::SetStatus { channel, status } => {
            let (channel_id, channel_title) = tracked_or_resolved(&db, &pipeline, &channel).await?;
            match status {
                StatusChoice::Active => db.set_channel_status(&channel_id, "active")?,
                StatusChoice::Muted => db.set_channel_status(&channel_id, "muted")?,
                StatusChoice::Auto => db.set_channel_status_auto(&channel_id)?,
            }

            // Print what is stored now, not what was asked for.
            let stored = db
                .get_channel(&channel_id)?
                .map(|c| c.status)
                .unwrap_or_else(|| "unknown".to_string());
            let how = if db.is_channel_status_manual(&channel_id)? {
                "set by hand; the automatic rule does not change it"
            } else {
"automatic: follows the average slop score once 3 videos are judged on quality"
            };
            println!(
                "Channel '{}' ({}) is now: {} ({})",
                channel_title.bold(),
                channel_id.dimmed(),
                stored.bold(),
                how
            );
        }

        Commands::ChannelHealth => {
            let channels = db.list_channels_with_health()?;
            println!("\n{}", "===========================================================================".cyan());
            println!("  {}", "CHANNEL HEALTH SCORECARD & SLOP RATE MONITOR".bold());
            println!("  The average counts only videos an LLM judged on quality. Off-topic videos,");
            println!("  videos with not enough information and disqualified ones are left out.");
            println!("---------------------------------------------------------------------------");

            for c in channels {
                let manual = db.is_channel_status_manual(&c.id)?;
                let status_label = match c.status.as_str() {
                    "high_signal" => "🟢 HIGH SIGNAL (low average slop over 3+ quality-judged videos)".green().bold(),
                    "muted" if manual => "🔴 MUTED (by hand)".red().bold(),
                    "muted" => "🔴 MUTED (auto-muted: average slop > 75 over 3+ quality-judged videos)".red().bold(),
                    _ => "🔵 ACTIVE (Monitoring)".cyan().bold(),
                };
                let avg_slop = c.avg_slop_score
                    .map(|s| format!("{:.1}/100", s))
                    .unwrap_or_else(|| "No video judged on quality yet".to_string());

                println!("  Channel:  {}", c.title.bold());
                println!("  ID:       {}", c.id.dimmed());
                println!("  Status:   {}", status_label);
                println!(
                    "  Set by:   {}",
                    if manual { "hand (hand back with: ytc set-status <channel> auto)" } else { "automatic rule" }
                );
                println!("  Slop Avg: {}", avg_slop);
                println!(
                    "  Videos:   {} kept | {} below the quality cutoff | {} off-topic (fringe / nonsense / politics) | {} not enough information | {} disqualified by hand",
                    c.kept_count, c.below_cutoff_count, c.off_topic_count, c.insufficient_count, c.excluded_count
                );
                if c.status == "muted" {
                    println!("  Action:   {}", "Channel is skipped by crawls until it is set back to active or auto.".yellow());
                } else if c.status == "high_signal" {
                    println!("  Action:   {}", "Channel is crawled before active channels.".green());
                }
                println!("{}\n", "-".repeat(75).dimmed());
            }
        }

        Commands::RelatedChannels { input, offline } => {
            let (target_id, target_title) = tracked_or_resolved(&db, &pipeline, &input).await?;
            println!("Channel: {} ({})", target_title.bold(), target_id.dimmed());
            println!(
                "Sources: creator names in this channel's kept insights (local database){}\n",
                if offline { "" } else { " + channel items on its YouTube page (one request)" }
            );

            let related = pipeline.related_channels(&target_id, !offline).await?;
            if related.is_empty() {
                println!(
                    "{}",
                    if offline {
                        "No creator is named in this channel's kept insights."
                    } else {
                        "No creator is named in this channel's kept insights and no channel item was found on its page."
                    }
                    .yellow()
                );
                return Ok(());
            }

            println!("Found {} candidate(s):", related.len());
            println!("{}", "-".repeat(75).dimmed());
            for r in &related {
                let source = if r.source == rss::RELATED_SOURCE_MENTIONS {
                    format!("mentioned in {} kept insight(s)", r.mentions)
                } else {
                    "featured on the channel page".to_string()
                };
                let status = match (&r.channel_id, r.tracked) {
                    (Some(id), true) => format!("already tracked ({})", id).green(),
                    (Some(id), false) => format!("not tracked (ytc add-channel {})", id).yellow(),
                    (None, _) => "name only, no channel known (see: ytc harvest-mentions --resolve N)".yellow(),
                };
                println!("  • {:<30} {:<32} {}", r.title.bold(), source.dimmed(), status);
            }
            println!();
        }

        Commands::Discover { query, limit, delay } => {
            println!(
                "\n{}",
                format!("🔍 Running Discovery Harvest for: '{}' (Candidate Limit: {})...", query, limit)
                    .cyan()
                    .bold()
            );

            let res = pipeline.run_discover(&query, limit, delay).await?;
            let mut counts = OutcomeCounts::default();
            for item in &res.items {
                // The stored row decides; the item's flag is the fallback when it cannot be read.
                counts.add(match db.get_insight_meta(&item.video_id)? {
                    Some(meta) => pipeline::video_outcome(&meta),
                    None if item.is_worth_keeping => VideoOutcome::Kept,
                    None => VideoOutcome::BelowCutoff,
                });
            }
            let added: Vec<&str> = res.items.iter().filter_map(|i| i.auto_promoted_creator.as_deref()).collect();
            println!(
                "\n{} Discovery finished: {} search result(s), {} analysed: {} (quality cutoff: slop <= {}).",
                "✓".green(),
                res.candidates_found,
                res.items.len(),
                counts.line(),
                pipeline.max_slop
            );
            if !added.is_empty() {
                println!("  Added to the watchlist: {}", added.join(", "));
            }
        }

        Commands::Watch { interval, limit, delay, discover, queries, discover_limit, discover_delay, wiki } => {
            let queries = watch_queries(queries.as_deref());
            if discover && queries.is_empty() {
                anyhow::bail!("--discover needs --queries \"first subject,second subject\": there are no built-in queries.");
            }
            let options = WatchOptions {
                interval_minutes: interval,
                limit_per_channel: limit,
                delay_secs: delay,
                discover,
                queries,
                discover_limit,
                discover_delay_secs: discover_delay,
                wiki_dir: effective_wiki_dir(wiki, &wiki_dir),
            };
            run_watch_loop(&pipeline, &options).await?;
        }

        Commands::Wiki { output, prune } => {
            let target_dir = effective_wiki_dir(output, &wiki_dir);
            println!("{}", format!("Writing the Knowledge Wiki to: {:?} ...", target_dir).cyan());
            let stats = wiki::WikiGenerator::generate_with_options(&db, &target_dir, prune)?;
            wiki::WikiGenerator::print_wiki_summary(&stats);

            // Absolute, so the printed folder and command are right from any directory.
            let wiki_abs = absolute_dir(&target_dir);
            let pruned_dir = wiki_abs.join(PRUNED_DIR_NAME);
            let remaining = stats.stale_pages.saturating_sub(stats.pruned_pages);
            // After a prune the summary above names the exact folder the pages went to.
            if remaining > 0 && !prune {
                println!(
                    "{} stale generated page(s) are still in {:?}. Move them to {:?} with: {}",
                    remaining,
                    wiki_abs,
                    pruned_dir,
                    wiki_prune_command(&target_dir)
                );
            } else if remaining > 0 {
                println!(
                    "{} {} stale page(s) could not be moved; they are still in {:?}.",
                    "⚠".yellow(),
                    remaining,
                    wiki_abs
                );
            }
        }

        Commands::Serve { port } => {
            server::start_server(pipeline, port).await?;
        }

        Commands::HarvestMentions { resolve, add, limit } => {
            if add && resolve == 0 {
                anyhow::bail!("--add needs --resolve N: without a lookup there is no channel to add");
            }
            harvest_mentions(&db, &pipeline, resolve, add, limit).await?;
        }

        Commands::Crawl { channel, limit, delay, wiki } => {
            println!(
                "{}",
                format!(
                    "Starting crawl across tracked channels (limit: {} new video(s) per channel, quality cutoff: slop <= {})...",
                    limit, pipeline.max_slop
                )
                .cyan()
                .bold()
            );

            let summary = if wiki {
                pipeline.run_crawl_and_wiki(channel, limit, delay).await?
            } else {
                pipeline.run_crawl(channel, limit, delay).await?
            };
            println!("\n{} Crawl finished: {}", "✓".green(), crawl_summary_line(&summary));

            if let Some(err) = &summary.wiki_error {
                anyhow::bail!("The crawl results are stored, but the wiki could not be written to {:?}: {}", pipeline.wiki_dir, err);
            }
            if summary.wiki_generated {
                println!("  Wiki written to {:?}.", pipeline.wiki_dir);
            } else if summary.videos_kept > 0 {
                println!("  The wiki was not updated. Run 'ytc wiki', or use 'ytc crawl --wiki' next time.");
            }
        }

        Commands::ProcessVideo { input, json: _ } => {
            let video_id = extract_video_id(&input)?;
            let url = format!("https://www.youtube.com/watch?v={}", video_id);

            println!("{}", format!("Analyzing video: {}", url).cyan());

            let existing = db.get_video(&video_id)?;
            // A row whose title or channel is a placeholder (the metadata could not be
            // fetched when it was created) is treated like missing data, so running
            // the command again repairs it.
            let needs_title = existing.as_ref().is_none_or(|v| is_placeholder_title(&v.title));
            let needs_channel = existing.as_ref().is_none_or(|v| is_placeholder_channel(&v.channel_id));

            let video = match existing {
                Some(v) if !needs_title && !needs_channel => v,
                existing => {
                    let is_new = existing.is_none();
                    let now = chrono::Utc::now().to_rfc3339();
                    let mut v = existing.unwrap_or_else(|| Video {
                        id: video_id.clone(),
                        channel_id: "direct_ingest".to_string(),
                        title: format!("Video {}", video_id),
                        url: url.clone(),
                        published_at: now.clone(),
                        status: "pending".to_string(),
                        created_at: now.clone(),
                    });

                    // One yt-dlp call for title, channel and the real upload date.
                    let meta = pipeline.extractor.fetch_video_meta(&video_id);
                    match &meta {
                        Some(m) => {
                            if needs_title {
                                v.title = m.title.clone();
                            }
                            if needs_channel {
                                if let Some(id) = &m.channel_id {
                                    v.channel_id = id.clone();
                                }
                            }
                            // A placeholder row stored the ingest time as its date, so it is replaced too.
                            match &m.upload_date {
                                Some(date) => v.published_at = date.clone(),
                                None if is_new => eprintln!(
                                    "{} yt-dlp reported no upload date for {}: today's date is stored as its publication date.",
                                    "Warning:".yellow(),
                                    video_id
                                ),
                                None => {}
                            }
                        }
                        None => {
                            if needs_title {
                                if let Some(title) = pipeline.extractor.fetch_title(&video_id) {
                                    v.title = title;
                                }
                            }
                            if is_new {
                                eprintln!(
                                    "{} Video details could not be read through yt-dlp: today's date is stored as the publication date of {}.",
                                    "Warning:".yellow(),
                                    video_id
                                );
                            }
                        }
                    }
                    if needs_channel && is_placeholder_channel(&v.channel_id) {
                        if let Ok((channel_id, _)) = rss::resolve_channel(&pipeline.http_client, &url).await {
                            v.channel_id = channel_id;
                        }
                    }
                    db.upsert_video(&v)?;
                    v
                }
            };

            let insight = pipeline.process_video(&video).await?;
            let meta = db.get_insight_meta(&video.id)?;
            let coverage = db.get_transcript_coverage(&video.id)?;
            print_insight_card(&video, &insight, meta.as_ref(), coverage.as_ref());
        }

        Commands::Reanalyze { video_id, analyzer, all, limit } => {
            let selection = reanalyze_selection(video_id, analyzer, all)?;
            reanalyze(&db, &pipeline, selection, limit).await?;
        }

        Commands::Exclude { video, reason } => {
            exclude_video(&db, &video, reason.as_deref(), &wiki_dir)?;
        }

        Commands::Include { video } => {
            include_video(&db, &video)?;
        }

        Commands::Category { action } => {
            category_command(&db, action)?;
        }

        Commands::SetCategory { video, name, none, auto } => {
            set_category(&db, &video, category_choice(name, none, auto)?)?;
        }

        Commands::Categorize { all, limit } => {
            categorize(&db, &pipeline, all, limit).await?;
        }

        Commands::Excluded => {
            let excluded = db.list_excluded()?;
            if excluded.is_empty() {
                println!("{}", "No video is disqualified by hand.".yellow());
                return Ok(());
            }
            println!("\n=== Videos disqualified by hand ({}) ===\n", excluded.len());
            println!("{:<13} {:<11} {:<24} {}", "ID".bold(), "SINCE".bold(), "CHANNEL".bold(), "TITLE".bold());
            println!("{}", "-".repeat(100).dimmed());
            for e in &excluded {
                // RFC 3339: the first ten characters are the date.
                let since = e
                    .excluded_at
                    .as_deref()
                    .map(|t| t.chars().take(10).collect::<String>())
                    .unwrap_or_else(|| "?".to_string());
                println!(
                    "{:<13} {:<11} {:<24} {}",
                    e.video_id.cyan(),
                    since,
                    truncate_chars(e.channel_title.as_deref().unwrap_or("(untracked channel)"), 24),
                    truncate_chars(&e.title, 48)
                );
                if let Some(reason) = &e.reason {
                    println!("{:<13} {}", "", format!("Reason: {}", reason).dimmed());
                }
            }
            println!("\nUndo with: ytc include <video id>\n");
        }

        Commands::MergeDb { other } => {
            println!(
                "{}",
                format!(
                    "Merging {} (opened read-only) into {} ...",
                    absolute_for_display(&other).display(),
                    absolute_for_display(&db_path).display()
                )
                .cyan()
            );
            let stats = merge_db(&db, &db_path, &other)?;
            println!("{}", stats);
            if stats.is_noop() {
                println!("\nNothing was changed: the current database already holds everything from that file.");
            } else {
                println!(
                    "\n{} Merge finished. The wiki is not rewritten by this command. Bring it in line with: {}",
                    "✓".green(),
                    wiki_prune_command(&wiki_dir)
                );
            }
        }

        Commands::List { kept_only, limit } => {
            let list = db.list_library(kept_only, limit)?;
            if list.is_empty() {
                println!("{}", "No videos found in library.".yellow());
                return Ok(());
            }
            let metas = db.list_insight_meta()?;

            println!(
                "\n=== {} ({}) ===\n",
                if kept_only { "Kept Videos" } else { "All Ingested Videos" },
                list.len()
            );

            println!("{:<13} {:<6} {:<36} {}", "ID".bold(), "SCORE".bold(), "RATING".bold(), "TITLE".bold());
            println!("{}", "-".repeat(100).dimmed());

            for (video, insight_opt) in list {
                let (score_str, rating_str) = match &insight_opt {
                    Some(i) => score_and_rating(i, metas.get(&video.id)),
                    None => ("N/A".to_string(), "Pending".to_string()),
                };

                println!(
                    "{:<13} {:<6} {:<36} {}",
                    video.id.cyan(),
                    score_str,
                    truncate_chars(&rating_str, 36),
                    truncate_chars(&video.title, 42)
                );
            }
            println!();
        }

        Commands::Show { video_id } => {
            let vid = extract_video_id(&video_id)?;
            let video = db.get_video(&vid)?.ok_or_else(|| anyhow::anyhow!("Video not found in database"))?;
            let insight = db.get_insight(&vid)?.ok_or_else(|| anyhow::anyhow!("Insights not found for this video"))?;
            let meta = db.get_insight_meta(&vid)?;
            let coverage = db.get_transcript_coverage(&vid)?;
            print_insight_card(&video, &insight, meta.as_ref(), coverage.as_ref());
            if let Some(t) = db.get_transcript(&vid)? {
                println!(
                    "  TRANSCRIPT: {} -> {} chars, {:.0}s removed in {} SponsorBlock segment(s)\n",
                    t.raw_char_count, t.cleaned_char_count, t.stripped_ad_seconds, t.stripped_segments_count
                );
            }
        }

        Commands::Search { query } => {
            println!("{}", format!("Searching library for: '{}' ...", query).cyan());
            let hits = db.search(&query)?;
            if hits.is_empty() {
                println!("{}", "No matching insights found.".yellow());
                return Ok(());
            }

            println!("\nFound {} match(es):\n", hits.len());
            for h in hits {
                // The stored row and its meta carry what the hit does not: the
                // analyzer type, the reasons and the coverage figures.
                let Some(insight) = db.get_insight(&h.video_id)? else {
                    continue;
                };
                let meta = db.get_insight_meta(&h.video_id)?;
                let meta = meta.as_ref();
                let coverage = db.get_transcript_coverage(&h.video_id)?;

                let (score, rating) = score_and_rating(&insight, meta);
                let score = if score == "-" { "Not scored".to_string() } else { format!("Slop Score: {}/100", score) };
                println!("{} [{}] {}", h.title.bold(), h.video_id.cyan(), score);

                let rating = match outcome_of(&insight, meta) {
                    VideoOutcome::Kept if is_heuristic(&insight) => format!("{} (kept)", rating).yellow(),
                    VideoOutcome::Kept => format!("{} (kept)", rating).green(),
                    VideoOutcome::BelowCutoff => format!("{} (not kept: below the quality cutoff)", rating).red(),
                    VideoOutcome::Insufficient => format!("{} (not kept)", rating).yellow(),
                    VideoOutcome::OffTopic | VideoOutcome::Disqualified => format!("{} (not kept)", rating).red(),
                };
                println!("  Rating: {}", rating);
                if let Some(note) = read_note(&insight, meta) {
                    println!("  Read: {}", note);
                }
                if let Some(note) = transcript_source_note(coverage.as_ref()) {
                    println!("  Transcript: {}", note);
                }
                println!("  Summary: {}", h.summary);
                if !h.topics.is_empty() {
                    println!("  Topics: {}", h.topics.join(", ").dimmed());
                }
                if let Some(top_insight) = h.key_insights.first() {
                    println!("  Top Insight: {}", top_insight);
                }
                println!();
            }
        }

        Commands::Export { video_id, output } => {
            let vid = extract_video_id(&video_id)?;
            let video = db.get_video(&vid)?.ok_or_else(|| anyhow::anyhow!("Video not found"))?;
            let insight = db.get_insight(&vid)?.ok_or_else(|| anyhow::anyhow!("Insights not found"))?;
            let meta = db.get_insight_meta(&vid)?;
            let coverage = db.get_transcript_coverage(&vid)?;

            let filename = output.unwrap_or_else(|| PathBuf::from(format!("ytc_{}.md", vid)));
            std::fs::write(&filename, export_markdown(&video, &insight, meta.as_ref(), coverage.as_ref()))?;
            println!("{} Exported Markdown note to: {:?}", "✓".green(), filename);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Small pure helpers
// ---------------------------------------------------------------------------

/// A subcommand's own directory flag wins over the global `--wiki-dir`.
fn effective_wiki_dir(local: Option<PathBuf>, global: &Path) -> PathBuf {
    local
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| global.to_path_buf())
}

/// Absolute form of a directory without "." components ("./wiki" becomes
/// <current dir>/wiki). Not canonicalized, see `absolute_for_display`.
fn absolute_dir(dir: &Path) -> PathBuf {
    let cleaned: PathBuf = dir
        .components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .collect();
    absolute_for_display(&cleaned)
}

/// The prune command for exactly this wiki folder. The folder is always
/// named, as an absolute path: the default "./wiki" and any other relative
/// path would mean a different folder when the command is run elsewhere.
fn wiki_prune_command(target: &Path) -> String {
    format!("ytc wiki --prune --wiki-dir \"{}\"", absolute_dir(target).display())
}

/// Shown under `crawl --help`.
fn failure_limit_help() -> String {
    format!(
        "A video whose transcript or analysis failed {n} times is skipped by every later crawl; the crawl prints 'Given up after {n} failed attempts' for it. Failures caused by a YouTube rate limit (HTTP 429) or by an unreachable LLM are not counted. To try such a video again, run: ytc process-video <video id>",
        n = pipeline::MAX_VIDEO_FAILURES
    )
}

/// Shown under `process-video --help`.
fn process_video_retry_help() -> String {
    format!(
        "This command always makes a new attempt: it is the way to retry a video that 'crawl' and 'watch' skip after {} failed attempts ('Given up after ...').",
        pipeline::MAX_VIDEO_FAILURES
    )
}

/// Splits `--queries`; without it (or with nothing usable in it) the list is empty.
fn watch_queries(raw: Option<&str>) -> Vec<String> {
    raw.map(|q| q.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default()
}

/// Host of the LLM endpoint when it is not this machine. None for loopback
/// addresses and for a URL that cannot be read (the request then fails anyway).
fn remote_llm_host(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url.trim()).ok()?;
    let host = parsed.host_str()?.trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    let local = host == "localhost"
        || host.ends_with(".localhost")
        || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    (!local).then_some(host)
}

/// Commands that hand transcript text to the LLM endpoint.
fn command_sends_transcripts(command: &Commands) -> bool {
    matches!(
        command,
        Commands::Crawl { .. }
            | Commands::Discover { .. }
            | Commands::Watch { .. }
            | Commands::ProcessVideo { .. }
            | Commands::Reanalyze { .. }
            | Commands::Serve { .. }
    )
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() > max {
        format!("{}...", text.chars().take(max.saturating_sub(3)).collect::<String>())
    } else {
        text.to_string()
    }
}

fn new_active_channel(channel_id: &str, title: &str) -> Channel {
    Channel {
        id: channel_id.to_string(),
        title: title.to_string(),
        url: format!("https://www.youtube.com/channel/{}", channel_id),
        status: "active".to_string(),
        rolling_slop_score: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        last_crawled_at: None,
    }
}

fn is_heuristic(insight: &InsightResult) -> bool {
    insight.analyzer_type == analyzer::HEURISTIC_ANALYZER
}

/// "llm" or "heuristic": which kind of code judged the video.
fn engine_kind(insight: &InsightResult) -> &'static str {
    if is_heuristic(insight) {
        "heuristic"
    } else {
        "llm"
    }
}

fn is_insufficient(insight: &InsightResult, meta: Option<&InsightMeta>) -> bool {
    insight.analyzer_type == pipeline::INSUFFICIENT_ANALYZER || meta.is_some_and(|m| m.is_insufficient())
}

/// What became of a stored video. With the meta row this is the database's
/// one kept rule. Without it only the insight itself can be asked: its flag
/// is the quality decision, which `process_video` and `reanalyze_video`
/// already return as the effective one.
fn outcome_of(insight: &InsightResult, meta: Option<&InsightMeta>) -> VideoOutcome {
    match meta {
        Some(m) => pipeline::video_outcome(m),
        None if is_insufficient(insight, None) => VideoOutcome::Insufficient,
        None if insight.is_worth_keeping => VideoOutcome::Kept,
        None => VideoOutcome::BelowCutoff,
    }
}

/// First words of the confirmation `ytc exclude` prints. For a video that
/// was disqualified already it says what really happened to the stored
/// reason, compared before and after the call, instead of always "updated".
fn exclude_label(was_excluded: bool, reason_before: Option<&str>, reason_after: Option<&str>) -> &'static str {
    let clean = |r: Option<&str>| r.map(str::trim).filter(|r| !r.is_empty()).map(str::to_string);
    match (was_excluded, clean(reason_before), clean(reason_after)) {
        (false, _, _) => "Disqualified by hand",
        (true, before, after) if before == after => "Was already disqualified; nothing changed",
        (true, _, Some(_)) => "Was already disqualified; reason updated",
        (true, _, None) => "Was already disqualified; reason removed",
    }
}

fn with_reason(label: &str, reason: Option<&str>) -> String {
    match reason.map(str::trim).filter(|r| !r.is_empty()) {
        Some(reason) => format!("{} ({})", label, reason),
        None => label.to_string(),
    }
}

/// Why a video is out of the library for another reason than its quality
/// score: "Disqualified by hand (..)", "Off-topic: politics (..)" or "Not
/// enough information (..)". None for a kept video and for one below the
/// quality cutoff.
fn rejection_note(insight: &InsightResult, meta: Option<&InsightMeta>) -> Option<String> {
    match outcome_of(insight, meta) {
        VideoOutcome::Kept | VideoOutcome::BelowCutoff => None,
        VideoOutcome::Disqualified => Some(with_reason(
            "Disqualified by hand",
            meta.and_then(|m| m.excluded_reason.as_deref()),
        )),
        VideoOutcome::OffTopic => {
            let verdict = meta.and_then(|m| m.topic_verdict.as_deref()).unwrap_or("rejected");
            Some(with_reason(
                &format!("Off-topic: {}", verdict),
                meta.and_then(|m| m.topic_reason.as_deref()),
            ))
        }
        VideoOutcome::Insufficient => Some(with_reason(
            "Not enough information",
            meta.and_then(|m| m.info_reason.as_deref()),
        )),
    }
}

/// How the analysis read the transcript: "whole transcript (3 sections)".
/// None when nothing was read (too little content) or nothing was recorded.
fn read_note(insight: &InsightResult, meta: Option<&InsightMeta>) -> Option<String> {
    if is_insufficient(insight, meta) {
        return None;
    }
    meta?.whole_transcript_note()
}

/// "YouTube transcript (reaches 97% of the video)": where the transcript came
/// from and how far into the video it reaches, as far as recorded.
fn transcript_source_note(coverage: Option<&TranscriptCoverage>) -> Option<String> {
    let c = coverage?;
    let reach = c.video_coverage_pct().map(|pct| format!("reaches {}% of the video", pct));
    match (c.source_label(), reach) {
        (Some(label), Some(reach)) => Some(format!("{} ({})", label, reach)),
        (Some(label), None) => Some(label.to_string()),
        (None, reach) => reach,
    }
}

/// The SCORE and RATING cells of a list row. A row without a measurement
/// (too little content: the stored 0 is a placeholder) shows no score. A
/// video that is out for its topic, for too little information or by the
/// owner's decision says so in the rating cell. A heuristic row has an
/// estimate that no LLM reviewed.
fn score_and_rating(insight: &InsightResult, meta: Option<&InsightMeta>) -> (String, String) {
    let score = if is_insufficient(insight, meta) { "-".to_string() } else { insight.slop_score.to_string() };
    let rating = match rejection_note(insight, meta) {
        Some(note) => note,
        None if is_heuristic(insight) => UNREVIEWED_LABEL.to_string(),
        None => insight.quality_rating.clone(),
    };
    (score, rating)
}

/// One-phrase judgment for before/after lines: "37/100, topic ok",
/// "62/100, topic: politics", "40/100, Unreviewed (heuristic), topic not
/// judged" or "Not enough information (<reason>)". Without a meta row only
/// the score part is known.
fn score_with_engine(insight: &InsightResult, meta: Option<&InsightMeta>) -> String {
    if is_insufficient(insight, meta) {
        return with_reason("Not enough information", meta.and_then(|m| m.info_reason.as_deref()));
    }
    let mut phrase = format!("{}/100", insight.slop_score);
    if is_heuristic(insight) {
        phrase.push_str(&format!(", {}", UNREVIEWED_LABEL));
    }
    if let Some(m) = meta {
        match m.topic_verdict.as_deref() {
            Some(models::TOPIC_OK) => phrase.push_str(", topic ok"),
            Some(verdict) => phrase.push_str(&format!(", topic: {}", verdict)),
            None => phrase.push_str(", topic not judged"),
        }
        if m.excluded_manual {
            phrase.push_str(", disqualified by hand");
        }
    }
    phrase
}

/// Counts of video outcomes for a result line. Every video is in exactly one.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct OutcomeCounts {
    kept: usize,
    below_cutoff: usize,
    off_topic: usize,
    insufficient: usize,
    disqualified: usize,
}

impl OutcomeCounts {
    fn add(&mut self, outcome: VideoOutcome) {
        match outcome {
            VideoOutcome::Kept => self.kept += 1,
            VideoOutcome::BelowCutoff => self.below_cutoff += 1,
            VideoOutcome::OffTopic => self.off_topic += 1,
            VideoOutcome::Insufficient => self.insufficient += 1,
            VideoOutcome::Disqualified => self.disqualified += 1,
        }
    }

    /// Same wording as `CrawlSummary::counts_line`.
    fn line(&self) -> String {
        let mut line = format!(
            "{} kept, {} below the quality cutoff, {} off-topic (fringe / nonsense / politics), {} not enough information",
            self.kept, self.below_cutoff, self.off_topic, self.insufficient
        );
        if self.disqualified > 0 {
            line.push_str(&format!(", {} disqualified by hand", self.disqualified));
        }
        line
    }
}

fn crawl_summary_line(summary: &pipeline::CrawlSummary) -> String {
    format!("{} channel(s) checked | {}", summary.channels_checked, summary.counts_line())
}

/// "Kept in the library" or "Not kept: <why>", for notes and JSON consumers.
fn kept_status_text(insight: &InsightResult, meta: Option<&InsightMeta>) -> String {
    match (outcome_of(insight, meta), rejection_note(insight, meta)) {
        (VideoOutcome::Kept, _) => "Kept in the library".to_string(),
        (_, Some(note)) => format!("Not kept: {}", note),
        (_, None) => "Not kept: below the quality cutoff".to_string(),
    }
}

/// `ytc category add | rename | describe | remove | list`.
fn category_command(db: &Database, action: CategoryAction) -> anyhow::Result<()> {
    match action {
        CategoryAction::Add { name, description } => {
            let category = db.add_category(&name, description.as_deref())?;
            println!("{} Added category: {}", "✔".green(), category.name.bold());
            if let Some(description) = &category.description {
                println!("  {}", description.dimmed());
            }
            println!("  Videos analysed from now on can be assigned to it; 'ytc categorize' asks the LLM for the videos that are already analysed.");
        }
        CategoryAction::Rename { old, new } => {
            let (before, after, moved) = db.rename_category(&old, &new)?;
            println!("{} Renamed category: {} -> {}", "✔".green(), before, after.bold());
            println!("  {} video(s) are in it and moved along. The next 'ytc wiki' writes its page under the new name; 'ytc wiki --prune' moves the old page out.", moved);
        }
        CategoryAction::Describe { name, text } => {
            let category = db.describe_category(&name, text.as_deref())?;
            match &category.description {
                Some(description) => {
                    println!("{} Description of {}: {}", "✔".green(), category.name.bold(), description)
                }
                None => println!("{} {} has no description now.", "✔".green(), category.name.bold()),
            }
            println!("  The LLM reads it for videos analysed from now on; stored categories do not change.");
        }
        CategoryAction::Remove { name } => {
            let (removed, cleared) = db.remove_category(&name)?;
            println!("{} Removed category: {}", "✔".green(), removed.bold());
            println!("  {} video(s) were in it. They stay in the library, without a category, also those put into it by hand.", cleared);
        }
        CategoryAction::List => {
            let categories = db.list_categories_with_counts()?;
            if categories.is_empty() {
                println!("{}", "No categories defined. Add one with: ytc category add <name> [--description <text>]".yellow());
                return Ok(());
            }
            println!("\n=== Categories ({}) ===\n", categories.len());
            println!("{:<32} {:<6} {}", "NAME".bold(), "KEPT".bold(), "DESCRIPTION".bold());
            println!("{}", "-".repeat(100).dimmed());
            for (category, kept) in &categories {
                println!(
                    "{:<32} {:<6} {}",
                    truncate_chars(&category.name, 32).cyan(),
                    kept,
                    category.description.as_deref().unwrap_or("-")
                );
            }
            println!();
        }
    }
    Ok(())
}

/// The category of a video for `show` and `export`, with "(set by hand)"
/// when the owner chose it. A video put into no category by hand says so;
/// one that simply has none gets no line (None).
fn category_label(insight: &InsightResult, meta: Option<&InsightMeta>) -> Option<String> {
    let by_hand = meta.is_some_and(|m| m.category_manual);
    match (insight.category.as_deref(), by_hand) {
        (Some(category), true) => Some(format!("{} (set by hand)", category)),
        (Some(category), false) => Some(category.to_string()),
        (None, true) => Some("none (set by hand)".to_string()),
        (None, false) => None,
    }
}

/// What `ytc set-category` was asked to do with a video.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CategoryChoice {
    /// Into this category, by hand.
    Named(String),
    /// Into no category, by hand.
    NoCategory,
    /// The model decides again.
    Auto,
}

/// The choice the command line asks for. clap already makes sure exactly one
/// is given; the error is for callers that bypass it.
fn category_choice(name: Option<String>, none: bool, auto: bool) -> anyhow::Result<CategoryChoice> {
    match (name, none, auto) {
        (Some(name), false, false) => Ok(CategoryChoice::Named(name)),
        (None, true, false) => Ok(CategoryChoice::NoCategory),
        (None, false, true) => Ok(CategoryChoice::Auto),
        _ => anyhow::bail!("Give exactly one of: a category name, --none, --auto"),
    }
}

/// `ytc set-category`: the owner's choice, stored on the insight row.
fn set_category(db: &Database, input: &str, choice: CategoryChoice) -> anyhow::Result<()> {
    let video_id = extract_video_id(input)?;
    if db.get_insight_meta(&video_id)?.is_none() {
        anyhow::bail!("Video {} has no analysis in this database. Nothing was changed.", video_id);
    }
    match &choice {
        CategoryChoice::Named(name) => {
            db.set_video_category(&video_id, Some(name))?;
        }
        CategoryChoice::NoCategory => {
            db.set_video_category(&video_id, None)?;
        }
        CategoryChoice::Auto => db.set_video_category_auto(&video_id)?,
    }

    // Print what is stored now, not what was asked for.
    let title = title_of(db, &video_id)?;
    let stored = db.get_insight(&video_id)?.and_then(|i| i.category);
    let stored = stored.as_deref().unwrap_or("none");
    println!("{} {} [{}]", "✓".green(), title.bold(), video_id.cyan());
    if choice == CategoryChoice::Auto {
        println!(
            "  Category: {} (automatic: the next analysis or 'ytc categorize --all' lets the model choose again)",
            stored.bold()
        );
    } else {
        println!(
            "  Category: {} (set by hand; no analysis changes it. Hand it back with: ytc set-category {} --auto)",
            stored.bold(),
            video_id
        );
    }
    Ok(())
}

/// `ytc categorize`: asks the LLM for the category of analysed videos.
async fn categorize(db: &Database, pipeline: &IngestionPipeline, all: bool, limit: Option<usize>) -> anyhow::Result<()> {
    if db.list_categories()?.is_empty() {
        println!(
            "{}",
            "No categories defined, so there is nothing to assign. Add one with: ytc category add <name> [--description <text>]".yellow()
        );
        return Ok(());
    }
    let (_, _, url) = pipeline.analyzer.describe();
    if let Some(host) = remote_llm_host(&url) {
        eprintln!(
            "{} The LLM endpoint is on '{}', not on this machine: titles and stored summaries are sent there.",
            "Warning:".yellow().bold(),
            host
        );
    }

    let summary = pipeline.run_categorize(all, limit).await?;
    if summary.total == 0 {
        println!("Nothing to categorize.");
        return Ok(());
    }
    println!("\nCategorized {} video(s): {}.", summary.total, summary.counts_line());
    for (name, count) in &summary.assigned {
        println!("  {:<32} {}", truncate_chars(name, 32).cyan(), count);
    }
    println!("  {:<32} {}", "(no category)", summary.none);
    println!("  {:<32} {}", "(failed, unchanged)", summary.failed);
    if summary.assigned_total() > 0 {
        println!(
            "Wiki pages are not rewritten by this command. Bring the wiki in line with: {}",
            wiki_prune_command(&pipeline.wiki_dir)
        );
    }
    if summary.stopped.is_some() {
        println!("Start the same command again to go on: it takes the videos that were not asked about yet first.");
    }
    if summary.failed > 0 {
        anyhow::bail!("{} of {} video(s) could not be categorized", summary.failed + summary.not_attempted, summary.total);
    }
    Ok(())
}

fn export_markdown(
    video: &Video,
    insight: &InsightResult,
    meta: Option<&InsightMeta>,
    coverage: Option<&TranscriptCoverage>,
) -> String {
    let heuristic = is_heuristic(insight);
    let mut md = String::new();
    md.push_str(&format!("# {}\n\n", video.title));
    md.push_str(&format!("- **Video URL:** {}\n", video.url));
    if let Some(category) = category_label(insight, meta) {
        md.push_str(&format!("- **Category:** {}\n", category));
    }
    if is_insufficient(insight, meta) {
        md.push_str("- **Quality Rating:** Not scored\n");
        md.push_str(&format!(
            "- **Slop Score:** {}\n",
            with_reason("Not enough information", meta.and_then(|m| m.info_reason.as_deref()))
        ));
    } else if heuristic {
        md.push_str(&format!("- **Quality Rating:** {}\n", UNREVIEWED_LABEL));
        md.push_str(&format!("- **Slop Score:** {}/100 (word-count estimate, not reviewed by an LLM)\n", insight.slop_score));
    } else {
        md.push_str(&format!("- **Quality Rating:** {}\n", insight.quality_rating));
        md.push_str(&format!("- **Slop Score:** {}/100\n", insight.slop_score));
    }
    if let Some(m) = meta.filter(|m| !m.topic_ok()) {
        md.push_str(&format!(
            "- **Topic:** {}\n",
            with_reason(m.topic_verdict.as_deref().unwrap_or("rejected"), m.topic_reason.as_deref())
        ));
    }
    md.push_str(&format!("- **Status:** {}\n", kept_status_text(insight, meta)));
    if let Some(note) = read_note(insight, meta) {
        md.push_str(&format!("- **Read:** {}\n", note));
    }
    if let Some(note) = transcript_source_note(coverage) {
        md.push_str(&format!("- **Transcript Source:** {}\n", note));
    }
    md.push_str(&format!("- **Analyzer:** {}\n\n", insight.analyzer_type));

    md.push_str(if heuristic { "## Summary (extracted sentences, no LLM)\n\n" } else { "## Executive Summary\n\n" });
    md.push_str(&format!("{}\n\n", insight.summary));

    if !insight.key_insights.is_empty() {
        md.push_str(if heuristic { "## Transcript Excerpts (no LLM)\n\n" } else { "## Core Insights\n\n" });
        for ins in &insight.key_insights {
            md.push_str(&format!("* {}\n", ins));
        }
        md.push('\n');
    }

    if !insight.takeaways.is_empty() {
        md.push_str(if heuristic { "## Further Transcript Excerpts (no LLM)\n\n" } else { "## Actionable Takeaways\n\n" });
        for t in &insight.takeaways {
            md.push_str(&format!("* {}\n", t));
        }
        md.push('\n');
    }

    if !insight.quotes.is_empty() {
        md.push_str(if heuristic { "## Transcript Sentences (no LLM)\n\n" } else { "## Notable Quotes\n\n" });
        for q in &insight.quotes {
            md.push_str(&format!("> {}\n\n", q));
        }
    }
    md
}

/// The object `process-video --json` prints. `slop_score` is null when
/// nothing was measured: for a video with too little content the 0 inside
/// `insight` is a placeholder. `is_worth_keeping` is the effective decision (quality, topic,
/// information and the owner's disqualification together); the flag inside
/// `insight` is the quality part alone.
fn insight_json(
    video: &Video,
    insight: &InsightResult,
    meta: Option<&InsightMeta>,
    coverage: Option<&TranscriptCoverage>,
) -> serde_json::Value {
    let not_scored = is_insufficient(insight, meta);
    let score_label = if not_scored {
        with_reason("Not enough information", meta.and_then(|m| m.info_reason.as_deref()))
    } else {
        insight.score_label()
    };
    serde_json::json!({
        "video_id": video.id,
        "engine": engine_kind(insight),
        "not_scored": not_scored,
        "slop_score": if not_scored { None } else { Some(insight.slop_score) },
        "score_label": score_label,
        "is_worth_keeping": outcome_of(insight, meta) == VideoOutcome::Kept,
        "status": kept_status_text(insight, meta),
        "topic_verdict": meta.and_then(|m| m.topic_verdict.clone()),
        "topic_reason": meta.and_then(|m| m.topic_reason.clone()),
        "info_status": meta.and_then(|m| m.info_status.clone()),
        "info_reason": meta.and_then(|m| m.info_reason.clone()),
        "excluded": meta.is_some_and(|m| m.excluded_manual),
        "excluded_reason": meta.and_then(|m| m.excluded_reason.clone()),
        "sections_total": meta.and_then(|m| m.sections_total),
        "transcript_chars": meta.and_then(|m| m.transcript_chars),
        "transcript_source": coverage.and_then(|c| c.source.clone()),
        "video_duration_sec": coverage.and_then(|c| c.video_duration_sec),
        "covered_until_sec": coverage.and_then(|c| c.covered_until_sec),
        "video": video,
        "insight": insight,
    })
}

// ---------------------------------------------------------------------------
// exclude / include / merge-db
// ---------------------------------------------------------------------------

fn title_of(db: &Database, video_id: &str) -> anyhow::Result<String> {
    Ok(db.get_video(video_id)?.map(|v| v.title).unwrap_or_else(|| "(not in the videos table)".to_string()))
}

/// `ytc exclude`: the owner's decision, stored on the insight row.
fn exclude_video(db: &Database, input: &str, reason: Option<&str>, wiki_dir: &Path) -> anyhow::Result<()> {
    let video_id = extract_video_id(input)?;
    let Some(before) = db.get_insight_meta(&video_id)? else {
        anyhow::bail!(
            "Video {} has no analysis in this database, so there is nothing to disqualify. Nothing was changed.",
            video_id
        );
    };
    db.exclude_video(&video_id, reason)?;

    let title = title_of(db, &video_id)?;
    let stored_reason = db.get_insight_meta(&video_id)?.and_then(|m| m.excluded_reason);
    println!(
        "{} {}: {} [{}]",
        "✓".green(),
        with_reason(
            exclude_label(before.excluded_manual, before.excluded_reason.as_deref(), stored_reason.as_deref()),
            stored_reason.as_deref()
        ),
        title.bold(),
        video_id.cyan()
    );
    println!("  It is never kept, whatever a later analysis says, and does not count in its channel's average.");
    if before.kept {
        println!(
            "  It was in the library: the next 'ytc wiki' stops listing it, and its page is moved out of the wiki with: {}",
            wiki_prune_command(wiki_dir)
        );
    } else {
        println!("  It was not in the library before, so the wiki does not change.");
    }
    println!("  Undo with: ytc include {}", video_id);
    Ok(())
}

/// `ytc include`: undoes `exclude`; the stored judgment decides again.
fn include_video(db: &Database, input: &str) -> anyhow::Result<()> {
    let video_id = extract_video_id(input)?;
    let Some(before) = db.get_insight_meta(&video_id)? else {
        anyhow::bail!("Video {} has no analysis in this database. Nothing was changed.", video_id);
    };
    let title = title_of(db, &video_id)?;
    if !before.excluded_manual {
        println!("Video {} [{}] is not disqualified. Nothing was changed.", title.bold(), video_id.cyan());
        return Ok(());
    }
    db.include_video(&video_id)?;

    println!("{} Not disqualified any more: {} [{}]", "✓".green(), title.bold(), video_id.cyan());
    let after = db.get_insight_meta(&video_id)?;
    match (db.get_insight(&video_id)?, after) {
        (Some(insight), Some(meta)) if meta.kept => {
            println!(
                "  It is in the library again ({}). Run 'ytc wiki' to write its page.",
                score_with_engine(&insight, Some(&meta))
            );
        }
        (Some(insight), Some(meta)) => println!(
            "  It stays out of the library on its stored judgment: {}.",
            rejection_note(&insight, Some(&meta)).unwrap_or_else(|| format!(
                "below the quality cutoff ({})",
                score_with_engine(&insight, Some(&meta))
            ))
        ),
        _ => {}
    }
    Ok(())
}

/// True when both paths name the same existing file, however they are spelled.
fn is_same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// `ytc merge-db`: merges `other` into the open database `db` (the file
/// `db_path`). Refuses, before anything is touched, a file that does not
/// exist and the database itself.
fn merge_db(db: &Database, db_path: &Path, other: &Path) -> anyhow::Result<models::MergeStats> {
    if !other.is_file() {
        anyhow::bail!("{} does not exist or is not a file. Nothing was changed.", other.display());
    }
    if is_same_file(db_path, other) {
        anyhow::bail!(
            "{} is the database in use; a database cannot be merged into itself. Nothing was changed.",
            other.display()
        );
    }
    db.merge_from(other)
}

// ---------------------------------------------------------------------------
// Channel lookup
// ---------------------------------------------------------------------------

/// Finds a tracked channel without a request: by its ID, by a URL containing
/// `/channel/<id>`, or by its exact title (case-insensitive, only when a
/// single channel has that title).
fn find_tracked_channel<'a>(channels: &'a [Channel], input: &str) -> Option<&'a Channel> {
    let wanted = input.trim();
    if wanted.is_empty() {
        return None;
    }
    if let Some(c) = channels
        .iter()
        .find(|c| c.id == wanted || wanted.contains(&format!("/channel/{}", c.id)))
    {
        return Some(c);
    }
    let mut by_title = channels.iter().filter(|c| c.title.trim().to_lowercase() == wanted.to_lowercase());
    match (by_title.next(), by_title.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

/// (channel id, title) of the channel the user means: a tracked channel when
/// the input names one, otherwise whatever YouTube resolves the input to.
async fn tracked_or_resolved(db: &Database, pipeline: &IngestionPipeline, input: &str) -> anyhow::Result<(String, String)> {
    let channels = db.list_channels()?;
    if let Some(c) = find_tracked_channel(&channels, input) {
        return Ok((c.id.clone(), c.title.clone()));
    }
    rss::resolve_channel(&pipeline.http_client, input).await
}

// ---------------------------------------------------------------------------
// process-video --json
// ---------------------------------------------------------------------------

/// Arguments for the worker process of `process-video --json`: the same
/// command line without the flag.
fn args_without_json(args: impl Iterator<Item = std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    args.filter(|a| a != "--json").collect()
}

/// `process-video --json`: the ingestion prints its progress with println!, so
/// it runs in a child process whose stdout is this process's stderr. Afterwards
/// the stored row is read back and printed as the only thing on stdout.
fn process_video_json(db_path: &Path, input: &str) -> anyhow::Result<()> {
    let video_id = extract_video_id(input)?;

    let exe = std::env::current_exe()?;
    let status = std::process::Command::new(exe)
        .args(args_without_json(std::env::args_os().skip(1)))
        // The worker must use this very file, also when --db was not given.
        .env("YTC_DB", absolute_for_display(db_path))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(std::io::stderr()))
        .status()?;
    if !status.success() {
        anyhow::bail!("process-video failed ({}); see the messages above. Nothing was printed to stdout.", status);
    }

    let db = Database::open(db_path)?;
    let video = db
        .get_video(&video_id)?
        .ok_or_else(|| anyhow::anyhow!("Video {} is not in the database after processing", video_id))?;
    let insight = db
        .get_insight(&video_id)?
        .ok_or_else(|| anyhow::anyhow!("No insight is stored for {} after processing", video_id))?;
    let meta = db.get_insight_meta(&video_id)?;
    let coverage = db.get_transcript_coverage(&video_id)?;

    println!("{}", serde_json::to_string(&insight_json(&video, &insight, meta.as_ref(), coverage.as_ref()))?);
    Ok(())
}

// ---------------------------------------------------------------------------
// reanalyze
// ---------------------------------------------------------------------------

/// What `ytc reanalyze` was asked to work on.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReanalyzeSelection {
    /// One video, named by ID or URL.
    Video(String),
    /// Every video whose analyzer type starts with the prefix.
    AnalyzerPrefix(String),
    /// Every analysed video.
    All,
}

impl ReanalyzeSelection {
    fn describe(&self) -> String {
        match self {
            Self::Video(input) => format!("video {}", input),
            Self::AnalyzerPrefix(prefix) => format!("analyzer type starts with '{}'", prefix),
            Self::All => "every analysed video".to_string(),
        }
    }
}

/// The selection the command line asks for. clap already makes sure exactly
/// one is given; the error is for callers that bypass it.
fn reanalyze_selection(video_id: Option<String>, analyzer: Option<String>, all: bool) -> anyhow::Result<ReanalyzeSelection> {
    let mut given: Vec<ReanalyzeSelection> = Vec::new();
    given.extend(video_id.map(ReanalyzeSelection::Video));
    if let Some(prefix) = analyzer {
        let prefix = prefix.trim();
        if prefix.is_empty() {
            anyhow::bail!("--analyzer needs a non-empty prefix, e.g. 'heuristic' or 'llm:'");
        }
        given.push(ReanalyzeSelection::AnalyzerPrefix(prefix.to_string()));
    }
    given.extend(all.then_some(ReanalyzeSelection::All));
    match (given.pop(), given.is_empty()) {
        (Some(selection), true) => Ok(selection),
        _ => anyhow::bail!("Give exactly one of: a video ID, --analyzer <prefix>, --all"),
    }
}

/// LLM requests needed to read a transcript of `chars` completely: one when
/// it fits a single request, otherwise one per section plus the final
/// judgment. Very long videos need a few more to condense the notes.
///
/// `chars` is the length in UTF-8 bytes (`str::len`): that is the unit of
/// `ContextBudget` and of the analyzer's own comparison, whatever the names say.
fn llm_requests_for(chars: usize, single_chars: usize, section_chars: usize) -> usize {
    if chars <= single_chars || section_chars == 0 {
        1
    } else {
        chars.div_ceil(section_chars) + 1
    }
}

/// Length of a stored transcript in the unit of the analyzer's budget (UTF-8
/// bytes of the trimmed text, not characters: a Cyrillic, Greek or CJK
/// transcript takes two or three bytes per character).
fn transcript_budget_len(text: &str) -> usize {
    text.trim().len()
}

/// What a `reanalyze` run will do, decided before the first LLM request.
#[derive(Debug, Default, PartialEq, Eq)]
struct ReanalyzePlan {
    /// Videos to analyse, in the order of the selection.
    targets: Vec<String>,
    /// Disqualified by hand: skipped by every bulk selection.
    disqualified: Vec<String>,
    /// Selected, but no transcript is stored: cannot be analysed from storage.
    no_transcript: Vec<String>,
    /// Forecast: LLM requests for all targets, and targets read in sections.
    requests: usize,
    in_sections: usize,
    /// Videos that would have been analysed without --limit.
    beyond_limit: usize,
}

/// Sorts the selected videos into what is analysed and what is left out.
///
/// `matching` is in the order of the selection (for a bulk selection: the
/// video analysed longest ago first), and `limit` cuts it after that many
/// videos that can be analysed.
/// `single` is the selection of one video by name: it is always attempted, so
/// that the exact error is shown, and never skipped.
/// `transcript_len` returns the stored transcript's length in budget units,
/// None when no transcript is stored.
fn plan_reanalyze(
    matching: &[String],
    metas: &std::collections::HashMap<String, InsightMeta>,
    single: bool,
    limit: Option<usize>,
    budget: (usize, usize),
    mut transcript_len: impl FnMut(&str) -> anyhow::Result<Option<usize>>,
) -> anyhow::Result<ReanalyzePlan> {
    let (single_chars, section_chars) = budget;
    let mut plan = ReanalyzePlan::default();
    for id in matching {
        if !single && metas.get(id).is_some_and(|m| m.excluded_manual) {
            // The owner's decision stands whatever a new analysis says.
            plan.disqualified.push(id.clone());
            continue;
        }
        match transcript_len(id)?.filter(|len| *len > 0) {
            // Not stopped at the limit: the list of videos without a
            // transcript must be complete, and the count of what is left exact.
            Some(_) if !single && limit.is_some_and(|n| plan.targets.len() >= n) => plan.beyond_limit += 1,
            Some(len) => {
                let needed = llm_requests_for(len, single_chars, section_chars);
                plan.requests += needed;
                if needed > 1 {
                    plan.in_sections += 1;
                }
                plan.targets.push(id.clone());
            }
            // Named directly: attempted, so that the exact error is shown.
            None if single => plan.targets.push(id.clone()),
            None => plan.no_transcript.push(id.clone()),
        }
    }
    Ok(plan)
}

/// How many LLM failures in a row end a bulk run; the crawl uses the same number.
const LLM_FAILURES_BEFORE_STOP: usize = 2;

/// `Some(engine_down)` when the error is the LLM not answering (connection
/// refused, timeout, HTTP error), None for anything else: an unusable answer
/// or a missing transcript is the video's own problem.
fn llm_failure_of(err: &anyhow::Error) -> Option<bool> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<analyzer::LlmUnavailable>())
        .map(|unavailable| unavailable.engine_down)
}

/// The stop rule of the crawl, for one more result of a bulk re-analysis.
/// `llm_failure` is `llm_failure_of` for a failed video and None for a
/// finished one or another kind of error. Returns the new number of LLM
/// failures in a row and whether the run must stop: at once when the engine
/// is down, otherwise after `LLM_FAILURES_BEFORE_STOP` such failures in a row.
/// A hung server would otherwise cost one full request timeout per remaining video.
fn llm_stop_rule(llm_failure: Option<bool>, failures_in_a_row: usize) -> (usize, bool) {
    match llm_failure {
        Some(engine_down) => {
            let in_a_row = failures_in_a_row + 1;
            (in_a_row, engine_down || in_a_row >= LLM_FAILURES_BEFORE_STOP)
        }
        None => (0, false),
    }
}

/// The line about the videos --limit left out.
fn beyond_limit_line(beyond_limit: usize) -> String {
    format!(
        "{} further selected video(s) are left for a later run (--limit): run the same command again, it continues with them.",
        beyond_limit
    )
}

/// What to do after a run that stopped because of the LLM.
const STOP_RESUME_LINE: &str =
    "When the LLM works again, start the same command again: it continues with the videos not analysed again yet.";

async fn reanalyze(
    db: &Database,
    pipeline: &IngestionPipeline,
    selection: ReanalyzeSelection,
    limit: Option<usize>,
) -> anyhow::Result<()> {
    let (_, model, url) = pipeline.analyzer.describe();
    if !pipeline.analyzer.has_llm() {
        anyhow::bail!("reanalyze needs an LLM and none is configured (--llm-url / YTC_LLM_URL). Nothing was changed.");
    }
    if !pipeline.analyzer.llm_reachable().await {
        anyhow::bail!("The LLM at {} is not reachable (no answer from /health or /models within 2 s). Nothing was changed.", url);
    }

    let metas = db.list_insight_meta()?;
    let single = matches!(selection, ReanalyzeSelection::Video(_));
    let matching: Vec<String> = match &selection {
        ReanalyzeSelection::Video(input) => vec![extract_video_id(input)?],
        ReanalyzeSelection::AnalyzerPrefix(prefix) => db.list_analysed_video_ids(Some(prefix))?,
        ReanalyzeSelection::All => db.list_analysed_video_ids(None)?,
    };
    if !single {
        println!("{} video(s) selected: {}.", matching.len(), selection.describe());
    }

    // What is really ahead: which videos, and how many LLM requests reading
    // every transcript completely takes.
    let budget = pipeline.analyzer.context_budget();
    let plan = plan_reanalyze(
        &matching,
        &metas,
        single,
        limit,
        (budget.single_chars, budget.section_chars),
        |id: &str| Ok(db.get_transcript(id)?.map(|t| transcript_budget_len(&t.text))),
    )?;
    let ReanalyzePlan { targets, disqualified, no_transcript, requests, in_sections, beyond_limit } = plan;

    if single {
        for id in matching.iter().filter(|id| metas.get(*id).is_some_and(|m| m.excluded_manual)) {
            println!(
                "{} {} is disqualified by hand: it is analysed again and stays out of the library until 'ytc include {}'.",
                "Note:".yellow().bold(),
                id,
                id
            );
        }
    }
    if !disqualified.is_empty() {
        println!(
            "{} {} disqualified video(s) skipped (undo with 'ytc include <video id>'): {}",
            "Note:".yellow().bold(),
            disqualified.len(),
            truncate_chars(&disqualified.join(", "), 120)
        );
    }
    if !no_transcript.is_empty() {
        // The complete list, not a cut one: these IDs are what the owner
        // needs to fetch the videos again.
        println!(
            "{} {} selected video(s) cannot be analysed again: no transcript is stored for them. 'ytc process-video <video id>' fetches the transcript and judges the video. IDs: {}",
            "Note:".yellow().bold(),
            no_transcript.len(),
            no_transcript.join(", ")
        );
    }
    if beyond_limit > 0 {
        println!("{}", beyond_limit_line(beyond_limit));
    }
    if targets.is_empty() {
        println!("Nothing to re-analyse.");
        return Ok(());
    }

    println!(
        "{}",
        format!(
            "Re-analysing {} video(s) from stored transcripts with LLM {} (quality cutoff: slop <= {})...",
            targets.len(),
            model,
            pipeline.max_slop
        )
        .cyan()
        .bold()
    );
    println!(
        "Every transcript is read completely: about {} LLM request(s) in total, {} video(s) are read in sections. One request takes from seconds to a few minutes depending on the model and the hardware, so a long video can take several minutes; the total time is not known in advance. Ctrl+C stops between requests; finished videos stay stored.",
        requests, in_sections
    );

    let total = targets.len();
    let mut counts = OutcomeCounts::default();
    let (mut done, mut failed, mut attempted) = (0usize, 0usize, 0usize);
    let mut llm_failures_in_a_row = 0usize;
    for (idx, id) in targets.iter().enumerate() {
        attempted += 1;
        let title = title_of(db, id)?;
        let before = match db.get_insight(id)? {
            Some(stored) => {
                let meta = db.get_insight_meta(id)?;
                format!(
                    "{} ({})",
                    score_with_engine(&stored, meta.as_ref()),
                    if outcome_of(&stored, meta.as_ref()) == VideoOutcome::Kept { "kept" } else { "not kept" }
                )
            }
            None => "no insight".to_string(),
        };
        println!("\n[{}/{}] {} {}", idx + 1, total, id.cyan(), truncate_chars(&title, 60));

        match pipeline.reanalyze_video(id).await {
            Ok(after) => {
                done += 1;
                llm_failures_in_a_row = 0;
                let meta = db.get_insight_meta(id)?;
                let meta = meta.as_ref();
                let outcome = outcome_of(&after, meta);
                counts.add(outcome);
                let decision = match (outcome, rejection_note(&after, meta)) {
                    (VideoOutcome::Kept, _) => "kept".green(),
                    (VideoOutcome::Insufficient, _) => "not kept".yellow(),
                    (VideoOutcome::BelowCutoff, _) => "not kept: below the quality cutoff".red(),
                    (_, Some(note)) => format!("not kept: {}", note).red(),
                    (_, None) => "not kept".red(),
                };
                println!(
                    "    {} before: {}  ->  after: {} ({})",
                    "✓".green(),
                    before,
                    score_with_engine(&after, meta),
                    decision
                );
                if let Some(note) = read_note(&after, meta) {
                    println!("      {}", format!("read: {}", note).dimmed());
                }
            }
            Err(e) => {
                failed += 1;
                println!("    {} unchanged ({}): {:#}", "✗".red(), before, e);
                // Same rule as the crawl: at once when the engine is down,
                // otherwise after two LLM failures in a row (a hung server, a
                // wrong model name or key fail the same way for every video).
                let llm_failure = llm_failure_of(&e);
                let stop;
                (llm_failures_in_a_row, stop) = llm_stop_rule(llm_failure, llm_failures_in_a_row);
                if stop {
                    let why = if llm_failure == Some(true) {
                        "The LLM stopped answering".to_string()
                    } else {
                        format!("The LLM failed for {} videos in a row", llm_failures_in_a_row)
                    };
                    println!(
                        "    {} {}; the run stops here and the remaining {} video(s) are left unchanged. {}",
                        "⚠".yellow(),
                        why,
                        total - attempted,
                        STOP_RESUME_LINE
                    );
                    break;
                }
            }
        }
    }

    println!(
        "\nRe-analysed {} of {} video(s): {}; {} failed (stored insight unchanged), {} not attempted.",
        done,
        total,
        counts.line(),
        failed,
        total - attempted
    );
    if done > 0 {
        println!(
            "Wiki pages are not rewritten by this command. Bring the wiki in line with: {}",
            wiki_prune_command(&pipeline.wiki_dir)
        );
    }
    if failed > 0 {
        anyhow::bail!("{} of {} video(s) could not be re-analysed", failed + (total - attempted), total);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// harvest-mentions
// ---------------------------------------------------------------------------

/// One creator name with the kept videos whose insight lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MentionGroup {
    name: String,
    /// Titles of the videos mentioning the name; `videos.len()` is the count.
    videos: Vec<String>,
}

/// Lower-cased letters and digits only, so "X Collective" equals "x-collective".
fn normalize_name(name: &str) -> String {
    name.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// Groups (creator name, video title) pairs by name, case-insensitively. The
/// spelling seen first is kept, a video counts once per name, and the result
/// is sorted by number of videos (highest first), then by name.
fn group_mentions(pairs: &[(String, String)]) -> Vec<MentionGroup> {
    let mut groups: Vec<MentionGroup> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (raw_name, video_title) in pairs {
        let name = raw_name.trim().trim_start_matches('@').trim();
        if name.is_empty() {
            continue;
        }
        let pos = *index.entry(name.to_lowercase()).or_insert_with(|| {
            groups.push(MentionGroup { name: name.to_string(), videos: Vec::new() });
            groups.len() - 1
        });
        if !groups[pos].videos.contains(video_title) {
            groups[pos].videos.push(video_title.clone());
        }
    }
    groups.sort_by(|a, b| {
        b.videos
            .len()
            .cmp(&a.videos.len())
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    groups
}

/// The tracked channel whose title is this name (ignoring case, spaces and
/// punctuation). Whole-name comparison: a channel called "Go" does not match
/// every name containing "go".
fn tracked_channel_for_name<'a>(channels: &'a [Channel], name: &str) -> Option<&'a Channel> {
    let wanted = normalize_name(name);
    if wanted.is_empty() {
        return None;
    }
    channels.iter().find(|c| normalize_name(&c.title) == wanted)
}

/// The handle to try for a mentioned name: "@" + the name without spaces.
/// None when the name has characters a handle cannot contain or is too short
/// to be a handle (YouTube handles have at least 3 characters).
fn handle_guess(name: &str) -> Option<String> {
    let compact: String = name.trim().trim_start_matches('@').chars().filter(|c| !c.is_whitespace()).collect();
    let valid = compact.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'));
    (valid && compact.chars().count() >= 3).then(|| format!("@{}", compact))
}

/// True when a resolved channel title plausibly is the mentioned name: equal
/// after normalisation, or one contains the other and the shorter has at least
/// 4 characters ("Example" / "Alice Example").
fn titles_match(mentioned: &str, resolved: &str) -> bool {
    let a = normalize_name(mentioned);
    let b = normalize_name(resolved);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a == b {
        return true;
    }
    let (short, long) = if a.chars().count() <= b.chars().count() { (&a, &b) } else { (&b, &a) };
    short.chars().count() >= 4 && long.contains(short.as_str())
}

async fn harvest_mentions(
    db: &Database,
    pipeline: &IngestionPipeline,
    resolve: usize,
    add: bool,
    limit: usize,
) -> anyhow::Result<()> {
    println!("\n{}", "===========================================================================".cyan());
    println!("  {}", "CREATOR NAMES LISTED IN KEPT INSIGHTS".bold());
    println!("  Source: the 'mentioned creators' field the analysis stored for each kept video.");
    println!("  Transcripts are not scanned again. Counts are videos per name.");
    println!("---------------------------------------------------------------------------");

    let groups = group_mentions(&db.get_all_mentioned_creators()?);
    if groups.is_empty() {
        println!("  No creator names are stored in kept insights yet.\n");
        return Ok(());
    }
    let tracked = db.list_channels()?;

    let shown = if limit == 0 { groups.len() } else { limit.min(groups.len()) };
    for g in groups.iter().take(shown) {
        let status = match tracked_channel_for_name(&tracked, &g.name) {
            Some(c) => format!("tracked as '{}' ({})", c.title, c.id).green(),
            None => "not tracked".yellow(),
        };
        println!("  {:>3}x  {}  [{}]", g.videos.len(), g.name.bold(), status);
        for title in g.videos.iter().take(5) {
            println!("          • {}", truncate_chars(title, 80).dimmed());
        }
        if g.videos.len() > 5 {
            println!("          … and {} more video(s)", g.videos.len() - 5);
        }
    }
    println!("{}", "-".repeat(75).dimmed());
    let untracked: Vec<&MentionGroup> = groups
        .iter()
        .filter(|g| tracked_channel_for_name(&tracked, &g.name).is_none())
        .collect();
    println!(
        "  {} name(s), {} shown, {} not matching a tracked channel title.",
        groups.len(),
        shown,
        untracked.len()
    );

    if resolve == 0 {
        println!("  Offline report: no name was looked up. Use --resolve N to try the top N untracked names.\n");
        return Ok(());
    }

    println!(
        "\n  Looking up the top {} untracked name(s) by handle guess{}:",
        resolve.min(untracked.len()),
        if add { " (matching channels are added)" } else { " (nothing is added without --add)" }
    );
    let (mut resolved, mut added) = (0usize, 0usize);
    for (i, g) in untracked.iter().take(resolve).enumerate() {
        let Some(handle) = handle_guess(&g.name) else {
            println!("  • {}: no handle can be guessed from this name; not looked up", g.name.bold());
            continue;
        };
        if i > 0 {
            // Stay polite between lookups.
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        match rss::resolve_channel(&pipeline.http_client, &handle).await {
            Ok((id, title)) => {
                resolved += 1;
                let matches = titles_match(&g.name, &title);
                let already = db.get_channel(&id)?.is_some();
                let verdict = if already {
                    "already tracked under this channel ID".green()
                } else if !matches {
                    format!("title differs from the name: not added (check it, then: ytc add-channel {})", id).yellow()
                } else if add {
                    db.upsert_channel(&new_active_channel(&id, &title))?;
                    added += 1;
                    "added to the watchlist as 'active'".green()
                } else {
                    format!("title matches (add with --add or: ytc add-channel {})", id).cyan()
                };
                println!("  • {} -> {} '{}' ({}): {}", g.name.bold(), handle, title, id.dimmed(), verdict);
            }
            Err(e) => println!("  • {} -> {}: no channel found ({})", g.name.bold(), handle, e),
        }
    }
    println!(
        "\n  {} of {} lookup(s) found a channel; {} added. A handle guess can hit a different creator: check the titles.\n",
        resolved,
        resolve.min(untracked.len()),
        added
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// watch
// ---------------------------------------------------------------------------

struct WatchOptions {
    interval_minutes: u64,
    limit_per_channel: usize,
    delay_secs: u64,
    discover: bool,
    queries: Vec<String>,
    discover_limit: usize,
    discover_delay_secs: u64,
    wiki_dir: PathBuf,
}

fn is_llm_unavailable(err: &anyhow::Error) -> bool {
    err.chain().any(|c| c.downcast_ref::<analyzer::LlmUnavailable>().is_some())
        || err.to_string().contains("LLM engine not available")
}

/// Stop request of the watcher, shared with the Ctrl+C listener task.
#[derive(Default)]
struct StopSignal {
    /// Ctrl+C was pressed.
    requested: AtomicBool,
    /// The watch loop saw the request and is returning.
    acknowledged: AtomicBool,
    /// The wiki is being written (a short, blocking step): the listener waits
    /// for it instead of ending the process between two half-written pages.
    writing_wiki: AtomicBool,
}

/// Runs `step` until it finishes or a stop is requested. None means stopped:
/// the step was dropped at its current await point and the request is marked
/// as acknowledged.
///
/// One signal future inside a `select!` around the sleep is not enough: on
/// Unix tokio's SIGINT handler stays installed after the first use, so a
/// Ctrl+C during a later crawl was swallowed and never seen.
async fn run_unless_stopped<F: std::future::Future>(stop: &StopSignal, step: F) -> Option<F::Output> {
    tokio::pin!(step);
    loop {
        if stop.requested.load(Ordering::SeqCst) {
            stop.acknowledged.store(true, Ordering::SeqCst);
            return None;
        }
        tokio::select! {
            out = &mut step => return Some(out),
            _ = tokio::time::sleep(STOP_POLL) => {}
        }
    }
}

/// Listens for Ctrl+C on a runtime thread for as long as the watcher runs.
/// The watch loop itself cannot do that everywhere: transcript extraction
/// (yt-dlp, Whisper) blocks its thread for minutes without an await point.
/// When the loop has not acknowledged the request after `STOP_GRACE`, the
/// process is ended from here, which is what Ctrl+C did before on Windows.
fn spawn_ctrl_c_listener(stop: Arc<StopSignal>) {
    tokio::spawn(async move {
        // Without a handler the platform default applies: Ctrl+C ends the process.
        if tokio::signal::ctrl_c().await.is_err() {
            return;
        }
        stop.requested.store(true, Ordering::SeqCst);
        let asked = std::time::Instant::now();
        loop {
            tokio::time::sleep(STOP_POLL).await;
            if stop.acknowledged.load(Ordering::SeqCst) {
                return;
            }
            if asked.elapsed() >= STOP_GRACE && !stop.writing_wiki.load(Ordering::SeqCst) {
                println!("\n{} Watcher stopped.", "👋".yellow().bold());
                // 130 = ended by Ctrl+C, the shell convention.
                std::process::exit(130);
            }
        }
    });
}

/// The watch cycle: crawl, optional discovery, wiki, wait. Every number comes
/// from `options`; nothing is fixed here.
async fn run_watch_loop(pipeline: &IngestionPipeline, options: &WatchOptions) -> anyhow::Result<()> {
    println!("\n{}", "===========================================================================".cyan());
    println!("  {}", "YTC WATCHER".bold());
    println!("  Cycle Interval:      {} minute(s) after each cycle", options.interval_minutes);
    println!("  Max Videos/Channel:  {}", options.limit_per_channel);
    println!("  Delay Between Videos: {} s (crawl starts at {} s and backs off on HTTP 429)", options.delay_secs, options.delay_secs.max(10));
    if options.discover {
        println!(
            "  Discovery:           {} query(ies), {} candidate(s) each, {} s between candidates",
            options.queries.len(),
            options.discover_limit,
            options.discover_delay_secs
        );
        for q in &options.queries {
            println!("                       - {}", q);
        }
    } else {
        println!("  Discovery:           off");
    }
    println!("  Analysis Engine:     {}", pipeline.analyzer.engine_label());
    println!("  Quality Cutoff:      slop score <= {}", pipeline.max_slop);
    println!("  Wiki Directory:      {:?}", options.wiki_dir);
    println!("---------------------------------------------------------------------------");

    let stop = Arc::new(StopSignal::default());
    spawn_ctrl_c_listener(stop.clone());

    let mut cycle: u64 = 1;
    // Left only through a stop request; every long await is wrapped so that
    // Ctrl+C ends the loop during any step, not only during the wait.
    'watch: loop {
        println!(
            "\n{} [{}] Starting cycle #{} ...",
            "⏱".cyan(),
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            cycle
        );

        // 0. Ask the LLM first. Without this check a cycle fetched every feed and
        // one transcript from YouTube (or ran Whisper) before it noticed, and did
        // so again in every cycle while the LLM stayed down.
        let mut llm_down = false;
        if pipeline.analyzer.has_llm() {
            let Some(reachable) = run_unless_stopped(&stop, pipeline.analyzer.llm_reachable()).await else {
                break 'watch;
            };
            llm_down = !reachable;
        }

        // 1. Crawl. No wiki step here: it is written once, after discovery.
        if llm_down {
            let (_, _, url) = pipeline.analyzer.describe();
            println!(
                "  {} The LLM at {} is not reachable: crawl skipped in this cycle, no request is sent to YouTube.",
                "⚠".yellow(),
                url
            );
        } else {
            let crawl = pipeline.run_crawl_with_wiki(None, options.limit_per_channel, options.delay_secs, None);
            let Some(result) = run_unless_stopped(&stop, crawl).await else {
                break 'watch;
            };
            match result {
                Ok(summary) => println!("  Cycle #{} crawl: {}", cycle, crawl_summary_line(&summary)),
                Err(e) => {
                    println!("  {} Crawl ended with an error: {:#}", "⚠".yellow(), e);
                    llm_down = is_llm_unavailable(&e);
                }
            }
        }

        // 2. Discovery, pointless while the LLM is down.
        if options.discover && llm_down {
            println!("  Discovery skipped in this cycle: the LLM is not available.");
        } else if options.discover {
            for q in &options.queries {
                println!("\n{} Discovery query: '{}'", "🔍".cyan(), q);
                let discovery = pipeline.run_discover(q, options.discover_limit, options.discover_delay_secs);
                let Some(result) = run_unless_stopped(&stop, discovery).await else {
                    break 'watch;
                };
                match result {
                    Ok(found) => {
                        let kept = found.items.iter().filter(|i| i.is_worth_keeping).count();
                        println!("  {} candidate(s) analysed, {} kept", found.items.len(), kept);
                    }
                    Err(e) => {
                        println!("  {} Discovery query failed: {:#}", "⚠".yellow(), e);
                        if is_llm_unavailable(&e) {
                            break;
                        }
                    }
                }
            }
        }

        // 3. Wiki. Scoped so the connection is closed before the wait. It has no
        // await point, so the listener is told to let it finish (seconds) and the
        // stop request is looked at right before it; the wait below sees one
        // that arrived meanwhile.
        if stop.requested.load(Ordering::SeqCst) {
            stop.acknowledged.store(true, Ordering::SeqCst);
            break 'watch;
        }
        println!("\n{} Writing the wiki to {:?} ...", "📚".cyan(), options.wiki_dir);
        stop.writing_wiki.store(true, Ordering::SeqCst);
        let written = pipeline
            .db()
            .and_then(|db| wiki::WikiGenerator::generate_with_options(&db, &options.wiki_dir, false));
        stop.writing_wiki.store(false, Ordering::SeqCst);
        match written {
            Ok(stats) => {
                println!(
                    "  Wiki written: {} insights, {} topics, {} creators.",
                    stats.insights_count, stats.topics_count, stats.creators_count
                );
                if stats.stale_pages > 0 {
                    // Names the folder this loop wrote to, not the default one.
                    println!(
                        "  {} stale generated page(s) left in place (move them out of the wiki with: {}).",
                        stats.stale_pages,
                        wiki_prune_command(&options.wiki_dir)
                    );
                }
            }
            Err(e) => println!("  {} The wiki could not be written: {:#}", "⚠".yellow(), e),
        }

        let wait = std::time::Duration::from_secs(options.interval_minutes.saturating_mul(60));
        // None only for an interval too large for a calendar date.
        let next_run = chrono::Duration::from_std(wait)
            .ok()
            .and_then(|d| chrono::Utc::now().checked_add_signed(d))
            .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| format!("in {} minutes", options.interval_minutes));
        println!(
            "\n{} Cycle #{} finished. Next cycle: {} (Ctrl+C to stop).",
            "✓".green(),
            cycle,
            next_run
        );

        if run_unless_stopped(&stop, tokio::time::sleep(wait)).await.is_none() {
            break 'watch;
        }
        cycle += 1;
    }

    println!("\n{} Watcher stopped.", "👋".yellow().bold());
    Ok(())
}

// ---------------------------------------------------------------------------
// Insight card
// ---------------------------------------------------------------------------

fn print_insight_card(
    video: &Video,
    insight: &InsightResult,
    meta: Option<&InsightMeta>,
    coverage: Option<&TranscriptCoverage>,
) {
    let heuristic = is_heuristic(insight);
    let no_score = is_insufficient(insight, meta);
    let outcome = outcome_of(insight, meta);

    println!("\n{}", "=".repeat(75).dimmed());
    println!("  TITLE: {}", video.title.bold());
    println!("  URL:   {}", video.url.cyan());

    if let Some(category) = category_label(insight, meta) {
        println!("  CATEGORY: {}", category.blue().bold());
    }

    if no_score {
        println!("  SCORE: {}", "Not scored".yellow());
    } else if heuristic {
        println!(
            "  SCORE: {}/100 — {}",
            insight.slop_score,
            format!("{}: word-count estimate, not reviewed by an LLM", UNREVIEWED_LABEL).yellow()
        );
    } else {
        println!("  SCORE: {}/100 (quality only: 0 = dense, 100 = slop; the subject does not enter)", insight.slop_score);
    }

    // Shown only when it speaks against the video; "ok" needs no line.
    if let Some(m) = meta.filter(|m| !m.topic_ok()) {
        println!(
            "  TOPIC: {}",
            with_reason(m.topic_verdict.as_deref().unwrap_or("rejected"), m.topic_reason.as_deref()).red()
        );
    }

    let rating = if heuristic { UNREVIEWED_LABEL } else { insight.quality_rating.as_str() };
    let note = rejection_note(insight, meta).unwrap_or_default();
    let status = match outcome {
        VideoOutcome::Kept => format!("{} (kept in the library)", rating).green().bold(),
        VideoOutcome::BelowCutoff => format!("{} (not kept: below the quality cutoff)", rating).red().bold(),
        VideoOutcome::OffTopic => format!("{} (not kept)", note).red().bold(),
        VideoOutcome::Insufficient => {
            format!("{} (not kept; not slop, and not counted in the channel average)", note).yellow().bold()
        }
        VideoOutcome::Disqualified => {
            format!("{} (not kept; undo with: ytc include {})", note, video.id).red().bold()
        }
    };
    println!("  STATUS: {}", status);
    println!("  ENGINE: {}", insight.analyzer_type.dimmed());
    if let Some(note) = read_note(insight, meta) {
        println!("  READ:   {}", note);
    }
    if let Some(note) = transcript_source_note(coverage) {
        println!("  SOURCE: {}", note);
    }
    println!("{}", "-".repeat(75).dimmed());

    // A heuristic row holds sentences picked from the transcript, so its
    // sections are not called insights, takeaways or quotes.
    let heading = |llm: &'static str, no_llm: &'static str| if heuristic { no_llm } else { llm };

    println!("\n{}", heading("EXECUTIVE SUMMARY:", "SUMMARY (extracted sentences, no LLM):").bold());
    println!("  {}", insight.summary);

    if !insight.key_insights.is_empty() {
        println!("\n{}", heading("KEY INSIGHTS:", "TRANSCRIPT EXCERPTS (no LLM):").bold());
        for (i, ins) in insight.key_insights.iter().enumerate() {
            println!("  {}. {}", i + 1, ins);
        }
    }

    if !insight.takeaways.is_empty() {
        println!("\n{}", heading("ACTIONABLE TAKEAWAYS:", "FURTHER TRANSCRIPT EXCERPTS (no LLM):").bold());
        for (i, t) in insight.takeaways.iter().enumerate() {
            println!("  {}. {}", i + 1, t);
        }
    }

    if !insight.quotes.is_empty() {
        println!("\n{}", heading("NOTABLE QUOTES:", "TRANSCRIPT SENTENCES (no LLM):").bold());
        for q in &insight.quotes {
            println!("  • {}", q.italic());
        }
    }

    if !insight.topics.is_empty() {
        println!("\n{}", "CONCEPTS & TOPICS:".bold());
        println!("  {}", insight.topics.join("  •  ").cyan());
    }

    if !insight.mentioned_creators.is_empty() {
        println!("\n{}", "MENTIONED CREATORS / GUESTS:".bold());
        println!("  {}", insight.mentioned_creators.join(", ").yellow());
    }

    println!("{}\n", "=".repeat(75).dimmed());
}

fn extract_video_id(input: &str) -> anyhow::Result<String> {
    let trimmed = input.trim();

    let bare = regex::Regex::new(r"^[A-Za-z0-9_-]{11}$")?;
    if bare.is_match(trimmed) {
        return Ok(trimmed.to_string());
    }

    // The trailing group stops a longer path segment or parameter from being cut to 11 chars.
    let in_url = regex::Regex::new(
        r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
    )?;
    if let Some(m) = in_url.captures(trimmed).and_then(|c| c.get(1)) {
        return Ok(m.as_str().to_string());
    }

    anyhow::bail!("Could not parse an 11-character YouTube video ID from '{}'", input)
}

fn is_placeholder_title(title: &str) -> bool {
    title.starts_with("Video ")
}

fn is_placeholder_channel(channel_id: &str) -> bool {
    channel_id == "direct_ingest" || channel_id == "unknown"
}

/// Absolute form of the DB path for the startup message. Not canonicalized: on Windows
/// that would print the verbatim (`\\?\`) prefix.
fn absolute_for_display(path: &std::path::Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "AbCdEfGh123";

    #[test]
    fn extract_video_id_bare_id() {
        assert_eq!(extract_video_id(ID).unwrap(), ID);
        assert_eq!(extract_video_id("  AbCdEfGh123\n").unwrap(), ID);
    }

    #[test]
    fn extract_video_id_leading_hyphen() {
        assert_eq!(extract_video_id("-AbCdEfGh12").unwrap(), "-AbCdEfGh12");
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?v=-AbCdEfGh12").unwrap(),
            "-AbCdEfGh12"
        );
    }

    #[test]
    fn extract_video_id_url_forms() {
        let cases = [
            "https://www.youtube.com/watch?v=AbCdEfGh123&t=30s",
            "https://www.youtube.com/watch?feature=share&v=AbCdEfGh123",
            "https://youtu.be/AbCdEfGh123?t=5",
            "https://www.youtube.com/shorts/AbCdEfGh123",
            "https://www.youtube.com/embed/AbCdEfGh123",
            "https://www.youtube.com/live/AbCdEfGh123?si=abc",
            "https://www.youtube.com/shorts/AbCdEfGh123/",
            "https://www.youtube.com/watch?v=AbCdEfGh123#t=30",
            "youtu.be/AbCdEfGh123",
        ];
        for case in cases {
            assert_eq!(extract_video_id(case).unwrap(), ID, "input: {case}");
        }
    }

    #[test]
    fn extract_video_id_rejects_garbage() {
        for case in [
            "",
            "not a video id",
            "hello world",
            "dQw4w9WgXc",
            "https://www.youtube.com/@somechannel",
            "https://www.youtube.com/watch?v=AbCdEfGh123toolong",
        ] {
            assert!(extract_video_id(case).is_err(), "input: {case:?}");
        }
    }

    #[test]
    fn extract_video_id_cli_accepts_leading_hyphen() {
        for sub in ["show", "export", "process-video"] {
            let cli = Cli::try_parse_from(["ytc", sub, "-AbCdEfGh12"]);
            assert!(cli.is_ok(), "{sub} rejected a leading-hyphen id");
        }
    }

    // --- argument parsing of the options and commands added later ---

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("ytc").chain(args.iter().copied()))
    }

    #[test]
    fn global_max_slop_and_wiki_dir_are_parsed_before_the_subcommand() {
        let cli = parse(&["--max-slop", "30", "--wiki-dir", "D:/notes/wiki", "list"]).ok().expect("valid");
        assert_eq!(cli.max_slop, 30);
        assert_eq!(cli.wiki_dir, PathBuf::from("D:/notes/wiki"));

        assert_eq!(parse(&["--max-slop", "0", "list"]).ok().expect("0 is allowed").max_slop, 0);
        assert_eq!(parse(&["--max-slop", "100", "list"]).ok().expect("100 is allowed").max_slop, 100);
    }

    #[test]
    fn max_slop_outside_the_range_is_rejected() {
        for bad in ["101", "-1", "abc", "45.5"] {
            assert!(parse(&["--max-slop", bad, "list"]).is_err(), "accepted --max-slop {bad}");
        }
    }

    #[test]
    fn global_defaults_apply_without_flags_or_env() {
        // The env vars take part in parsing; only assert when they are not set here.
        if std::env::var_os("YTC_MAX_SLOP").is_some() || std::env::var_os("YTC_WIKI_DIR").is_some() {
            return;
        }
        let cli = parse(&["list"]).ok().expect("valid");
        assert_eq!(cli.max_slop, pipeline::KEEP_THRESHOLD);
        assert_eq!(cli.wiki_dir, PathBuf::from(pipeline::DEFAULT_WIKI_DIR));
    }

    #[test]
    fn subcommand_wiki_directory_wins_over_the_global_one() {
        let global = Path::new("./wiki");
        assert_eq!(effective_wiki_dir(Some(PathBuf::from("out")), global), PathBuf::from("out"));
        assert_eq!(effective_wiki_dir(None, global), PathBuf::from("./wiki"));
        assert_eq!(effective_wiki_dir(Some(PathBuf::new()), global), PathBuf::from("./wiki"));
    }

    #[test]
    fn prune_command_names_the_wiki_folder_as_an_absolute_path() {
        // Relative folders, the default one included, are tied to the current directory.
        for relative in [pipeline::DEFAULT_WIKI_DIR, "out", "./notes/wiki"] {
            let dir = absolute_dir(Path::new(relative));
            assert!(dir.is_absolute(), "{relative} -> {dir:?}");
            assert!(
                !dir.components().any(|c| matches!(c, std::path::Component::CurDir)),
                "{relative} -> {dir:?}"
            );
            assert_eq!(
                wiki_prune_command(Path::new(relative)),
                format!("ytc wiki --prune --wiki-dir \"{}\"", dir.display())
            );
        }
        assert!(absolute_dir(Path::new("./notes/wiki")).ends_with(Path::new("notes").join("wiki")));

        // An absolute folder is printed as given.
        let abs = std::env::temp_dir().join("ytc wiki");
        assert_eq!(absolute_dir(&abs), abs);
        assert_eq!(wiki_prune_command(&abs), format!("ytc wiki --prune --wiki-dir \"{}\"", abs.display()));
    }

    #[test]
    fn printed_prune_command_parses_and_targets_that_folder() {
        let dir = std::env::temp_dir().join("ytc wiki");
        let dir_text = dir.display().to_string();
        assert!(wiki_prune_command(&dir).starts_with("ytc wiki --prune --wiki-dir "));
        // The words of the printed command, as a shell hands them over.
        let cli = parse(&["wiki", "--prune", "--wiki-dir", &dir_text]).ok().expect("valid");
        assert_eq!(cli.wiki_dir, dir);
        match cli.command {
            Commands::Wiki { output, prune } => {
                assert!(prune);
                assert_eq!(effective_wiki_dir(output, &cli.wiki_dir), dir);
            }
            _ => panic!("not the wiki command"),
        }
    }

    #[test]
    fn global_options_are_accepted_before_and_after_the_subcommand() {
        let options = [
            "--db", "x.db", "--llm-url", "http://127.0.0.1:9/v1", "--llm-key", "k", "--llm-model", "m",
            "--max-slop", "30", "--wiki-dir", "notes",
        ];
        let before: Vec<&str> = options.iter().copied().chain(["crawl", "-d", "7"]).collect();
        let after: Vec<&str> = ["crawl", "-d", "7"].into_iter().chain(options.iter().copied()).collect();
        // Mixed: some before, some after, around the subcommand's own flags.
        let mixed: Vec<&str> = options[..6]
            .iter()
            .copied()
            .chain(["crawl"])
            .chain(options[6..].iter().copied())
            .chain(["-d", "7"])
            .collect();

        for args in [before, after, mixed] {
            let cli = parse(&args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
            assert_eq!(cli.db, Some(PathBuf::from("x.db")), "{args:?}");
            assert!(cli.db_short.is_none(), "{args:?}");
            assert_eq!(cli.llm_url.as_deref(), Some("http://127.0.0.1:9/v1"), "{args:?}");
            assert_eq!(cli.llm_key.as_deref(), Some("k"), "{args:?}");
            assert_eq!(cli.llm_model, "m", "{args:?}");
            assert_eq!(cli.max_slop, 30, "{args:?}");
            assert_eq!(cli.wiki_dir, PathBuf::from("notes"), "{args:?}");
            match cli.command {
                // -d after the subcommand stays the delay.
                Commands::Crawl { delay, .. } => assert_eq!(delay, 7, "{args:?}"),
                _ => panic!("not the crawl command"),
            }
        }
    }

    #[test]
    fn global_options_work_after_every_subcommand_shape() {
        // Positional that accepts hyphen values, flags with -d / -w / -o / -l shorts.
        for args in [
            vec!["process-video", "-AbCdEfGh12", "--max-slop", "30", "--wiki-dir", "notes"],
            vec!["process-video", "--max-slop", "30", "--wiki-dir", "notes", "-AbCdEfGh12"],
            vec!["watch", "-i", "5", "--max-slop", "30", "-d", "9", "--wiki-dir", "notes"],
            vec!["discover", "some query", "-d", "1", "--wiki-dir", "notes", "--max-slop", "30"],
            vec!["reanalyze", "--analyzer", "heuristic", "--max-slop", "30", "--wiki-dir", "notes"],
            vec!["serve", "--wiki-dir", "notes", "--max-slop", "30", "-p", "9000"],
            vec!["list", "--max-slop", "30", "--wiki-dir", "notes"],
        ] {
            let cli = parse(&args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
            assert_eq!(cli.max_slop, 30, "{args:?}");
            assert_eq!(cli.wiki_dir, PathBuf::from("notes"), "{args:?}");
            if let Commands::ProcessVideo { input, .. } = &cli.command {
                assert_eq!(input, "-AbCdEfGh12", "{args:?}");
            }
        }
        // Still validated after the subcommand.
        assert!(parse(&["crawl", "--max-slop", "101"]).is_err());
    }

    #[test]
    fn short_db_flag_still_works_before_the_subcommand_only() {
        let cli = parse(&["-d", "x.db", "crawl", "-d", "7"]).ok().expect("valid");
        assert_eq!(cli.db_short, Some(PathBuf::from("x.db")));
        assert!(cli.db.is_none());
        match cli.command {
            Commands::Crawl { delay, .. } => assert_eq!(delay, 7),
            _ => panic!("not the crawl command"),
        }
        // No subcommand-level -d on 'list': it must not be read as the database.
        assert!(parse(&["list", "-d", "x.db"]).is_err());
        assert!(parse(&["-d", "x.db", "--db", "y.db", "list"]).is_err(), "two databases");
    }

    #[test]
    fn subcommand_directory_flags_win_over_the_global_wiki_dir_in_both_positions() {
        for args in [
            ["--wiki-dir", "global", "wiki", "--output", "local"],
            ["wiki", "--wiki-dir", "global", "--output", "local"],
            ["wiki", "--output", "local", "--wiki-dir", "global"],
        ] {
            let cli = parse(&args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
            assert_eq!(cli.wiki_dir, PathBuf::from("global"), "{args:?}");
            match cli.command {
                Commands::Wiki { output, .. } => {
                    assert_eq!(effective_wiki_dir(output, &cli.wiki_dir), PathBuf::from("local"), "{args:?}")
                }
                _ => panic!("not the wiki command"),
            }
        }
        for args in [
            ["--wiki-dir", "global", "watch", "--wiki", "local"],
            ["watch", "--wiki-dir", "global", "--wiki", "local"],
            ["watch", "-w", "local", "--wiki-dir", "global"],
        ] {
            let cli = parse(&args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
            assert_eq!(cli.wiki_dir, PathBuf::from("global"), "{args:?}");
            match cli.command {
                Commands::Watch { wiki, .. } => {
                    assert_eq!(effective_wiki_dir(wiki, &cli.wiki_dir), PathBuf::from("local"), "{args:?}")
                }
                _ => panic!("not the watch command"),
            }
        }
        // Without the local flag the global one is used, wherever it stands.
        let cli = parse(&["watch", "--wiki-dir", "global"]).ok().expect("valid");
        match cli.command {
            Commands::Watch { wiki, .. } => assert_eq!(effective_wiki_dir(wiki, &cli.wiki_dir), PathBuf::from("global")),
            _ => panic!("not the watch command"),
        }
        // 'crawl --wiki' is a switch and must not swallow or be confused with --wiki-dir.
        let cli = parse(&["crawl", "--wiki", "--wiki-dir", "global"]).ok().expect("valid");
        assert_eq!(cli.wiki_dir, PathBuf::from("global"));
        assert!(matches!(cli.command, Commands::Crawl { wiki: true, .. }));
    }

    fn subcommand_help(name: &str) -> String {
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        // Propagates the global options, as a real run does before printing help.
        cmd.build();
        cmd.find_subcommand_mut(name)
            .unwrap_or_else(|| panic!("no subcommand {name}"))
            .render_long_help()
            .to_string()
    }

    #[test]
    fn help_documents_the_failure_limit_and_the_prune_folder() {
        let limit = format!("{} failed attempts", pipeline::MAX_VIDEO_FAILURES);
        let crawl = subcommand_help("crawl");
        assert!(crawl.contains(&format!("failed {} times", pipeline::MAX_VIDEO_FAILURES)), "{crawl}");
        assert!(crawl.contains("ytc process-video <video id>"), "{crawl}");
        assert!(subcommand_help("process-video").contains(&limit));

        let wiki = subcommand_help("wiki");
        assert!(wiki.contains(PRUNED_DIR_NAME), "{wiki}");
        assert!(wiki.contains("nothing is deleted"), "{wiki}");

        // Global options are listed for a subcommand, so 'ytc crawl --help' shows them.
        for option in ["--max-slop", "--wiki-dir", "--db", "--llm-url", "--llm-key", "--llm-model"] {
            assert!(crawl.contains(option), "crawl help lacks {option}");
        }
    }

    #[tokio::test]
    async fn a_stop_request_ends_a_long_step_promptly() {
        let stop = Arc::new(StopSignal::default());
        let setter = stop.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            setter.requested.store(true, Ordering::SeqCst);
        });

        let started = std::time::Instant::now();
        let hour = tokio::time::sleep(std::time::Duration::from_secs(3600));
        assert!(run_unless_stopped(&stop, hour).await.is_none());
        assert!(started.elapsed() < std::time::Duration::from_secs(5), "took {:?}", started.elapsed());
        assert!(stop.acknowledged.load(Ordering::SeqCst), "the listener must learn that the loop reacted");

        // Once requested, a further step is not even started.
        let mut polled = false;
        let step = async { polled = true };
        assert!(run_unless_stopped(&stop, step).await.is_none());
        assert!(!polled);
    }

    #[tokio::test]
    async fn a_step_runs_to_its_result_without_a_stop_request() {
        let stop = StopSignal::default();
        // Longer than one poll period: the step must survive the polling.
        let step = async {
            tokio::time::sleep(STOP_POLL * 3).await;
            7
        };
        assert_eq!(run_unless_stopped(&stop, step).await, Some(7));
        assert!(!stop.acknowledged.load(Ordering::SeqCst));
    }

    #[test]
    fn wiki_prune_and_output_are_parsed() {
        match parse(&["wiki"]).ok().expect("valid").command {
            Commands::Wiki { output, prune } => {
                assert!(output.is_none());
                assert!(!prune);
            }
            _ => panic!("not the wiki command"),
        }
        match parse(&["wiki", "--prune", "-o", "out"]).ok().expect("valid").command {
            Commands::Wiki { output, prune } => {
                assert_eq!(output, Some(PathBuf::from("out")));
                assert!(prune);
            }
            _ => panic!("not the wiki command"),
        }
    }

    #[test]
    fn reanalyze_takes_a_video_or_an_analyzer_prefix() {
        match parse(&["reanalyze", "-AbCdEfGh12"]).ok().expect("valid").command {
            Commands::Reanalyze { video_id, analyzer, limit, .. } => {
                assert_eq!(video_id.as_deref(), Some("-AbCdEfGh12"));
                assert!(analyzer.is_none());
                assert!(limit.is_none());
            }
            _ => panic!("not the reanalyze command"),
        }
        match parse(&["reanalyze", "--analyzer", "heuristic", "--limit", "5"]).ok().expect("valid").command {
            Commands::Reanalyze { video_id, analyzer, limit, .. } => {
                assert!(video_id.is_none());
                assert_eq!(analyzer.as_deref(), Some("heuristic"));
                assert_eq!(limit, Some(5));
            }
            _ => panic!("not the reanalyze command"),
        }
        assert!(parse(&["reanalyze"]).is_err(), "neither a video nor a prefix");
        assert!(parse(&["reanalyze", ID, "--analyzer", "heuristic"]).is_err(), "both at once");
    }

    #[test]
    fn process_video_json_flag_is_parsed_in_either_position() {
        for args in [
            ["process-video", "-AbCdEfGh12", "--json"],
            ["process-video", "--json", "-AbCdEfGh12"],
        ] {
            match parse(&args).ok().expect("valid").command {
                Commands::ProcessVideo { input, json } => {
                    assert_eq!(input, "-AbCdEfGh12");
                    assert!(json, "{args:?}");
                }
                _ => panic!("not the process-video command"),
            }
        }
        match parse(&["process-video", ID]).ok().expect("valid").command {
            Commands::ProcessVideo { json, .. } => assert!(!json),
            _ => panic!("not the process-video command"),
        }
    }

    #[test]
    fn worker_arguments_drop_only_the_json_flag() {
        let args = ["--db", "x.db", "process-video", "--json", "-AbCdEfGh12"]
            .iter()
            .map(std::ffi::OsString::from);
        let expected: Vec<std::ffi::OsString> =
            ["--db", "x.db", "process-video", "-AbCdEfGh12"].iter().map(std::ffi::OsString::from).collect();
        assert_eq!(args_without_json(args), expected);
    }

    #[test]
    fn set_status_accepts_active_muted_auto_only() {
        for (word, expected) in [
            ("active", StatusChoice::Active),
            ("muted", StatusChoice::Muted),
            ("auto", StatusChoice::Auto),
        ] {
            match parse(&["set-status", "UC0000000000000000000001", word]).ok().expect("valid").command {
                Commands::SetStatus { channel, status } => {
                    assert_eq!(channel, "UC0000000000000000000001");
                    assert_eq!(status, expected);
                }
                _ => panic!("not the set-status command"),
            }
        }
        assert!(parse(&["set-status", "UC0000000000000000000001", "high_signal"]).is_err());
        assert!(parse(&["set-status", "UC0000000000000000000001"]).is_err());
    }

    #[test]
    fn harvest_mentions_is_offline_by_default() {
        match parse(&["harvest-mentions"]).ok().expect("valid").command {
            Commands::HarvestMentions { resolve, add, limit } => {
                assert_eq!(resolve, 0);
                assert!(!add);
                assert_eq!(limit, 50);
            }
            _ => panic!("not the harvest-mentions command"),
        }
        match parse(&["harvest-mentions", "--resolve", "3", "--add", "--limit", "0"]).ok().expect("valid").command {
            Commands::HarvestMentions { resolve, add, limit } => {
                assert_eq!(resolve, 3);
                assert!(add);
                assert_eq!(limit, 0);
            }
            _ => panic!("not the harvest-mentions command"),
        }
    }

    #[test]
    fn watch_numbers_come_from_flags() {
        match parse(&["watch"]).ok().expect("valid").command {
            Commands::Watch { interval, limit, delay, discover, queries, discover_limit, discover_delay, wiki } => {
                assert_eq!((interval, limit, delay), (60, 2, 15));
                assert_eq!((discover_limit, discover_delay), (2, 2));
                assert!(!discover);
                assert!(queries.is_none());
                assert!(wiki.is_none());
            }
            _ => panic!("not the watch command"),
        }
        let args = [
            "watch", "-i", "5", "-l", "4", "--delay", "0", "--discover", "--queries", "a, b",
            "--discover-limit", "7", "--discover-delay", "9", "--wiki", "out",
        ];
        match parse(&args).ok().expect("valid").command {
            Commands::Watch { interval, limit, delay, discover, queries, discover_limit, discover_delay, wiki } => {
                assert_eq!((interval, limit, delay), (5, 4, 0));
                assert_eq!((discover_limit, discover_delay), (7, 9));
                assert!(discover);
                assert_eq!(queries.as_deref(), Some("a, b"));
                assert_eq!(wiki, Some(PathBuf::from("out")));
            }
            _ => panic!("not the watch command"),
        }
        assert!(parse(&["watch", "--interval", "0"]).is_err(), "a zero interval would loop without pause");
    }

    #[test]
    fn watch_queries_are_only_the_ones_given() {
        assert_eq!(watch_queries(Some(" a , ,b ")), vec!["a".to_string(), "b".to_string()]);
        assert!(watch_queries(None).is_empty());
        assert!(watch_queries(Some(" , ")).is_empty());
    }

    #[test]
    fn crawl_wiki_and_related_offline_flags_are_parsed() {
        match parse(&["crawl", "--wiki", "-l", "1"]).ok().expect("valid").command {
            Commands::Crawl { channel, limit, delay, wiki } => {
                assert!(channel.is_none());
                assert_eq!((limit, delay), (1, 10));
                assert!(wiki);
            }
            _ => panic!("not the crawl command"),
        }
        match parse(&["crawl"]).ok().expect("valid").command {
            Commands::Crawl { wiki, .. } => assert!(!wiki),
            _ => panic!("not the crawl command"),
        }
        match parse(&["related-channels", "@somebody", "--offline"]).ok().expect("valid").command {
            Commands::RelatedChannels { input, offline } => {
                assert_eq!(input, "@somebody");
                assert!(offline);
            }
            _ => panic!("not the related-channels command"),
        }
    }

    // --- pure helpers ---

    fn insight(analyzer_type: &str, slop_score: u32, quality_rating: &str) -> InsightResult {
        InsightResult {
            video_id: ID.to_string(),
            slop_score,
            is_worth_keeping: slop_score <= 45,
            quality_rating: quality_rating.to_string(),
            summary: "Summary text.".to_string(),
            key_insights: vec!["First point".to_string()],
            takeaways: vec!["Do this".to_string()],
            quotes: vec!["A quote".to_string()],
            topics: vec!["glaciers".to_string()],
            mentioned_creators: Vec::new(),
            analyzer_type: analyzer_type.to_string(),
            category: None,
        }
    }

    fn video() -> Video {
        Video {
            id: ID.to_string(),
            channel_id: "UC0000000000000000000001".to_string(),
            title: "A Video".to_string(),
            url: format!("https://www.youtube.com/watch?v={ID}"),
            published_at: "2026-01-01T00:00:00+00:00".to_string(),
            status: "processed".to_string(),
            created_at: "2026-01-02T00:00:00+00:00".to_string(),
        }
    }

    fn channel(id: &str, title: &str) -> Channel {
        new_active_channel(id, title)
    }

    #[test]
    fn list_cells_show_no_rating_of_its_own_for_a_heuristic_row() {
        // The rating a heuristic row carries is not shown as a judgment.
        let heuristic = insight(analyzer::HEURISTIC_ANALYZER, 40, "High Signal");
        assert_eq!(score_and_rating(&heuristic, None), ("40".to_string(), UNREVIEWED_LABEL.to_string()));

        let llm = insight("llm:some-model", 20, "High Signal");
        assert_eq!(score_and_rating(&llm, None), ("20".to_string(), "High Signal".to_string()));
    }

    #[test]
    fn before_after_score_names_the_engine_when_it_is_not_an_llm() {
        assert_eq!(score_with_engine(&insight("llm:m", 20, "High Signal"), None), "20/100");
        assert_eq!(
            score_with_engine(&insight(analyzer::HEURISTIC_ANALYZER, 40, "High Signal"), None),
            "40/100, Unreviewed (heuristic)"
        );
    }

    #[test]
    fn export_labels_llm_and_heuristic_rows() {
        let read = judged(Some("ok"), Some(2));
        let md = export_markdown(&video(), &insight("llm:m", 20, "High Signal"), Some(&read), None);
        assert!(md.contains("- **Slop Score:** 20/100\n"));
        assert!(md.contains("- **Read:** whole transcript (2 sections)\n"));
        assert!(md.contains("## Core Insights"));

        // Nothing recorded about the reading: no line.
        let md = export_markdown(&video(), &insight("llm:m", 20, "High Signal"), Some(&InsightMeta::default()), None);
        assert!(!md.contains("**Read:**"), "{md}");

        let md = export_markdown(&video(), &insight(analyzer::HEURISTIC_ANALYZER, 40, "High Signal"), None, None);
        assert!(md.contains("- **Quality Rating:** Unreviewed (heuristic)\n"));
        assert!(md.contains("not reviewed by an LLM"));
        assert!(!md.contains("High Signal"));
        assert!(!md.contains("## Core Insights"));
        assert!(!md.contains("## Actionable Takeaways"));
        assert!(!md.contains("## Notable Quotes"));
    }

    #[test]
    fn json_output_is_one_object_on_one_line() {
        let read = judged(Some("ok"), Some(3));
        let value = insight_json(&video(), &insight("llm:m", 20, "High Signal"), Some(&read), None);
        assert!(value.is_object());
        assert_eq!(value["video_id"], ID);
        assert_eq!(value["engine"], "llm");
        assert_eq!(value["slop_score"], 20);
        assert_eq!(value["not_scored"], false);
        assert_eq!(value["sections_total"], 3);
        assert_eq!(value["transcript_chars"], 36_000);
        assert_eq!(value["video"]["title"], "A Video");
        assert_eq!(value["insight"]["quality_rating"], "High Signal");
        // One line: scripts read it with a single json.loads.
        assert!(!serde_json::to_string(&value).expect("serialisable").contains('\n'));

        let value = insight_json(&video(), &insight(analyzer::HEURISTIC_ANALYZER, 40, "x"), None, None);
        assert_eq!(value["engine"], "heuristic");
    }

    #[test]
    fn mentions_are_counted_per_video_and_sorted_by_count() {
        let pair = |name: &str, title: &str| (name.to_string(), title.to_string());
        let groups = group_mentions(&[
            pair("Zed Person", "Video A"),
            pair("Alice Example", "Video A"),
            pair("alice example", "Video B"),
            pair("@Alice Example", "Video B"), // same video twice: counted once
            pair("Alice Example", "Video C"),
            pair("Ada Person", "Video C"),
            pair("  ", "Video C"),
        ]);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].name, "Alice Example");
        assert_eq!(groups[0].videos, vec!["Video A", "Video B", "Video C"]);
        // Equal counts: by name.
        assert_eq!(groups[1].name, "Ada Person");
        assert_eq!(groups[2].name, "Zed Person");
        assert!(group_mentions(&[]).is_empty());
    }

    #[test]
    fn a_short_channel_title_does_not_mark_other_names_as_tracked() {
        let channels = [channel("UC0000000000000000000001", "Go"), channel("UC0000000000000000000002", "X Collective")];
        assert!(tracked_channel_for_name(&channels, "Go Examples").is_none());
        assert!(tracked_channel_for_name(&channels, "Zed Person").is_none());
        assert_eq!(
            tracked_channel_for_name(&channels, "x-collective").map(|c| c.id.as_str()),
            Some("UC0000000000000000000002")
        );
        assert!(tracked_channel_for_name(&channels, "  ").is_none());
    }

    #[test]
    fn handle_guesses_and_title_checks() {
        assert_eq!(handle_guess("Alice Example").as_deref(), Some("@AliceExample"));
        assert_eq!(handle_guess("@3example1name").as_deref(), Some("@3example1name"));
        assert!(handle_guess("X Collective (XC)").is_none());
        assert!(handle_guess("Go").is_none());
        assert!(handle_guess("").is_none());

        assert!(titles_match("Alice Example", "alice example"));
        assert!(titles_match("Example", "Alice Example"));
        assert!(titles_match("X Collective", "X-Collective"));
        assert!(!titles_match("Go", "Go Examples"), "too short to accept a partial match");
        assert!(!titles_match("Alice Example", "Some Fan Channel"));
        assert!(!titles_match("", "Anything"));
    }

    #[test]
    fn tracked_channels_are_found_without_a_request() {
        let channels = [
            channel("UC0000000000000000000001", "First Channel"),
            channel("UC0000000000000000000002", "Twin"),
            channel("UC0000000000000000000003", "Twin"),
        ];
        let id_of = |input: &str| find_tracked_channel(&channels, input).map(|c| c.id.clone());
        assert_eq!(id_of("UC0000000000000000000002").as_deref(), Some("UC0000000000000000000002"));
        assert_eq!(
            id_of("https://www.youtube.com/channel/UC0000000000000000000003/videos").as_deref(),
            Some("UC0000000000000000000003")
        );
        assert_eq!(id_of(" first channel ").as_deref(), Some("UC0000000000000000000001"));
        assert!(id_of("Twin").is_none(), "an ambiguous title must not pick one");
        assert!(id_of("@firstchannel").is_none());
        assert!(id_of("").is_none());
    }

    #[test]
    fn only_non_loopback_llm_hosts_are_reported() {
        for local in [
            "http://127.0.0.1:8080/v1",
            "http://localhost:11434/v1",
            "http://[::1]:8080/v1",
            "http://llm.localhost/v1",
            "not a url",
        ] {
            assert!(remote_llm_host(local).is_none(), "{local}");
        }
        assert_eq!(remote_llm_host("https://api.example.com/v1").as_deref(), Some("api.example.com"));
        assert_eq!(remote_llm_host("http://192.168.1.20:8080/v1").as_deref(), Some("192.168.1.20"));
    }

    // --- disqualify, merge-db, one database, reanalyze selections ---

    #[test]
    fn exclude_include_and_excluded_are_parsed() {
        match parse(&["exclude", ID, "--reason", "duplicate of another talk"]).ok().expect("valid").command {
            Commands::Exclude { video, reason } => {
                assert_eq!(video, ID);
                assert_eq!(reason.as_deref(), Some("duplicate of another talk"));
            }
            _ => panic!("not the exclude command"),
        }
        // IDs can start with a hyphen; the flag is found on either side.
        for args in [
            ["exclude", "-AbCdEfGh12", "--reason", "x"],
            ["exclude", "--reason", "x", "-AbCdEfGh12"],
        ] {
            match parse(&args).unwrap_or_else(|e| panic!("{args:?}: {e}")).command {
                Commands::Exclude { video, reason } => {
                    assert_eq!(video, "-AbCdEfGh12", "{args:?}");
                    assert_eq!(reason.as_deref(), Some("x"), "{args:?}");
                }
                _ => panic!("not the exclude command"),
            }
        }
        match parse(&["exclude", ID]).ok().expect("valid").command {
            Commands::Exclude { reason, .. } => assert!(reason.is_none()),
            _ => panic!("not the exclude command"),
        }
        match parse(&["include", "-AbCdEfGh12"]).ok().expect("valid").command {
            Commands::Include { video } => assert_eq!(video, "-AbCdEfGh12"),
            _ => panic!("not the include command"),
        }
        assert!(matches!(parse(&["excluded"]).ok().expect("valid").command, Commands::Excluded));
        assert!(parse(&["exclude"]).is_err(), "a video is required");
        assert!(parse(&["include"]).is_err(), "a video is required");
        assert!(parse(&["include", ID, "--reason", "x"]).is_err(), "include takes no reason");
    }

    #[test]
    fn category_commands_parse_and_change_only_the_category_list() {
        match parse(&["category", "add", "Baking", "--description", "bread and pastry"]).ok().expect("valid").command {
            Commands::Category { action: CategoryAction::Add { name, description } } => {
                assert_eq!((name.as_str(), description.as_deref()), ("Baking", Some("bread and pastry")));
            }
            _ => panic!("not category add"),
        }
        match parse(&["category", "add", "Star Gazing"]).ok().expect("valid").command {
            Commands::Category { action: CategoryAction::Add { name, description } } => {
                assert_eq!((name.as_str(), description), ("Star Gazing", None));
            }
            _ => panic!("not category add"),
        }
        match parse(&["category", "remove", "Baking"]).ok().expect("valid").command {
            Commands::Category { action: CategoryAction::Remove { name } } => assert_eq!(name, "Baking"),
            _ => panic!("not category remove"),
        }
        assert!(matches!(
            parse(&["category", "list"]).ok().expect("valid").command,
            Commands::Category { action: CategoryAction::List }
        ));
        assert!(parse(&["category"]).is_err(), "an action is required");
        assert!(parse(&["category", "add"]).is_err(), "a name is required");
        assert!(parse(&["category", "remove"]).is_err(), "a name is required");

        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open test db");
        // Nothing is defined until the user adds a category.
        category_command(&db, CategoryAction::List).expect("empty list");
        assert!(db.list_categories().expect("list").is_empty());
        let add = |name: &str| CategoryAction::Add { name: name.to_string(), description: None };
        category_command(&db, add("Baking")).expect("add");
        assert!(category_command(&db, add("baking")).is_err(), "the same name in another case");
        category_command(&db, CategoryAction::List).expect("list");
        category_command(&db, CategoryAction::Remove { name: "BAKING".to_string() }).expect("remove");
        assert!(category_command(&db, CategoryAction::Remove { name: "Baking".to_string() }).is_err());
        assert!(db.list_categories().expect("list").is_empty());
    }

    #[test]
    fn category_rename_and_describe_parse_and_change_the_category() {
        match parse(&["category", "rename", "Baking", "Bread and Pastry"]).ok().expect("valid").command {
            Commands::Category { action: CategoryAction::Rename { old, new } } => {
                assert_eq!((old.as_str(), new.as_str()), ("Baking", "Bread and Pastry"));
            }
            _ => panic!("not category rename"),
        }
        match parse(&["category", "describe", "Baking", "bread and pastry"]).ok().expect("valid").command {
            Commands::Category { action: CategoryAction::Describe { name, text } } => {
                assert_eq!((name.as_str(), text.as_deref()), ("Baking", Some("bread and pastry")));
            }
            _ => panic!("not category describe"),
        }
        match parse(&["category", "describe", "Baking"]).ok().expect("valid").command {
            Commands::Category { action: CategoryAction::Describe { text, .. } } => assert_eq!(text, None),
            _ => panic!("not category describe"),
        }
        assert!(parse(&["category", "rename", "Baking"]).is_err(), "a new name is required");
        assert!(parse(&["category", "describe"]).is_err(), "a name is required");

        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open test db");
        db.add_category("Baking", Some("bread")).expect("add");
        db.add_category("Astronomy", None).expect("add");
        let rename = |old: &str, new: &str| CategoryAction::Rename { old: old.to_string(), new: new.to_string() };
        assert!(category_command(&db, rename("baking", "astronomy")).is_err(), "the name of another category");
        assert!(category_command(&db, rename("Gardening", "Plants")).is_err(), "an unknown category");
        category_command(&db, rename("baking", "Bread and Pastry")).expect("rename");
        let describe = |text: Option<&str>| CategoryAction::Describe {
            name: "bread and pastry".to_string(),
            text: text.map(str::to_string),
        };
        category_command(&db, describe(Some("loaves, rolls and ovens"))).expect("describe");
        let listed = db.list_categories().expect("list");
        assert_eq!(listed[1].name, "Bread and Pastry");
        assert_eq!(listed[1].description.as_deref(), Some("loaves, rolls and ovens"));
        for removing in [None, Some("")] {
            category_command(&db, describe(Some("something"))).expect("describe");
            category_command(&db, describe(removing)).expect("remove the description");
            assert_eq!(db.list_categories().expect("list")[1].description, None);
        }
    }

    #[test]
    fn set_category_takes_a_name_or_none_or_auto() {
        match parse(&["set-category", ID, "Baking"]).ok().expect("valid").command {
            Commands::SetCategory { video, name, none, auto } => {
                assert_eq!((video.as_str(), name.as_deref(), none, auto), (ID, Some("Baking"), false, false));
                assert_eq!(category_choice(name, none, auto).expect("choice"), CategoryChoice::Named("Baking".to_string()));
            }
            _ => panic!("not set-category"),
        }
        // A video ID may start with a hyphen; the flag is still a flag.
        for (flag, expected) in [("--none", CategoryChoice::NoCategory), ("--auto", CategoryChoice::Auto)] {
            match parse(&["set-category", "-AbCdEfGh12", flag]).ok().expect("valid").command {
                Commands::SetCategory { video, name, none, auto } => {
                    assert_eq!(video, "-AbCdEfGh12");
                    assert_eq!(category_choice(name, none, auto).expect("choice"), expected);
                }
                _ => panic!("not set-category"),
            }
        }
        // Exactly one of the three.
        assert!(parse(&["set-category", ID]).is_err());
        assert!(parse(&["set-category", ID, "Baking", "--none"]).is_err());
        assert!(parse(&["set-category", ID, "Baking", "--auto"]).is_err());
        assert!(parse(&["set-category", ID, "--none", "--auto"]).is_err());
        assert!(parse(&["set-category"]).is_err());
        assert!(category_choice(None, false, false).is_err());
        assert!(category_choice(Some("Baking".to_string()), true, false).is_err());

        match parse(&["categorize"]).ok().expect("valid").command {
            Commands::Categorize { all, limit } => assert_eq!((all, limit), (false, None)),
            _ => panic!("not categorize"),
        }
        match parse(&["categorize", "--all", "--limit", "25"]).ok().expect("valid").command {
            Commands::Categorize { all, limit } => assert_eq!((all, limit), (true, Some(25))),
            _ => panic!("not categorize"),
        }
        assert!(parse(&["categorize", "extra"]).is_err());
    }

    #[test]
    fn a_category_set_by_hand_is_stored_and_shown_as_such() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open");
        db.upsert_channel(&channel("UC0000000000000000000001", "A Channel")).expect("channel");
        db.upsert_video(&video()).expect("video");
        db.save_insight(&insight("llm:m", 20, "High Signal")).expect("insight");
        db.add_category("Baking", None).expect("add");
        db.add_category("Astronomy", None).expect("add");
        let shown = |db: &Database| {
            let stored = db.get_insight(ID).expect("insight").expect("row");
            let meta = db.get_insight_meta(ID).expect("meta");
            let md = export_markdown(&video(), &stored, meta.as_ref(), None);
            (category_label(&stored, meta.as_ref()), md)
        };
        // No category and no choice: no line at all.
        let (label, md) = shown(&db);
        assert_eq!(label, None);
        assert!(!md.contains("**Category:**"), "{md}");

        // A URL names the same video; the name matches whatever its case.
        set_category(&db, &format!("https://youtu.be/{ID}"), CategoryChoice::Named("bAKING".to_string())).expect("set");
        let (label, md) = shown(&db);
        assert_eq!(label.as_deref(), Some("Baking (set by hand)"));
        assert!(md.contains("- **Category:** Baking (set by hand)\n"), "{md}");

        // An unknown name changes nothing and lists the names there are.
        let err = set_category(&db, ID, CategoryChoice::Named("Gardening".to_string())).expect_err("unknown").to_string();
        assert!(err.contains("Your categories: Astronomy, Baking"), "{err}");
        assert_eq!(shown(&db).0.as_deref(), Some("Baking (set by hand)"));

        set_category(&db, ID, CategoryChoice::NoCategory).expect("none");
        let (label, md) = shown(&db);
        assert_eq!(label.as_deref(), Some("none (set by hand)"));
        assert!(md.contains("- **Category:** none (set by hand)\n"), "{md}");

        // Handed back: the category stays, the mark goes.
        set_category(&db, ID, CategoryChoice::Named("Astronomy".to_string())).expect("set");
        set_category(&db, ID, CategoryChoice::Auto).expect("auto");
        let (label, md) = shown(&db);
        assert_eq!(label.as_deref(), Some("Astronomy"));
        assert!(md.contains("- **Category:** Astronomy\n"), "{md}");

        let err = set_category(&db, "aaaaaaaaaaa", CategoryChoice::NoCategory).expect_err("no analysis").to_string();
        assert!(err.contains("has no analysis"), "{err}");
        assert!(set_category(&db, "not a video", CategoryChoice::Auto).is_err());
    }

    #[tokio::test]
    async fn categorize_without_categories_prints_a_hint_and_does_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db_path = dir.path().join("ytc.db");
        let db = Database::open(&db_path).expect("open");
        db.upsert_channel(&channel("UC0000000000000000000001", "A Channel")).expect("channel");
        db.upsert_video(&video()).expect("video");
        db.save_insight(&insight("llm:m", 20, "High Signal")).expect("insight");
        // No LLM is configured either: without categories that is never looked at.
        let pipeline = IngestionPipeline::new(&db_path, None, None, None).expect("pipeline");
        categorize(&db, &pipeline, false, None).await.expect("a hint, not an error");
        categorize(&db, &pipeline, true, Some(5)).await.expect("a hint, not an error");
        assert_eq!(db.get_insight(ID).expect("insight").expect("row").category, None);

        // With a category the missing LLM is the error.
        db.add_category("Baking", None).expect("add");
        let err = categorize(&db, &pipeline, false, None).await.expect_err("no LLM").to_string();
        assert!(err.contains("needs an LLM"), "{err}");
    }

    #[test]
    fn merge_db_takes_one_file_and_the_global_db() {
        let cli = parse(&["merge-db", "laptop/ytc.db", "--db", "desk.db"]).ok().expect("valid");
        assert_eq!(cli.db, Some(PathBuf::from("desk.db")));
        match cli.command {
            Commands::MergeDb { other } => assert_eq!(other, PathBuf::from("laptop/ytc.db")),
            _ => panic!("not the merge-db command"),
        }
        assert!(parse(&["merge-db"]).is_err(), "a file is required");
        assert!(parse(&["merge-db", "a.db", "b.db"]).is_err(), "one file at a time");
    }

    #[test]
    fn reanalyze_needs_exactly_one_selection() {
        let flags = |args: &[&str]| match parse(args).unwrap_or_else(|e| panic!("{args:?}: {e}")).command {
            Commands::Reanalyze { video_id, analyzer, all, limit } => {
                (reanalyze_selection(video_id, analyzer, all).expect("one selection"), limit)
            }
            _ => panic!("not the reanalyze command"),
        };
        assert_eq!(flags(&["reanalyze", "--all"]), (ReanalyzeSelection::All, None));
        assert_eq!(flags(&["reanalyze", "-l", "2", "--all"]), (ReanalyzeSelection::All, Some(2)));
        assert_eq!(
            flags(&["reanalyze", "--analyzer", " llm: ", "--limit", "3"]),
            (ReanalyzeSelection::AnalyzerPrefix("llm:".to_string()), Some(3))
        );
        assert_eq!(flags(&["reanalyze", ID]), (ReanalyzeSelection::Video(ID.to_string()), None));

        for two in [vec!["reanalyze", "--all", "--analyzer", "llm:"], vec!["reanalyze", ID, "--all"]] {
            assert!(parse(&two).is_err(), "accepted {two:?}");
        }
        assert!(parse(&["reanalyze", "--limit", "3"]).is_err(), "a limit selects nothing");

        // The same rule without clap in front of it.
        assert!(reanalyze_selection(None, None, false).is_err());
        assert!(reanalyze_selection(Some(ID.to_string()), None, true).is_err());
        assert!(reanalyze_selection(None, Some("  ".to_string()), false).is_err(), "blank prefix");

        let help = subcommand_help("reanalyze");
        for text in ["--all", "--analyzer", "--limit", "longest ago first", "two videos in a row", "minutes per long video"] {
            assert!(help.contains(text), "reanalyze help lacks '{text}': {help}");
        }
    }

    /// The meta of an analysis that read the transcript in `sections` sections.
    fn judged(verdict: Option<&str>, sections: Option<u32>) -> InsightMeta {
        InsightMeta {
            topic_verdict: verdict.map(str::to_string),
            info_status: verdict.map(|_| models::INFO_SUFFICIENT.to_string()),
            sections_total: sections,
            transcript_chars: sections.map(|s| u64::from(s) * 12_000),
            ..Default::default()
        }
    }

    #[test]
    fn request_estimate_is_one_per_section_plus_the_judgment() {
        // Budget of the defaults: 14,547 characters in one request, sections of 21,216.
        assert_eq!(llm_requests_for(1, 14_547, 21_216), 1);
        assert_eq!(llm_requests_for(14_547, 14_547, 21_216), 1);
        assert_eq!(llm_requests_for(14_548, 14_547, 21_216), 2);
        assert_eq!(llm_requests_for(21_216, 14_547, 21_216), 2);
        assert_eq!(llm_requests_for(21_217, 14_547, 21_216), 3);
        assert_eq!(llm_requests_for(100_000, 14_547, 21_216), 6);
        assert_eq!(llm_requests_for(100_000, 14_547, 0), 1, "no division by zero");
    }

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    /// A library of four judged videos, one of them disqualified by hand.
    fn small_library() -> std::collections::HashMap<String, InsightMeta> {
        [
            ("apples", judged(Some("ok"), Some(3))),
            ("bread", judged(None, Some(1))),
            ("cheese", InsightMeta { excluded_manual: true, ..judged(Some("ok"), Some(1)) }),
            ("dates", judged(Some("politics"), Some(2))),
        ]
        .into_iter()
        .map(|(id, meta)| (id.to_string(), meta))
        .collect()
    }

    /// Plans a bulk run over `matching`; every video in `transcripts` has a
    /// stored transcript of that length.
    fn plan_for(matching: &[&str], transcripts: &[(&str, usize)], limit: Option<usize>) -> ReanalyzePlan {
        plan_reanalyze(&ids(matching), &small_library(), false, limit, (14_547, 21_216), |id| {
            Ok(transcripts.iter().find(|(name, _)| *name == id).map(|(_, len)| *len))
        })
        .expect("plan")
    }

    const LIBRARY_TRANSCRIPTS: [(&str, usize); 4] = [("apples", 40_000), ("bread", 10_000), ("cheese", 10_000), ("dates", 36_000)];

    #[test]
    fn a_bulk_run_keeps_the_given_order_and_leaves_out_disqualified_videos() {
        // The order of the selection (analysed longest ago first), not the order of the IDs.
        let plan = plan_for(&["dates", "cheese", "apples", "bread"], &LIBRARY_TRANSCRIPTS, None);
        assert_eq!(plan.targets, ids(&["dates", "apples", "bread"]));
        assert_eq!(plan.disqualified, ids(&["cheese"]));
        assert!(plan.no_transcript.is_empty());
        assert_eq!(plan.beyond_limit, 0);
        // 36,000 and 40,000 are read in sections (2 + 1 requests each), the other in one.
        assert_eq!((plan.requests, plan.in_sections), (3 + 3 + 1, 2));
    }

    #[test]
    fn a_limit_takes_the_first_videos_and_counts_the_rest() {
        let plan = plan_for(&["dates", "cheese", "apples", "bread"], &LIBRARY_TRANSCRIPTS, Some(2));
        assert_eq!(plan.targets, ids(&["dates", "apples"]));
        assert_eq!(plan.beyond_limit, 1, "bread; the disqualified video is not waiting");
        assert_eq!((plan.requests, plan.in_sections), (6, 2), "the forecast covers the targets only");
        assert!(beyond_limit_line(1).contains("left for a later run"));
        assert!(STOP_RESUME_LINE.contains("same command again"));

        // A limit of 0 analyses nothing and reports everything as left.
        let none = plan_for(&["dates", "cheese", "apples", "bread"], &LIBRARY_TRANSCRIPTS, Some(0));
        assert!(none.targets.is_empty());
        assert_eq!(none.beyond_limit, 3);
    }

    #[test]
    fn videos_without_a_stored_transcript_are_listed_completely() {
        // An empty stored transcript is no transcript.
        let plan = plan_for(&["apples", "bread", "dates"], &[("apples", 0), ("dates", 36_000)], Some(0));
        // Complete also behind the limit.
        assert_eq!(plan.no_transcript, ids(&["apples", "bread"]));
        assert!(plan.targets.is_empty());
        assert_eq!(plan.beyond_limit, 1);
    }

    #[test]
    fn a_video_named_directly_is_always_attempted() {
        let metas = small_library();
        // Disqualified, a limit of 0: none of it applies to one named video.
        let plan = plan_reanalyze(&ids(&["cheese"]), &metas, true, Some(0), (14_547, 21_216), |_| Ok(Some(40_000)))
            .expect("plan");
        assert_eq!(plan.targets, ids(&["cheese"]));
        assert_eq!((plan.beyond_limit, plan.disqualified.len()), (0, 0));

        // Without a transcript it is attempted as well, so that the exact error is shown.
        let plan = plan_reanalyze(&ids(&["figs"]), &metas, true, None, (14_547, 21_216), |_| Ok(None)).expect("plan");
        assert_eq!(plan.targets, ids(&["figs"]));
        assert!(plan.no_transcript.is_empty());

        // A database error ends the planning instead of being read as "no transcript".
        let failed = plan_reanalyze(&ids(&["bread"]), &metas, false, None, (14_547, 21_216), |_| {
            Err(anyhow::anyhow!("database is locked"))
        });
        assert!(failed.is_err());
    }

    #[test]
    fn a_bulk_run_stops_when_the_llm_is_down_or_fails_twice_in_a_row() {
        let timeout = analyzer::LlmUnavailable { reason: "error sending request".to_string(), engine_down: false };
        let refused = analyzer::LlmUnavailable { reason: "connection refused".to_string(), engine_down: true };
        // As reanalyze_video returns them: wrapped in a context line.
        let timeout_err = anyhow::Error::new(timeout).context("Stored insight of abc left unchanged");
        let refused_err = anyhow::Error::new(refused).context("Stored insight of abc left unchanged");
        let other_err = anyhow::anyhow!("The LLM answer could not be used");

        assert_eq!(llm_failure_of(&timeout_err), Some(false));
        assert_eq!(llm_failure_of(&refused_err), Some(true));
        assert_eq!(llm_failure_of(&other_err), None);

        // Engine down: at once.
        assert_eq!(llm_stop_rule(llm_failure_of(&refused_err), 0), (1, true));
        // A timeout: the next video may work, the second one in a row stops the run.
        assert_eq!(llm_stop_rule(llm_failure_of(&timeout_err), 0), (1, false));
        assert_eq!(llm_stop_rule(llm_failure_of(&timeout_err), 1), (2, true));
        // An error of the video itself does not count and breaks the row.
        assert_eq!(llm_stop_rule(llm_failure_of(&other_err), 1), (0, false));
        assert_eq!(llm_stop_rule(None, 0), (0, false));

        // timeout, unusable answer, timeout, timeout: stops at the fourth video.
        let mut in_a_row = 0;
        let mut stopped_at = None;
        for (n, err) in [&timeout_err, &other_err, &timeout_err, &timeout_err, &timeout_err].into_iter().enumerate() {
            let stop;
            (in_a_row, stop) = llm_stop_rule(llm_failure_of(err), in_a_row);
            if stop {
                stopped_at = Some(n + 1);
                break;
            }
        }
        assert_eq!(stopped_at, Some(4));
        assert_eq!(LLM_FAILURES_BEFORE_STOP, 2, "the crawl stops after two as well");
    }

    #[test]
    fn the_request_forecast_uses_the_unit_of_the_analyzer_budget() {
        // 9,599 Cyrillic letters are 19,198 bytes: over the one-request budget
        // of 14,547 although the character count is under it.
        let cyrillic = format!("  {}\n", "ж".repeat(9_599));
        assert_eq!(cyrillic.trim().chars().count(), 9_599);
        assert_eq!(transcript_budget_len(&cyrillic), 19_198);
        assert_eq!(llm_requests_for(transcript_budget_len(&cyrillic), 14_547, 21_216), 2);
        // Latin text: bytes and characters agree, surrounding whitespace is not counted.
        assert_eq!(transcript_budget_len("  plain text \n"), 10);
        assert_eq!(transcript_budget_len("   "), 0);
    }

    #[test]
    fn a_repeated_exclude_says_what_happened_to_the_reason() {
        assert_eq!(exclude_label(false, None, None), "Disqualified by hand");
        assert_eq!(exclude_label(false, None, Some("fluff")), "Disqualified by hand");
        // Again without a reason: the stored one stays, and the line says so.
        assert_eq!(exclude_label(true, Some("fluff"), Some("fluff")), "Was already disqualified; nothing changed");
        assert_eq!(exclude_label(true, None, None), "Was already disqualified; nothing changed");
        assert_eq!(exclude_label(true, Some(" fluff "), Some("fluff")), "Was already disqualified; nothing changed");
        assert_eq!(exclude_label(true, Some("fluff"), Some("an ad")), "Was already disqualified; reason updated");
        assert_eq!(exclude_label(true, None, Some("an ad")), "Was already disqualified; reason updated");
        // Never "updated" for a reason that is gone.
        assert_eq!(exclude_label(true, Some("fluff"), None), "Was already disqualified; reason removed");
        assert_eq!(exclude_label(true, Some("fluff"), Some("  ")), "Was already disqualified; reason removed");
    }

    #[test]
    fn a_database_is_not_merged_into_itself() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("data").join("ytc.db");
        let db = Database::open(&path).expect("open");
        db.upsert_channel(&channel("UC0000000000000000000001", "Mine")).expect("channel");

        // The same file, spelled differently.
        let respelled = dir.path().join("data").join(".").join("ytc.db");
        for same in [path.clone(), respelled] {
            let err = merge_db(&db, &path, &same).expect_err("must refuse").to_string();
            assert!(err.contains("cannot be merged into itself"), "{err}");
            assert!(err.contains("Nothing was changed"), "{err}");
        }
        let err = merge_db(&db, &path, &dir.path().join("missing.db")).expect_err("must refuse").to_string();
        assert!(err.contains("does not exist"), "{err}");
        assert_eq!(db.list_channels().expect("list").len(), 1);

        // Another database is imported, and a second run changes nothing.
        let other_path = dir.path().join("other.db");
        {
            let other = Database::open(&other_path).expect("open other");
            other.upsert_channel(&channel("UC0000000000000000000002", "Theirs")).expect("channel");
        }
        let stats = merge_db(&db, &path, &other_path).expect("merge");
        assert_eq!(stats.channels_added, 1);
        assert!(!stats.is_noop());
        assert_eq!(db.list_channels().expect("list").len(), 2);
        assert!(merge_db(&db, &path, &other_path).expect("merge again").is_noop());
    }

    #[test]
    fn exclude_and_include_change_the_effective_decision_only() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open");
        db.upsert_channel(&channel("UC0000000000000000000001", "A Channel")).expect("channel");
        db.upsert_video(&video()).expect("video");
        db.save_insight(&insight("llm:m", 20, "High Signal")).expect("insight");
        assert!(db.get_insight_meta(ID).expect("meta").expect("row").kept);

        exclude_video(&db, ID, Some("duplicate"), Path::new("wiki")).expect("exclude");
        let meta = db.get_insight_meta(ID).expect("meta").expect("row");
        assert!(!meta.kept);
        let stored = db.get_insight(ID).expect("insight").expect("row");
        assert!(stored.is_worth_keeping, "the model's quality decision is not erased");
        assert_eq!(outcome_of(&stored, Some(&meta)), VideoOutcome::Disqualified);
        assert_eq!(rejection_note(&stored, Some(&meta)).as_deref(), Some("Disqualified by hand (duplicate)"));
        assert_eq!(db.list_excluded().expect("list").len(), 1);

        // A URL names the same video.
        include_video(&db, &format!("https://youtu.be/{ID}")).expect("include");
        assert!(db.get_insight_meta(ID).expect("meta").expect("row").kept);
        assert!(db.list_excluded().expect("list").is_empty());
        include_video(&db, ID).expect("including twice is not an error");

        let err = exclude_video(&db, "aaaaaaaaaaa", None, Path::new("wiki")).expect_err("no analysis").to_string();
        assert!(err.contains("nothing to disqualify"), "{err}");
        assert!(include_video(&db, "aaaaaaaaaaa").is_err());
        assert!(exclude_video(&db, "not a video", None, Path::new("wiki")).is_err());
    }

    #[test]
    fn rows_say_why_a_video_is_not_kept() {
        let llm = insight("llm:m", 20, "High Signal");
        let kept = InsightMeta { kept: true, ..judged(Some("ok"), Some(3)) };
        assert_eq!(score_and_rating(&llm, Some(&kept)), ("20".to_string(), "High Signal".to_string()));
        assert_eq!(score_with_engine(&llm, Some(&kept)), "20/100, topic ok");
        assert_eq!(rejection_note(&llm, Some(&kept)), None);
        assert_eq!(kept_status_text(&llm, Some(&kept)), "Kept in the library");

        // A good score does not keep a political video.
        let politics = InsightMeta {
            topic_reason: Some("election campaign commentary".to_string()),
            ..judged(Some("politics"), Some(1))
        };
        assert_eq!(
            score_and_rating(&llm, Some(&politics)),
            ("20".to_string(), "Off-topic: politics (election campaign commentary)".to_string())
        );
        assert_eq!(score_with_engine(&llm, Some(&politics)), "20/100, topic: politics");
        assert_eq!(
            kept_status_text(&llm, Some(&politics)),
            "Not kept: Off-topic: politics (election campaign commentary)"
        );

        // Too little content: no score is shown, in no view.
        let short_row = insight(pipeline::INSUFFICIENT_ANALYZER, 0, "Not enough information");
        let short = InsightMeta {
            info_status: Some(models::INFO_INSUFFICIENT.to_string()),
            info_reason: Some("only 40 words in 31 seconds".to_string()),
            ..Default::default()
        };
        let expected = "Not enough information (only 40 words in 31 seconds)";
        assert_eq!(score_and_rating(&short_row, Some(&short)), ("-".to_string(), expected.to_string()));
        assert_eq!(score_with_engine(&short_row, Some(&short)), expected);
        assert_eq!(score_and_rating(&short_row, None), ("-".to_string(), "Not enough information".to_string()));
        assert_eq!(read_note(&short_row, Some(&short)), None);

        // The owner's decision comes first and keeps the measured score visible.
        let excluded = InsightMeta { excluded_manual: true, excluded_reason: Some(" spam ".to_string()), ..kept.clone() };
        let excluded = InsightMeta { kept: false, ..excluded };
        assert_eq!(
            score_and_rating(&llm, Some(&excluded)),
            ("20".to_string(), "Disqualified by hand (spam)".to_string())
        );
        assert_eq!(score_with_engine(&llm, Some(&excluded)), "20/100, topic ok, disqualified by hand");

        // Plain quality: the rating itself is the explanation.
        let weak = insight("llm:m", 80, "Low Signal");
        let below = judged(Some("ok"), Some(1));
        assert_eq!(score_and_rating(&weak, Some(&below)), ("80".to_string(), "Low Signal".to_string()));
        assert_eq!(kept_status_text(&weak, Some(&below)), "Not kept: below the quality cutoff");

        // A row without a verdict.
        assert_eq!(
            score_with_engine(&llm, Some(&InsightMeta { kept: true, ..Default::default() })),
            "20/100, topic not judged"
        );
    }

    #[test]
    fn read_and_source_notes_state_only_what_is_recorded() {
        let llm = insight("llm:m", 20, "High Signal");
        assert_eq!(
            read_note(&llm, Some(&judged(Some("ok"), Some(3)))).as_deref(),
            Some("whole transcript (3 sections)")
        );
        assert_eq!(read_note(&llm, Some(&judged(Some("ok"), Some(1)))).as_deref(), Some("whole transcript (1 section)"));
        assert_eq!(read_note(&llm, Some(&judged(None, None))), None);
        assert_eq!(read_note(&llm, None), None);

        let coverage = |source: Option<&str>, duration: Option<f64>, until: Option<f64>| TranscriptCoverage {
            video_duration_sec: duration,
            covered_until_sec: until,
            source: source.map(str::to_string),
        };
        assert_eq!(
            transcript_source_note(Some(&coverage(Some("whisper"), Some(600.0), Some(598.0)))).as_deref(),
            Some("local Whisper transcription of the audio (reaches 99% of the video)")
        );
        assert_eq!(
            transcript_source_note(Some(&coverage(Some("builtin"), None, Some(598.0)))).as_deref(),
            Some("YouTube transcript")
        );
        assert_eq!(
            transcript_source_note(Some(&coverage(None, Some(600.0), Some(300.0)))).as_deref(),
            Some("reaches 50% of the video")
        );
        assert_eq!(transcript_source_note(Some(&coverage(None, None, None))), None);
        assert_eq!(transcript_source_note(None), None);
    }

    #[test]
    fn export_and_json_carry_verdict_information_and_coverage() {
        let llm = insight("llm:m", 20, "High Signal");
        let kept = InsightMeta { kept: true, ..judged(Some("ok"), Some(3)) };
        let coverage = TranscriptCoverage {
            video_duration_sec: Some(600.0),
            covered_until_sec: Some(594.0),
            source: Some("subtitles".to_string()),
        };

        let md = export_markdown(&video(), &llm, Some(&kept), Some(&coverage));
        assert!(md.contains("- **Status:** Kept in the library\n"), "{md}");
        assert!(md.contains("- **Read:** whole transcript (3 sections)\n"), "{md}");
        assert!(md.contains("- **Transcript Source:** YouTube subtitle track (reaches 99% of the video)\n"), "{md}");
        assert!(!md.contains("**Topic:**"), "an accepted topic needs no line: {md}");

        let value = insight_json(&video(), &llm, Some(&kept), Some(&coverage));
        assert_eq!(value["is_worth_keeping"], true);
        assert_eq!(value["topic_verdict"], "ok");
        assert!(value["topic_reason"].is_null());
        assert_eq!(value["info_status"], "sufficient");
        assert!(value["info_reason"].is_null());
        assert_eq!(value["excluded"], false);
        assert_eq!(value["sections_total"], 3);
        assert_eq!(value["transcript_source"], "subtitles");
        assert_eq!(value["video_duration_sec"], 600.0);
        assert_eq!(value["covered_until_sec"], 594.0);
        // Every key the help text promises is present, also without any stored extras.
        let bare = insight_json(&video(), &llm, None, None);
        for key in [
            "topic_verdict", "topic_reason", "info_status", "info_reason", "excluded", "sections_total",
            "transcript_source", "video_duration_sec", "covered_until_sec",
        ] {
            assert!(bare.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(bare["excluded"], false);
        assert!(bare["transcript_source"].is_null());

        // Disqualified: the effective decision is false although the stored flag is true.
        let excluded = InsightMeta {
            kept: false,
            excluded_manual: true,
            excluded_reason: Some("duplicate".to_string()),
            ..kept.clone()
        };
        let value = insight_json(&video(), &llm, Some(&excluded), None);
        assert_eq!(value["is_worth_keeping"], false);
        assert_eq!(value["insight"]["is_worth_keeping"], true);
        assert_eq!(value["excluded"], true);
        assert_eq!(value["excluded_reason"], "duplicate");
        assert_eq!(value["status"], "Not kept: Disqualified by hand (duplicate)");
        let md = export_markdown(&video(), &llm, Some(&excluded), None);
        assert!(md.contains("- **Status:** Not kept: Disqualified by hand (duplicate)\n"), "{md}");

        // Off-topic by the model's verdict: the quality score stays visible next to it.
        let politics = InsightMeta { topic_reason: Some("campaign talk".to_string()), ..judged(Some("politics"), Some(1)) };
        let md = export_markdown(&video(), &llm, Some(&politics), None);
        assert!(md.contains("- **Slop Score:** 20/100\n"), "{md}");
        assert!(md.contains("- **Topic:** politics (campaign talk)\n"), "{md}");
        assert!(md.contains("- **Status:** Not kept: Off-topic: politics (campaign talk)\n"), "{md}");

        // Not enough information: no number anywhere.
        let short_row = insight(pipeline::INSUFFICIENT_ANALYZER, 0, "Not enough information");
        let short = InsightMeta {
            info_status: Some(models::INFO_INSUFFICIENT.to_string()),
            info_reason: Some("only 40 words".to_string()),
            ..Default::default()
        };
        let value = insight_json(&video(), &short_row, Some(&short), None);
        assert!(value["slop_score"].is_null());
        assert_eq!(value["not_scored"], true);
        assert_eq!(value["is_worth_keeping"], false);
        assert_eq!(value["info_status"], "insufficient");
        assert_eq!(value["info_reason"], "only 40 words");
        assert_eq!(value["score_label"], "Not enough information (only 40 words)");
        let md = export_markdown(&video(), &short_row, Some(&short), None);
        assert!(md.contains("- **Slop Score:** Not enough information (only 40 words)\n"), "{md}");
        assert!(!md.contains("0/100"), "{md}");
    }

    #[test]
    fn outcome_counts_use_the_crawl_wording() {
        let mut counts = OutcomeCounts::default();
        for outcome in [
            VideoOutcome::Kept,
            VideoOutcome::Kept,
            VideoOutcome::BelowCutoff,
            VideoOutcome::OffTopic,
            VideoOutcome::Insufficient,
        ] {
            counts.add(outcome);
        }
        assert_eq!(
            counts.line(),
            "2 kept, 1 below the quality cutoff, 1 off-topic (fringe / nonsense / politics), 1 not enough information"
        );
        counts.add(VideoOutcome::Disqualified);
        assert!(counts.line().ends_with(", 1 disqualified by hand"));

        let summary = pipeline::CrawlSummary { channels_checked: 4, videos_kept: 2, videos_insufficient: 1, ..Default::default() };
        let line = crawl_summary_line(&summary);
        assert!(line.starts_with("4 channel(s) checked | 2 kept, "), "{line}");
        assert!(line.contains("1 not enough information"), "{line}");
    }

    #[test]
    fn help_describes_the_database_rule_the_topic_verdicts_and_the_commands() {
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        cmd.build();
        let mut help = cmd.render_long_help().to_string();
        for sub in cmd.get_subcommands_mut() {
            help.push_str(&sub.render_long_help().to_string());
        }
        assert!(help.contains("<app folder>/data/ytc.db"), "{help}");
        let lower = help.to_lowercase();
        assert!(lower.contains("fringe") && lower.contains("nonsense") && lower.contains("politics"));

        for name in ["exclude", "include", "excluded", "merge-db", "category", "set-category", "categorize"] {
            assert!(cmd.find_subcommand(name).is_some(), "no subcommand {name}");
        }
        assert!(subcommand_help("exclude").contains("wiki --prune"));
        assert!(subcommand_help("merge-db").contains("read-only"));
    }

    #[test]
    fn long_text_is_cut_on_character_boundaries() {
        assert_eq!(truncate_chars("short", 10), "short");
        assert_eq!(truncate_chars("äöüäöüäöüäöü", 6), "äöü...");
    }
}
