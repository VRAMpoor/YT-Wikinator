use crate::db::Database;
use crate::extractor::{transcript_completeness, Completeness};
use crate::models::{Channel, InsightMeta, InsightResult, TranscriptCoverage, Video};
use crate::pipeline::{video_outcome, IngestionPipeline, VideoOutcome};
use axum::{
    extract::{Path as AxumPath, Query, Request, State},
    http::{header, Method, StatusCode},
    middleware::Next,
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use colored::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

pub struct AppState {
    pub pipeline: Arc<IngestionPipeline>,
    /// The one long job (crawl, discover, re-analysis, wiki prune, category
    /// assignment) that may run at a time: they share the single LLM server
    /// and the wiki folder.
    pub jobs: Arc<JobSlot>,
}

/// The long jobs of the web page. The number is what `JobSlot` stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    Crawl = 1,
    Discover = 2,
    Reanalyze = 3,
    WikiPrune = 4,
    Categorize = 5,
}

impl JobKind {
    /// Name of the job for the refusal message.
    fn label(self) -> &'static str {
        match self {
            JobKind::Crawl => "a crawl",
            JobKind::Discover => "a discover run",
            JobKind::Reanalyze => "a re-analysis",
            JobKind::WikiPrune => "a wiki sync that moves removed pages",
            JobKind::Categorize => "a category assignment",
        }
    }

    /// Every kind of job.
    const ALL: [JobKind; 5] =
        [JobKind::Crawl, JobKind::Discover, JobKind::Reanalyze, JobKind::WikiPrune, JobKind::Categorize];

    fn from_code(code: u8) -> Option<JobKind> {
        Self::ALL.into_iter().find(|k| *k as u8 == code)
    }
}

/// Busy marker shared by crawl, discover and re-analysis. 0 = idle, else the
/// `JobKind` that holds it. One atomic compare-and-set both checks and
/// claims, so two requests arriving together cannot both start.
#[derive(Debug, Default)]
pub struct JobSlot(AtomicU8);

impl JobSlot {
    /// Claims the slot for `kind`. `Err` carries the message for the caller
    /// when another job holds it. The slot is free again when the returned
    /// guard is dropped.
    fn try_claim(self: &Arc<Self>, kind: JobKind) -> Result<JobGuard, String> {
        match self.0.compare_exchange(0, kind as u8, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => Ok(JobGuard { slot: Arc::clone(self) }),
            Err(held) => Err(busy_message(JobKind::from_code(held), kind)),
        }
    }
}

/// Refusal text when `wanted` cannot start because `running` holds the slot.
fn busy_message(running: Option<JobKind>, wanted: JobKind) -> String {
    let running = running.map_or("another job", JobKind::label);
    format!(
        "Not started: {} is running, and only one long job can run at a time (they share the one LLM engine and the wiki folder). Wait for it to finish, then start {} again.",
        running,
        wanted.label()
    )
}

/// Holds the job slot. Releasing happens in `drop`, so every way out of a
/// job frees the slot: the normal end, an error, a panic that unwinds the
/// task, and a handler that is dropped before it could start its task.
#[derive(Debug)]
struct JobGuard {
    slot: Arc<JobSlot>,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        self.slot.0.store(0, Ordering::SeqCst);
    }
}

/// JSON refusal of a job request.
fn busy_json(message: String) -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "error", "busy": true, "message": message}))
}

#[derive(Deserialize)]
struct SearchParams {
    q: Option<String>,
}

#[derive(Deserialize)]
struct AddChannelPayload {
    input: String,
}

#[derive(Deserialize)]
struct AddCategoryPayload {
    name: String,
    description: Option<String>,
}

#[derive(Deserialize)]
struct RemoveCategoryPayload {
    name: String,
}

#[derive(Deserialize)]
struct RenameCategoryPayload {
    name: String,
    new_name: String,
}

#[derive(Deserialize)]
struct DescribeCategoryPayload {
    name: String,
    /// Missing, null or blank removes the description.
    description: Option<String>,
}

/// Body of POST /api/insights/:video_id/category. `auto: true` hands the
/// choice back to the model. Otherwise `category` is the owner's choice: one
/// of the user's categories, or null for "no category".
#[derive(Deserialize)]
struct SetVideoCategoryPayload {
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    auto: bool,
}

#[derive(Deserialize)]
struct CategorizePayload {
    /// Also the videos the model already put into a category.
    all: Option<bool>,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct CrawlPayload {
    limit: Option<usize>,
    delay: Option<u64>,
    channel: Option<String>,
}

#[derive(Deserialize)]
struct DiscoverPayload {
    query: String,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct ReanalyzePayload {
    video_id: String,
}

#[derive(Deserialize)]
struct SyncWikiParams {
    /// `prune=true` also moves generated pages that do not belong to
    /// anything in the library into `<wiki>/.ytc-pruned/`. Nothing is deleted.
    prune: Option<bool>,
}

#[derive(Deserialize)]
struct RelatedParams {
    /// `page=false` skips the request to the creator's YouTube page and
    /// answers from the local database only.
    page: Option<bool>,
}

pub async fn start_server(pipeline: Arc<IngestionPipeline>, port: u16) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        pipeline: pipeline.clone(),
        jobs: Arc::new(JobSlot::default()),
    });

    let app = app_router(state, port);

    // Bind first: if the port is taken we must fail before claiming to be live.
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        anyhow::anyhow!(
            "Cannot listen on http://localhost:{} ({}). Another YTC server or program is already using that port.",
            port,
            e
        )
    })?;

    println!("\n{}", "===========================================================================".cyan());
    println!("  {}", "YTC INTERACTIVE KNOWLEDGE WIKI IS LIVE".green().bold());
    println!("  URL: {}", format!("http://localhost:{}", port).cyan().bold().underline());
    println!("  Storage DB: {}", pipeline.db_path.display().to_string().dimmed());
    println!("  Wiki folder: {}", pipeline.wiki_dir.display().to_string().dimmed());
    println!("  Analysis engine: {}", pipeline.analyzer.engine_label().dimmed());
    println!("  Press Ctrl+C to stop the server");
    println!("{}\n", "===========================================================================".cyan());

    axum::serve(listener, app).await?;
    Ok(())
}

/// Every route of the server behind the one request guard. `port` is the
/// port the server is reached on: the guard accepts requests made to a local
/// address with that port only.
fn app_router(state: Arc<AppState>, port: u16) -> Router {
    Router::new()
        .route("/", get(handle_index))
        .route("/api/status", get(handle_status))
        .route("/api/insights", get(handle_get_insights))
        .route("/api/insights/:video_id/exclude", post(handle_exclude_video))
        .route("/api/insights/:video_id/include", post(handle_include_video))
        .route("/api/insights/:video_id/category", post(handle_set_video_category))
        .route("/api/excluded", get(handle_get_excluded))
        .route("/api/topics", get(handle_get_topics))
        .route("/api/categories", get(handle_get_categories))
        .route("/api/categories/add", post(handle_add_category))
        .route("/api/categories/remove", post(handle_remove_category))
        .route("/api/categories/rename", post(handle_rename_category))
        .route("/api/categories/describe", post(handle_describe_category))
        .route("/api/categorize", post(handle_categorize))
        .route("/api/categorize/progress", get(handle_categorize_progress))
        .route("/api/channels", get(handle_get_channels))
        .route("/api/channels/add", post(handle_add_channel))
        .route("/api/channels/:id/toggle", post(handle_toggle_channel))
        .route("/api/channels/:id/delete", post(handle_delete_channel))
        .route("/api/channels/:id/related", get(handle_related_channels))
        .route("/api/transcripts/:id", get(handle_get_transcript))
        .route("/api/search", get(handle_search))
        .route("/api/sync-wiki", post(handle_sync_wiki))
        .route("/api/crawl", post(handle_crawl))
        .route("/api/crawl/progress", get(handle_crawl_progress))
        .route("/api/discover", post(handle_discover))
        .route("/api/reanalyze", post(handle_reanalyze))
        // One guard for every route: without it any web page the user visits
        // while the server runs could post to it (see request_allowed).
        .layer(axum::middleware::from_fn(move |req: Request, next: Next| {
            guard_local_requests(port, req, next)
        }))
        .with_state(state)
}

/// How long GET /api/status waits for the LLM probe. The page shows the
/// library without waiting for the status, but the chip should not stay on
/// "checking" for long when the LLM server is simply not started.
const STATUS_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(2500);

/// The three names under which a browser on this machine can reach the server.
const LOCAL_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "[::1]"];

/// True when `value` is `<prefix><local host>:<port>`. Browsers leave the
/// port out when it is the default one (80), so that form is accepted for
/// port 80 only. Host names are not case-sensitive.
fn is_local_authority(value: &str, prefix: &str, port: u16) -> bool {
    let value = value.trim().to_ascii_lowercase();
    let Some(authority) = value.strip_prefix(prefix) else {
        return false;
    };
    LOCAL_HOSTS.iter().any(|host| {
        authority == format!("{}:{}", host, port) || (port == 80 && authority == *host)
    })
}

/// Decides whether a request may reach the handlers.
///
/// The server listens on 127.0.0.1 only, but the user's browser can still be
/// made to call it by any web page that is open at the same time:
/// - DNS rebinding: a foreign name that resolves to 127.0.0.1 makes the page
///   same-origin with this server. Such a request carries the foreign name in
///   `Host`, so every request must name this server by a local address.
/// - Cross-site forms and `fetch(..., {mode: 'no-cors'})` can send a POST
///   without asking first. Browsers add `Origin` to those, so a request that
///   changes something is refused when `Origin` is present and is not this
///   server, or when the browser itself calls it cross-site.
///
/// A missing `Origin` is accepted: command-line clients do not send one, and
/// a browser always does for a cross-origin POST. GET requests are judged by
/// `Host` alone, because links and the address bar send no `Origin`, and a
/// foreign page cannot read the answer of a cross-origin GET.
fn request_allowed(
    method: &Method,
    host: Option<&str>,
    origin: Option<&str>,
    sec_fetch_site: Option<&str>,
    port: u16,
) -> bool {
    if !host.is_some_and(|h| is_local_authority(h, "", port)) {
        return false;
    }
    if method == Method::GET || method == Method::HEAD {
        return true;
    }
    if sec_fetch_site.is_some_and(|s| s.trim().eq_ignore_ascii_case("cross-site")) {
        return false;
    }
    match origin {
        // Includes the literal "null" that sandboxed frames and file pages send.
        Some(o) => is_local_authority(o, "http://", port),
        None => true,
    }
}

/// Middleware in front of every route: answers 403 for requests that
/// `request_allowed` refuses.
async fn guard_local_requests(port: u16, req: Request, next: Next) -> Response {
    // In its own block: the borrow of the request must end before the request
    // is handed on, or the future of this middleware is not `Send`.
    let allowed = {
        // A header that is not valid text cannot be one of the accepted
        // values: it is passed on as an empty string, which matches nothing.
        let text = |name: header::HeaderName| req.headers().get(name).map(|v| v.to_str().unwrap_or(""));
        request_allowed(
            req.method(),
            text(header::HOST),
            text(header::ORIGIN),
            text(header::HeaderName::from_static("sec-fetch-site")),
            port,
        )
    };
    if !allowed {
        return (
            StatusCode::FORBIDDEN,
            format!(
                "Refused: this YTC server only answers requests made to http://localhost:{} from its own page.",
                port
            ),
        )
            .into_response();
    }
    next.run(req).await
}

/// Error answer of the list endpoints. An empty list with HTTP 200 would make a
/// locked or broken database look like an empty library.
fn list_error(e: impl std::fmt::Display) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": e.to_string() })),
    )
}

/// Host (and port) of the LLM base URL for display: no scheme, no path, and
/// above all no `user:password@` part, because the page shows this value.
fn url_host(url: &str) -> String {
    let rest = url.trim();
    let rest = rest.split_once("://").map(|(_, r)| r).unwrap_or(rest);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit_once('@').map(|(_, h)| h).unwrap_or(authority);
    host.to_string()
}

/// True for results of the word-count heuristic: no LLM has read the video.
fn is_heuristic_analyzer(analyzer_type: &str) -> bool {
    analyzer_type.trim().starts_with(crate::analyzer::HEURISTIC_ANALYZER)
}

/// The library counts of GET /api/status. Every insight row is in exactly
/// one of them, by the same precedence as the crawl summary
/// (`pipeline::video_outcome`), so they add up to the number of rows.
fn library_counts_json(metas: &HashMap<String, InsightMeta>) -> serde_json::Value {
    let (mut kept, mut below_cutoff, mut off_topic, mut insufficient, mut excluded) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for meta in metas.values() {
        match video_outcome(meta) {
            VideoOutcome::Kept => kept += 1,
            VideoOutcome::BelowCutoff => below_cutoff += 1,
            VideoOutcome::OffTopic => off_topic += 1,
            VideoOutcome::Insufficient => insufficient += 1,
            VideoOutcome::Disqualified => excluded += 1,
        }
    }
    serde_json::json!({
        "kept": kept,
        "below_cutoff": below_cutoff,
        "off_topic": off_topic,
        "insufficient": insufficient,
        "excluded": excluded,
    })
}

/// Body of GET /api/status. `describe` is `VideoAnalyzer::describe()`:
/// ("llm", model, base URL) or ("heuristic", "", ""). `counts` is
/// `library_counts_json`, or None when the database could not be read: the
/// page then shows no numbers instead of zeros.
fn status_json(
    describe: (String, String, String),
    llm_reachable: bool,
    max_slop: u32,
    wiki_dir: &std::path::Path,
    db_path: &std::path::Path,
    counts: Option<serde_json::Value>,
) -> serde_json::Value {
    let (mode, model, url) = describe;
    let has_llm = mode == "llm";
    serde_json::json!({
        "analyzer_mode": mode,
        "llm_model": if has_llm { Some(model) } else { None },
        "llm_url_host": if has_llm { Some(url_host(&url)) } else { None },
        // Without a configured LLM there is nothing to reach.
        "llm_reachable": has_llm && llm_reachable,
        "max_slop": max_slop,
        "wiki_dir": wiki_dir.display().to_string(),
        "db_path": db_path.display().to_string(),
        "version": env!("CARGO_PKG_VERSION"),
        "counts": counts,
    })
}

/// Whether the stored transcript reaches the end of the video, by the rule
/// the extractor applies when it accepts a caption track. None = the video
/// length or the last cue was not recorded.
fn transcript_complete(tc: &TranscriptCoverage) -> Option<bool> {
    match transcript_completeness(tc.video_duration_sec, tc.covered_until_sec) {
        Completeness::Unknown => None,
        Completeness::Complete => Some(true),
        Completeness::Incomplete { .. } => Some(false),
    }
}

/// One row of GET /api/insights. `meta` carries what the insight row itself
/// does not: whether it was scored at all and how the transcript was read.
fn insight_json(
    v: &Video,
    c: &Channel,
    i: &InsightResult,
    meta: Option<&InsightMeta>,
    transcript: Option<&TranscriptCoverage>,
) -> serde_json::Value {
    // Too little to judge: the score column holds a placeholder.
    let not_scored = meta.is_some_and(|m| m.is_insufficient());
    let score_label = match meta {
        Some(m) => m.score_label(i),
        None => i.score_label(),
    };
    serde_json::json!({
        "video_id": v.id,
        "title": v.title,
        "url": v.url,
        "published_at": v.published_at,
        "channel_id": c.id,
        "channel_title": c.title,
        "channel_status": c.status,
        // For a not_scored row this is a placeholder, not a measurement:
        // the page shows score_label instead.
        "slop_score": i.slop_score,
        "not_scored": not_scored,
        "score_label": score_label,
        "quality_rating": i.quality_rating,
        "summary": i.summary,
        "key_insights": i.key_insights,
        "takeaways": i.takeaways,
        "quotes": i.quotes,
        "topics": i.topics,
        "mentioned_creators": i.mentioned_creators,
        "analyzer": i.analyzer_type,
        "unreviewed": is_heuristic_analyzer(&i.analyzer_type),
        // One of the user's categories, or null.
        "category": i.category,
        // True when the owner chose the category (or "no category") by hand.
        "category_manual": meta.is_some_and(|m| m.category_manual),
        // The two judgments, separately: subject, and enough material to judge.
        "topic_verdict": meta.and_then(|m| m.topic_verdict.as_deref()),
        "topic_reason": meta.and_then(|m| m.topic_reason.as_deref()),
        "info_status": meta.and_then(|m| m.info_status.as_deref()),
        "info_reason": meta.and_then(|m| m.info_reason.as_deref()),
        "excluded": meta.is_some_and(|m| m.excluded_manual),
        "excluded_reason": meta.and_then(|m| m.excluded_reason.as_deref()),
        "excluded_at": meta.and_then(|m| m.excluded_at.as_deref()),
        // The effective decision and, when it is "no", the reason in words.
        "kept": meta.map(|m| m.kept),
        "rejection": meta.and_then(|m| m.rejection_label()),
        // How the analysis read the transcript ("whole transcript (3
        // sections)"); null when no analysis ran: the page then shows no line.
        "sections_total": meta.and_then(|m| m.sections_total),
        "read_note": meta.and_then(InsightMeta::whole_transcript_note),
        "transcript_source": transcript.and_then(|t| t.source.as_deref()),
        "video_duration_sec": transcript.and_then(|t| t.video_duration_sec),
        "covered_until_sec": transcript.and_then(|t| t.covered_until_sec),
        "video_coverage_pct": transcript.and_then(|t| t.video_coverage_pct()),
        "transcript_complete": transcript.and_then(transcript_complete),
    })
}

async fn handle_index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

/// What actually runs: analysis engine, whether its server answers right now,
/// the keep threshold and the folders in use. The page takes every engine
/// name and the threshold from here instead of from fixed text.
async fn handle_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let pipeline = &state.pipeline;
    // No request is made without an LLM. A probe that does not finish in time
    // counts as "not reachable": the page must get its answer either way.
    let reachable = tokio::time::timeout(STATUS_PROBE_TIMEOUT, pipeline.analyzer.llm_reachable())
        .await
        .unwrap_or(false);
    // The counts are extra information: a database that cannot be read right
    // now must not take the engine status away from the page.
    let counts = pipeline
        .db()
        .and_then(|db| db.list_insight_meta())
        .map(|metas| library_counts_json(&metas))
        .ok();
    Json(status_json(
        pipeline.analyzer.describe(),
        reachable,
        pipeline.max_slop,
        &pipeline.wiki_dir,
        &pipeline.db_path,
        counts,
    ))
}

async fn handle_get_insights(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return list_error(e),
    };

    let items = match db.get_all_kept_videos_with_details() {
        Ok(items) => items,
        Err(e) => return list_error(e),
    };
    let meta = match db.list_insight_meta() {
        Ok(m) => m,
        Err(e) => return list_error(e),
    };
    let coverage = match db.list_transcript_coverage() {
        Ok(c) => c,
        Err(e) => return list_error(e),
    };
    let res: Vec<_> = items
        .iter()
        .map(|(v, c, i)| insight_json(v, c, i, meta.get(&v.id), coverage.get(&v.id)))
        .collect();
    (StatusCode::OK, Json(serde_json::json!(res)))
}

/// Longest disqualification reason that is stored; the rest is cut off.
const MAX_EXCLUDE_REASON_CHARS: usize = 300;

/// Reads the optional body `{"reason": "..."}` of POST .../exclude. No body
/// at all is a disqualification without a reason, so a plain command-line
/// POST works. A body that is not such an object is an error: guessing
/// would drop a reason the caller meant to give.
fn parse_exclude_reason(body: &[u8]) -> Result<Option<String>, String> {
    #[derive(Deserialize)]
    struct ExcludePayload {
        reason: Option<String>,
    }
    if body.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(None);
    }
    // Through a JSON value first: the derived reader alone would also take
    // an array and read its first element as the reason.
    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| format!("The request body must be empty or a JSON object with an optional text field 'reason' ({}).", e))?;
    if !(value.is_object() || value.is_null()) {
        return Err("The request body must be empty or a JSON object with an optional text field 'reason'.".to_string());
    }
    let payload: Option<ExcludePayload> = serde_json::from_value(value)
        .map_err(|e| format!("The request body must be empty or JSON with an optional text field 'reason' ({}).", e))?;
    Ok(payload
        .and_then(|p| p.reason)
        .map(|r| r.trim().chars().take(MAX_EXCLUDE_REASON_CHARS).collect::<String>())
        .filter(|r| !r.is_empty()))
}

/// Title of a video for an answer; the id when the video row is missing.
fn video_title(db: &Database, video_id: &str) -> String {
    match db.get_video(video_id) {
        Ok(Some(v)) => v.title,
        _ => video_id.to_string(),
    }
}

/// Answer of POST /api/insights/:video_id/exclude. The wiki files are not
/// rewritten here (`wiki_synced` is always false): the page tells the owner
/// to run Sync Wiki.
fn exclude_video_json(db: &Database, video_id: &str, reason: Option<&str>) -> serde_json::Value {
    let video_id = video_id.trim();
    if video_id.is_empty() {
        return serde_json::json!({"status": "error", "message": "video_id is missing."});
    }
    match db.exclude_video(video_id, reason) {
        Ok(()) => serde_json::json!({
            "status": "success",
            "video_id": video_id,
            "title": video_title(db, video_id),
            "excluded": true,
            "reason": reason,
            "wiki_synced": false,
        }),
        Err(e) => serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    }
}

/// Answer of POST /api/insights/:video_id/include. `kept` says whether the
/// video is back in the library; when it is not, `rejection` says why (the
/// model's judgment decides again once the disqualification is gone).
fn include_video_json(db: &Database, video_id: &str) -> serde_json::Value {
    let video_id = video_id.trim();
    if video_id.is_empty() {
        return serde_json::json!({"status": "error", "message": "video_id is missing."});
    }
    let was_excluded = match db.get_insight_meta(video_id) {
        Ok(meta) => meta.is_some_and(|m| m.excluded_manual),
        Err(e) => return serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    };
    if let Err(e) = db.include_video(video_id) {
        return serde_json::json!({"status": "error", "message": format!("{:#}", e)});
    }
    match db.get_insight_meta(video_id) {
        Ok(meta) => serde_json::json!({
            "status": "success",
            "video_id": video_id,
            "title": video_title(db, video_id),
            "excluded": false,
            "was_excluded": was_excluded,
            "kept": meta.as_ref().is_some_and(|m| m.kept),
            "rejection": meta.as_ref().and_then(|m| m.rejection_label()),
            "wiki_synced": false,
        }),
        Err(e) => serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    }
}

/// Rows of GET /api/excluded: the disqualified videos, newest decision first.
fn excluded_list_json(db: &Database) -> anyhow::Result<serde_json::Value> {
    let rows: Vec<_> = db
        .list_excluded()?
        .into_iter()
        .map(|x| {
            serde_json::json!({
                "video_id": x.video_id,
                "title": x.title,
                "channel_id": x.channel_id,
                "channel_title": x.channel_title,
                "reason": x.reason,
                "excluded_at": x.excluded_at,
                // The model's score from before the disqualification.
                "slop_score": x.slop_score,
            })
        })
        .collect();
    Ok(serde_json::json!(rows))
}

/// Disqualifies a video by the owner's decision. Only the database changes.
async fn handle_exclude_video(
    State(state): State<Arc<AppState>>,
    AxumPath(video_id): AxumPath<String>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let reason = match parse_exclude_reason(&body) {
        Ok(r) => r,
        Err(message) => return Json(serde_json::json!({"status": "error", "message": message})),
    };
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    Json(exclude_video_json(&db, &video_id, reason.as_deref()))
}

/// Undoes a disqualification. Only the database changes.
async fn handle_include_video(
    State(state): State<Arc<AppState>>,
    AxumPath(video_id): AxumPath<String>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    Json(include_video_json(&db, &video_id))
}

async fn handle_get_excluded(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return list_error(e),
    };
    match excluded_list_json(&db) {
        Ok(rows) => (StatusCode::OK, Json(rows)),
        Err(e) => list_error(e),
    }
}

async fn handle_get_topics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return list_error(e),
    };

    match db.list_all_topics() {
        Ok(topics) => {
            let res: Vec<_> = topics
                .into_iter()
                .map(|(name, count)| serde_json::json!({ "name": name, "count": count }))
                .collect();
            (StatusCode::OK, Json(serde_json::json!(res)))
        }
        Err(e) => list_error(e),
    }
}

/// Rows of GET /api/categories: the user's categories, each with the number
/// of kept videos in it. An empty list until the user defines one.
fn categories_json(db: &Database) -> anyhow::Result<serde_json::Value> {
    let rows: Vec<_> = db
        .list_categories_with_counts()?
        .into_iter()
        .map(|(category, kept)| {
            serde_json::json!({
                "name": category.name,
                "description": category.description,
                "created_at": category.created_at,
                "kept": kept,
            })
        })
        .collect();
    Ok(serde_json::json!(rows))
}

/// Answer of POST /api/categories/add.
fn add_category_json(db: &Database, name: &str, description: Option<&str>) -> serde_json::Value {
    match db.add_category(name, description) {
        Ok(category) => serde_json::json!({
            "status": "success",
            "name": category.name,
            "description": category.description,
        }),
        Err(e) => serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    }
}

/// Answer of POST /api/categories/remove. `cleared` is the number of videos
/// that were in the category; they stay in the library without one.
fn remove_category_json(db: &Database, name: &str) -> serde_json::Value {
    match db.remove_category(name) {
        Ok((removed, cleared)) => serde_json::json!({
            "status": "success",
            "name": removed,
            "cleared": cleared,
        }),
        Err(e) => serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    }
}

/// Answer of POST /api/categories/rename. `moved` is the number of videos in
/// the category; they follow the new name.
fn rename_category_json(db: &Database, name: &str, new_name: &str) -> serde_json::Value {
    match db.rename_category(name, new_name) {
        Ok((before, after, moved)) => serde_json::json!({
            "status": "success",
            "old_name": before,
            "name": after,
            "moved": moved,
        }),
        Err(e) => serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    }
}

/// Answer of POST /api/categories/describe. A missing or blank description
/// removes the stored one.
fn describe_category_json(db: &Database, name: &str, description: Option<&str>) -> serde_json::Value {
    match db.describe_category(name, description) {
        Ok(category) => serde_json::json!({
            "status": "success",
            "name": category.name,
            "description": category.description,
        }),
        Err(e) => serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    }
}

/// Answer of POST /api/insights/:video_id/category. With `auto` the choice
/// goes back to the model and the stored category stays as it is; otherwise
/// `category` (None: no category) is set by hand. The answer carries what is
/// stored afterwards.
fn set_video_category_json(db: &Database, video_id: &str, category: Option<&str>, auto: bool) -> serde_json::Value {
    let video_id = video_id.trim();
    if video_id.is_empty() {
        return serde_json::json!({"status": "error", "message": "video_id is missing."});
    }
    let changed = if auto {
        db.set_video_category_auto(video_id)
    } else {
        // A blank name is "no category", as the page sends null for it.
        let name = category.map(str::trim).filter(|name| !name.is_empty());
        db.set_video_category(video_id, name).map(|_| ())
    };
    if let Err(e) = changed {
        return serde_json::json!({"status": "error", "message": format!("{:#}", e)});
    }
    match (db.get_insight(video_id), db.get_insight_meta(video_id)) {
        (Ok(insight), Ok(meta)) => serde_json::json!({
            "status": "success",
            "video_id": video_id,
            "title": video_title(db, video_id),
            "category": insight.and_then(|i| i.category),
            "category_manual": meta.is_some_and(|m| m.category_manual),
        }),
        (Err(e), _) | (_, Err(e)) => serde_json::json!({"status": "error", "message": format!("{:#}", e)}),
    }
}

/// Success answer of POST /api/categorize.
fn categorize_answer_json(summary: &crate::pipeline::CategorizeSummary) -> serde_json::Value {
    let per_category: Vec<_> = summary
        .assigned
        .iter()
        .map(|(name, count)| serde_json::json!({ "name": name, "count": count }))
        .collect();
    serde_json::json!({
        "status": "success",
        "total": summary.total,
        "beyond_limit": summary.beyond_limit,
        "assigned": summary.assigned_total(),
        "per_category": per_category,
        "none": summary.none,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "not_attempted": summary.not_attempted,
        "stopped": summary.stopped,
        "counts_line": summary.counts_line(),
    })
}

async fn handle_rename_category(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RenameCategoryPayload>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    Json(rename_category_json(&db, &payload.name, &payload.new_name))
}

async fn handle_describe_category(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DescribeCategoryPayload>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    Json(describe_category_json(&db, &payload.name, payload.description.as_deref()))
}

/// Sets the category of a video by hand, or hands the choice back to the
/// model. Only the database changes.
async fn handle_set_video_category(
    State(state): State<Arc<AppState>>,
    AxumPath(video_id): AxumPath<String>,
    Json(payload): Json<SetVideoCategoryPayload>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    Json(set_video_category_json(&db, &video_id, payload.category.as_deref(), payload.auto))
}

async fn handle_categorize_progress(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let progress = state.pipeline.categorize_progress.read().await;
    Json(progress.clone())
}

/// Asks the LLM for the category of the analysed videos that have none (with
/// `all`: of every video whose category was not set by hand). Started and
/// reported like a crawl: the work runs as its own task, GET
/// /api/categorize/progress says where it stands, and this request answers
/// when it is over.
async fn handle_categorize(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Option<CategorizePayload>>,
) -> impl IntoResponse {
    let all = payload.as_ref().and_then(|p| p.all).unwrap_or(false);
    let limit = payload.as_ref().and_then(|p| p.limit);

    // Same slot as crawl, discover and re-analysis: one LLM server, one job.
    let job = match state.jobs.try_claim(JobKind::Categorize) {
        Ok(guard) => guard,
        Err(message) => return busy_json(message),
    };
    {
        let mut p = state.pipeline.categorize_progress.write().await;
        *p = crate::pipeline::CategorizeProgress {
            is_running: true,
            stage: "starting".to_string(),
            message: "Starting the category assignment...".to_string(),
            ..Default::default()
        };
    }

    // Own supervised task, same reason as in handle_crawl: a closed tab must
    // not cut the work off, and a panic must still end the "running" state.
    let pipeline = state.pipeline.clone();
    let task = tokio::spawn(async move {
        let _job = job;
        let worker_pipeline = pipeline.clone();
        let worker = tokio::spawn(async move { worker_pipeline.run_categorize(all, limit).await });
        let result = match worker.await {
            Ok(r) => r,
            Err(join_err) => Err(anyhow::anyhow!("The category assignment crashed: {}", join_err)),
        };
        if let Err(e) = &result {
            let mut p = pipeline.categorize_progress.write().await;
            p.is_running = false;
            p.stage = "error".to_string();
            p.message = format!("Category assignment failed: {}", e);
        }
        result
    });

    match task.await {
        Ok(Ok(summary)) => Json(categorize_answer_json(&summary)),
        Ok(Err(e)) => Json(serde_json::json!({"status": "error", "message": e.to_string()})),
        Err(join_err) => Json(serde_json::json!({"status": "error", "message": join_err.to_string()})),
    }
}

async fn handle_get_categories(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return list_error(e),
    };
    match categories_json(&db) {
        Ok(rows) => (StatusCode::OK, Json(rows)),
        Err(e) => list_error(e),
    }
}

async fn handle_add_category(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AddCategoryPayload>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    Json(add_category_json(&db, &payload.name, payload.description.as_deref()))
}

async fn handle_remove_category(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RemoveCategoryPayload>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    Json(remove_category_json(&db, &payload.name))
}

async fn handle_get_channels(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return list_error(e),
    };

    match db.list_channels_with_health() {
        Ok(channels) => (StatusCode::OK, Json(serde_json::json!(channels))),
        Err(e) => list_error(e),
    }
}

async fn handle_add_channel(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AddChannelPayload>,
) -> impl IntoResponse {
    let (ch_id, ch_title) = match crate::rss::resolve_channel(&state.pipeline.http_client, &payload.input).await {
        Ok(res) => res,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": format!("Could not resolve channel: {}", e)})),
    };

    let channel = crate::models::Channel {
        id: ch_id.clone(),
        title: ch_title.clone(),
        url: format!("https://www.youtube.com/channel/{}", ch_id),
        status: "active".to_string(),
        rolling_slop_score: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        last_crawled_at: None,
    };

    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };

    match db.upsert_channel(&channel) {
        Ok(_) => Json(serde_json::json!({
            "status": "success",
            "channel_id": ch_id,
            "title": ch_title
        })),
        Err(e) => Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    }
}

async fn handle_toggle_channel(
    State(state): State<Arc<AppState>>,
    AxumPath(channel_id): AxumPath<String>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };

    match db.toggle_channel_status(&channel_id) {
        Ok(new_status) => Json(serde_json::json!({"status": "success", "new_status": new_status})),
        Err(e) => Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    }
}

/// Stops tracking a channel. `Ok(false)` when the id is not tracked: the
/// DELETE statement succeeds for any id, so without this lookup a mistyped or
/// already removed channel would be reported as removed.
fn delete_tracked_channel(db: &Database, channel_id: &str) -> anyhow::Result<bool> {
    if db.get_channel(channel_id)?.is_none() {
        return Ok(false);
    }
    db.delete_channel(channel_id)?;
    Ok(true)
}

async fn handle_delete_channel(
    State(state): State<Arc<AppState>>,
    AxumPath(channel_id): AxumPath<String>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };

    match delete_tracked_channel(&db, &channel_id) {
        Ok(true) => Json(serde_json::json!({"status": "success"})),
        Ok(false) => Json(serde_json::json!({
            "status": "error",
            "message": format!("No tracked channel has the id {}.", channel_id)
        })),
        Err(e) => Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    }
}

/// Suggestions for a channel, each with the source it came from:
/// creators named in this channel's kept insights (local database) and, unless
/// `?page=false`, channel items on the creator's YouTube page (one request).
/// Nothing is tracked here; the page sends the chosen ones to /api/channels/add.
async fn handle_related_channels(
    State(state): State<Arc<AppState>>,
    AxumPath(channel_id): AxumPath<String>,
    Query(params): Query<RelatedParams>,
) -> impl IntoResponse {
    let include_page = params.page.unwrap_or(true);
    match state.pipeline.related_channels(&channel_id, include_page).await {
        Ok(related) => Json(serde_json::json!({
            "status": "success",
            "page_checked": include_page,
            "related": related
        })),
        Err(e) => Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    }
}

async fn handle_get_transcript(
    State(state): State<Arc<AppState>>,
    AxumPath(video_id): AxumPath<String>,
) -> impl IntoResponse {
    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"error": e.to_string()})),
    };

    // Read note and coverage are extra information: a failed lookup must not
    // hide the transcript.
    let meta = db.get_insight_meta(&video_id).ok().flatten();
    let coverage = db.get_transcript_coverage(&video_id).ok().flatten();

    match db.get_transcript(&video_id) {
        Ok(Some(t)) => Json(serde_json::json!({
            "video_id": t.video_id,
            "raw_char_count": t.raw_char_count,
            "cleaned_char_count": t.cleaned_char_count,
            "stripped_ad_seconds": t.stripped_ad_seconds,
            "stripped_segments_count": t.stripped_segments_count,
            "read_note": meta.as_ref().and_then(InsightMeta::whole_transcript_note),
            "transcript_source": coverage.as_ref().and_then(|c| c.source.as_deref()),
            "video_coverage_pct": coverage.as_ref().and_then(|c| c.video_coverage_pct()),
            "transcript_complete": coverage.as_ref().and_then(transcript_complete),
            "text": t.text
        })),
        Ok(None) => Json(serde_json::json!({"error": "Transcript not found in database"})),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

/// One row of GET /api/search. A "not enough information" row carries a
/// placeholder (0) in the score column. It is not shown as a number: "0/100"
/// would present an unjudged video as a perfect one.
fn search_hit_json(h: crate::db::SearchHit) -> serde_json::Value {
    let score_label = h.score_label();
    let not_scored = h.has_no_score();
    serde_json::json!({
        "video_id": h.video_id,
        "title": h.title,
        // For a not_scored hit this is a placeholder, not a measurement:
        // score_label says what the row is.
        "slop_score": h.slop_score,
        "not_scored": not_scored,
        "score_label": score_label,
        // The effective decision; the fields after it say why a hit with a
        // low score is not kept.
        "is_worth_keeping": h.is_worth_keeping,
        "topic_verdict": h.topic_verdict,
        "info_status": h.info_status,
        "info_reason": h.info_reason,
        "excluded": h.excluded_manual,
        "quality_rating": h.quality_rating,
        "summary": h.summary,
        "key_insights": h.key_insights,
        "topics": h.topics,
        "category": h.category,
    })
}

async fn handle_search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let query = params.q.unwrap_or_default();
    if query.trim().is_empty() {
        return (StatusCode::OK, Json(serde_json::json!([])));
    }

    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return list_error(e),
    };

    match db.search(&query) {
        Ok(hits) => {
            let res: Vec<_> = hits.into_iter().map(search_hit_json).collect();
            (StatusCode::OK, Json(serde_json::json!(res)))
        }
        Err(e) => list_error(e),
    }
}

/// Success answer of POST /api/sync-wiki. `skipped_user_files` is passed on
/// so the page can say that the wiki is incomplete: those pages were not
/// written because a file that ytc did not generate has the same name.
fn wiki_sync_json(stats: &crate::wiki::WikiStats) -> serde_json::Value {
    serde_json::json!({
        "status": "success",
        "topics": stats.topics_count,
        "creators": stats.creators_count,
        "categories": stats.categories_count,
        "insights": stats.insights_count,
        "stale_pages": stats.stale_pages,
        "pruned_pages": stats.pruned_pages,
        // Where the moved pages are now; null when nothing was moved.
        "pruned_to": stats.pruned_to.as_ref().map(|p| p.display().to_string()),
        "skipped_user_files": stats.skipped_user_files,
        "wiki_dir": stats.output_dir.display().to_string()
    })
}

/// Writes the wiki files. Without `?prune=true` nothing is moved: the answer
/// only says how many generated pages belong to nothing any more
/// (`stale_pages`). With it those pages are moved to `<wiki>/.ytc-pruned/`
/// (`pruned_to`), never deleted; the page asks the owner before it sends that.
async fn handle_sync_wiki(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SyncWikiParams>,
) -> impl IntoResponse {
    let prune = params.prune.unwrap_or(false);
    // A running job writes pages of videos this request has not read from the
    // database yet; pruning beside it would move those away as stale. A plain
    // sync moves nothing and needs no guard.
    let _job = if prune {
        match state.jobs.try_claim(JobKind::WikiPrune) {
            Ok(guard) => Some(guard),
            Err(message) => return busy_json(message),
        }
    } else {
        None
    };

    let db = match state.pipeline.db() {
        Ok(d) => d,
        Err(e) => return Json(serde_json::json!({"status": "error", "message": e.to_string()})),
    };

    match crate::wiki::WikiGenerator::generate_with_options(&db, &state.pipeline.wiki_dir, prune) {
        Ok(stats) => Json(wiki_sync_json(&stats)),
        Err(e) => Json(serde_json::json!({"status": "error", "message": format!("{:#}", e)})),
    }
}

async fn handle_crawl_progress(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let progress = state.pipeline.progress.read().await;
    Json(progress.clone())
}

async fn handle_crawl(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Option<CrawlPayload>>,
) -> impl IntoResponse {
    let limit = payload.as_ref().and_then(|p| p.limit).unwrap_or(2);
    let delay = payload.as_ref().and_then(|p| p.delay).unwrap_or(15);
    let channel = payload.as_ref().and_then(|p| p.channel.clone());

    // One atomic check-and-claim shared with discover and re-analysis, so two
    // requests arriving together cannot both start a long job.
    let job = match state.jobs.try_claim(JobKind::Crawl) {
        Ok(guard) => guard,
        Err(message) => return busy_json(message),
    };
    {
        let mut p = state.pipeline.progress.write().await;
        // percent stays 0: nothing has been done yet.
        *p = crate::pipeline::CrawlProgress {
            is_running: true,
            stage: "starting".to_string(),
            message: "Starting crawl...".to_string(),
            ..Default::default()
        };
    }

    // The crawl runs as its own task. If the browser tab is closed or reloaded,
    // only this handler is dropped; the crawl finishes and resets its progress.
    // The supervisor task also outlives the handler, so even a panic inside the
    // crawl releases the "running" flag when nobody is waiting for the answer.
    let pipeline = state.pipeline.clone();
    let task = tokio::spawn(async move {
        // Held by the supervisor until the crawl has ended in any way.
        let _job = job;
        let worker_pipeline = pipeline.clone();
        // The wiki is written inside the crawl (to the pipeline's wiki folder),
        // while the progress still says "running". It is not generated again here.
        let worker = tokio::spawn(async move { worker_pipeline.run_crawl_and_wiki(channel, limit, delay).await });
        let result = match worker.await {
            Ok(r) => r,
            Err(join_err) => Err(anyhow::anyhow!("Crawl crashed: {}", join_err)),
        };
        if let Err(e) = &result {
            let mut p = pipeline.progress.write().await;
            p.is_running = false;
            p.stage = "error".to_string();
            p.message = format!("Crawl failed: {}", e);
        }
        result
    });

    match task.await {
        Ok(Ok(summary)) => Json(crawl_answer_json(&summary)),
        Ok(Err(e)) => Json(serde_json::json!({"status": "error", "message": e.to_string()})),
        Err(join_err) => Json(serde_json::json!({"status": "error", "message": join_err.to_string()})),
    }
}

/// Success answer of POST /api/crawl. Every processed video is in exactly one
/// of kept, filtered_slop, off_topic, insufficient, disqualified and failed.
fn crawl_answer_json(summary: &crate::pipeline::CrawlSummary) -> serde_json::Value {
    serde_json::json!({
        "status": "success",
        "channels_checked": summary.channels_checked,
        "videos_evaluated": summary.videos_evaluated,
        "videos_kept": summary.videos_kept,
        "videos_filtered_slop": summary.videos_filtered_slop,
        "videos_failed": summary.videos_failed,
        "videos_off_topic": summary.videos_off_topic,
        "videos_insufficient": summary.videos_insufficient,
        "videos_disqualified": summary.videos_disqualified,
        "videos_waiting": summary.videos_waiting,
        "counts_line": summary.counts_line(),
        "wiki_generated": summary.wiki_generated,
        "wiki_error": summary.wiki_error
    })
}

/// Success answer of POST /api/reanalyze. `kept` is the effective decision
/// read back from the database; `rejection` says why a video is not kept
/// (off-topic, not enough information, disqualified, below the cutoff).
fn reanalyze_answer_json(
    insight: &InsightResult,
    meta: Option<&InsightMeta>,
    max_slop: u32,
    wiki_error: Option<String>,
) -> serde_json::Value {
    let kept = meta.map_or(insight.is_worth_keeping, |m| m.kept);
    serde_json::json!({
        "status": "success",
        "video_id": insight.video_id,
        "slop_score": insight.slop_score,
        "is_worth_keeping": kept,
        "kept": kept,
        "rejection": meta.and_then(|m| m.rejection_label()),
        "not_scored": meta.is_some_and(|m| m.is_insufficient()),
        "quality_rating": insight.quality_rating,
        "analyzer": insight.analyzer_type,
        "max_slop": max_slop,
        "wiki_error": wiki_error
    })
}

/// Writes the wiki to the pipeline's folder after a change outside a crawl.
/// Returns the error text instead of dropping it, so the page can say that the
/// database changed but the wiki files did not.
fn regenerate_wiki(pipeline: &IngestionPipeline) -> Option<String> {
    let db = match pipeline.db() {
        Ok(db) => db,
        Err(e) => return Some(e.to_string()),
    };
    crate::wiki::WikiGenerator::generate(&db, &pipeline.wiki_dir)
        .err()
        .map(|e| e.to_string())
}

async fn handle_discover(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DiscoverPayload>,
) -> impl IntoResponse {
    let limit = payload.limit.unwrap_or(3);

    // Discover, crawl and re-analysis share the one local LLM server. Run
    // together, they time each other out, and the crawl counts those timeouts
    // as failures.
    let job = match state.jobs.try_claim(JobKind::Discover) {
        Ok(guard) => guard,
        Err(message) => return busy_json(message),
    };

    // Own task, same reason as in handle_crawl: a closed tab must not cut the
    // work off half-way through a video.
    let pipeline = state.pipeline.clone();
    let query = payload.query;
    let task = tokio::spawn(async move {
        // Dropped when the task ends, also by an error or a panic.
        let _job = job;
        let result = pipeline.run_discover(&query, limit, 2).await;
        let wiki_error = if result.is_ok() { regenerate_wiki(&pipeline) } else { None };
        (result, wiki_error)
    });

    match task.await {
        Ok((Ok(disc), wiki_error)) => Json(serde_json::json!({
            "status": "success",
            "query": disc.query,
            "candidates_found": disc.candidates_found,
            "evaluated": disc.items.len(),
            "evaluate_limit": limit,
            "items": disc.items,
            "wiki_error": wiki_error
        })),
        Ok((Err(e), _)) => Json(serde_json::json!({"status": "error", "message": e.to_string()})),
        Err(join_err) => Json(serde_json::json!({"status": "error", "message": join_err.to_string()})),
    }
}

/// Runs the LLM again over the stored transcript of one video (no YouTube
/// request). The stored insight is replaced only by a valid LLM answer; in
/// every other case the answer is an error and the stored row is unchanged.
async fn handle_reanalyze(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ReanalyzePayload>,
) -> impl IntoResponse {
    let video_id = payload.video_id.trim().to_string();
    if video_id.is_empty() {
        return Json(serde_json::json!({"status": "error", "message": "video_id is missing."}));
    }

    // Same reason as in handle_discover: one LLM server, one job at a time.
    // A long video is read in many sections, so this can take minutes.
    let job = match state.jobs.try_claim(JobKind::Reanalyze) {
        Ok(guard) => guard,
        Err(message) => return busy_json(message),
    };

    // Own task: a closed tab must not cut the LLM call off before the result is stored.
    let pipeline = state.pipeline.clone();
    let task = tokio::spawn(async move {
        // Dropped when the task ends, also by an error or a panic.
        let _job = job;
        let result = pipeline.reanalyze_video(&video_id).await;
        let wiki_error = if result.is_ok() { regenerate_wiki(&pipeline) } else { None };
        // Extra information for the answer: without it the insight's own flag is used.
        let meta = match &result {
            Ok(insight) => pipeline
                .db()
                .and_then(|db| db.get_insight_meta(&insight.video_id))
                .ok()
                .flatten(),
            Err(_) => None,
        };
        (result, meta, wiki_error)
    });

    match task.await {
        Ok((Ok(insight), meta, wiki_error)) => Json(reanalyze_answer_json(
            &insight,
            meta.as_ref(),
            state.pipeline.max_slop,
            wiki_error,
        )),
        // {:#} keeps the cause chain (why the stored insight was left unchanged).
        Ok((Err(e), _, _)) => Json(serde_json::json!({"status": "error", "message": format!("{:#}", e)})),
        Err(join_err) => Json(serde_json::json!({"status": "error", "message": join_err.to_string()})),
    }
}

const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YT-Wikinator — Autonomous YouTube Knowledge Wiki</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --card-hover: #1c2128;
      --border: #30363d;
      --border-accent: #388bfd;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --text-dim: #8b949e;
      --accent: #58a6ff;
      --accent-hover: #79c0ff;
      --success: #238636;
      --badge-green: #2ea043;
      --badge-red: #da3633;
      --badge-yellow: #d29922;
      --badge-purple: #a371f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding-bottom: 60px;
    }
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--border);
      padding: 16px 36px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .logo {
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--text-bright);
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
    }
    .logo span { font-size: 1.5rem; }
    .stats-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.88rem;
      flex-wrap: wrap;
    }
    .stat-pill {
      background: #21262d;
      padding: 6px 14px;
      border-radius: 20px;
      border: 1px solid var(--border);
      color: var(--text);
      cursor: pointer;
      font-size: 0.86rem;
      font-family: inherit;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      outline: none;
    }
    .stat-pill:hover {
      background: #30363d;
      border-color: var(--accent);
      color: var(--text-bright);
      transform: translateY(-1px);
    }
    .stat-pill.active {
      background: #1f6feb22;
      border-color: var(--accent);
      color: var(--text-bright);
      box-shadow: 0 0 10px rgba(88, 166, 255, 0.35);
    }
    .stat-pill strong { color: var(--accent); font-weight: 700; }
    .btn {
      background: #21262d;
      color: var(--text-bright);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { background: #30363d; border-color: var(--text-dim); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-primary { background: #238636; border-color: #2ea043; color: #fff; }
    .btn-primary:hover { background: #2ea043; }
    .btn-accent { background: #1f6feb; border-color: #388bfd; color: #fff; }
    .btn-accent:hover { background: #388bfd; }
    .btn-warn { background: #9e6a03; border-color: #bb8009; color: #fff; }
    .btn-warn:hover { background: #bb8009; }
    .btn-danger { background: #da3633; border-color: #f85149; color: #fff; }
    .btn-danger:hover { background: #b62324; }
    .btn-sm { padding: 4px 10px; font-size: 0.78rem; }
    
    .nav-tabs {
      display: flex;
      gap: 12px;
      padding: 14px 36px 0;
      border-bottom: 1px solid var(--border);
      background: #11151c;
    }
    .nav-tab {
      padding: 10px 18px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.95rem;
      color: var(--text-dim);
      border-bottom: 2px solid transparent;
      transition: color 0.15s, border-color 0.15s;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .nav-tab:hover { color: var(--text-bright); }
    .nav-tab.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }
    .nav-tab .tab-badge {
      background: #21262d;
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 10px;
      color: var(--text);
    }
    .nav-tab.active .tab-badge {
      background: #1f6feb33;
      color: var(--accent);
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px 36px;
    }

    /* Universal Controls & Filter Bar */
    .controls-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 12px 18px;
      border-radius: 8px;
      margin-bottom: 20px;
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      color: var(--text-dim);
    }
    .control-group select {
      background: #21262d;
      color: var(--text-bright);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
    }
    .control-group select:focus { border-color: var(--accent); }

    .view-toggle-btns {
      display: flex;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }
    .btn-toggle {
      background: #21262d;
      color: var(--text-dim);
      border: none;
      padding: 6px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-toggle.active {
      background: #1f6feb;
      color: #fff;
    }

    /* Active Filter Banner */
    .active-filter-banner {
      display: none;
      align-items: center;
      justify-content: space-between;
      background: #1f6feb18;
      border: 1px solid #1f6feb55;
      padding: 8px 16px;
      border-radius: 8px;
      margin-bottom: 20px;
      font-size: 0.88rem;
    }
    .active-filter-content {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .filter-tag-pill {
      background: #1f6feb33;
      color: var(--accent);
      padding: 2px 10px;
      border-radius: 12px;
      font-weight: 600;
    }

    /* Library filter pills: all, one per category of the user's, disqualified */
    .filter-pills {
      display: flex;
      gap: 10px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }
    .filter-pill {
      background: #21262d;
      color: var(--text-dim);
      padding: 6px 16px;
      border-radius: 20px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border);
      transition: all 0.15s;
    }
    .filter-pill:hover {
      color: var(--text-bright);
      border-color: #484f58;
    }
    .filter-pill.active {
      background: #1f6feb22;
      color: #58a6ff;
      border-color: #1f6feb;
    }
    .category-manager {
      display: none;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 18px;
    }

    .search-section {
      margin-bottom: 22px;
      display: flex;
      gap: 12px;
    }
    .search-input {
      width: 100%;
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text-bright);
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .search-input:focus { border-color: var(--accent); }

    /* Card Grid */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(390px, 1fr));
      gap: 20px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
    }
    .card:hover {
      border-color: #484f58;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .card-title {
      font-size: 1.08rem;
      font-weight: 600;
      color: var(--text-bright);
      text-decoration: none;
      line-height: 1.4;
    }
    .card-title:hover { color: var(--accent); }
    .badge {
      padding: 3px 8px;
      border-radius: 12px;
      font-size: 0.72rem;
      font-weight: 700;
      white-space: nowrap;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-green { background: #23863633; color: #3fb950; border: 1px solid #238636; }
    .badge-yellow { background: #d2992233; color: #e3b341; border: 1px solid #bb8009; }
    .badge-red { background: #da363333; color: #f85149; border: 1px solid #da3633; }
    .badge-category { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb; font-size: 0.72rem; cursor: pointer; }
    /* Category of one video: chosen by the model, or set by hand */
    .category-select {
      background: #21262d;
      color: var(--text-dim);
      border: 1px solid var(--border);
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 0.74rem;
      max-width: 190px;
      outline: none;
      cursor: pointer;
    }
    .category-select:focus { border-color: var(--accent); }
    .category-by-hand { font-size: 0.72rem; color: var(--text-dim); font-style: italic; }
    
    .card-meta {
      font-size: 0.82rem;
      color: var(--text-dim);
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .wiki-creator-link {
      color: var(--accent);
      cursor: pointer;
      font-weight: 600;
      transition: color 0.15s;
    }
    .wiki-creator-link:hover {
      color: var(--accent-hover);
      text-decoration: underline;
    }
    .card-summary {
      font-size: 0.90rem;
      color: var(--text);
      line-height: 1.5;
    }
    .quote-box {
      border-left: 3px solid var(--accent);
      padding: 8px 14px;
      background: #1c212888;
      border-radius: 0 6px 6px 0;
      font-size: 0.85rem;
      font-style: italic;
      color: #d2a8ff;
    }
    .section-title {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 2px;
    }
    .insights-list {
      list-style-type: disc;
      padding-left: 18px;
      font-size: 0.85rem;
      color: #b1bac4;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: auto;
      padding-top: 8px;
    }
    .tag {
      background: #21262d;
      color: var(--accent);
      padding: 3px 9px;
      border-radius: 6px;
      font-size: 0.74rem;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.15s;
      border: 1px solid transparent;
    }
    .tag:hover {
      background: #30363d;
      color: var(--accent-hover);
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    .tag-mention {
      background: #a371f722;
      color: #d2a8ff;
      border-color: #a371f744;
    }
    .card-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 8px;
      padding-top: 10px;
      border-top: 1px solid #21262d;
    }

    /* Dense Wiki Table */
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--card-bg);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    th, td {
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 0.90rem;
    }
    th {
      background: #21262d;
      color: var(--text-bright);
      font-weight: 600;
      user-select: none;
    }
    tr:hover td { background: var(--card-hover); }

    /* Topic Taxonomy Grid */
    .topic-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      transition: all 0.15s;
    }
    .topic-card:hover {
      border-color: var(--accent);
      background: var(--card-hover);
      transform: translateY(-1px);
    }

    /* Modals & Backdrop */
    .modal-backdrop {
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      z-index: 1000;
      justify-content: center;
      align-items: center;
    }
    .modal {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      width: 90%;
      max-width: 850px;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    }
    .modal-header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-title { font-size: 1.2rem; font-weight: 700; color: var(--text-bright); }
    .modal-close {
      font-size: 1.5rem;
      color: var(--text-dim);
      cursor: pointer;
      line-height: 1;
    }
    .modal-close:hover { color: var(--text-bright); }
    .modal-body {
      padding: 20px 24px;
      overflow-y: auto;
      font-size: 0.92rem;
      white-space: pre-wrap;
      line-height: 1.7;
      color: #b1bac4;
      font-family: monospace;
    }
    .modal-meta-box {
      background: #21262d;
      padding: 12px 18px;
      border-radius: 8px;
      margin-bottom: 16px;
      display: flex;
      gap: 16px;
      font-size: 0.85rem;
      flex-wrap: wrap;
    }
    .toast {
      position: fixed;
      bottom: 25px;
      right: 25px;
      max-width: min(640px, 90vw);
      background: #1f6feb;
      color: #fff;
      padding: 12px 20px;
      border-radius: 8px;
      font-weight: 600;
      box-shadow: 0 4px 15px rgba(0,0,0,0.5);
      z-index: 9999;
      display: none;
    }

    /* Live Crawl Progress Dashboard */
    .crawl-progress-card {
      display: none;
      background: #161b22;
      border: 1px solid #1f6feb;
      border-radius: 8px;
      margin: 14px 24px;
      padding: 14px 18px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      animation: fadeInDown 0.25s ease-out;
    }
    @keyframes fadeInDown {
      from { opacity: 0; transform: translateY(-8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .crawl-progress-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .crawl-progress-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-bright);
    }
    .crawl-progress-pulse {
      display: inline-block;
      width: 10px;
      height: 10px;
      background: #58a6ff;
      border-radius: 50%;
      box-shadow: 0 0 8px #58a6ff;
      animation: pulseDot 1.4s infinite;
    }
    @keyframes pulseDot {
      0% { transform: scale(0.9); opacity: 0.7; }
      50% { transform: scale(1.3); opacity: 1; }
      100% { transform: scale(0.9); opacity: 0.7; }
    }
    .crawl-progress-badges {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
    }
    .badge-kept-pill {
      background: rgba(63, 185, 80, 0.18);
      color: #3fb950;
      border: 1px solid rgba(63, 185, 80, 0.4);
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 600;
    }
    .badge-filtered-pill {
      background: rgba(248, 81, 73, 0.18);
      color: #f85149;
      border: 1px solid rgba(248, 81, 73, 0.4);
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 600;
    }
    .crawl-pct-pill {
      color: #58a6ff;
      font-weight: 700;
      font-size: 0.95rem;
      min-width: 44px;
      text-align: right;
    }
    .crawl-progress-track {
      height: 9px;
      background: #21262d;
      border-radius: 5px;
      overflow: hidden;
      margin-bottom: 8px;
      position: relative;
    }
    .crawl-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #1f6feb, #58a6ff);
      border-radius: 5px;
      transition: width 0.35s ease, background 0.3s;
    }
    .crawl-progress-bar.complete {
      background: #3fb950 !important;
    }
    .crawl-progress-details {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      color: var(--text-dim);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      gap: 16px;
    }
    .crawl-detail-channel {
      color: #8b949e;
      flex-shrink: 0;
    }
    .crawl-detail-video {
      color: var(--text-bright);
      overflow: hidden;
      text-overflow: ellipsis;
      flex-grow: 1;
      text-align: right;
    }
    .badge-neutral-pill {
      background: #21262d;
      color: var(--text-dim);
      border: 1px solid var(--border);
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 600;
    }

    /* Engine status chip: filled from GET /api/status */
    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.76rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #21262d;
      color: var(--text-dim);
      cursor: pointer;
      white-space: nowrap;
    }
    .status-chip .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-dim);
    }
    .status-chip.online { color: #3fb950; border-color: rgba(63, 185, 80, 0.4); }
    .status-chip.online .status-dot { background: #3fb950; }
    .status-chip.offline { color: #f85149; border-color: rgba(248, 81, 73, 0.4); }
    .status-chip.offline .status-dot { background: #f85149; }
    .status-chip.heuristic { color: var(--badge-yellow); border-color: rgba(210, 153, 34, 0.45); }
    .status-chip.heuristic .status-dot { background: var(--badge-yellow); }

    /* Not reviewed by an LLM / not scored at all */
    .badge-unreviewed {
      background: rgba(210, 153, 34, 0.15);
      color: var(--badge-yellow);
      border: 1px dashed rgba(210, 153, 34, 0.6);
    }
    .badge-not-scored {
      background: #21262d;
      color: var(--text-dim);
      border: 1px solid var(--border);
    }
    .coverage-note {
      font-size: 0.76rem;
      color: var(--badge-yellow);
    }
    /* How the model read the transcript, and where the transcript came from */
    .read-note {
      font-size: 0.76rem;
      color: var(--text-dim);
    }
    .read-note.read-note-warn { color: var(--badge-yellow); }
    /* Library counts under the header: filled from GET /api/status */
    .library-counts {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      padding: 8px 36px;
      background: #11151c;
      border-bottom: 1px solid var(--border);
      font-size: 0.8rem;
      color: var(--text-dim);
    }
    .count-chip {
      background: #21262d;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2px 10px;
      color: var(--text-dim);
      font-size: 0.8rem;
      font-family: inherit;
    }
    .count-chip strong { color: var(--text-bright); }
    .count-chip-link { cursor: pointer; }
    .count-chip-link:hover { border-color: var(--accent); color: var(--text-bright); }
    .btn-remove { color: #f85149; }
    .btn-remove:hover { border-color: #f85149; }
    .toast-action {
      margin-left: 12px;
      background: #fff;
      color: #1f6feb;
      border: none;
      border-radius: 6px;
      padding: 3px 10px;
      font-weight: 700;
      cursor: pointer;
    }
    /* A mentioned name that is not a tracked channel: text, not a filter */
    .mention-plain {
      color: var(--text-dim);
      font-size: 0.76rem;
      display: inline-flex;
      align-items: center;
      gap: 3px;
    }
    .mention-search {
      cursor: pointer;
      color: var(--accent);
      background: none;
      border: none;
      font-size: 0.76rem;
      padding: 0 2px;
    }
    .related-row {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      background: #21262d;
      padding: 8px 10px;
      border-radius: 6px;
      margin-bottom: 6px;
      font-size: 0.86rem;
    }
    .related-source {
      font-size: 0.74rem;
      color: var(--text-dim);
    }
  </style>

</head>
<body>
  <header>
    <div class="logo" onclick="resetAllFilters()">
      <span>📚</span> YT-Wikinator
    </div>
    <!-- Filled from GET /api/status: what really analyses the videos, and whether it answers -->
    <span id="status-chip" class="status-chip" onclick="loadStatus(true)" title="Checking the analysis engine...">
      <span class="status-dot"></span><span id="status-chip-text">Engine: checking...</span>
    </span>
    <div class="stats-bar">
      <!-- Clickable Interactive Stat Buttons -->
      <button class="stat-pill" id="pill-insights" onclick="switchTab('insights')" title="Click to view all video insights">
        <strong id="stat-insights">-</strong> Insights
      </button>
      <button class="stat-pill" id="pill-topics" onclick="switchTab('topics')" title="Click to view topics taxonomy">
        <strong id="stat-topics">-</strong> Topics
      </button>
      <button class="stat-pill" id="pill-channels" onclick="switchTab('channels')" title="Click to view creator scorecards">
        <strong id="stat-channels">-</strong> Creators
      </button>
      <button class="stat-pill" id="pill-slop" onclick="sortBySignalPurity()" title="Click to sort library by signal purity (lowest slop first)">
        Avg Slop: <strong id="stat-slop">-</strong>
      </button>
      
      <div style="display: flex; align-items: center; gap: 4px; background: #21262d; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--border);">
        <span style="font-size: 0.75rem; color: var(--text-dim);">Depth:</span>
        <select id="crawl-depth-select" style="background: transparent; color: var(--accent); border: none; font-size: 0.8rem; font-weight: bold; cursor: pointer; outline: none;">
          <option value="1">1 vid</option>
          <option value="2">2 vids</option>
          <option value="3" selected>3 vids</option>
          <option value="5">5 vids</option>
          <option value="10">10 vids</option>
        </select>
      </div>
      <button class="btn btn-accent" id="btn-quick-crawl" onclick="runQuickCrawl()">⚡ Crawl All</button>
      <button class="btn" onclick="openDiscoverModal()">🔍 Discover Topic</button>
      <button class="btn btn-primary" onclick="syncWiki()">Sync Wiki Files</button>
    </div>
  </header>

  <!-- Every analysed video is in exactly one of these groups (GET /api/status).
       The sixth one is shown only when the library holds such rows. -->
  <div id="library-counts" class="library-counts">
    <span>All analysed videos:</span>
    <span class="count-chip" title="In the library and in the wiki"><strong id="lib-count-kept">-</strong> Kept</span>
    <span class="count-chip" title="Judged on quality only: the slop score is above the keep threshold"><strong id="lib-count-below">-</strong> Below quality cutoff</span>
    <span class="count-chip" title="The subject itself is fringe, nonsense or politics. Every other subject is welcome."><strong id="lib-count-offtopic">-</strong> Off-topic</span>
    <span class="count-chip" title="Very short video or almost no speech: not judged, and not counted in the channel's rolling slop"><strong id="lib-count-insufficient">-</strong> Not enough information</span>
    <button class="count-chip count-chip-link" onclick="showExcludedView()" title="Removed from the wiki by you. Click to list them and to restore one."><strong id="lib-count-excluded">-</strong> Disqualified</button>
  </div>

  <!-- Live Crawl Progress Dashboard Bar -->
  <div id="crawl-progress-card" class="crawl-progress-card">
    <div class="crawl-progress-header">
      <div class="crawl-progress-title">
        <span class="crawl-progress-pulse" id="crawl-pulse-dot"></span>
        <span id="crawl-progress-msg">Starting crawl...</span>
      </div>
      <div class="crawl-progress-badges">
        <span class="badge-neutral-pill" id="crawl-stat-count" title="Videos processed / new videos found in the feeds. The total is known once all feeds are checked.">-</span>
        <span class="badge-kept-pill" id="crawl-stat-kept" title="Acceptable subject and a slop score at or below the keep threshold">0 Kept</span>
        <span class="badge-filtered-pill" id="crawl-stat-filtered" title="Judged on quality only: slop score above the keep threshold">0 Below cutoff</span>
        <span class="badge-neutral-pill" id="crawl-stat-policy" title="The subject itself is fringe, nonsense or politics. Every other subject is welcome.">0 Off-topic</span>
        <span class="badge-neutral-pill" id="crawl-stat-insufficient" title="Very short video or almost no speech: not judged, and not counted in the channel's rolling slop">0 Not enough info</span>
        <span class="badge-neutral-pill" id="crawl-stat-failed" title="Could not be processed in this crawl">0 Failed</span>
        <span class="crawl-pct-pill" id="crawl-stat-percent">0%</span>
      </div>
    </div>
    <div class="crawl-progress-track">
      <div class="crawl-progress-bar" id="crawl-progress-fill" style="width: 0%;"></div>
    </div>
    <div class="crawl-progress-details">
      <span class="crawl-detail-channel" id="crawl-detail-channel">Channel: -</span>
      <span class="crawl-detail-video" id="crawl-detail-video"></span>
    </div>
  </div>


  <!-- Navigation Tabs -->
  <div class="nav-tabs">
    <div class="nav-tab active" id="tab-btn-insights" onclick="switchTab('insights')">
      💡 Insights Library <span class="tab-badge" id="tab-count-insights">-</span>
    </div>
    <div class="nav-tab" id="tab-btn-topics" onclick="switchTab('topics')">
      🧭 Concepts & Topics Taxonomy <span class="tab-badge" id="tab-count-topics">-</span>
    </div>
    <div class="nav-tab" id="tab-btn-channels" onclick="switchTab('channels')">
      🎙️ Creators & Scorecards <span class="tab-badge" id="tab-count-channels">-</span>
    </div>
  </div>

  <div class="container">
    <!-- Shown when the library cannot be loaded from the server -->
    <div id="load-error-banner" style="display: none; background: #da363322; border: 1px solid #da3633; color: #f85149; padding: 10px 16px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9rem;"></div>

    <!-- Active Filter Notification Bar -->
    <div id="active-filter-banner" class="active-filter-banner">
      <div class="active-filter-content">
        <span>🔎 Active Filter:</span>
        <span id="active-filter-label" class="filter-tag-pill">-</span>
        <span id="active-filter-count" style="color: var(--text-dim); font-size: 0.82rem;">(0 results)</span>
      </div>
      <button class="btn btn-sm" onclick="resetAllFilters()">✕ Clear Filter</button>
    </div>

    <!-- Library filter: everything, one pill per category the user defined (none by default), disqualified -->
    <div id="library-filter-bar" class="filter-pills">
      <div class="filter-pill active" data-filter="all" onclick="filterLibrary('all')">🌟 All Library (<span id="count-all">0</span>)</div>
      <span id="category-pills" style="display: none;"></span>
      <div class="filter-pill" data-filter="excluded" onclick="filterLibrary('excluded')" title="Videos you removed from the wiki. Listed here so that you can restore them.">🚫 Disqualified (<span id="count-excluded">0</span>)</div>
      <div class="filter-pill" id="btn-category-manager" onclick="toggleCategoryManager()" title="Define your own categories. There are none until you add one.">🗂️ Categories…</div>
    </div>

    <!-- Category manager: list, add, rename, describe and remove the user's own categories -->
    <div id="category-manager" class="category-manager">
      <p style="font-size: 0.86rem; color: var(--text-dim); margin-bottom: 12px;">
        Categories are yours to define; there are none by default. Once you have some, the LLM assigns each video it analyses
        to one of them, or to none. A category is only a label: it is not used when deciding whether a video is kept. Adding or changing categories can shift individual scores slightly.
        Videos that are already analysed get a category with the button below, without being analysed again. You can also set the category of a video yourself on its card: a category set by hand stays.
        Renaming a category takes its videos along; removing it keeps them.
      </p>
      <div style="display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap;">
        <input type="text" id="add-category-name" class="search-input" style="flex: 1; min-width: 180px;" maxlength="60" placeholder="Category name" />
        <input type="text" id="add-category-description" class="search-input" style="flex: 2; min-width: 240px;" maxlength="300" placeholder="What belongs in it (optional; the LLM reads this)" />
        <button class="btn btn-primary" onclick="submitAddCategory()">+ Add Category</button>
      </div>
      <table>
        <thead>
          <tr>
            <th style="width: 240px;">Category</th>
            <th>Description</th>
            <th style="width: 120px;">Kept videos</th>
            <th style="width: 280px; text-align: right;">Action</th>
          </tr>
        </thead>
        <tbody id="categories-table-body"></tbody>
      </table>
      <div style="display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; align-items: center;">
        <button class="btn" id="btn-categorize" onclick="runCategorize()" title="Asks the LLM for the category of every analysed video that has none. One short request per video, from its stored summary: no transcript is read again and no score changes. Videos whose category you set by hand are left alone.">🏷️ Assign categories to existing videos</button>
        <span id="categorize-status" style="font-size: 0.84rem; color: var(--text-dim);"></span>
      </div>
    </div>

    <!-- INSIGHTS VIEW CONTROLS TOOLBAR -->
    <div id="insights-controls-bar" class="controls-bar">
      <div class="control-group">
        <label>Sort By:</label>
        <select id="insights-sort-select" onchange="applySortingAndFiltering()">
          <option value="date-desc">📅 Date (Newest First)</option>
          <option value="date-asc">📅 Date (Oldest First)</option>
          <option value="title-asc">🔤 Alphabetical (Title A-Z)</option>
          <option value="title-desc">🔤 Alphabetical (Title Z-A)</option>
          <option value="channel-asc">🎙️ Creator / Channel (A-Z)</option>
          <option value="slop-asc">🎯 Signal Purity (Lowest Slop First)</option>
          <option value="slop-desc">🚨 Slop Level (Highest Slop First)</option>
          <option value="category-asc">🏷️ Category</option>
        </select>
      </div>

      <div class="control-group">
        <label>Creator:</label>
        <select id="insights-channel-select" onchange="onChannelDropdownChange()">
          <option value="all">All Tracked Creators</option>
        </select>
      </div>

      <div class="control-group">
        <label>Signal:</label>
        <select id="insights-signal-select" onchange="applySortingAndFiltering()">
          <option value="all">All Tiers</option>
          <option value="pure">🟢 Pure Signal (Slop &le; 20)</option>
          <option value="high">🔵 High Signal (Slop &le; 35)</option>
          <option value="moderate">🟡 Moderate (Slop &gt; 35)</option>
        </select>
      </div>

      <div class="control-group">
        <label>Engine:</label>
        <select id="insights-engine-select" onchange="applySortingAndFiltering()">
          <option value="all">All</option>
          <option value="llm">Reviewed by an LLM</option>
          <option value="heuristic">Unreviewed (heuristic)</option>
        </select>
      </div>

      <div class="control-group" style="margin-left: auto;">
        <label>Layout:</label>
        <div class="view-toggle-btns">
          <button id="btn-layout-cards" class="btn-toggle active" onclick="setLayoutMode('cards')">🗂️ Cards</button>
          <button id="btn-layout-table" class="btn-toggle" onclick="setLayoutMode('table')">📋 Wiki Index</button>
        </div>
      </div>
    </div>

    <!-- TOPICS VIEW CONTROLS TOOLBAR -->
    <div id="topics-controls-bar" class="controls-bar" style="display: none;">
      <div class="control-group">
        <label>Sort Topics By:</label>
        <select id="topics-sort-select" onchange="onTopicSearch(document.getElementById('topics-filter-input').value)">
          <option value="count-desc">🔢 Video Count (Most Videos First)</option>
          <option value="count-asc">🔢 Video Count (Fewest First)</option>
          <option value="name-asc">🔤 Alphabetical (A-Z)</option>
          <option value="name-desc">🔤 Alphabetical (Z-A)</option>
        </select>
      </div>
      <div class="control-group" style="flex: 1; max-width: 450px;">
        <input type="text" id="topics-filter-input" class="search-input" style="padding: 6px 14px; font-size: 0.85rem;" placeholder="Filter topics by keyword..." oninput="onTopicSearch(this.value)">
      </div>
      <div style="font-size: 0.85rem; color: var(--text-dim); margin-left: auto;">
        💡 Click any topic to view all connected dossiers & cross-references
      </div>
    </div>

    <!-- CHANNELS VIEW CONTROLS TOOLBAR -->
    <div id="channels-controls-bar" class="controls-bar" style="display: none;">
      <div class="control-group">
        <label>Sort Creators By:</label>
        <select id="channels-sort-select" onchange="onChannelSearch(document.getElementById('channels-filter-input').value)">
          <option value="health-asc">🟢 Health Status (Lowest Slop First)</option>
          <option value="kept-desc">📚 Kept Dossiers (Most First)</option>
          <option value="title-asc">🔤 Creator Name (A-Z)</option>
          <option value="title-desc">🔤 Creator Name (Z-A)</option>
          <option value="total-desc">📊 Total Evaluated</option>
        </select>
      </div>
      <div class="control-group" style="flex: 1; max-width: 450px;">
        <input type="text" id="channels-filter-input" class="search-input" style="padding: 6px 14px; font-size: 0.85rem;" placeholder="Filter creators by handle or title..." oninput="onChannelSearch(this.value)">
      </div>
      <div style="font-size: 0.85rem; color: var(--text-dim); margin-left: auto;">
        🎙️ Click creator name to explore their dossiers in the wiki
      </div>
    </div>

    <!-- Global Search Bar -->
    <div class="search-section" id="global-search-container">
      <input type="text" id="search-box" class="search-input" placeholder="Search titles, summaries, concepts, categories, takeaways or quotes..." oninput="onSearch(this.value)">
    </div>

    <!-- 1. INSIGHTS VIEW (Cards Grid) -->
    <div id="view-insights-cards" class="grid"></div>

    <!-- 1B. INSIGHTS VIEW (Dense Wiki Index Table) -->
    <div id="view-insights-table" style="display: none;">
      <table>
        <thead>
          <tr>
            <th style="width: 100px; cursor: pointer;" onclick="toggleTableSort('date')">📅 Date</th>
            <th style="width: 90px; cursor: pointer;" onclick="toggleTableSort('slop')">🎯 Slop</th>
            <th style="cursor: pointer;" onclick="toggleTableSort('title')">Video Title & Dossier</th>
            <th style="width: 180px; cursor: pointer;" onclick="toggleTableSort('channel')">🎙️ Creator</th>
            <th style="width: 140px; cursor: pointer;" onclick="toggleTableSort('category')">🏷️ Category</th>
            <th>Key Concept Topics</th>
            <th style="width: 140px; text-align: right;">Action</th>
          </tr>
        </thead>
        <tbody id="insights-table-body"></tbody>
      </table>
    </div>

    <!-- 1C. DISQUALIFIED VIEW: videos the owner removed, each with a Restore button -->
    <div id="view-excluded" style="display: none;">
      <p style="font-size: 0.86rem; color: var(--text-dim); margin-bottom: 12px;">
        Videos you removed from the wiki. They stay out of the library, also after a re-analysis, until you restore them.
        Sync Wiki takes a removed video out of the index, topic and creator pages. Its own page file stays in the wiki folder
        until you move it out; a restored video gets its page back with Sync Wiki.
        <button class="btn btn-sm" style="margin-left: 6px;" title="Writes the wiki files and moves pages of videos, topics, creators and categories that have left the library into the folder .ytc-pruned inside the wiki folder. Nothing is deleted." onclick="syncWikiAndMove()">Sync wiki and move removed pages</button>
      </p>
      <table>
        <thead>
          <tr>
            <th>Video</th>
            <th style="width: 200px;">🎙️ Creator</th>
            <th>Reason</th>
            <th style="width: 120px;">Removed on</th>
            <th style="width: 120px; text-align: right;">Action</th>
          </tr>
        </thead>
        <tbody id="excluded-table-body"></tbody>
      </table>
    </div>

    <!-- 2. TOPICS VIEW -->
    <div id="view-topics" class="grid" style="display: none;"></div>

    <!-- 3. CHANNELS VIEW -->
    <div id="view-channels" style="display: none;">
      <div class="add-channel-bar" style="display: flex; gap: 12px; margin-bottom: 24px; background: var(--card-bg); padding: 16px; border-radius: 8px; border: 1px solid var(--border);">
        <input type="text" id="add-channel-input" class="search-input" placeholder="Enter creator @handle (e.g. @somechannel), channel URL, or UC ID..." />
        <button class="btn btn-primary" onclick="submitAddChannel()">+ Track Channel</button>
      </div>

      <table>
        <thead>
          <tr>
            <th>Creator / Channel</th>
            <th>Health Status</th>
            <th>Rolling Slop</th>
            <th>Kept / Evaluated</th>
            <th>Channel ID</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="channels-table-body"></tbody>
      </table>
    </div>
  </div>

  <!-- Transcript Modal -->
  <div id="transcript-modal" class="modal-backdrop" onclick="closeModal(event)">
    <div class="modal" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div id="modal-video-title" class="modal-title">Cleaned Transcript</div>
        <div class="modal-close" onclick="closeModal()">&times;</div>
      </div>
      <div style="padding: 16px 24px 0;">
        <div class="modal-meta-box">
          <div title="Time removed from the transcript in segments submitted to SponsorBlock by its community (sponsor, self-promotion, intro, outro and similar). Videos without submitted segments lose nothing. Transcripts stored by older versions show the summed length of the submitted segments instead.">✂️ Removed (SponsorBlock): <strong id="modal-stripped-secs">-</strong> s in <strong id="modal-stripped-segs">-</strong> segments</div>
          <div>Character Count: <strong id="modal-char-count">-</strong></div>
          <div id="modal-coverage" class="coverage-note" style="display: none;"></div>
        </div>
      </div>
      <div id="modal-transcript-text" class="modal-body">Loading transcript...</div>
    </div>
  </div>

  <!-- Discover Modal -->
  <div id="discover-modal" class="modal-backdrop" onclick="closeDiscoverModal(event)">
    <div class="modal" style="max-width: 680px;" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-title">🔍 Discover Topics & Add Creators</div>
        <div class="modal-close" onclick="closeDiscoverModal()">&times;</div>
      </div>
      <div style="padding: 20px 24px; display: flex; flex-direction: column; gap: 14px;">
        <p style="font-size: 0.9rem; color: #8b949e;">
          Searches YouTube for candidate videos and scores the first few with
          <span id="discover-engine-text">the configured analysis engine</span>.
          When an LLM gives a video a low slop score, its creator is added to the watchlist as a normal (active) channel.
          Any subject is welcome: only fringe, nonsense and politics are rejected for their topic.
        </p>
        <input type="text" id="discover-input" class="search-input" placeholder="Type a subject to search for..." />

        <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 6px;">
          <button class="btn" onclick="closeDiscoverModal()">Cancel</button>
          <button class="btn btn-accent" id="btn-submit-discover" onclick="submitDiscover()">Harvest Candidates</button>
        </div>
        <div id="discover-results" style="margin-top: 10px; max-height: 350px; overflow-y: auto; display: none;"></div>
      </div>
    </div>
  </div>

  <!-- Related Channels Modal -->
  <div id="related-modal" class="modal-backdrop" onclick="closeRelatedModal(event)">
    <div class="modal" style="max-width: 680px;" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div id="related-modal-title" class="modal-title">🕸️ Related creators</div>
        <div class="modal-close" onclick="closeRelatedModal()">&times;</div>
      </div>
      <div style="padding: 20px 24px; display: flex; flex-direction: column; gap: 12px;">
        <p style="font-size: 0.86rem; color: #8b949e;">
          Suggestions come from two sources, shown on each line: creators named in this channel's kept dossiers,
          and channel items on the creator's YouTube page. Only the ones you tick are tracked.
        </p>
        <div id="related-results" style="max-height: 380px; overflow-y: auto;"></div>
        <div style="display: flex; justify-content: flex-end; gap: 10px;">
          <button class="btn" onclick="closeRelatedModal()">Close</button>
          <button class="btn btn-primary" id="btn-track-related" onclick="trackSelectedRelated()" disabled>Track selected</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Toast Notification -->
  <div id="toast" class="toast"></div>

  <script>
    let allInsights = [];
    let allTopics = [];
    let allChannels = [];
    // Answer of GET /api/excluded: the videos the owner disqualified.
    let allExcluded = [];
    // Answer of GET /api/categories: the categories the user defined.
    let allCategories = [];
    let activeTab = 'insights';
    // What the library list shows: 'all', 'excluded', or 'category' (then
    // selectedCategory names one of allCategories).
    let libraryFilter = 'all';
    let selectedCategory = null;
    let activeCreatorFilter = null;
    let activeTopicFilter = null;
    let layoutMode = 'cards'; // 'cards' or 'table'
    // Answer of GET /api/status; null until it has been read (or when it failed).
    let appStatus = null;
    // True once the library has been read from the server at least once.
    let libraryLoaded = false;

    // One timer for all toasts: a timer left over from the toast before must
    // not hide a later one (the Undo toast has to stay for its whole time).
    let toastTimer = null;

    // action = { label, run }: adds one button to the toast, for example Undo.
    function showToast(msg, duration = 3500, action = null) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      if (action) {
        const btn = document.createElement('button');
        btn.className = 'toast-action';
        btn.innerText = action.label;
        btn.onclick = () => {
          toast.style.display = 'none';
          action.run();
        };
        toast.appendChild(btn);
      }
      toast.style.display = 'block';
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => {
        toastTimer = null;
        toast.style.display = 'none';
      }, duration);
    }

    // The keep threshold the server really applies; null while unknown.
    function maxSlop() {
      const n = appStatus ? Number(appStatus.max_slop) : NaN;
      return Number.isFinite(n) ? n : null;
    }

    function hasLlm() {
      return !!appStatus && appStatus.analyzer_mode === 'llm';
    }

    // Name of the analysis engine for sentences, taken from the server.
    function engineName() {
      if (!appStatus) return 'the configured analysis engine';
      if (hasLlm()) return `the LLM ${appStatus.llm_model || '(unnamed model)'}`;
      return 'the word-count heuristic (no LLM is configured)';
    }

    // Green up to 20, yellow up to the keep threshold, red above it. While the
    // threshold is unknown nothing is painted red.
    function slopBadgeClass(score) {
      if (score <= 20) return 'badge-green';
      const limit = maxSlop();
      return (limit === null || score <= limit) ? 'badge-yellow' : 'badge-red';
    }

    function renderStatusChip() {
      const chip = document.getElementById('status-chip');
      const text = document.getElementById('status-chip-text');
      if (!chip || !text) return;
      chip.classList.remove('online', 'offline', 'heuristic');
      if (!appStatus) {
        text.innerText = 'Engine: unknown';
        chip.title = 'The server did not answer /api/status. Click to check again.';
      } else if (hasLlm()) {
        const up = appStatus.llm_reachable === true;
        chip.classList.add(up ? 'online' : 'offline');
        text.innerText = `LLM: ${appStatus.llm_model || '(unnamed model)'} ${up ? 'online' : 'offline'}`;
        chip.title = `LLM server: ${appStatus.llm_url_host || '?'}\n` +
          (up ? 'It answered just now.' : 'It did not answer within about 2 seconds. Crawls stop until it is started.') +
          `\nKeep threshold: slop score up to ${appStatus.max_slop}\nWiki folder: ${appStatus.wiki_dir}\nDatabase: ${appStatus.db_path}\nytc ${appStatus.version}\nClick to check again.`;
      } else {
        chip.classList.add('heuristic');
        text.innerText = 'Heuristic mode (no LLM)';
        chip.title = 'No LLM is configured. Videos are scored by counting words and phrases and are stored as unreviewed.' +
          `\nKeep threshold: slop score up to ${appStatus.max_slop}\nWiki folder: ${appStatus.wiki_dir}\nDatabase: ${appStatus.db_path}\nytc ${appStatus.version}\nClick to check again.`;
      }
      const eng = document.getElementById('discover-engine-text');
      if (eng) eng.innerText = engineName();
    }

    // The library counts under the header. A dash while they are unknown:
    // a zero would claim that nothing was analysed.
    function renderLibraryCounts() {
      const counts = appStatus ? appStatus.counts : null;
      const set = (id, key) => {
        const el = document.getElementById(id);
        if (!el) return;
        const n = counts ? Number(counts[key]) : NaN;
        el.innerText = Number.isFinite(n) ? n : '-';
      };
      set('lib-count-kept', 'kept');
      set('lib-count-below', 'below_cutoff');
      set('lib-count-offtopic', 'off_topic');
      set('lib-count-insufficient', 'insufficient');
      set('lib-count-excluded', 'excluded');
    }

    // Reads what really runs on the server. A failure leaves the shown
    // answer in place when there is one.
    async function loadStatus(announce) {
      try {
        const res = await fetchJson('/api/status');
        if (!res || typeof res.analyzer_mode !== 'string') throw new Error('unexpected answer');
        appStatus = res;
      } catch (e) {
        console.warn('Status check failed:', e);
        if (announce) showToast('Could not read the engine status from the server.');
      }
      renderStatusChip();
      renderLibraryCounts();
      if (announce && appStatus) {
        showToast(hasLlm()
          ? `LLM ${appStatus.llm_model || ''} is ${appStatus.llm_reachable ? 'online' : 'offline'}`
          : 'Heuristic mode: no LLM is configured');
        // Badge colours and the Re-analyse buttons depend on the status.
        renderCurrentTab();
      }
    }

    // Every string that comes from YouTube, the database or the LLM goes through
    // esc() before it is put into innerHTML or into an attribute.
    function esc(s) {
      return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    // Argument of an inline handler: a JS literal first, then HTML-escaped,
    // because the browser decodes the attribute before it runs the code.
    function jsArg(v) {
      return esc(JSON.stringify(v ?? ''));
    }

    // Only http(s) links are rendered as links.
    function safeUrl(u) {
      return /^https?:\/\//i.test(u || '') ? esc(u) : 'about:blank';
    }

    async function fetchJson(url, options) {
      const r = await fetch(url, options);
      if (!r.ok) {
        let detail = '';
        try {
          const body = await r.json();
          detail = body.error || body.message || '';
        } catch (_) {}
        throw new Error(`${url} answered HTTP ${r.status}${detail ? ' (' + detail + ')' : ''}`);
      }
      return r.json();
    }

    function showLoadError(msg) {
      const el = document.getElementById('load-error-banner');
      if (!el) return;
      el.innerText = msg || '';
      el.style.display = msg ? 'block' : 'none';
    }

    // Returns false when the server could not be read. What is on screen stays
    // as it is in that case: an emptied library would look like lost data.
    async function loadData() {
      let insRes, topRes, chRes, exRes, catRes;
      try {
        [insRes, topRes, chRes, exRes, catRes] = await Promise.all([
          fetchJson('/api/insights'),
          fetchJson('/api/topics'),
          fetchJson('/api/channels'),
          // The disqualified list and the categories are side views: when one
          // cannot be read, the library is still shown and the category list stays as it is.
          fetchJson('/api/excluded').catch(() => null),
          fetchJson('/api/categories').catch(() => null)
        ]);
        if (!Array.isArray(insRes) || !Array.isArray(topRes) || !Array.isArray(chRes)) {
          throw new Error('the server sent an unexpected answer');
        }
      } catch (e) {
        showLoadError(`Cannot load the library from the YTC server: ${e.message || e}. Check that the server is still running, then reload this page.`);
        return false;
      }
      showLoadError('');

      allInsights = insRes;
      allTopics = topRes;
      allChannels = chRes;
      if (Array.isArray(exRes)) allExcluded = exRes;
      if (Array.isArray(catRes)) allCategories = catRes;
      const firstLoad = !libraryLoaded;
      libraryLoaded = true;

      // Update stat pill counters & tab badges
      document.getElementById('stat-insights').innerText = allInsights.length;
      document.getElementById('stat-topics').innerText = allTopics.length;
      document.getElementById('stat-channels').innerText = allChannels.length;

      document.getElementById('tab-count-insights').innerText = allInsights.length;
      document.getElementById('tab-count-topics').innerText = allTopics.length;
      document.getElementById('tab-count-channels').innerText = allChannels.length;

      document.getElementById('count-all').innerText = allInsights.length;
      renderCategoryPills();
      renderCategoryManager();
      document.getElementById('count-excluded').innerText = allExcluded.length;

      // Rows that were filtered without being scored carry a sort value, not a
      // score: they are left out of the average.
      const scored = allInsights.filter(i => !i.not_scored);
      if (scored.length > 0) {
        const avg = scored.reduce((acc, i) => acc + i.slop_score, 0) / scored.length;
        document.getElementById('stat-slop').innerText = `${avg.toFixed(1)}/100`;
      } else {
        document.getElementById('stat-slop').innerText = 'N/A';
      }

      // Populate creator filter dropdown
      populateCreatorDropdown();

      // Check initial URL hash
      handleHashRouting();

      renderCurrentTab();
      // The counts under the header come with the status. The first load has
      // its own status request (see the end of this script).
      if (!firstLoad) loadStatus(false);
      return true;
    }

    function populateCreatorDropdown() {
      const select = document.getElementById('insights-channel-select');
      const currentVal = select.value;
      const creators = Array.from(new Set(allInsights.map(i => i.channel_title).filter(Boolean))).sort();

      select.innerHTML = '<option value="all">All Tracked Creators</option>' +
        creators.map(c => {
          const count = allInsights.filter(i => i.channel_title === c).length;
          return `<option value="${esc(c)}">${esc(c)} (${count})</option>`;
        }).join('');

      if (currentVal) select.value = currentVal;
    }

    function switchTab(tab) {
      activeTab = tab;
      window.location.hash = tab;

      // Update stat pills active state
      document.querySelectorAll('.stat-pill').forEach(el => el.classList.remove('active'));
      const activePill = document.getElementById(`pill-${tab}`);
      if (activePill) activePill.classList.add('active');

      // Update nav tabs active state
      document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
      const activeTabBtn = document.getElementById(`tab-btn-${tab}`);
      if (activeTabBtn) activeTabBtn.classList.add('active');

      // Toggle views visibility
      document.getElementById('view-insights-cards').style.display = 'none';
      document.getElementById('view-insights-table').style.display = 'none';
      document.getElementById('view-excluded').style.display = 'none';
      document.getElementById('view-topics').style.display = 'none';
      document.getElementById('view-channels').style.display = 'none';

      document.getElementById('insights-controls-bar').style.display = (tab === 'insights') ? 'flex' : 'none';
      document.getElementById('topics-controls-bar').style.display = (tab === 'topics') ? 'flex' : 'none';
      document.getElementById('channels-controls-bar').style.display = (tab === 'channels') ? 'flex' : 'none';
      document.getElementById('library-filter-bar').style.display = (tab === 'insights') ? 'flex' : 'none';
      if (tab !== 'insights') document.getElementById('category-manager').style.display = 'none';
      document.getElementById('global-search-container').style.display = (tab === 'insights') ? 'flex' : 'none';

      if (tab === 'insights') {
        if (layoutMode === 'cards') {
          document.getElementById('view-insights-cards').style.display = 'grid';
        } else {
          document.getElementById('view-insights-table').style.display = 'block';
        }
      } else if (tab === 'topics') {
        document.getElementById('view-topics').style.display = 'grid';
      } else if (tab === 'channels') {
        document.getElementById('view-channels').style.display = 'block';
      }

      renderCurrentTab();
    }

    function setLayoutMode(mode) {
      layoutMode = mode;
      document.getElementById('btn-layout-cards').classList.toggle('active', mode === 'cards');
      document.getElementById('btn-layout-table').classList.toggle('active', mode === 'table');

      if (activeTab === 'insights') {
        document.getElementById('view-insights-cards').style.display = (mode === 'cards') ? 'grid' : 'none';
        document.getElementById('view-insights-table').style.display = (mode === 'table') ? 'block' : 'none';
      }
      renderCurrentTab();
    }

    function sortBySignalPurity() {
      switchTab('insights');
      document.getElementById('insights-sort-select').value = 'slop-asc';
      applySortingAndFiltering();
      showToast('🎯 Sorted by signal purity: Lowest slop first!');
    }

    // The list of disqualified videos, reached from the pill and from the
    // count under the header.
    function showExcludedView() {
      switchTab('insights');
      filterLibrary('excluded');
    }

    function markFilterPill() {
      document.querySelectorAll('.filter-pill[data-filter]').forEach(el => {
        const selected = el.dataset.filter === libraryFilter &&
          (libraryFilter !== 'category' || el.dataset.category === selectedCategory);
        el.classList.toggle('active', selected);
      });
    }

    function setLibraryFilter(kind, category) {
      libraryFilter = kind;
      selectedCategory = (kind === 'category') ? category : null;
      markFilterPill();
    }

    // 'all' or 'excluded'.
    function filterLibrary(kind) {
      setLibraryFilter(kind, null);
      applySortingAndFiltering();
    }

    function filterByCategory(name) {
      setLibraryFilter('category', name);
      applySortingAndFiltering();
    }

    // One pill per category the user defined, with the number of videos of
    // the library in it. Without categories there are no such pills.
    function renderCategoryPills() {
      const box = document.getElementById('category-pills');
      box.innerHTML = allCategories.map(c => {
        const count = allInsights.filter(i => i.category === c.name).length;
        return `<div class="filter-pill" data-filter="category" data-category="${esc(c.name)}" title="${esc(c.description || '')}" onclick="filterByCategory(${jsArg(c.name)})">🏷️ ${esc(c.name)} (${count})</div>`;
      }).join('');
      box.style.display = allCategories.length > 0 ? 'contents' : 'none';
      // A category that does not exist any more cannot stay selected.
      if (libraryFilter === 'category' && !allCategories.some(c => c.name === selectedCategory)) {
        setLibraryFilter('all', null);
      }
      markFilterPill();
    }

    function toggleCategoryManager() {
      const panel = document.getElementById('category-manager');
      panel.style.display = (panel.style.display === 'block') ? 'none' : 'block';
    }

    function renderCategoryManager() {
      const body = document.getElementById('categories-table-body');
      if (allCategories.length === 0) {
        body.innerHTML = '<tr><td colspan="4" style="color: var(--text-dim);">No categories defined. The library works without them.</td></tr>';
        return;
      }
      body.innerHTML = allCategories.map(c => c.name === editingCategory ? `
        <tr>
          <td><input type="text" id="edit-category-name" class="search-input" style="width: 100%; padding: 6px 10px;" maxlength="60" value="${esc(c.name)}" onkeydown="categoryEditKey(event, ${jsArg(c.name)})" /></td>
          <td><input type="text" id="edit-category-description" class="search-input" style="width: 100%; padding: 6px 10px;" maxlength="300" value="${esc(c.description || '')}" placeholder="What belongs in it (optional; the LLM reads this)" onkeydown="categoryEditKey(event, ${jsArg(c.name)})" /></td>
          <td>${esc(c.kept)}</td>
          <td style="text-align: right; white-space: nowrap;">
            <button class="btn btn-sm btn-primary" onclick="saveCategory(${jsArg(c.name)})">Save</button>
            <button class="btn btn-sm" onclick="cancelCategoryEdit()">Cancel</button>
          </td>
        </tr>
      ` : `
        <tr>
          <td><strong>${esc(c.name)}</strong></td>
          <td style="font-size: 0.86rem;">${esc(c.description || '')}</td>
          <td>${esc(c.kept)}</td>
          <td style="text-align: right; white-space: nowrap;">
            <button class="btn btn-sm" onclick="editCategory(${jsArg(c.name)})">Edit</button>
            <button class="btn btn-sm" onclick="removeCategory(${jsArg(c.name)})">Remove</button>
          </td>
        </tr>
      `).join('');
    }

    // POST with a JSON body; the answer is the server's JSON object.
    function postJson(url, body) {
      return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      }).then(r => r.json());
    }

    // The category whose row shows input fields instead of text, or null.
    let editingCategory = null;

    function editCategory(name) {
      editingCategory = name;
      renderCategoryManager();
      const field = document.getElementById('edit-category-name');
      if (field) { field.focus(); field.select(); }
    }

    function cancelCategoryEdit() {
      editingCategory = null;
      renderCategoryManager();
    }

    // Enter saves the row that is being edited, Escape leaves it as it was.
    function categoryEditKey(event, name) {
      if (event.key === 'Enter') { event.preventDefault(); saveCategory(name); }
      if (event.key === 'Escape') { event.preventDefault(); cancelCategoryEdit(); }
    }

    // Saves the edited row: the name first (its videos follow), then the
    // description. Only what was changed is sent.
    async function saveCategory(name) {
      const current = allCategories.find(c => c.name === name) || {};
      const newName = document.getElementById('edit-category-name').value.trim();
      const description = document.getElementById('edit-category-description').value.trim();
      if (!newName) { alert('A category needs a name.'); return; }
      const done = [];
      let stored = name;
      try {
        if (newName !== name) {
          const res = await postJson('/api/categories/rename', { name, new_name: newName });
          if (res.status !== 'success') { alert('Error: ' + res.message); return; }
          stored = res.name;
          // The filter follows the category it was showing.
          if (libraryFilter === 'category' && selectedCategory === res.old_name) selectedCategory = res.name;
          done.push(`renamed: ${res.old_name} → ${res.name} (${res.moved} video${res.moved === 1 ? '' : 's'}). Sync Wiki brings the wiki files in line.`);
        }
        if (description !== (current.description || '')) {
          const res = await postJson('/api/categories/describe', { name: stored, description });
          if (res.status !== 'success') { alert('Error: ' + res.message); await loadData(); return; }
          done.push(res.description ? `description of ${res.name} saved` : `${res.name} has no description now`);
        }
      } catch (e) {
        alert('Failed to save the category: ' + e);
        return;
      }
      editingCategory = null;
      if (done.length) {
        showToast('✓ Category ' + done.join('; '), 7000);
        await loadData();
      } else {
        renderCategoryManager();
      }
    }

    // The category of one video, chosen in its dropdown: 'auto' hands the
    // choice back to the model, 'none' and 'c:<name>' set it by hand.
    async function setVideoCategory(videoId, select) {
      const value = select.value;
      const body = value === 'auto'
        ? { auto: true }
        : { category: value === 'none' ? null : value.slice(2) };
      select.disabled = true;
      try {
        const res = await postJson(`/api/insights/${encodeURIComponent(videoId)}/category`, body);
        if (res.status === 'success') {
          const where = res.category ? `in ${res.category}` : 'in no category';
          showToast(res.category_manual
            ? `✓ "${res.title}" is ${where} (set by hand). Sync Wiki brings the wiki files in line.`
            : `✓ The model decides the category of "${res.title}" again; until its next analysis it stays ${where}.`, 6000);
        } else {
          alert('The category was not changed.\n\n' + (res.message || 'unknown error'));
        }
      } catch (e) {
        alert('The category was not changed: the server did not answer (' + e + ').');
      }
      // Drawn again from the server either way, so the card shows what is stored.
      await loadData();
    }

    let categorizePollInterval = null;

    function showCategorizeProgress(p) {
      const status = document.getElementById('categorize-status');
      const btn = document.getElementById('btn-categorize');
      if (!status || !btn) return;
      btn.disabled = p.is_running === true;
      if (p.is_running) {
        const total = Number(p.total_videos) || 0;
        status.innerText = total > 0
          ? `⏳ ${p.current_index || 0}/${total}: ${p.assigned || 0} assigned, ${p.none || 0} in no category, ${p.failed || 0} failed` +
            (p.current_video_title ? ` • ${p.current_video_title}` : '')
          : (p.message || 'Starting...');
      } else if (p.stage === 'complete') {
        status.innerText = '✓ ' + (p.message || 'Done.');
      } else if (p.stage === 'error') {
        status.innerText = '⚠️ ' + (p.message || 'The category assignment failed.');
      }
    }

    function stopCategorizePolling() {
      if (categorizePollInterval) clearInterval(categorizePollInterval);
      categorizePollInterval = null;
    }

    // Follows GET /api/categorize/progress until the run is over.
    function startCategorizePolling() {
      stopCategorizePolling();
      const interval = setInterval(async () => {
        try {
          const p = await fetchJson('/api/categorize/progress');
          // An answer that arrives after the polling was stopped is dropped.
          if (categorizePollInterval !== interval) return;
          if (p.is_running) {
            showCategorizeProgress(p);
          } else {
            // Over, or never started (the request was refused).
            stopCategorizePolling();
            showCategorizeProgress(p);
            if (p.stage === 'complete' || p.stage === 'error') loadData();
          }
        } catch (e) {
          console.warn('Categorize progress poll error:', e);
        }
      }, 600);
      categorizePollInterval = interval;
    }

    async function runCategorize() {
      if (allCategories.length === 0) {
        alert('Define at least one category first.');
        return;
      }
      if (!hasLlm()) {
        alert('Assigning categories needs an LLM, and none is configured on the server.');
        return;
      }
      if (!confirm('Assign categories to existing videos?\n\nThe LLM is asked once per analysed video that has no category yet, from its stored summary. No transcript is read again and no score changes. Videos whose category you set by hand are left alone.')) return;
      showCategorizeProgress({ is_running: true, message: 'Starting...' });
      startCategorizePolling();
      try {
        // The request returns when the run is over, whatever the outcome.
        const res = await postJson('/api/categorize', {});
        stopCategorizePolling();
        if (res.status === 'success') {
          const text = res.total === 0
            ? 'No analysed video is without a category.'
            : `${res.stopped ? res.stopped + ': ' : 'Categories assigned: '}${res.counts_line}.`;
          showCategorizeProgress({ is_running: false, stage: res.stopped ? 'error' : 'complete', message: text });
          showToast((res.stopped ? '⚠️ ' : '✓ ') + text + (res.assigned > 0 ? ' Sync Wiki brings the wiki files in line.' : ''), 9000);
          await loadData();
        } else {
          showCategorizeProgress({ is_running: false, stage: 'error', message: res.message || 'unknown error' });
          // Another job holds the slot: this run did not start, a run that
          // is already under way keeps its progress line.
          if (res.busy) releaseCategorizeStatusIfIdle();
          alert('Categories were not assigned.\n\n' + (res.message || 'unknown error'));
        }
      } catch (e) {
        stopCategorizePolling();
        showCategorizeProgress({ is_running: false, stage: 'error', message: 'The server did not answer.' });
        alert('Assigning categories failed: the server did not answer (' + e + ').');
      }
      loadStatus(false);
    }

    // After a refused request: shows what the server says about a run that
    // may already be under way, and follows it when there is one.
    async function releaseCategorizeStatusIfIdle() {
      try {
        const p = await fetchJson('/api/categorize/progress');
        if (p.is_running) {
          showCategorizeProgress(p);
          startCategorizePolling();
        }
      } catch (_) {}
    }

    async function submitAddCategory() {
      const nameInput = document.getElementById('add-category-name');
      const descriptionInput = document.getElementById('add-category-description');
      const name = nameInput.value.trim();
      if (!name) return;
      try {
        const res = await fetch('/api/categories/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, description: descriptionInput.value.trim() })
        }).then(r => r.json());
        if (res.status === 'success') {
          showToast(`✓ Category added: ${res.name}`);
          nameInput.value = '';
          descriptionInput.value = '';
          await loadData();
        } else {
          alert('Error: ' + res.message);
        }
      } catch (e) {
        alert('Failed to add the category: ' + e);
      }
    }

    async function removeCategory(name) {
      if (!confirm(`Remove the category "${name}"?\n\nIts videos stay in the library; they are in no category afterwards.`)) return;
      try {
        const res = await fetch('/api/categories/remove', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name })
        }).then(r => r.json());
        if (res.status === 'success') {
          showToast(`Category removed: ${res.name} (${res.cleared} video${res.cleared === 1 ? '' : 's'} now without a category)`);
          await loadData();
        } else {
          alert('Error: ' + res.message);
        }
      } catch (e) {
        alert('Failed to remove the category: ' + e);
      }
    }

    function filterByCreator(creatorName) {
      activeCreatorFilter = creatorName;
      activeTopicFilter = null;
      switchTab('insights');
      document.getElementById('insights-channel-select').value = creatorName;
      applySortingAndFiltering();
    }

    // The tracked channel a mentioned name refers to, compared without case.
    function trackedChannelByName(name) {
      const wanted = String(name ?? '').trim().toLowerCase();
      if (!wanted) return null;
      return allChannels.find(c => String(c.title ?? '').trim().toLowerCase() === wanted) || null;
    }

    // For a mentioned name that is not a tracked channel: look the name up in
    // the text of the whole library instead of filtering on a creator that
    // does not exist.
    function searchForName(name) {
      activeCreatorFilter = null;
      activeTopicFilter = null;
      setLibraryFilter('all', null);
      document.getElementById('insights-channel-select').value = 'all';
      document.getElementById('search-box').value = name;
      switchTab('insights');
      showToast(`Searching the library for "${name}"`);
    }

    function mentionHtml(name) {
      const ch = trackedChannelByName(name);
      if (ch) {
        return `<span class="tag tag-mention" title="Tracked creator: show their dossiers" onclick="filterByCreator(${jsArg(ch.title)})">@${esc(name)}</span>`;
      }
      return `<span class="mention-plain" title="Not a tracked channel">${esc(name)}<button class="mention-search" title="Search the library for this name" onclick="searchForName(${jsArg(name)})">🔍 search</button></span>`;
    }

    function onChannelDropdownChange() {
      const val = document.getElementById('insights-channel-select').value;
      activeCreatorFilter = (val === 'all') ? null : val;
      applySortingAndFiltering();
    }

    function filterByTopic(topicName) {
      activeTopicFilter = topicName;
      switchTab('insights');
      applySortingAndFiltering();
    }

    function resetAllFilters() {
      activeCreatorFilter = null;
      activeTopicFilter = null;
      setLibraryFilter('all', null);
      document.getElementById('search-box').value = '';
      document.getElementById('insights-channel-select').value = 'all';
      document.getElementById('insights-signal-select').value = 'all';
      document.getElementById('insights-engine-select').value = 'all';
      document.getElementById('active-filter-banner').style.display = 'none';

      applySortingAndFiltering();
      showToast('Filters reset to default library');
    }

    function applySortingAndFiltering() {
      if (activeTab !== 'insights') return;

      // The Disqualified pill replaces the library by the list of removed videos.
      const showExcluded = libraryFilter === 'excluded';
      document.getElementById('view-excluded').style.display = showExcluded ? 'block' : 'none';
      document.getElementById('view-insights-cards').style.display = (!showExcluded && layoutMode === 'cards') ? 'grid' : 'none';
      document.getElementById('view-insights-table').style.display = (!showExcluded && layoutMode === 'table') ? 'block' : 'none';
      if (showExcluded) {
        updateActiveFilterBanner(allExcluded.length);
        renderExcluded();
        return;
      }

      let list = [...allInsights];

      // 1. Category Filter
      if (libraryFilter === 'category') {
        list = list.filter(i => i.category === selectedCategory);
      }

      // 2. Creator Filter
      if (activeCreatorFilter) {
        list = list.filter(i => i.channel_title === activeCreatorFilter);
      }

      // 3. Topic Filter
      if (activeTopicFilter) {
        list = list.filter(i => (i.topics || []).some(t => t.toLowerCase() === activeTopicFilter.toLowerCase()));
      }

      // 4. Signal Tier Filter
      // A row that was never scored belongs to no tier.
      const signalTier = document.getElementById('insights-signal-select')?.value || 'all';
      if (signalTier === 'pure') {
        list = list.filter(i => !i.not_scored && i.slop_score <= 20);
      } else if (signalTier === 'high') {
        list = list.filter(i => !i.not_scored && i.slop_score <= 35);
      } else if (signalTier === 'moderate') {
        // No upper bound: only kept videos are listed, and a kept video above 60
        // must stay reachable through one of the tiers.
        list = list.filter(i => !i.not_scored && i.slop_score > 35);
      }

      // 4b. Engine filter: reviewed by an LLM, or heuristic output nobody reviewed
      const engine = document.getElementById('insights-engine-select')?.value || 'all';
      if (engine === 'heuristic') {
        list = list.filter(i => i.unreviewed);
      } else if (engine === 'llm') {
        list = list.filter(i => !i.unreviewed && !i.not_scored);
      }

      // 5. Search Text Query
      const query = document.getElementById('search-box')?.value.trim().toLowerCase();
      if (query) {
        const has = v => String(v ?? '').toLowerCase().includes(query);
        const anyHas = arr => Array.isArray(arr) && arr.some(has);
        list = list.filter(i =>
          has(i.title) || has(i.summary) || has(i.channel_title) ||
          has(i.category) ||
          anyHas(i.topics) || anyHas(i.key_insights) ||
          anyHas(i.takeaways) || anyHas(i.quotes)
        );
      }

      // 6. Sorting
      const sortKey = document.getElementById('insights-sort-select')?.value || 'date-desc';
      list.sort((a, b) => {
        if (sortKey === 'date-desc') return new Date(b.published_at || 0) - new Date(a.published_at || 0);
        if (sortKey === 'date-asc') return new Date(a.published_at || 0) - new Date(b.published_at || 0);
        if (sortKey === 'title-asc') return (a.title || '').localeCompare(b.title || '');
        if (sortKey === 'title-desc') return (b.title || '').localeCompare(a.title || '');
        if (sortKey === 'channel-asc') return (a.channel_title || '').localeCompare(b.channel_title || '');
        if (sortKey === 'slop-asc') return a.slop_score - b.slop_score;
        if (sortKey === 'slop-desc') return b.slop_score - a.slop_score;
        // Videos without a category come last.
        if (sortKey === 'category-asc') return (a.category ? 0 : 1) - (b.category ? 0 : 1) || (a.category || '').localeCompare(b.category || '');
        return 0;
      });

      // Update Filter Banner UI
      updateActiveFilterBanner(list.length);

      // Render into active layout
      if (layoutMode === 'cards') {
        renderInsightsCards(list);
      } else {
        renderInsightsTable(list);
      }
    }

    function updateActiveFilterBanner(count) {
      const banner = document.getElementById('active-filter-banner');
      const label = document.getElementById('active-filter-label');
      const countEl = document.getElementById('active-filter-count');

      const parts = [];
      if (activeCreatorFilter) parts.push(`Creator: "${activeCreatorFilter}"`);
      if (activeTopicFilter) parts.push(`Topic: #${activeTopicFilter}`);
      if (libraryFilter === 'excluded') {
        // The other filters do not apply to this list.
        parts.length = 0;
        parts.push('Disqualified videos');
      } else if (libraryFilter === 'category') {
        parts.push(`Category: ${selectedCategory}`);
      }
      const engineFilter = document.getElementById('insights-engine-select')?.value || 'all';
      if (libraryFilter !== 'excluded') {
        if (engineFilter === 'heuristic') parts.push('Engine: unreviewed (heuristic)');
        if (engineFilter === 'llm') parts.push('Engine: reviewed by an LLM');
      }

      if (parts.length > 0) {
        banner.style.display = 'flex';
        label.innerText = parts.join(' • ');
        countEl.innerText = libraryFilter === 'excluded'
          ? `(${count} video${count === 1 ? '' : 's'})`
          : `(${count} matching dossier${count === 1 ? '' : 's'})`;
      } else {
        banner.style.display = 'none';
      }
    }

    function renderCurrentTab() {
      if (activeTab === 'insights') {
        applySortingAndFiltering();
      } else if (activeTab === 'topics') {
        // Through the filter inputs, so a reload keeps what the user typed.
        onTopicSearch(document.getElementById('topics-filter-input').value);
      } else if (activeTab === 'channels') {
        onChannelSearch(document.getElementById('channels-filter-input').value);
      }
    }

    // A fresh database has nothing to filter: say what to do next instead.
    function emptyInsightsMessage() {
      if (allInsights.length === 0) {
        if (allChannels.length === 0) {
          return 'No insights yet. Open the <a href="javascript:void(0)" onclick="switchTab(\'channels\')" style="color: var(--accent);">Creators tab</a>, add a channel, then press Crawl.';
        }
        return 'No insights yet. Press <strong>⚡ Crawl All</strong> at the top to evaluate the newest videos of your tracked channels.';
      }
      return 'No insights match the current filter or query. <a href="javascript:void(0)" onclick="resetAllFilters()" style="color: var(--accent);">Reset filters</a>.';
    }

    // One place for the category badge, so cards and table agree. A video
    // without a category has no badge.
    function categoryBadgeHtml(category) {
      if (!category) return '';
      return `<span class="badge badge-category" title="Show the videos of this category" onclick="filterByCategory(${jsArg(category)})">🏷️ ${esc(category)}</span>`;
    }

    // Marker next to a category (or next to nothing, for "no category") that
    // the owner set by hand.
    function categoryByHandHtml(i) {
      if (!i.category_manual) return '';
      const what = i.category ? 'This category was set by hand' : 'This video was put into no category by hand';
      return `<span class="category-by-hand" title="${what}. No analysis changes it; choose Automatic to hand it back to the model.">${i.category ? 'set by hand' : 'no category, set by hand'}</span>`;
    }

    // Dropdown that sets the category of one video: the user's categories,
    // none, or the model's choice. Not shown while the user has no categories
    // and nothing was set by hand.
    function categorySelectHtml(i) {
      if (allCategories.length === 0 && !i.category_manual) return '';
      const manual = i.category_manual === true;
      const option = (value, label, selected) => `<option value="${esc(value)}"${selected ? ' selected' : ''}>${esc(label)}</option>`;
      const options = [
        option('auto', 'Automatic (model decides)', !manual),
        option('none', 'No category', manual && !i.category),
        ...allCategories.map(c => option('c:' + c.name, c.name, manual && i.category === c.name))
      ].join('');
      return `<select class="category-select" title="Category of this video. Automatic: the model chooses when it analyses the video. Anything else is set by hand and stays." onchange="setVideoCategory(${jsArg(i.video_id)}, this)">${options}</select>`;
    }

    const TRANSCRIPT_SOURCE_LABELS = {
      builtin: 'YouTube captions',
      subtitles: 'YouTube captions (subtitle track)',
      whisper: 'Whisper (local transcription of the audio)'
    };

    // What the model read and what the transcript is, as plain text parts.
    // Takes the fields that /api/insights and /api/transcripts both send.
    // Nothing is said about a fact the server did not record.
    function readLineParts(i) {
      const parts = [];
      if (i.read_note) parts.push(`Read: ${i.read_note}`);
      const source = TRANSCRIPT_SOURCE_LABELS[i.transcript_source];
      if (source) parts.push(`Transcript: ${source}`);
      if (i.transcript_complete === true) {
        parts.push('covers the whole video');
      } else if (i.transcript_complete === false) {
        parts.push(i.video_coverage_pct == null
          ? 'does not reach the end of the video'
          : `reaches only ${i.video_coverage_pct}% of the video`);
      }
      return { parts, warn: i.transcript_complete === false };
    }

    function readLineHtml(i) {
      const { parts, warn } = readLineParts(i);
      if (parts.length === 0) return '';
      return `<div class="read-note${warn ? ' read-note-warn' : ''}">${warn ? '⚠️' : '📖'} ${parts.map(esc).join(' • ')}</div>`;
    }

    function removeButtonHtml(i, short) {
      return `<button class="btn btn-sm btn-remove" title="Take this video out of the library. It stays out, also after a re-analysis, until you restore it. Its wiki page file is not touched: the Disqualified list offers to move it out." onclick="removeFromWiki(${jsArg(i.video_id)}, ${jsArg(i.title)}, this)">🚫 ${short ? 'Remove' : 'Remove from wiki'}</button>`;
    }

    // Score badge. A row without a score shows why it has none instead of
    // the placeholder stored in its score column.
    function scoreBadgeHtml(i, short) {
      if (i.not_scored) {
        const label = i.score_label || 'Not scored';
        return `<span class="badge badge-not-scored" title="${esc(label)}. This video was not scored.">${esc(short ? 'Not scored' : label)}</span>`;
      }
      return `<span class="badge ${slopBadgeClass(i.slop_score)}" title="Slop Score (0=pure signal, 100=pure slop)">${esc(i.slop_score)}${short ? '' : '/100'}</span>`;
    }

    // Heuristic rows are word counts and transcript excerpts, not a review.
    function unreviewedBadgeHtml(i) {
      if (!i.unreviewed) return '';
      return `<span class="badge badge-unreviewed" title="Scored by counting words and phrases. No LLM has read this video; the text below is transcript excerpts.">Unreviewed (heuristic)</span>`;
    }

    function reanalyseButtonHtml(i) {
      if (!i.unreviewed) return '';
      if (!hasLlm()) {
        return `<button class="btn btn-sm" disabled title="Re-analysis needs an LLM, and none is configured on the server.">🧠 Re-analyse</button>`;
      }
      return `<button class="btn btn-sm btn-warn" title="Run ${esc(engineName())} over the stored transcript and replace this entry" onclick="reanalyzeVideo(${jsArg(i.video_id)}, this)">🧠 Re-analyse</button>`;
    }

    function renderInsightsCards(list) {
      const container = document.getElementById('view-insights-cards');
      if (list.length === 0) {
        container.innerHTML = `<p style="color: #8b949e; grid-column: 1/-1; padding: 40px; text-align: center;">${emptyInsightsMessage()}</p>`;
        return;
      }

      container.innerHTML = list.map(i => {
        const tags = (i.topics || []).map(t => `<span class="tag" onclick="filterByTopic(${jsArg(t)})">#${esc(t)}</span>`).join('');
        const findings = (i.key_insights || []).slice(0, 3).map(k => `<li>${esc(k)}</li>`).join('');
        const coverageHtml = readLineHtml(i);
        const quoteHtml = (i.quotes && i.quotes.length > 0) ? `<div class="quote-box">"${esc(i.quotes[0])}"</div>` : '';

        const categoryBadge = categoryBadgeHtml(i.category);

        // Mentioned creators
        let mentionsHtml = '';
        if (i.mentioned_creators && i.mentioned_creators.length > 0) {
          mentionsHtml = `
            <div style="font-size: 0.76rem; color: var(--text-dim); display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
              <span>Mentions:</span>
              ${i.mentioned_creators.map(mentionHtml).join('')}
            </div>
          `;
        }

        return `
          <div class="card">
            <div class="card-header">
              <a href="${safeUrl(i.url)}" target="_blank" rel="noopener" class="card-title">${esc(i.title)}</a>
              ${scoreBadgeHtml(i, false)}
            </div>
            <div class="card-meta">
              ${categoryBadge}
              ${categoryByHandHtml(i)}
              ${categorySelectHtml(i)}
              ${unreviewedBadgeHtml(i)}
              <span class="wiki-creator-link" onclick="filterByCreator(${jsArg(i.channel_title)})">🎙️ ${esc(i.channel_title || 'Unknown')}</span>
              <span>📅 ${esc(new Date(i.published_at).toLocaleDateString())}</span>
            </div>
            <div class="card-summary">${esc(i.summary)}</div>
            ${coverageHtml}
            ${quoteHtml}
            <div class="section-title">${i.unreviewed ? 'Transcript Excerpts (no LLM)' : 'Key Findings'}</div>
            <ul class="insights-list">${findings}</ul>
            ${mentionsHtml}
            <div class="tags">${tags}</div>
            <div class="card-actions">
              <a href="${safeUrl(i.url)}" target="_blank" rel="noopener" style="font-size: 0.78rem; color: var(--accent); text-decoration: none;">▶ YouTube</a>
              ${reanalyseButtonHtml(i)}
              <button class="btn btn-sm" onclick="openTranscript(${jsArg(i.video_id)}, ${jsArg(i.title)})">📄 Read Cleaned Transcript</button>
              ${removeButtonHtml(i, false)}
            </div>
          </div>
        `;
      }).join('');
    }

    function renderInsightsTable(list) {
      const tbody = document.getElementById('insights-table-body');
      if (list.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #8b949e; padding: 30px;">${emptyInsightsMessage()}</td></tr>`;
        return;
      }

      tbody.innerHTML = list.map(i => {
        const tags = (i.topics || []).slice(0, 4).map(t => `<span class="tag" style="margin-right: 4px;" onclick="filterByTopic(${jsArg(t)})">#${esc(t)}</span>`).join('');

        return `
          <tr>
            <td style="white-space: nowrap; font-size: 0.82rem; color: var(--text-dim);">${esc(new Date(i.published_at).toLocaleDateString())}</td>
            <td>${scoreBadgeHtml(i, true)}</td>
            <td>
              <a href="${safeUrl(i.url)}" target="_blank" rel="noopener" style="color: var(--text-bright); text-decoration: none; font-weight: 600; line-height: 1.4; display: block;">
                ${esc(i.title)}
              </a>
              <div style="font-size: 0.8rem; color: var(--text-dim); margin-top: 4px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
                ${esc(i.summary)}
              </div>
              ${unreviewedBadgeHtml(i)}
              ${readLineHtml(i)}
            </td>
            <td>
              <span class="wiki-creator-link" onclick="filterByCreator(${jsArg(i.channel_title)})">🎙️ ${esc(i.channel_title || 'Unknown')}</span>
            </td>
            <td style="font-size: 0.8rem;">${categoryBadgeHtml(i.category)} ${categoryByHandHtml(i)}<div style="margin-top: 4px;">${categorySelectHtml(i)}</div></td>
            <td><div style="display: flex; flex-wrap: wrap; gap: 4px;">${tags}</div></td>
            <td style="text-align: right; white-space: nowrap;">
              ${reanalyseButtonHtml(i)}
              <button class="btn btn-sm" onclick="openTranscript(${jsArg(i.video_id)}, ${jsArg(i.title)})">📄 Transcript</button>
              ${removeButtonHtml(i, true)}
            </td>
          </tr>
        `;
      }).join('');
    }

    function dateText(value) {
      const d = new Date(value || '');
      return isNaN(d) ? '-' : d.toLocaleDateString();
    }

    function renderExcluded() {
      const tbody = document.getElementById('excluded-table-body');
      if (allExcluded.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #8b949e; padding: 30px;">No video is disqualified. The button <strong>🚫 Remove from wiki</strong> on an insight card takes a video out of the wiki.</td></tr>';
        return;
      }
      tbody.innerHTML = allExcluded.map(x => `
        <tr>
          <td>
            <a href="https://www.youtube.com/watch?v=${encodeURIComponent(x.video_id)}" target="_blank" rel="noopener" style="color: var(--text-bright); text-decoration: none; font-weight: 600;">
              ${esc(x.title || x.video_id)}
            </a>
          </td>
          <td>${esc(x.channel_title || x.channel_id || 'Unknown')}</td>
          <td>${x.reason ? esc(x.reason) : '<span style="color: var(--text-dim);">(no reason given)</span>'}</td>
          <td style="white-space: nowrap; font-size: 0.82rem; color: var(--text-dim);">${esc(dateText(x.excluded_at))}</td>
          <td style="text-align: right;">
            <button class="btn btn-sm btn-primary" title="Undo the removal: the analysis decides again whether the video is kept" onclick="restoreVideo(${jsArg(x.video_id)}, ${jsArg(x.title)}, this)">↩ Restore</button>
          </td>
        </tr>
      `).join('');
    }

    // Disqualifies a video by the owner's decision. Only the database
    // changes. The next Sync Wiki takes the video out of the index; its own
    // page file stays until syncWikiAndMove() moves it out.
    async function removeFromWiki(videoId, title, btn) {
      const name = title || videoId;
      if (!confirm(`Remove "${name}" from the wiki?\n\nThe video leaves the library and stays out, also after a re-analysis. You can restore it at any time in the Disqualified list.\n\nNo wiki file is changed or deleted by this.`)) return;
      const typed = prompt('Optional: why do you remove it? Leave empty for no reason. Cancel keeps the video.', '');
      if (typed === null) return;
      const reason = typed.trim();
      if (btn) btn.disabled = true;
      try {
        const res = await fetch(`/api/insights/${encodeURIComponent(videoId)}/exclude`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(reason ? { reason } : {})
        }).then(r => r.json());
        if (res.status !== 'success') {
          alert('The video was not removed: ' + (res.message || 'unknown error'));
          if (btn) btn.disabled = false;
          return;
        }
        // The card disappears at once, also when the reload below fails.
        allInsights = allInsights.filter(i => i.video_id !== videoId);
        renderCurrentTab();
        showToast(`Removed "${name}" from the library. The wiki files are unchanged: Sync Wiki takes it out of the index, and its page file stays in the wiki folder until you use 'Sync wiki and move removed pages' in the Disqualified list.`, 15000, {
          label: 'Undo',
          run: () => restoreVideo(videoId, name, null)
        });
        await loadData();
      } catch (e) {
        alert('The video was not removed: the server did not answer (' + e + ').');
        if (btn) btn.disabled = false;
      }
    }

    // Undoes a disqualification. The stored analysis decides again, so a
    // restored video can still be out of the library; the toast says why.
    async function restoreVideo(videoId, title, btn) {
      const name = title || videoId;
      if (btn) btn.disabled = true;
      try {
        const res = await fetch(`/api/insights/${encodeURIComponent(videoId)}/include`, { method: 'POST' }).then(r => r.json());
        if (res.status !== 'success') {
          alert('The video was not restored: ' + (res.message || 'unknown error'));
          if (btn) btn.disabled = false;
          return;
        }
        const outcome = res.kept
          ? 'It is back in the library.'
          : `It is not disqualified any more, but the analysis keeps it out: ${res.rejection || 'not kept'}.`;
        showToast(`Restored "${name}". ${outcome} Run Sync Wiki to update the wiki files.`, 9000);
        await loadData();
      } catch (e) {
        alert('The video was not restored: the server did not answer (' + e + ').');
        if (btn) btn.disabled = false;
      }
    }

    function toggleTableSort(col) {
      const select = document.getElementById('insights-sort-select');
      if (col === 'date') {
        select.value = (select.value === 'date-desc') ? 'date-asc' : 'date-desc';
      } else if (col === 'title') {
        select.value = (select.value === 'title-asc') ? 'title-desc' : 'title-asc';
      } else if (col === 'slop') {
        select.value = (select.value === 'slop-asc') ? 'slop-desc' : 'slop-asc';
      } else if (col === 'channel') {
        select.value = 'channel-asc';
      } else if (col === 'category') {
        select.value = 'category-asc';
      }
      applySortingAndFiltering();
    }

    function renderTopics(topics) {
      const container = document.getElementById('view-topics');
      if (topics.length === 0) {
        const msg = allTopics.length === 0
          ? 'No topics yet. They appear after the first crawl: add a channel in the Creators tab, then press Crawl.'
          : 'No topics match this filter.';
        container.innerHTML = `<p style="color: #8b949e; grid-column: 1/-1;">${msg}</p>`;
        return;
      }

      // Sort topics
      const sortKey = document.getElementById('topics-sort-select')?.value || 'count-desc';
      const sorted = [...topics].sort((a, b) => {
        if (sortKey === 'count-desc') return b.count - a.count;
        if (sortKey === 'count-asc') return a.count - b.count;
        if (sortKey === 'name-asc') return (a.name || '').localeCompare(b.name || '');
        if (sortKey === 'name-desc') return (b.name || '').localeCompare(a.name || '');
        return 0;
      });

      container.innerHTML = sorted.map(t => `
        <div class="topic-card" onclick="filterByTopic(${jsArg(t.name)})">
          <div>
            <strong style="color: var(--text-bright); display: block; font-size: 0.98rem;">#${esc(t.name)}</strong>
            <span style="font-size: 0.78rem; color: var(--accent); margin-top: 2px; display: inline-block;">Explore dossiers &rarr;</span>
          </div>
          <span class="badge badge-green">${esc(t.count)} video${t.count === 1 ? '' : 's'}</span>
        </div>
      `).join('');
    }

    function onTopicSearch(query) {
      const q = (query || '').trim().toLowerCase();
      const filtered = allTopics.filter(t => (t.name || '').toLowerCase().includes(q));
      renderTopics(filtered);
    }

    function renderChannels(channels) {
      const tbody = document.getElementById('channels-table-body');

      if (channels.length === 0) {
        const msg = allChannels.length === 0
          ? 'No channels tracked yet. Add a channel above, then press Crawl.'
          : 'No creators match this filter.';
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #8b949e; padding: 30px;">${msg}</td></tr>`;
        return;
      }

      // Sort channels
      const sortKey = document.getElementById('channels-sort-select')?.value || 'health-asc';
      const sorted = [...channels].sort((a, b) => {
        // 0 is the best possible score, not a missing one: only null sorts last.
        if (sortKey === 'health-asc') return (a.avg_slop_score ?? 999) - (b.avg_slop_score ?? 999);
        if (sortKey === 'kept-desc') return b.kept_count - a.kept_count;
        if (sortKey === 'title-asc') return (a.title || '').localeCompare(b.title || '');
        if (sortKey === 'title-desc') return (b.title || '').localeCompare(a.title || '');
        if (sortKey === 'total-desc') return b.total_evaluated - a.total_evaluated;
        return 0;
      });

      tbody.innerHTML = sorted.map(c => {
        let badgeClass = 'badge-green';
        let label = '🟢 High Signal';
        if (c.status === 'muted') {
          badgeClass = 'badge-red';
          label = '🔴 Muted';
        } else if (c.status === 'active') {
          badgeClass = 'badge-yellow';
          label = '🔵 Active';
        }

        const slop = c.avg_slop_score == null ? 'Pending' : `${Number(c.avg_slop_score).toFixed(0)}/100`;
        const toggleBtnLabel = c.status === 'muted' ? 'Unmute' : 'Mute';
        const toggleClass = c.status === 'muted' ? 'btn-primary' : 'btn-warn';
        const name = c.title || c.id;
        // Why videos of this channel are not in the library, one group each.
        const notKept = [
          [c.below_cutoff_count, 'below quality cutoff'],
          [c.off_topic_count, 'off-topic'],
          [c.insufficient_count, 'not enough information'],
          [c.excluded_count, 'disqualified']
        ].filter(([n]) => Number(n) > 0).map(([n, label]) => `${esc(n)} ${label}`).join(' • ');

        return `
          <tr>
            <td>
              <span class="wiki-creator-link" style="font-size: 1rem;" onclick="filterByCreator(${jsArg(c.title)})">
                ${esc(name)}
              </span>
              <a href="https://www.youtube.com/channel/${encodeURIComponent(c.id)}" target="_blank" rel="noopener" style="margin-left: 6px; font-size: 0.78rem; color: var(--text-dim); text-decoration: none;">🔗 YouTube</a>
            </td>
            <td><span class="badge ${badgeClass}">${label}</span></td>
            <td><strong>${slop}</strong></td>
            <td>
              <span style="cursor: pointer; color: var(--accent); font-weight: 600;" onclick="filterByCreator(${jsArg(c.title)})">
                ${esc(c.kept_count)} kept
              </span> / ${esc(c.total_evaluated)}
              ${notKept ? `<div style="font-size: 0.76rem; color: var(--text-dim);" title="Off-topic means the subject is fringe, nonsense or politics. Videos with not enough information and disqualified videos do not count in the rolling slop.">${notKept}</div>` : ''}
            </td>
            <td style="color: #8b949e; font-family: monospace; font-size: 0.85rem;">${esc(c.id)}</td>
            <td>
              <div style="display: flex; gap: 6px;">
                <button class="btn btn-sm btn-primary" onclick="filterByCreator(${jsArg(c.title)})">📚 Dossiers</button>
                <button class="btn btn-sm btn-accent" onclick="crawlChannel(${jsArg(c.id)})" title="Crawl this creator with current depth">⚡ Crawl</button>
                <button class="btn btn-sm ${toggleClass}" onclick="toggleMute(${jsArg(c.id)})">${toggleBtnLabel}</button>
                <button class="btn btn-sm" onclick="findRelated(${jsArg(c.id)}, ${jsArg(name)})">🕸️ Related</button>
                <button class="btn btn-sm btn-danger" title="Stop tracking this channel" onclick="deleteChannel(${jsArg(c.id)}, ${jsArg(name)})">&times;</button>
              </div>
            </td>
          </tr>
        `;
      }).join('');
    }

    function onChannelSearch(query) {
      const q = (query || '').trim().toLowerCase();
      const filtered = allChannels.filter(c => (c.title || '').toLowerCase().includes(q) || (c.id || '').toLowerCase().includes(q));
      renderChannels(filtered);
    }

    function onSearch(query) {
      applySortingAndFiltering();
    }

    function handleHashRouting() {
      const hash = window.location.hash.replace('#', '');
      if (!hash) return;

      if (hash.startsWith('creator:')) {
        const creator = decodeURIComponent(hash.replace('creator:', ''));
        filterByCreator(creator);
      } else if (hash.startsWith('topic:')) {
        const topic = decodeURIComponent(hash.replace('topic:', ''));
        filterByTopic(topic);
      } else if (['insights', 'topics', 'channels'].includes(hash)) {
        switchTab(hash);
      }
    }

    async function openTranscript(videoId, title) {
      document.getElementById('modal-video-title').innerText = title || `Video: ${videoId}`;
      document.getElementById('modal-transcript-text').innerText = 'Loading cleaned transcript...';
      // Otherwise a failed load would keep showing the numbers of another transcript.
      document.getElementById('modal-stripped-secs').innerText = '-';
      document.getElementById('modal-stripped-segs').innerText = '-';
      document.getElementById('modal-char-count').innerText = '-';
      const coverage = document.getElementById('modal-coverage');
      coverage.style.display = 'none';
      coverage.innerText = '';
      document.getElementById('transcript-modal').style.display = 'flex';

      try {
        const res = await fetch(`/api/transcripts/${encodeURIComponent(videoId)}`).then(r => r.json());
        if (res.error) {
          document.getElementById('modal-transcript-text').innerText = 'Error: ' + res.error;
          return;
        }
        document.getElementById('modal-stripped-secs').innerText = (Number(res.stripped_ad_seconds) || 0).toFixed(1);
        document.getElementById('modal-stripped-segs').innerText = res.stripped_segments_count || 0;
        document.getElementById('modal-char-count').innerText = (res.cleaned_char_count || 0).toLocaleString();
        // What the analysis read and where the transcript came from, as far
        // as the server recorded it. Yellow only when something is missing.
        const { parts, warn } = readLineParts(res);
        if (parts.length > 0) {
          coverage.innerText = `${warn ? '⚠️' : '📖'} ${parts.join(' • ')}`;
          coverage.style.color = warn ? '' : 'var(--text-dim)';
          coverage.style.display = 'block';
        }
        document.getElementById('modal-transcript-text').innerText = res.text || '(Empty transcript)';
      } catch (err) {
        document.getElementById('modal-transcript-text').innerText = 'Failed to load: ' + err;
      }
    }

    function closeModal() {
      document.getElementById('transcript-modal').style.display = 'none';
    }

    function openDiscoverModal() {
      document.getElementById('discover-modal').style.display = 'flex';
      document.getElementById('discover-input').focus();
    }

    function closeDiscoverModal() {
      document.getElementById('discover-modal').style.display = 'none';
    }

    async function submitAddChannel() {
      const input = document.getElementById('add-channel-input').value.trim();
      if (!input) return;

      showToast(`Resolving and tracking channel: ${input}...`);
      try {
        const res = await fetch('/api/channels/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input })
        }).then(r => r.json());

        if (res.status === 'success') {
          showToast(`✓ Tracked: ${res.title}`);
          document.getElementById('add-channel-input').value = '';
          await loadData();
        } else {
          alert('Error: ' + res.message);
        }
      } catch (e) {
        alert('Failed to add channel: ' + e);
      }
    }

    async function toggleMute(channelId) {
      try {
        const res = await fetch(`/api/channels/${encodeURIComponent(channelId)}/toggle`, { method: 'POST' }).then(r => r.json());
        if (res.status === 'success') {
          showToast(`Channel status updated to: ${res.new_status}`);
          await loadData();
        } else {
          alert('Could not change the channel status: ' + (res.message || 'unknown error'));
        }
      } catch (e) {
        alert('Toggle failed: ' + e);
      }
    }

    async function deleteChannel(channelId, title) {
      if (!confirm(`Stop tracking "${title || channelId}"?\n\nIts existing dossiers stay in the library.`)) return;
      try {
        const res = await fetch(`/api/channels/${encodeURIComponent(channelId)}/delete`, { method: 'POST' }).then(r => r.json());
        if (res.status === 'success') {
          showToast('Channel removed');
          await loadData();
        } else {
          alert('Could not remove the channel: ' + (res.message || 'unknown error'));
        }
      } catch (e) {
        alert('Delete failed: ' + e);
      }
    }

    // Suggestions of the open Related dialog, as the server sent them.
    let relatedItems = [];

    const RELATED_SOURCE_LABELS = {
      mentioned_in_kept_insights: 'Named in the kept dossiers of this channel',
      featured_on_channel_page: 'Channel item on the YouTube page of this creator'
    };

    function relatedSourceLabel(item) {
      const base = RELATED_SOURCE_LABELS[item.source] || `Source: ${item.source || 'unknown'}`;
      return item.mentions > 0 ? `${base} (${item.mentions} time${item.mentions === 1 ? '' : 's'})` : base;
    }

    function closeRelatedModal() {
      document.getElementById('related-modal').style.display = 'none';
    }

    function updateTrackRelatedButton() {
      const n = document.querySelectorAll('#related-results input[type=checkbox]:checked').length;
      const btn = document.getElementById('btn-track-related');
      btn.disabled = n === 0;
      btn.innerText = n > 0 ? `Track selected (${n})` : 'Track selected';
    }

    function relatedRowHtml(item, idx) {
      let control;
      let note = '';
      if (item.tracked) {
        control = '<span title="Already on the watchlist">✓</span>';
        note = 'Already tracked.';
      } else {
        control = `<input type="checkbox" data-idx="${idx}" onchange="updateTrackRelatedButton()">`;
        if (!item.channel_id) {
          note = 'Only the name is known. Tracking tries it as a YouTube @handle: it can fail or resolve to another channel, and the line shows what was found.';
        }
      }
      return `
        <div class="related-row">
          <div style="padding-top: 2px;">${control}</div>
          <div style="flex: 1;">
            <div style="font-weight: 600; color: var(--text-bright);">${esc(item.title || item.channel_id || '(no name)')}</div>
            <div class="related-source">${esc(relatedSourceLabel(item))}${item.channel_id ? ' • ' + esc(item.channel_id) : ''}</div>
            ${note ? `<div class="related-source">${esc(note)}</div>` : ''}
            <div class="related-source" id="related-outcome-${idx}"></div>
          </div>
        </div>
      `;
    }

    // Lists the suggestions with their source. Nothing is tracked here: that
    // happens in trackSelectedRelated, for ticked lines only.
    async function findRelated(channelId, title) {
      relatedItems = [];
      const results = document.getElementById('related-results');
      document.getElementById('related-modal-title').innerText = `🕸️ Related creators for ${title}`;
      results.innerHTML = '<p style="color: #8b949e;">Reading the local library and the YouTube page of this creator...</p>';
      updateTrackRelatedButton();
      document.getElementById('related-modal').style.display = 'flex';

      try {
        const res = await fetch(`/api/channels/${encodeURIComponent(channelId)}/related`).then(r => r.json());
        if (res.status !== 'success') {
          results.innerHTML = `<p style="color: #f85149;">Could not look up related creators: ${esc(res.message || 'unknown error')}</p>`;
          return;
        }
        relatedItems = Array.isArray(res.related) ? res.related : [];
        if (relatedItems.length === 0) {
          results.innerHTML = `<p style="color: var(--text);">Nothing found for ${esc(title)}: no other creator is named in the kept dossiers of this channel${res.page_checked ? ', and its YouTube page showed no channel items' : ''}. Nothing was tracked.</p>`;
          return;
        }
        results.innerHTML = relatedItems.map(relatedRowHtml).join('');
        updateTrackRelatedButton();
      } catch (e) {
        results.innerHTML = `<p style="color: #f85149;">Failed to look up related creators: ${esc(e)}</p>`;
      }
    }

    async function trackSelectedRelated() {
      const boxes = Array.from(document.querySelectorAll('#related-results input[type=checkbox]:checked'));
      if (boxes.length === 0) return;
      const btn = document.getElementById('btn-track-related');
      btn.disabled = true;
      btn.innerText = '⏳ Tracking...';

      let added = 0;
      for (const box of boxes) {
        const item = relatedItems[Number(box.dataset.idx)];
        if (!item) continue;
        let outcome;
        try {
          const addRes = await fetch('/api/channels/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: item.channel_id || item.title })
          }).then(r => r.json());
          if (addRes.status === 'success') {
            added++;
            outcome = `✓ Tracked as "${addRes.title}" (${addRes.channel_id})`;
            box.checked = false;
            box.disabled = true;
          } else {
            outcome = `✗ Not tracked: ${addRes.message || 'unknown error'}`;
          }
        } catch (e) {
          outcome = `✗ Not tracked: ${e}`;
        }
        const out = document.getElementById(`related-outcome-${box.dataset.idx}`);
        if (out) out.innerText = outcome;
      }

      showToast(`Tracked ${added} of ${boxes.length} selected creator${boxes.length === 1 ? '' : 's'}.`, 6000);
      updateTrackRelatedButton();
      if (added > 0) await loadData();
    }

    let crawlPollInterval = null;
    let crawlPollToken = 0;
    let crawlCardHideTimer = null;

    function hideCrawlCard() {
      const card = document.getElementById('crawl-progress-card');
      if (card) card.style.display = 'none';
    }

    // The token makes an answer that is still in flight from a stopped poll harmless.
    function stopCrawlPolling(hideCard) {
      if (crawlPollInterval) clearInterval(crawlPollInterval);
      crawlPollInterval = null;
      crawlPollToken++;
      if (hideCard) hideCrawlCard();
    }

    function hideCrawlCardLater(ms) {
      if (crawlCardHideTimer) clearTimeout(crawlCardHideTimer);
      crawlCardHideTimer = setTimeout(() => {
        crawlCardHideTimer = null;
        hideCrawlCard();
      }, ms);
    }

    function showCrawlProgress(msg, channelTitle = '') {
      const card = document.getElementById('crawl-progress-card');
      if (!card) return;
      // A hide timer left over from the crawl before must not close this one.
      if (crawlCardHideTimer) {
        clearTimeout(crawlCardHideTimer);
        crawlCardHideTimer = null;
      }
      card.style.display = 'block';
      document.getElementById('crawl-progress-msg').innerText = msg || 'Crawl in progress...';
      document.getElementById('crawl-detail-channel').innerText = channelTitle ? `Channel: ${channelTitle}` : 'Channel: All Active Channels';
      document.getElementById('crawl-detail-video').innerText = '';
      // Nothing is done yet: the bar starts empty and only moves with the
      // numbers the server reports.
      const bar = document.getElementById('crawl-progress-fill');
      if (bar) {
        bar.style.width = '0%';
        bar.style.background = '';
        bar.classList.remove('complete');
      }
      document.getElementById('crawl-stat-percent').innerText = '0%';
      showCrawlCounters({ stage: 'starting' });
    }

    // The pills of the progress card, straight from the server's counters.
    function showCrawlCounters(res) {
      const count = document.getElementById('crawl-stat-count');
      const total = Number(res.total_videos) || 0;
      const done = Number(res.current_index) || 0;
      if (res.stage === 'checking_feed') {
        // The number of new videos is not known before all feeds are read.
        count.innerText = 'Checking feeds';
      } else if (res.stage === 'updating_wiki') {
        count.innerText = 'Writing wiki';
      } else if (res.stage === 'starting') {
        count.innerText = 'Starting';
      } else {
        count.innerText = `${done}/${total} video${total === 1 ? '' : 's'}`;
      }
      document.getElementById('crawl-stat-kept').innerText = `${res.videos_kept || 0} Kept`;
      document.getElementById('crawl-stat-filtered').innerText = `${res.videos_filtered || 0} Below cutoff`;
      document.getElementById('crawl-stat-policy').innerText = `${res.videos_off_topic || 0} Off-topic`;
      document.getElementById('crawl-stat-insufficient').innerText = `${res.videos_insufficient || 0} Not enough info`;
      document.getElementById('crawl-stat-failed').innerText = `${res.videos_failed || 0} Failed`;
    }

    // One sentence for the toast at the end of a crawl, from the counters of
    // GET /api/crawl/progress. Every processed video is in exactly one group.
    function crawlCountsText(res) {
      const n = key => Number(res[key]) || 0;
      let text = `${n('videos_kept')} kept, ${n('videos_filtered')} below the quality cutoff, ` +
        `${n('videos_off_topic')} off-topic (fringe / nonsense / politics), ` +
        `${n('videos_insufficient')} not enough information, ${n('videos_failed')} failed`;
      if (n('videos_disqualified') > 0) text += `, ${n('videos_disqualified')} disqualified by hand`;
      return text;
    }

    function startCrawlProgressPolling(onCompleteCallback) {
      stopCrawlPolling(false);
      const token = crawlPollToken;
      let idlePolls = 0;
      let failedPolls = 0;

      const poll = async () => {
        const msg = document.getElementById('crawl-progress-msg');
        try {
          const res = await fetchJson('/api/crawl/progress');
          if (token !== crawlPollToken) return;
          failedPolls = 0;
          const card = document.getElementById('crawl-progress-card');
          const bar = document.getElementById('crawl-progress-fill');
          const pct = document.getElementById('crawl-stat-percent');
          const ch = document.getElementById('crawl-detail-channel');
          const vid = document.getElementById('crawl-detail-video');

          if (!card || !bar) return;

          // The server sets is_running to false only after the last step (the
          // wiki) is done, so "running" alone decides; no stage is assumed.
          if (res.is_running) {
            idlePolls = 0;
            card.style.display = 'block';
            const percent = Number(res.percent) || 0;
            bar.style.width = percent + '%';
            pct.innerText = percent + '%';
            msg.innerText = res.message || 'Crawl in progress...';
            showCrawlCounters(res);
            if (res.current_channel_title) ch.innerText = `Channel: ${res.current_channel_title}`;
            if (res.stage === 'updating_wiki') {
              vid.innerText = '';
            } else if (res.current_video_title) {
              vid.innerText = `Video: ${res.current_video_title}`;
            }
          } else if (res.stage === 'complete') {
            bar.style.width = '100%';
            bar.classList.add('complete');
            pct.innerText = '100%';
            msg.innerText = res.message || '✓ Crawl Complete!';
            showCrawlCounters(res);
            stopCrawlPolling(false);
            if (onCompleteCallback) onCompleteCallback(res);
            hideCrawlCardLater(5000);
          } else if (res.stage === 'error') {
            msg.innerText = '⚠️ ' + (res.message || 'Crawl failed');
            bar.style.background = '#f85149';
            showCrawlCounters(res);
            stopCrawlPolling(false);
            // Videos analysed before the stop are stored: show them.
            loadData();
            loadStatus(false);
            // Long enough to read; the card has no close button.
            hideCrawlCardLater(15000);
          } else {
            // Nothing is running (for example after a server restart). A few idle
            // answers are tolerated, because the first poll can arrive before
            // POST /api/crawl has claimed the running flag.
            idlePolls++;
            if (idlePolls >= 5) stopCrawlPolling(true);
          }
        } catch (e) {
          if (token !== crawlPollToken) return;
          console.warn('Progress poll error:', e);
          failedPolls++;
          if (failedPolls >= 8) {
            stopCrawlPolling(false);
            if (msg) msg.innerText = '⚠️ Lost contact with the YTC server.';
            hideCrawlCardLater(15000);
          }
        }
      };

      // No immediate poll: the first one must come after POST /api/crawl has
      // claimed the running flag, or it would read the end state of the crawl before.
      crawlPollInterval = setInterval(poll, 450);
    }

    // After a crawl request that was refused: when no crawl is running and there is
    // no error state for the poll to show, the bar would stay on "Starting..." for ever.
    // A crawl that is already running keeps the bar: the poll follows it to its end.
    async function releaseCrawlCardIfIdle() {
      let p = null;
      try {
        p = await fetchJson('/api/crawl/progress');
      } catch (_) {}
      if (p && (p.is_running || p.stage === 'error')) return;
      stopCrawlPolling(true);
    }

    async function runQuickCrawl() {
      const depth = parseInt(document.getElementById('crawl-depth-select')?.value) || 3;
      const btn = document.getElementById('btn-quick-crawl');
      btn.disabled = true;
      btn.innerText = '⏳ Crawling...';

      showCrawlProgress(`Starting crawl (depth: ${depth} videos/channel)...`);
      startCrawlProgressPolling(async (summary) => {
        btn.disabled = false;
        btn.innerText = '⚡ Crawl All';
        showToast(`✓ Crawl complete: ${crawlCountsText(summary)}. Wiki files updated.`, 7000);
        await loadData();
      });

      try {
        const res = await fetch('/api/crawl', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ limit: depth, delay: 10 })
        }).then(r => r.json());

        // The request returns when the crawl is over, whatever the outcome.
        btn.disabled = false;
        btn.innerText = '⚡ Crawl All';
        if (res.status !== 'success') {
          await releaseCrawlCardIfIdle();
          alert('Crawl error: ' + (res.message || 'unknown error'));
        }
      } catch (e) {
        stopCrawlPolling(true);
        alert('Crawl failed: ' + e);
        btn.disabled = false;
        btn.innerText = '⚡ Crawl All';
      }
    }

    async function crawlChannel(channelId) {
      const depth = parseInt(document.getElementById('crawl-depth-select')?.value) || 3;
      const chObj = allChannels.find(c => c.id === channelId);
      const chTitle = chObj ? chObj.title : channelId;

      showCrawlProgress(`Crawling ${chTitle} (depth: ${depth} videos)...`, chTitle);
      startCrawlProgressPolling(async (summary) => {
        showToast(`✓ Channel crawl complete: ${crawlCountsText(summary)}. Wiki files updated.`, 7000);
        await loadData();
      });

      try {
        const res = await fetch('/api/crawl', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ limit: depth, delay: 10, channel: channelId })
        }).then(r => r.json());

        if (res.status !== 'success') {
          await releaseCrawlCardIfIdle();
          alert('Crawl error: ' + (res.message || 'unknown error'));
        }
      } catch (e) {
        stopCrawlPolling(true);
        alert('Crawl failed: ' + e);
      }
    }

    async function submitDiscover() {
      const q = document.getElementById('discover-input').value.trim();
      if (!q) return;

      const btn = document.getElementById('btn-submit-discover');
      const resultsDiv = document.getElementById('discover-results');
      btn.disabled = true;
      btn.innerText = '⏳ Harvesting...';
      resultsDiv.style.display = 'block';
      resultsDiv.innerHTML = `<p style="color: #8b949e;">Searching candidates and scoring them with ${esc(engineName())}...</p>`;

      try {
        const res = await fetch('/api/discover', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, limit: 2 })
        }).then(r => r.json());

        if (res.status === 'success') {
          const items = Array.isArray(res.items) ? res.items : [];
          // candidates_found is what the search returned; only the first few are evaluated.
          const found = Number.isFinite(Number(res.candidates_found)) ? Number(res.candidates_found) : null;
          const foundText = found === null ? '' : ` of ${found} candidate${found === 1 ? '' : 's'} found by the search`;
          const limitText = res.evaluate_limit ? ` (at most ${esc(res.evaluate_limit)} are evaluated per run)` : '';
          let html = `<div style="font-size: 0.85rem; margin-bottom: 8px; color: var(--accent);">Evaluated ${items.length}${foundText}${limitText}:</div>`;
          items.forEach(it => {
            // "rejection" says in words why a video is not kept: off-topic
            // (fringe / nonsense / politics), not enough information,
            // disqualified, or below the quality cutoff.
            let verdict;
            if (it.is_worth_keeping) {
              verdict = `Score: ${esc(it.slop_score)}/100 (🟢 KEPT)`;
            } else if (it.not_scored) {
              verdict = `Not scored. ${esc(it.rejection || 'Not kept')}`;
            } else {
              verdict = `Score: ${esc(it.slop_score)}/100 (🔴 ${esc(it.rejection || 'Below quality cutoff')})`;
            }
            const promo = it.auto_promoted_creator ? ` • Added to the watchlist: ${esc(it.auto_promoted_creator)}` : '';
            html += `
              <div style="background: #21262d; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-weight: 600; color: #fff;">${esc(it.title)}</div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px;">
                  ${verdict}${promo}
                </div>
              </div>
            `;
          });
          if (res.wiki_error) {
            html += `<p style="color: #f85149; font-size: 0.82rem;">The results are stored, but the wiki files could not be written: ${esc(res.wiki_error)}</p>`;
          }
          resultsDiv.innerHTML = html;
          showToast('Discovery complete!');
          await loadData();
        } else {
          resultsDiv.innerHTML = `<p style="color: #f85149;">Error: ${esc(res.message || 'unknown error')}</p>`;
        }
      } catch (e) {
        resultsDiv.innerHTML = `<p style="color: #f85149;">Failed: ${esc(e)}</p>`;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Harvest Candidates';
      }
    }

    // Runs the LLM over the stored transcript of a heuristic entry. The server
    // replaces the entry only when the LLM gave a valid answer.
    async function reanalyzeVideo(videoId, btn) {
      if (btn) {
        btn.disabled = true;
        btn.innerText = '⏳ Re-analysing...';
      }
      try {
        const res = await fetch('/api/reanalyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video_id: videoId })
        }).then(r => r.json());

        if (res.status === 'success') {
          // The reason comes from the server: off-topic, not enough
          // information, disqualified, or below the cutoff.
          const kept = res.is_worth_keeping
            ? 'kept'
            : `not kept (${res.rejection || 'above the keep threshold of ' + res.max_slop}), so it leaves the library`;
          const score = res.not_scored ? 'not scored' : `${res.slop_score}/100`;
          showToast(`✓ Re-analysed by ${res.analyzer}: ${score}, ${kept}.` +
            (res.wiki_error ? ` The wiki files could not be written: ${res.wiki_error}` : ''), 7000);
          await loadData();
          return;
        }
        alert('Re-analysis did not happen; the entry is unchanged.\n\n' + (res.message || 'unknown error'));
      } catch (e) {
        alert('Re-analysis failed: the server did not answer (' + e + ').');
      }
      if (btn) {
        btn.disabled = false;
        btn.innerText = '🧠 Re-analyse';
      }
      // The engine may have gone offline: show what the server says now.
      loadStatus(false);
    }

    // Writes the wiki files. With prune === true the server also moves
    // generated pages that belong to nothing in the library any more into
    // the folder .ytc-pruned inside the wiki folder; it never deletes a page.
    async function syncWiki(prune) {
      const move = prune === true;
      try {
        const res = await fetch(move ? '/api/sync-wiki?prune=true' : '/api/sync-wiki', { method: 'POST' }).then(r => r.json());
        if (res.status === 'success') {
          const stale = Number(res.stale_pages) || 0;
          const moved = Number(res.pruned_pages) || 0;
          const left = stale - moved;
          const skipped = Number(res.skipped_user_files) || 0;
          let leftText = '';
          if (left > 0 && move) {
            leftText = ` ${left} page${left === 1 ? '' : 's'} could not be moved and ${left === 1 ? 'is' : 'are'} still in the wiki folder (the server window names ${left === 1 ? 'it' : 'them'}).`;
          } else if (left > 0) {
            leftText = ` ${left} generated page${left === 1 ? '' : 's'} of videos, topics, creators or categories that left the library ${left === 1 ? 'is' : 'are'} still in the wiki folder.`;
          }
          // A sync that left something undone stays on screen longer, and
          // offers the step that moves the leftover pages out.
          showToast(`✓ Wiki written to ${res.wiki_dir}: ${res.insights} insights, ${res.topics} topics, ${res.creators} creators.` +
            (skipped > 0 ? ` ${skipped} page${skipped === 1 ? ' was' : 's were'} NOT written, because a file that YTC did not generate has the same name. Move or rename that file, then sync again.` : '') +
            (moved > 0 ? ` ${moved} page${moved === 1 ? '' : 's'} that left the library ${moved === 1 ? 'was' : 'were'} moved to ${res.pruned_to || 'the folder .ytc-pruned inside the wiki folder'}. Nothing was deleted.` : '') +
            leftText,
            (skipped > 0 || left > 0 || moved > 0) ? 15000 : 7000,
            (left > 0 && !move) ? { label: 'Sync wiki and move removed pages', run: () => syncWikiAndMove() } : null);
        } else {
          alert('Failed to sync wiki: ' + (res.message || 'unknown error'));
        }
      } catch (e) {
        alert('Failed to sync wiki: the server did not answer (' + e + ').');
      }
    }

    // The explicit step after "Remove from wiki": the page file of a removed
    // video leaves the wiki folder. Asks first and says what happens to it.
    async function syncWikiAndMove() {
      if (!confirm('Sync the wiki and move removed pages?\n\nThe wiki files are written again. Generated pages of videos, topics, creators and categories that left the library are moved into the folder .ytc-pruned inside the wiki folder.\n\nNothing is deleted: a moved page can be put back from that folder.')) return;
      await syncWiki(true);
    }

    // Initial load. The library and the status are fetched side by side: the
    // status can take seconds when the LLM server is down, and an empty page
    // during that time looks like a lost library. Badge colours and the
    // Re-analyse buttons depend on the status, so the tab is drawn again when
    // it arrives - but only over a loaded library, never over an empty one.
    loadData();
    loadStatus(false).then(() => { if (libraryLoaded) renderCurrentTab(); });

    // Check if a background crawl is already active
    fetch('/api/crawl/progress').then(r => r.json()).then(res => {
      if (res && res.is_running) {
        showCrawlProgress(res.message, res.current_channel_title);
        startCrawlProgressPolling(async () => {
          await loadData();
        });
      }
    }).catch(() => {});

    // The same for a category assignment that is under way.
    fetchJson('/api/categorize/progress').then(p => {
      if (p && p.is_running) {
        showCategorizeProgress(p);
        startCategorizePolling();
      }
    }).catch(() => {});
  </script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn video() -> Video {
        Video {
            id: "vid00000001".to_string(),
            channel_id: "UCxxxxxxxxxxxxxxxxxxxxxx".to_string(),
            title: "A title".to_string(),
            url: "https://www.youtube.com/watch?v=vid00000001".to_string(),
            published_at: "2026-01-02T00:00:00+00:00".to_string(),
            status: "processed".to_string(),
            created_at: "2026-01-03T00:00:00+00:00".to_string(),
        }
    }

    fn channel() -> Channel {
        Channel {
            id: "UCxxxxxxxxxxxxxxxxxxxxxx".to_string(),
            title: "Some Creator".to_string(),
            url: "https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx".to_string(),
            status: "active".to_string(),
            rolling_slop_score: None,
            created_at: "2026-01-01T00:00:00+00:00".to_string(),
            last_crawled_at: None,
        }
    }

    fn insight(analyzer_type: &str, slop_score: u32) -> InsightResult {
        InsightResult {
            video_id: "vid00000001".to_string(),
            slop_score,
            is_worth_keeping: true,
            quality_rating: "High Signal".to_string(),
            summary: "Summary".to_string(),
            key_insights: vec![],
            takeaways: vec![],
            quotes: vec![],
            topics: vec![],
            mentioned_creators: vec![],
            analyzer_type: analyzer_type.to_string(),
            category: None,
        }
    }

    const PORT: u16 = 8088;

    #[test]
    fn same_origin_post_from_the_page_is_allowed() {
        for (host, origin) in [
            ("localhost:8088", "http://localhost:8088"),
            ("127.0.0.1:8088", "http://127.0.0.1:8088"),
            ("[::1]:8088", "http://[::1]:8088"),
            ("LocalHost:8088", "http://LOCALHOST:8088"),
        ] {
            assert!(
                request_allowed(&Method::POST, Some(host), Some(origin), Some("same-origin"), PORT),
                "{} / {}",
                host,
                origin
            );
        }
    }

    #[test]
    fn post_without_origin_is_allowed_for_command_line_clients() {
        assert!(request_allowed(&Method::POST, Some("127.0.0.1:8088"), None, None, PORT));
        assert!(request_allowed(&Method::POST, Some("localhost:8088"), None, None, PORT));
    }

    #[test]
    fn post_with_a_foreign_origin_is_refused() {
        let host = Some("localhost:8088");
        for origin in [
            "https://evil.example",
            "http://evil.example:8088",
            // Another program's page on this machine is still another origin.
            "http://localhost:3000",
            "https://localhost:8088",
            "http://localhost:8088.evil.example",
            "http://localhost:80880",
            "null",
            "",
        ] {
            assert!(
                !request_allowed(&Method::POST, host, Some(origin), None, PORT),
                "origin {:?} must be refused",
                origin
            );
        }
        // The browser's own verdict counts even when Origin is missing or local.
        assert!(!request_allowed(&Method::POST, host, None, Some("cross-site"), PORT));
        assert!(!request_allowed(&Method::POST, host, Some("http://localhost:8088"), Some("Cross-Site"), PORT));
    }

    #[test]
    fn a_foreign_host_is_refused_for_every_method() {
        // DNS rebinding: the page and its Origin look consistent, but the name
        // is not a local one.
        for method in [Method::GET, Method::POST, Method::HEAD, Method::OPTIONS] {
            for host in ["attacker.example", "attacker.example:8088", "localhost:9999", "localhost", "127.0.0.1", ""] {
                assert!(
                    !request_allowed(&method, Some(host), Some("http://attacker.example:8088"), None, PORT),
                    "{} with host {:?} must be refused",
                    method,
                    host
                );
                assert!(!request_allowed(&method, Some(host), None, None, PORT));
            }
            assert!(!request_allowed(&method, None, None, None, PORT));
        }
    }

    #[test]
    fn get_with_a_foreign_origin_depends_on_the_host_only() {
        let origin = Some("https://evil.example");
        assert!(request_allowed(&Method::GET, Some("localhost:8088"), origin, Some("cross-site"), PORT));
        assert!(!request_allowed(&Method::GET, Some("evil.example:8088"), origin, Some("cross-site"), PORT));
        // Plain navigation: no Origin at all.
        assert!(request_allowed(&Method::GET, Some("127.0.0.1:8088"), None, Some("none"), PORT));
    }

    #[test]
    fn the_port_may_be_left_out_only_when_it_is_80() {
        assert!(request_allowed(&Method::POST, Some("localhost"), Some("http://localhost"), None, 80));
        assert!(request_allowed(&Method::POST, Some("localhost:80"), Some("http://localhost:80"), None, 80));
        assert!(!request_allowed(&Method::POST, Some("localhost"), Some("http://localhost"), None, PORT));
    }

    #[test]
    fn deleting_a_channel_that_is_not_tracked_is_reported() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = crate::db::Database::open(dir.path().join("ytc.db")).expect("open test db");
        db.upsert_channel(&channel()).expect("store channel");

        assert!(!delete_tracked_channel(&db, "UCnotTrackedAtAll0000000").expect("lookup"));
        // The unknown id must not have removed anything.
        assert!(db.get_channel(&channel().id).expect("read").is_some());

        assert!(delete_tracked_channel(&db, &channel().id).expect("delete"));
        assert!(db.get_channel(&channel().id).expect("read").is_none());
        // A second request for the same id is an error, not a second success.
        assert!(!delete_tracked_channel(&db, &channel().id).expect("lookup"));
    }

    #[test]
    fn the_library_is_loaded_without_waiting_for_the_status() {
        let script = page_script();
        assert!(!script.contains("loadStatus(false).then(() => loadData())"));
        let data = script.rfind("\n    loadData();").expect("initial loadData call");
        let status = script.rfind("loadStatus(false).then(").expect("initial loadStatus call");
        assert!(data < status, "the library request must be sent before the status is awaited");
    }

    #[test]
    fn a_channel_average_of_zero_is_a_score_in_the_page() {
        let script = page_script();
        // Truthiness tests would turn the best score into "no score".
        assert!(!script.contains("c.avg_slop_score ?"));
        assert!(!script.contains("avg_slop_score ||"));
        assert!(script.contains("c.avg_slop_score == null ? 'Pending'"));
        assert!(script.contains("(a.avg_slop_score ?? 999) - (b.avg_slop_score ?? 999)"));
    }

    #[test]
    fn wiki_sync_answer_reports_stale_and_skipped_pages() {
        let stats = crate::wiki::WikiStats {
            topics_count: 4,
            creators_count: 2,
            categories_count: 5,
            insights_count: 9,
            output_dir: std::path::PathBuf::from("D:/notes/wiki"),
            stale_pages: 3,
            pruned_pages: 1,
            pruned_to: Some(std::path::PathBuf::from("D:/notes/wiki/.ytc-pruned/20260102-030405")),
            skipped_user_files: 2,
            ..Default::default()
        };
        let j = wiki_sync_json(&stats);
        assert_eq!(j["pruned_to"], "D:/notes/wiki/.ytc-pruned/20260102-030405");
        // Nothing moved: no folder is named.
        assert!(wiki_sync_json(&crate::wiki::WikiStats::default())["pruned_to"].is_null());
        assert_eq!(j["status"], "success");
        assert_eq!(j["topics"], 4);
        assert_eq!(j["creators"], 2);
        assert_eq!(j["categories"], 5);
        assert_eq!(j["insights"], 9);
        assert_eq!(j["stale_pages"], 3);
        assert_eq!(j["pruned_pages"], 1);
        assert_eq!(j["skipped_user_files"], 2);
        assert_eq!(j["wiki_dir"], "D:/notes/wiki");

        // The page reads exactly these names for its toast.
        let script = page_script();
        assert!(script.contains("Number(res.skipped_user_files) || 0"));
        assert!(script.contains("Number(res.stale_pages) || 0"));
    }

    #[test]
    fn url_host_drops_scheme_path_and_credentials() {
        assert_eq!(url_host("http://127.0.0.1:8080/v1"), "127.0.0.1:8080");
        assert_eq!(url_host(" https://api.example.com/v1/chat?x=1 "), "api.example.com");
        assert_eq!(url_host("https://user:secret@llm.local:1234/v1"), "llm.local:1234");
        assert_eq!(url_host("localhost:8080"), "localhost:8080");
        assert_eq!(url_host(""), "");
    }

    #[test]
    fn status_names_the_configured_llm() {
        let s = status_json(
            ("llm".to_string(), "some-model".to_string(), "http://user:pw@127.0.0.1:8080/v1".to_string()),
            true,
            30,
            Path::new("D:/notes/wiki"),
            Path::new("D:/data/ytc.db"),
            None,
        );
        assert_eq!(s["analyzer_mode"], "llm");
        assert_eq!(s["llm_model"], "some-model");
        assert_eq!(s["llm_url_host"], "127.0.0.1:8080");
        assert_eq!(s["llm_reachable"], true);
        assert_eq!(s["max_slop"], 30);
        assert_eq!(s["wiki_dir"], "D:/notes/wiki");
        assert_eq!(s["db_path"], "D:/data/ytc.db");
        assert_eq!(s["version"], env!("CARGO_PKG_VERSION"));
        // Unknown counts are null, never zeros.
        assert!(s["counts"].is_null());
        // The password of the URL must not be anywhere in the answer.
        assert!(!s.to_string().contains("pw"));
    }

    #[test]
    fn status_without_an_llm_reports_heuristic_and_never_reachable() {
        let s = status_json(
            ("heuristic".to_string(), String::new(), String::new()),
            // Even a wrong "true" from the caller must not show an LLM as online.
            true,
            45,
            Path::new("./wiki"),
            Path::new("ytc.db"),
            None,
        );
        assert_eq!(s["analyzer_mode"], "heuristic");
        assert!(s["llm_model"].is_null());
        assert!(s["llm_url_host"].is_null());
        assert_eq!(s["llm_reachable"], false);
    }

    #[test]
    fn scored_llm_insight_keeps_its_score_and_has_no_badges() {
        let j = insight_json(&video(), &channel(), &insight("llm:some-model", 12), None, None);
        assert_eq!(j["slop_score"], 12);
        assert_eq!(j["score_label"], "12/100");
        assert_eq!(j["not_scored"], false);
        assert_eq!(j["unreviewed"], false);
        assert!(j["read_note"].is_null());
        assert_eq!(j["channel_title"], "Some Creator");
    }

    #[test]
    fn heuristic_insight_is_marked_unreviewed() {
        let j = insight_json(&video(), &channel(), &insight(crate::analyzer::HEURISTIC_ANALYZER, 40), None, None);
        assert_eq!(j["unreviewed"], true);
        assert_eq!(j["not_scored"], false);
        assert_eq!(j["score_label"], "40/100");
        assert!(!is_heuristic_analyzer("llm:heuristic-sounding-model"));
    }

    #[test]
    fn the_read_note_names_the_sections_and_is_null_when_no_analysis_ran() {
        let read = |sections: Option<u32>| {
            let meta = InsightMeta { sections_total: sections, ..Default::default() };
            insight_json(&video(), &channel(), &insight("llm:m", 10), Some(&meta), None)["read_note"].clone()
        };
        assert_eq!(read(Some(3)), "whole transcript (3 sections)");
        assert_eq!(read(Some(1)), "whole transcript (1 section)");
        // Nothing recorded: no line at all, not a guess.
        assert!(read(None).is_null());
    }

    #[test]
    fn insight_row_has_every_field_and_null_where_nothing_is_stored() {
        // No meta row found, no transcript coverage.
        let j = insight_json(&video(), &channel(), &insight("llm:m", 10), None, None);
        for key in [
            "topic_verdict", "topic_reason", "info_status", "info_reason", "excluded_reason", "excluded_at",
            "sections_total", "transcript_source", "video_duration_sec", "covered_until_sec",
            "video_coverage_pct", "transcript_complete", "read_note", "rejection", "kept",
        ] {
            assert!(j.get(key).is_some(), "field {} is missing", key);
            assert!(j[key].is_null(), "field {} must be null, is {}", key, j[key]);
        }
        assert_eq!(j["excluded"], false);

        let meta = InsightMeta {
            topic_verdict: Some("ok".to_string()),
            topic_reason: Some("a history lecture".to_string()),
            info_status: Some("sufficient".to_string()),
            sections_total: Some(4),
            transcript_chars: Some(80_000),
            kept: true,
            ..Default::default()
        };
        let tc = TranscriptCoverage {
            video_duration_sec: Some(600.0),
            covered_until_sec: Some(590.0),
            source: Some("whisper".to_string()),
        };
        let mut i = insight("llm:m", 10);
        i.category = Some("Ancient History".to_string());
        let j = insight_json(&video(), &channel(), &i, Some(&meta), Some(&tc));
        assert_eq!(j["topic_verdict"], "ok");
        assert_eq!(j["topic_reason"], "a history lecture");
        assert_eq!(j["info_status"], "sufficient");
        assert!(j["info_reason"].is_null());
        assert_eq!(j["excluded"], false);
        assert_eq!(j["kept"], true);
        assert!(j["rejection"].is_null());
        assert_eq!(j["sections_total"], 4);
        assert_eq!(j["read_note"], "whole transcript (4 sections)");
        assert_eq!(j["transcript_source"], "whisper");
        assert_eq!(j["video_duration_sec"], 600.0);
        assert_eq!(j["covered_until_sec"], 590.0);
        // 10 seconds before the end: complete by the extractor's rule, and
        // the percentage still does not round up to 100.
        assert_eq!(j["transcript_complete"], true);
        assert_eq!(j["video_coverage_pct"], 98);
        assert_eq!(j["category"], "Ancient History");
        // No category is null, not an empty string or a default name.
        let plain = insight_json(&video(), &channel(), &insight("llm:m", 10), Some(&meta), Some(&tc));
        assert!(plain["category"].is_null());
    }

    #[test]
    fn a_transcript_that_stops_early_is_reported_as_incomplete() {
        let tc = TranscriptCoverage {
            video_duration_sec: Some(3600.0),
            covered_until_sec: Some(1800.0),
            source: Some("builtin".to_string()),
        };
        let j = insight_json(&video(), &channel(), &insight("llm:m", 10), None, Some(&tc));
        assert_eq!(j["transcript_complete"], false);
        assert_eq!(j["video_coverage_pct"], 50);

        // Without the video length nothing is claimed either way.
        let unknown = TranscriptCoverage {
            source: Some("subtitles".to_string()),
            ..Default::default()
        };
        let j = insight_json(&video(), &channel(), &insight("llm:m", 10), None, Some(&unknown));
        assert!(j["transcript_complete"].is_null());
        assert!(j["video_coverage_pct"].is_null());
        assert_eq!(j["transcript_source"], "subtitles");
    }

    #[test]
    fn insufficient_and_disqualified_rows_say_so_instead_of_showing_a_score() {
        let short = InsightMeta {
            info_status: Some("insufficient".to_string()),
            info_reason: Some("only 40 words in 31 seconds".to_string()),
            ..Default::default()
        };
        // 0 is the placeholder the pipeline stores for such a row.
        let j = insight_json(&video(), &channel(), &insight("info:insufficient", 0), Some(&short), None);
        assert_eq!(j["not_scored"], true);
        assert_eq!(j["score_label"], "Not enough information (only 40 words in 31 seconds)");
        assert_eq!(j["rejection"], "Not enough information (only 40 words in 31 seconds)");
        assert_eq!(j["kept"], false);

        let removed = InsightMeta {
            excluded_manual: true,
            excluded_reason: Some("duplicate".to_string()),
            excluded_at: Some("2026-11-11T10:00:00+00:00".to_string()),
            ..Default::default()
        };
        let j = insight_json(&video(), &channel(), &insight("llm:m", 10), Some(&removed), None);
        assert_eq!(j["excluded"], true);
        assert_eq!(j["excluded_reason"], "duplicate");
        assert_eq!(j["excluded_at"], "2026-11-11T10:00:00+00:00");
        assert_eq!(j["rejection"], "Disqualified (duplicate)");
        // The model's score is still shown: it was measured.
        assert_eq!(j["score_label"], "10/100");

        let politics = InsightMeta {
            topic_verdict: Some("politics".to_string()),
            topic_reason: Some("election commentary".to_string()),
            ..Default::default()
        };
        let j = insight_json(&video(), &channel(), &insight("llm:m", 15), Some(&politics), None);
        assert_eq!(j["topic_verdict"], "politics");
        assert_eq!(j["rejection"], "Off-topic: politics (election commentary)");
        assert_eq!(j["not_scored"], false);
    }

    #[test]
    fn status_counts_put_every_row_in_exactly_one_group() {
        let mut metas = HashMap::new();
        let mut add = |id: &str, meta: InsightMeta| {
            metas.insert(id.to_string(), meta);
        };
        add("kept1", InsightMeta { kept: true, ..Default::default() });
        add("kept2", InsightMeta { kept: true, topic_verdict: Some("ok".to_string()), ..Default::default() });
        add("slop", InsightMeta::default());
        add("fringe", InsightMeta { topic_verdict: Some("fringe".to_string()), ..Default::default() });
        add("short", InsightMeta { info_status: Some("insufficient".to_string()), ..Default::default() });
        // Disqualified wins over every other reason.
        add(
            "gone",
            InsightMeta {
                excluded_manual: true,
                topic_verdict: Some("politics".to_string()),
                info_status: Some("insufficient".to_string()),
                ..Default::default()
            },
        );
        let counts = library_counts_json(&metas);
        assert_eq!(
            counts,
            serde_json::json!({
                "kept": 2,
                "below_cutoff": 1,
                "off_topic": 1,
                "insufficient": 1,
                "excluded": 1
            })
        );
        // The groups add up to the number of rows.
        let sum: u64 = counts.as_object().expect("object").values().filter_map(|v| v.as_u64()).sum();
        assert_eq!(sum, 6);

        let s = status_json(
            ("heuristic".to_string(), String::new(), String::new()),
            false,
            45,
            Path::new("./wiki"),
            Path::new("ytc.db"),
            Some(counts),
        );
        assert_eq!(s["counts"]["kept"], 2);
        assert_eq!(s["counts"]["below_cutoff"], 1);
        assert_eq!(s["counts"]["off_topic"], 1);
        assert_eq!(s["counts"]["insufficient"], 1);
        assert_eq!(s["counts"]["excluded"], 1);

        // An empty library is all zeros, which is different from "unknown".
        assert_eq!(
            library_counts_json(&HashMap::new()),
            serde_json::json!({"kept": 0, "below_cutoff": 0, "off_topic": 0, "insufficient": 0, "excluded": 0})
        );
    }

    #[test]
    fn exclude_reason_body_is_optional_and_checked() {
        assert_eq!(parse_exclude_reason(b""), Ok(None));
        assert_eq!(parse_exclude_reason(b"  \n"), Ok(None));
        assert_eq!(parse_exclude_reason(b"{}"), Ok(None));
        assert_eq!(parse_exclude_reason(b"null"), Ok(None));
        assert_eq!(parse_exclude_reason(br#"{"reason": null}"#), Ok(None));
        assert_eq!(parse_exclude_reason(br#"{"reason": "   "}"#), Ok(None));
        assert_eq!(
            parse_exclude_reason(br#"{"reason": "  duplicate of another video "}"#),
            Ok(Some("duplicate of another video".to_string()))
        );
        // Cut by characters, not bytes: a multi-byte text must not panic.
        let long = format!(r#"{{"reason": "{}"}}"#, "ä".repeat(1000));
        let cut = parse_exclude_reason(long.as_bytes()).expect("valid body").expect("a reason");
        assert_eq!(cut.chars().count(), MAX_EXCLUDE_REASON_CHARS);

        assert!(parse_exclude_reason(b"not json").is_err());
        assert!(parse_exclude_reason(br#"{"reason": 5}"#).is_err());
        assert!(parse_exclude_reason(br#"["duplicate"]"#).is_err());
    }

    /// A database with one tracked channel and one kept, LLM-reviewed video.
    fn library_with_one_kept_video() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open test db");
        db.upsert_channel(&channel()).expect("store channel");
        db.upsert_video(&video()).expect("store video");
        db.save_insight(&insight("llm:m", 12)).expect("store insight");
        (dir, db)
    }

    #[test]
    fn excluding_a_video_takes_it_out_and_lists_it_until_it_is_included_again() {
        let (_dir, db) = library_with_one_kept_video();
        let id = video().id;
        assert_eq!(db.get_all_kept_videos_with_details().expect("kept").len(), 1);
        assert_eq!(excluded_list_json(&db).expect("list"), serde_json::json!([]));

        let j = exclude_video_json(&db, &format!(" {} ", id), Some("off my interests"));
        assert_eq!(j["status"], "success");
        assert_eq!(j["video_id"], id.as_str());
        assert_eq!(j["title"], "A title");
        assert_eq!(j["excluded"], true);
        assert_eq!(j["reason"], "off my interests");
        // The page relies on this to tell the owner to run Sync Wiki.
        assert_eq!(j["wiki_synced"], false);

        // GET /api/insights lists kept videos only: the card is gone.
        assert!(db.get_all_kept_videos_with_details().expect("kept").is_empty());

        let list = excluded_list_json(&db).expect("list");
        let rows = list.as_array().expect("an array");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row["video_id"], id.as_str());
        assert_eq!(row["title"], "A title");
        assert_eq!(row["channel_id"], channel().id.as_str());
        assert_eq!(row["channel_title"], "Some Creator");
        assert_eq!(row["reason"], "off my interests");
        assert_eq!(row["slop_score"], 12);
        let at = row["excluded_at"].as_str().expect("a date");
        assert!(chrono::DateTime::parse_from_rfc3339(at).is_ok(), "excluded_at {:?} is not a date", at);

        // The status counts move with it.
        let counts = library_counts_json(&db.list_insight_meta().expect("meta"));
        assert_eq!(counts["kept"], 0);
        assert_eq!(counts["excluded"], 1);

        let j = include_video_json(&db, &id);
        assert_eq!(j["status"], "success");
        assert_eq!(j["video_id"], id.as_str());
        assert_eq!(j["excluded"], false);
        assert_eq!(j["was_excluded"], true);
        assert_eq!(j["kept"], true);
        assert!(j["rejection"].is_null());
        assert_eq!(j["wiki_synced"], false);
        assert_eq!(db.get_all_kept_videos_with_details().expect("kept").len(), 1);
        assert_eq!(excluded_list_json(&db).expect("list"), serde_json::json!([]));

        // Including a video that is not disqualified changes nothing and says so.
        let j = include_video_json(&db, &id);
        assert_eq!(j["status"], "success");
        assert_eq!(j["was_excluded"], false);
        assert_eq!(j["kept"], true);
    }

    #[test]
    fn a_disqualification_without_a_reason_is_stored_without_one() {
        let (_dir, db) = library_with_one_kept_video();
        let j = exclude_video_json(&db, &video().id, None);
        assert_eq!(j["status"], "success");
        assert!(j["reason"].is_null());
        let list = excluded_list_json(&db).expect("list");
        assert!(list[0]["reason"].is_null());
    }

    #[test]
    fn a_restored_video_that_the_analysis_rejects_reports_why() {
        let (_dir, db) = library_with_one_kept_video();
        let id = video().id;
        let mut slop = insight("llm:m", 80);
        slop.is_worth_keeping = false;
        db.save_insight(&slop).expect("store insight");
        assert_eq!(exclude_video_json(&db, &id, None)["status"], "success");

        let j = include_video_json(&db, &id);
        assert_eq!(j["status"], "success");
        assert_eq!(j["was_excluded"], true);
        assert_eq!(j["kept"], false);
        assert_eq!(j["rejection"], "Below quality cutoff");
    }

    #[test]
    fn excluding_or_including_an_unknown_video_is_an_error() {
        let (_dir, db) = library_with_one_kept_video();
        for j in [
            exclude_video_json(&db, "noSuchVideo", Some("x")),
            include_video_json(&db, "noSuchVideo"),
            exclude_video_json(&db, "  ", None),
            include_video_json(&db, ""),
        ] {
            assert_eq!(j["status"], "error", "{}", j);
            assert!(j["message"].as_str().is_some_and(|m| !m.is_empty()), "{}", j);
        }
        // Nothing was disqualified by the failed requests.
        assert_eq!(excluded_list_json(&db).expect("list"), serde_json::json!([]));
        assert_eq!(db.get_all_kept_videos_with_details().expect("kept").len(), 1);
    }

    #[test]
    fn crawl_answer_carries_the_counters_and_the_summary_line() {
        let summary = crate::pipeline::CrawlSummary {
            channels_checked: 3,
            videos_evaluated: 5,
            videos_kept: 4,
            videos_filtered_slop: 1,
            videos_failed: 2,
            videos_off_topic: 3,
            videos_insufficient: 2,
            videos_disqualified: 1,
            wiki_generated: true,
            ..Default::default()
        };
        let j = crawl_answer_json(&summary);
        assert_eq!(j["status"], "success");
        assert_eq!(j["channels_checked"], 3);
        assert_eq!(j["videos_evaluated"], 5);
        assert_eq!(j["videos_kept"], 4);
        assert_eq!(j["videos_filtered_slop"], 1);
        assert_eq!(j["videos_failed"], 2);
        assert_eq!(j["videos_off_topic"], 3);
        assert_eq!(j["videos_insufficient"], 2);
        assert_eq!(j["videos_disqualified"], 1);
        assert_eq!(j["counts_line"], summary.counts_line());
        assert_eq!(j["wiki_generated"], true);
        assert!(j["wiki_error"].is_null());
    }

    #[test]
    fn reanalyze_answer_uses_the_effective_decision() {
        // The model liked it, but the owner disqualified it: not kept.
        let liked = insight("llm:m", 10);
        let removed = InsightMeta {
            excluded_manual: true,
            ..Default::default()
        };
        let j = reanalyze_answer_json(&liked, Some(&removed), 45, None);
        assert_eq!(j["status"], "success");
        assert_eq!(j["is_worth_keeping"], false);
        assert_eq!(j["kept"], false);
        assert_eq!(j["rejection"], "Disqualified");
        assert_eq!(j["not_scored"], false);
        assert_eq!(j["slop_score"], 10);
        assert_eq!(j["max_slop"], 45);
        assert!(j["wiki_error"].is_null());

        let short = InsightMeta {
            info_status: Some("insufficient".to_string()),
            ..Default::default()
        };
        let j = reanalyze_answer_json(&insight("info:insufficient", 0), Some(&short), 45, Some("disk full".to_string()));
        assert_eq!(j["kept"], false);
        assert_eq!(j["not_scored"], true);
        assert_eq!(j["rejection"], "Not enough information");
        assert_eq!(j["wiki_error"], "disk full");

        // Without a meta row the insight's own flag decides.
        let j = reanalyze_answer_json(&liked, None, 45, None);
        assert_eq!(j["kept"], true);
        assert!(j["rejection"].is_null());
    }

    #[test]
    fn the_raw_string_of_the_page_cannot_end_early() {
        // A quote followed by a hash would close the Rust raw string.
        assert!(!INDEX_HTML.contains("\"#"));
        assert!(INDEX_HTML.trim_end().ends_with("</html>"));
    }

    #[test]
    fn the_page_has_the_five_plain_counts() {
        let script = page_script();
        for (id, label) in [
            ("lib-count-kept", "Kept"),
            ("lib-count-below", "Below quality cutoff"),
            ("lib-count-offtopic", "Off-topic"),
            ("lib-count-insufficient", "Not enough information"),
            ("lib-count-excluded", "Disqualified"),
        ] {
            assert!(
                INDEX_HTML.contains(&format!(r#"<strong id="{}">-</strong> {}"#, id, label)),
                "count {} with label {:?} is missing",
                id,
                label
            );
            assert!(script.contains(&format!("set('{}'", id)), "count {} is never filled", id);
        }
        // The names the script reads are the names the server sends.
        let counts = library_counts_json(&HashMap::new());
        for key in ["kept", "below_cutoff", "off_topic", "insufficient", "excluded"] {
            assert!(counts.get(key).is_some());
            assert!(script.contains(&format!(", '{}');", key)), "the page does not read counts.{}", key);
        }
    }

    #[test]
    fn the_page_filters_by_the_users_categories_and_ships_none_of_its_own() {
        let script = page_script();
        // The only filters the page itself ships: everything and disqualified.
        let markup = &INDEX_HTML[..INDEX_HTML.find("<script>").expect("page has a script")];
        assert_eq!(markup.matches("data-filter=").count(), 2);
        assert!(markup.contains(r#"data-filter="all""#));
        assert!(markup.contains(r#"data-filter="excluded""#));
        assert!(INDEX_HTML.contains(r#"<span id="category-pills" style="display: none;"></span>"#));
        // Category pills are made from the server's list and hidden without one.
        assert!(script.contains("fetchJson('/api/categories')"));
        assert!(script.contains("box.style.display = allCategories.length > 0 ? 'contents' : 'none';"));
        assert!(script.contains("list = list.filter(i => i.category === selectedCategory);"));
        // A video without a category gets no badge.
        assert!(script.contains("if (!category) return '';"));
        // The manager adds and removes through the two POST routes.
        assert!(script.contains("fetch('/api/categories/add'"));
        assert!(script.contains("fetch('/api/categories/remove'"));
        for id in ["category-manager", "add-category-name", "add-category-description", "categories-table-body"] {
            assert!(INDEX_HTML.contains(&format!(r#"id="{}""#, id)), "{} is missing", id);
        }
    }

    #[test]
    fn categories_are_listed_added_and_removed_through_the_api_helpers() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open test db");
        assert_eq!(categories_json(&db).expect("list"), serde_json::json!([]));

        let added = add_category_json(&db, " Baking ", Some("bread and pastry"));
        assert_eq!(added["status"], "success");
        assert_eq!(added["name"], "Baking");
        assert_eq!(add_category_json(&db, "Star Gazing", Some("  "))["description"], serde_json::Value::Null);
        let duplicate = add_category_json(&db, "baking", None);
        assert_eq!(duplicate["status"], "error");
        assert!(duplicate["message"].as_str().is_some_and(|m| m.contains("already exists")), "{}", duplicate);
        assert_eq!(add_category_json(&db, "  ", None)["status"], "error");

        db.upsert_video(&video()).expect("video");
        let mut row = insight("llm:m", 10);
        row.category = Some("Baking".to_string());
        db.save_insight(&row).expect("insight");

        let listed = categories_json(&db).expect("list");
        assert_eq!(listed[0]["name"], "Baking");
        assert_eq!(listed[0]["description"], "bread and pastry");
        assert_eq!(listed[0]["kept"], 1);
        assert_eq!(listed[1]["name"], "Star Gazing");
        assert_eq!(listed[1]["kept"], 0);

        let removed = remove_category_json(&db, "BAKING");
        assert_eq!((removed["status"].as_str(), removed["name"].as_str(), removed["cleared"].as_u64()), (Some("success"), Some("Baking"), Some(1)));
        // The video is still there, without a category.
        let kept = db.get_insight(&row.video_id).expect("get").expect("the insight stays");
        assert_eq!(kept.category, None);
        assert_eq!(categories_json(&db).expect("list").as_array().map(Vec::len), Some(1));
        assert_eq!(remove_category_json(&db, "Baking")["status"], "error");
    }

    #[test]
    fn categories_are_renamed_and_described_through_the_api_helpers() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open test db");
        add_category_json(&db, "Baking", Some("bread"));
        add_category_json(&db, "Star Gazing", None);
        db.upsert_video(&video()).expect("video");
        let mut row = insight("llm:m", 10);
        row.category = Some("Baking".to_string());
        db.save_insight(&row).expect("insight");

        let taken = rename_category_json(&db, "baking", "star gazing");
        assert_eq!(taken["status"], "error");
        assert!(taken["message"].as_str().is_some_and(|m| m.contains("already exists")), "{}", taken);
        assert_eq!(rename_category_json(&db, "Gardening", "Plants")["status"], "error");
        assert_eq!(rename_category_json(&db, "Baking", "  ")["status"], "error");

        let renamed = rename_category_json(&db, "BAKING", " Bread and Pastry ");
        assert_eq!(
            (renamed["status"].as_str(), renamed["old_name"].as_str(), renamed["name"].as_str(), renamed["moved"].as_u64()),
            (Some("success"), Some("Baking"), Some("Bread and Pastry"), Some(1))
        );
        assert_eq!(db.get_insight(&row.video_id).expect("get").expect("row").category.as_deref(), Some("Bread and Pastry"));
        let listed = categories_json(&db).expect("list");
        assert_eq!((listed[0]["name"].as_str(), listed[0]["kept"].as_u64()), (Some("Bread and Pastry"), Some(1)));
        assert_eq!(listed[0]["description"], "bread");

        let described = describe_category_json(&db, "bread and pastry", Some(" loaves and ovens "));
        assert_eq!((described["status"].as_str(), described["description"].as_str()), (Some("success"), Some("loaves and ovens")));
        assert_eq!(described["name"], "Bread and Pastry");
        for blank in [None, Some(""), Some("  ")] {
            describe_category_json(&db, "Bread and Pastry", Some("something"));
            let removed = describe_category_json(&db, "Bread and Pastry", blank);
            assert_eq!(removed["status"], "success");
            assert!(removed["description"].is_null());
            assert!(categories_json(&db).expect("list")[0]["description"].is_null());
        }
        assert_eq!(describe_category_json(&db, "Gardening", Some("plants"))["status"], "error");
    }

    #[test]
    fn the_category_of_a_video_is_set_by_hand_and_handed_back_through_the_api_helper() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("ytc.db")).expect("open test db");
        add_category_json(&db, "Baking", None);
        add_category_json(&db, "Star Gazing", None);
        db.upsert_video(&video()).expect("video");
        db.save_insight(&insight("llm:m", 10)).expect("insight");
        let id = "vid00000001";
        let row_json = |db: &Database| {
            let stored = db.get_insight(id).expect("get").expect("row");
            let meta = db.get_insight_meta(id).expect("meta");
            insight_json(&video(), &channel(), &stored, meta.as_ref(), None)
        };
        assert_eq!(row_json(&db)["category_manual"], false);
        // Without the extra facts of the row nothing claims "by hand".
        assert_eq!(insight_json(&video(), &channel(), &insight("llm:m", 10), None, None)["category_manual"], false);

        let set = set_video_category_json(&db, " vid00000001 ", Some("star gazing"), false);
        assert_eq!(
            (set["status"].as_str(), set["category"].as_str(), set["category_manual"].as_bool(), set["title"].as_str()),
            (Some("success"), Some("Star Gazing"), Some(true), Some("A title"))
        );
        let row = row_json(&db);
        assert_eq!((row["category"].as_str(), row["category_manual"].as_bool()), (Some("Star Gazing"), Some(true)));

        // An unknown name changes nothing; the message lists the names.
        let unknown = set_video_category_json(&db, id, Some("Gardening"), false);
        assert_eq!(unknown["status"], "error");
        assert!(unknown["message"].as_str().is_some_and(|m| m.contains("Your categories: Baking, Star Gazing")), "{}", unknown);
        assert_eq!(row_json(&db)["category"], "Star Gazing");

        // null (and a blank name) is "no category", by hand.
        for none in [None, Some("  ")] {
            set_video_category_json(&db, id, Some("Baking"), false);
            let cleared = set_video_category_json(&db, id, none, false);
            assert_eq!(cleared["status"], "success");
            assert!(cleared["category"].is_null());
            assert_eq!(cleared["category_manual"], true);
        }

        // Automatic: the mark goes, the category stays as it is.
        set_video_category_json(&db, id, Some("Baking"), false);
        let auto = set_video_category_json(&db, id, None, true);
        assert_eq!((auto["status"].as_str(), auto["category"].as_str(), auto["category_manual"].as_bool()), (Some("success"), Some("Baking"), Some(false)));
        // A category sent along with auto is not applied.
        let auto = set_video_category_json(&db, id, Some("Star Gazing"), true);
        assert_eq!((auto["category"].as_str(), auto["category_manual"].as_bool()), (Some("Baking"), Some(false)));

        assert_eq!(set_video_category_json(&db, "vid00000002", Some("Baking"), false)["status"], "error");
        assert_eq!(set_video_category_json(&db, "vid00000002", None, true)["status"], "error");
        assert_eq!(set_video_category_json(&db, "  ", None, false)["status"], "error");
    }

    #[test]
    fn the_categorize_answer_names_every_group() {
        let summary = crate::pipeline::CategorizeSummary {
            total: 9,
            beyond_limit: 4,
            assigned: vec![("Baking".to_string(), 3), ("Star Gazing".to_string(), 0)],
            none: 2,
            failed: 2,
            skipped: 1,
            not_attempted: 1,
            stopped: Some("The LLM failed for 2 videos in a row".to_string()),
        };
        let j = categorize_answer_json(&summary);
        assert_eq!(j["status"], "success");
        assert_eq!((j["total"].as_u64(), j["assigned"].as_u64(), j["none"].as_u64(), j["failed"].as_u64()), (Some(9), Some(3), Some(2), Some(2)));
        assert_eq!((j["skipped"].as_u64(), j["not_attempted"].as_u64(), j["beyond_limit"].as_u64()), (Some(1), Some(1), Some(4)));
        assert_eq!(j["per_category"], serde_json::json!([{"name": "Baking", "count": 3}, {"name": "Star Gazing", "count": 0}]));
        assert_eq!(j["stopped"], "The LLM failed for 2 videos in a row");
        assert_eq!(
            j["counts_line"],
            "3 assigned (Baking: 3), 2 in no category, 2 failed, 1 set by hand meanwhile, 1 not attempted"
        );
        assert!(categorize_answer_json(&crate::pipeline::CategorizeSummary::default())["stopped"].is_null());

        // The page reads exactly these names.
        let script = page_script();
        for name in ["res.counts_line", "res.stopped", "res.total", "res.assigned", "p.total_videos", "p.current_index"] {
            assert!(script.contains(name), "{} is not read by the page", name);
        }
    }

    #[test]
    fn the_page_manages_categories_and_sets_the_category_of_a_video() {
        let script = page_script();
        // Name and description of every category are edited in its table
        // row and saved through their routes; only what changed is sent.
        assert!(script.contains(r#"onclick="editCategory(${jsArg(c.name)})">Edit</button>"#));
        assert!(script.contains(r#"onclick="saveCategory(${jsArg(c.name)})">Save</button>"#));
        assert!(script.contains("postJson('/api/categories/rename', { name, new_name: newName })"));
        assert!(script.contains("postJson('/api/categories/describe', { name: stored, description })"));
        // No browser question box: it cannot be styled and some browsers hide it.
        assert!(!script.contains("prompt(`New name"));
        // The assignment for existing videos: started by its button, followed by polling.
        assert!(INDEX_HTML.contains(r#"id="btn-categorize" onclick="runCategorize()""#));
        assert!(INDEX_HTML.contains("Assign categories to existing videos</button>"));
        assert!(INDEX_HTML.contains(r#"id="categorize-status""#));
        assert!(script.contains("postJson('/api/categorize', {})"));
        assert!(script.contains("fetchJson('/api/categorize/progress')"));
        // The dropdown on every video: the user's categories, none, automatic.
        assert!(script.contains("option('auto', 'Automatic (model decides)', !manual)"));
        assert!(script.contains("option('none', 'No category', manual && !i.category)"));
        assert!(script.contains("allCategories.map(c => option('c:' + c.name, c.name, manual && i.category === c.name))"));
        assert!(script.contains("postJson(`/api/insights/${encodeURIComponent(videoId)}/category`, body)"));
        assert!(script.contains("? { auto: true }"));
        assert_eq!(script.matches("${categorySelectHtml(i)}").count(), 2, "cards and table");
        assert_eq!(script.matches("${categoryByHandHtml(i)}").count(), 2, "cards and table");
        // No dropdown while the user has no categories and nothing was set by hand.
        assert!(script.contains("if (allCategories.length === 0 && !i.category_manual) return '';"));
        // The marker appears only for a choice made by hand.
        assert!(script.contains("if (!i.category_manual) return '';"));
        assert!(script.contains("'set by hand'"));
    }

    #[tokio::test]
    async fn the_category_endpoints_refuse_requests_from_another_origin() {
        let dir = tempfile::tempdir().expect("temp dir");
        let pipeline = Arc::new(IngestionPipeline::new(&dir.path().join("ytc.db"), None, None, None).expect("pipeline"));
        let db = pipeline.db().expect("db");
        db.upsert_channel(&channel()).expect("channel");
        db.upsert_video(&video()).expect("video");
        db.save_insight(&insight("llm:m", 10)).expect("insight");
        db.add_category("Baking", Some("bread")).expect("category");

        // The real router with its guard, on a loopback port the OS picked.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let state = Arc::new(AppState { pipeline: pipeline.clone(), jobs: Arc::new(JobSlot::default()) });
        let app = app_router(state.clone(), port);
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let client = reqwest::Client::builder().no_proxy().build().expect("client");
        let base = format!("http://127.0.0.1:{}", port);
        let own_origin = base.clone();
        let posts = [
            ("/api/categories/describe", serde_json::json!({"name": "Baking", "description": "loaves"})),
            ("/api/insights/vid00000001/category", serde_json::json!({"category": "Baking"})),
            ("/api/insights/vid00000001/category", serde_json::json!({"auto": true})),
            ("/api/categorize", serde_json::json!({"all": true})),
            ("/api/categories/rename", serde_json::json!({"name": "Baking", "new_name": "Bread"})),
        ];

        for (path, body) in &posts {
            let url = format!("{}{}", base, path);
            for origin in ["https://evil.example", "http://localhost:3000", "null"] {
                let refused = client.post(&url).header("Origin", origin).json(body).send().await.expect("answer");
                assert_eq!(refused.status(), StatusCode::FORBIDDEN, "{} from {}", path, origin);
            }
            // The browser's own verdict, without or with a local Origin.
            let cross = client.post(&url).header("Sec-Fetch-Site", "cross-site").json(body).send().await.expect("answer");
            assert_eq!(cross.status(), StatusCode::FORBIDDEN, "{} cross-site", path);
            let cross = client
                .post(&url)
                .header("Origin", &own_origin)
                .header("Sec-Fetch-Site", "cross-site")
                .json(body)
                .send()
                .await
                .expect("answer");
            assert_eq!(cross.status(), StatusCode::FORBIDDEN, "{} cross-site with a local origin", path);
            // DNS rebinding: a foreign name in Host.
            let rebound = client
                .post(&url)
                .header("Host", format!("attacker.example:{}", port))
                .json(body)
                .send()
                .await
                .expect("answer");
            assert_eq!(rebound.status(), StatusCode::FORBIDDEN, "{} with a foreign host", path);
        }
        // Nothing reached a handler.
        let listed = db.list_categories().expect("list");
        assert_eq!((listed.len(), listed[0].name.as_str(), listed[0].description.as_deref()), (1, "Baking", Some("bread")));
        assert_eq!(db.get_insight("vid00000001").expect("get").expect("row").category, None);
        assert!(!db.get_insight_meta("vid00000001").expect("meta").expect("row").category_manual);
        assert_eq!(pipeline.categorize_progress.read().await.stage, "");
        assert!(state.jobs.try_claim(JobKind::Categorize).is_ok(), "no job was started");

        // The same requests from the server's own page are answered.
        let mut answers = Vec::new();
        for (path, body) in &posts {
            let response = client
                .post(format!("{}{}", base, path))
                .header("Origin", &own_origin)
                .json(body)
                .send()
                .await
                .expect("answer");
            assert_eq!(response.status(), StatusCode::OK, "{}", path);
            answers.push(response.json::<serde_json::Value>().await.expect("json"));
        }
        assert_eq!(answers[0]["description"], "loaves");
        assert_eq!((answers[1]["category"].as_str(), answers[1]["category_manual"].as_bool()), (Some("Baking"), Some(true)));
        assert_eq!((answers[2]["category"].as_str(), answers[2]["category_manual"].as_bool()), (Some("Baking"), Some(false)));
        // No LLM is configured here: the run is refused by the handler, not by the guard.
        assert_eq!(answers[3]["status"], "error");
        assert!(answers[3]["message"].as_str().is_some_and(|m| m.contains("needs an LLM")), "{}", answers[3]);
        assert_eq!((answers[4]["status"].as_str(), answers[4]["name"].as_str()), (Some("success"), Some("Bread")));
        assert_eq!(db.get_insight("vid00000001").expect("get").expect("row").category.as_deref(), Some("Bread"));
        // The failed run released the job slot and left an error state.
        assert!(state.jobs.try_claim(JobKind::Categorize).is_ok());
        let progress = client.get(format!("{}/api/categorize/progress", base)).send().await.expect("answer");
        assert_eq!(progress.status(), StatusCode::OK);
        let progress = progress.json::<serde_json::Value>().await.expect("json");
        assert_eq!((progress["is_running"].as_bool(), progress["stage"].as_str()), (Some(false), Some("error")));
    }

    #[test]
    fn the_page_can_remove_and_restore_a_video_and_points_to_sync_wiki() {
        let script = page_script();
        assert!(script.contains("/exclude`"));
        assert!(script.contains("/include`"));
        assert!(script.contains("fetchJson('/api/excluded')"));
        assert!(script.contains("Remove from wiki"));
        assert!(script.contains("label: 'Undo'"));
        assert!(script.contains("↩ Restore"));
        // Removing asks first and names the video.
        assert!(script.contains("if (!confirm(`Remove \"${name}\" from the wiki?"));
        // Neither action rewrites wiki files. Restoring points to Sync Wiki;
        // removing says that the page file stays and where to move it out.
        assert_eq!(script.matches("Run Sync Wiki to update the wiki files.").count(), 1);
        assert!(script.contains("The wiki files are unchanged: Sync Wiki takes it out of the index, and its page file stays in the wiki folder until you use 'Sync wiki and move removed pages'"));
        assert!(script.contains("No wiki file is changed or deleted by this."));
        assert!(INDEX_HTML.contains(r#"id="view-excluded""#));
        assert!(INDEX_HTML.contains(r#"data-filter="excluded""#));
    }

    #[test]
    fn the_page_shows_the_crawl_counters_in_plain_wording() {
        let script = page_script();
        for id in ["crawl-stat-kept", "crawl-stat-filtered", "crawl-stat-policy", "crawl-stat-insufficient", "crawl-stat-failed"] {
            assert!(INDEX_HTML.contains(&format!(r#"id="{}""#, id)), "pill {} is missing", id);
            assert!(script.contains(&format!("getElementById('{}')", id)), "pill {} is never filled", id);
        }
        assert!(script.contains("res.videos_off_topic"));
        assert!(script.contains("res.videos_insufficient"));
        assert!(script.contains("it.rejection"));
        assert!(script.contains("Read: ${i.read_note}"));
    }

    #[test]
    fn only_one_long_job_can_hold_the_slot() {
        let slot = Arc::new(JobSlot::default());
        let crawl = slot.try_claim(JobKind::Crawl).expect("an idle slot can be claimed");

        // Every kind is refused while the crawl holds it, with a message that
        // names what is running and what was not started.
        for kind in JobKind::ALL {
            let message = slot.try_claim(kind).expect_err("the slot is taken");
            assert!(message.starts_with("Not started: a crawl is running"), "{}", message);
            assert!(message.contains(kind.label()), "{}", message);
            assert!(message.contains("Wait for it to finish"), "{}", message);
        }
        // A refused claim must not free the slot of the job that holds it.
        assert!(slot.try_claim(JobKind::Reanalyze).is_err());

        drop(crawl);
        let reanalysis = slot.try_claim(JobKind::Reanalyze).expect("free again after the crawl");
        let message = slot.try_claim(JobKind::Crawl).expect_err("held by the re-analysis");
        assert!(message.starts_with("Not started: a re-analysis is running"), "{}", message);
        drop(reanalysis);
        assert!(slot.try_claim(JobKind::Discover).is_ok());

        // An unknown code still gives a usable message.
        assert!(busy_message(None, JobKind::Crawl).starts_with("Not started: another job is running"));
        for kind in JobKind::ALL {
            assert_eq!(JobKind::from_code(kind as u8), Some(kind));
        }
        assert_eq!(JobKind::from_code(0), None);
        // A category assignment and a crawl shut each other out.
        let categorize = slot.try_claim(JobKind::Categorize).expect("idle");
        let message = slot.try_claim(JobKind::Crawl).expect_err("held by the category assignment");
        assert!(message.starts_with("Not started: a category assignment is running"), "{}", message);
        drop(categorize);

        let j = busy_json("m".to_string()).0;
        assert_eq!(j["status"], "error");
        assert_eq!(j["busy"], true);
        assert_eq!(j["message"], "m");
    }

    #[test]
    fn requests_arriving_together_start_exactly_one_job() {
        let slot = Arc::new(JobSlot::default());
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|n| {
                let slot = Arc::clone(&slot);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let kind = if n % 2 == 0 { JobKind::Crawl } else { JobKind::Reanalyze };
                    // The winner's guard is handed back, so it is still held
                    // while the others try.
                    slot.try_claim(kind).ok()
                })
            })
            .collect();
        let guards: Vec<_> = threads
            .into_iter()
            .filter_map(|t| t.join().expect("thread finished"))
            .collect();
        assert_eq!(guards.len(), 1);
        drop(guards);
        assert!(slot.try_claim(JobKind::Discover).is_ok());
    }

    #[tokio::test]
    async fn the_job_slot_is_released_when_a_job_fails_or_panics() {
        let slot = Arc::new(JobSlot::default());

        // A job that ends with an error, as the handlers run it: the guard
        // lives in the spawned task.
        let job = slot.try_claim(JobKind::Discover).expect("idle");
        let task = tokio::spawn(async move {
            let _job = job;
            Err::<(), anyhow::Error>(anyhow::anyhow!("LLM engine not available"))
        });
        assert!(task.await.expect("no panic").is_err());
        assert!(slot.try_claim(JobKind::Crawl).is_ok(), "an error must free the slot");

        // A job that panics half-way.
        let job = slot.try_claim(JobKind::Reanalyze).expect("idle");
        let task = tokio::spawn(async move {
            let _job = job;
            if std::hint::black_box(true) {
                panic!("job crashed (expected in this test)");
            }
        });
        assert!(task.await.expect_err("the task panicked").is_panic());
        assert!(slot.try_claim(JobKind::Crawl).is_ok(), "a panic must free the slot");

        // A task that is cancelled before it finished.
        let job = slot.try_claim(JobKind::Crawl).expect("idle");
        let task = tokio::spawn(async move {
            let _job = job;
            std::future::pending::<()>().await;
        });
        task.abort();
        assert!(task.await.expect_err("aborted").is_cancelled());
        assert!(slot.try_claim(JobKind::Crawl).is_ok(), "a cancelled job must free the slot");
    }

    fn search_hit(slop_score: u32) -> crate::db::SearchHit {
        crate::db::SearchHit {
            video_id: "vid00000001".to_string(),
            title: "A title".to_string(),
            slop_score,
            is_worth_keeping: false,
            quality_rating: "Not scored".to_string(),
            summary: "Summary".to_string(),
            key_insights: vec![],
            topics: vec![],
            category: None,
            topic_verdict: None,
            info_status: None,
            info_reason: None,
            excluded_manual: false,
        }
    }

    #[test]
    fn search_hits_without_a_score_are_never_shown_as_a_number() {
        // "Not enough information" rows hold the placeholder 0.
        let j = search_hit_json(crate::db::SearchHit {
            info_status: Some("insufficient".to_string()),
            ..search_hit(0)
        });
        assert_eq!(j["score_label"], "Not enough information");
        assert_eq!(j["not_scored"], true);
        assert_eq!(j["info_status"], "insufficient");
        let j = search_hit_json(crate::db::SearchHit {
            info_status: Some("insufficient".to_string()),
            info_reason: Some("only 40 words".to_string()),
            ..search_hit(0)
        });
        assert_eq!(j["score_label"], "Not enough information (only 40 words)");
        assert_eq!(j["info_reason"], "only 40 words");

        // A real 0 stays a score.
        let j = search_hit_json(crate::db::SearchHit {
            is_worth_keeping: true,
            topic_verdict: Some("ok".to_string()),
            info_status: Some("sufficient".to_string()),
            ..search_hit(0)
        });
        assert_eq!(j["score_label"], "0/100");
        assert_eq!(j["not_scored"], false);
        assert_eq!(j["is_worth_keeping"], true);
        assert_eq!(search_hit_json(search_hit(37))["score_label"], "37/100");
    }

    #[test]
    fn the_page_offers_to_move_removed_wiki_pages_and_says_nothing_is_deleted() {
        let script = page_script();
        assert!(script.contains("async function syncWikiAndMove()"));
        assert!(script.contains("'/api/sync-wiki?prune=true'"));
        // The move happens only after a question that says what it does.
        assert!(script.contains("if (!confirm('Sync the wiki and move removed pages?"));
        assert!(script.contains("Nothing is deleted: a moved page can be put back from that folder."));
        assert!(script.contains("await syncWiki(true);"));
        // The plain button never moves anything.
        assert!(INDEX_HTML.contains(r#"onclick="syncWiki()""#));
        assert!(script.contains("const move = prune === true;"));
        // Offered where pages are left over, and in the Disqualified list.
        assert!(script.contains("{ label: 'Sync wiki and move removed pages', run: () => syncWikiAndMove() }"));
        assert!(INDEX_HTML.contains(r#"onclick="syncWikiAndMove()">Sync wiki and move removed pages</button>"#));
        assert!(script.contains("res.pruned_to"));
        assert!(script.contains("Nothing was deleted."));
        assert!(INDEX_HTML.contains(".ytc-pruned"));
    }

    /// The inline script of the embedded page.
    fn page_script() -> &'static str {
        let start = INDEX_HTML.find("<script>").expect("page has a script") + "<script>".len();
        let end = INDEX_HTML.rfind("</script>").expect("script is closed");
        &INDEX_HTML[start..end]
    }

    #[test]
    fn every_inline_handler_of_the_page_is_defined_in_its_script() {
        let script = page_script();
        let attr = regex::Regex::new(r#"\bon(?:click|change|input)="([^"]*)""#).unwrap();
        let call = regex::Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*)\(").unwrap();
        // Calls inside a handler that are browser API, not page functions.
        let builtin = ["getElementById", "stopPropagation"];
        let mut checked = 0;
        for a in attr.captures_iter(INDEX_HTML) {
            for c in call.captures_iter(&a[1]) {
                let name = &c[1];
                if builtin.contains(&name) {
                    continue;
                }
                assert!(
                    script.contains(&format!("function {}(", name)),
                    "inline handler calls {}(), which the script does not define",
                    name
                );
                checked += 1;
            }
        }
        assert!(checked > 20, "handler scan found too little ({}): the pattern is broken", checked);
    }

    #[test]
    fn the_page_takes_the_keep_threshold_from_the_server() {
        let script = page_script();
        // The keep threshold comes from /api/status, not from a literal.
        assert!(!script.contains("<= 45"));
        assert!(script.contains("/api/status"));
        assert!(script.contains("/api/reanalyze"));
    }
}
