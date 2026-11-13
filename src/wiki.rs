use crate::db::Database;
use crate::models::{Category, Channel, InsightMeta, InsightResult, TranscriptCoverage, Video};
use anyhow::{Context, Result};
use colored::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Creator page for kept videos whose channel is not tracked.
const INDEPENDENT_SLUG: &str = "independent-analysis";

/// Frontmatter line that marks a page as written by this generator. Only
/// pages carrying it are ever counted as stale or pruned.
const GENERATED_MARKER: &str = "generated_by: ytc";
/// Frontmatter block of generated pages that have no other metadata.
const GENERATED_FRONTMATTER: &str = "---\ngenerated_by: ytc\n---\n\n";

// Fixed lines of the generated sub-pages.
const HOME_CRUMB: &str = "⬅️ [[../index|Wiki Home]]";
const INSIGHT_CRUMB_PREFIX: &str = "⬅️ [[../index|Wiki Home]] | 🎙️ Creator: [[../creators/";
const TOPIC_HEADING_PREFIX: &str = "# Concept: ";
const CATEGORY_HEADING_PREFIX: &str = "# Category: ";
const CREATOR_HEADING_PREFIX: &str = "# Creator Profile: ";
const TOPIC_SECTION_HEADING: &str = "## 📚 Contributing Video Analyses";

// First lines of the root pages. They carry no frontmatter, so their fixed
// head is what tells a generated root page from a user's file of the same name.
const INDEX_HEAD: &str = "# 🌌 YTC Knowledge Wiki\n\n> ";

/// Folder inside the wiki that receives pruned pages, one subfolder per run.
const PRUNED_DIR: &str = ".ytc-pruned";

type KeptVideo = (Video, Channel, InsightResult);

/// The kinds of page the generator owns: the four folders it fills
/// one-to-one, and the root pages (index/Home).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageKind {
    Insight,
    Topic,
    Creator,
    /// One page per category of the user's that holds kept videos.
    Category,
    Index,
}

/// Writes pages and remembers their names, so the sync step knows which
/// files belong to the current run.
#[derive(Default)]
struct PageWriter {
    written: HashSet<PathBuf>,
    /// Existing files that are not recognisably generated and were therefore
    /// left as they are instead of being overwritten.
    skipped: Vec<PathBuf>,
}

impl PageWriter {
    fn write(&mut self, dir: &Path, file_name: &str, kind: PageKind, content: &str) -> Result<()> {
        let path = dir.join(file_name);
        // The wiki folder may be shared with the user's own notes (a vault):
        // a file of the same name that ytc did not write is never replaced.
        if is_foreign_file(&path, kind) {
            eprintln!(
                "Warning: {:?} exists and was not written by ytc; it is left untouched and this page is not written (rename or move the file to let ytc write it).",
                path
            );
            self.skipped.push(path);
            return Ok(());
        }
        fs::write(&path, content).with_context(|| format!("Failed to write wiki page {:?}", path))?;
        self.written.insert(page_key(dir, file_name));
        Ok(())
    }

    fn wrote(&self, dir: &Path, file_name: &str) -> bool {
        self.written.contains(&page_key(dir, file_name))
    }
}

/// True when `path` holds content that is not recognisably a generated page
/// of `kind`. A missing or empty file is not foreign: nothing would be lost.
fn is_foreign_file(path: &Path, kind: PageKind) -> bool {
    match fs::read(path) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => !text.trim().is_empty() && !is_generated_page(kind, &text),
            // The generator only writes UTF-8.
            Err(_) => true,
        },
        // Not there (or not readable as a file): the write itself reports real problems.
        Err(_) => false,
    }
}

/// True when `video_id` may be used as the file name of an insight page and
/// inside the wiki links that point to it: 1 to 64 characters of
/// `[A-Za-z0-9_-]`, which every YouTube id is. Ids are copied verbatim from
/// imported databases (`ytc merge-db`); one holding a path separator, a dot or
/// a drive letter would otherwise make the generator write outside the wiki
/// folder. Windows device names (`NUL`, `COM1`, ..) are refused as well,
/// because `NUL.md` still names the device there.
fn is_safe_page_id(video_id: &str) -> bool {
    const DEVICE_NAMES: [&str; 4] = ["CON", "PRN", "AUX", "NUL"];
    let upper = video_id.to_ascii_uppercase();
    let is_device = DEVICE_NAMES.contains(&upper.as_str())
        || ((upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.len() == 4
            && upper.ends_with(|c: char| c.is_ascii_digit()));
    !video_id.is_empty()
        && video_id.len() <= 64
        && video_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        && !is_device
}

/// Lower-cased so that on a case-insensitive file system a page whose name on
/// disk differs only in case from the name just written is never taken for stale.
fn page_key(dir: &Path, file_name: &str) -> PathBuf {
    dir.join(file_name.to_lowercase())
}

/// The YAML frontmatter and the rest of a page, if the page starts with one.
fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("---\n")?;
    let end = rest.find("\n---\n")?;
    Some((&rest[..end], &rest[end + "\n---\n".len()..]))
}

/// True only for pages this generator wrote: a sub-page carries the
/// `generated_by: ytc` frontmatter line, a root page starts with the fixed
/// head. Anything else (user notes, a note that merely looks like a generated
/// page) is left alone.
fn is_generated_page(kind: PageKind, raw: &str) -> bool {
    let text = raw.trim_start_matches('\u{feff}').replace("\r\n", "\n");
    if let Some((fm, _)) = split_frontmatter(&text) {
        if fm.lines().any(|line| line.trim() == GENERATED_MARKER) {
            return true;
        }
    }
    kind == PageKind::Index && text.starts_with(INDEX_HEAD)
}

/// A folder under `<wiki>/.ytc-pruned/` named after the current UTC time that
/// does not exist yet. It is created when the first page is moved into it.
fn new_pruned_dir(wiki_root: &Path) -> PathBuf {
    let base = wiki_root.join(PRUNED_DIR);
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let mut candidate = base.join(&stamp);
    let mut n = 2;
    // Two prune runs within the same second must not share a folder.
    while candidate.exists() {
        candidate = base.join(format!("{}-{}", stamp, n));
        n += 1;
    }
    candidate
}

/// Moves a stale page to `<pruned_dir>/<its subfolder>/<its name>`.
fn move_to_pruned(page: &Path, pruned_dir: &Path) -> Result<()> {
    let file_name = page.file_name().context("page has no file name")?;
    let subfolder = page
        .parent()
        .and_then(Path::file_name)
        .context("page has no parent folder")?;
    let target_dir = pruned_dir.join(subfolder);
    fs::create_dir_all(&target_dir).with_context(|| format!("Failed to create {:?}", target_dir))?;
    let target = target_dir.join(file_name);
    // rename replaces an existing file on some platforms; never lose one that way.
    if target.exists() {
        anyhow::bail!("{:?} already exists", target);
    }
    fs::rename(page, &target).with_context(|| format!("Failed to move to {:?}", target))?;
    Ok(())
}

/// Generated `*.md` files directly in `dir` that this run did not write.
fn find_stale_pages(dir: &Path, kind: PageKind, writer: &PageWriter) -> Result<Vec<PathBuf>> {
    let mut stale = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("Failed to list wiki folder {:?}", dir))? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if writer.wrote(dir, file_name) {
            continue;
        }
        // Unreadable or non-UTF-8 files are not recognisably ours: keep them.
        match fs::read_to_string(&path) {
            Ok(content) if is_generated_page(kind, &content) => stale.push(path),
            _ => {}
        }
    }
    stale.sort();
    Ok(stale)
}

/// A category of the user's that holds kept videos, with the file name of its page.
struct CategoryPage<'a> {
    category: Category,
    slug: String,
    videos: Vec<&'a KeptVideo>,
}

/// All spellings of a topic that map to the same page file, merged.
struct MergedTopic {
    name: String,
    slug: String,
    videos: Vec<(Video, InsightResult)>,
}

#[derive(Debug, Clone, Default)]
pub struct WikiStats {
    pub topics_count: usize,
    pub creators_count: usize,
    pub insights_count: usize,
    /// Categories of the user's that hold kept videos: one page each.
    pub categories_count: usize,
    pub output_dir: PathBuf,
    /// Generated pages under insights/, topics/, creators/, categories/ that
    /// correspond to nothing in the database any more (found before pruning).
    pub stale_pages: usize,
    /// How many of the stale pages were moved out of the wiki into
    /// `pruned_to`; 0 unless pruning was requested.
    pub pruned_pages: usize,
    /// Folder (`<wiki>/.ytc-pruned/<UTC timestamp>`) that received the pruned
    /// pages, in their `insights/`, `topics/`, `creators/`, `categories/` subfolders. None
    /// when nothing was moved. Pruned pages can be restored from here.
    pub pruned_to: Option<PathBuf>,
    /// Pages that were not written because a file of the same name exists
    /// that ytc did not generate. Each one is named in a warning on stderr.
    pub skipped_user_files: usize,
}

pub struct WikiGenerator;

impl WikiGenerator {
    /// Writes all pages and reports stale ones without moving anything.
    pub fn generate<P: AsRef<Path>>(db: &Database, output_dir: P) -> Result<WikiStats> {
        Self::generate_with_options(db, output_dir, false)
    }

    /// Writes all pages. Generated pages of videos, topics, creators or
    /// categories that are gone from the database are counted in `WikiStats::stale_pages` and,
    /// with `prune`, moved to `<wiki>/.ytc-pruned/<UTC timestamp>/` (never
    /// deleted: a stale page can be the only copy of an analysis). Pruning is
    /// refused when the database has no kept insights at all. Files that are
    /// not recognisably generated are neither moved nor overwritten.
    pub fn generate_with_options<P: AsRef<Path>>(db: &Database, output_dir: P, prune: bool) -> Result<WikiStats> {
        let out = output_dir.as_ref();
        let topics_dir = out.join("topics");
        let creators_dir = out.join("creators");
        let insights_dir = out.join("insights");
        let categories_dir = out.join("categories");

        fs::create_dir_all(&topics_dir)?;
        fs::create_dir_all(&creators_dir)?;
        fs::create_dir_all(&insights_dir)?;

        let mut all_kept = db.get_all_kept_videos_with_details()?;
        // A video id becomes a file name and a link target. One that is not a
        // plain id (possible only in an imported or hand-edited database) gets
        // no page and no link, so nothing is written outside the wiki folder.
        all_kept.retain(|(video, _, _)| {
            let safe = is_safe_page_id(&video.id);
            if !safe {
                eprintln!(
                    "Warning: video id {:?} cannot be used as a file name; no wiki page is written for it.",
                    video.id
                );
            }
            safe
        });
        let all_channels = db.list_channels_with_health()?;
        let all_topics = merge_topics_by_slug(db)?;
        let creator_slugs = build_creator_slugs(&all_channels);
        let category_pages = build_category_pages(db.list_categories()?, &all_kept);
        let category_slugs: HashMap<&str, &str> =
            category_pages.iter().map(|page| (page.category.name.as_str(), page.slug.as_str())).collect();
        let insight_meta = db.list_insight_meta()?;
        let transcript_coverage = db.list_transcript_coverage()?;
        let mut writer = PageWriter::default();

        // 1. Generate Individual Insight Pages. `all_kept` holds only videos
        // that pass the kept rule: a video that is disqualified, topic-rejected
        // or too short to judge gets no page (an existing one becomes stale).
        for (video, channel, insight) in &all_kept {
            // `all_kept` holds safe ids only (see above).
            let filename = format!("{}.md", video.id);
            let content = Self::render_insight_page(
                video,
                channel,
                insight,
                insight_meta.get(&video.id),
                transcript_coverage.get(&video.id),
                &creator_slugs,
                &category_slugs,
            );
            writer.write(&insights_dir, &filename, PageKind::Insight, &content)?;
        }

        // 2. Generate Topic Pages (one per slug, so spelling variants share a page)
        for topic in &all_topics {
            let filename = format!("{}.md", topic.slug);
            let content = Self::render_topic_page(&topic.name, &topic.videos);
            writer.write(&topics_dir, &filename, PageKind::Topic, &content)?;
        }

        // 3. Generate Creator Pages
        for ch in &all_channels {
            let filename = format!("{}.md", creator_slug(&creator_slugs, &ch.id));
            let content = Self::render_creator_page(ch, &all_kept);
            writer.write(&creators_dir, &filename, PageKind::Creator, &content)?;
        }

        // Insight pages of untracked or deleted channels link here.
        let independent: Vec<&KeptVideo> = all_kept
            .iter()
            .filter(|(v, _, _)| !creator_slugs.contains_key(&v.channel_id))
            .collect();
        if !independent.is_empty() {
            let content = Self::render_independent_page(&independent);
            writer.write(&creators_dir, &format!("{}.md", INDEPENDENT_SLUG), PageKind::Creator, &content)?;
        }

        // 4. Generate Category Pages: one per category of the user's that
        // holds kept videos. Without such a category nothing is written and
        // the folder is not created.
        if !category_pages.is_empty() {
            fs::create_dir_all(&categories_dir)?;
        }
        for page in &category_pages {
            let filename = format!("{}.md", page.slug);
            let content = Self::render_category_page(page, &creator_slugs);
            writer.write(&categories_dir, &filename, PageKind::Category, &content)?;
        }

        // 5. Generate Main Index (Home.md and index.md)
        let index_content =
            Self::render_index_page(&all_channels, &all_topics, &all_kept, &creator_slugs, &category_pages);
        writer.write(out, "index.md", PageKind::Index, &index_content)?;
        writer.write(out, "Home.md", PageKind::Index, &index_content)?;

        // 6. Synchronize: pages of videos, topics, creators and categories
        // that left the database (a removed category, or one whose last kept
        // video is gone). Counted always, moved out of the wiki only on request.
        let mut stale: Vec<PathBuf> = Vec::new();
        for (dir, kind) in [
            (&insights_dir, PageKind::Insight),
            (&topics_dir, PageKind::Topic),
            (&creators_dir, PageKind::Creator),
            (&categories_dir, PageKind::Category),
        ] {
            // categories/ exists only once a category page was written.
            if dir.is_dir() {
                stale.extend(find_stale_pages(dir, kind, &writer)?);
            }
        }

        let mut pruned_pages = 0;
        let mut pruned_to = None;
        if prune && !stale.is_empty() {
            // Staleness is relative to the database that was opened. An empty
            // or wrong one makes every generated page look stale, and those
            // pages may be the only copy of their analyses.
            if all_kept.is_empty() {
                anyhow::bail!(
                    "Refusing to prune: the database has no kept insights, so all {} generated page(s) under {:?} count as stale. This usually means an empty or different database was opened (check --db). Nothing was moved or deleted.",
                    stale.len(),
                    out
                );
            }
            let pruned_dir = new_pruned_dir(out);
            for path in &stale {
                match move_to_pruned(path, &pruned_dir) {
                    Ok(()) => pruned_pages += 1,
                    Err(e) => eprintln!("Warning: could not move stale wiki page {:?}: {:#}", path, e),
                }
            }
            if pruned_pages > 0 {
                pruned_to = Some(pruned_dir);
            }
        }

        Ok(WikiStats {
            topics_count: all_topics.len(),
            creators_count: all_channels.len(),
            insights_count: all_kept.len(),
            categories_count: category_pages.len(),
            output_dir: out.to_path_buf(),
            stale_pages: stale.len(),
            pruned_pages,
            pruned_to,
            skipped_user_files: writer.skipped.len(),
        })
    }

    fn render_index_page(
        channels: &[crate::models::ChannelHealth],
        topics: &[MergedTopic],
        kept_videos: &[KeptVideo],
        creator_slugs: &HashMap<String, String>,
        category_pages: &[CategoryPage],
    ) -> String {
        let mut s = String::new();
        s.push_str("# 🌌 YTC Knowledge Wiki\n\n");
        s.push_str("> Autonomous YouTube intelligence library: slop & sponsors stripped, high-signal insights preserved.\n\n");

        // The user's categories that hold kept videos; no section without any.
        if !category_pages.is_empty() {
            s.push_str("## 🗂️ Categories\n\n");
            for page in category_pages {
                s.push_str(&format!(
                    "- **[[categories/{}|{}]]** ({})",
                    page.slug,
                    page.category.name,
                    page.videos.len()
                ));
                if let Some(description) = &page.category.description {
                    s.push_str(&format!(": {}", description));
                }
                s.push('\n');
            }
            s.push('\n');
        }

        // Quick Stats
        let total_kept = kept_videos.len();
        let total_channels = channels.len();
        let total_topics = topics.len();
        let scored: Vec<f64> = kept_videos.iter().map(|(_, _, i)| i.slop_score as f64).collect();
        let avg_slop: f64 = if !scored.is_empty() {
            scored.iter().sum::<f64>() / scored.len() as f64
        } else {
            0.0
        };

        s.push_str("## 📊 Knowledge Base Stats\n\n");
        s.push_str(&format!("- **Total High-Signal Insights:** {}\n", total_kept));
        s.push_str(&format!("- **Tracked Channels:** {}\n", total_channels));
        s.push_str(&format!("- **Indexed Concepts/Topics:** {}\n", total_topics));
        s.push_str(&format!("- **Avg Slop Score of Library:** {:.1}/100\n\n", avg_slop));

        // Topic Navigation
        s.push_str("## 🧭 Concepts & Topics\n\n");
        s.push_str("| Concept / Topic | Mentions | Wiki Link |\n");
        s.push_str("| :--- | :--- | :--- |\n");
        for topic in topics {
            // An unescaped '|' ends the table cell, also inside a wikilink.
            s.push_str(&format!(
                "| **{}** | {} | [[topics/{}\\|Explore Topic]] |\n",
                table_cell(&topic.name), topic.videos.len(), topic.slug
            ));
        }
        s.push_str("\n");

        // Creator Directory
        s.push_str("## 🎙️ Creator & Channel Directory\n\n");
        s.push_str("| Creator / Channel | Status | Signal / Slop Avg | Kept Insights | Channel Wiki |\n");
        s.push_str("| :--- | :--- | :--- | :--- | :--- |\n");
        for ch in channels {
            let slug = creator_slug(creator_slugs, &ch.id);
            let status_badge = match ch.status.as_str() {
                "high_signal" => "🟢 High Signal",
                "muted" => "🔴 Muted",
                _ => "🔵 Active",
            };
            let slop_display = ch.avg_slop_score
                .map(|score| format!("{:.0}/100", score))
                .unwrap_or_else(|| "N/A".to_string());

            s.push_str(&format!(
                "| **{}** | {} | {} | {} / {} | [[creators/{}\\|View Creator]] |\n",
                table_cell(&ch.title), status_badge, slop_display, ch.kept_count, ch.total_evaluated, slug
            ));
        }
        s.push_str("\n");

        // Recent High-Signal Insights
        s.push_str("## ⚡ Latest High-Signal Insights\n\n");
        for (video, channel, insight) in kept_videos.iter().take(6) {
            let ch_slug = creator_slug(creator_slugs, &video.channel_id);
            let category_badge = insight.category.as_deref().map(|name| format!("[{}] ", name)).unwrap_or_default();
            s.push_str(&format!(
                "- {}**[[insights/{}|{}]]** by [[creators/{}|{}]] (Slop Score: **{}**)\n  > {}\n",
                category_badge, video.id, video.title, ch_slug, channel.title, insight.score_label(), insight.summary
            ));
        }
        s.push_str("\n---\n*Generated by YTC Autonomous Crawler & Curator.*\n");
        s
    }

    fn render_category_page(page: &CategoryPage, creator_slugs: &HashMap<String, String>) -> String {
        let mut s = String::new();
        s.push_str(GENERATED_FRONTMATTER);
        s.push_str(&format!("{}\n\n", HOME_CRUMB));
        s.push_str(&format!("{}{}\n\n", CATEGORY_HEADING_PREFIX, page.category.name));
        if let Some(description) = &page.category.description {
            s.push_str(&format!("> {}\n\n", description));
        }
        s.push_str(&format!("Kept videos in this category: **{}**\n\n", page.videos.len()));

        s.push_str("## 📚 Curated Analyses\n\n");
        for (video, channel, insight) in page.videos.iter().copied() {
            let ch_slug = creator_slug(creator_slugs, &video.channel_id);
            s.push_str(&format!("### [[../insights/{}|{}]]\n\n", video.id, video.title));
            s.push_str(&format!("- **Creator:** [[../creators/{}|{}]]\n", ch_slug, channel.title));
            s.push_str(&format!("- **Slop Score:** {} ({})\n", insight.score_label(), insight.quality_rating));
            s.push_str(&format!("- **Executive Summary:** {}\n\n", insight.summary));

            if !insight.key_insights.is_empty() {
                s.push_str("**Core Findings:**\n");
                for (idx, item) in insight.key_insights.iter().take(3).enumerate() {
                    s.push_str(&format!("{}. {}\n", idx + 1, item));
                }
                s.push_str("\n");
            }

            if !insight.takeaways.is_empty() {
                s.push_str("**Takeaways:**\n");
                for t in insight.takeaways.iter().take(2) {
                    s.push_str(&format!("- {}\n", t));
                }
                s.push_str("\n");
            }

            s.push_str("---\n\n");
        }

        s
    }

    fn render_insight_page(
        video: &Video,
        channel: &Channel,
        insight: &InsightResult,
        meta: Option<&InsightMeta>,
        transcript: Option<&TranscriptCoverage>,
        creator_slugs: &HashMap<String, String>,
        category_slugs: &HashMap<&str, &str>,
    ) -> String {
        let ch_slug = creator_slug(creator_slugs, &video.channel_id);
        let sections = meta.and_then(|m| m.sections_total);
        let transcript_source = transcript.and_then(|t| t.source.as_deref());
        let mut s = String::new();

        // Obsidian / Foam YAML frontmatter
        s.push_str("---\n");
        s.push_str(GENERATED_MARKER);
        s.push('\n');
        s.push_str(&format!("title: \"{}\"\n", video.title.replace('"', "\\\"")));
        s.push_str(&format!("video_id: \"{}\"\n", video.id));
        s.push_str(&format!("channel_id: \"{}\"\n", video.channel_id));
        s.push_str(&format!("channel_title: \"{}\"\n", channel.title.replace('"', "\\\"")));
        if let Some(ref category) = insight.category {
            s.push_str(&format!("category: \"{}\"\n", category.replace('"', "\\\"")));
        }
        s.push_str(&format!("slop_score: {}\n", insight.slop_score));
        s.push_str(&format!("quality_rating: \"{}\"\n", insight.quality_rating));
        s.push_str(&format!("published_at: \"{}\"\n", video.published_at));
        s.push_str(&format!("analyzer: \"{}\"\n", insight.analyzer_type));
        if let Some(total) = sections {
            s.push_str(&format!("sections_total: {}\n", total));
        }
        if let Some(source) = transcript_source {
            s.push_str(&format!("transcript_source: \"{}\"\n", source));
        }
        s.push_str("topics:\n");
        for t in &insight.topics {
            s.push_str(&format!("  - \"{}\"\n", t));
        }
        s.push_str(&format!("url: \"{}\"\n", video.url));
        s.push_str("---\n\n");

        // Breadcrumbs & Header
        s.push_str(&format!("{}{}|{}]]\n\n", INSIGHT_CRUMB_PREFIX, ch_slug, channel.title));
        s.push_str(&format!("# {}\n\n", video.title));

        if let Some(category) = insight.category.as_deref() {
            // Linked when the category has a page; a name without one is shown plainly.
            match category_slugs.get(category) {
                Some(slug) => s.push_str(&format!("- **Category:** [[../categories/{}|{}]]\n", slug, category)),
                None => s.push_str(&format!("- **Category:** {}\n", category)),
            }
        }
        s.push_str(&format!("- **Original Video:** [{}]({})\n", video.url, video.url));
        s.push_str(&format!("- **Creator:** [[../creators/{}|{}]]\n", ch_slug, channel.title));
        // The score measures substance (density, concreteness, padding, bait), never the subject.
        let score_label = match meta {
            Some(m) => m.score_label(insight),
            None => insight.score_label(),
        };
        s.push_str(&format!("- **Quality Rating:** {} (Slop Score: **{}**)\n", insight.quality_rating, score_label));
        s.push_str(&format!("- **Curator Engine:** `{}`\n", insight.analyzer_type));
        // How the analysis read the transcript. No line when no analysis ran.
        if let Some(note) = meta.and_then(InsightMeta::whole_transcript_note) {
            s.push_str(&format!("- **Read:** {}\n", note));
        }
        if let Some(label) = transcript.and_then(TranscriptCoverage::source_label) {
            match transcript.and_then(TranscriptCoverage::video_coverage_pct) {
                Some(pct) => s.push_str(&format!("- **Transcript Source:** {} (reaches {}% of the video)\n", label, pct)),
                None => s.push_str(&format!("- **Transcript Source:** {}\n", label)),
            }
        }
        s.push('\n');

        // Executive Summary
        s.push_str("## 📋 Executive Summary\n\n");
        s.push_str(&insight.summary);
        s.push_str("\n\n");

        // Core Insights
        s.push_str("## 💡 Core Insights\n\n");
        for (idx, item) in insight.key_insights.iter().enumerate() {
            s.push_str(&format!("{}. {}\n", idx + 1, item));
        }
        s.push_str("\n");

        // Actionable Takeaways
        if !insight.takeaways.is_empty() {
            s.push_str("## 🛠️ Actionable Takeaways\n\n");
            for (idx, item) in insight.takeaways.iter().enumerate() {
                s.push_str(&format!("{}. {}\n", idx + 1, item));
            }
            s.push_str("\n");
        }

        // Notable Quotes
        if !insight.quotes.is_empty() {
            s.push_str("## 💬 Notable Quotes\n\n");
            for q in &insight.quotes {
                s.push_str(&format!("> \"{}\"\n\n", q));
            }
        }

        // Topics & Tags
        if !insight.topics.is_empty() {
            s.push_str("## 🏷️ Related Concepts & Wiki Links\n\n");
            for t in &insight.topics {
                let slug = sanitize_slug(t);
                s.push_str(&format!("- [[../topics/{}|{}]]\n", slug, t));
            }
            s.push_str("\n");
        }

        // Mentioned Creators
        if !insight.mentioned_creators.is_empty() {
            s.push_str("## 👥 Cited Creators & Entities\n\n");
            for m in &insight.mentioned_creators {
                s.push_str(&format!("- `{}`\n", m));
            }
            s.push_str("\n");
        }

        s
    }

    fn render_topic_page(topic: &str, videos: &[(Video, InsightResult)]) -> String {
        let mut s = String::new();
        s.push_str(GENERATED_FRONTMATTER);
        s.push_str(&format!("{}\n\n", HOME_CRUMB));
        s.push_str(&format!("{}{}\n\n", TOPIC_HEADING_PREFIX, topic));
        // The page lists each video's own summary; nothing is synthesized across videos.
        s.push_str(&format!("> {} insight(s) tagged with this topic.\n\n", videos.len()));

        s.push_str(&format!("{}\n\n", TOPIC_SECTION_HEADING));
        for (video, insight) in videos {
            s.push_str(&format!(
                "### [[../insights/{}|{}]]\n\n",
                video.id, video.title
            ));
            s.push_str(&format!("- **Slop Score:** {} ({})\n", insight.score_label(), insight.quality_rating));
            s.push_str(&format!("- **Summary:** {}\n\n", insight.summary));

            if !insight.key_insights.is_empty() {
                s.push_str("**Key Findings:**\n");
                for (idx, item) in insight.key_insights.iter().take(3).enumerate() {
                    s.push_str(&format!("{}. {}\n", idx + 1, item));
                }
                s.push_str("\n");
            }
        }

        s
    }

    fn render_creator_page(
        channel: &crate::models::ChannelHealth,
        all_videos: &[KeptVideo],
    ) -> String {
        let mut s = String::new();
        s.push_str(GENERATED_FRONTMATTER);
        s.push_str(&format!("{}\n\n", HOME_CRUMB));
        s.push_str(&format!("{}{}\n\n", CREATOR_HEADING_PREFIX, channel.title));

        let status_badge = match channel.status.as_str() {
            "high_signal" => "🟢 High Signal (Trusted Creator)".to_string(),
            // A mute is either the user's toggle or the auto-mute rule; the page cannot tell which.
            "muted" => format!(
                "🔴 Muted (manually, or rolling slop average > {:.0})",
                crate::db::AUTO_MUTE_SLOP_AVG
            ),
            _ => "🔵 Active (Monitoring)".to_string(),
        };
        let slop_display = channel
            .avg_slop_score
            .map(|score| format!("{:.1}/100", score))
            .unwrap_or_else(|| "Pending Evaluations".to_string());

        s.push_str(&format!("- **Channel ID:** `{}`\n", channel.id));
        s.push_str(&format!("- **Health Status:** {}\n", status_badge));
        s.push_str(&format!("- **Rolling Slop Score Average:** **{}**\n", slop_display));
        s.push_str(&format!("- **Evaluated Videos:** {} total ({} kept in Knowledge Base)\n", channel.total_evaluated, channel.kept_count));
        // Where every analysed video of the channel ended up. The lines are disjoint.
        s.push_str(&format!("  - Kept: {}\n", channel.kept_count));
        s.push_str(&format!("  - Below quality cutoff: {}\n", channel.below_cutoff_count));
        s.push_str(&format!("  - Off-topic (fringe / nonsense / politics): {}\n", channel.off_topic_count));
        s.push_str(&format!(
            "  - Not enough information (too short to judge; not counted in the average): {}\n",
            channel.insufficient_count
        ));
        s.push_str(&format!("  - Disqualified by hand: {}\n", channel.excluded_count));
        s.push('\n');

        let channel_videos: Vec<_> = all_videos
            .iter()
            .filter(|(v, _, _)| v.channel_id == channel.id)
            .collect();

        if !channel_videos.is_empty() {
            s.push_str("## 📚 Curated Contributions in Knowledge Base\n\n");
            for (v, _, i) in channel_videos {
                s.push_str(&format!(
                    "### [[../insights/{}|{}]]\n\n",
                    v.id, v.title
                ));
                s.push_str(&format!("- **Slop Score:** {} ({})\n", i.score_label(), i.quality_rating));
                s.push_str(&format!("- **Summary:** {}\n\n", i.summary));
            }
        } else {
            s.push_str("## 📚 Curated Contributions\n\n*No videos currently indexed in high-signal library.*\n");
        }

        s
    }

    fn render_independent_page(videos: &[&KeptVideo]) -> String {
        let mut s = String::new();
        s.push_str(GENERATED_FRONTMATTER);
        s.push_str(&format!("{}\n\n", HOME_CRUMB));
        s.push_str(&format!("{}Independent Analysis\n\n", CREATOR_HEADING_PREFIX));
        s.push_str("> Kept videos whose channel is not on the watchlist: single videos added directly, discovery candidates, and videos of channels that were taken off the watchlist.\n\n");
        s.push_str(&format!("- **Kept in Knowledge Base:** {}\n\n", videos.len()));

        s.push_str("## 📚 Curated Contributions in Knowledge Base\n\n");
        for (v, _, i) in videos {
            s.push_str(&format!(
                "### [[../insights/{}|{}]]\n\n",
                v.id, v.title
            ));
            s.push_str(&format!("- **Slop Score:** {} ({})\n", i.score_label(), i.quality_rating));
            s.push_str(&format!("- **Summary:** {}\n\n", i.summary));
        }

        s
    }

    pub fn print_wiki_summary(stats: &WikiStats) {
        // "Synchronized" only when no stale generated page is left behind.
        let remaining = stats.stale_pages.saturating_sub(stats.pruned_pages);
        let headline = if remaining > 0 {
            "YTC KNOWLEDGE WIKI GENERATED (stale pages remain)".yellow().bold()
        } else if stats.skipped_user_files > 0 {
            "YTC KNOWLEDGE WIKI GENERATED (some pages not written)".yellow().bold()
        } else {
            "YTC KNOWLEDGE WIKI SYNCHRONIZED".green().bold()
        };
        println!("\n{}", "===========================================================================".cyan());
        println!("  {}", headline);
        println!("  Path: {:?}", stats.output_dir);
        if stats.pruned_pages > 0 {
            match &stats.pruned_to {
                Some(dir) => println!(
                    "  • {} stale generated page(s) moved to {:?} (nothing was deleted; move a file back to restore it)",
                    stats.pruned_pages, dir
                ),
                None => println!("  • {} stale generated page(s) moved out of the wiki", stats.pruned_pages),
            }
        }
        if stats.skipped_user_files > 0 {
            println!(
                "  • {} page(s) NOT written: a file of the same name exists that ytc did not generate (see the warnings above)",
                stats.skipped_user_files
            );
        }
        if remaining > 0 {
            println!(
                "  • {} stale generated page(s) left in place (videos, topics, creators or categories that left the library; not pruned)",
                remaining
            );
        }
        println!("---------------------------------------------------------------------------");
        println!("  • {} Concept & Topic Pages generated", stats.topics_count);
        println!("  • {} Creator Profiles indexed", stats.creators_count);
        println!("  • {} High-Signal Video Insights cross-linked", stats.insights_count);
        println!("  • {} Category Pages generated", stats.categories_count);
        println!("  • Bidirectional [[Wikilinks]] enabled for Obsidian / Foam / Logseq");
        println!("{}\n", "===========================================================================".cyan());
    }
}

fn sanitize_slug(name: &str) -> String {
    let mut s = name.to_lowercase();
    s = s.replace('&', "and");
    s = s.replace(|c: char| !c.is_alphanumeric() && c != '-', "-");
    let mut parts: Vec<&str> = s.split('-').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        parts.push("unnamed");
    }
    parts.join("-")
}

fn table_cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

fn creator_slug<'a>(slugs: &'a HashMap<String, String>, channel_id: &str) -> &'a str {
    slugs.get(channel_id).map(String::as_str).unwrap_or(INDEPENDENT_SLUG)
}

/// One page slug per tracked channel. Channels whose titles slugify identically
/// (two channels both titled "Fakechannel") would overwrite each other's page, so
/// every member of such a group gets a suffix taken from its channel id.
fn build_creator_slugs(channels: &[crate::models::ChannelHealth]) -> HashMap<String, String> {
    let mut groups: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for ch in channels {
        groups.entry(sanitize_slug(&ch.title)).or_default().push(ch.id.as_str());
    }

    let mut used: HashSet<String> = HashSet::new();
    used.insert(INDEPENDENT_SLUG.to_string());
    let mut slugs = HashMap::new();

    for (slug, ids) in &groups {
        if ids.len() == 1 && slug != INDEPENDENT_SLUG {
            used.insert(slug.clone());
            slugs.insert(ids[0].to_string(), slug.clone());
        }
    }

    for (slug, ids) in &mut groups {
        if ids.len() == 1 && slug != INDEPENDENT_SLUG {
            continue;
        }
        // Sorted so the result does not depend on the listing order.
        ids.sort_unstable();
        for id in ids.iter() {
            let id_slug = sanitize_slug(id);
            let chars: Vec<char> = id_slug.chars().collect();
            let tail: String = chars[chars.len().saturating_sub(6)..].iter().collect();
            let mut candidate = format!("{}-{}", slug, tail.trim_matches('-'));
            if used.contains(&candidate) {
                candidate = format!("{}-{}", slug, id_slug);
            }
            let mut n = 2;
            while used.contains(&candidate) {
                candidate = format!("{}-{}-{}", slug, id_slug, n);
                n += 1;
            }
            used.insert(candidate.clone());
            slugs.insert(id.to_string(), candidate);
        }
    }

    slugs
}

/// The pages of the categories that hold kept videos, ordered by name. Two
/// names that slugify identically ("Art & Craft" / "Art and Craft") would
/// share a file, so the later one gets a number appended.
fn build_category_pages(categories: Vec<Category>, kept: &[KeptVideo]) -> Vec<CategoryPage<'_>> {
    let mut used: HashSet<String> = HashSet::new();
    let mut pages = Vec::new();
    for category in categories {
        let videos: Vec<&KeptVideo> =
            kept.iter().filter(|(_, _, i)| i.category.as_deref() == Some(category.name.as_str())).collect();
        if videos.is_empty() {
            continue;
        }
        let base = sanitize_slug(&category.name);
        let mut slug = base.clone();
        let mut n = 2;
        while !used.insert(slug.clone()) {
            slug = format!("{}-{}", base, n);
            n += 1;
        }
        pages.push(CategoryPage { category, slug, videos });
    }
    pages
}

/// Topic names that differ only in case or punctuation ("Chain of Thought" /
/// "Chain-of-Thought") share one file name. Merging them keeps every video on
/// the page instead of only those of the spelling written last.
fn merge_topics_by_slug(db: &Database) -> Result<Vec<MergedTopic>> {
    let mut merged: BTreeMap<String, MergedTopic> = BTreeMap::new();

    // list_all_topics is ordered by count, so the first spelling seen is the most common one.
    for (name, _count) in db.list_all_topics()? {
        let slug = sanitize_slug(&name);
        let videos = db.get_topic_videos(&name)?;
        let entry = merged.entry(slug.clone()).or_insert_with(|| MergedTopic {
            name: name.clone(),
            slug,
            videos: Vec::new(),
        });
        for (video, insight) in videos {
            // No page is written for such an id, so no topic page links to it.
            if !is_safe_page_id(&video.id) {
                continue;
            }
            if !entry.videos.iter().any(|(v, _)| v.id == video.id) {
                entry.videos.push((video, insight));
            }
        }
    }

    let mut topics: Vec<MergedTopic> = merged.into_values().filter(|t| !t.videos.is_empty()).collect();
    for t in &mut topics {
        t.videos.sort_by(|(va, ia), (vb, ib)| {
            ia.slop_score
                .cmp(&ib.slop_score)
                .then_with(|| vb.published_at.cmp(&va.published_at))
        });
    }
    topics.sort_by(|a, b| b.videos.len().cmp(&a.videos.len()).then_with(|| a.name.cmp(&b.name)));
    Ok(topics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(id: &str, title: &str) -> Channel {
        Channel {
            id: id.to_string(),
            title: title.to_string(),
            url: format!("https://example.invalid/channel/{}", id),
            status: "active".to_string(),
            rolling_slop_score: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            last_crawled_at: None,
        }
    }

    fn video(id: &str, channel_id: &str) -> Video {
        Video {
            id: id.to_string(),
            channel_id: channel_id.to_string(),
            title: format!("Video {}", id),
            url: format!("https://example.invalid/watch?v={}", id),
            published_at: "2026-02-01T00:00:00Z".to_string(),
            status: "processed".to_string(),
            created_at: "2026-02-01T00:00:00Z".to_string(),
        }
    }

    fn insight(video_id: &str, keep: bool, topics: &[&str]) -> InsightResult {
        InsightResult {
            video_id: video_id.to_string(),
            slop_score: 20,
            is_worth_keeping: keep,
            quality_rating: "High Signal".to_string(),
            summary: format!("Summary of {}", video_id),
            key_insights: vec!["A finding".to_string()],
            takeaways: vec![],
            quotes: vec![],
            topics: topics.iter().map(|t| t.to_string()).collect(),
            mentioned_creators: vec![],
            analyzer_type: "llm:test-model".to_string(),
            category: None,
        }
    }

    /// Database and wiki folder inside one temp directory.
    fn fixture() -> (tempfile::TempDir, Database, PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("db").join("ytc.db")).expect("open test db");
        let wiki = dir.path().join("wiki");
        db.upsert_channel(&channel("UC1", "Test Creator")).expect("channel");
        for (id, topics) in [("vidA", ["Sourdough", "Rye Flour"]), ("vidB", ["Sourdough", "Ovens"])] {
            db.upsert_video(&video(id, "UC1")).expect("video");
            db.save_insight(&insight(id, true, &topics)).expect("insight");
        }
        (dir, db, wiki)
    }

    fn files_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("read dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn generated_pages_are_recognised_and_user_notes_are_not() {
        // The marker line in the frontmatter.
        assert!(is_generated_page(PageKind::Topic, "---\ngenerated_by: ytc\n---\n\nanything"));
        assert!(is_generated_page(PageKind::Insight, "---\r\ngenerated_by: ytc\r\ntitle: \"x\"\r\n---\r\n\r\nbody"));

        // A user's note that merely looks like a generated page is not one.
        assert!(!is_generated_page(PageKind::Topic, "⬅️ [[../index|Wiki Home]]\n\n# Concept: Sourdough\n\n> text"));
        assert!(!is_generated_page(PageKind::Topic, "⬅️ [[../index|Wiki Home]]\n\n# Concept: Sourdough\n\nMy own notes about sourdough.\n\n## Reading list\n"));
        assert!(!is_generated_page(PageKind::Creator, "⬅️ [[../index|Wiki Home]]\n\n# Creator Profile: X\n\nWhy I follow this channel.\n"));

        // Root pages are recognised by their fixed head.
        assert!(is_generated_page(PageKind::Index, "# 🌌 YTC Knowledge Wiki\n\n> Autonomous YouTube intelligence library\n"));
        assert!(!is_generated_page(PageKind::Index, "# Home\n\nMy vault.\n"));

        assert!(is_generated_page(PageKind::Category, "---\ngenerated_by: ytc\n---\n\n⬅️ [[../index|Wiki Home]]\n\n# Category: Baking\n"));
        assert!(!is_generated_page(PageKind::Category, "⬅️ [[../index|Wiki Home]]\n\n# Category: Baking\n\nMy own baking notes.\n"));
        assert!(!is_generated_page(PageKind::Category, "# Baking\n\nRecipes I want to try.\n"));

        // User files.
        assert!(!is_generated_page(PageKind::Insight, "# My notes\n\nSome thoughts about video_id: \"abc\"\n"));
        assert!(!is_generated_page(PageKind::Insight, "---\ntitle: \"Mine\"\ntags: [note]\n---\n\n# Mine\n"));
        assert!(!is_generated_page(PageKind::Topic, "# Concept: Sourdough\n\nmy own page"));
        // The marker in the body, not in the frontmatter, proves nothing.
        assert!(!is_generated_page(PageKind::Topic, "# Notes\n\ngenerated_by: ytc\n"));
        assert!(!is_generated_page(PageKind::Topic, ""));
    }

    #[test]
    fn every_page_the_generator_writes_is_recognised_as_its_own() {
        let (_dir, db, wiki) = fixture();
        let stats = WikiGenerator::generate(&db, &wiki).expect("generate");
        assert_eq!((stats.insights_count, stats.topics_count, stats.creators_count), (2, 3, 1));
        assert_eq!((stats.stale_pages, stats.pruned_pages), (0, 0));

        for (sub, kind, expected) in [
            ("insights", PageKind::Insight, 2),
            ("topics", PageKind::Topic, 3),
            ("creators", PageKind::Creator, 1),
        ] {
            let mut count = 0;
            for entry in fs::read_dir(wiki.join(sub)).expect("read dir") {
                let path = entry.expect("entry").path();
                let content = fs::read_to_string(&path).expect("read page");
                assert!(is_generated_page(kind, &content), "{:?} not recognised", path);
                assert!(content.starts_with("---\ngenerated_by: ytc\n"), "{:?} lacks the marker", path);
                count += 1;
            }
            assert_eq!(count, expected, "pages in {}", sub);
        }

        // Root pages too, otherwise the next run would refuse to refresh them.
        for (name, kind) in [("index.md", PageKind::Index), ("Home.md", PageKind::Index)] {
            let content = fs::read_to_string(wiki.join(name)).expect("read root page");
            assert!(is_generated_page(kind, &content), "{} not recognised", name);
        }
        let stats = WikiGenerator::generate(&db, &wiki).expect("second run");
        assert_eq!(stats.skipped_user_files, 0);
    }

    #[test]
    fn files_not_written_by_ytc_are_never_overwritten() {
        let (_dir, db, wiki) = fixture();
        fs::create_dir_all(wiki.join("topics")).expect("dir");
        fs::create_dir_all(wiki.join("insights")).expect("dir");
        let user_files = [
            ("Home.md", "# Home\n\nMy vault start page.\n"),
            ("topics/sourdough.md", "⬅️ [[../index|Wiki Home]]\n\n# Concept: Sourdough\n\nMy own notes about sourdough.\n"),
            ("insights/vidA.md", "# vidA\n\nWatched it, notes to self.\n"),
        ];
        for (name, content) in user_files {
            fs::write(wiki.join(name), content).expect("write user file");
        }

        // Twice: the files must survive every run, not only the first.
        for _ in 0..2 {
            let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("generate");
            assert_eq!(stats.skipped_user_files, 3);
            assert_eq!((stats.stale_pages, stats.pruned_pages), (0, 0));
            assert_eq!(stats.pruned_to, None);
            for (name, content) in user_files {
                assert_eq!(fs::read_to_string(wiki.join(name)).expect("read"), content, "{} was changed", name);
            }
        }
        // Everything else is still generated.
        for generated in ["index.md", "topics/rye-flour.md", "insights/vidB.md", "creators/test-creator.md"] {
            assert!(wiki.join(generated).exists(), "{} missing", generated);
        }
    }

    #[test]
    fn stale_pages_are_reported_and_only_pruned_on_request() {
        let (_dir, db, wiki) = fixture();
        WikiGenerator::generate(&db, &wiki).expect("first run");

        let stale_insight = wiki.join("insights").join("goneVideo.md");
        let stale_topic = wiki.join("topics").join("gone-topic.md");
        let user_note = wiki.join("insights").join("my-notes.md");
        let user_topic = wiki.join("topics").join("reading-list.md");
        let other_file = wiki.join("creators").join("avatar.txt");
        let gone_page = format!("---\ngenerated_by: ytc\nvideo_id: \"goneVideo\"\n---\n\n# Gone\n");
        fs::write(&stale_insight, &gone_page).expect("write");
        fs::write(&stale_topic, format!("{}{}\n\n{}Gone\n", GENERATED_FRONTMATTER, HOME_CRUMB, TOPIC_HEADING_PREFIX)).expect("write");
        fs::write(&user_note, "# My notes\n\nNot generated.\n").expect("write");
        fs::write(&user_topic, "---\ntitle: \"Reading list\"\n---\n\n- a book\n").expect("write");
        fs::write(&other_file, "generated_by: ytc").expect("write");

        // Default: count, delete nothing.
        let stats = WikiGenerator::generate(&db, &wiki).expect("report run");
        assert_eq!((stats.stale_pages, stats.pruned_pages), (2, 0));
        assert_eq!(stats.pruned_to, None);
        assert!(stale_insight.exists() && stale_topic.exists());
        assert!(!wiki.join(PRUNED_DIR).exists());

        // Prune: exactly the two stale generated pages leave the wiki folders,
        // and they are moved, not deleted.
        let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("prune run");
        assert_eq!((stats.stale_pages, stats.pruned_pages), (2, 2));
        assert!(!stale_insight.exists() && !stale_topic.exists());
        assert!(user_note.exists() && user_topic.exists() && other_file.exists());
        for current in ["insights/vidA.md", "insights/vidB.md", "topics/sourdough.md", "topics/rye-flour.md", "topics/ovens.md", "creators/test-creator.md"] {
            assert!(wiki.join(current).exists(), "{} must survive pruning", current);
        }

        let pruned_to = stats.pruned_to.expect("the trash folder is reported");
        assert_eq!(pruned_to.parent(), Some(wiki.join(PRUNED_DIR).as_path()));
        assert_eq!(files_in(&wiki.join(PRUNED_DIR)).len(), 1);
        assert_eq!(files_in(&pruned_to), ["insights", "topics"]);
        assert_eq!(
            fs::read_to_string(pruned_to.join("insights").join("goneVideo.md")).expect("moved insight"),
            gone_page
        );
        assert!(pruned_to.join("topics").join("gone-topic.md").exists());

        // The trash folder is not rescanned, and an idle prune creates none.
        let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("clean run");
        assert_eq!((stats.stale_pages, stats.pruned_pages), (0, 0));
        assert_eq!(stats.pruned_to, None);
        assert_eq!(files_in(&wiki.join(PRUNED_DIR)).len(), 1);
    }

    #[test]
    fn prune_is_refused_when_the_database_has_no_kept_insights() {
        let (dir, db, wiki) = fixture();
        WikiGenerator::generate(&db, &wiki).expect("first run");

        // The same wiki, seen through an empty database (wrong --db, fresh file).
        let empty = Database::open(dir.path().join("other").join("ytc.db")).expect("open empty db");
        let stats = WikiGenerator::generate(&empty, &wiki).expect("report run");
        assert_eq!((stats.insights_count, stats.stale_pages, stats.pruned_pages), (0, 6, 0));

        let err = WikiGenerator::generate_with_options(&empty, &wiki, true).expect_err("prune must be refused");
        assert!(err.to_string().contains("Refusing to prune"), "{}", err);
        for page in ["insights/vidA.md", "insights/vidB.md", "topics/sourdough.md", "topics/rye-flour.md", "topics/ovens.md", "creators/test-creator.md"] {
            assert!(wiki.join(page).exists(), "{} must not be touched", page);
        }
        assert!(!wiki.join(PRUNED_DIR).exists());

        // Nothing stale: an empty database is no reason to fail.
        let fresh_wiki = dir.path().join("fresh-wiki");
        let stats = WikiGenerator::generate_with_options(&empty, &fresh_wiki, true).expect("nothing to prune");
        assert_eq!((stats.stale_pages, stats.pruned_pages), (0, 0));
    }

    #[test]
    fn two_prune_runs_never_share_a_trash_folder() {
        let dir = tempfile::tempdir().expect("temp dir");
        let first = new_pruned_dir(dir.path());
        fs::create_dir_all(&first).expect("create");
        let second = new_pruned_dir(dir.path());
        assert_ne!(first, second);
        assert_eq!(second.parent(), first.parent());

        // A file already in the trash is never replaced by a move.
        let page = dir.path().join("topics").join("a.md");
        fs::create_dir_all(first.join("topics")).expect("create");
        fs::create_dir_all(dir.path().join("topics")).expect("create");
        fs::write(first.join("topics").join("a.md"), "first run").expect("write");
        fs::write(&page, "later").expect("write");
        assert!(move_to_pruned(&page, &first).is_err());
        assert!(page.exists());
        assert_eq!(fs::read_to_string(first.join("topics").join("a.md")).expect("read"), "first run");
    }

    #[test]
    fn pages_of_videos_that_left_the_library_become_stale() {
        let (_dir, db, wiki) = fixture();
        WikiGenerator::generate(&db, &wiki).expect("first run");

        // vidB is re-analysed as slop: its page and its only-own topic are stale.
        let mut slop = insight("vidB", false, &["Sourdough", "Ovens"]);
        slop.slop_score = 90;
        db.save_insight(&slop).expect("save");

        let stats = WikiGenerator::generate(&db, &wiki).expect("report run");
        assert_eq!((stats.insights_count, stats.topics_count), (1, 2));
        assert_eq!((stats.stale_pages, stats.pruned_pages), (2, 0));

        let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("prune run");
        assert_eq!(stats.pruned_pages, 2);
        assert!(!wiki.join("insights").join("vidB.md").exists());
        assert!(!wiki.join("topics").join("ovens.md").exists());
        let pruned_to = stats.pruned_to.expect("the trash folder is reported");
        assert!(pruned_to.join("insights").join("vidB.md").exists());
        assert!(pruned_to.join("topics").join("ovens.md").exists());
        assert!(wiki.join("insights").join("vidA.md").exists());
        assert!(wiki.join("topics").join("sourdough.md").exists());
    }

    #[test]
    fn insight_page_has_no_read_or_source_line_when_nothing_is_recorded() {
        let v = video("vidA", "UC1");
        let c = channel("UC1", "Test Creator");
        let i = insight("vidA", true, &["Sourdough"]);
        let slugs: HashMap<String, String> = HashMap::from([("UC1".to_string(), "test-creator".to_string())]);

        for meta in [None, Some(InsightMeta::default())] {
            let page = WikiGenerator::render_insight_page(&v, &c, &i, meta.as_ref(), None, &slugs, &HashMap::new());
            assert!(!page.contains("**Read:**"), "{}", page);
            assert!(!page.contains("Transcript Source"));
            assert!(!page.contains("sections_total"));
            assert!(page.contains("(Slop Score: **20/100**)"));
        }
    }

    #[test]
    fn insight_page_says_that_the_whole_transcript_was_read_and_where_it_came_from() {
        let v = video("vidA", "UC1");
        let c = channel("UC1", "Test Creator");
        let i = insight("vidA", true, &["Sourdough"]);
        let slugs: HashMap<String, String> = HashMap::from([("UC1".to_string(), "test-creator".to_string())]);

        let whole = InsightMeta {
            transcript_chars: Some(41_000),
            sections_total: Some(3),
            topic_verdict: Some("ok".to_string()),
            kept: true,
            ..Default::default()
        };
        let whisper = TranscriptCoverage {
            video_duration_sec: Some(1_800.0),
            covered_until_sec: Some(1_795.0),
            source: Some("whisper".to_string()),
        };
        let page = WikiGenerator::render_insight_page(&v, &c, &i, Some(&whole), Some(&whisper), &slugs, &HashMap::new());
        assert!(page.contains("- **Read:** whole transcript (3 sections)\n"), "{}", page);
        assert!(page.contains("- **Transcript Source:** local Whisper transcription of the audio (reaches 99% of the video)\n"), "{}", page);
        assert!(page.contains("sections_total: 3\n"));
        assert!(page.contains("transcript_source: \"whisper\"\n"));
        assert!(page.contains("(Slop Score: **20/100**)"));
        assert!(is_generated_page(PageKind::Insight, &page));

        // A source without the figures to compare names the source alone.
        let builtin = TranscriptCoverage { source: Some("builtin".to_string()), ..Default::default() };
        let page = WikiGenerator::render_insight_page(&v, &c, &i, Some(&whole), Some(&builtin), &slugs, &HashMap::new());
        assert!(page.contains("- **Read:** whole transcript (3 sections)\n"), "{}", page);
        assert!(page.contains("- **Transcript Source:** YouTube transcript\n"), "{}", page);
    }

    fn in_category(video_id: &str, category: &str, topics: &[&str]) -> InsightResult {
        InsightResult { category: Some(category.to_string()), ..insight(video_id, true, topics) }
    }

    #[test]
    fn without_categories_the_wiki_has_no_category_pages() {
        let (_dir, db, wiki) = fixture();
        let stats = WikiGenerator::generate(&db, &wiki).expect("generate");
        assert_eq!(stats.categories_count, 0);
        assert!(!wiki.join("categories").exists());
        let mut top = files_in(&wiki);
        top.retain(|n| n.ends_with(".md"));
        assert_eq!(top, vec!["Home.md", "index.md"]);

        let index = fs::read_to_string(wiki.join("index.md")).expect("index.md");
        assert!(!index.to_lowercase().contains("categor"), "{}", index);
        assert!(index.contains("- **[[insights/vidA|Video vidA]]** by [[creators/test-creator|Test Creator]]"), "{}", index);
        let page = fs::read_to_string(wiki.join("insights").join("vidA.md")).expect("insight page");
        assert!(!page.to_lowercase().contains("categor"), "{}", page);

        // A defined category without a kept video gets no page either.
        db.add_category("Baking", None).expect("category");
        let stats = WikiGenerator::generate(&db, &wiki).expect("generate");
        assert_eq!(stats.categories_count, 0);
        assert!(!wiki.join("categories").exists());
    }

    #[test]
    fn every_category_with_kept_videos_gets_one_page() {
        let (_dir, db, wiki) = fixture();
        db.add_category("Baking", Some("Bread, pastry and ovens")).expect("category");
        db.add_category("Star Gazing", None).expect("category");
        db.add_category("Unused", None).expect("category");
        db.save_insight(&in_category("vidA", "Baking", &["Sourdough", "Rye Flour"])).expect("insight");
        db.upsert_video(&video("vidC", "UC1")).expect("video");
        db.save_insight(&in_category("vidC", "Baking", &[])).expect("insight");
        db.upsert_video(&video("vidS", "UC1")).expect("video");
        db.save_insight(&in_category("vidS", "Star Gazing", &[])).expect("insight");
        // Not kept: it is on no category page and does not make one.
        db.upsert_video(&video("vidU", "UC1")).expect("video");
        let mut slop = in_category("vidU", "Unused", &[]);
        slop.is_worth_keeping = false;
        db.save_insight(&slop).expect("insight");

        let stats = WikiGenerator::generate(&db, &wiki).expect("generate");
        assert_eq!((stats.insights_count, stats.categories_count, stats.stale_pages), (4, 2, 0));
        assert_eq!(files_in(&wiki.join("categories")), ["baking.md", "star-gazing.md"]);

        let baking = fs::read_to_string(wiki.join("categories").join("baking.md")).expect("category page");
        assert!(is_generated_page(PageKind::Category, &baking));
        assert!(baking.starts_with("---\ngenerated_by: ytc\n---\n\n⬅️ [[../index|Wiki Home]]\n\n# Category: Baking\n\n> Bread, pastry and ovens\n\n"), "{}", baking);
        assert!(baking.contains("Kept videos in this category: **2**"), "{}", baking);
        assert!(baking.contains("### [[../insights/vidA|Video vidA]]"), "{}", baking);
        assert!(baking.contains("### [[../insights/vidC|Video vidC]]"), "{}", baking);
        assert!(baking.contains("- **Creator:** [[../creators/test-creator|Test Creator]]"), "{}", baking);
        for other in ["vidB", "vidS", "vidU"] {
            assert!(!baking.contains(other), "{} is not in Baking", other);
        }
        let stars = fs::read_to_string(wiki.join("categories").join("star-gazing.md")).expect("category page");
        assert!(stars.contains("# Category: Star Gazing\n\nKept videos in this category: **1**"), "{}", stars);

        let index = fs::read_to_string(wiki.join("index.md")).expect("index.md");
        assert!(index.contains("## 🗂️ Categories\n\n- **[[categories/baking|Baking]]** (2): Bread, pastry and ovens\n- **[[categories/star-gazing|Star Gazing]]** (1)\n\n"), "{}", index);
        assert!(!index.contains("Unused"), "{}", index);
        assert!(index.contains("- [Baking] **[[insights/vidA|Video vidA]]**"), "{}", index);
        assert!(index.contains("- **[[insights/vidB|Video vidB]]**"), "{}", index);

        let page = fs::read_to_string(wiki.join("insights").join("vidA.md")).expect("insight page");
        assert!(page.contains("category: \"Baking\"\n"), "{}", page);
        assert!(page.contains("- **Category:** [[../categories/baking|Baking]]\n"), "{}", page);
        let plain = fs::read_to_string(wiki.join("insights").join("vidB.md")).expect("insight page");
        assert!(!plain.to_lowercase().contains("categor"), "{}", plain);

        // The next run refreshes its own pages.
        let stats = WikiGenerator::generate(&db, &wiki).expect("second run");
        assert_eq!((stats.skipped_user_files, stats.stale_pages), (0, 0));
    }

    #[test]
    fn the_page_of_a_removed_category_is_stale_and_pruned_like_any_other() {
        let (_dir, db, wiki) = fixture();
        db.add_category("Baking", None).expect("category");
        db.add_category("Star Gazing", None).expect("category");
        db.save_insight(&in_category("vidA", "Baking", &["Sourdough", "Rye Flour"])).expect("insight");
        db.save_insight(&in_category("vidB", "Star Gazing", &["Sourdough", "Ovens"])).expect("insight");
        WikiGenerator::generate(&db, &wiki).expect("first run");
        let user_note = wiki.join("categories").join("my-list.md");
        fs::write(&user_note, "# Categories I still want\n\n- gardening\n").expect("write");

        // Removing the category keeps the video: only the category page goes.
        db.remove_category("Baking").expect("remove");
        let stats = WikiGenerator::generate(&db, &wiki).expect("report run");
        assert_eq!((stats.insights_count, stats.categories_count), (2, 1));
        assert_eq!((stats.stale_pages, stats.pruned_pages), (1, 0));
        assert!(wiki.join("categories").join("baking.md").exists());

        let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("prune run");
        assert_eq!((stats.stale_pages, stats.pruned_pages), (1, 1));
        assert!(!wiki.join("categories").join("baking.md").exists());
        let pruned_to = stats.pruned_to.expect("the trash folder is reported");
        assert!(pruned_to.join("categories").join("baking.md").exists());
        assert!(wiki.join("categories").join("star-gazing.md").exists());
        assert!(user_note.exists());
        let page = fs::read_to_string(wiki.join("insights").join("vidA.md")).expect("the video keeps its page");
        assert!(!page.to_lowercase().contains("categor"), "{}", page);
        assert!(!fs::read_to_string(wiki.join("index.md")).expect("index").contains("Baking"));

        // A category whose last kept video left the library loses its page too.
        let mut slop = in_category("vidB", "Star Gazing", &["Sourdough", "Ovens"]);
        slop.is_worth_keeping = false;
        db.save_insight(&slop).expect("save");
        let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("prune run");
        assert_eq!(stats.categories_count, 0);
        assert!(!wiki.join("categories").join("star-gazing.md").exists());
        assert!(user_note.exists());
    }

    #[test]
    fn a_users_own_file_in_the_categories_folder_is_never_overwritten() {
        let (_dir, db, wiki) = fixture();
        db.add_category("Baking", None).expect("category");
        db.save_insight(&in_category("vidA", "Baking", &["Sourdough", "Rye Flour"])).expect("insight");
        fs::create_dir_all(wiki.join("categories")).expect("dir");
        let own = "# Baking\n\nRecipes I want to try.\n";
        fs::write(wiki.join("categories").join("baking.md"), own).expect("write");

        let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("generate");
        assert_eq!((stats.skipped_user_files, stats.stale_pages, stats.pruned_pages), (1, 0, 0));
        assert_eq!(fs::read_to_string(wiki.join("categories").join("baking.md")).expect("read"), own);
    }

    #[test]
    fn categories_whose_names_make_the_same_file_name_get_separate_pages() {
        let category = |name: &str| Category {
            name: name.to_string(),
            description: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let kept: Vec<KeptVideo> = ["Art & Craft", "Art and Craft", "Art-and-Craft"]
            .iter()
            .enumerate()
            .map(|(n, name)| {
                let id = format!("vid{}", n);
                (video(&id, "UC1"), channel("UC1", "Test Creator"), in_category(&id, name, &[]))
            })
            .collect();
        let categories = vec![category("Art & Craft"), category("Art and Craft"), category("Art-and-Craft"), category("Empty")];
        let pages = build_category_pages(categories, &kept);
        let slugs: Vec<(&str, &str, usize)> =
            pages.iter().map(|p| (p.category.name.as_str(), p.slug.as_str(), p.videos.len())).collect();
        assert_eq!(
            slugs,
            [("Art & Craft", "art-and-craft", 1), ("Art and Craft", "art-and-craft-2", 1), ("Art-and-Craft", "art-and-craft-3", 1)]
        );
    }

    #[test]
    fn rejected_and_disqualified_videos_get_no_page_and_the_page_they_had_is_pruned() {
        let (_dir, db, wiki) = fixture();
        for (id, topic) in [("vidP", "Elections"), ("vidS", "Shorts"), ("vidD", "Duplicates")] {
            db.upsert_video(&video(id, "UC1")).expect("video");
            db.save_insight(&insight(id, true, &[topic])).expect("insight");
        }
        let stats = WikiGenerator::generate(&db, &wiki).expect("first run");
        assert_eq!((stats.insights_count, stats.stale_pages), (5, 0));

        // A good score does not keep a political video, a too-short one or a
        // video the owner threw out.
        db.set_insight_verdict("vidP", "politics", Some("election commentary")).expect("verdict");
        db.set_insight_info_status("vidS", "insufficient", Some("40 words")).expect("info");
        db.exclude_video("vidD", Some("duplicate")).expect("exclude");

        let stats = WikiGenerator::generate(&db, &wiki).expect("report run");
        assert_eq!(stats.insights_count, 2);
        // Three insight pages and three topic pages that only they carried.
        assert_eq!((stats.stale_pages, stats.pruned_pages), (6, 0));
        for root in ["index.md", "creators/test-creator.md", "topics/sourdough.md"] {
            let text = fs::read_to_string(wiki.join(root)).expect("read");
            for gone in ["vidP", "vidS", "vidD"] {
                assert!(!text.contains(gone), "{} still lists {}", root, gone);
            }
        }
        let creator = fs::read_to_string(wiki.join("creators").join("test-creator.md")).expect("creator page");
        assert!(creator.contains("  - Kept: 2\n"), "{}", creator);
        assert!(creator.contains("  - Below quality cutoff: 0\n"), "{}", creator);
        assert!(creator.contains("  - Off-topic (fringe / nonsense / politics): 1\n"), "{}", creator);
        assert!(creator.contains("  - Not enough information (too short to judge; not counted in the average): 1\n"), "{}", creator);
        assert!(creator.contains("  - Disqualified by hand: 1\n"), "{}", creator);

        let stats = WikiGenerator::generate_with_options(&db, &wiki, true).expect("prune run");
        assert_eq!(stats.pruned_pages, 6);
        let pruned_to = stats.pruned_to.expect("the trash folder is reported");
        for gone in ["vidP", "vidS", "vidD"] {
            assert!(!wiki.join("insights").join(format!("{}.md", gone)).exists());
            assert!(pruned_to.join("insights").join(format!("{}.md", gone)).exists());
        }

        // The undo path: including the video again brings its page back.
        db.include_video("vidD").expect("include");
        let stats = WikiGenerator::generate(&db, &wiki).expect("after include");
        assert_eq!(stats.insights_count, 3);
        assert!(wiki.join("insights").join("vidD.md").exists());
    }

    #[test]
    fn a_video_id_is_only_a_file_name_when_it_is_a_plain_id() {
        for ok in ["AbCdEfGh123", "-AbCdEfGh12", "a_b-C9", "x"] {
            assert!(is_safe_page_id(ok), "{}", ok);
        }
        let too_long = "a".repeat(65);
        for bad in [
            "", ".", "..", "..\\..\\escaped", "../../escaped", "a/b", "a\\b", "C:evil", "a.b", "name.md",
            "id with space", "x');alert(1", "nul", "COM1", "lpt9", "héllo", too_long.as_str(),
        ] {
            assert!(!is_safe_page_id(bad), "{:?}", bad);
        }
        assert!(is_safe_page_id(&"a".repeat(64)));
        // Four letters starting with COM are only a device with a digit.
        assert!(is_safe_page_id("COMA"));
    }

    #[test]
    fn an_imported_id_with_path_parts_never_leaves_the_wiki_folder() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(dir.path().join("db").join("ytc.db")).expect("open test db");
        // Two levels below the temp dir, so "..\..\" would still land inside it.
        let wiki = dir.path().join("deep").join("a").join("wiki");
        db.upsert_channel(&channel("UC1", "Test Creator")).expect("channel");
        let hostile = ["..\\..\\escaped_bs", "../../escaped_fs", "..", "sub/page", "dotted.id"];
        for id in hostile.iter().copied().chain(["vidA"]) {
            db.upsert_video(&video(id, "UC1")).expect("video");
            db.save_insight(&insight(id, true, &["Sourdough"])).expect("insight");
        }

        let stats = WikiGenerator::generate(&db, &wiki).expect("generate");
        // Only the plain id got a page, and it is the only one counted.
        assert_eq!(stats.insights_count, 1);
        assert_eq!(files_in(&wiki.join("insights")), vec!["vidA.md"]);

        // Nothing was created anywhere outside the wiki folder.
        assert_eq!(files_in(&dir.path().join("deep")), vec!["a"]);
        assert_eq!(files_in(&dir.path().join("deep").join("a")), vec!["wiki"]);
        assert_eq!(files_in(dir.path()), vec!["db", "deep"]);
        let mut top = files_in(&wiki);
        top.retain(|n| n.ends_with(".md"));
        assert_eq!(top, vec!["Home.md", "index.md"]);

        // And no page links to a page that was not written.
        let index = fs::read_to_string(wiki.join("index.md")).expect("read");
        assert!(index.contains("insights/vidA|"));
        assert!(!index.contains("escaped") && !index.contains("dotted.id"));
        for page in [wiki.join("topics").join("sourdough.md"), wiki.join("creators").join("test-creator.md")] {
            let text = fs::read_to_string(&page).expect("read");
            assert!(text.contains("../insights/vidA|"), "{:?}", page);
            assert!(!text.contains("escaped") && !text.contains("sub/page") && !text.contains("dotted.id"), "{:?}", page);
        }
    }

    #[test]
    fn creator_page_says_where_every_analysed_video_ended_up() {
        let health = crate::models::ChannelHealth {
            id: "UC1".to_string(),
            title: "Test Creator".to_string(),
            status: "active".to_string(),
            total_evaluated: 13,
            kept_count: 3,
            avg_slop_score: Some(30.0),
            last_crawled_at: None,
            below_cutoff_count: 1,
            off_topic_count: 3,
            insufficient_count: 5,
            excluded_count: 1,
        };
        let page = WikiGenerator::render_creator_page(&health, &[]);
        assert!(page.contains("- **Evaluated Videos:** 13 total (3 kept in Knowledge Base)\n"));
        assert!(page.contains("  - Kept: 3\n"));
        assert!(page.contains("  - Below quality cutoff: 1\n"));
        assert!(page.contains("  - Off-topic (fringe / nonsense / politics): 3\n"));
        assert!(page.contains("  - Not enough information (too short to judge; not counted in the average): 5\n"));
        assert!(page.contains("  - Disqualified by hand: 1\n"));
        assert!(is_generated_page(PageKind::Creator, &page));
    }

    #[test]
    fn an_average_of_zero_is_shown_as_a_score() {
        let health = |avg: Option<f64>| crate::models::ChannelHealth {
            id: "UC1".to_string(),
            title: "Test Creator".to_string(),
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
        let slugs: HashMap<String, String> = HashMap::from([("UC1".to_string(), "test-creator".to_string())]);

        // 0 is the best possible score, not a missing one.
        let page = WikiGenerator::render_creator_page(&health(Some(0.0)), &[]);
        assert!(page.contains("- **Rolling Slop Score Average:** **0.0/100**\n"));
        let index = WikiGenerator::render_index_page(&[health(Some(0.0))], &[], &[], &slugs, &[]);
        assert!(index.contains("| 🟢 High Signal | 0/100 |"));

        let page = WikiGenerator::render_creator_page(&health(None), &[]);
        assert!(page.contains("**Pending Evaluations**"));
        let index = WikiGenerator::render_index_page(&[health(None)], &[], &[], &slugs, &[]);
        assert!(index.contains("| 🟢 High Signal | N/A |"));
    }
}
