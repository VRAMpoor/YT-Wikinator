use anyhow::{anyhow, Context, Result};
use regex::Regex;
use reqwest::Client;
use std::time::Duration;

/// Wall-clock limit for the yt-dlp call in `resolve_channel`. yt-dlp's
/// --socket-timeout limits one socket operation only; with retries or a slow
/// host a request handler would otherwise wait without an end.
const RESOLVE_TIMEOUT: Duration = Duration::from_secs(45);

/// Hosts a channel URL may point to. Anything else is not passed to yt-dlp or
/// fetched: the input comes from a web form.
const YOUTUBE_HOSTS: &[&str] = &["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"];

fn is_youtube_url(url: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| YOUTUBE_HOSTS.contains(&host.as_str()))
}

/// Runs a helper program without blocking an async worker thread and gives up
/// after `limit`; the child is killed then. None when it could not be started
/// or did not finish in time.
async fn output_with_timeout(command: std::process::Command, limit: Duration) -> Option<std::process::Output> {
    let mut command = tokio::process::Command::from(command);
    command.stdin(std::process::Stdio::null()).kill_on_drop(true);
    match tokio::time::timeout(limit, command.output()).await {
        Ok(result) => result.ok(),
        Err(_) => {
            eprintln!("[Resolve] yt-dlp did not answer within {} s; stopped.", limit.as_secs());
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct RssVideoItem {
    pub video_id: String,
    pub title: String,
    pub url: String,
    pub published_at: String,
}

#[derive(Debug, Clone)]
pub struct ChannelFeed {
    pub channel_id: String,
    pub channel_title: String,
    pub videos: Vec<RssVideoItem>,
}

/// Resolves a YouTube channel input (URL, @handle, or channel ID) to (channel_id, channel_title)
pub async fn resolve_channel(client: &Client, input: &str) -> Result<(String, String)> {
    let trimmed = input.trim();

    // Check if it's already a UC channel ID (e.g. UCxxxxxxxxxxxxxxxxxxxxxx)
    if trimmed.starts_with("UC") && trimmed.len() == 24 {
        // Fetch RSS directly to get the title
        if let Ok(feed) = fetch_channel_feed(client, trimmed).await {
            return Ok((feed.channel_id, feed.channel_title));
        }
        return Ok((trimmed.to_string(), trimmed.to_string()));
    }

    // yt-dlp only accepts URLs: a bare "@handle" is rejected as "not a valid URL".
    // So turn handles and plain names into the channel URL first.
    let target_url = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else if trimmed.starts_with('@') {
        format!("https://www.youtube.com/{}", trimmed)
    } else if trimmed.contains('/') || trimmed.contains("youtube.com") || trimmed.contains("youtu.be") {
        // Address-bar copy without the scheme, e.g. "youtube.com/@handle"
        format!("https://{}", trimmed)
    } else {
        format!("https://www.youtube.com/@{}", trimmed)
    };

    if !is_youtube_url(&target_url) {
        return Err(anyhow!(
            "'{}' is not a YouTube address. Use a channel URL on youtube.com, an @handle or a UC... channel ID.",
            trimmed
        ));
    }

    // Try resolving via yt-dlp first as it handles handles, custom URLs, and channel links reliably
    let mut command = crate::extractor::python_command(&crate::extractor::detect_python());
    command.args([
        "-m",
        "yt_dlp",
        "--print",
        "channel_id",
        "--print",
        "channel",
        "--playlist-items",
        "1",
        "--socket-timeout",
        "15",
        "--no-warnings",
        &target_url,
    ]);
    let output = output_with_timeout(command, RESOLVE_TIMEOUT).await;

    if let Some(out) = output {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
            if lines.len() >= 2 {
                let channel_id = lines[0].trim().to_string();
                let channel_title = lines[1].trim().to_string();
                if channel_id.starts_with("UC") {
                    return Ok((channel_id, channel_title));
                }
            } else if lines.len() == 1 && lines[0].starts_with("UC") {
                let channel_id = lines[0].trim().to_string();
                return Ok((channel_id.clone(), channel_id));
            }
        }
    }

    // Fallback: Fetch the webpage directly if it's a handle or URL
    let resp = client
        .get(&target_url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .await
        .context("Failed to request YouTube channel page")?
        .text()
        .await?;

    // The page owner's ID is in the "identifier" meta tag and in "externalId".
    // A plain "channelId" also appears for featured / recommended channels and the
    // first such match is often NOT the owner, so it is only the last resort.
    let patterns = [
        r#"itemprop="identifier" content="(UC[a-zA-Z0-9_-]{22})""#,
        r#""externalId":"(UC[a-zA-Z0-9_-]{22})""#,
        r#"channelId":"(UC[a-zA-Z0-9_-]{22})""#,
    ];
    let channel_id = patterns
        .iter()
        .find_map(|pat| {
            Regex::new(pat)
                .unwrap()
                .captures(&resp)
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_string())
        })
        .ok_or_else(|| anyhow!("Could not resolve YouTube channel ID from '{}'", input))?;

    // Extract title
    let title_re = Regex::new(r#"<title>([^<]+) - YouTube</title>"#).unwrap();
    let channel_title = if let Some(caps) = title_re.captures(&resp) {
        caps.get(1)
            .map(|m| html_escape_decode(m.as_str().trim()))
            .unwrap_or_else(|| channel_id.clone())
    } else {
        channel_id.clone()
    };

    Ok((channel_id, channel_title))
}

/// Fetches and parses the latest videos from a YouTube channel RSS feed
pub async fn fetch_channel_feed(client: &Client, channel_id: &str) -> Result<ChannelFeed> {
    let rss_url = format!("https://www.youtube.com/feeds/videos.xml?channel_id={}", channel_id);

    let xml_text = client
        .get(&rss_url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        .send()
        .await
        .context(format!("Failed to fetch RSS feed for channel {}", channel_id))?
        // A 404 / 429 / 500 page has no <entry>, so without this check a blocked
        // or wrong feed looked like "no new videos".
        .error_for_status()
        .with_context(|| format!("RSS feed for channel {} answered with an error", channel_id))?
        .text()
        .await?;

    // Parse channel title
    let channel_title_re = Regex::new(r#"<author>\s*<name>([^<]+)</name>"#).unwrap();
    let channel_title = channel_title_re
        .captures(&xml_text)
        .and_then(|c| c.get(1))
        .map(|m| html_escape_decode(m.as_str().trim()))
        .unwrap_or_else(|| channel_id.to_string());

    // Parse entries
    let entry_re = Regex::new(r#"(?s)<entry>(.*?)</entry>"#).unwrap();
    let vid_re = Regex::new(r#"<yt:videoId>([^<]+)</yt:videoId>"#).unwrap();
    let title_re = Regex::new(r#"<title>([^<]+)</title>"#).unwrap();
    let published_re = Regex::new(r#"<published>([^<]+)</published>"#).unwrap();

    let mut videos = Vec::new();
    for entry_match in entry_re.find_iter(&xml_text) {
        let entry_str = entry_match.as_str();

        let video_id = match vid_re.captures(entry_str).and_then(|c| c.get(1)) {
            Some(m) => m.as_str().trim().to_string(),
            None => continue,
        };

        let title = match title_re.captures(entry_str).and_then(|c| c.get(1)) {
            Some(m) => html_escape_decode(m.as_str().trim()),
            None => "Untitled Video".to_string(),
        };

        let published_at = match published_re.captures(entry_str).and_then(|c| c.get(1)) {
            Some(m) => m.as_str().trim().to_string(),
            None => chrono::Utc::now().to_rfc3339(),
        };

        let url = format!("https://www.youtube.com/watch?v={}", video_id);

        videos.push(RssVideoItem {
            video_id,
            title,
            url,
            published_at,
        });
    }

    Ok(ChannelFeed {
        channel_id: channel_id.to_string(),
        channel_title,
        videos,
    })
}

#[derive(Debug, Clone)]
pub struct SearchResultItem {
    pub video_id: String,
}

/// Fast, lightweight YouTube search without API keys
pub async fn search_youtube(client: &Client, query: &str, limit: usize) -> Result<Vec<SearchResultItem>> {
    // .query() percent-encodes the value. Formatting it into the URL lost "+", "&"
    // and "#" ("C++", "R&D", "C#").
    let resp = client
        .get("https://www.youtube.com/results")
        .query(&[("search_query", query)])
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .await
        .context("Failed to query YouTube search results")?
        .text()
        .await?;

    let vid_re = Regex::new(r#""videoId":"([a-zA-Z0-9_-]{11})""#).unwrap();
    let mut seen_ids = std::collections::HashSet::new();
    let mut items = Vec::new();

    for cap in vid_re.captures_iter(&resp) {
        let vid = cap.get(1).unwrap().as_str().to_string();
        if !seen_ids.contains(&vid) {
            seen_ids.insert(vid.clone());
            items.push(SearchResultItem { video_id: vid });
            if items.len() >= limit {
                break;
            }
        }
    }

    Ok(items)
}

/// Source label: the creator is named in this channel's kept insights (local
/// database, no request). The channel id is only known if such a channel is tracked.
pub const RELATED_SOURCE_MENTIONS: &str = "mentioned_in_kept_insights";
/// Source label: the channel is shown as a channel item (featured-channels
/// shelf) on the creator's YouTube page.
pub const RELATED_SOURCE_PAGE: &str = "featured_on_channel_page";

/// One related-channel candidate with where it came from. Nothing here is a
/// graph traversal: both sources are one hop from the given channel.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RelatedChannel {
    /// None for a mentioned name that has not been resolved to a channel yet
    /// (resolve it with `resolve_channel` when the user picks it).
    pub channel_id: Option<String>,
    pub title: String,
    /// `RELATED_SOURCE_MENTIONS` or `RELATED_SOURCE_PAGE`.
    pub source: String,
    /// In how many kept insights the name occurs; 0 for page hits.
    pub mentions: usize,
    /// True when `channel_id` is a channel already on the watchlist.
    pub tracked: bool,
}

/// Counts creator names case-insensitively. The spelling seen first is kept.
/// Sorted by count (highest first), then by name.
pub fn rank_mentions<S: AsRef<str>>(names: &[S]) -> Vec<(String, usize)> {
    let mut ranked: Vec<(String, usize)> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for name in names {
        let name = name.as_ref().trim().trim_start_matches('@').trim();
        if name.is_empty() {
            continue;
        }
        match index.get(&name.to_lowercase()) {
            Some(&pos) => ranked[pos].1 += 1,
            None => {
                index.insert(name.to_lowercase(), ranked.len());
                ranked.push((name.to_string(), 1));
            }
        }
    }
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase())));
    ranked
}

fn is_channel_id(id: &str) -> bool {
    id.len() == 24 && id.starts_with("UC") && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Text of a YouTube title object: {"simpleText": ..}, {"runs":[{"text": ..}]}
/// or the view-model form {"content": ..}.
fn renderer_text(value: &serde_json::Value) -> Option<String> {
    let text = value
        .get("simpleText")
        .or_else(|| value.get("content"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| {
            let runs = value.get("runs")?.as_array()?;
            Some(runs.iter().filter_map(|r| r.get("text")?.as_str()).collect::<String>())
        })?;
    let text = text.trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// Collects channel items below `value`: channelRenderer / gridChannelRenderer
/// objects, and lockupViewModel objects whose content type is a channel.
/// Video and playlist items carry their owner's id in other fields
/// (browseId, videoOwner...), which are deliberately not read: a channel whose
/// video merely appears in a shelf is not a channel the creator features.
fn collect_channel_items(value: &serde_json::Value, out: &mut Vec<(String, String)>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let hit = match key.as_str() {
                    "channelRenderer" | "gridChannelRenderer" => child
                        .get("channelId")
                        .and_then(|v| v.as_str())
                        .map(|id| (id, child.get("title").and_then(renderer_text))),
                    "lockupViewModel"
                        if child.get("contentType").and_then(|v| v.as_str())
                            == Some("LOCKUP_CONTENT_TYPE_CHANNEL") =>
                    {
                        child.get("contentId").and_then(|v| v.as_str()).map(|id| {
                            let title = child
                                .pointer("/metadata/lockupMetadataViewModel/title")
                                .and_then(renderer_text);
                            (id, title)
                        })
                    }
                    _ => None,
                };
                if let Some((id, title)) = hit {
                    if is_channel_id(id) {
                        out.push((id.to_string(), title.unwrap_or_default()));
                    }
                }
                collect_channel_items(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_channel_items(item, out);
            }
        }
        _ => {}
    }
}

/// Channels a creator features on their page, from the page HTML: the channel
/// items inside `ytInitialData`, in page order, without duplicates and without
/// the page owner. The title is empty when the item carries none.
///
/// Errors when the page has no readable `ytInitialData` (consent wall, error
/// page, changed layout), so "could not read the page" is never reported as
/// "this creator features nobody".
pub fn parse_featured_channels(html: &str, own_channel_id: &str) -> Result<Vec<(String, String)>> {
    // `var ytInitialData = {...};` or `window["ytInitialData"] = {...};`
    static ASSIGN_RE: std::sync::OnceLock<Option<Regex>> = std::sync::OnceLock::new();
    let json_start = ASSIGN_RE
        .get_or_init(|| Regex::new(r#"ytInitialData["'\]]*\s*=\s*\{"#).ok())
        .as_ref()
        .and_then(|re| re.find(html))
        .map(|m| m.end() - 1)
        .ok_or_else(|| anyhow!("Channel page has no ytInitialData object (consent or error page?)"))?;
    // The object is followed by ";</script>": read exactly one JSON value.
    let data: serde_json::Value = serde_json::Deserializer::from_str(&html[json_start..])
        .into_iter::<serde_json::Value>()
        .next()
        .ok_or_else(|| anyhow!("Channel page has an empty ytInitialData"))?
        .context("ytInitialData of the channel page is not valid JSON")?;

    let mut found = Vec::new();
    collect_channel_items(&data, &mut found);

    let mut seen = std::collections::HashSet::new();
    seen.insert(own_channel_id.to_string());
    Ok(found.into_iter().filter(|(id, _)| seen.insert(id.clone())).collect())
}

/// Channels the creator features on their YouTube page (channel items only), as
/// (channel_id, title). An empty list means the page was read and shows no
/// featured channels; most creators have none. For the offline source (creators
/// mentioned in kept insights) see `IngestionPipeline::related_channels`.
pub async fn fetch_related_channels(client: &Client, channel_id: &str) -> Result<Vec<(String, String)>> {
    let target_url = format!("https://www.youtube.com/channel/{}", channel_id);

    let resp = client
        .get(&target_url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .await
        .context("Failed to request channel page for related creators")?
        .error_for_status()
        .context("Channel page answered with an error")?
        .text()
        .await?;

    let mut related = parse_featured_channels(&resp, channel_id)?;
    related.truncate(MAX_FEATURED_CHANNELS);

    // An item without a title: ask the channel's feed for the name. A channel
    // whose feed does not load keeps its id as the name instead of being dropped.
    for (cid, title) in related.iter_mut() {
        if title.is_empty() {
            *title = match fetch_channel_feed(client, cid).await {
                Ok(feed) => feed.channel_title,
                Err(_) => cid.clone(),
            };
        }
    }

    Ok(related)
}

/// Upper bound of page hits returned (and of follow-up feed requests for names).
const MAX_FEATURED_CHANNELS: usize = 8;

/// Decodes the XML/HTML entities found in feed and page titles: the five named
/// ones and numeric ones (&#39; &#x27;). One pass over the text, so "&amp;lt;"
/// becomes "&lt;" and is not decoded a second time.
pub(crate) fn html_escape_decode(s: &str) -> String {
    static ENTITY_RE: std::sync::OnceLock<Option<Regex>> = std::sync::OnceLock::new();
    let Some(re) = ENTITY_RE
        .get_or_init(|| Regex::new(r"&(amp|lt|gt|quot|apos|#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7});").ok())
        .as_ref()
    else {
        return s.to_string();
    };

    re.replace_all(s, |caps: &regex::Captures| {
        let entity = &caps[1];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => {
                let digits = &entity[1..];
                let code = match digits.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok(),
                    None => digits.parse::<u32>().ok(),
                };
                // Control characters would break the one-line titles in UI and wiki.
                code.and_then(char::from_u32).filter(|c| !c.is_control())
            }
        };
        match decoded {
            Some(c) => c.to_string(),
            // Not a valid code point: keep the text as it was.
            None => caps[0].to_string(),
        }
    })
    .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_youtube_hosts_are_accepted_as_channel_urls() {
        assert!(is_youtube_url("https://www.youtube.com/@fakechannel"));
        assert!(is_youtube_url("https://youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx"));
        assert!(is_youtube_url("http://M.YouTube.com/@handle"));
        assert!(is_youtube_url("https://youtu.be/AbCdEfGh123"));

        assert!(!is_youtube_url("http://127.0.0.1:8080/admin"));
        assert!(!is_youtube_url("https://example.com/youtube.com/@handle"));
        assert!(!is_youtube_url("https://youtube.com.example.org/@handle"));
        assert!(!is_youtube_url("https://www.youtube.com@example.org/x"));
        assert!(!is_youtube_url("not a url"));
    }

    #[tokio::test]
    async fn a_foreign_url_is_refused_before_any_request() {
        let client = Client::new();
        for input in ["http://127.0.0.1:9/feed", "example.org/@handle"] {
            let err = resolve_channel(&client, input).await.expect_err("not YouTube");
            assert!(err.to_string().contains("not a YouTube address"), "{}", err);
        }
    }

    #[tokio::test]
    async fn a_helper_that_cannot_be_started_or_is_too_slow_yields_none() {
        let missing = std::process::Command::new("ytc-no-such-helper-program");
        assert!(output_with_timeout(missing, Duration::from_secs(5)).await.is_none());

        // The test binary itself, listing its tests, is a program that exists
        // on every platform; a zero limit ends it before it can answer.
        let exe = std::env::current_exe().expect("test binary path");
        let mut slow = std::process::Command::new(&exe);
        slow.arg("--list");
        assert!(output_with_timeout(slow, Duration::ZERO).await.is_none());

        let mut quick = std::process::Command::new(&exe);
        quick.arg("--list");
        let out = output_with_timeout(quick, Duration::from_secs(60)).await.expect("finishes in time");
        assert!(out.status.success());
    }

    const OWN: &str = "UC0000000000000000000own";
    const FEAT_A: &str = "UC00000000000000000featA";
    const FEAT_B: &str = "UC00000000000000000featB";
    const FEAT_C: &str = "UC00000000000000000featC";
    const VIDEO_OWNER: &str = "UC00000000000videoOwner_";

    /// Hand-made page in the shape of a channel home tab: a video shelf whose
    /// items name another channel only as the video's owner, and a
    /// featured-channels shelf with the three channel item forms.
    fn fixture() -> String {
        format!(
            r#"<html><body><script nonce="x">var ytInitialData = {{"contents":{{"twoColumnBrowseResultsRenderer":{{"tabs":[{{"tabRenderer":{{"title":"Home","content":{{"sectionListRenderer":{{"contents":[
{{"itemSectionRenderer":{{"contents":[{{"shelfRenderer":{{"title":{{"runs":[{{"text":"Uploads"}}]}},"content":{{"horizontalListRenderer":{{"items":[
  {{"lockupViewModel":{{"contentId":"LOP8ul3KqQI","contentType":"LOCKUP_CONTENT_TYPE_VIDEO","metadata":{{"lockupMetadataViewModel":{{"title":{{"content":"A video"}}}}}}}}}},
  {{"gridVideoRenderer":{{"videoId":"abcdefghijk","shortBylineText":{{"runs":[{{"text":"Guest","navigationEndpoint":{{"browseEndpoint":{{"browseId":"{video_owner}"}}}}}}]}}}}}}
]}}}}}}}}]}}}},
{{"itemSectionRenderer":{{"contents":[{{"shelfRenderer":{{"title":{{"runs":[{{"text":"Channels I like"}}]}},"content":{{"horizontalListRenderer":{{"items":[
  {{"gridChannelRenderer":{{"channelId":"{a}","title":{{"simpleText":"Alpha & Co"}},"subscriberCountText":{{"simpleText":"1M"}}}}}},
  {{"channelRenderer":{{"channelId":"{b}","title":{{"runs":[{{"text":"Beta "}},{{"text":"Labs"}}]}}}}}},
  {{"lockupViewModel":{{"contentId":"{c}","contentType":"LOCKUP_CONTENT_TYPE_CHANNEL","metadata":{{"lockupMetadataViewModel":{{"title":{{"content":"Gamma"}}}}}}}}}},
  {{"gridChannelRenderer":{{"channelId":"{a}","title":{{"simpleText":"Alpha again"}}}}}},
  {{"gridChannelRenderer":{{"channelId":"{own}","title":{{"simpleText":"Myself"}}}}}},
  {{"gridChannelRenderer":{{"channelId":"not-a-channel-id","title":{{"simpleText":"Broken"}}}}}}
]}}}}}}}}]}}}}
]}}}}}}}}]}}}},"metadata":{{"channelMetadataRenderer":{{"externalId":"{own}"}}}}}};</script><script>var other = {{}};</script></body></html>"#,
            video_owner = VIDEO_OWNER,
            a = FEAT_A,
            b = FEAT_B,
            c = FEAT_C,
            own = OWN,
        )
    }

    #[test]
    fn only_channel_items_of_the_page_are_returned() {
        let found = parse_featured_channels(&fixture(), OWN).expect("fixture has ytInitialData");
        assert_eq!(
            found,
            vec![
                (FEAT_A.to_string(), "Alpha & Co".to_string()),
                (FEAT_B.to_string(), "Beta Labs".to_string()),
                (FEAT_C.to_string(), "Gamma".to_string()),
            ]
        );
        // The owner of a video in a shelf is not a featured channel.
        assert!(found.iter().all(|(id, _)| id != VIDEO_OWNER));
    }

    #[test]
    fn a_page_without_channel_items_is_an_empty_list_not_an_error() {
        let html = r#"<script>var ytInitialData = {"contents":{"items":[{"gridVideoRenderer":{"videoId":"abcdefghijk"}}]}};</script>"#;
        assert!(parse_featured_channels(html, OWN).expect("readable page").is_empty());
    }

    #[test]
    fn an_unreadable_page_is_an_error() {
        assert!(parse_featured_channels("<html>Before you continue to YouTube</html>", OWN).is_err());
        assert!(parse_featured_channels("var ytInitialData = ;", OWN).is_err());
        assert!(parse_featured_channels("if (window.ytInitialData) { run(); }", OWN).is_err());
        assert!(parse_featured_channels(r#"var ytInitialData = {"contents": [;"#, OWN).is_err());
    }

    #[test]
    fn mentions_are_counted_case_insensitively_and_ranked() {
        let names = ["Alice Example", "sam sample", "alice example", " @Fakechannel ", "", "Sam Sample", "ALICE EXAMPLE"];
        assert_eq!(
            rank_mentions(&names),
            vec![
                ("Alice Example".to_string(), 3),
                ("sam sample".to_string(), 2),
                ("Fakechannel".to_string(), 1),
            ]
        );
        assert!(rank_mentions::<&str>(&[]).is_empty());
    }
}
