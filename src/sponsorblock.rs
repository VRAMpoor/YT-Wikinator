use crate::models::{CleanedTranscript, SponsorSegment, TranscriptCue};
use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Segment categories requested from SponsorBlock. Everything in this list is
/// removed from the transcript, so "stripped seconds" covers intros, outros and
/// filler as well as sponsor reads.
pub const SPONSORBLOCK_CATEGORIES: [&str; 6] = ["sponsor", "selfpromo", "interaction", "intro", "outro", "filler"];

/// Action types that mean "this part is not content". Others ("full", "poi",
/// "chapter") label a video or a point in it and remove nothing.
const REMOVING_ACTION_TYPES: [&str; 2] = ["skip", "mute"];

/// Length of the SHA-256 prefix sent to the API. Four hex characters is the
/// shortest the API accepts: the server learns one of 65,536 buckets, not the video.
const HASH_PREFIX_LEN: usize = 4;

/// A cue longer than this (Whisper segments, merged captions) is trimmed word by
/// word instead of being kept or dropped as a whole.
const LONG_CUE_SECS: f64 = 10.0;

/// A short cue is dropped when more than this share of it lies inside a segment.
const DROP_OVERLAP_RATIO: f64 = 0.4;

#[derive(Debug, Deserialize)]
struct SponsorApiResponseItem {
    #[serde(default)]
    pub category: String,
    #[serde(rename = "actionType")]
    pub action_type: Option<String>,
    #[serde(default)]
    pub segment: Vec<f64>,
}

/// One entry of the hash-prefix answer: every video whose ID hash starts with
/// the requested prefix.
#[derive(Debug, Deserialize)]
struct HashPrefixVideo {
    #[serde(rename = "videoID")]
    video_id: String,
    #[serde(default)]
    segments: Vec<SponsorApiResponseItem>,
}

#[derive(Clone)]
pub struct SponsorBlockClient {
    client: Client,
    base_url: String,
}

/// Limit for one SponsorBlock request, body included. reqwest has no default
/// timeout: a server that accepts the connection and then stalls would hang
/// the crawl for good, with its progress stuck on "running".
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn build_client(request_timeout: Duration) -> Client {
    Client::builder()
        .user_agent("YTC-YouTubeCrawler/1.0")
        .connect_timeout(CONNECT_TIMEOUT.min(request_timeout))
        .timeout(request_timeout)
        .build()
        .unwrap()
}

impl Default for SponsorBlockClient {
    fn default() -> Self {
        Self {
            client: build_client(REQUEST_TIMEOUT),
            base_url: "https://sponsor.ajay.app".to_string(),
        }
    }
}

/// SHA-256 (FIPS 180-4) as lower-case hex. Only used to build the privacy
/// prefix for SponsorBlock; the crate has no hashing dependency, and the
/// function is checked against the standard test vectors below.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Padding: 0x80, zeros up to 56 mod 64, then the bit length as 64-bit big-endian.
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (state, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *state = state.wrapping_add(value);
        }
    }

    h.iter().map(|v| format!("{:08x}", v)).collect()
}

/// The part of the video ID's hash that is sent to SponsorBlock.
fn hash_prefix(video_id: &str) -> String {
    let mut hex = sha256_hex(video_id.as_bytes());
    hex.truncate(HASH_PREFIX_LEN);
    hex
}

/// Picks this video's segments out of a hash-prefix answer, which lists every
/// video sharing the prefix. The video ID comparison happens here, locally.
fn parse_hash_prefix_response(body: &str, video_id: &str) -> Result<Vec<SponsorSegment>> {
    let videos: Vec<HashPrefixVideo> =
        serde_json::from_str(body).context("SponsorBlock answer is not the expected JSON list")?;

    let mut segments = Vec::new();
    for video in videos.into_iter().filter(|v| v.video_id == video_id) {
        for item in video.segments {
            if item.segment.len() >= 2 {
                segments.push(SponsorSegment {
                    category: item.category,
                    action_type: item.action_type.unwrap_or_else(|| "skip".to_string()),
                    start_sec: item.segment[0],
                    end_sec: item.segment[1],
                });
            }
        }
    }
    Ok(segments)
}

/// Time ranges that are really to be removed: only skip/mute segments with a
/// positive length, sorted, with overlapping or touching ranges merged so no
/// second is counted twice.
fn merged_removal_ranges(segments: &[SponsorSegment]) -> Vec<(f64, f64)> {
    let mut ranges: Vec<(f64, f64)> = segments
        .iter()
        .filter(|s| REMOVING_ACTION_TYPES.contains(&s.action_type.as_str()))
        .filter(|s| s.start_sec.is_finite() && s.end_sec.is_finite() && s.end_sec > s.start_sec)
        .map(|s| (s.start_sec.max(0.0), s.end_sec))
        .collect();
    merge_ranges(&mut ranges);
    ranges
}

/// Sorts and merges overlapping or touching ranges in place.
fn merge_ranges(ranges: &mut Vec<(f64, f64)>) {
    ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges.iter() {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    *ranges = merged;
}

fn in_any_range(t: f64, ranges: &[(f64, f64)]) -> bool {
    ranges.iter().any(|&(start, end)| t >= start && t < end)
}

impl SponsorBlockClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetches the community-submitted segments of a video from SponsorBlock.
    ///
    /// Uses the hash-prefix endpoint: only the first four hex characters of
    /// SHA-256(video ID) leave this machine, and the answer (all videos with
    /// that prefix) is filtered locally. A video nobody submitted segments for
    /// yields an empty list.
    pub async fn fetch_segments(&self, video_id: &str) -> Result<Vec<SponsorSegment>> {
        let url = format!("{}/api/skipSegments/{}", self.base_url, hash_prefix(video_id));
        let categories = serde_json::to_string(&SPONSORBLOCK_CATEGORIES).unwrap_or_default();
        let action_types = serde_json::to_string(&REMOVING_ACTION_TYPES).unwrap_or_default();

        let resp = match self
            .client
            .get(&url)
            .query(&[("categories", categories.as_str()), ("actionTypes", action_types.as_str())])
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                // If SponsorBlock network request fails, return empty gracefully without crashing
                eprintln!("[SponsorBlock] Warning: Failed to query API: {}", e);
                return Ok(Vec::new());
            }
        };

        if resp.status().as_u16() == 404 {
            // No segments registered for any video with this prefix
            return Ok(Vec::new());
        }

        if !resp.status().is_success() {
            eprintln!(
                "[SponsorBlock] Warning: API returned HTTP status {}",
                resp.status()
            );
            return Ok(Vec::new());
        }

        let body = match resp.text().await {
            Ok(text) => text,
            Err(e) => {
                eprintln!("[SponsorBlock] Warning: Failed to read response: {}", e);
                return Ok(Vec::new());
            }
        };

        match parse_hash_prefix_response(&body, video_id) {
            Ok(segments) => Ok(segments),
            Err(e) => {
                eprintln!("[SponsorBlock] Warning: {}", e);
                Ok(Vec::new())
            }
        }
    }

    /// Removes SponsorBlock segments from the transcript cues.
    ///
    /// - Only skip/mute segments count; overlapping ones are merged first.
    /// - A cue with usable timing up to `LONG_CUE_SECS` long is dropped when more
    ///   than 40% of it lies inside a segment.
    /// - A longer cue is trimmed: its words are spread evenly over its duration
    ///   and those falling inside a segment are removed.
    /// - A cue without a duration is dropped only when its start lies inside a segment.
    ///
    /// `stripped_ad_seconds` is the transcript time actually removed (dropped
    /// cues and trimmed parts, overlaps counted once) and
    /// `stripped_segments_count` the number of merged segments that removed
    /// something. Both are 0 when the segments matched no cue.
    pub fn strip_sponsors(
        &self,
        video_id: &str,
        cues: &[TranscriptCue],
        segments: &[SponsorSegment],
    ) -> CleanedTranscript {
        let raw_text: String = cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        let raw_char_count = raw_text.len();

        let ranges = merged_removal_ranges(segments);
        let mut kept_texts: Vec<String> = Vec::with_capacity(cues.len());
        // Time taken out of the transcript, as ranges, so that the overlapping
        // cues of rolling auto-captions are not counted twice.
        let mut removed: Vec<(f64, f64)> = Vec::new();
        // Which merged segment took something out, including cues without duration.
        let mut segment_used = vec![false; ranges.len()];

        for cue in cues {
            let duration = cue.end_sec - cue.start_sec;
            let has_timing = cue.start_sec.is_finite() && cue.end_sec.is_finite() && duration > 0.0;

            if !has_timing {
                match ranges.iter().position(|&(s, e)| cue.start_sec >= s && cue.start_sec < e) {
                    Some(idx) => segment_used[idx] = true,
                    None => kept_texts.push(cue.text.clone()),
                }
                continue;
            }

            let overlaps: Vec<(usize, f64, f64)> = ranges
                .iter()
                .enumerate()
                .filter_map(|(idx, &(s, e))| {
                    let start = cue.start_sec.max(s);
                    let end = cue.end_sec.min(e);
                    (end > start).then_some((idx, start, end))
                })
                .collect();
            if overlaps.is_empty() {
                kept_texts.push(cue.text.clone());
                continue;
            }

            if duration <= LONG_CUE_SECS {
                let overlap: f64 = overlaps.iter().map(|&(_, s, e)| e - s).sum();
                if overlap / duration > DROP_OVERLAP_RATIO {
                    removed.push((cue.start_sec, cue.end_sec));
                    for &(idx, _, _) in &overlaps {
                        segment_used[idx] = true;
                    }
                } else {
                    kept_texts.push(cue.text.clone());
                }
                continue;
            }

            // Long cue: keep the words outside the segments.
            let words: Vec<&str> = cue.text.split_whitespace().collect();
            let word_count = words.len().max(1) as f64;
            let kept_words: Vec<&str> = words
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    let position = cue.start_sec + (*i as f64 + 0.5) / word_count * duration;
                    !in_any_range(position, &ranges)
                })
                .map(|(_, w)| *w)
                .collect();
            if kept_words.len() < words.len() {
                for &(idx, start, end) in &overlaps {
                    removed.push((start, end));
                    segment_used[idx] = true;
                }
            }
            if !kept_words.is_empty() {
                kept_texts.push(kept_words.join(" "));
            }
        }

        merge_ranges(&mut removed);
        let stripped_ad_seconds: f64 = removed.iter().map(|&(s, e)| e - s).sum();
        let stripped_segments_count = segment_used.iter().filter(|used| **used).count();

        let cleaned_text = kept_texts.join(" ");

        CleanedTranscript {
            video_id: video_id.to_string(),
            raw_char_count,
            cleaned_char_count: cleaned_text.len(),
            stripped_ad_seconds,
            stripped_segments_count,
            text: cleaned_text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start: f64, end: f64, text: &str) -> TranscriptCue {
        TranscriptCue { start_sec: start, end_sec: end, text: text.to_string() }
    }

    fn seg(action: &str, start: f64, end: f64) -> SponsorSegment {
        SponsorSegment {
            category: "sponsor".to_string(),
            action_type: action.to_string(),
            start_sec: start,
            end_sec: end,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[tokio::test]
    async fn a_stalled_server_ends_in_no_segments_instead_of_hanging() {
        // Accepts the connection and never answers.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for stream in listener.incoming().flatten() {
                held.push(stream);
            }
        });

        let client = SponsorBlockClient {
            client: build_client(Duration::from_millis(300)),
            base_url: format!("http://{}", addr),
        };
        let started = std::time::Instant::now();
        let segments = client.fetch_segments("wantedVid01").await.expect("degrades to no segments");
        assert!(segments.is_empty());
        assert!(started.elapsed() < Duration::from_secs(10), "{:?}", started.elapsed());
    }

    #[test]
    fn sha256_matches_the_standard_test_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 56 bytes: the padding spills into a second block.
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(hash_prefix("abc"), "ba78");
    }

    #[test]
    fn hash_prefix_answer_is_filtered_by_video_id() {
        let body = r#"[
            {"videoID":"otherVideo1","hash":"ba78aaaa","segments":[
                {"category":"sponsor","actionType":"skip","segment":[1.0,2.0],"UUID":"x"}]},
            {"videoID":"wantedVid01","hash":"ba78bbbb","segments":[
                {"category":"intro","actionType":"skip","segment":[0.0,12.5],"UUID":"y"},
                {"category":"sponsor","segment":[100.0,130.0]},
                {"category":"poi_highlight","actionType":"poi","segment":[40.0]}]}
        ]"#;
        let segments = parse_hash_prefix_response(body, "wantedVid01").expect("valid answer");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].category, "intro");
        assert!(close(segments[0].end_sec, 12.5));
        // A missing actionType is the API default, "skip".
        assert_eq!(segments[1].action_type, "skip");

        assert!(parse_hash_prefix_response(body, "unknownVid0").expect("valid answer").is_empty());
        assert!(parse_hash_prefix_response("Not Found", "wantedVid01").is_err());
    }

    #[test]
    fn overlapping_segments_are_counted_once() {
        let cues: Vec<TranscriptCue> = (0..10)
            .map(|i| cue(i as f64 * 2.0, i as f64 * 2.0 + 2.0, &format!("cue{}", i)))
            .collect();
        // 4..10 and 8..14 overlap: together they cover 4..14, i.e. 10 s, not 12 s.
        let segments = vec![seg("skip", 4.0, 10.0), seg("skip", 8.0, 14.0)];
        let cleaned = SponsorBlockClient::new().strip_sponsors("vid", &cues, &segments);

        assert_eq!(cleaned.text, "cue0 cue1 cue7 cue8 cue9");
        assert!(close(cleaned.stripped_ad_seconds, 10.0), "{}", cleaned.stripped_ad_seconds);
        assert_eq!(cleaned.stripped_segments_count, 1);
        assert_eq!(cleaned.cleaned_char_count, cleaned.text.len());
    }

    #[test]
    fn long_cue_is_trimmed_by_overlap_not_kept_whole() {
        // One 30 s Whisper-style cue with 30 words; the sponsor read is 10..20 s.
        // That is 33% of the cue, below the share at which a whole cue is dropped.
        let words: Vec<String> = (0..30).map(|i| format!("w{}", i)).collect();
        let cues = vec![cue(0.0, 30.0, &words.join(" "))];
        let cleaned = SponsorBlockClient::new().strip_sponsors("vid", &cues, &[seg("skip", 10.0, 20.0)]);

        let kept: Vec<&str> = cleaned.text.split_whitespace().collect();
        assert_eq!(kept.len(), 20);
        assert!(kept.contains(&"w9") && kept.contains(&"w20"));
        assert!(!kept.contains(&"w10") && !kept.contains(&"w19"));
        assert!(close(cleaned.stripped_ad_seconds, 10.0));
        assert_eq!(cleaned.stripped_segments_count, 1);
        assert!(cleaned.cleaned_char_count < cleaned.raw_char_count);
    }

    #[test]
    fn nothing_removed_reports_zero() {
        let cues = vec![cue(0.0, 2.0, "hello there"), cue(2.0, 4.0, "general remarks")];
        // A segment after the last cue, one that only grazes a cue (25%), and
        // labels that are not removals.
        let segments = vec![
            seg("skip", 500.0, 560.0),
            seg("skip", 3.5, 4.0),
            seg("poi", 0.0, 4.0),
            seg("full", 0.0, 4.0),
            seg("chapter", 0.0, 4.0),
            seg("skip", 3.0, 3.0),
        ];
        let cleaned = SponsorBlockClient::new().strip_sponsors("vid", &cues, &segments);

        assert_eq!(cleaned.text, "hello there general remarks");
        assert_eq!(cleaned.stripped_ad_seconds, 0.0);
        assert_eq!(cleaned.stripped_segments_count, 0);
        assert_eq!(cleaned.raw_char_count, cleaned.cleaned_char_count);
    }

    #[test]
    fn rolling_caption_cues_do_not_double_count_removed_time() {
        // Auto-caption cues overlap each other in time.
        let cues = vec![
            cue(0.0, 4.0, "intro words"),
            cue(2.0, 6.0, "sponsor one"),
            cue(4.0, 8.0, "sponsor two"),
            cue(8.0, 12.0, "content again"),
        ];
        let cleaned = SponsorBlockClient::new().strip_sponsors("vid", &cues, &[seg("mute", 3.0, 8.0)]);
        assert_eq!(cleaned.text, "intro words content again");
        // Removed cues span 2..8 = 6 s, not 4 + 4.
        assert!(close(cleaned.stripped_ad_seconds, 6.0), "{}", cleaned.stripped_ad_seconds);
        assert_eq!(cleaned.stripped_segments_count, 1);
    }

    #[test]
    fn cue_without_duration_is_dropped_only_inside_a_segment() {
        let cues = vec![cue(5.0, 5.0, "inside"), cue(50.0, 50.0, "outside")];
        let cleaned = SponsorBlockClient::new().strip_sponsors("vid", &cues, &[seg("skip", 0.0, 10.0)]);
        assert_eq!(cleaned.text, "outside");
        // The cue had no duration, so no seconds can be claimed for it.
        assert_eq!(cleaned.stripped_ad_seconds, 0.0);
        assert_eq!(cleaned.stripped_segments_count, 1);
    }
}
