use crate::models::{match_category, Category, CleanedTranscript, InsightResult, TOPIC_OK};
use anyhow::{Context, Result};
use colored::*;
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Serialize)]
struct ChatCompletionMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatCompletionMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // Left out for plain-text answers (section notes): not every server accepts null.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionResponseMessage,
    /// "length" when the answer was cut off at max_tokens or the context limit.
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponseMessage {
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

// Local models do not follow the schema exactly: they send null for a field they
// have nothing for, a score as 35.0 or "35", a single string where a list is asked.
// The lenient_* helpers accept these, so one sloppy field does not fail the video.
#[derive(Debug, Deserialize)]
struct LlmInsightJson {
    /// None when the answer has no usable score at all.
    #[serde(default, deserialize_with = "lenient_score")]
    slop_score: Option<u32>,
    // "is_worth_keeping" of the answer is not read: the keep decision is
    // computed from the parts (topic verdict, score, threshold).
    /// Raw topic verdict; read with `parse_topic_verdict`. None when the
    /// answer has no verdict at all (missing or null).
    #[serde(default, deserialize_with = "lenient_verdict_text")]
    topic_verdict: Option<String>,
    #[serde(default, deserialize_with = "lenient_opt_string")]
    topic_reason: Option<String>,
    #[serde(default, deserialize_with = "lenient_string")]
    quality_rating: String,
    #[serde(default, deserialize_with = "lenient_string")]
    summary: String,
    #[serde(default, deserialize_with = "lenient_string_list")]
    key_insights: Vec<String>,
    #[serde(default, deserialize_with = "lenient_string_list")]
    takeaways: Vec<String>,
    #[serde(default, deserialize_with = "lenient_string_list")]
    quotes: Vec<String>,
    #[serde(default, deserialize_with = "lenient_string_list")]
    topics: Vec<String>,
    #[serde(default, deserialize_with = "lenient_string_list")]
    mentioned_creators: Vec<String>,
    /// Raw category name; read with `models::match_category`.
    #[serde(default, deserialize_with = "lenient_opt_string")]
    category: Option<String>,
}

/// The answer of a category request (`CATEGORIZE_SYSTEM_PROMPT`).
#[derive(Debug, Deserialize)]
struct LlmCategoryJson {
    /// Raw category name; read with `models::match_category`.
    #[serde(default, deserialize_with = "lenient_opt_string")]
    category: Option<String>,
}

fn default_quality() -> String {
    "Moderate Signal".to_string()
}

/// The rejecting verdict a single lower-case word stands for, if any:
/// "political" and "geopolitics" are "politics", "pseudoscience" is "fringe".
fn verdict_of_word(word: &str) -> Option<&'static str> {
    const STEMS: &[(&str, &str)] = &[
        ("fringe", "fringe"),
        ("pseudo", "fringe"),
        ("paranormal", "fringe"),
        ("conspirac", "fringe"),
        ("nonsens", "nonsense"),
        ("politic", "politics"),
        ("geopolitic", "politics"),
    ];
    STEMS.iter().find(|(stem, _)| word.starts_with(stem)).map(|&(_, verdict)| verdict)
}

/// The model's topic verdict as one of `models::TOPIC_VERDICTS`, or None when
/// the answer cannot be read as one of them (the caller treats the whole
/// answer as unusable and asks again).
///
/// Case, whitespace and punctuation do not matter, and common variants are
/// understood ("Politics.", "political", "fringe/pseudoscience",
/// "ok (history lecture)"). A missing verdict is "ok": a model that left the
/// field out raised no objection. Anything else is NOT turned into "ok": "not ok", "off-topic",
/// "ok / politics" or an invented label would otherwise keep a video the
/// model wanted to reject, with a stored row that claims the model said "ok".
fn parse_topic_verdict(raw: Option<&str>) -> Option<&'static str> {
    let text = raw.unwrap_or_default().trim().to_lowercase();
    if text.is_empty() {
        return Some(TOPIC_OK);
    }
    let words: Vec<&str> = text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let rejecting = words.iter().enumerate().find_map(|(pos, w)| verdict_of_word(w).map(|v| (pos, v)));
    let starts_ok = matches!(words.first(), Some(&"ok") | Some(&"okay"));
    match rejecting {
        // "ok", "OK.", "okay", "ok (history lecture)".
        None if starts_ok => Some(TOPIC_OK),
        // "sports", "off-topic", "not ok", "rejected", "...".
        None => None,
        // "ok / politics", "ok (not politics)": says both.
        Some(_) if starts_ok => None,
        // "not politics", "non-political": a denial, not a verdict.
        Some((pos, _)) if words[..pos].iter().any(|w| matches!(*w, "not" | "no" | "non" | "isn" | "never")) => None,
        Some((_, verdict)) => Some(verdict),
    }
}

fn value_to_text(v: &serde_json::Value) -> Option<String> {
    let text = match v {
        serde_json::Value::String(s) => Some(s.trim().to_string()),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => Some(v.to_string()),
        // null has no text; a nested list or object is dropped rather than stored as raw JSON.
        _ => None,
    };
    text.filter(|s| !s.is_empty())
}

/// A list with exactly one element stands for that element: local models
/// sometimes wrap a single string in brackets (`"topic_verdict": ["politics"]`).
fn single_element(v: &serde_json::Value) -> &serde_json::Value {
    match v {
        serde_json::Value::Array(items) if items.len() == 1 => &items[0],
        other => other,
    }
}

fn lenient_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(value_to_text(single_element(&v)).unwrap_or_default())
}

fn lenient_opt_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(value_to_text(single_element(&v)).filter(|s| !s.eq_ignore_ascii_case("null")))
}

/// The topic verdict as text, without losing anything: None only for a
/// missing or null value. A list, object, number or boolean is kept as its
/// JSON text, so `parse_topic_verdict` sees it and can refuse it; dropped
/// (as `lenient_string` does) it would read as "no verdict", which is "ok".
fn lenient_verdict_text<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match single_element(&v) {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    })
}

fn lenient_string_list<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match &v {
        serde_json::Value::Array(items) => items.iter().filter_map(value_to_text).collect(),
        other => value_to_text(other).into_iter().collect(),
    })
}

/// Number or numeric string, rounded and clamped to 0..=100: -5 is 0, 45.5 is
/// 46, 150 is 100. Anything that is not a number ("20%", "low", a list) is
/// None, which makes the whole answer unusable (asked again once).
fn lenient_score<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u32>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    let number = match &v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().trim_end_matches("/100").trim().parse::<f64>().ok(),
        _ => None,
    };
    Ok(number.filter(|n| n.is_finite()).map(|n| n.round().clamp(0.0, 100.0) as u32))
}

/// `analyzer_type` of results produced without an LLM.
pub const HEURISTIC_ANALYZER: &str = "heuristic";
/// Quality rating of a heuristic result: nobody reviewed the content.
pub const HEURISTIC_QUALITY_RATING: &str = "Unreviewed (heuristic, no LLM)";

/// Env var: context window of the LLM server in tokens (llama-server `-c`).
pub const LLM_CONTEXT_TOKENS_ENV: &str = "YTC_LLM_CONTEXT_TOKENS";
/// Context window assumed when the env var is not set.
pub const DEFAULT_LLM_CONTEXT_TOKENS: usize = 8192;
/// Below this the prompt and the answer leave no room for a transcript.
pub const MIN_LLM_CONTEXT_TOKENS: usize = 4096;
pub const MAX_LLM_CONTEXT_TOKENS: usize = 2_000_000;

/// Env var: how many characters are counted as one token when the transcript
/// is budgeted.
pub const LLM_CHARS_PER_TOKEN_ENV: &str = "YTC_LLM_CHARS_PER_TOKEN";
/// English prose is about 4 characters per token; 3 leaves room for numbers,
/// names, German and auto-caption text, which tokenize worse. Lower it when
/// the server answers "context size exceeded".
pub const DEFAULT_CHARS_PER_TOKEN: f64 = 3.0;
const MIN_CHARS_PER_TOKEN: f64 = 1.0;
const MAX_CHARS_PER_TOKEN: f64 = 6.0;

/// Env var: overrides the computed budget with a fixed number of transcript
/// characters per LLM request (whole-transcript request and section alike).
pub const LLM_CONTEXT_CHARS_ENV: &str = "YTC_LLM_CONTEXT_CHARS";
/// Below this a section is too short to take notes on.
pub const MIN_LLM_CONTEXT_CHARS: usize = 3_000;
/// About 50k tokens: more than this in one request is a configuration mistake
/// for the local models this tool is used with.
pub const MAX_LLM_CONTEXT_CHARS: usize = 200_000;

/// Answer budget of the judgment request.
const JUDGMENT_MAX_TOKENS: u32 = 1500;
/// Answer budget of a category request: one short JSON object, with room for
/// a model that thinks before it answers.
const CATEGORIZE_MAX_TOKENS: u32 = 400;
/// Answer budget of a notes request (one section, or a group of notes).
const NOTES_MAX_TOKENS: u32 = 600;
/// Tokens reserved per request for the chat template, the title and headings.
const REQUEST_OVERHEAD_TOKENS: usize = 256;

/// How many transcript characters fit one LLM request. Lengths are bytes of
/// UTF-8 text (the unit of `CleanedTranscript::cleaned_char_count`), which is
/// never less than the number of characters, so the budget errs on the safe side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    /// A transcript up to this length is judged in one request. Also the
    /// limit for all section notes together in the final judgment request.
    pub single_chars: usize,
    /// Longest section of a transcript that is read in sections.
    pub section_chars: usize,
}

impl ContextBudget {
    /// Budget for a context window: what is left after the system prompt, the
    /// answer and the request overhead, times `chars_per_token`. A fixed
    /// `override_chars` (YTC_LLM_CONTEXT_CHARS) replaces both results.
    pub fn compute(context_tokens: usize, chars_per_token: f64, override_chars: Option<usize>) -> Self {
        if let Some(chars) = override_chars {
            let chars = chars.clamp(MIN_LLM_CONTEXT_CHARS, MAX_LLM_CONTEXT_CHARS);
            return Self { single_chars: chars, section_chars: chars };
        }
        let context = context_tokens.clamp(MIN_LLM_CONTEXT_TOKENS, MAX_LLM_CONTEXT_TOKENS);
        let ratio = if chars_per_token.is_finite() {
            chars_per_token.clamp(MIN_CHARS_PER_TOKEN, MAX_CHARS_PER_TOKEN)
        } else {
            DEFAULT_CHARS_PER_TOKEN
        };
        let tokens_of = |text: &str| (text.len() as f64 / ratio).ceil() as usize;
        let chars_left = |answer_tokens: u32, system_prompt: &str| {
            let used = answer_tokens as usize + tokens_of(system_prompt) + REQUEST_OVERHEAD_TOKENS;
            ((context.saturating_sub(used) as f64 * ratio) as usize).max(MIN_LLM_CONTEXT_CHARS)
        };
        // One section size for both kinds of notes request: budget for the longer prompt.
        let notes_prompt = if NOTES_SYSTEM_PROMPT.len() >= MERGE_SYSTEM_PROMPT.len() {
            NOTES_SYSTEM_PROMPT
        } else {
            MERGE_SYSTEM_PROMPT
        };
        Self {
            single_chars: chars_left(JUDGMENT_MAX_TOKENS, JUDGMENT_SYSTEM_PROMPT),
            section_chars: chars_left(NOTES_MAX_TOKENS, notes_prompt),
        }
    }

    /// Budget from raw env values. Unset, empty or unreadable values fall
    /// back to the defaults.
    fn from_raw(context_tokens: Option<&str>, chars_per_token: Option<&str>, context_chars: Option<&str>) -> Self {
        let tokens = context_tokens
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_LLM_CONTEXT_TOKENS);
        let ratio = chars_per_token
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|r| r.is_finite() && *r > 0.0)
            .unwrap_or(DEFAULT_CHARS_PER_TOKEN);
        let override_chars = context_chars.and_then(|v| v.trim().parse::<usize>().ok());
        Self::compute(tokens, ratio, override_chars)
    }

    /// Budget from YTC_LLM_CONTEXT_TOKENS, YTC_LLM_CHARS_PER_TOKEN and
    /// YTC_LLM_CONTEXT_CHARS.
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok();
        Self::from_raw(
            var(LLM_CONTEXT_TOKENS_ENV).as_deref(),
            var(LLM_CHARS_PER_TOKEN_ENV).as_deref(),
            var(LLM_CONTEXT_CHARS_ENV).as_deref(),
        )
    }
}

/// How an analysis read its transcript. It always reads all of it: an
/// analysis that cannot read a section fails instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisCoverage {
    /// Length of the transcript in bytes of the cleaned text, the same unit
    /// as `CleanedTranscript::cleaned_char_count`.
    pub transcript_chars: usize,
    /// Consecutive sections the transcript was read in (1 = one request).
    pub sections: usize,
}

impl AnalysisCoverage {
    /// An analysis that read all of `text` in `sections` sections.
    fn whole(text: &str, sections: usize) -> Self {
        Self { transcript_chars: text.len(), sections }
    }
}

/// Result of `VideoAnalyzer::analyze_with_coverage`.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// `is_worth_keeping` is the quality-and-topic part of the keep decision
    /// (topic ok AND score at or below the threshold). The pipeline adds
    /// "enough information" and "not disqualified by hand".
    pub insight: InsightResult,
    /// One of `models::TOPIC_VERDICTS`. None when no model judged the topic
    /// (heuristic engine).
    pub topic_verdict: Option<String>,
    pub topic_reason: Option<String>,
    pub coverage: AnalysisCoverage,
}

/// Largest index <= `idx` that is a char boundary of `s`.
fn floor_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Where to end the first section of `rest`: after the last sentence end
/// (terminal punctuation or a line break, followed by whitespace) at or
/// before `hi`; without one, after the last whitespace (auto-captions have no
/// punctuation; cues are joined by spaces); without any, at `hi` itself.
/// Boundaries in the first half of the ideal length are ignored so that one
/// early full stop does not produce a tiny section.
fn section_cut(rest: &str, ideal: usize, max: usize) -> usize {
    let first_char = rest.chars().next().map_or(0, char::len_utf8);
    // At least one character per section, so the split always advances.
    let hi = floor_boundary(rest, max.min(ideal + max / 10)).max(first_char);
    let lo = ideal / 2;

    let mut last_sentence = None;
    let mut last_space = None;
    let mut prev: Option<char> = None;
    for (i, c) in rest[..hi].char_indices() {
        if c.is_whitespace() {
            let end = i + c.len_utf8();
            if end > lo {
                last_space = Some(end);
                if c == '\n' || matches!(prev, Some('.' | '!' | '?' | '…' | '。')) {
                    last_sentence = Some(end);
                }
            }
        }
        prev = Some(c);
    }
    last_sentence.or(last_space).unwrap_or(hi)
}

/// Splits a transcript into consecutive sections of at most `max_chars`
/// bytes each. The sections are slices of `text`: concatenated they give the
/// text back exactly, so nothing is lost and nothing is read twice. Cuts are
/// made at sentence ends where there are any, else between words, and never
/// inside a character. The sections come out about equally long (no tiny
/// last one). A text within the limit is one section.
pub fn split_into_sections(text: &str, max_chars: usize) -> Vec<&str> {
    let max = max_chars.max(1);
    let mut sections = Vec::new();
    let mut rest = text;
    while rest.len() > max {
        let sections_left = rest.len().div_ceil(max);
        let ideal = rest.len().div_ceil(sections_left);
        let (head, tail) = rest.split_at(section_cut(rest, ideal, max));
        sections.push(head);
        rest = tail;
    }
    if !rest.is_empty() || sections.is_empty() {
        sections.push(rest);
    }
    sections
}

/// Notes on a run of consecutive transcript sections (`first..=last`, 1-based).
#[derive(Debug, Clone, PartialEq, Eq)]
struct SectionNote {
    first: usize,
    last: usize,
    text: String,
}

impl SectionNote {
    fn heading(&self, total: usize) -> String {
        if self.first == self.last {
            format!("### Notes on section {} of {}", self.first, total)
        } else {
            format!("### Notes on sections {}-{} of {}", self.first, self.last, total)
        }
    }
}

/// All notes in order, each under its heading: the body of the final
/// judgment request (and of a merge request).
fn render_notes(notes: &[SectionNote], total: usize) -> String {
    notes
        .iter()
        .map(|n| format!("{}\n{}", n.heading(total), n.text.trim()))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Plans one round of condensing: consecutive notes are grouped so that a
/// group stays within `max_group_chars`. Every group has at least two notes
/// (that is what makes a round shrink the list), except a single note left
/// over at the end, which is carried over unchanged.
fn merge_groups(lens: &[usize], max_group_chars: usize) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    while start < lens.len() {
        let mut end = start + 1;
        let mut size = lens[start];
        while end < lens.len() && (end - start < 2 || size + lens[end] <= max_group_chars) {
            size += lens[end];
            end += 1;
        }
        groups.push(start..end);
        start = end;
    }
    groups
}

/// The text after a reasoning model's `</think>` block (Qwen3, DeepSeek-R1).
fn strip_think(raw: &str) -> &str {
    match raw.find("</think>") {
        Some(end) => &raw[end + "</think>".len()..],
        None => raw,
    }
}

/// Notes text of an answer, or why it cannot be used.
fn usable_notes(content: &str) -> Result<String> {
    let notes = strip_think(content).trim();
    if notes.is_empty() {
        anyhow::bail!("LLM returned empty notes");
    }
    // A reasoning block that was never closed: the model was still thinking
    // when the answer ended, so this text is not notes on the section.
    if notes.contains("<think>") {
        anyhow::bail!("LLM returned an unfinished reasoning block instead of notes");
    }
    Ok(notes.to_string())
}

/// Added to a notes request that is repeated because the first answer was
/// cut off at `NOTES_MAX_TOKENS`.
const BRIEF_NOTES_REMINDER: &str =
    "Your last answer was too long and was cut off. Be brief: short bullet lines, at most 120 words in total.";

/// The answer text of one chat completion.
///
/// It is `message.content`. `reasoning_content` (the chain of thought of a
/// reasoning model) is used only when the content is empty AND the answer
/// finished normally: some servers put the whole answer there. For an answer
/// that was cut off ("length") with empty content the result is an empty
/// string: the model was still thinking, and its thoughts are not an answer.
/// The callers refuse a cut-off answer by its finish reason.
fn answer_text(message: &ChatCompletionResponseMessage, finish_reason: Option<&str>) -> Result<String> {
    let filled = |text: &Option<String>| text.as_deref().filter(|t| !t.trim().is_empty()).map(str::to_string);
    if let Some(content) = filled(&message.content) {
        return Ok(content);
    }
    if finish_reason == Some("length") {
        return Ok(String::new());
    }
    let finished_normally = matches!(finish_reason, None | Some("stop"));
    match filled(&message.reasoning_content) {
        Some(reasoning) if finished_normally => Ok(reasoning),
        _ => Err(anyhow::anyhow!(
            "Empty content in LLM choices (finish_reason: {})",
            finish_reason.unwrap_or("not reported")
        )),
    }
}

// The placeholders are replaced before sending (`judgment_system_prompt`):
// {MAX_SLOP} by the threshold in use, {CATEGORY_SECTION} and {CATEGORY_KEY} by
// the user's categories, or by nothing when there are none.
const JUDGMENT_SYSTEM_PROMPT: &str = r#"You are YTC (YouTube Knowledge Curator). You judge ONE video and return ONE JSON object.
The user message holds either the complete transcript or, for a long video, notes on every consecutive section of the transcript. Judge the whole video either way.

You make TWO INDEPENDENT judgments. Never mix them.

JUDGMENT 1 - TOPIC ("topic_verdict"): is the subject itself acceptable?
- "ok": any real field of knowledge or practice. History, mathematics, physics, biology, medicine, economics, law, philosophy, languages, technology, music, cooking, crafts, sport, travel: all "ok".
- "fringe": the subject itself is paranormal, pseudoscience or conspiracy belief presented as true: astrology, horoscopes, tarot, psychics, telepathy, near-death or afterlife testimony, past lives, manifestation, UFO or alien visitation claims, flat earth, miracle cures.
- "nonsense": no informational intent at all: pranks, pure reaction or noise, challenges, gibberish, ambience, clips without commentary.
- "politics": partisan politics, election campaigns, commentary on parties or politicians, culture-war debate, commentary on current geopolitical conflicts.
A sober treatment of history, science, economics or religion is "ok" even when it involves wars, governments or beliefs: a lecture on the Roman Senate or on the causes of World War 1 is "ok"; a scientist explaining why astrology fails is "ok". When unsure, answer "ok".

JUDGMENT 2 - QUALITY ("slop_score", integer 0-100): how much real information does the video deliver for its length, whatever its subject?
- 0-20: dense and concrete. Specific facts, numbers, mechanisms, worked examples, steps, original analysis or first-hand experience. Almost no filler.
- 21-45: solid. Clear substance and useful detail, with some padding or repetition.
- 46-70: thin. A few real points stretched out: generic or second-hand statements, long intros, repetition, claims without evidence.
- 71-100: slop. Clickbait or hype with little behind it, engagement bait, recycled lists, empty motivation, a sales pitch without content.
The subject NEVER raises or lowers the slop score. Off-topic content is NOT scored higher: its topic_verdict says that. Score a fringe or political video on its information density like any other.

EXAMPLES:
- A dense university lecture on the Peloponnesian War, or on linear algebra: topic_verdict "ok", slop_score about 10.
- "This ONE trick CHANGES EVERYTHING": fifteen minutes of hype without one concrete detail: topic_verdict "ok", slop_score about 85.
- A weekly astrology reading: topic_verdict "fringe".
- A cooking video that gives quantities, temperatures and times for every step: "ok", about 15. "The secret recipe chefs don't want you to know", with no quantities and no method: "ok", about 90.
{CATEGORY_SECTION}
Return a STRICT JSON object with these exact keys:
{
  "topic_verdict": <"ok", "fringe", "nonsense" or "politics">,
  "topic_reason": <one sentence: what the video is about and why this verdict>,
{CATEGORY_KEY}  "slop_score": <integer 0-100, quality only, see JUDGMENT 2>,
  "is_worth_keeping": <boolean, true if topic_verdict is "ok" and slop_score <= {MAX_SLOP}>,
  "quality_rating": <"High Signal", "Moderate Signal", or "Low Signal / Slop">,
  "summary": <concise 2-3 sentence overview of the core argument or thesis>,
  "key_insights": [<3-6 strings: specific, concrete insights or findings>],
  "takeaways": [<2-4 items the viewer can use, test, or remember>],
  "quotes": [<0-3 notable verbatim quotes from the speaker>],
  "topics": [<3-6 concise concept tags for the wiki, e.g. "Roman Republic", "Fourier Transform", "Sourdough Starter">],
  "mentioned_creators": [<names or handles of other creators, guests, or channels mentioned, or empty list>]
}"#;

/// The two pieces of the judgment prompt that exist only when the user has
/// defined categories: the section that lists them (name and description)
/// and the line of the JSON shape that asks for one. Both are empty without
/// categories, so the prompt then does not mention categories at all.
fn category_prompt_parts(categories: &[Category]) -> (String, String) {
    if categories.is_empty() {
        return (String::new(), String::new());
    }
    let mut section = String::from(
        "\nCATEGORY (\"category\"): a label only. It never affects the score or the verdict. Choose the ONE category from this list that fits the video best, written exactly as listed, or null when none fits:\n",
    );
    section.push_str(&category_list_lines(categories));
    let key = "  \"category\": <one of the category names listed above, or null>,\n".to_string();
    (section, key)
}

/// The user's categories as the prompts list them: one line per category
/// with its name and, when it has one, its description.
fn category_list_lines(categories: &[Category]) -> String {
    let mut lines = String::new();
    for category in categories {
        match category.description.as_deref().map(str::trim).filter(|d| !d.is_empty()) {
            Some(description) => lines.push_str(&format!("- \"{}\": {}\n", category.name, description)),
            None => lines.push_str(&format!("- \"{}\"\n", category.name)),
        }
    }
    lines
}

// {CATEGORY_LIST} is replaced before sending (`categorize_system_prompt`) by
// the user's categories.
/// System prompt of a category request (`VideoAnalyzer::categorize`): the
/// model gets the stored insight of an analysed video, not its transcript,
/// and answers with a category only.
const CATEGORIZE_SYSTEM_PROMPT: &str = r#"You sort ONE video of a video library into the owner's categories. The user message holds the title of the video and what an earlier analysis stored about it: a summary, its key insights and its topic tags.
A category is a label only. Do not judge or score the video.
Choose the ONE category from this list that fits the video best, written exactly as listed, or null when none fits:
{CATEGORY_LIST}
Return a STRICT JSON object with this exact key and nothing else:
{
  "category": <one of the category names listed above, or null>
}"#;

/// The system prompt of a category request for the user's categories.
fn categorize_system_prompt(categories: &[Category]) -> String {
    CATEGORIZE_SYSTEM_PROMPT.replace("{CATEGORY_LIST}\n", &category_list_lines(categories))
}

/// The user message of a category request: title, summary, key insights and
/// topics of a stored insight. Parts the insight does not have are left out.
fn categorize_user_prompt(video_title: &str, insight: &InsightResult) -> String {
    let mut text = format!("Video Title: {}\n\nSummary: {}\n", video_title.trim(), insight.summary.trim());
    if !insight.key_insights.is_empty() {
        text.push_str("\nKey insights:\n");
        for item in &insight.key_insights {
            text.push_str(&format!("- {}\n", item.trim()));
        }
    }
    if !insight.topics.is_empty() {
        text.push_str(&format!("\nTopics: {}\n", insight.topics.join(", ")));
    }
    text.push_str("\nProvide the JSON answer now.");
    text
}

/// The system prompt of the judgment request for a keep threshold and the
/// user's categories.
fn judgment_system_prompt(max_slop: u32, categories: &[Category]) -> String {
    let (section, key) = category_prompt_parts(categories);
    JUDGMENT_SYSTEM_PROMPT
        .replace("{MAX_SLOP}", &max_slop.to_string())
        .replace("{CATEGORY_SECTION}", &section)
        .replace("{CATEGORY_KEY}", &key)
}

/// System prompt of a section request. No score is asked for: the judgment
/// is made once, from the notes on all sections.
const NOTES_SYSTEM_PROMPT: &str = r#"You are reading ONE section of a longer video transcript. Other sections are read separately; a final step judges the whole video from the notes on all sections.
Write compact notes on THIS section only, as plain text:
- Short bullet lines with everything concrete: facts, claims, numbers, names, tools, examples, arguments, how-to steps.
- One line starting "TOPIC:" that says what this section is about.
- One line starting "FILLER:" that says how much of this section is padding, repetition, sponsor or self-promotion talk, or engagement bait (none / some / most), with a short example if there is any.
- Optionally one line starting "QUOTE:" with one short verbatim sentence worth quoting.
Do not score or judge the video. Do not add anything that is not in the text. At most 200 words."#;

/// System prompt of a condensing request, used only when the notes on all
/// sections together do not fit the final judgment request.
const MERGE_SYSTEM_PROMPT: &str = r#"You are condensing notes on consecutive sections of one video transcript into ONE shorter set of notes. A final step judges the whole video from it.
Keep, as plain text:
- Short bullet lines with the concrete content of all the notes, in order: facts, claims, numbers, names, tools, examples, arguments, how-to steps. Drop duplicates, never invent.
- One line starting "TOPIC:" for what these sections are about.
- One line starting "FILLER:" that sums up how much padding, repetition, promotion or engagement bait the notes report (none / some / most).
- At most one line starting "QUOTE:", copied unchanged from the notes.
Do not score or judge the video. At most 200 words."#;

/// Health endpoint of an OpenAI-compatible base URL: llama-server serves
/// `/health` next to `/v1`, not below it.
fn health_url(base_url: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    format!("{}/health", base.strip_suffix("/v1").unwrap_or(base))
}

/// True for an LLM answer status that points at the configuration (wrong API
/// key: 401/403; wrong model name or URL: 404). Such a request fails for every
/// video alike. 400 is not in the list: llama-server answers it when one
/// transcript does not fit the context window, which is specific to the video.
fn is_llm_config_status(status: u16) -> bool {
    matches!(status, 401 | 403 | 404)
}

/// Why a parsed LLM answer must not be stored, or None when it is usable.
/// A cut-off or score-only answer still parses after the JSON repair, with
/// every missing field empty; stored, it would replace a good insight.
fn unusable_answer_reason(finish_reason: Option<&str>, summary: &str) -> Option<&'static str> {
    // (A cut-off answer without any text is refused before this, in `judgment_from_answer`.)
    if finish_reason == Some("length") {
        return Some("LLM answer was cut off at max_tokens");
    }
    // key_insights may be empty: a plain slop verdict can have none.
    if summary.trim().is_empty() {
        return Some("LLM answer has no summary");
    }
    None
}

/// The configured LLM could not be reached, is still loading its model, or timed
/// out. This is different from "the LLM answered with something unusable": the
/// video must stay unprocessed so a later crawl analyses it properly.
#[derive(Debug)]
pub struct LlmUnavailable {
    pub reason: String,
    /// true: nothing is listening, or the model is still loading (HTTP 503), so
    /// no video can be analysed right now. false: this one request failed
    /// (timeout, other 5xx) and the next video may well work. Also false for a
    /// configuration error (HTTP 401/403/404): the engine answers, but the same
    /// way for every video, so two such answers in a row stop a crawl.
    pub engine_down: bool,
}

/// Start of `LlmUnavailable::reason` for a request that ran into the request
/// timeout (see `LlmUnavailable::is_timeout`).
const TIMEOUT_REASON_PREFIX: &str = "request timed out after ";

impl LlmUnavailable {
    /// True when the engine was reachable but one request did not finish
    /// within the request timeout (`LLM_TIMEOUT_SECS_ENV`).
    pub fn is_timeout(&self) -> bool {
        self.reason.starts_with(TIMEOUT_REASON_PREFIX)
    }
}

impl std::fmt::Display for LlmUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_timeout() {
            // The engine is there: "not available" would send the user looking
            // for a stopped server instead of a too short timeout.
            write!(f, "LLM {}. Video left unprocessed for a later retry.", self.reason)
        } else {
            write!(f, "LLM engine not available ({}). Video left unprocessed for a later retry.", self.reason)
        }
    }
}

impl std::error::Error for LlmUnavailable {}

/// Env var: seconds one LLM request (a section, a judgment) may take.
pub const LLM_TIMEOUT_SECS_ENV: &str = "YTC_LLM_TIMEOUT_SECS";
/// Timeout of a request of ordinary size when the env var is not set. A
/// request carries a whole section (about 21,000 characters at the default
/// context size); on a CPU or a partly offloaded model that takes several
/// minutes.
pub const DEFAULT_LLM_TIMEOUT_SECS: u64 = 600;
/// Request size the default timeout is meant for. A larger request (a larger
/// context window is configured) gets proportionally more time.
const TIMEOUT_BASE_REQUEST_CHARS: usize = 24_000;
/// Upper limit of any request timeout: one day.
const MAX_LLM_TIMEOUT_SECS: u64 = 86_400;

/// The request timeout when `LLM_TIMEOUT_SECS_ENV` is not set: scaled to the
/// size of the request (system prompt plus user message, in bytes), never
/// below `DEFAULT_LLM_TIMEOUT_SECS`.
fn default_timeout_secs(request_chars: usize) -> u64 {
    let steps = request_chars.div_ceil(TIMEOUT_BASE_REQUEST_CHARS).max(1) as u64;
    DEFAULT_LLM_TIMEOUT_SECS.saturating_mul(steps).min(MAX_LLM_TIMEOUT_SECS)
}

/// The fixed request timeout from the raw env value: whole seconds, at least
/// 1. None (use the scaled default) when unset, empty, zero or unreadable.
fn timeout_from_raw(raw: Option<&str>) -> Option<std::time::Duration> {
    let secs = raw?.trim().parse::<u64>().ok().filter(|s| *s > 0)?;
    Some(std::time::Duration::from_secs(secs.min(MAX_LLM_TIMEOUT_SECS)))
}

/// "600" or "0.3": a duration in seconds for a message.
fn seconds_text(duration: std::time::Duration) -> String {
    if duration.subsec_nanos() == 0 {
        duration.as_secs().to_string()
    } else {
        format!("{:.1}", duration.as_secs_f64())
    }
}

/// The fixed request timeout from `LLM_TIMEOUT_SECS_ENV`, if set.
fn timeout_from_env() -> Option<std::time::Duration> {
    timeout_from_raw(std::env::var(LLM_TIMEOUT_SECS_ENV).ok().as_deref())
}

#[derive(Clone)]
pub struct VideoAnalyzer {
    client: Client,
    llm_url: Option<String>,
    llm_key: Option<String>,
    llm_model: String,
    /// A video is kept only when its slop score is at or below this value.
    max_slop: u32,
    /// How much transcript text one request may carry.
    budget: ContextBudget,
    /// Fixed timeout of one chat request (YTC_LLM_TIMEOUT_SECS). None: the
    /// default, scaled to the size of each request (`default_timeout_secs`).
    request_timeout: Option<std::time::Duration>,
    /// The user's categories the model may choose from. Empty: the prompt
    /// does not mention categories and no insight gets one.
    categories: Vec<Category>,
}

impl Default for VideoAnalyzer {
    fn default() -> Self {
        let llm_url = std::env::var("YTC_LLM_URL").ok();
        let llm_key = std::env::var("YTC_LLM_KEY").ok();
        let llm_model = std::env::var("YTC_LLM_MODEL").unwrap_or_else(|_| "llama3.2".to_string());

        Self {
            // No client-wide timeout: every request sets its own (`timeout_for`;
            // 2 s for the health check), because a whole section on slow
            // hardware needs far longer than a short request.
            client: Client::new(),
            llm_url,
            llm_key,
            llm_model,
            max_slop: crate::pipeline::KEEP_THRESHOLD,
            budget: ContextBudget::from_env(),
            request_timeout: timeout_from_env(),
            categories: Vec::new(),
        }
    }
}

impl VideoAnalyzer {
    pub fn new(llm_url: Option<String>, llm_key: Option<String>, llm_model: Option<String>) -> Self {
        Self {
            // No client-wide timeout, see `Default`.
            client: Client::new(),
            llm_url,
            llm_key,
            llm_model: llm_model.unwrap_or_else(|| "llama3.2".to_string()),
            max_slop: crate::pipeline::KEEP_THRESHOLD,
            budget: ContextBudget::from_env(),
            request_timeout: timeout_from_env(),
            categories: Vec::new(),
        }
    }

    /// Fixes the timeout of one chat request, replacing YTC_LLM_TIMEOUT_SECS
    /// and the scaled default.
    #[cfg(test)]
    pub fn with_request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// How long a chat request of `request_chars` bytes may take.
    fn timeout_for(&self, request_chars: usize) -> std::time::Duration {
        self.request_timeout
            .unwrap_or_else(|| std::time::Duration::from_secs(default_timeout_secs(request_chars)))
    }

    /// The error of a request that got no complete HTTP answer. A timeout is
    /// named as one, with the setting that raises it: the engine is reachable
    /// (`engine_down` false), this one request took too long. Connection
    /// refused means nothing is listening; `answer_started` (the response
    /// headers arrived) rules that out.
    fn transport_error(
        endpoint: &str,
        err: &reqwest::Error,
        timeout: std::time::Duration,
        answer_started: bool,
    ) -> anyhow::Error {
        let reason = if err.is_timeout() {
            format!(
                "{}{} s: {} did not finish its answer in time. To allow more time set {} to a larger number of seconds; to send smaller requests lower {}",
                TIMEOUT_REASON_PREFIX,
                seconds_text(timeout),
                endpoint,
                LLM_TIMEOUT_SECS_ENV,
                LLM_CONTEXT_CHARS_ENV
            )
        } else {
            format!("{} - {}", endpoint, err)
        };
        anyhow::Error::new(LlmUnavailable {
            engine_down: !answer_started && !err.is_timeout() && err.is_connect(),
            reason,
        })
    }

    /// Sets the categories the model may choose from (see `Database::list_categories`).
    pub fn with_categories(mut self, categories: Vec<Category>) -> Self {
        self.categories = categories;
        self
    }

    /// Characters of transcript or notes the judgment request may carry: the
    /// budget less the room the category list takes in its system prompt.
    fn judgment_chars(&self) -> usize {
        let (section, key) = category_prompt_parts(&self.categories);
        self.budget.single_chars.saturating_sub(section.len() + key.len())
    }

    /// Sets the keep threshold (0..=100) used for the keep flag and written into
    /// the LLM prompt.
    pub fn with_max_slop(mut self, max_slop: u32) -> Self {
        self.max_slop = max_slop.min(100);
        self
    }

    /// Replaces the context budget read from the environment.
    #[cfg(test)]
    pub fn with_context_budget(mut self, budget: ContextBudget) -> Self {
        self.budget = budget;
        self
    }

    /// The context budget in use.
    pub fn context_budget(&self) -> ContextBudget {
        self.budget
    }

    /// The keep threshold in use.
    #[cfg(test)]
    pub fn max_slop(&self) -> u32 {
        self.max_slop
    }

    /// The configured LLM base URL, if any.
    fn configured_url(&self) -> Option<&str> {
        self.llm_url.as_deref().map(str::trim).filter(|u| !u.is_empty())
    }

    /// True when an LLM endpoint is configured (it may still be down).
    pub fn has_llm(&self) -> bool {
        self.configured_url().is_some()
    }

    /// (mode, model, url): ("llm", model name, base URL) or
    /// ("heuristic", "", "") when no LLM endpoint is configured.
    pub fn describe(&self) -> (String, String, String) {
        match self.configured_url() {
            Some(url) => ("llm".to_string(), self.llm_model.clone(), url.to_string()),
            None => (HEURISTIC_ANALYZER.to_string(), String::new(), String::new()),
        }
    }

    /// Engine name for log lines: "LLM <model>" or the heuristic.
    pub fn engine_label(&self) -> String {
        if self.has_llm() {
            format!("LLM {}", self.llm_model)
        } else {
            "word-count heuristic (no LLM configured)".to_string()
        }
    }

    /// True when the configured LLM server answers within 2 seconds: first
    /// `<base without /v1>/health` (llama-server; 503 while the model loads),
    /// then `<base>/models` for servers without a health route. False when no
    /// LLM is configured.
    pub async fn llm_reachable(&self) -> bool {
        let Some(base_url) = self.configured_url() else {
            return false;
        };
        let timeout = std::time::Duration::from_secs(2);
        let models_url = format!("{}/models", base_url.trim_end_matches('/'));
        for url in [health_url(base_url), models_url] {
            let mut req = self.client.get(&url).timeout(timeout);
            if let Some(ref key) = self.llm_key {
                req = req.bearer_auth(key);
            }
            if let Ok(resp) = req.send().await {
                if resp.status().is_success() {
                    return true;
                }
            }
        }
        false
    }

    /// Analyzes a cleaned transcript with the LLM.
    /// - No LLM configured: the heuristic engine is used.
    /// - LLM configured but down / loading / timed out: returns `LlmUnavailable`
    ///   so the video is NOT stored as processed with a heuristic score.
    /// - LLM answered but the answer could not be used: one retry, then a plain
    ///   error. Only this video is marked failed and a later crawl tries it again.
    ///   The heuristic is not used here: its "summary" of auto-captions is a few
    ///   ranked sentences, which must not be stored as the insight of an LLM run.
    ///
    /// Returns the insight with the topic verdict and how the transcript was
    /// read.
    ///
    /// The WHOLE transcript is read. One that fits the context budget goes
    /// into a single request. A longer one is split into consecutive sections
    /// (`split_into_sections`); the model takes notes on every section, and
    /// the judgment is made from the title and all notes. Nothing is sampled:
    /// when a section cannot be read the analysis fails.
    pub async fn analyze_with_coverage(&self, video_title: &str, transcript: &CleanedTranscript) -> Result<Analysis> {
        if let Some(base_url) = self.configured_url() {
            let (material, sections) = self.read_transcript(base_url, video_title, &transcript.text).await?;
            let coverage = AnalysisCoverage::whole(&transcript.text, sections);

            let mut attempt = 1;
            loop {
                match self.judge(base_url, video_title, &transcript.video_id, &material).await {
                    Ok((insight, verdict, topic_reason)) => {
                        return Ok(Analysis {
                            insight,
                            topic_verdict: Some(verdict.to_string()),
                            topic_reason,
                            coverage,
                        })
                    }
                    Err(err) if err.downcast_ref::<LlmUnavailable>().is_some() => return Err(err),
                    Err(err) if attempt < 2 => {
                        eprintln!("[Analyzer] LLM answer unusable ({}). Retrying once.", err);
                        attempt += 1;
                    }
                    Err(err) => {
                        return Err(err.context(
                            "LLM answer unusable after 2 attempts. Video left unprocessed for a later retry",
                        ));
                    }
                }
            }
        }

        // Built-in heuristic (no LLM configured). It reads the whole text and
        // cannot judge the subject: no topic verdict is recorded.
        Ok(Analysis {
            insight: self.analyze_heuristically(video_title, transcript),
            topic_verdict: None,
            topic_reason: None,
            coverage: AnalysisCoverage::whole(&transcript.text, 1),
        })
    }

    /// Reads the whole transcript and returns what the judgment request will
    /// carry, with the number of sections it was read in.
    async fn read_transcript(&self, base_url: &str, video_title: &str, text: &str) -> Result<(Material, usize)> {
        if text.len() <= self.judgment_chars() {
            return Ok((Material::Transcript(text.to_string()), 1));
        }

        let sections = split_into_sections(text, self.budget.section_chars);
        let total = sections.len();
        println!(
            "    {} The transcript ({} characters) does not fit one request: reading all of it in {} sections.",
            "📖".dimmed(),
            text.len(),
            total
        );

        let mut notes: Vec<SectionNote> = Vec::with_capacity(total);
        for (idx, section) in sections.iter().enumerate() {
            let number = idx + 1;
            println!("    {} Reading section {} of {} ...", "📖".dimmed(), number, total);
            let user = format!(
                "Video Title: {}\n\nSection {} of {} of the transcript:\n{}\n\nWrite the notes on this section now.",
                video_title, number, total, section
            );
            let what = format!("Section {} of {}", number, total);
            let note = self.notes_request(base_url, NOTES_SYSTEM_PROMPT, &user, &what).await?;
            notes.push(SectionNote { first: number, last: number, text: note });
        }

        let notes = self.condense_notes(base_url, video_title, notes, total).await?;
        Ok((Material::Notes { text: render_notes(&notes, total), sections: total }, total))
    }

    /// Makes the notes fit the final judgment request. Usually they do. When
    /// there are too many, consecutive notes are condensed in groups, round
    /// after round, until they fit: every section stays represented.
    async fn condense_notes(
        &self,
        base_url: &str,
        video_title: &str,
        mut notes: Vec<SectionNote>,
        total: usize,
    ) -> Result<Vec<SectionNote>> {
        let limit = self.judgment_chars();
        let mut round = 0;
        loop {
            let size = render_notes(&notes, total).len();
            if size <= limit {
                return Ok(notes);
            }
            if notes.len() < 2 {
                anyhow::bail!(
                    "The notes ({} characters) do not fit the context budget of {} characters. Raise {} or {}",
                    size,
                    limit,
                    LLM_CONTEXT_TOKENS_ENV,
                    LLM_CONTEXT_CHARS_ENV
                );
            }
            round += 1;
            println!(
                "    {} The notes on {} parts are too long for one request: condensing them (round {}) ...",
                "📖".dimmed(),
                notes.len(),
                round
            );

            let lens: Vec<usize> = notes.iter().map(|n| n.heading(total).len() + n.text.len() + 2).collect();
            let mut condensed = Vec::new();
            for group in merge_groups(&lens, self.budget.section_chars) {
                let members = &notes[group];
                if let [single] = members {
                    condensed.push(single.clone());
                    continue;
                }
                let (first, last) = (members[0].first, members[members.len() - 1].last);
                let user = format!(
                    "Video Title: {}\n\n{}\n\nCondense these notes now.",
                    video_title,
                    render_notes(members, total)
                );
                let what = format!("Notes on sections {}-{} of {}", first, last, total);
                let text = self.notes_request(base_url, MERGE_SYSTEM_PROMPT, &user, &what).await?;
                condensed.push(SectionNote { first, last, text });
            }

            // A model that answers longer than its input would loop forever.
            if render_notes(&condensed, total).len() >= size {
                anyhow::bail!(
                    "The notes on {} sections could not be condensed to fit the context budget of {} characters",
                    total,
                    limit
                );
            }
            notes = condensed;
        }
    }

    /// One plain-text notes request (a section, or a group of notes). The
    /// engine being unavailable ends the analysis at once. Any other failure
    /// is retried once and then ends it too: the video is never judged on a
    /// partial reading. Notes that were cut off at `NOTES_MAX_TOKENS` are
    /// such a failure: their end is missing, or (reasoning model) they were
    /// never begun. The retry then asks for a shorter answer.
    async fn notes_request(&self, base_url: &str, system: &str, user: &str, what: &str) -> Result<String> {
        let mut attempt = 1;
        let mut ask_for_brevity = false;
        loop {
            let prompt = if ask_for_brevity {
                format!("{}\n\n{}", user, BRIEF_NOTES_REMINDER)
            } else {
                user.to_string()
            };
            let result = match self.chat(base_url, system, prompt, NOTES_MAX_TOKENS, false).await {
                Ok(answer) if answer.finish_reason.as_deref() == Some("length") => {
                    ask_for_brevity = true;
                    Err(anyhow::anyhow!(
                        "the notes were cut off at max_tokens ({} tokens)",
                        NOTES_MAX_TOKENS
                    ))
                }
                Ok(answer) => usable_notes(&answer.content),
                Err(err) => Err(err),
            };
            match result {
                Ok(notes) => return Ok(notes),
                Err(err) if err.downcast_ref::<LlmUnavailable>().is_some() => return Err(err),
                Err(err) if attempt < 2 => {
                    eprintln!("[Analyzer] {} could not be read ({}). Retrying once.", what, err);
                    attempt += 1;
                }
                Err(err) => {
                    return Err(err.context(format!(
                        "{} could not be read after 2 attempts. A video is not judged on part of its transcript: left unprocessed for a later retry",
                        what
                    )));
                }
            }
        }
    }

    /// The judgment request: title plus the whole transcript or all notes.
    async fn judge(
        &self,
        base_url: &str,
        video_title: &str,
        video_id: &str,
        material: &Material,
    ) -> Result<(InsightResult, &'static str, Option<String>)> {
        let system_prompt = judgment_system_prompt(self.max_slop, &self.categories);
        let user_prompt = match material {
            Material::Transcript(text) => format!(
                "Video Title: {}\nVideo ID: {}\n\nComplete cleaned transcript (SponsorBlock segments removed):\n{}\n\nProvide the JSON analysis now.",
                video_title, video_id, text
            ),
            Material::Notes { text, sections } => {
                println!(
                    "    {} Judging the whole video from the notes on all {} sections ...",
                    "🧠".dimmed(),
                    sections
                );
                // Do not tell the model it has the transcript when it has notes.
                format!(
                    "Video Title: {}\nVideo ID: {}\n\nThe transcript was too long for one request. It was read completely, in {} consecutive sections. Below are the notes on every section, in order. Judge the whole video from them. In \"quotes\" use only sentences the notes mark as QUOTE.\n\n{}\n\nProvide the JSON analysis now.",
                    video_title, video_id, sections, text
                )
            }
        };

        let answer = self.chat(base_url, &system_prompt, user_prompt, JUDGMENT_MAX_TOKENS, true).await?;
        self.judgment_from_answer(video_id, &answer.content, answer.finish_reason.as_deref())
    }

    /// One chat-completions request. `json` asks the server for a JSON object.
    /// Returns `LlmUnavailable` when the engine is down, loading, timed out or
    /// misconfigured; a plain error for anything specific to this request.
    async fn chat(&self, base_url: &str, system: &str, user: String, max_tokens: u32, json: bool) -> Result<ChatAnswer> {
        let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        // From connecting until the last byte of the answer.
        let timeout = self.timeout_for(system.len() + user.len());

        let mut req_builder = self.client.post(&endpoint).timeout(timeout).json(&ChatCompletionRequest {
            model: self.llm_model.clone(),
            messages: vec![
                ChatCompletionMessage {
                    role: "system".to_string(),
                    content: system.to_string(),
                },
                ChatCompletionMessage {
                    role: "user".to_string(),
                    content: user,
                },
            ],
            temperature: 0.2,
            max_tokens: Some(max_tokens),
            response_format: json.then(|| ResponseFormat {
                format_type: "json_object".to_string(),
            }),
        });

        if let Some(ref key) = self.llm_key {
            req_builder = req_builder.bearer_auth(key);
        }

        // Connection refused, timeout, or the connection dropping mid-answer all
        // mean "no LLM answer right now", not "this video is slop".
        let resp = req_builder
            .send()
            .await
            .map_err(|e| Self::transport_error(&endpoint, &e, timeout, false))?;
        let status = resp.status();
        let body_text = resp
            .text()
            .await
            .map_err(|e| Self::transport_error(&endpoint, &e, timeout, true))?;
        if status.is_server_error() {
            let detail: String = body_text.chars().take(200).collect();
            return Err(anyhow::Error::new(LlmUnavailable {
                // llama-server answers 503 while the model is still loading.
                engine_down: status.as_u16() == 503,
                reason: format!(
                    "{} answered with a server error {}: {}",
                    endpoint,
                    status.as_u16(),
                    detail.trim()
                ),
            }));
        }
        if is_llm_config_status(status.as_u16()) {
            // Not the video's fault: the caller must not count this as a failure
            // of the video.
            let detail: String = body_text.chars().take(200).collect();
            return Err(anyhow::Error::new(LlmUnavailable {
                engine_down: false,
                reason: format!(
                    "{} answered HTTP {}: check the LLM URL, model name and API key. {}",
                    endpoint,
                    status.as_u16(),
                    detail.trim()
                ),
            }));
        }
        if status.as_u16() == 400 {
            // llama-server answers 400 when a request is larger than its context window.
            return Err(anyhow::anyhow!(
                "LLM endpoint returned HTTP {}: {} (if the request exceeded the context window: set {} to the server's real context size, or lower {})",
                status,
                body_text,
                LLM_CONTEXT_TOKENS_ENV,
                LLM_CHARS_PER_TOKEN_ENV
            ));
        }
        if !status.is_success() {
            return Err(anyhow::anyhow!("LLM endpoint returned HTTP {}: {}", status, body_text));
        }

        let resp: ChatCompletionResponse = serde_json::from_str(&body_text)
            .with_context(|| format!("Failed to parse ChatCompletionResponse from body: {}", body_text))?;
        let choice = resp
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("Empty content in LLM choices (no choice in the answer)"))?;
        let finish_reason = choice.finish_reason.clone();
        let content = answer_text(&choice.message, finish_reason.as_deref())?;
        Ok(ChatAnswer { content, finish_reason })
    }

    /// Turns the text of a judgment answer into an insight, the topic verdict
    /// (one of `models::TOPIC_VERDICTS`) and its reason. The keep flag is
    /// computed here from verdict and score; what the model put into
    /// "is_worth_keeping" is not read.
    fn judgment_from_answer(
        &self,
        video_id: &str,
        raw_content: &str,
        finish_reason: Option<&str>,
    ) -> Result<(InsightResult, &'static str, Option<String>)> {
        // A reasoning model that used up max_tokens while thinking: there is
        // no answer text at all (`answer_text` does not pass the thoughts on).
        if finish_reason == Some("length") && raw_content.trim().is_empty() {
            anyhow::bail!("LLM answer was cut off at max_tokens before any answer text was written");
        }

        // If the model is a reasoning/thinking model (e.g. Qwen3, DeepSeek-R1), strip <think>...</think>
        let content_after_think = strip_think(raw_content);

        // Extract JSON if wrapped in markdown blocks
        // A truncated answer can have its only '}' before the first '{'; slicing
        // start..=end would then panic. Take everything from '{' and let
        // repair_json close it.
        let json_str = match (content_after_think.find('{'), content_after_think.rfind('}')) {
            (Some(start), Some(end)) if end > start => &content_after_think[start..=end],
            (Some(start), _) => &content_after_think[start..],
            _ => content_after_think,
        };

        let parsed: LlmInsightJson = match serde_json::from_str(json_str) {
            Ok(p) => p,
            Err(_) => {
                let repaired = repair_json(json_str);
                match serde_json::from_str(&repaired) {
                    Ok(p) => p,
                    Err(_) => {
                        // Fallback: trim to last complete property delimiter (comma) and re-close
                        if let Some(last_comma) = json_str.rfind(',') {
                            let truncated_to_comma = repair_json(&json_str[..last_comma]);
                            serde_json::from_str(&truncated_to_comma)
                                .with_context(|| format!("Failed to parse JSON even after repair: {}", repaired))?
                        } else {
                            serde_json::from_str(&repaired)
                                .with_context(|| format!("Failed to parse JSON even after repair: {}", repaired))?
                        }
                    }
                }
            }
        };

        // An object without a score (e.g. "{}" or a truncated answer cut before the
        // score) is not an analysis. Defaulting to 0 would store it as a masterpiece.
        let slop_score = parsed
            .slop_score
            .ok_or_else(|| anyhow::anyhow!("LLM answer has no usable slop_score"))?;
        // Checked after the score so "{}" keeps its own message.
        if let Some(reason) = unusable_answer_reason(finish_reason, &parsed.summary) {
            anyhow::bail!(reason);
        }

        // A verdict that cannot be read is not "ok": the answer is unusable
        // (asked again once), so no video is kept on a misread rejection.
        let topic_verdict = parse_topic_verdict(parsed.topic_verdict.as_deref()).ok_or_else(|| {
            let shown: String = parsed.topic_verdict.as_deref().unwrap_or_default().chars().take(80).collect();
            anyhow::anyhow!(
                "LLM answer has an unrecognised topic_verdict ({:?}); expected one of: {}",
                shown,
                crate::models::TOPIC_VERDICTS.join(", ")
            )
        })?;
        // Only a name the user defined is stored, in the user's spelling;
        // anything else the model puts there is no category.
        let category = parsed.category.as_deref().and_then(|raw| match_category(&self.categories, raw));

        let insight = InsightResult {
            video_id: video_id.to_string(),
            slop_score,
            is_worth_keeping: topic_verdict == TOPIC_OK && slop_score <= self.max_slop,
            quality_rating: if parsed.quality_rating.is_empty() { default_quality() } else { parsed.quality_rating },
            summary: parsed.summary,
            key_insights: parsed.key_insights,
            takeaways: parsed.takeaways,
            quotes: parsed.quotes,
            topics: parsed.topics,
            mentioned_creators: parsed.mentioned_creators,
            analyzer_type: format!("llm:{}", self.llm_model),
            category,
        };
        Ok((insight, topic_verdict, parsed.topic_reason))
    }

    /// Asks the model for the category of an already analysed video, in one
    /// short request built from its stored insight (title, summary, key
    /// insights, topics); the transcript is not read. Returns one of the
    /// user's categories in the user's spelling, or None when the model
    /// names none of them.
    ///
    /// Errors as `chat`: `LlmUnavailable` when the engine is down, loading,
    /// timed out or misconfigured; a plain error when the answer cannot be
    /// read. Fails without a configured LLM and without categories.
    pub async fn categorize(&self, video_title: &str, insight: &InsightResult) -> Result<Option<String>> {
        let Some(base_url) = self.configured_url() else {
            anyhow::bail!("Assigning categories needs an LLM and none is configured");
        };
        if self.categories.is_empty() {
            anyhow::bail!("No categories are defined, so there is nothing to choose from");
        }
        let system_prompt = categorize_system_prompt(&self.categories);
        let user_prompt = categorize_user_prompt(video_title, insight);
        let answer = self.chat(base_url, &system_prompt, user_prompt, CATEGORIZE_MAX_TOKENS, true).await?;
        self.category_from_answer(&answer.content, answer.finish_reason.as_deref())
    }

    /// Reads the answer of a category request. The rule of the judgment
    /// answer applies: only a name the user defined counts, in the user's
    /// spelling; anything else in "category" is no category. An answer that
    /// is no JSON object at all is an error.
    fn category_from_answer(&self, raw_content: &str, finish_reason: Option<&str>) -> Result<Option<String>> {
        if finish_reason == Some("length") && raw_content.trim().is_empty() {
            anyhow::bail!("LLM answer was cut off at max_tokens before any answer text was written");
        }
        let content = strip_think(raw_content);
        let json_str = match (content.find('{'), content.rfind('}')) {
            (Some(start), Some(end)) if end > start => &content[start..=end],
            (Some(start), _) => &content[start..],
            _ => anyhow::bail!("LLM answer holds no JSON object: {}", content.chars().take(200).collect::<String>()),
        };
        let parsed: LlmCategoryJson = match serde_json::from_str(json_str) {
            Ok(parsed) => parsed,
            Err(_) => {
                let repaired = repair_json(json_str);
                serde_json::from_str(&repaired)
                    .with_context(|| format!("Failed to parse JSON even after repair: {}", repaired))?
            }
        };
        Ok(parsed.category.as_deref().and_then(|raw| match_category(&self.categories, raw)))
    }
}

/// What the judgment request carries.
enum Material {
    /// The complete cleaned transcript.
    Transcript(String),
    /// Notes on all `sections` sections of the transcript, in order.
    Notes { text: String, sections: usize },
}

/// Text and finish reason of one chat completion.
struct ChatAnswer {
    content: String,
    /// "length" when the answer was cut off at max_tokens or the context limit.
    finish_reason: Option<String>,
}

fn repair_json(input: &str) -> String {
    let mut s = input.trim().to_string();
    if !s.starts_with('{') {
        if let Some(pos) = s.find('{') {
            s = s[pos..].to_string();
        }
    }
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_str = false;
    let mut escape = false;

    for c in s.chars() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' {
            in_str = !in_str;
            continue;
        }
        if in_str {
            continue;
        }
        match c {
            '{' => open_braces += 1,
            '}' => open_braces -= 1,
            '[' => open_brackets += 1,
            ']' => open_brackets -= 1,
            _ => {}
        }
    }

    if in_str {
        s.push('"');
    }
    while open_brackets > 0 {
        s.push(']');
        open_brackets -= 1;
    }
    while open_braces > 0 {
        s.push('}');
        open_braces -= 1;
    }
    s
}

impl VideoAnalyzer {
    /// Heuristic used only when no LLM is configured. It does not understand the
    /// content, does not judge the subject and assigns no category. It is a
    /// rough guess from surface signs that hold for any subject. The score is:
    /// base 45, plus hits of a list of filler and call-to-action phrases, minus
    /// hits of a list of phrases that mark an explanation (an example, a
    /// reason, a comparison), plus/minus 10 for vocabulary diversity, plus 20
    /// for fewer than 200 words, plus 15 for "???", "!!!" or "OMG" in the title.
    ///
    /// The result says so: quality rating `HEURISTIC_QUALITY_RATING`, a summary
    /// that states no model wrote it, and transcript sentences stored as
    /// excerpts. The pipeline stores the video with status "heuristic", so a
    /// later run with an LLM analyses it properly.
    pub fn analyze_heuristically(&self, video_title: &str, transcript: &CleanedTranscript) -> InsightResult {
        let text = &transcript.text;
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len().max(1);

        // Phrase lists: filler and calls to action on one side, signs of an
        // explanation on the other. Neither names a subject.
        let slop_phrases = [
            "smash the like",
            "hit subscribe",
            "leave a comment",
            "before we begin",
            "before we dive in",
            "you won't believe",
            "mind blowing",
            "this changes everything",
            "secret trick",
            "nobody is talking about",
            "in today's video",
            "what do you think",
            "let me know in the comments",
            "link in the description",
            "check out the link",
            "turn on notifications",
        ];

        let substance_markers = [
            "for example",
            "for instance",
            "in contrast",
            "on the other hand",
            "compared to",
            "specifically",
            "in other words",
            "the reason why",
            "which means",
            "as a result",
            "consequently",
            "resulted in",
            "according to",
        ];

        let lower_text = text.to_lowercase();
        let mut slop_hits = 0;
        for phrase in &slop_phrases {
            let count = lower_text.matches(phrase).count();
            slop_hits += count * 5;
        }

        let mut substance_hits = 0;
        for marker in &substance_markers {
            let count = lower_text.matches(marker).count();
            substance_hits += count * 4;
        }

        // Vocabulary diversity: Type-Token Ratio (TTR)
        let mut unique_words = HashSet::new();
        for word in &words {
            let cleaned = word.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
            if cleaned.len() > 3 {
                unique_words.insert(cleaned);
            }
        }
        let ttr = unique_words.len() as f64 / total_words as f64;

        // Base score starts at 45
        let mut score: i32 = 45;

        // Slop penalties
        score += (slop_hits as f64 * 100.0 / total_words as f64 * 50.0) as i32;

        // Title sensationalism check (excessive punctuation or "OMG")
        if video_title.contains("???") || video_title.contains("!!!") || video_title.contains("OMG") {
            score += 15;
        }

        // Short transcript penalty (less than 200 words is usually a short or minimal content)
        if total_words < 200 {
            score += 20;
        }

        // Substance bonus (reduces slop score)
        let substance_ratio = substance_hits as f64 * 100.0 / total_words as f64;
        score -= (substance_ratio * 40.0) as i32;

        // Lexical diversity adjustment (rich vocabulary lowers slop score)
        if ttr > 0.45 {
            score -= 10;
        } else if ttr < 0.25 {
            score += 10;
        }

        // Clamp to 0..100
        let final_slop_score = score.clamp(5, 95) as u32;
        // The no-LLM mode still has to keep something, so the keep flag follows
        // the score; the rating below says that nobody reviewed the content.
        let is_worth_keeping = final_slop_score <= self.max_slop;

        // Sentences ranked by summed word frequency. They are the speaker's
        // words, not conclusions: stored as excerpts, never as takeaways or as
        // "notable quotes" somebody selected.
        let sentences = extract_ranked_sentences(text);
        let summary = format!(
            "Not reviewed by an LLM. Score from phrase counts over {} words; the entries below are transcript sentences picked by word frequency, not a summary.",
            total_words
        );
        let key_insights: Vec<String> = sentences
            .iter()
            .take(HEURISTIC_EXCERPTS)
            .map(|s| format!("{}{}", HEURISTIC_EXCERPT_PREFIX, s))
            .collect();

        InsightResult {
            video_id: transcript.video_id.clone(),
            slop_score: final_slop_score,
            is_worth_keeping,
            quality_rating: HEURISTIC_QUALITY_RATING.to_string(),
            summary,
            key_insights,
            takeaways: Vec::new(),
            quotes: Vec::new(),
            topics: Vec::new(),
            mentioned_creators: Vec::new(),
            analyzer_type: HEURISTIC_ANALYZER.to_string(),
            category: None,
        }
    }
}

/// How many ranked transcript sentences a heuristic result carries.
const HEURISTIC_EXCERPTS: usize = 6;
/// Prefix of every such sentence, so no reader takes it for a model's insight.
pub const HEURISTIC_EXCERPT_PREFIX: &str = "Transcript excerpt (no LLM): ";

/// Splits text into sentences and ranks them by summed word frequency
/// (stop words excluded), normalised by sentence length. Not TextRank.
fn extract_ranked_sentences(text: &str) -> Vec<String> {
    let sentence_re = Regex::new(r"([.!?]+)\s+").unwrap();
    let raw_sentences: Vec<&str> = sentence_re.split(text).collect();

    // Word frequencies
    let mut word_freq: HashMap<String, usize> = HashMap::new();
    let stop_words: HashSet<&str> = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
        "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
        "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
        "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    ]
    .into_iter()
    .collect();

    for word in text.split_whitespace() {
        let cleaned = word.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
        if cleaned.len() > 3 && !stop_words.contains(cleaned.as_str()) {
            *word_freq.entry(cleaned).or_insert(0) += 1;
        }
    }

    let mut scored: Vec<(f64, String)> = Vec::new();
    for s in raw_sentences {
        let trimmed = s.trim();
        let word_count = trimmed.split_whitespace().count();
        // Ignore sentences that are too short or run-on
        if word_count < 7 || word_count > 45 {
            continue;
        }

        let mut score = 0.0;
        for word in trimmed.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
            if let Some(&freq) = word_freq.get(&cleaned) {
                score += freq as f64;
            }
        }

        // Normalize by sentence length
        let normalized_score = score / (word_count as f64).sqrt();
        scored.push((normalized_score, trimmed.to_string()));
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, s)| s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(text: &str) -> CleanedTranscript {
        CleanedTranscript {
            video_id: "vid".to_string(),
            raw_char_count: text.len(),
            cleaned_char_count: text.len(),
            stripped_ad_seconds: 0.0,
            stripped_segments_count: 0,
            text: text.to_string(),
        }
    }

    #[test]
    fn health_url_sits_next_to_v1() {
        assert_eq!(health_url("http://127.0.0.1:8080/v1"), "http://127.0.0.1:8080/health");
        assert_eq!(health_url("http://127.0.0.1:8080/v1/"), "http://127.0.0.1:8080/health");
        assert_eq!(health_url("http://host:1234"), "http://host:1234/health");
    }

    #[test]
    fn describe_names_the_configured_engine() {
        let llm = VideoAnalyzer::new(
            Some(" http://127.0.0.1:8080/v1 ".to_string()),
            None,
            Some("my-model".to_string()),
        );
        assert!(llm.has_llm());
        assert_eq!(
            llm.describe(),
            ("llm".to_string(), "my-model".to_string(), "http://127.0.0.1:8080/v1".to_string())
        );
        assert_eq!(llm.engine_label(), "LLM my-model");

        // An empty URL is "not configured", as in `analyze`.
        let none = VideoAnalyzer::new(Some("  ".to_string()), None, None);
        assert!(!none.has_llm());
        assert_eq!(none.describe(), ("heuristic".to_string(), String::new(), String::new()));
    }

    #[test]
    fn max_slop_setter_is_clamped_and_defaults_to_the_constant() {
        let analyzer = VideoAnalyzer::new(None, None, None);
        assert_eq!(analyzer.max_slop(), crate::pipeline::KEEP_THRESHOLD);
        assert_eq!(analyzer.clone().with_max_slop(60).max_slop(), 60);
        assert_eq!(analyzer.with_max_slop(250).max_slop(), 100);
    }

    #[test]
    fn heuristic_results_are_labelled_and_carry_excerpts_only() {
        let text = "The hydration of the dough matters because the flour decides the crumb. \
                    For example the dough was rested overnight and the crumb opened up. \
                    We tested the long rest against a short rest for three weeks. "
            .repeat(30);
        let analyzer = VideoAnalyzer::new(None, None, None);
        let result = analyzer.analyze_heuristically("Dough hydration notes", &transcript(&text));

        assert_eq!(result.analyzer_type, HEURISTIC_ANALYZER);
        assert_eq!(result.quality_rating, HEURISTIC_QUALITY_RATING);
        assert!(result.summary.starts_with("Not reviewed by an LLM"));
        assert!(result.takeaways.is_empty());
        assert!(result.quotes.is_empty());
        assert!(!result.key_insights.is_empty());
        assert!(result.key_insights.len() <= HEURISTIC_EXCERPTS);
        for excerpt in &result.key_insights {
            assert!(excerpt.starts_with(HEURISTIC_EXCERPT_PREFIX), "{}", excerpt);
        }
    }

    #[test]
    fn heuristic_keep_flag_follows_the_configured_threshold() {
        // 50 identical words: base 45, +20 (under 200 words), +10 (low diversity).
        let text = "word ".repeat(50);
        let strict = VideoAnalyzer::new(None, None, None);
        let result = strict.analyze_heuristically("Plain title", &transcript(&text));
        assert_eq!(result.slop_score, 75);
        assert!(!result.is_worth_keeping);

        let lenient = VideoAnalyzer::new(None, None, None).with_max_slop(80);
        assert!(lenient.analyze_heuristically("Plain title", &transcript(&text)).is_worth_keeping);
    }

    #[test]
    fn the_heuristic_assigns_no_category_even_when_some_are_defined() {
        let analyzer = VideoAnalyzer::new(None, None, None).with_categories(vec![category("Baking", None)]);
        let baking = transcript(&"The baking of bread starts with flour, water and salt. ".repeat(40));
        assert_eq!(analyzer.analyze_heuristically("Baking bread", &baking).category, None);
    }

    #[tokio::test]
    async fn without_an_llm_the_whole_transcript_counts_as_analysed() {
        let analyzer = VideoAnalyzer::new(None, None, None);
        let text = "Some sentence about a system that was measured carefully over time. ".repeat(400);
        let analysis = analyzer
            .analyze_with_coverage("Title", &transcript(&text))
            .await
            .expect("heuristic never fails");
        assert_eq!(analysis.insight.analyzer_type, HEURISTIC_ANALYZER);
        assert_eq!(analysis.coverage, AnalysisCoverage { transcript_chars: text.len(), sections: 1 });
        // The heuristic cannot judge a subject and does not pretend to.
        assert_eq!(analysis.topic_verdict, None);
        // No endpoint configured: not reachable, and no request is made.
        assert!(!analyzer.llm_reachable().await);
    }

    #[test]
    fn cut_off_and_summary_less_answers_are_refused() {
        assert_eq!(unusable_answer_reason(Some("stop"), "A summary."), None);
        assert_eq!(unusable_answer_reason(None, "A summary."), None);
        // Cut off at max_tokens, even when the repaired JSON has a summary.
        assert_eq!(
            unusable_answer_reason(Some("length"), "The video expl"),
            Some("LLM answer was cut off at max_tokens")
        );
        // {"slop_score": 20} parses, but it is not an analysis.
        assert_eq!(unusable_answer_reason(Some("stop"), ""), Some("LLM answer has no summary"));
        assert_eq!(unusable_answer_reason(None, "  
"), Some("LLM answer has no summary"));
    }

    #[test]
    fn finish_reason_is_read_when_the_server_sends_it() {
        let with: ChatCompletionResponse =
            serde_json::from_str(r#"{"choices":[{"message":{"content":"x"},"finish_reason":"length"}]}"#)
                .expect("valid");
        assert_eq!(with.choices[0].finish_reason.as_deref(), Some("length"));
        // Servers that omit the field, or send null, still parse.
        let without: ChatCompletionResponse =
            serde_json::from_str(r#"{"choices":[{"message":{"content":"x"}}]}"#).expect("valid");
        assert_eq!(without.choices[0].finish_reason, None);
        let null: ChatCompletionResponse =
            serde_json::from_str(r#"{"choices":[{"message":{"content":"x"},"finish_reason":null}]}"#)
                .expect("valid");
        assert_eq!(null.choices[0].finish_reason, None);
    }

    #[test]
    fn only_key_model_and_url_errors_count_as_configuration() {
        for status in [401, 403, 404] {
            assert!(is_llm_config_status(status), "{}", status);
        }
        // 400: context overflow for one transcript. 429/5xx are handled elsewhere.
        for status in [200, 400, 422, 429, 500, 503] {
            assert!(!is_llm_config_status(status), "{}", status);
        }
    }

    // ---- context budget ---------------------------------------------------

    #[test]
    fn default_budget_fits_the_default_context_window() {
        let budget = ContextBudget::compute(DEFAULT_LLM_CONTEXT_TOKENS, DEFAULT_CHARS_PER_TOKEN, None);
        let tokens = |chars: usize| (chars as f64 / DEFAULT_CHARS_PER_TOKEN).ceil() as usize;
        // Whole-transcript request: system prompt + transcript + answer + overhead.
        let judgment = tokens(JUDGMENT_SYSTEM_PROMPT.len())
            + tokens(budget.single_chars)
            + JUDGMENT_MAX_TOKENS as usize
            + REQUEST_OVERHEAD_TOKENS;
        assert!(judgment <= DEFAULT_LLM_CONTEXT_TOKENS, "{} tokens", judgment);
        // Section request, with either notes prompt.
        for prompt in [NOTES_SYSTEM_PROMPT, MERGE_SYSTEM_PROMPT] {
            let section =
                tokens(prompt.len()) + tokens(budget.section_chars) + NOTES_MAX_TOKENS as usize + REQUEST_OVERHEAD_TOKENS;
            assert!(section <= DEFAULT_LLM_CONTEXT_TOKENS, "{} tokens", section);
        }
        // Most of the window is used.
        assert!(budget.single_chars >= 12_000, "{:?}", budget);
        assert!(budget.section_chars > budget.single_chars, "{:?}", budget);
    }

    #[test]
    fn budget_follows_context_size_ratio_and_override() {
        let base = ContextBudget::compute(8192, 3.0, None);
        let wide = ContextBudget::compute(32_768, 3.0, None);
        assert!(wide.single_chars > base.single_chars * 3, "{:?}", wide);
        let cautious = ContextBudget::compute(8192, 2.0, None);
        assert!(cautious.single_chars < base.single_chars);
        assert!(cautious.section_chars < base.section_chars);

        // A context too small for prompt and answer still yields a usable minimum.
        let tiny = ContextBudget::compute(10, 3.0, None);
        assert_eq!(tiny, ContextBudget::compute(MIN_LLM_CONTEXT_TOKENS, 3.0, None));
        assert!(tiny.single_chars >= MIN_LLM_CONTEXT_CHARS);
        // Nonsense ratios are not used as they are.
        assert_eq!(ContextBudget::compute(8192, f64::NAN, None), base);
        assert_eq!(ContextBudget::compute(8192, 0.0, None), ContextBudget::compute(8192, MIN_CHARS_PER_TOKEN, None));

        // YTC_LLM_CONTEXT_CHARS fixes both figures, clamped.
        let fixed = ContextBudget::compute(8192, 3.0, Some(24_000));
        assert_eq!(fixed, ContextBudget { single_chars: 24_000, section_chars: 24_000 });
        assert_eq!(ContextBudget::compute(8192, 3.0, Some(10)).single_chars, MIN_LLM_CONTEXT_CHARS);
        assert_eq!(ContextBudget::compute(8192, 3.0, Some(999_999_999)).section_chars, MAX_LLM_CONTEXT_CHARS);
    }

    #[test]
    fn budget_comes_from_the_env_values() {
        let default = ContextBudget::compute(DEFAULT_LLM_CONTEXT_TOKENS, DEFAULT_CHARS_PER_TOKEN, None);
        assert_eq!(ContextBudget::from_raw(None, None, None), default);
        assert_eq!(ContextBudget::from_raw(Some(""), Some("many"), Some("lots")), default);
        assert_eq!(ContextBudget::from_raw(Some("-1"), Some("-3"), None), default);
        assert_eq!(
            ContextBudget::from_raw(Some(" 16384 "), Some(" 3.5 "), None),
            ContextBudget::compute(16_384, 3.5, None)
        );
        assert_eq!(
            ContextBudget::from_raw(Some("16384"), None, Some(" 24000 ")),
            ContextBudget { single_chars: 24_000, section_chars: 24_000 }
        );
    }

    // ---- splitting --------------------------------------------------------

    /// `n` different sentences, so every part of the text can be told apart.
    fn numbered_text(n: usize) -> String {
        (0..n)
            .map(|i| format!("Fact number {} is that the value was measured at {} units. ", i, i * 7))
            .collect()
    }

    fn assert_lossless(text: &str, max: usize) -> Vec<&str> {
        let sections = split_into_sections(text, max);
        assert_eq!(sections.concat(), text, "sections must give the text back exactly (max {})", max);
        for (i, section) in sections.iter().enumerate() {
            assert!(section.len() <= max, "section {} has {} bytes, limit {}", i, section.len(), max);
            assert!(!section.is_empty() || text.is_empty(), "empty section {}", i);
        }
        sections
    }

    #[test]
    fn a_text_within_the_limit_is_one_section() {
        assert_eq!(split_into_sections("", 100), vec![""]);
        assert_eq!(split_into_sections("short text", 100), vec!["short text"]);
        let exact = "a".repeat(100);
        assert_eq!(split_into_sections(&exact, 100), vec![exact.as_str()]);
    }

    #[test]
    fn sections_lose_nothing_and_respect_the_limit() {
        let text = numbered_text(400);
        for max in [80, 200, 999, 1_500, 4_096, 20_000] {
            let sections = assert_lossless(&text, max);
            assert!(sections.len() >= text.len().div_ceil(max));
        }
        // Not a multiple of anything, with odd whitespace.
        let ragged = format!("  {}\n\n{}\t{}", numbered_text(7), "word ".repeat(333), numbered_text(3));
        for max in [50, 77, 1_000] {
            assert_lossless(&ragged, max);
        }
    }

    #[test]
    fn sections_end_at_sentence_ends_when_there_are_any() {
        let text = numbered_text(120);
        let sections = assert_lossless(&text, 1_000);
        assert!(sections.len() > 1);
        for section in &sections {
            assert!(section.ends_with(". "), "cut inside a sentence: ...{:?}", &section[section.len() - 20..]);
            assert!(section.starts_with("Fact number"), "{:?}", &section[..20]);
        }
    }

    #[test]
    fn unpunctuated_captions_are_cut_between_words() {
        // Auto-captions: no punctuation at all.
        let text = "alpha bravo charlie delta echo foxtrot ".repeat(300);
        let sections = assert_lossless(&text, 700);
        let words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"];
        for section in &sections {
            assert!(section.ends_with(' '), "cut inside a word");
            assert!(section.split_whitespace().all(|w| words.contains(&w)));
        }
        // A line break counts as a sentence end.
        let lines = "first line of the captions\nsecond line of the captions\n".repeat(40);
        for section in assert_lossless(&lines, 300) {
            assert!(section.ends_with('\n'));
        }
    }

    #[test]
    fn text_without_any_boundary_is_cut_hard_but_never_inside_a_character() {
        let ascii = "x".repeat(2_500);
        assert_eq!(assert_lossless(&ascii, 1_000).len(), 3);

        // 2 bytes per character, 3 bytes per character, 4 bytes per character.
        for (unit, max) in [("é", 101), ("語", 100), ("🎓", 99), ("é", 1), ("🎓", 3)] {
            let text = unit.repeat(500);
            let sections = split_into_sections(&text, max);
            assert_eq!(sections.concat(), text);
            // One character can be wider than an absurdly small limit; nothing else may exceed it.
            let limit = max.max(unit.len());
            assert!(sections.iter().all(|s| !s.is_empty() && s.len() <= limit), "unit {} max {}", unit, max);
        }
        // Mixed scripts with spaces.
        let mixed = "Größe und Maß sind wichtig. 日本語のテキストです。 Ünïcödé everywhere! ".repeat(60);
        assert_lossless(&mixed, 256);
    }

    #[test]
    fn sections_are_about_equally_long() {
        // 2.1 limits of text: three sections of about 0.7, not 1 + 1 + 0.1.
        let text = "word ".repeat(2_100);
        let sections = assert_lossless(&text, 5_000);
        assert_eq!(sections.len(), 3);
        for section in &sections {
            assert!(section.len() > 2_500, "{} bytes", section.len());
        }
        // An early full stop does not produce a tiny first section.
        let early = format!("Hi. {}", "word ".repeat(2_000));
        let sections = assert_lossless(&early, 5_000);
        assert!(sections[0].len() > 2_500, "{} bytes", sections[0].len());
    }

    // ---- notes ------------------------------------------------------------

    #[test]
    fn merge_groups_cover_all_notes_in_order_and_always_shrink_the_list() {
        // Pairs when three do not fit.
        assert_eq!(merge_groups(&[180; 8], 500), vec![0..2, 2..4, 4..6, 6..8]);
        // As many as fit.
        assert_eq!(merge_groups(&[100; 7], 350), vec![0..3, 3..6, 6..7]);
        // A leftover single note is carried over, not merged with nothing.
        assert_eq!(merge_groups(&[180; 5], 500), vec![0..2, 2..4, 4..5]);
        // Two notes that do not fit together are still paired: a round must shrink the list.
        assert_eq!(merge_groups(&[400, 400, 400], 500), vec![0..2, 2..3]);
        assert_eq!(merge_groups(&[10, 10], 500), vec![0..2]);
        assert_eq!(merge_groups(&[10], 500), vec![0..1]);
        assert!(merge_groups(&[], 500).is_empty());

        for (lens, max) in [(vec![120usize; 23], 500usize), (vec![50, 900, 50, 50, 300, 10], 400)] {
            let groups = merge_groups(&lens, max);
            // Consecutive, complete, no overlap.
            let mut next = 0;
            for g in &groups {
                assert_eq!(g.start, next);
                assert!(g.end > g.start);
                next = g.end;
            }
            assert_eq!(next, lens.len());
            assert!(groups.len() < lens.len());
        }
    }

    #[test]
    fn notes_are_rendered_in_order_under_their_section_numbers() {
        let notes = vec![
            SectionNote { first: 1, last: 1, text: " - a fact\n".to_string() },
            SectionNote { first: 2, last: 4, text: "- merged".to_string() },
        ];
        assert_eq!(
            render_notes(&notes, 4),
            "### Notes on section 1 of 4\n- a fact\n\n### Notes on sections 2-4 of 4\n- merged"
        );
    }

    #[test]
    fn empty_notes_are_refused_and_think_blocks_removed() {
        assert_eq!(usable_notes("  - a fact \n").expect("notes"), "- a fact");
        assert_eq!(usable_notes("<think>hmm</think>\n- a fact").expect("notes"), "- a fact");
        assert!(usable_notes("   ").is_err());
        assert!(usable_notes("<think>only thinking</think>  ").is_err());
        assert_eq!(strip_think("no block"), "no block");
        // A reasoning block that never ended is not notes.
        assert!(usable_notes("<think>Okay, the user wants me to summarise this section. Let me").is_err());
    }

    #[test]
    fn the_answer_is_the_content_and_reasoning_only_stands_in_for_a_finished_answer() {
        let message = |content: Option<&str>, reasoning: Option<&str>| ChatCompletionResponseMessage {
            content: content.map(str::to_string),
            reasoning_content: reasoning.map(str::to_string),
        };
        // Content wins whenever there is any, whatever the finish reason.
        for finish in [Some("stop"), Some("length"), None] {
            let text = answer_text(&message(Some("- a fact"), Some("thinking...")), finish).expect("content");
            assert_eq!(text, "- a fact");
        }
        // Empty content, answer finished: the server put the answer into reasoning_content.
        for finish in [Some("stop"), None] {
            let text = answer_text(&message(Some("  "), Some("- a fact")), finish).expect("finished answer");
            assert_eq!(text, "- a fact");
            assert_eq!(answer_text(&message(None, Some("- a fact")), finish).expect("finished answer"), "- a fact");
        }
        // Empty content, cut off: the thoughts are NOT the answer.
        assert_eq!(answer_text(&message(Some(""), Some("Okay, the user wants")), Some("length")).expect("empty"), "");
        assert_eq!(answer_text(&message(None, None), Some("length")).expect("empty"), "");
        // Any other abnormal end with empty content, or nothing at all: an error.
        assert!(answer_text(&message(None, Some("thoughts")), Some("content_filter")).is_err());
        assert!(answer_text(&message(None, None), Some("stop")).is_err());
        assert!(answer_text(&message(Some(" "), Some(" ")), None).is_err());
    }

    // ---- request timeout ----------------------------------------------------

    #[test]
    fn the_request_timeout_comes_from_the_env_value_or_scales_with_the_request() {
        use std::time::Duration;
        assert_eq!(timeout_from_raw(Some("900")), Some(Duration::from_secs(900)));
        assert_eq!(timeout_from_raw(Some(" 45 ")), Some(Duration::from_secs(45)));
        assert_eq!(timeout_from_raw(Some("999999999")), Some(Duration::from_secs(MAX_LLM_TIMEOUT_SECS)));
        // Unset or unusable: the scaled default applies.
        for raw in [None, Some(""), Some("0"), Some("-5"), Some("ten"), Some("1.5")] {
            assert_eq!(timeout_from_raw(raw), None, "{:?}", raw);
        }

        // Never below 600 s; the largest requests of the default budget fit the base size.
        assert_eq!(DEFAULT_LLM_TIMEOUT_SECS, 600);
        assert_eq!(default_timeout_secs(0), 600);
        assert_eq!(default_timeout_secs(9_000), 600);
        let budget = ContextBudget::compute(DEFAULT_LLM_CONTEXT_TOKENS, DEFAULT_CHARS_PER_TOKEN, None);
        let largest = budget.section_chars.max(budget.single_chars + JUDGMENT_SYSTEM_PROMPT.len()) + 1_000;
        assert!(largest <= TIMEOUT_BASE_REQUEST_CHARS, "{} characters", largest);
        assert_eq!(default_timeout_secs(largest), 600);
        // A larger context window sends larger requests: more time.
        assert_eq!(default_timeout_secs(TIMEOUT_BASE_REQUEST_CHARS + 1), 1_200);
        assert_eq!(default_timeout_secs(100_000), 3_000);
        assert_eq!(default_timeout_secs(usize::MAX), MAX_LLM_TIMEOUT_SECS);

        let analyzer = VideoAnalyzer::new(None, None, None).with_request_timeout(Duration::from_secs(30));
        // A fixed timeout is used as it is, whatever the request size.
        assert_eq!(analyzer.timeout_for(500_000), Duration::from_secs(30));
        assert_eq!(seconds_text(Duration::from_secs(600)), "600");
        assert_eq!(seconds_text(Duration::from_millis(300)), "0.3");
    }

    #[test]
    fn only_a_timeout_reason_reads_as_a_timeout() {
        let timeout = LlmUnavailable {
            reason: format!("{}600 s: http://x/v1/chat/completions did not finish", TIMEOUT_REASON_PREFIX),
            engine_down: false,
        };
        assert!(timeout.is_timeout());
        let shown = timeout.to_string();
        assert!(shown.starts_with("LLM request timed out after 600 s"), "{}", shown);
        assert!(!shown.contains("not available"), "{}", shown);

        let down = LlmUnavailable { reason: "http://x/v1 - connection refused".to_string(), engine_down: true };
        assert!(!down.is_timeout());
        // Scripts and the CLI recognise an outage by this wording.
        assert!(down.to_string().starts_with("LLM engine not available ("), "{}", down);
    }

    // ---- judgment parsing ---------------------------------------------------

    #[test]
    fn topic_verdicts_are_normalised_and_an_unreadable_one_is_never_ok() {
        // Clearly "ok", in any case and with any punctuation. No verdict at
        // all is "ok" too.
        for ok in ["ok", "OK", " Ok. ", "okay", "\"ok\"", "ok (history lecture)", "OK - a cooking video", "", "   "] {
            assert_eq!(parse_topic_verdict(Some(ok)), Some("ok"), "{:?}", ok);
        }
        assert_eq!(parse_topic_verdict(None), Some("ok"));

        for (raw, verdict) in [
            ("fringe", "fringe"),
            ("Fringe / pseudoscience", "fringe"),
            ("fringe/pseudoscience", "fringe"),
            ("pseudo-science", "fringe"),
            ("PARANORMAL", "fringe"),
            ("conspiracy theory", "fringe"),
            ("nonsense", "nonsense"),
            ("Nonsense (no informational intent)", "nonsense"),
            ("politics", "politics"),
            ("Politics.", "politics"),
            ("  POLITICS \n", "politics"),
            ("political", "politics"),
            ("Political commentary", "politics"),
            ("geopolitics", "politics"),
            ("politics (election commentary), not history", "politics"),
            // A list the model sent instead of one string.
            (r#"["politics","fringe"]"#, "politics"),
        ] {
            assert_eq!(parse_topic_verdict(Some(raw)), Some(verdict), "{:?}", raw);
            assert!(crate::models::TOPIC_VERDICTS.contains(&verdict));
        }

        // Not one of the four verdicts and not clearly "ok": unusable, never "ok".
        for unusable in [
            "not ok",
            "off-topic",
            "rejected",
            "sports",
            "on-topic",
            "history",
            "acceptable",
            "ok / politics",
            "ok (not politics)",
            "not politics",
            "non-political",
            "apolitical",
            "true",
            "3",
            "...",
            "{}",
        ] {
            assert_eq!(parse_topic_verdict(Some(unusable)), None, "{:?}", unusable);
        }
    }

    #[test]
    fn verdict_and_score_fields_are_read_leniently_but_not_invented() {
        let parse = |json: &str| serde_json::from_str::<LlmInsightJson>(json).expect("parses");

        // One string wrapped in a list is that string.
        let wrapped = parse(r#"{"topic_verdict":["politics"],"summary":["One sentence."],"slop_score":10}"#);
        assert_eq!(wrapped.topic_verdict.as_deref(), Some("politics"));
        assert_eq!(wrapped.summary, "One sentence.");
        // Missing and null are "no verdict"; anything else is kept for the parser to refuse.
        assert_eq!(parse(r#"{"slop_score":10}"#).topic_verdict, None);
        assert_eq!(parse(r#"{"topic_verdict":null}"#).topic_verdict, None);
        let list = parse(r#"{"topic_verdict":["ok","politics"]}"#).topic_verdict;
        assert_eq!(parse_topic_verdict(list.as_deref()), None);
        let object = parse(r#"{"topic_verdict":{"label":"sports"}}"#).topic_verdict;
        assert_eq!(parse_topic_verdict(object.as_deref()), None);
        assert_eq!(parse_topic_verdict(parse(r#"{"topic_verdict":false}"#).topic_verdict.as_deref()), None);

        // Scores: rounded and clamped to 0..=100.
        for (raw, score) in [
            ("20", 20),
            ("\"20\"", 20),
            ("\" 35/100 \"", 35),
            ("45.4", 45),
            ("45.5", 46),
            ("0.2", 0),
            ("-5", 0),
            ("-0.4", 0),
            ("150", 100),
            ("1e9", 100),
        ] {
            let parsed = parse(&format!(r#"{{"slop_score":{}}}"#, raw));
            assert_eq!(parsed.slop_score, Some(score), "{}", raw);
        }
        // Not a number: no score, so the answer is unusable.
        for raw in ["\"20%\"", "\"low\"", "\"\"", "null", "[20]", "true", "{}"] {
            let parsed = parse(&format!(r#"{{"slop_score":{}}}"#, raw));
            assert_eq!(parsed.slop_score, None, "{}", raw);
        }
        assert_eq!(parse("{}").slop_score, None);
    }

    fn category(name: &str, description: Option<&str>) -> Category {
        Category {
            name: name.to_string(),
            description: description.map(str::to_string),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn answer(topic_verdict: &str, slop: u32, model_keep: bool) -> String {
        answer_with_category(topic_verdict, slop, model_keep, serde_json::Value::Null)
    }

    fn answer_with_category(topic_verdict: &str, slop: u32, model_keep: bool, category: serde_json::Value) -> String {
        serde_json::json!({
            "topic_verdict": topic_verdict, "topic_reason": "Because of what it is about.",
            "category": category, "slop_score": slop, "is_worth_keeping": model_keep,
            "quality_rating": "High Signal", "summary": "A summary.", "key_insights": ["One"],
            "takeaways": [], "quotes": [], "topics": ["T"], "mentioned_creators": []
        })
        .to_string()
    }

    #[test]
    fn the_keep_flag_is_computed_from_verdict_and_score_not_taken_from_the_model() {
        let analyzer = VideoAnalyzer::new(None, None, Some("m".to_string())); // threshold 45
        let judge = |text: String| analyzer.judgment_from_answer("vid", &text, Some("stop")).expect("usable answer");

        // (verdict, score, what the model claims) -> kept
        let table = [
            ("ok", 10, true, true),
            ("ok", 45, true, true),
            // The model's own flag is ignored in both directions.
            ("ok", 10, false, true),
            ("ok", 46, true, false),
            ("ok", 90, true, false),
            // A rejected subject is never kept, however dense the video.
            ("fringe", 5, true, false),
            ("nonsense", 5, true, false),
            ("politics", 5, true, false),
            // Variants of a rejecting verdict reject as well.
            ("Politics.", 5, true, false),
            ("fringe/pseudoscience", 5, true, false),
        ];
        for (verdict, score, model_keep, kept) in table {
            let (insight, parsed_verdict, reason) = judge(answer(verdict, score, model_keep));
            assert_eq!(insight.is_worth_keeping, kept, "{} at {}", verdict, score);
            // The score is the model's quality score, whatever the subject.
            assert_eq!(insight.slop_score, score);
            assert_eq!(parsed_verdict == "ok", verdict == "ok");
            assert_eq!(reason.as_deref(), Some("Because of what it is about."));
        }

        // A verdict that cannot be read is an unusable answer: it is neither
        // stored as "ok" (kept) nor guessed to be a rejection.
        for unreadable in ["sports", "not ok", "off-topic", "ok / politics"] {
            let err = analyzer
                .judgment_from_answer("vid", &answer(unreadable, 20, true), Some("stop"))
                .expect_err("unreadable verdict");
            let text = err.to_string();
            assert!(text.contains("unrecognised topic_verdict"), "{}", text);
            assert!(text.contains(unreadable), "{}", text);
        }

        // The threshold in use decides.
        let lenient = VideoAnalyzer::new(None, None, None).with_max_slop(60);
        let (insight, _, _) =
            lenient.judgment_from_answer("vid", &answer("ok", 55, false), None).expect("usable");
        assert!(insight.is_worth_keeping);
    }

    #[test]
    fn only_a_category_the_user_defined_is_stored_in_the_users_spelling() {
        let analyzer = VideoAnalyzer::new(None, None, Some("m".to_string()))
            .with_categories(vec![category("Ancient History", Some("antiquity")), category("Baking", None)]);
        let stored = |value: serde_json::Value| {
            let (insight, verdict, _) = analyzer
                .judgment_from_answer("vid", &answer_with_category("ok", 12, true, value), None)
                .expect("usable");
            assert_eq!(verdict, "ok");
            assert!(insight.is_worth_keeping);
            assert_eq!(insight.analyzer_type, "llm:m");
            insight.category
        };
        assert_eq!(stored(serde_json::json!("Baking")).as_deref(), Some("Baking"));
        assert_eq!(stored(serde_json::json!("  ancient history ")).as_deref(), Some("Ancient History"));
        assert_eq!(stored(serde_json::json!(["BAKING"])).as_deref(), Some("Baking"));
        // Anything else is no category: an invented name, null, a list, a number.
        for other in [
            serde_json::json!("Gardening"),
            serde_json::json!("Baking, Ancient History"),
            serde_json::Value::Null,
            serde_json::json!("null"),
            serde_json::json!(""),
            serde_json::json!(["Baking", "Ancient History"]),
            serde_json::json!(3),
        ] {
            assert_eq!(stored(other.clone()), None, "{}", other);
        }
    }

    #[test]
    fn the_category_never_changes_score_verdict_or_the_keep_decision() {
        let plain = VideoAnalyzer::new(None, None, Some("m".to_string()));
        let with_categories = plain.clone().with_categories(vec![category("Baking", None)]);
        for (verdict, score) in [("ok", 10), ("ok", 46), ("fringe", 5), ("politics", 30)] {
            let text = answer_with_category(verdict, score, true, serde_json::json!("Baking"));
            let (a, verdict_a, _) = plain.judgment_from_answer("vid", &text, None).expect("usable");
            let (b, verdict_b, _) = with_categories.judgment_from_answer("vid", &text, None).expect("usable");
            assert_eq!((a.slop_score, a.is_worth_keeping, verdict_a), (b.slop_score, b.is_worth_keeping, verdict_b));
            // Without categories of the user's own nothing is stored.
            assert_eq!(a.category, None);
            assert_eq!(b.category.as_deref(), Some("Baking"));
        }
    }

    #[test]
    fn without_categories_the_prompt_does_not_mention_them() {
        let prompt = judgment_system_prompt(45, &[]);
        assert!(!prompt.to_lowercase().contains("categor"), "{}", prompt);
        assert!(!prompt.contains("{CATEGORY"), "{}", prompt);
        assert!(!prompt.contains("{MAX_SLOP}"));
        assert!(prompt.contains("slop_score <= 45"));
        assert_eq!(category_prompt_parts(&[]), (String::new(), String::new()));
    }

    #[test]
    fn with_categories_the_prompt_lists_each_name_and_description_and_asks_for_one() {
        let categories = [category("Ancient History", Some("Greece, Rome and their neighbours")), category("Baking", None)];
        let prompt = judgment_system_prompt(60, &categories);
        assert!(prompt.contains("- \"Ancient History\": Greece, Rome and their neighbours\n"), "{}", prompt);
        assert!(prompt.contains("- \"Baking\"\n"), "{}", prompt);
        assert!(prompt.contains("\"category\": <one of the category names listed above, or null>"), "{}", prompt);
        assert!(prompt.contains("It never affects the score or the verdict"));
        assert!(!prompt.contains("{CATEGORY"));
        assert!(prompt.contains("slop_score <= 60"));
        // The list is the only difference to the prompt without categories.
        let (section, key) = category_prompt_parts(&categories);
        assert_eq!(prompt.len(), judgment_system_prompt(60, &[]).len() + section.len() + key.len());
    }

    #[test]
    fn the_category_list_takes_its_room_from_the_judgment_request() {
        let budget = ContextBudget { single_chars: 5_000, section_chars: 4_000 };
        let plain = VideoAnalyzer::new(None, None, None).with_context_budget(budget);
        assert_eq!(plain.judgment_chars(), 5_000);
        let categories = vec![category("Baking", Some("bread and pastry"))];
        let (section, key) = category_prompt_parts(&categories);
        let listed = plain.clone().with_categories(categories);
        assert_eq!(listed.judgment_chars(), 5_000 - section.len() - key.len());
        let tiny = listed.with_context_budget(ContextBudget { single_chars: 10, section_chars: 10 });
        assert_eq!(tiny.judgment_chars(), 0);
    }

    #[test]
    fn a_loosely_formed_answer_without_a_verdict_parses_with_topic_ok() {
        let analyzer = VideoAnalyzer::new(None, None, None);
        let loose = r#"Here you go: {"slop_score":"20","is_worth_keeping":"yes",
            "summary":"S.","key_insights":"single string"}"#;
        let (insight, verdict, reason) = analyzer.judgment_from_answer("vid", loose, None).expect("tolerant");
        assert_eq!((verdict, reason), ("ok", None));
        assert_eq!(insight.category, None);
        assert_eq!(insight.slop_score, 20);
        assert!(insight.is_worth_keeping);
        assert_eq!(insight.key_insights, vec!["single string".to_string()]);

        // Still refused: no score, no summary, cut off.
        assert!(analyzer.judgment_from_answer("vid", "{}", None).is_err());
        assert!(analyzer.judgment_from_answer("vid", r#"{"slop_score":20}"#, None).is_err());
        assert!(analyzer
            .judgment_from_answer("vid", &answer("ok", 20, true), Some("length"))
            .is_err());
    }

    #[test]
    fn the_prompt_separates_topic_from_quality() {
        let prompt = JUDGMENT_SYSTEM_PROMPT;
        assert!(prompt.contains("The subject NEVER raises or lowers the slop score"));
        for verdict in crate::models::TOPIC_VERDICTS {
            assert!(prompt.contains(&format!("\"{}\"", verdict)), "{}", verdict);
        }
        for band in ["0-20", "21-45", "46-70", "71-100"] {
            assert!(prompt.contains(band), "{}", band);
        }
        assert!(prompt.contains("{MAX_SLOP}"));
        // The notes prompts ask for no score.
        for notes_prompt in [NOTES_SYSTEM_PROMPT, MERGE_SYSTEM_PROMPT] {
            assert!(notes_prompt.contains("Do not score"));
            assert!(!notes_prompt.contains("slop_score"));
        }
    }

    // ---- reading with a stub LLM ---------------------------------------------

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Notes,
        Merge,
        Judgment,
        Category,
    }

    /// A chat-completions endpoint on a loopback port that records every
    /// request (kind and user message) and the system prompt of every
    /// judgment request.
    struct Stub {
        url: String,
        log: std::sync::Arc<std::sync::Mutex<Vec<(Kind, String)>>>,
        judgment_prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Stub {
        fn users(&self, kind: Kind) -> Vec<String> {
            let log = self.log.lock().expect("log");
            log.iter().filter(|(k, _)| *k == kind).map(|(_, user)| user.clone()).collect()
        }

        fn count(&self, kind: Kind) -> usize {
            self.users(kind).len()
        }
    }

    /// A successful answer with `content` as the assistant message.
    fn says(content: &str) -> (&'static str, String) {
        (
            "200 OK",
            serde_json::json!({ "choices": [{ "message": { "content": content }, "finish_reason": "stop" }] }).to_string(),
        )
    }

    /// `reply(kind, n)` answers the n-th request (1-based) of that kind with
    /// (status line, response body).
    fn stub_llm(reply: impl Fn(Kind, usize) -> (&'static str, String) + Send + 'static) -> Stub {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let thread_log = log.clone();
        let judgment_prompts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let thread_prompts = judgment_prompts.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                // Read the headers and then as many body bytes as they announce.
                let mut request = Vec::new();
                let mut chunk = [0u8; 8192];
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
                        if request.len() >= body_start + body_len {
                            break;
                        }
                    }
                }
                let body: serde_json::Value = serde_json::from_slice(&request[body_start..]).unwrap_or_default();
                let message = |i: usize| body["messages"][i]["content"].as_str().unwrap_or_default().to_string();
                let system = message(0);
                let kind = if system == NOTES_SYSTEM_PROMPT {
                    Kind::Notes
                } else if system == MERGE_SYSTEM_PROMPT {
                    Kind::Merge
                } else if system.starts_with("You sort ONE video") {
                    Kind::Category
                } else {
                    Kind::Judgment
                };
                if kind == Kind::Judgment {
                    thread_prompts.lock().expect("prompts").push(system);
                }
                let nth = {
                    let mut log = thread_log.lock().expect("log");
                    log.push((kind, message(1)));
                    log.iter().filter(|(k, _)| *k == kind).count()
                };
                let (status_line, response_body) = reply(kind, nth);
                let response = format!(
                    "HTTP/1.1 {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    status_line,
                    response_body.len(),
                    response_body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        Stub { url: format!("http://{}/v1", addr), log, judgment_prompts }
    }

    const SHORT_NOTES: &str = "- a concrete fact\nTOPIC: measurements\nFILLER: none";

    fn default_reply(kind: Kind, _nth: usize) -> (&'static str, String) {
        match kind {
            Kind::Notes => says(SHORT_NOTES),
            Kind::Merge => says("TOPIC: merged"),
            Kind::Judgment => says(&answer("ok", 15, true)),
            Kind::Category => says(r#"{"category": null}"#),
        }
    }

    /// A stored insight as `categorize` gets it.
    fn stored_insight() -> InsightResult {
        InsightResult {
            video_id: "vid".to_string(),
            slop_score: 18,
            is_worth_keeping: true,
            quality_rating: "High Signal".to_string(),
            summary: "How a rye starter is fed and kept.".to_string(),
            key_insights: vec!["Feed it every twelve hours".to_string(), "Rye ferments faster than wheat".to_string()],
            takeaways: vec!["A takeaway that is not sent".to_string()],
            quotes: vec!["A quote that is not sent".to_string()],
            topics: vec!["Sourdough Starter".to_string(), "Rye Flour".to_string()],
            mentioned_creators: vec![],
            analyzer_type: "llm:earlier-model".to_string(),
            category: None,
        }
    }

    #[test]
    fn the_category_prompt_lists_the_categories_and_asks_for_nothing_else() {
        let categories = [category("Ancient History", Some("Greece, Rome and their neighbours")), category("Baking", None)];
        let prompt = categorize_system_prompt(&categories);
        assert!(prompt.contains("- \"Ancient History\": Greece, Rome and their neighbours\n- \"Baking\"\nReturn"), "{}", prompt);
        assert!(prompt.contains("\"category\": <one of the category names listed above, or null>"), "{}", prompt);
        assert!(!prompt.contains("{CATEGORY"), "{}", prompt);
        // No score, verdict or summary is asked for.
        for key in ["slop_score", "topic_verdict", "\"summary\"", "is_worth_keeping", "key_insights"] {
            assert!(!prompt.contains(key), "{} in {}", key, prompt);
        }
        // The list is the one the judgment prompt shows.
        assert!(category_prompt_parts(&categories).0.ends_with(&category_list_lines(&categories)));

        let user = categorize_user_prompt(" Feeding a starter ", &stored_insight());
        assert_eq!(
            user,
            "Video Title: Feeding a starter\n\nSummary: How a rye starter is fed and kept.\n\nKey insights:\n- Feed it every twelve hours\n- Rye ferments faster than wheat\n\nTopics: Sourdough Starter, Rye Flour\n\nProvide the JSON answer now."
        );
        let bare = InsightResult { key_insights: vec![], topics: vec![], ..stored_insight() };
        assert_eq!(
            categorize_user_prompt("T", &bare),
            "Video Title: T\n\nSummary: How a rye starter is fed and kept.\n\nProvide the JSON answer now."
        );
    }

    #[test]
    fn a_category_answer_counts_only_when_it_names_one_of_the_users_categories() {
        let analyzer = VideoAnalyzer::new(None, None, Some("m".to_string()))
            .with_categories(vec![category("Ancient History", None), category("Baking", None)]);
        let read = |text: &str| analyzer.category_from_answer(text, Some("stop")).expect("readable answer");
        assert_eq!(read(r#"{"category": "Baking"}"#).as_deref(), Some("Baking"));
        assert_eq!(read(r#"{"category": "  ancient HISTORY "}"#).as_deref(), Some("Ancient History"));
        assert_eq!(read(r#"{"category": ["baking"]}"#).as_deref(), Some("Baking"));
        assert_eq!(read("<think>bread, so baking</think>\n```json\n{\"category\": \"Baking\"}\n```").as_deref(), Some("Baking"));
        // Cut off before the closing brace.
        assert_eq!(read(r#"{"category": "Baking""#).as_deref(), Some("Baking"));
        // Anything else is no category.
        for other in [
            r#"{"category": "Gardening"}"#,
            r#"{"category": "Baking, Ancient History"}"#,
            r#"{"category": null}"#,
            r#"{"category": "null"}"#,
            r#"{"category": ""}"#,
            r#"{"category": ["Baking", "Ancient History"]}"#,
            r#"{"category": 3}"#,
            r#"{}"#,
            r#"{"label": "Baking"}"#,
        ] {
            assert_eq!(read(other), None, "{}", other);
        }
        // No JSON object at all is an error, not "no category".
        for unreadable in ["Baking", "", "I think it is about bread."] {
            assert!(analyzer.category_from_answer(unreadable, Some("stop")).is_err(), "{:?}", unreadable);
        }
        assert!(analyzer.category_from_answer("", Some("length")).is_err());
    }

    #[tokio::test]
    async fn a_category_request_carries_the_stored_insight_and_the_category_list() {
        let reply = |kind: Kind, nth: usize| match kind {
            Kind::Category => says(r#"{"category": "baking"}"#),
            other => default_reply(other, nth),
        };
        let stub = stub_llm(reply);
        let categories = vec![category("Baking", Some("bread and pastry")), category("Star Gazing", None)];
        let analyzer = stub_analyzer(&stub, 10_000, 8_000).with_categories(categories);

        let chosen = analyzer.categorize("Feeding a starter", &stored_insight()).await.expect("answer");
        assert_eq!(chosen.as_deref(), Some("Baking"));
        // One request, of the category kind: no notes, no judgment.
        assert_eq!((stub.count(Kind::Category), stub.count(Kind::Judgment), stub.count(Kind::Notes)), (1, 0, 0));
        let user = &stub.users(Kind::Category)[0];
        assert_eq!(user, &categorize_user_prompt("Feeding a starter", &stored_insight()));
        assert!(!user.contains("not sent"), "{}", user);

        // Without categories or without an LLM nothing is asked.
        let plain = stub_analyzer(&stub, 10_000, 8_000);
        assert!(plain.categorize("T", &stored_insight()).await.is_err());
        let no_llm = VideoAnalyzer::new(None, None, None).with_categories(vec![category("Baking", None)]);
        assert!(no_llm.categorize("T", &stored_insight()).await.is_err());
        assert_eq!(stub.count(Kind::Category), 1);
    }

    fn stub_analyzer(stub: &Stub, single_chars: usize, section_chars: usize) -> VideoAnalyzer {
        VideoAnalyzer::new(Some(stub.url.clone()), None, Some("stub-model".to_string()))
            .with_context_budget(ContextBudget { single_chars, section_chars })
    }

    #[tokio::test]
    async fn a_transcript_that_fits_is_sent_whole_in_one_request() {
        let stub = stub_llm(default_reply);
        let analyzer = stub_analyzer(&stub, 10_000, 8_000);
        let text = numbered_text(100);
        assert!(text.len() <= 10_000);

        let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("analysis");
        assert_eq!(stub.count(Kind::Notes), 0);
        assert_eq!(stub.count(Kind::Judgment), 1);
        // The request carries the transcript from the first to the last sentence.
        let user = &stub.users(Kind::Judgment)[0];
        assert!(user.contains(text.trim()), "whole transcript in the request");
        assert!(user.contains("Video Title: Measurements"));
        assert_eq!(analysis.coverage, AnalysisCoverage::whole(&text, 1));
        assert_eq!(analysis.topic_verdict.as_deref(), Some("ok"));
        assert_eq!(analysis.insight.slop_score, 15);
        assert!(analysis.insight.is_worth_keeping);
    }

    #[tokio::test]
    async fn the_judgment_request_names_the_users_categories_only_when_there_are_some() {
        let reply = |kind: Kind, nth: usize| match kind {
            Kind::Judgment => says(&answer_with_category("ok", 15, true, serde_json::json!("baking"))),
            other => default_reply(other, nth),
        };
        let stub = stub_llm(reply);
        let text = numbered_text(50);

        // No categories: the prompt is silent about them and nothing is stored,
        // whatever the model volunteers.
        let plain = stub_analyzer(&stub, 10_000, 8_000);
        let analysis = plain.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("analysis");
        assert_eq!(analysis.insight.category, None);

        let categories = vec![category("Baking", Some("bread and pastry")), category("Star Gazing", None)];
        let listed = stub_analyzer(&stub, 10_000, 8_000).with_categories(categories);
        let analysis = listed.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("analysis");
        assert_eq!(analysis.insight.category.as_deref(), Some("Baking"));
        assert_eq!((analysis.insight.slop_score, analysis.insight.is_worth_keeping), (15, true));

        let prompts = stub.judgment_prompts.lock().expect("prompts");
        assert_eq!(prompts.len(), 2);
        assert!(!prompts[0].to_lowercase().contains("categor"), "{}", prompts[0]);
        assert!(prompts[1].contains("- \"Baking\": bread and pastry\n- \"Star Gazing\"\n"), "{}", prompts[1]);
        assert!(prompts[1].contains("\"category\": <one of the category names listed above, or null>"));
    }

    #[tokio::test]
    async fn a_long_transcript_is_read_section_by_section_and_judged_from_all_notes() {
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Notes => says(&format!("- fact from request {}\nTOPIC: t\nFILLER: none", nth)),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let text = numbered_text(100);
        let sections = split_into_sections(&text, 1_500);
        assert!(sections.len() >= 4, "{} sections", sections.len());

        let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("analysis");

        // One notes request per section, in order, each with exactly its section.
        let notes_requests = stub.users(Kind::Notes);
        assert_eq!(notes_requests.len(), sections.len());
        for (i, (request, section)) in notes_requests.iter().zip(&sections).enumerate() {
            assert!(request.contains(&format!("Section {} of {}", i + 1, sections.len())), "{}", request);
            assert!(request.contains(section.trim()), "section {} is in its request", i + 1);
        }
        // Every sentence of the transcript was sent, and only once.
        for i in 0..100 {
            let sentence = format!("Fact number {} is", i);
            let hits = notes_requests.iter().filter(|r| r.contains(&sentence)).count();
            assert_eq!(hits, 1, "{}", sentence);
        }

        // No condensing needed; one judgment from the title and ALL notes, in order.
        assert_eq!(stub.count(Kind::Merge), 0);
        let judgments = stub.users(Kind::Judgment);
        assert_eq!(judgments.len(), 1);
        let judgment = &judgments[0];
        assert!(judgment.contains("Video Title: Measurements"));
        let mut last_pos = 0;
        for i in 1..=sections.len() {
            let heading = format!("### Notes on section {} of {}", i, sections.len());
            let pos = judgment.find(&heading).unwrap_or_else(|| panic!("missing {}", heading));
            assert!(pos >= last_pos, "notes out of order");
            last_pos = pos;
            assert!(judgment.contains(&format!("- fact from request {}", i)));
        }
        // The transcript itself is not in the judgment request, and it is not called one.
        assert!(!judgment.contains("Fact number 50 is"));
        assert!(judgment.contains("notes on every section"));

        assert_eq!(
            analysis.coverage,
            AnalysisCoverage { transcript_chars: text.len(), sections: sections.len() }
        );
        assert_eq!(analysis.insight.analyzer_type, "llm:stub-model");
    }

    #[tokio::test]
    async fn too_many_notes_are_condensed_in_groups_before_the_judgment() {
        // Notes of 150 characters: eight of them do not fit 600 characters.
        let long_note = format!("- {}", "detail ".repeat(21));
        let stub = stub_llm(move |kind, nth| match kind {
            Kind::Notes => says(&long_note),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 600, 500);
        let text = numbered_text(60);
        let sections = split_into_sections(&text, 500).len();
        assert!(sections >= 6, "{} sections", sections);

        let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("analysis");

        assert_eq!(stub.count(Kind::Notes), sections);
        // Two notes fit a merge request, three do not: one merge per pair, one round.
        assert_eq!(stub.count(Kind::Merge), sections / 2);
        assert_eq!(stub.count(Kind::Judgment), 1);
        let merges = stub.users(Kind::Merge);
        assert!(merges[0].contains("### Notes on section 1 of"));
        assert!(merges[0].contains("### Notes on section 2 of"));
        assert!(!merges[0].contains("### Notes on section 3 of"));
        let judgment = &stub.users(Kind::Judgment)[0];
        assert!(judgment.contains(&format!("### Notes on sections 1-2 of {}", sections)), "{}", judgment);
        assert!(judgment.contains("TOPIC: merged"));
        assert!(judgment.len() < 600 + 600, "the notes were condensed");
        // All sections were read; condensing does not change that.
        assert_eq!(analysis.coverage, AnalysisCoverage::whole(&text, sections));
    }

    #[tokio::test]
    async fn notes_that_cannot_be_condensed_fail_the_analysis() {
        let long_note = format!("- {}", "detail ".repeat(21));
        let stub = stub_llm(move |kind, nth| match kind {
            Kind::Notes => says(&long_note),
            // Longer than what it was given.
            Kind::Merge => says(&"blah ".repeat(200)),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 600, 500);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&numbered_text(60)))
            .await
            .expect_err("cannot fit");
        assert!(format!("{:#}", err).contains("could not be condensed"), "{:#}", err);
        assert_eq!(stub.count(Kind::Judgment), 0);
    }

    #[tokio::test]
    async fn a_section_that_fails_once_is_retried() {
        let stub = stub_llm(|kind, nth| match (kind, nth) {
            // The first attempt at section 2.
            (Kind::Notes, 2) => ("400 Bad Request", r#"{"error":"hiccup"}"#.to_string()),
            _ => default_reply(kind, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let text = numbered_text(100);
        let sections = split_into_sections(&text, 1_500).len();

        let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("retry works");
        assert_eq!(stub.count(Kind::Notes), sections + 1);
        assert_eq!(stub.count(Kind::Judgment), 1);
        assert_eq!(analysis.coverage.sections, sections);
        // An empty answer is retried the same way.
        let stub = stub_llm(|kind, nth| match (kind, nth) {
            (Kind::Notes, 1) => says("   "),
            _ => default_reply(kind, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("retry works");
        assert_eq!(stub.count(Kind::Notes), sections + 1);
    }

    #[tokio::test]
    async fn a_section_that_fails_twice_fails_the_analysis_instead_of_a_partial_judgment() {
        let stub = stub_llm(|kind, nth| match (kind, nth) {
            (Kind::Notes, 2) | (Kind::Notes, 3) => ("400 Bad Request", r#"{"error":"still broken"}"#.to_string()),
            _ => default_reply(kind, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&numbered_text(100)))
            .await
            .expect_err("section 2 cannot be read");
        let text = format!("{:#}", err);
        assert!(text.contains("Section 2 of"), "{}", text);
        assert!(text.contains("not judged on part of its transcript"), "{}", text);
        // The video's own failure, not an engine outage.
        assert!(err.downcast_ref::<LlmUnavailable>().is_none());
        // Reading stopped there and nothing was judged.
        assert_eq!(stub.count(Kind::Notes), 3);
        assert_eq!(stub.count(Kind::Judgment), 0);
    }

    #[tokio::test]
    async fn an_unavailable_engine_ends_the_reading_at_once() {
        let stub = stub_llm(|kind, nth| match (kind, nth) {
            (Kind::Notes, 2) => ("503 Service Unavailable", r#"{"error":"loading model"}"#.to_string()),
            _ => default_reply(kind, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&numbered_text(100)))
            .await
            .expect_err("engine down");
        let unavailable = err.downcast_ref::<LlmUnavailable>().expect("typed, so the video stays unprocessed");
        assert!(unavailable.engine_down);
        // No retry, no further section, no judgment.
        assert_eq!(stub.count(Kind::Notes), 2);
        assert_eq!(stub.count(Kind::Judgment), 0);
    }

    #[tokio::test]
    async fn an_unusable_judgment_is_retried_without_reading_the_sections_again() {
        let stub = stub_llm(|kind, nth| match (kind, nth) {
            (Kind::Judgment, 1) => says("{}"),
            _ => default_reply(kind, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let text = numbered_text(100);
        let sections = split_into_sections(&text, 1_500).len();
        analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("second answer is good");
        assert_eq!(stub.count(Kind::Judgment), 2);
        assert_eq!(stub.count(Kind::Notes), sections);
    }

    /// A successful answer with every part of the assistant message given.
    fn says_with(content: &str, reasoning: Option<&str>, finish_reason: &str) -> (&'static str, String) {
        (
            "200 OK",
            serde_json::json!({
                "choices": [{
                    "message": { "content": content, "reasoning_content": reasoning },
                    "finish_reason": finish_reason
                }]
            })
            .to_string(),
        )
    }

    #[tokio::test]
    async fn an_unreadable_verdict_is_asked_again_once_and_then_fails_the_analysis() {
        // First answer "sports", second answer usable.
        let stub = stub_llm(|kind, nth| match (kind, nth) {
            (Kind::Judgment, 1) => says(&answer("sports", 15, true)),
            _ => default_reply(kind, nth),
        });
        let analyzer = stub_analyzer(&stub, 10_000, 8_000);
        let text = numbered_text(50);
        let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("second answer");
        assert_eq!(stub.count(Kind::Judgment), 2);
        assert_eq!(analysis.topic_verdict.as_deref(), Some("ok"));

        // Unreadable both times: the video fails, nothing is kept as "ok".
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Judgment => says(&answer("not ok", 15, true)),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 10_000, 8_000);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&text))
            .await
            .expect_err("unreadable verdict twice");
        let shown = format!("{:#}", err);
        assert!(shown.contains("unrecognised topic_verdict"), "{}", shown);
        assert!(shown.contains("after 2 attempts"), "{}", shown);
        // The video's own failure, not an engine outage.
        assert!(err.downcast_ref::<LlmUnavailable>().is_none());
        assert_eq!(stub.count(Kind::Judgment), 2);

        // A rejecting verdict in an unusual spelling is understood at once.
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Judgment => says(&answer("Politics.", 15, true)),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 10_000, 8_000);
        let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("usable");
        assert_eq!(analysis.topic_verdict.as_deref(), Some("politics"));
        assert!(!analysis.insight.is_worth_keeping);
        assert_eq!(stub.count(Kind::Judgment), 1);
    }

    #[tokio::test]
    async fn a_non_numeric_score_is_unusable_and_an_odd_number_is_rounded_and_clamped() {
        let with_score = |score: serde_json::Value| {
            let mut json: serde_json::Value = serde_json::from_str(&answer("ok", 15, true)).expect("json");
            json["slop_score"] = score;
            json.to_string()
        };
        let text = numbered_text(50);

        let percent = with_score(serde_json::json!("20%"));
        let stub = stub_llm(move |kind, nth| match kind {
            Kind::Judgment => says(&percent),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 10_000, 8_000);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&text))
            .await
            .expect_err("no usable score");
        assert!(format!("{:#}", err).contains("no usable slop_score"), "{:#}", err);
        assert_eq!(stub.count(Kind::Judgment), 2);

        for (raw, stored) in [(serde_json::json!(-5), 0), (serde_json::json!(44.6), 45), (serde_json::json!(250), 100)] {
            let body = with_score(raw.clone());
            let stub = stub_llm(move |kind, nth| match kind {
                Kind::Judgment => says(&body),
                other => default_reply(other, nth),
            });
            let analyzer = stub_analyzer(&stub, 10_000, 8_000);
            let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("usable");
            assert_eq!(analysis.insight.slop_score, stored, "{}", raw);
        }
    }

    #[tokio::test]
    async fn notes_come_from_the_content_and_never_from_the_thoughts_of_a_cut_off_answer() {
        let text = numbered_text(100);
        let sections = split_into_sections(&text, 1_500).len();

        // Content and reasoning both there: the content is the notes.
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Notes => says_with("- the real notes\nTOPIC: t\nFILLER: none", Some("Okay, the user wants me to"), "stop"),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("analysis");
        let judgment = &stub.users(Kind::Judgment)[0];
        assert!(judgment.contains("- the real notes"), "{}", judgment);
        assert!(!judgment.contains("the user wants me to"), "{}", judgment);

        // A reasoning model that used up its budget while thinking: content
        // empty, finish_reason "length". Its thoughts are not notes; the
        // request is repeated once, and then the analysis fails.
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Notes => says_with("", Some("Okay, the user wants me to write notes. Let me think"), "length"),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&text))
            .await
            .expect_err("thoughts are not notes");
        let shown = format!("{:#}", err);
        assert!(shown.contains("cut off at max_tokens"), "{}", shown);
        assert!(shown.contains("Section 1 of"), "{}", shown);
        assert!(err.downcast_ref::<LlmUnavailable>().is_none());
        assert_eq!(stub.count(Kind::Notes), 2);
        assert_eq!(stub.count(Kind::Judgment), 0);

        // Empty content with a normally finished answer: the server delivered
        // the answer in reasoning_content, and it is used.
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Notes => says_with("", Some("- notes delivered as reasoning\nTOPIC: t"), "stop"),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("analysis");
        assert_eq!(stub.count(Kind::Notes), sections);
        assert!(stub.users(Kind::Judgment)[0].contains("- notes delivered as reasoning"));
    }

    #[tokio::test]
    async fn cut_off_notes_are_asked_again_briefly_once_and_then_fail_the_analysis() {
        let text = numbered_text(100);
        let sections = split_into_sections(&text, 1_500).len();

        // Section 2 is cut off on the first attempt only.
        let stub = stub_llm(|kind, nth| match (kind, nth) {
            (Kind::Notes, 2) => says_with("- a fact that stops in the mid", None, "length"),
            _ => default_reply(kind, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let analysis = analyzer.analyze_with_coverage("Measurements", &transcript(&text)).await.expect("retry works");
        assert_eq!(analysis.coverage.sections, sections);
        let requests = stub.users(Kind::Notes);
        assert_eq!(requests.len(), sections + 1);
        // The repeated request is the same section, with the order to be brief.
        assert!(!requests[1].contains(BRIEF_NOTES_REMINDER));
        assert!(requests[2].contains(BRIEF_NOTES_REMINDER), "{}", requests[2]);
        assert!(requests[2].starts_with(requests[1].as_str()));
        assert!(!requests[3].contains(BRIEF_NOTES_REMINDER));
        // The cut-off text is not among the notes the video is judged from.
        let judgment = &stub.users(Kind::Judgment)[0];
        assert!(!judgment.contains("stops in the mid"), "{}", judgment);

        // Cut off twice: no judgment on notes with a missing end.
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Notes => says_with("- a fact that stops in the mid", None, "length"),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 2_000, 1_500);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&text))
            .await
            .expect_err("cut off twice");
        let shown = format!("{:#}", err);
        assert!(shown.contains("cut off at max_tokens"), "{}", shown);
        assert!(shown.contains("not judged on part of its transcript"), "{}", shown);
        assert_eq!(stub.count(Kind::Notes), 2);
        assert_eq!(stub.count(Kind::Judgment), 0);
    }

    #[tokio::test]
    async fn a_judgment_cut_off_while_thinking_is_refused() {
        let stub = stub_llm(|kind, nth| match kind {
            Kind::Judgment => says_with("", Some(r#"The answer could be {"slop_score": 10, "summary": "x"}"#), "length"),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 10_000, 8_000);
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&numbered_text(50)))
            .await
            .expect_err("no answer text");
        let shown = format!("{:#}", err);
        assert!(shown.contains("cut off at max_tokens before any answer text"), "{}", shown);
        assert_eq!(stub.count(Kind::Judgment), 2);

        // Finished normally with the JSON in reasoning_content: used.
        let body = answer("ok", 15, true);
        let stub = stub_llm(move |kind, nth| match kind {
            Kind::Judgment => says_with("", Some(body.as_str()), "stop"),
            other => default_reply(other, nth),
        });
        let analyzer = stub_analyzer(&stub, 10_000, 8_000);
        let analysis = analyzer
            .analyze_with_coverage("Measurements", &transcript(&numbered_text(50)))
            .await
            .expect("answer in reasoning_content");
        assert_eq!(analysis.insight.slop_score, 15);
    }

    #[tokio::test]
    async fn a_request_that_takes_too_long_is_a_named_timeout_of_a_reachable_engine() {
        // The stub accepts the request and answers after the timeout has passed.
        let stub = stub_llm(|kind, nth| {
            std::thread::sleep(std::time::Duration::from_millis(1_500));
            default_reply(kind, nth)
        });
        let analyzer = stub_analyzer(&stub, 10_000, 8_000).with_request_timeout(std::time::Duration::from_millis(300));
        let started = std::time::Instant::now();
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&numbered_text(50)))
            .await
            .expect_err("timeout");
        assert!(started.elapsed() < std::time::Duration::from_millis(1_400), "the request was not cut short");

        let unavailable = err.downcast_ref::<LlmUnavailable>().expect("typed: the video stays unprocessed");
        assert!(unavailable.is_timeout());
        // Something answered the connection: this request failed, the engine is not down.
        assert!(!unavailable.engine_down);
        let shown = err.to_string();
        assert!(shown.contains("timed out after 0.3 s"), "{}", shown);
        assert!(shown.contains(LLM_TIMEOUT_SECS_ENV), "{}", shown);
        assert!(!shown.contains("not available"), "{}", shown);
        // No retry within the video: a second wait of the same length helps nobody.
        assert_eq!(stub.count(Kind::Judgment), 1);
    }

    #[tokio::test]
    async fn nothing_listening_is_an_engine_that_is_down_not_a_timeout() {
        // A port that was free a moment ago.
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
            listener.local_addr().expect("local addr").port()
        };
        let analyzer = VideoAnalyzer::new(Some(format!("http://127.0.0.1:{}/v1", port)), None, None)
            .with_context_budget(ContextBudget { single_chars: 10_000, section_chars: 8_000 })
            .with_request_timeout(std::time::Duration::from_secs(20));
        let err = analyzer
            .analyze_with_coverage("Measurements", &transcript(&numbered_text(50)))
            .await
            .expect_err("connection refused");
        let unavailable = err.downcast_ref::<LlmUnavailable>().expect("typed");
        assert!(unavailable.engine_down);
        assert!(!unavailable.is_timeout());
        assert!(err.to_string().starts_with("LLM engine not available ("), "{}", err);
    }
}
