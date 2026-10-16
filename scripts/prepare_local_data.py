#!/usr/bin/env python
"""Turn the local Hub snapshots under /mnt/data/LLM/hf into the jsonl the recipe reads.

`doc/datasets.md` names the cleaning each source needs before it can enter a bucket.  Everything a
`parquet:` / `jsonl:` spec can read straight from the snapshot is left where it is; this script only
writes the sources whose cleaning happens at the text level and which therefore cannot be expressed as
a field expression:

    b-corpus       `.txt.gz` works, decompressed, engine markup/watermarks dropped, split into documents
    moegirl        raw wikitext: templates, tables, comments, scripts and markup stripped
    scraped novels ChineseNovels (whole book -> chapters) and Chinese_interactive_novels_3k (`img` rows)
    JSON arrays    LooksJuicy / catgirl answer sets, plus the catgirl chain-of-thought removal
    English sets   DailyDialog / EmpatheticDialogues / Topical-Chat / Cornell / ESConv noise

`run_pipeline.sh` reads the result through its `data/<bucket>/*.jsonl` globs, and never dereferences an
`hf:` spec, so a run needs no network access.  The snapshots themselves are described in
`/mnt/data/LLM/README.md`; `--snapshot` points at that directory.

    python scripts/prepare_local_data.py                     # every job into data/
    python scripts/prepare_local_data.py --only bcorpus_zh moegirl
    python scripts/prepare_local_data.py --list
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys

DEFAULT_SNAPSHOT = "/mnt/data/LLM/hf"
BCORPUS = "Limour--b-corpus"
# Documents are cut at ~this many characters, at line boundaries.  A whole visual-novel script or novel
# is one file; without the cut each would be a single document and `prepare_data --max_doc_chars` would
# silently keep only its head.
CHUNK_CHARS = 4000

JOBS = {}


def job(name, out, sources=()):
    def register(fn):
        JOBS[name] = dict(out=out, fn=fn, sources=list(sources))
        return fn
    return register


# ------------------------------------------------------------------------------------------------
# output
# ------------------------------------------------------------------------------------------------
class Sink:
    """Write `{"text": ...}` jsonl (or raw records) atomically, one file per bucket source."""

    def __init__(self, path, records=False):
        self.path = path
        self.records = records
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tmp = path + ".part"
        self.f = open(self.tmp, "w", encoding="utf-8")
        self.docs = self.chars = 0

    def write(self, text):
        text = text.strip()
        if not text:
            return
        self.f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.docs += 1
        self.chars += len(text)

    def write_record(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.docs += 1
        if "messages" in rec:
            self.chars += sum(len(m.get("content") or "") for m in rec["messages"])
        else:
            self.chars += len(rec.get("text") or "")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.f.close()
        if exc_type is None:
            os.replace(self.tmp, self.path)
        else:
            os.remove(self.tmp)


# ------------------------------------------------------------------------------------------------
# text helpers
# ------------------------------------------------------------------------------------------------
_QUOTE_START = "「『“‘\"（(【["


_SPEAKER = re.compile(r"^([^\s：:]{1,10})[：:]\s*(.*)$", re.S)
_SPEAKER_NAMES = {"user", "gpt", "system", "assistant", "narrator", "旁白", "旁白君", "系统", "画外音"}


def strip_speaker(line):
    """Drop a `角色名：` / `user：` prefix, keeping prose that merely contains a colon.

    b-corpus and ChatHaruhi tag every turn (`秦绢：「…」`, `gpt：…`), which would otherwise be trained as
    text.  A prefix is only removed when the rest starts with a quote/parenthesis or when the name is a
    known marker, so a narration line such as `相信我：这就是…` keeps its clause.  `Narrator：` has no
    quote after the colon, so the name has to be in the marker set to be recognised."""
    m = _SPEAKER.match(line)
    if not m:
        return line
    name, rest = m.group(1), m.group(2)
    if rest[:1] in _QUOTE_START or name.lower() in _SPEAKER_NAMES:
        return rest
    return line


# Platform directives in the kajiwoto chat log: `#mem:name=felix I'll remember that.` is the model's
# memory write, not dialogue.  The rest of the line is kept.
_MEM = re.compile(r"#mem:\S*")


def clean_line(line):
    """One chat-log line: `#mem:` directives, wikitext/placeholders, speaker tag."""
    line = strip_wikitext(_MEM.sub("", line.replace("\ufeff", "").strip()))
    return strip_speaker(line).rstrip("\\").strip()


def chunk_texts(texts, target=CHUNK_CHARS):
    """Group lines into documents of about `target` characters, breaking between lines."""
    buf, size = [], 0
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if size and size + len(t) + 1 > target:
            yield "\n".join(buf)
            buf, size = [], 0
        buf.append(t)
        size += len(t) + 1
    if buf:
        yield "\n".join(buf)


_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def han_ratio(text):
    return len(_HAN.findall(text)) / max(len(text), 1)


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


# ---- engine markup and site furniture ------------------------------------------------------------
# Visual-novel scripts keep their engine's inline markup: `<USER>` / `<usr>` is the player character's
# name (25.6k occurrences across v-corpus, mostly as a line-initial speaker tag), `<i>` /
# `<color=#FF0000>` / `<br>` is presentation, and Umineko spells incantations as runs of one-letter tags
# (`<e><k>:<e><k>:…`).  Only a known vocabulary is removed, so a tag that swallowed its own text
# (`<F16「太好了…好、很好」>`) survives instead of losing the line with it.
_PLAYER_SPEAKER = re.compile(r"^\s*<\s*(?:USER|usr|player|玩家)\s*>", re.I | re.M)
_PLAYER_WORD = re.compile(r"<\s*(?:USER|usr|player|玩家)\s*>", re.I)
_TAG = re.compile(r"</?\s*([A-Za-z][A-Za-z0-9]{0,9})\b[^<>]{0,80}?/?>")
_MARKUP_TAGS = {
    "user", "usr", "i", "b", "u", "s", "em", "strong", "strike", "del", "ins", "mark", "sub", "sup",
    "font", "color", "size", "span", "div", "p", "br", "hr", "center", "big", "small", "tt", "abbr",
    "cite", "q", "samp", "dfn", "bdi", "bdo", "ruby", "rt", "rb", "rp", "pre", "code", "nobr", "wbr",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "table", "tr", "td", "th", "tbody", "thead",
    "img", "a", "nowiki", "details", "summary", "figure", "figcaption", "blockquote", "section",
    "article", "marquee", "kbd", "var", "output", "time", "picture", "source", "audio", "video",
    "canvas", "iframe", "embed", "object", "param", "map", "area", "svg", "math",
}
_BREAK_TAGS = {"br", "p", "div", "tr", "li"}
# A run of one/two-letter tags is engine notation, never prose; `[^\s:：·・、-]{0,3}` keeps the colons
# and middle dots that separate the runes inside the run.
_RUNE_RUN = re.compile(r"(?:<\s*[A-Za-z]{1,2}\s*>[\s:：·・、\-]{0,3}){3,}")
_ENTITY = {"&nbsp;": " ", "&quot;": '"', "&apos;": "'", "&lt;": "<", "&gt;": ">", "&amp;": "&"}
_ENTITY_RE = re.compile(r"&(?:nbsp|quot|apos|lt|gt|amp|#\d{1,5});", re.I)
# NUL and the other C0 controls the scrapers left behind, plus U+FFFD from a broken decode.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")


def _entity(match):
    token = match.group(0)
    if token.lower().startswith("&#"):
        return chr(int(token[2:-1]))
    return _ENTITY[token.lower()]


def strip_markup(text, player="你"):
    """Drop engine/HTML markup from one line, keeping its text and turning `<br>`s into newlines.

    `player` replaces the inline player-name placeholder; a placeholder that opens a line is rewritten to
    `user` instead, so `strip_speaker` still recognises it as a speaker tag."""
    text = _PLAYER_SPEAKER.sub("user", text)
    text = _PLAYER_WORD.sub(player, text)

    def _sub(m):
        name = m.group(1).lower()
        # a one/two-letter tag is never prose: Umineko writes its runes as `<e>`/`<r>`/`<ng>`, and the
        # engines leave stray `<T 0>` markers behind.  Every real two-letter tag is in the set above.
        if len(name) > 2 and name not in _MARKUP_TAGS:
            return m.group(0)
        return "\n" if name in _BREAK_TAGS else ""

    text = _RUNE_RUN.sub("", text)
    text = _TAG.sub(_sub, text)
    for _ in range(2):  # scraped HTML is often escaped twice (`&amp;#9;`)
        text = _ENTITY_RE.sub(_entity, text)
    return text


# Scraped novels carry the site's own furniture between chapters: group/QQ numbers, off-site links and
# "read the latest chapter here" pointers, plus the work header the scraper kept (`投票 收藏到书柜`,
# `原创 / 男男 / …`).  The short-line guard keeps a prose sentence that merely mentions 投票 or 连载中.
_AD_LINE = re.compile(
    r"https?://|www\.[a-z0-9-]+\.|(?:qq|微信|weixin)\s*群|群\s*[:：]?\s*\d{5,}|"
    r"最新章节|请记住本站|手机版|全文阅读|加入书签|更多小说|资源尽在|整理于|【作家想说的话", re.I)
# Fan-work notices the platform asks the author to paste at the top of a chapter.  These phrases do not
# occur in narrative prose, so unlike the furniture above they need no length guard.  `勿上升` covers the
# variants the scrapers leave behind (`勿上升真人` / `勿上升正主` / `请勿上升` / `勿上升`).
_NOTICE = re.compile(r"勿上升|禁止抄袭|禁止转载|未经允许|侵权必究|纯属虚构|如有雷同|"
                     r"求收藏|求推荐票|求月票|排雷|避雷|可删|勿喷")
_FURNITURE = re.compile(r"投票|收藏到书柜|作品编号|连载中|^\s*原创\s*/")


def is_noise_line(line):
    """True for a blank line, an ad/watermark line, a notice, or a short line of site furniture."""
    s = line.strip()
    if not s:
        return True
    if _AD_LINE.search(s) or _NOTICE.search(s):
        return True
    return bool(len(s) <= 40 and _FURNITURE.search(s))


def clean_bcorpus_line(line, player="你"):
    """One `角色名：`-tagged visual-novel line: BOM, control characters, engine markup, speaker tag."""
    line = _CTRL.sub("", line.replace("\ufeff", "").strip())
    return strip_speaker(strip_markup(line, player)).rstrip("\\").strip()


# ---- wikitext ------------------------------------------------------------------------------------
_COMMENT = re.compile(r"<!--.*?-->", re.S)
# MediaWiki keeps `<script>`/`<style>` bodies verbatim; stripping only the tag leaves their JavaScript in
# the article text (measured 9 pages of the moegirl dump).
_SCRIPT = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>|<references\s*/>", re.S | re.I)
# `[[File:…]]` / `[[Category:…]]` are media and taxonomy, not prose; the doc keeps the body text only.
# A closing bracket may be missing (the scrapers truncate some of these), hence `\]\]?`.
_MEDIA = re.compile(r"\[\[(?:File|Image|文件|图像|图片|Category|分类)\s*:[^\[\]]*\]\]?", re.S | re.I)
# Any remaining `[[…]]`; the visible text is the part after the last `|`, so a broken `[[幼儿体型||…]]`
# or a section link `[[红色警戒#…||红色警戒3]]` keeps its label instead of surviving as markup.
_LINK = re.compile(r"\[\[[^\[\]]*\]\]?")
# Bracket pairs an earlier rule already half-consumed (`{{…[[…}}` leaves `]]`).
_BRACKETS = re.compile(r"\[\[|\]\]")
_EXT_LINK = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+$", re.M)
_MAGIC = re.compile(r"__[A-Z]+__")
_BREAK = re.compile(r"<br\s*/?>", re.I)
_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
_EMPHASIS = re.compile(r"'{2,5}")
_TABLE_ATTR = re.compile(r"^\s*[|!].*?\|", re.M)
_BLANK = re.compile(r"\n{3,}")
# A page the scraper stored as binary (an old GBK/Shift-JIS decode gone wrong) is mojibake, not text:
# replacement characters, private-use code points and C0/C1 controls.  5 pages of the moegirl dump are
# mostly this, and `strip_wikitext` cannot make prose of them, so the moegirl job drops such pages.
_JUNK = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd\ue000-\uf8ff]")


def junk_ratio(text):
    return len(_JUNK.findall(text)) / max(len(text), 1)


def _link_label(match):
    inner = match.group(0)[2:].rstrip("]")
    return inner.rsplit("|", 1)[-1] if "|" in inner else inner


def _strip_balanced(text, open_s, close_s):
    """Remove `open_s … close_s` spans, counting nesting (templates and tables both nest)."""
    out, depth, i, n = [], 0, 0, len(text)
    while i < n:
        if text.startswith(open_s, i):
            depth += 1
            i += len(open_s)
            continue
        if text.startswith(close_s, i):
            depth = max(0, depth - 1)
            i += len(close_s)
            continue
        if not depth:
            out.append(text[i])
        i += 1
    return "".join(out)


def strip_wikitext(text):
    """Reduce a raw wiki page to its readable prose (see doc/datasets.md 1.5)."""
    text = _COMMENT.sub("", text)
    text = _SCRIPT.sub("", text)
    text = _REF.sub("", text)
    text = _strip_balanced(text, "{|", "|}")
    text = _strip_balanced(text, "{{", "}}")
    text = _MEDIA.sub("", text)
    text = _LINK.sub(_link_label, text)
    text = _BRACKETS.sub("", text)
    text = _EXT_LINK.sub(lambda m: m.group(1), text)
    text = _BREAK.sub("\n", text)
    text = _HEADING.sub(lambda m: m.group(1), text)
    text = _MAGIC.sub("", text)
    for _ in range(2):  # scraped HTML is often escaped twice (`&amp;#9;`)
        text = _ENTITY_RE.sub(_entity, text)
    text = _HTML.sub("", text)
    text = _TABLE_ATTR.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _JUNK.sub("", text)
    text = _BLANK.sub("\n\n", text)
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


# ---- English small-set noise ---------------------------------------------------------------------
def clean_daily_dialog(turns):
    """DailyDialog ships its punctuation separated by a space (`after dinner ?`)."""
    return "\n".join(re.sub(r"\s+([?.!,;:])", r"\1", t).strip() for t in turns if t and t.strip())


def clean_empathetic(prompt, utterance):
    """EmpatheticDialogues escapes commas inside its prompts as `_comma_`."""
    return "\n".join(t.replace("_comma_", ",").strip() for t in (prompt, utterance) if t)


_TOPICAL_TURN = re.compile(r"</s>\s*<s>\s*\[agent_\d+\]|<s>\s*\[agent_\d+\]|</?s>|\[agent_\d+\]")


def clean_topical(conversation):
    """Topical-Chat wraps every turn in `<s>[agent_1]…</s>` markers."""
    return _TOPICAL_TURN.sub("\n", conversation or "").strip()


_CORNELL_SPEAKER = re.compile(r"^\s*[A-Z][A-Z0-9'.\- ]{0,30}?\s{2,}")
# The script dump keeps the site's editorial emphasis (`<u>rough</u>`) and escaped quotes (`&quot;`).
_DECOR_TAG = re.compile(r"</?\s*(?:u|i|b|s|em|strong|br)\s*/?>", re.I)


def clean_cornell(text):
    """Cornell rows are one line each, prefixed with the character name (`BIANCA   They do not!`)."""
    text = _ENTITY_RE.sub(_entity, _DECOR_TAG.sub("", (text or "").replace("\r", "")))
    return _CORNELL_SPEAKER.sub("", text).strip()


# ------------------------------------------------------------------------------------------------
# b-corpus (.txt.gz) and ChatHaruhi
# ------------------------------------------------------------------------------------------------
def _bcorpus_paths(root, patterns):
    base = os.path.join(root, BCORPUS)
    paths = []
    for pat in patterns:
        paths += glob.glob(os.path.join(base, pat), recursive=True)
    return sorted({p for p in paths if p.endswith(".gz")})


def _bcorpus(root, out, patterns, player="你"):
    written = 0
    with Sink(out) as sink:
        for path in _bcorpus_paths(root, patterns):
            lines = (clean_bcorpus_line(ln, player) for ln in read_gz(path))
            for doc in chunk_texts(x for x in lines if not is_noise_line(x)):
                sink.write(doc)
            written += 1
    return sink.docs, sink.chars, f"{written} files"


@job("bcorpus_zh", "chat/bcorpus_zh.jsonl", ["Limour--b-corpus/v-corpus-zh/**/*.txt.gz"])
def bcorpus_zh(root, out):
    return _bcorpus(root, out, ["v-corpus-zh/**/*.txt.gz"])


@job("bcorpus_en", "en_chat/bcorpus_en.jsonl", ["Limour--b-corpus/v-corpus-en/*.txt.gz"])
def bcorpus_en(root, out):
    return _bcorpus(root, out, ["v-corpus-en/*.txt.gz"], player="you")


@job("bcorpus_fiction", "fiction/bcorpus_fiction.jsonl",
     ["Limour--b-corpus/近现代小说/**/*.txt.gz", "Limour--b-corpus/翻译名著/**/*.txt.gz",
      "Limour--b-corpus/轻小说/**/*.txt.gz"])
def bcorpus_fiction(root, out):
    return _bcorpus(root, out, ["近现代小说/**/*.txt.gz", "翻译名著/**/*.txt.gz", "轻小说/**/*.txt.gz"])


@job("bcorpus_long", "long/bcorpus_long.jsonl",
     ["Limour--b-corpus/提高逼格/**/*.txt.gz", "Limour--b-corpus/畅销书/**/*.txt.gz",
      "Limour--b-corpus/散文集/**/*.txt.gz", "Limour--b-corpus/古籍/**/*.txt.gz"])
def bcorpus_long(root, out):
    return _bcorpus(root, out, ["提高逼格/**/*.txt.gz", "畅销书/**/*.txt.gz", "散文集/**/*.txt.gz",
                                "古籍/**/*.txt.gz"])


@job("bcorpus_roleplay", "roleplay/bcorpus_roleplay.jsonl", ["Limour--b-corpus/q-corpus/*.txt.gz"])
def bcorpus_roleplay(root, out):
    return _bcorpus(root, out, ["q-corpus/*.txt.gz"])


# ------------------------------------------------------------------------------------------------
# scraped novels: one whole book per record
# ------------------------------------------------------------------------------------------------
# `--max_doc_chars` keeps only the head of a document, and 21% of ChineseNovels' books are longer than
# that limit, so a book has to become chapters before it enters a bucket; `chunk_texts` then cuts each
# chapter at line boundaries.  Only 47% of the books carry explicit headings, the rest fall back to the
# ~4k-character chunks.
_CHAPTER_LINE = re.compile(
    r"^\s*(?:第\s*[0-9零一二三四五六七八九十百千万两]{1,12}\s*[章回节卷话集部篇][^\n]{0,40}"
    r"|(?:Chapter|CHAPTER)\s*\d{1,4}[^\n]{0,40}"
    r"|(?:序章|序言|楔子|尾声|后记|前言|番外|外传)[^\n]{0,30})\s*$")


def split_chapters(text):
    """Cut a whole book at its chapter headings, each heading staying with the body it introduces."""
    chapters, buf = [], []
    for line in text.splitlines():
        if buf and _CHAPTER_LINE.match(line):
            chapters.append("\n".join(buf))
            buf = []
        buf.append(line)
    if buf:
        chapters.append("\n".join(buf))
    return chapters


@job("chinesenovels", "fiction/chinesenovels.jsonl", ["Miaowuawa--ChineseNovels/haitang.jsonl"])
def chinesenovels(root, out):
    """Chapter-split, ad-stripped Miaowuawa/ChineseNovels (doc/datasets.md 1.3)."""
    path = os.path.join(root, "Miaowuawa--ChineseNovels", "haitang.jsonl")
    with Sink(out) as sink, open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            book = _CTRL.sub("", json.loads(line).get("text") or "")
            for chapter in split_chapters(book):
                lines = (strip_markup(ln) for ln in chapter.splitlines() if not is_noise_line(ln))
                for doc in chunk_texts(lines):
                    sink.write(doc)
    return sink.docs, sink.chars, "1 file"


_CONTENT_TAGS = ("narration", "dialog")


def interactive_chapter_text(chapter):
    """One `book_chapter` entry as text: `chapter_title` plus the narration/dialogue rows.

    `content_tag`/`role` are metadata (the old field expression dropped them, and `role` as a speaker tag
    would repeat the name the b-corpus cleaner strips); `img` rows are bare image URLs."""
    parts = [p.get("content") or "" for p in (chapter.get("content") or [])
             if (p.get("content_tag") or "") in _CONTENT_TAGS]
    title = (chapter.get("chapter_title") or "").strip()
    return "\n".join(x for x in (title, strip_markup(_CTRL.sub("", "\n".join(parts)))) if x)


@job("interactive_novels", "fiction/interactive_novels.jsonl",
     ["mrzjy--Chinese_interactive_novels_3k/default/train/*.parquet"])
def interactive_novels(root, out):
    """mrzjy/Chinese_interactive_novels_3k, which ships uncleaned (URLs in `img` rows, author notes)."""
    files = sorted(glob.glob(os.path.join(root, "mrzjy--Chinese_interactive_novels_3k", "default",
                                          "train", "*.parquet")))
    with Sink(out) as sink:
        for path in files:
            for rec in _parquet_rows(path, ["book_chapter"], batch_size=256):
                for chapter in rec.get("book_chapter") or []:
                    lines = [ln for ln in interactive_chapter_text(chapter).splitlines()
                             if not is_noise_line(ln)]
                    for doc in chunk_texts(lines):
                        sink.write(doc)
    return sink.docs, sink.chars, f"{len(files)} shards"


@job("kajiwoto", "en_chat/kajiwoto.jsonl", ["Limour--b-corpus/emoji/kajiwoto.txt.gz"])
def kajiwoto(root, out):
    """The `user：`/`gpt：` chat log, one line per turn (see doc/datasets.md 1.2)."""
    path = os.path.join(root, BCORPUS, "emoji", "kajiwoto.txt.gz")
    with Sink(out) as sink:
        for doc in chunk_texts(clean_line(ln) for ln in read_gz(path)):
            sink.write(doc)
    return sink.docs, sink.chars, "1 file"


@job("chat_haruhi", "roleplay_char/chat_haruhi.jsonl",
     ["Limour--b-corpus/ChatHaruhi/RolePlaying/*.txt.gz", "Limour--b-corpus/ChatHaruhi/Waifu/*.txt.gz"])
def chat_haruhi(root, out):
    """Per-character files; the leading `system：` line is dropped and the file is routed by its Han ratio."""
    base = os.path.join(root, BCORPUS, "ChatHaruhi")
    files = sorted(glob.glob(os.path.join(base, "*", "*.txt.gz")))
    out_dir = os.path.dirname(out)
    zh_path = os.path.join(out_dir, "chat_haruhi_zh.jsonl")
    en_path = os.path.join(out_dir, "chat_haruhi_en.jsonl")
    n_zh = n_en = 0
    with Sink(zh_path) as zh, Sink(en_path) as en:
        for path in files:
            lines = [ln for ln in read_gz(path) if ln.strip()]
            if not lines:
                continue
            body = lines[1:] if lines[0].lstrip().startswith(("system：", "system:")) else lines
            zh_side = han_ratio("".join(body)) >= 0.1
            sink = zh if zh_side else en
            for doc in chunk_texts(clean_line(ln) for ln in body):
                sink.write(doc)
            n_zh += zh_side
            n_en += not zh_side
    return zh.docs + en.docs, zh.chars + en.chars, f"{n_zh} zh / {n_en} en files"


# ------------------------------------------------------------------------------------------------
# JSON arrays
# ------------------------------------------------------------------------------------------------
@job("looksjuicy", "roleplay/looksjuicy.jsonl",
     ["LooksJuicy--Chinese-Roleplay-SingleTurn/Roleplay_Anime_xhj_alpaca.json"])
def looksjuicy(root, out):
    """`instruction` becomes `system_prompt`, or the character card is repeated in every user turn.

    The record keeps both spellings of the content: pretraining reads `text`, SFT reads the
    `system_prompt`/`input`/`output` fields through normalize_messages (doc/datasets.md 1.6).  The cards
    are copied from wiki pages, so `strip_wikitext` runs over all three fields instead of only the
    pretraining text (`{{黑幕|…}}`, `<s>…</s>`, `[[…]]`, `<ref>…</ref>`)."""
    path = os.path.join(root, "LooksJuicy--Chinese-Roleplay-SingleTurn", "Roleplay_Anime_xhj_alpaca.json")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    with Sink(out, records=True) as sink:
        for rec in records:
            card = strip_wikitext(rec.get("instruction") or "")
            user = strip_wikitext(rec.get("input") or "")
            reply = strip_wikitext(rec.get("output") or "")
            sink.write_record({"text": "\n".join(x for x in (card, user, reply) if x),
                               "system_prompt": card, "input": user, "output": reply})
    return sink.docs, sink.chars, "json array"


_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
_CAT_GIRL_FILES = ("v2/common-v2.json", "v2/create-v2.json", "v2/create-v2-next-nothink.json")


@job("catgirl", "roleplay_catgirl/catgirl.jsonl",
     ["kxdw2580--catgirl-dataset/v2/common-v2.json", "kxdw2580--catgirl-dataset/v2/create-v2.json",
      "kxdw2580--catgirl-dataset/v2/create-v2-next-nothink.json"])
def catgirl(root, out):
    """v2 only (the author rules out v1 and the qwen3 series); the inline chain of thought is dropped."""
    base = os.path.join(root, "kxdw2580--catgirl-dataset")
    with Sink(out, records=True) as sink:
        for rel in _CAT_GIRL_FILES:
            with open(os.path.join(base, rel), encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                inst, user = rec.get("instruction", ""), rec.get("input", "")
                reply = _THINK.sub("", rec.get("output", "")).strip()
                # Alpaca shape: the question lives in `instruction`, `input` only carries extra context.
                sink.write_record({"text": "\n".join(x for x in (inst, user, reply) if x),
                                   "instruction": inst, "input": user, "output": reply})
    return sink.docs, sink.chars, f"{len(_CAT_GIRL_FILES)} files"


# ------------------------------------------------------------------------------------------------
# wiki and the English small sets (parquet / jsonl inputs)
# ------------------------------------------------------------------------------------------------
def _parquet_rows(path, columns, batch_size=4096):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size):
        yield from batch.to_pylist()


@job("moegirl", "wiki/moegirl.jsonl",
     ["mrzjy--chinese_moegirl_wiki_corpus_raw/default/train/*.parquet"])
def moegirl(root, out):
    """Raw wikitext in, prose out: 99.6% of the pages carry `[[`, 91.5% `{{` (doc/datasets.md 1.5)."""
    files = sorted(glob.glob(os.path.join(root, "mrzjy--chinese_moegirl_wiki_corpus_raw",
                                          "default", "train", "*.parquet")))
    skipped = 0
    with Sink(out) as sink:
        for path in files:
            for rec in _parquet_rows(path, ["text"]):
                raw = rec.get("text") or ""
                if junk_ratio(raw) > 0.02:      # a page the scraper stored as binary, not wikitext
                    skipped += 1
                    continue
                sink.write(strip_wikitext(raw))
    return sink.docs, sink.chars, f"{len(files)} shards, {skipped} mojibake pages dropped"


@job("daily_dialog", "en_chat/daily_dialog.jsonl",
     ["li2017dailydialog--daily_dialog/default/train/*.parquet"])
def daily_dialog(root, out):
    files = sorted(glob.glob(os.path.join(root, "li2017dailydialog--daily_dialog", "default", "train",
                                          "*.parquet")))
    with Sink(out) as sink:
        for path in files:
            for rec in _parquet_rows(path, ["dialog"]):
                sink.write(clean_daily_dialog(rec.get("dialog") or []))
    return sink.docs, sink.chars, f"{len(files)} shards"


@job("empathetic", "en_chat/empathetic.jsonl",
     ["facebook--empathetic_dialogues/default/train/*.parquet"])
def empathetic(root, out):
    files = sorted(glob.glob(os.path.join(root, "facebook--empathetic_dialogues", "default", "train",
                                          "*.parquet")))
    with Sink(out) as sink:
        for path in files:
            for rec in _parquet_rows(path, ["prompt", "utterance"]):
                sink.write(clean_empathetic(rec.get("prompt"), rec.get("utterance")))
    return sink.docs, sink.chars, f"{len(files)} shards"


@job("topical_chat", "en_chat/topical_chat.jsonl", ["Fishball02--topical-chat/data/train-*.parquet"])
def topical_chat(root, out):
    files = sorted(glob.glob(os.path.join(root, "Fishball02--topical-chat", "data", "train-*.parquet")))
    with Sink(out) as sink:
        for path in files:
            for rec in _parquet_rows(path, ["conversation"]):
                sink.write(clean_topical(rec.get("conversation")))
    return sink.docs, sink.chars, f"{len(files)} shards"


@job("cornell", "en_chat/cornell.jsonl",
     ["mylesmharrison--cornell-movie-dialog/default/train/*.parquet"])
def cornell(root, out):
    """One line per utterance in the source, so consecutive lines are grouped back into exchanges."""
    files = sorted(glob.glob(os.path.join(root, "mylesmharrison--cornell-movie-dialog", "default",
                                          "train", "*.parquet")))
    with Sink(out) as sink:
        for path in files:
            lines = (clean_cornell(rec.get("text")) for rec in _parquet_rows(path, ["text"]))
            for doc in chunk_texts(lines):
                sink.write(doc)
    return sink.docs, sink.chars, f"{len(files)} shards"


@job("esconv", "en_chat/esconv.jsonl", ["thu-coai--esconv/ESConv.jsonl"])
def esconv(root, out):
    path = os.path.join(root, "thu-coai--esconv", "ESConv.jsonl")
    with Sink(out) as sink, open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sink.write("\n".join(t.get("content", "").strip() for t in rec.get("dialog") or []))
    return sink.docs, sink.chars, "1 file"


# ------------------------------------------------------------------------------------------------
# SFT slices
# ------------------------------------------------------------------------------------------------
def _spool_messages(out, specs):
    """Write the `{"messages": [...]}` shape ChatDataset reads, one record per usable source record.

    `--mm_data` accepts source specs directly, but the recipe's 74 / 21 / 5 slice is three items, so the
    two group files are written once here.  Running every record through normalize_messages also reports
    how many are unusable (`assistant` roles, instruction-only rows) before a training run starts."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from common_io import iter_records
    from hvlm.data_core import normalize_messages

    bad = 0
    with Sink(out, records=True) as sink:
        for spec in specs:
            for rec in iter_records(spec):
                try:
                    msgs = normalize_messages(dict(rec))
                    if not any(m["role"] == "self" for m in msgs):
                        raise ValueError("no 'self' turn")
                except Exception:
                    bad += 1
                    continue
                sink.write_record({"messages": msgs})
    return sink.docs, sink.chars, f"{len(specs)} sources, {bad} unusable"


@job("sft_human_dialog", "sft/human_dialog.jsonl", ["silver--lccc/base/train/*.parquet"])
def sft_human_dialog(root, out):
    """Human-to-human chat for SFT: LCCC base, the same source the pretraining chat bin reads first."""
    return _spool_messages(out, [f"parquet:{root}/silver--lccc/base/train/*.parquet:dialog"])


@job("sft_persona", "sft/persona.jsonl",
     ["shibing624--roleplay-zh/sharegpt_formatted_data-evol-gpt4.jsonl",
      "Johnson8187--role-play-chinese/alpaca_format_simplified.jsonl"])
def sft_persona(root, out):
    """Persona multi-turn sets; the third source is the `looksjuicy` job's output, so run that first."""
    return _spool_messages(out, [
        f"jsonl:{root}/shibing624--roleplay-zh/sharegpt_formatted_data-evol-gpt4.jsonl:-",
        f"jsonl:{root}/Johnson8187--role-play-chinese/alpaca_format_simplified.jsonl:-",
        "jsonl:data/roleplay/looksjuicy.jsonl:-",
    ])


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                    help=f"directory holding the hf/<org>--<name>/ snapshots (default {DEFAULT_SNAPSHOT})")
    ap.add_argument("--out", default="data", help="bucket root the job outputs land in (default data)")
    ap.add_argument("--only", nargs="+", metavar="JOB", help="run a subset of the jobs")
    ap.add_argument("--force", action="store_true", help="redo outputs that already exist")
    ap.add_argument("--list", action="store_true", help="print the jobs and exit")
    args = ap.parse_args()

    if args.list:
        for name, spec in JOBS.items():
            print(f"{name:16s} -> {spec['out']}")
            for src in spec["sources"]:
                print(f"{'':16s}    {src}")
        return
    names = args.only or list(JOBS)
    unknown = [n for n in names if n not in JOBS]
    if unknown:
        raise SystemExit(f"unknown job(s) {unknown}; --list shows the choices")
    if not os.path.isdir(args.snapshot):
        raise SystemExit(f"snapshot directory not found: {args.snapshot!r} (--snapshot points at it)")

    failed = []
    for name in names:
        spec = JOBS[name]
        out = os.path.join(args.out, spec["out"])
        if os.path.exists(out) and not args.force:
            print(f"[skip] {name}: {out} exists (--force to redo)")
            continue
        try:
            docs, chars, note = spec["fn"](args.snapshot, out)
        except FileNotFoundError as e:
            failed.append(name)
            print(f"[FAIL] {name}: {e}", file=sys.stderr)
            continue
        print(f"[ok]   {name}: {docs} documents, {chars / 1e6:.1f}M chars -> {out} ({note})")
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}")
    print("all jobs done; run_pipeline.sh reads these through its data/<bucket>/*.jsonl globs")


if __name__ == "__main__":
    main()
