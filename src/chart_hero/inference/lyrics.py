from __future__ import annotations

"""
Lyrics fetching and normalization utilities.

Goals
- Prefer synced lyrics with word timings (LRCLIB when available).
- Fallback to YouTube captions via yt_dlp when LRCLIB misses.
- As a last resort, accept unsynced text and time it heuristically.

This module intentionally has no hard dependencies beyond the stdlib and
yt_dlp (already in the project). If optional libraries are present, they are
used opportunistically:
- pronouncing: CMUdict-based syllabification
- pyphen: hyphenation fallback for syllabification
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
import json
import math
import re
import os
import urllib.parse
import urllib.request

try:  # optional
    import pronouncing  # type: ignore
except Exception:  # pragma: no cover - optional
    pronouncing = None  # type: ignore

try:  # optional
    import pyphen  # type: ignore
except Exception:  # pragma: no cover - optional
    pyphen = None  # type: ignore

try:
    from yt_dlp import YoutubeDL  # type: ignore
except Exception:  # pragma: no cover - yt_dlp is a project dep
    YoutubeDL = None  # type: ignore

# Optional: SyncedLyrics aggregator
try:  # optional
    import syncedlyrics  # type: ignore
except Exception:  # pragma: no cover - optional
    syncedlyrics = None  # type: ignore


@dataclass
class Syllable:
    text: str
    t0: float
    t1: float


@dataclass
class Word:
    text: str
    t0: float
    t1: float
    syllables: List[Syllable]


@dataclass
class Line:
    text: str
    t0: float
    t1: float
    words: List[Word]


@dataclass
class Lyrics:
    source: str  # e.g., "lrclib", "youtube_captions", "unsynced"
    confidence: float
    lines: List[Line]
    raw_lrc: Optional[str] = None


# ---- LRCLIB integration ----------------------------------------------------

LRCLIB_BASE = "https://lrclib.net/api"


def _log(msg: str) -> None:
    try:
        print(f"[LYRICS] {msg}")
    except Exception:
        pass


def _http_json(url: str, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "chart-hero/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        ctype = resp.headers.get("Content-Type", "application/json")
        data = resp.read()
        if "application/json" in ctype:
            return json.loads(data.decode("utf-8", errors="replace"))
        return data.decode("utf-8", errors="replace")


def fetch_lrclib_by_spotify_id(spotify_id: str) -> Optional[str]:
    """Return synced LRC text for a Spotify track ID, if present."""
    if not spotify_id:
        return None
    url = f"{LRCLIB_BASE}/get?spotifyId={urllib.parse.quote(spotify_id)}"
    try:
        _log(f"LRCLIB get by Spotify ID: {spotify_id}")
        js = _http_json(url)
        if isinstance(js, dict):
            # API usually exposes 'syncedLyrics' (string, may be empty)
            lrc = js.get("syncedLyrics") or js.get("synced_lyrics")
            if isinstance(lrc, str) and lrc.strip():
                _log("LRCLIB returned synced lyrics by Spotify ID.")
                return lrc
            _log("LRCLIB had no synced lyrics for Spotify ID.")
    except Exception as e:
        _log(
            "LRCLIB request by Spotify ID failed: "
            + f"error={e.__class__.__name__}: {str(e)}"
        )
        return None
    return None


def fetch_lrclib_search(
    *, track: str, artist: str | None, album: str | None, duration: float | None
) -> Optional[str]:
    """Search LRCLIB by metadata and return synced LRC if a confident match exists.

    Tries progressively looser queries: title+artist+album+duration, then drop album, then drop duration, then title only.
    Duration tolerance widened to 5 seconds to accommodate minor mismatches.
    """

    def _search(params: dict[str, str]) -> Optional[str]:
        url = f"{LRCLIB_BASE}/search?{urllib.parse.urlencode(params)}"
        try:
            _log("LRCLIB search: " + json.dumps(params, ensure_ascii=False))
            js = _http_json(url)
            if isinstance(js, list):
                _log(f"LRCLIB search returned {len(js)} candidate(s).")
                for it in js:
                    if not isinstance(it, dict):
                        continue
                    lrc = it.get("syncedLyrics") or it.get("synced_lyrics")
                    if not (isinstance(lrc, str) and lrc.strip()):
                        continue
                    if duration is not None and duration > 0:
                        d = it.get("duration") or it.get("durationMs")
                        try:
                            d_val = float(d) if not isinstance(d, str) else float(d)
                            if abs(d_val - float(duration)) > 5.0:
                                _log(
                                    f"Skip candidate: duration mismatch ({d} vs {duration})."
                                )
                                continue
                        except Exception:
                            pass
                    return lrc
        except Exception as e:
            _log(
                "LRCLIB search request failed: "
                + json.dumps(params, ensure_ascii=False)
                + f"; error={e.__class__.__name__}: {str(e)}"
            )
            return None
        return None

    params_base: dict[str, str] = {"track_name": track}
    if artist:
        params_base["artist_name"] = artist
    if album:
        params_base["album_name"] = album
    if duration is not None and duration > 0:
        params_base["duration"] = str(int(round(duration)))

    # Try with full params
    lrc = _search(params_base)
    if lrc:
        return lrc
    # Try without album
    params_wo_album = {k: v for k, v in params_base.items() if k != "album_name"}
    lrc = _search(params_wo_album)
    if lrc:
        return lrc
    # Try without duration
    params_wo_dur = {k: v for k, v in params_wo_album.items() if k != "duration"}
    lrc = _search(params_wo_dur)
    if lrc:
        return lrc
    # Try title only
    lrc = _search({"track_name": track})
    return lrc


# ---- YouTube captions fallback --------------------------------------------


def fetch_youtube_captions(
    link: str, *, lang_pref: Iterable[str] = ("en", "en-US", "en-GB")
) -> Optional[str]:
    """Return WebVTT captions text if available using yt_dlp without downloading.

    Prefers manually provided subtitles, falls back to automatic captions.
    """
    if YoutubeDL is None:
        _log("yt_dlp not available; skipping YouTube captions.")
        return None
    try:
        with YoutubeDL({"quiet": True, "writesubtitles": True}) as ydl:
            info = ydl.extract_info(link, download=False)
        if not isinstance(info, dict):
            return None
        # Try regular subtitles first
        for field in ("subtitles", "automatic_captions"):
            subs = info.get(field) or {}
            if not isinstance(subs, dict):
                continue
            _log(f"YouTube {field} languages: {sorted(list(subs.keys()))}")
            # find best language
            for lang in lang_pref:
                tracks = subs.get(lang)
                if not tracks:
                    continue
                # prefer vtt
                vtt = None
                for tr in tracks:
                    if tr.get("ext") == "vtt":
                        vtt = tr
                        break
                vtt = vtt or (tracks[0] if isinstance(tracks, list) else None)
                if vtt and vtt.get("url"):
                    url = vtt["url"]
                    try:
                        text = _http_json(url)  # not JSON, returns text
                        if isinstance(text, str) and text.strip():
                            _log(
                                f"Fetched YouTube captions from lang={lang} ({field})."
                            )
                            return text
                    except Exception:
                        _log("Failed to download YouTube captions text.")
                        continue
        return None
    except Exception:
        _log("yt_dlp extract_info failed.")
        return None


# ---- Parsing utilities -----------------------------------------------------

_LRC_TIME = re.compile(r"^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$")


def _time_to_seconds(txt: str) -> float:
    m = _LRC_TIME.match(txt)
    if not m:
        return 0.0
    mm = int(m.group(1))
    ss = int(m.group(2))
    ms = int((m.group(3) or "0").ljust(3, "0"))
    return mm * 60.0 + ss + ms / 1000.0


def parse_lrc(lrc: str) -> List[Line]:
    """Parse LRC with optional inline word timings <mm:ss.xx>word.

    Returns lines with word timings; if only line timings present, words share the line span.
    """
    lines: List[Line] = []
    for raw in lrc.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # extract timestamps like [mm:ss.xx] (may be multiple per line)
        stamps = re.findall(r"\[(\d{1,2}:\d{2}(?:\.\d{1,3})?)\]", raw)
        text = re.sub(r"\[(?:\d{1,2}:\d{2}(?:\.\d{1,3})?)\]", "", raw).strip()
        if not stamps:
            continue
        # Support a single timestamp per line (common case). Use next line/time for t1 later if possible.
        # For inline word times like <mm:ss.xx>word, prefer those
        words: List[Word] = []
        inline = list(
            re.finditer(r"<(?P<t>\d{1,2}:\d{2}(?:\.\d{1,3})?)>(?P<w>[^<]+)", text)
        )
        if inline:
            # produce word spans by next tag or end of line timebox (~+0.35s default min)
            temp_words: List[tuple[str, float]] = [
                (m.group("w").strip(), _time_to_seconds(m.group("t"))) for m in inline
            ]
            temp_words.sort(key=lambda x: x[1])
            for i, (w, t0) in enumerate(temp_words):
                t1 = temp_words[i + 1][1] if i + 1 < len(temp_words) else (t0 + 0.35)
                sylls = _syllabify_to_slices(w, t0, t1)
                words.append(Word(text=w, t0=t0, t1=t1, syllables=sylls))
            # line timing bounds from first/last word
            lt0 = words[0].t0
            lt1 = words[-1].t1
            lines.append(Line(text=text, t0=lt0, t1=lt1, words=words))
            continue

        # Only line-level timing: allocate evenly across whitespace-separated words
        t0 = _time_to_seconds(stamps[0])
        # Approximate end: next stamp on same line (rare), else +3s default window
        t1 = _time_to_seconds(stamps[1]) if len(stamps) > 1 else (t0 + 3.0)
        raw_words = [w for w in re.split(r"\s+", text) if w]
        if not raw_words:
            continue
        span = max(t1 - t0, 0.35)
        per = span / len(raw_words)
        cur = t0
        for w in raw_words:
            w0, w1 = cur, cur + per
            cur = w1
            sylls = _syllabify_to_slices(w, w0, w1)
            words.append(Word(text=w, t0=w0, t1=w1, syllables=sylls))
        lines.append(Line(text=text, t0=t0, t1=t1, words=words))
    return lines


def parse_vtt(vtt: str) -> List[Line]:
    """Parse a minimal WebVTT transcript to lines and rough word timings.

    Many YT captions are line-level only; we split line spans evenly across words.
    """

    def parse_vtt_time(ts: str) -> float:
        # Accept hh:mm:ss.mmm or mm:ss.mmm
        parts = ts.split("-->")
        if len(parts) == 2:
            start, end = parts[0].strip(), parts[1].strip().split()[0]
        else:
            start, end = ts.split()[0], ts.split()[2]

        def to_sec(x: str) -> float:
            xs = x.strip()
            h = 0
            if xs.count(":") == 2:
                h, m, s = xs.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            m, s = xs.split(":")
            return int(m) * 60 + float(s)

        return to_sec(start), to_sec(end)

    lines: List[Line] = []
    cur_text: List[str] = []
    cur_t0: Optional[float] = None
    cur_t1: Optional[float] = None
    for raw in vtt.splitlines():
        raw = raw.strip("\ufeff\n\r ")
        if not raw:
            if cur_text and cur_t0 is not None and cur_t1 is not None:
                full = " ".join(cur_text).strip()
                words: List[Word] = []
                if full:
                    toks = [w for w in re.split(r"\s+", full) if w]
                    span = max((cur_t1 - cur_t0), 0.35)
                    per = span / len(toks)
                    t = cur_t0
                    for w in toks:
                        w0, w1 = t, t + per
                        t = w1
                        sylls = _syllabify_to_slices(w, w0, w1)
                        words.append(Word(text=w, t0=w0, t1=w1, syllables=sylls))
                    lines.append(Line(text=full, t0=cur_t0, t1=cur_t1, words=words))
            cur_text, cur_t0, cur_t1 = [], None, None
            continue
        if "-->" in raw:
            try:
                t0, t1 = parse_vtt_time(raw)
                cur_t0, cur_t1 = t0, t1
                cur_text = []
            except Exception:
                continue
        elif raw.startswith("WEBVTT") or raw.isdigit():
            continue
        else:
            cur_text.append(raw)
    return lines


# ---- Syllabification -------------------------------------------------------

_VOWEL_GROUPS = re.compile(r"(?i:[aeiouy]+)")


def _syllables_pronouncing(word: str) -> List[str]:  # pragma: no cover - optional
    if pronouncing is None:
        return []
    phones = pronouncing.phones_for_word(re.sub(r"[^a-zA-Z']", "", word.lower()))
    if not phones:
        return []
    # Estimate syllable count from stressed vowel markers
    syl_count = pronouncing.syllable_count(phones[0])
    if syl_count <= 1:
        return [word]
    # No exact split boundaries provided; fallback to hyphenation/heuristic below
    return []


def _syllables_pyphen(word: str) -> List[str]:  # pragma: no cover - optional
    if pyphen is None:
        return []
    dic = pyphen.Pyphen(lang="en")
    hy = dic.inserted(word)
    return hy.split("-") if hy else []


def _rough_syllables(word: str) -> List[str]:
    # Very rough fallback: split around vowel groups; ensure non-empty
    w = word
    if len(w) <= 3:
        return [w]
    parts: List[str] = []
    last = 0
    for m in _VOWEL_GROUPS.finditer(w):
        end = m.end()
        parts.append(w[last:end])
        last = end
    if last < len(w):
        parts.append(w[last:])
    return [p for p in parts if p]


def syllabify_word(word: str) -> List[str]:
    if not word:
        return []
    # Try higher-quality splitters first
    s = _syllables_pronouncing(word)
    if not s:
        s = _syllables_pyphen(word)
    if not s:
        s = _rough_syllables(word)
    return s if s else [word]


def _syllabify_to_slices(word: str, t0: float, t1: float) -> List[Syllable]:
    syls = syllabify_word(word)
    if len(syls) <= 1 or (t1 <= t0):
        return [Syllable(text=word, t0=t0, t1=max(t1, t0 + 0.05))]
    # allocate time proportionally by character length of each syllable
    total_chars = sum(max(len(x), 1) for x in syls)
    span = max(t1 - t0, 0.05)
    out: List[Syllable] = []
    cur = t0
    for i, s in enumerate(syls):
        frac = max(len(s), 1) / total_chars
        dur = span * frac
        s0 = cur
        s1 = (t0 + span) if i == (len(syls) - 1) else (cur + dur)
        out.append(Syllable(text=s, t0=s0, t1=s1))
        cur = s1
    return out


# ---- Public entrypoint -----------------------------------------------------


def get_synced_lyrics(
    *,
    link: Optional[str] = None,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    duration: Optional[float] = None,
    spotify_id: Optional[str] = None,
) -> Optional[Lyrics]:
    """Fetch best-available synced lyrics and normalize to Lines/Words/Syllables.

    Strategy: LRCLIB by Spotify ID -> LRCLIB search -> YouTube captions.
    Returns None when nothing usable is found.
    """
    # 1) LRCLIB by Spotify ID
    if spotify_id:
        lrc = fetch_lrclib_by_spotify_id(spotify_id)
        if isinstance(lrc, str) and lrc.strip():
            lines = parse_lrc(lrc)
            if lines:
                return Lyrics(
                    source="lrclib", confidence=0.95, lines=lines, raw_lrc=lrc
                )

    # 2) LRCLIB textual search (try as-is, then a normalized title without qualifiers)
    def _normalize_title(t: str) -> str:
        # remove bracketed qualifiers and trailing dashes
        t2 = re.sub(r"\s*\([^)]*\)", "", t)
        t2 = re.sub(r"\s*\[[^\]]*\]", "", t2)
        t2 = re.sub(r"\s*-\s*[^-]+$", "", t2).strip()
        return t2 or t

    if title:
        tried_titles = [title]
        norm = _normalize_title(title)
        if norm != title:
            tried_titles.append(norm)
        for t in tried_titles:
            if t != title:
                _log(f"LRCLIB retry with normalized title: '{t}'")
            lrc = fetch_lrclib_search(
                track=t, artist=artist, album=album, duration=duration
            )
            if isinstance(lrc, str) and lrc.strip():
                lines = parse_lrc(lrc)
                if lines:
                    return Lyrics(
                        source="lrclib", confidence=0.8, lines=lines, raw_lrc=lrc
                    )

    # 3) SyncedLyrics (optional aggregator)
    try:
        use_sl = (syncedlyrics is not None) and (
            os.environ.get("USE_SYNCEDLYRICS", "1") != "0"
        )
        if use_sl and title:
            _log(
                "SyncedLyrics search: "
                + json.dumps(
                    {
                        "title": title,
                        "artist": artist,
                        "duration": duration,
                    },
                    ensure_ascii=False,
                )
            )
            lrc_sl: Optional[str] = None
            # Try common call patterns
            try:
                if hasattr(syncedlyrics, "search"):
                    # Prefer explicit fields if supported
                    try:
                        lrc_sl = syncedlyrics.search(
                            song=title,
                            artist=artist or "",
                            providers=None,
                            debug=False,
                        )
                    except TypeError:
                        # Fallback to single-query signature
                        q = f"{artist} - {title}" if artist else title
                        lrc_sl = syncedlyrics.search(q)
            except Exception:
                lrc_sl = None
            if isinstance(lrc_sl, str) and lrc_sl.strip():
                lines = parse_lrc(lrc_sl)
                if lines:
                    return Lyrics(
                        source="syncedlyrics",
                        confidence=0.9,
                        lines=lines,
                        raw_lrc=lrc_sl,
                    )
            _log("SyncedLyrics returned no usable LRC.")
    except Exception as e:
        _log(f"SyncedLyrics attempt failed: error={e.__class__.__name__}: {str(e)}")

    # 4) YouTube captions fallback
    if link:
        vtt = fetch_youtube_captions(link)
        if isinstance(vtt, str) and vtt.strip():
            lines = parse_vtt(vtt)
            if lines:
                return Lyrics(source="youtube_captions", confidence=0.5, lines=lines)

    # 5) AudD unsynced lyrics fallback (heuristic timing)
    try:
        api_token = os.environ.get("AUDD_API_TOKEN")
        if api_token and title:
            _log(
                "AudD unsynced fallback: findLyrics; "
                + json.dumps(
                    {
                        "title": title,
                        "artist": artist,
                        "duration": duration,
                        "spotify_id_present": bool(spotify_id),
                    },
                    ensure_ascii=False,
                )
            )
            txt = fetch_audd_unsynced_lyrics(
                title=title, artist=artist, api_token=api_token
            )
            if isinstance(txt, str) and txt.strip():
                if duration and duration > 0:
                    lines = time_unsynced_lyrics(txt, duration)
                else:
                    # Assume ~3s per line when duration unknown
                    est_dur = 3.0 * max(
                        1, len([ln for ln in txt.splitlines() if ln.strip()])
                    )
                    lines = time_unsynced_lyrics(txt, est_dur)
                if lines:
                    return Lyrics(source="unsynced", confidence=0.3, lines=lines)
            # If findLyrics had no results, try /lyrics endpoint
            _log(
                "AudD unsynced fallback: lyrics; "
                + json.dumps(
                    {
                        "title": title,
                        "artist": artist,
                        "duration": duration,
                        "spotify_id_present": bool(spotify_id),
                    },
                    ensure_ascii=False,
                )
            )
            txt2 = fetch_audd_lyrics_query(
                title=title, artist=artist, api_token=api_token
            )
            if isinstance(txt2, str) and txt2.strip():
                if duration and duration > 0:
                    lines = time_unsynced_lyrics(txt2, duration)
                else:
                    est_dur = 3.0 * max(
                        1, len([ln for ln in txt2.splitlines() if ln.strip()])
                    )
                    lines = time_unsynced_lyrics(txt2, est_dur)
                if lines:
                    return Lyrics(source="unsynced", confidence=0.3, lines=lines)
    except Exception:
        pass

    return None


def to_rb_tokens(lines: Iterable[Line]) -> List[tuple[Syllable, str]]:
    """Convert syllables to Rock Band-style lyric tokens used by Clone Hero.

    Adds a trailing '-' to all but the last syllable of each word.
    Returns a flat list of (syllable, token) preserving original order.
    """
    out: List[tuple[Syllable, str]] = []
    for line in lines:
        for w in line.words:
            for i, syl in enumerate(w.syllables):
                tok = syl.text
                if i < len(w.syllables) - 1:
                    tok = tok + "-"
                out.append((syl, tok))
    return out


# ---- AudD unsynced lyrics + heuristic timing ------------------------------


def fetch_audd_unsynced_lyrics(
    *, title: str, artist: Optional[str], api_token: str
) -> Optional[str]:
    q = title if not artist else f"{artist} - {title}"
    url = "https://api.audd.io/findLyrics/?" + urllib.parse.urlencode(
        {"q": q, "api_token": api_token}
    )
    try:
        _log("AudD findLyrics query: " + q)
        js = _http_json(url)
        if isinstance(js, dict):
            res = js.get("result")
            if isinstance(res, list) and res:
                # pick the first result that has lyrics text
                text_hits = 0
                preview: list[str] = []
                for it in res:
                    if isinstance(it, dict):
                        lyr = it.get("lyrics") or it.get("text")
                        if isinstance(lyr, str) and lyr.strip():
                            text_hits += 1
                            # keep first text for return and collect preview list
                            if not preview:
                                # Return the first text hit immediately after logging preview
                                # but first build preview of candidates for visibility
                                pass

                    # build preview line for this candidate
                    try:
                        cand_title = None
                        if isinstance(it, dict):
                            cand_title = it.get("title") or it.get("song_title") or None
                            cand_artist = (
                                it.get("artist") or it.get("artist_name") or None
                            )
                            snippet = (it.get("lyrics") or it.get("text") or "")[
                                :40
                            ].replace("\n", " ")
                            pv = f"{cand_artist or '?'} - {cand_title or '?'} | len={len(it.get('lyrics') or it.get('text') or '')} | {snippet}"
                            preview.append(pv)
                    except Exception:
                        pass
                # log summary of candidates
                _log(
                    "AudD findLyrics candidates: "
                    + json.dumps(preview[:5], ensure_ascii=False)
                    + (" …" if len(preview) > 5 else "")
                )
                # return first text hit if any
                for it in res:
                    if isinstance(it, dict):
                        lyr = it.get("lyrics") or it.get("text")
                        if isinstance(lyr, str) and lyr.strip():
                            _log("AudD findLyrics returned text.")
                            return lyr
        count = len(res) if isinstance(res, list) else 0
        _log(f"AudD findLyrics had no usable results (candidates={count}).")
        return None
    except Exception as e:
        _log(f"AudD findLyrics request failed: error={e.__class__.__name__}: {str(e)}")
        return None


def time_unsynced_lyrics(text: str, duration: float) -> List[Line]:
    """Heuristically assign timings to plain-text lyrics.

    Splits by non-empty lines; distributes duration proportionally by line length.
    Within a line, words split evenly across the line span and syllabified.
    """
    raw_lines = [ln.strip() for ln in text.splitlines()]
    lines_txt = [ln for ln in raw_lines if ln]
    if not lines_txt:
        return []
    # weight by character length to give longer lines more time, with min weight
    weights = [max(len(ln), 10) for ln in lines_txt]
    total_w = float(sum(weights))
    t_cursor = 0.0
    out: List[Line] = []
    for ln, w in zip(lines_txt, weights):
        span = duration * (w / total_w)
        t0 = t_cursor
        t1 = t0 + max(span, 0.75)
        t_cursor = t1
        words: List[Word] = []
        toks = [tok for tok in re.split(r"\s+", ln) if tok]
        if not toks:
            out.append(Line(text=ln, t0=t0, t1=t1, words=[]))
            continue
        per = max((t1 - t0) / len(toks), 0.15)
        cur = t0
        for tok in toks:
            w0, w1 = cur, cur + per
            cur = w1
            sylls = _syllabify_to_slices(tok, w0, w1)
            words.append(Word(text=tok, t0=w0, t1=w1, syllables=sylls))
        out.append(Line(text=ln, t0=t0, t1=t1, words=words))
    return out


def fetch_audd_lyrics_query(
    *, title: str, artist: Optional[str], api_token: str
) -> Optional[str]:
    """AudD lyrics endpoint; often returns a single best match object.

    API docs indicate /lyrics may return {result: {lyrics: "..."}} or a similar shape.
    """
    q = title if not artist else f"{artist} - {title}"
    url = "https://api.audd.io/lyrics/?" + urllib.parse.urlencode(
        {"q": q, "api_token": api_token}
    )
    try:
        js = _http_json(url)
        if isinstance(js, dict):
            res = js.get("result")
            if isinstance(res, dict):
                lyr = res.get("lyrics") or res.get("text")
                if isinstance(lyr, str) and lyr.strip():
                    _log("AudD lyrics returned text.")
                    return lyr
        try:
            keys = list(res.keys()) if isinstance(res, dict) else []
        except Exception:
            keys = []
        _log("AudD lyrics had no usable results." + (f" keys={keys}" if keys else ""))
        return None
    except Exception as e:
        _log(f"AudD lyrics request failed: error={e.__class__.__name__}: {str(e)}")
        return None
