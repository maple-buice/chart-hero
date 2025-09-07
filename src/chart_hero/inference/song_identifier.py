import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import requests

from chart_hero.inference.classes.audd import audd_song_response, audd_song_result


def identify_song(path: str) -> audd_song_result:
    api_token = os.environ.get("AUDD_API_TOKEN")
    if not api_token:
        raise ValueError("AUDD_API_TOKEN environment variable not set")

    data = {"api_token": api_token, "return": "musicbrainz,spotify,apple_music"}
    with open(path, "rb") as f:
        files = {"file": f}
        response = requests.post("https://api.audd.io/", data=data, files=files)
    response.raise_for_status()  # Raise an exception for bad status codes
    response_json = response.json()

    if not response_json or not response_json.get("result"):
        raise ValueError("No result found in AudD API response")

    audd_response = audd_song_response.from_dict(response_json)
    if not audd_response or not audd_response.result:
        raise ValueError("No result found in AudD API response")

    return audd_response.result


def get_data_from_acousticbrainz(song: audd_song_result) -> Dict[str, Any]:
    print(song.musicbrainz[0].id)
    result = requests.get(
        "https://acousticbrainz.org/api/v1/low-level?recording_ids="
        + song.musicbrainz[0].id
    )
    return cast(Dict[str, Any], result.json())


# -------------------- Simple on-disk cache for ID lookups -------------------


def _load_cache(p: str | os.PathLike[str]) -> Dict[str, Any]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_cache(p: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    try:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
    except Exception:
        pass


def _hash_file(path: str, max_bytes: int = 1 << 20) -> str:
    """Hash first max_bytes of file plus size+mtime to build a stable key."""
    try:
        h = hashlib.sha1()
        st = os.stat(path)
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
        with open(path, "rb") as f:
            chunk = f.read(max_bytes)
            if chunk:
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return hashlib.sha1(str(path).encode()).hexdigest()


def identify_song_cached(path: str, cache_path: str) -> audd_song_result:
    """Wrap identify_song with a small JSON cache keyed by file hash."""
    key = f"audd:{_hash_file(path)}"
    cache = _load_cache(cache_path)
    hit = cache.get(key)
    if isinstance(hit, dict):
        try:
            return audd_song_result.from_dict(hit)
        except Exception:
            pass
    result = identify_song(path)
    try:
        cache[key] = result.to_dict()
        _save_cache(cache_path, cache)
    except Exception:
        pass
    return result


# ---- MusicBrainz + AcousticBrainz direct helpers ---------------------------


def search_musicbrainz_recording(
    *,
    title: str,
    artist: Optional[str] = None,
    duration_sec: Optional[float] = None,
    limit: int = 5,
) -> Optional[Dict[str, Any]]:
    """Search MusicBrainz recordings by title/artist and return top candidate JSON.

    Uses WS /recording?query= with fmt=json. Picks highest score; if duration is
    provided, prefers closest duration (in ms).
    """
    if not title:
        return None
    q = f'recording:"{title}"'
    if artist:
        q += f' AND artist:"{artist}"'
    params = {
        "query": q,
        "fmt": "json",
        "limit": str(max(1, min(limit, 25))),
        "inc": "releases",
    }
    headers = {"User-Agent": "chart-hero/0.1 (MB search)"}
    try:
        resp = requests.get(
            "https://musicbrainz.org/ws/2/recording/",
            params=params,
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        js = resp.json()
        recs: List[Dict[str, Any]] = (
            js.get("recordings", []) if isinstance(js, dict) else []
        )
        if not recs:
            return None

        # Score and duration closeness
        def _score(r: Dict[str, Any]) -> Tuple[int, float]:
            sc = int(r.get("score", 0))
            length_ms = r.get("length")
            dur_pen = 0.0
            if duration_sec and isinstance(length_ms, (int, float)) and length_ms > 0:
                dur_pen = abs((float(length_ms) / 1000.0) - float(duration_sec))
            return sc, -dur_pen

        recs.sort(key=_score, reverse=True)
        return recs[0]
    except Exception:
        return None


def get_acousticbrainz_lowlevel_by_mbid(
    recording_mbid: str,
) -> Optional[Dict[str, Any]]:
    """Fetch AcousticBrainz low-level features for a recording MBID."""
    try:
        headers = {"User-Agent": "chart-hero/0.1 (AB lowlevel)"}
        url = (
            "https://acousticbrainz.org/api/v1/low-level?recording_ids="
            + recording_mbid
        )
        result = requests.get(url, headers=headers, timeout=10)
        result.raise_for_status()
        return cast(Dict[str, Any], result.json())
    except Exception:
        return None


def search_musicbrainz_recording_cached(
    *,
    title: str,
    artist: Optional[str] = None,
    duration_sec: Optional[float] = None,
    limit: int = 5,
    cache_path: str,
) -> Optional[Dict[str, Any]]:
    key = f"mb:{artist or ''}|{title}|{int(duration_sec or 0)}|{limit}"
    cache = _load_cache(cache_path)
    hit = cache.get(key)
    if isinstance(hit, dict):
        return hit
    res = search_musicbrainz_recording(
        title=title, artist=artist, duration_sec=duration_sec, limit=limit
    )
    if isinstance(res, dict):
        try:
            cache[key] = res
            _save_cache(cache_path, cache)
        except Exception:
            pass
    return res


def get_acousticbrainz_lowlevel_by_mbid_cached(
    recording_mbid: str, cache_path: str
) -> Optional[Dict[str, Any]]:
    key = f"ab:{recording_mbid}"
    cache = _load_cache(cache_path)
    hit = cache.get(key)
    if isinstance(hit, dict):
        return hit
    res = get_acousticbrainz_lowlevel_by_mbid(recording_mbid)
    if isinstance(res, dict):
        try:
            cache[key] = res
            _save_cache(cache_path, cache)
        except Exception:
            pass
    return res


def extract_bpm_from_acousticbrainz(payload: Dict[str, Any]) -> Optional[float]:
    """Extract a reasonable BPM from the AB low-level JSON.

    Prefers histogram first peak bpm when available, else rhythm.bpm.
    """
    try:
        recs = payload.get("recordings") or payload.get("results") or []
        if isinstance(recs, list) and recs:
            entry = recs[0]
            low = entry.get("lowlevel") or entry.get("analysis") or entry
            # Newer AB payloads may embed under lowlevel -> rhythm
            rhyth = low.get("rhythm", {}) if isinstance(low, dict) else {}
            cand = None
            for k in (
                "bpm_histogram_first_peak_bpm",
                "first_peak_bpm",
                "bpm",
            ):
                v = rhyth.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    cand = float(v)
                    break
            return cand
    except Exception:
        return None
    return None
