#!/usr/bin/env python3
"""
Quick MIDI repair utility for Clone Hero/Rock Band charts.

Scans one or more song roots for `notes.mid`, attempts to load each with mido,
and when mido raises data-byte range errors (common in community charts), repairs
the file by clamping channel-message data bytes to 0..127 inside each MTrk
chunk. Optionally writes a repaired copy or overwrites in place with backup.

Usage examples:
  python scripts/repair_bad_midis.py --songs-root CloneHero/Songs --dry-run
  python scripts/repair_bad_midis.py --songs-root CloneHero/Songs --apply --backup
  python scripts/repair_bad_midis.py --songs-root ./somepack --apply --out-suffix .fixed

Notes:
- Requires `mido`. It uses `clip=True` when available to be more tolerant when
  validating after repair.
- The repair focuses on clamping invalid data bytes in channel messages; it does
  not attempt deeper structural fixes. In practice this resolves most
  "data byte must be in range 0..127" errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
from typing import Iterable, Tuple


def _read_varlen(b: bytes, i: int) -> Tuple[int, int]:
    val = 0
    n = 0
    while i + n < len(b):
        c = b[i + n]
        val = (val << 7) | (c & 0x7F)
        n += 1
        if (c & 0x80) == 0:
            break
    return val, n


def _clamp_track_data(track: bytes) -> bytes:
    out = bytearray()
    i = 0
    running_status: int | None = None
    L = len(track)

    while i < L:
        # delta-time (varlen)
        dt, n = _read_varlen(track, i)
        out.extend(track[i : i + n])
        i += n
        if i >= L:
            break

        status = track[i]
        if status & 0x80:
            running_status = status
            out.append(status)
            i += 1
        # else: running status, do not consume; use previous running_status

        if running_status is None:
            # can't proceed safely; copy remainder
            out.extend(track[i:])
            break

        st = running_status
        if st == 0xFF:  # Meta event: 0xFF, type (1), length (varlen), data
            if i >= L:
                break
            meta_type = track[i]
            out.append(meta_type)
            i += 1
            length, n = _read_varlen(track, i)
            out.extend(track[i : i + n])
            i += n
            out.extend(track[i : i + length])
            i += length
            running_status = (
                None  # meta cancels running status per track parsing practice
            )
            continue
        if st in (0xF0, 0xF7):  # SysEx
            length, n = _read_varlen(track, i)
            out.extend(track[i : i + n])
            i += n
            out.extend(track[i : i + length])
            i += length
            running_status = None
            continue

        msg_type = st & 0xF0
        data_len = {0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2, 0xC0: 1, 0xD0: 1, 0xE0: 2}.get(
            msg_type, 0
        )
        # Clamp data bytes. If insufficient bytes remain, copy remainder and exit
        if i + data_len > L:
            out.extend(track[i:])
            break
        for k in range(data_len):
            out.append(track[i + k] & 0x7F)
        i += data_len

    return bytes(out)


def clamp_invalid_data_bytes(midi_bytes: bytes) -> bytes | None:
    # Minimal chunk reader: MThd + MTrk*
    i = 0
    if len(midi_bytes) < 14 or midi_bytes[:4] != b"MThd":
        return None
    out = bytearray()
    out.extend(midi_bytes[:8])  # MThd + length
    i = 8
    header_len = int.from_bytes(midi_bytes[4:8], "big")
    out.extend(midi_bytes[i : i + header_len])
    i += header_len

    while i + 8 <= len(midi_bytes):
        chunk_type = midi_bytes[i : i + 4]
        length = int.from_bytes(midi_bytes[i + 4 : i + 8], "big")
        i += 8
        data = midi_bytes[i : i + length]
        i += length
        if chunk_type == b"MTrk":
            fixed = _clamp_track_data(data)
            # Preserve original length by truncating/padding if necessary
            if len(fixed) != length:
                # Attempt to preserve length to keep offsets stable
                if len(fixed) > length:
                    fixed = fixed[:length]
                else:
                    fixed = fixed + bytes(length - len(fixed))
            out.extend(b"MTrk")
            out.extend(length.to_bytes(4, "big"))
            out.extend(fixed)
        else:
            out.extend(chunk_type)
            out.extend(length.to_bytes(4, "big"))
            out.extend(data)

    return bytes(out)


def find_midis(roots: Iterable[Path]) -> Iterable[Path]:
    for r in roots:
        if not r.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dp = Path(dirpath)
            if "notes.mid" in filenames:
                yield dp / "notes.mid"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan and repair invalid notes.mid files")
    ap.add_argument("--songs-root", type=str, nargs="+", required=True)
    ap.add_argument("--apply", action="store_true", help="Write repairs to disk")
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Backup original to notes.mid.bak when overwriting",
    )
    ap.add_argument(
        "--out-suffix",
        type=str,
        default=".repaired",
        help="Suffix for repaired copies when not overwriting (e.g., notes.repaired.mid)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite notes.mid in place (implies --apply)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.songs_root]
    apply = args.apply or args.overwrite

    try:
        import mido  # type: ignore
    except Exception:
        mido = None  # type: ignore

    total = 0
    repaired = 0
    failed = 0

    for midi_path in find_midis(roots):
        total += 1
        err = None
        ok = False
        if mido is not None:
            try:
                try:
                    mido.MidiFile(midi_path, clip=True)  # type: ignore[call-arg]
                except TypeError:
                    mido.MidiFile(midi_path)
                ok = True
            except Exception as e:
                err = str(e)
        else:
            # Without mido, attempt a blind repair pass
            err = "mido not available; attempting blind repair"

        if ok:
            # Already parsable; optionally normalize by re-saving via mido
            continue

        # Attempt byte-level clamp repair
        try:
            raw = midi_path.read_bytes()
        except Exception as e:
            print(f"ERROR reading {midi_path}: {e}")
            failed += 1
            continue

        fixed = clamp_invalid_data_bytes(raw)
        if not fixed:
            print(f"SKIP {midi_path}: unsupported/corrupt structure (no MThd)")
            failed += 1
            continue

        reparsed_ok = False
        if mido is not None:
            try:
                from io import BytesIO

                bio = BytesIO(fixed)
                try:
                    mido.MidiFile(file=bio, clip=True)  # type: ignore[call-arg]
                except TypeError:
                    mido.MidiFile(file=bio)
                reparsed_ok = True
            except Exception as e:
                print(f"REPAIR FAILED {midi_path}: still invalid after clamp: {e}")
        else:
            # Assume repaired for write if applying without mido
            reparsed_ok = True

        if not reparsed_ok:
            failed += 1
            continue

        if not apply:
            print(f"WOULD REPAIR {midi_path}: {err}")
            repaired += 1
            continue

        try:
            if args.overwrite:
                if args.backup:
                    bak = midi_path.with_suffix(".mid.bak")
                    if not bak.exists():
                        bak.write_bytes(raw)
                midi_path.write_bytes(fixed)
            else:
                outp = midi_path.with_name(
                    midi_path.stem + args.out_suffix + midi_path.suffix
                )
                outp.write_bytes(fixed)
            print(f"REPAIRED {midi_path}")
            repaired += 1
        except Exception as e:
            print(f"ERROR writing repair for {midi_path}: {e}")
            failed += 1

    print(f"Done. Total MIDIs: {total}, repaired: {repaired}, failed: {failed}")


if __name__ == "__main__":
    import os

    main()
