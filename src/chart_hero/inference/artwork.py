from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import requests
import io


def _download_image(url: str, timeout: float = 10.0) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _nice_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try a few common fonts; fall back to default
    for name in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_art(
    out_dir: Path,
    title: str,
    artist: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
) -> tuple[Path | None, Path | None]:
    """
    Create album.png (square) and background.jpg (16:9 blurred) in out_dir.
    Returns (album_path, background_path), either may be None if generation fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    album_path = out_dir / "album.png"
    bg_path = out_dir / "background.jpg"

    base_img: Optional[Image.Image] = None
    if thumbnail_url:
        base_img = _download_image(thumbnail_url)

    try:
        if base_img:
            square = _center_crop_square(base_img).resize((512, 512), Image.BICUBIC)
            square.save(album_path)

            bg = base_img.copy()
            bg = bg.resize((1920, 1080), Image.BICUBIC).filter(
                ImageFilter.GaussianBlur(radius=16)
            )
            # Dim
            overlay = Image.new("RGBA", bg.size, (0, 0, 0, 120))
            bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
            bg.save(bg_path, quality=90)
            return album_path, bg_path

        # Fallback: simple gradient with text
        square = Image.new("RGB", (512, 512), color=(24, 26, 32))
        draw = ImageDraw.Draw(square)
        # Simple diagonal gradient
        for y in range(512):
            g = int(24 + (y / 511.0) * 80)
            draw.line([(0, y), (512, y)], fill=(g, 28, 64))
        # Text
        title_font = _nice_font(36)
        artist_font = _nice_font(20)
        tw = draw.textlength(title, font=title_font)
        draw.text(((512 - tw) / 2, 200), title, fill=(240, 240, 240), font=title_font)
        if artist:
            aw = draw.textlength(artist, font=artist_font)
            draw.text(
                ((512 - aw) / 2, 250), artist, fill=(200, 200, 200), font=artist_font
            )
        square.save(album_path)

        bg = square.resize((1920, 1080), Image.BICUBIC).filter(
            ImageFilter.GaussianBlur(radius=12)
        )
        bg.save(bg_path, quality=90)
        return album_path, bg_path
    except Exception:
        return None, None
