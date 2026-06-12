#!/usr/bin/env python3
# scripts/make_og_image.py
"""
Generate frontend/public/og-image.png at the canonical Open Graph dimensions
(1200x630, 1.91:1). Composes:

    - A near-black background with a subtle dark-green radial highlight that
      matches the site's home-page treatment.
    - The AllIn logo, centered, scaled to a sensible height.
    - A short tagline beneath the logo (the page title's tagline).
    - The site's URL in the bottom-right corner as a quiet credit/CTA.

Re-run any time the tagline or logo changes:
    python scripts/make_og_image.py

Requires Pillow (`pip install Pillow`) -- not in the project's runtime
requirements; only the developer regenerating this asset needs it.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH  = os.path.join(REPO_ROOT, 'frontend', 'src', 'pages',
                          'AllIn_Black_Centered.png')
OUT_PATH   = os.path.join(REPO_ROOT, 'frontend', 'public', 'og-image.png')

W, H       = 1200, 630
BG         = (10, 10, 10)              # #0a0a0a (matches site)
HIGHLIGHT  = (12, 42, 31)              # #0c2a1f -- the home page's radial tint
TEXT_FG    = (235, 235, 235)
TEXT_MUTED = (140, 140, 140)
URL_MUTED  = (110, 110, 110)

TAGLINE    = "Heads-up Texas Hold'em poker AI"
URL        = "allin.jianrontan.com"


def _radial_background():
    """Vertical radial highlight in dark green, fading to #0a0a0a. Cheap PIL
    approximation: a single Gaussian-blurred ellipse over a flat black canvas."""
    bg = Image.new('RGB', (W, H), BG)
    overlay = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(overlay)
    # Ellipse centered slightly above middle; the blur softens it to a glow.
    cx, cy = W // 2, int(H * 0.32)
    rx, ry = int(W * 0.55), int(H * 0.45)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=HIGHLIGHT)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=120))
    return Image.blend(bg, overlay, alpha=0.55)


def _load_font(size, *, bold=False):
    """Best-effort font load. Falls back to PIL's default if no system font."""
    candidates = (
        # Windows
        'C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',
        # macOS
        '/System/Library/Fonts/Helvetica.ttc',
        # Linux (Debian/Ubuntu w/ ttf-dejavu)
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
            else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _center_text(draw, text, font, *, y, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((W - text_w) // 2, y), text, font=font, fill=fill)


def main():
    img = _radial_background()
    draw = ImageDraw.Draw(img)

    # --- Logo ---
    logo = Image.open(LOGO_PATH).convert('RGBA')
    # Scale the logo to a comfortable 220 px height (the source is 894x894).
    logo_h = 220
    logo_w = int(logo.width * (logo_h / logo.height))
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
    logo_x = (W - logo_w) // 2
    logo_y = int(H * 0.18)
    img.paste(logo, (logo_x, logo_y), logo)

    # --- Tagline ---
    tagline_font = _load_font(40, bold=True)
    _center_text(
        draw, TAGLINE, tagline_font,
        y=logo_y + logo_h + 36,
        fill=TEXT_FG,
    )

    # --- Bottom-right URL ---
    url_font = _load_font(22)
    url_bbox = draw.textbbox((0, 0), URL, font=url_font)
    url_w = url_bbox[2] - url_bbox[0]
    draw.text(
        (W - url_w - 44, H - 56),
        URL,
        font=url_font,
        fill=URL_MUTED,
    )

    # --- Hairline accent in the corner opposite the URL ---
    # A short horizontal line beneath the tagline, lightly tinted to match
    # the radial highlight. Adds visual structure without adding noise.
    line_y = logo_y + logo_h + 36 + tagline_font.getbbox(TAGLINE)[3] + 18
    line_w = 80
    draw.line(
        [(W // 2 - line_w // 2, line_y), (W // 2 + line_w // 2, line_y)],
        fill=TEXT_MUTED,
        width=1,
    )

    img.save(OUT_PATH, 'PNG', optimize=True)
    print(f"Wrote {OUT_PATH}  ({W}x{H})")


if __name__ == '__main__':
    main()
