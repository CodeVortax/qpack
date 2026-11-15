"""Image deskewing and region cropping utilities using PIL and NumPy."""

from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Photos of exam pages are usually taken with the phone held sideways.  Pixels
# alone cannot tell "turn left" from "turn right", so when a page is detected as
# sideways we first try a quarter turn counter-clockwise (= 270 degrees clockwise);
# looks_upside_down() then adds a half turn when the page came out upside down.
# `qpack scan --rotate 90|180|270` overrides it for a particular photo.
SIDEWAYS_DEFAULT_CW = 270

# A page counts as sideways only when the vertical line structure is clearly
# stronger.  On the sample photos upright pages scored a rows/cols ratio >= 3.2
# and sideways pages <= 0.41, so 0.6 leaves a wide safety margin.
SIDEWAYS_RATIO = 0.6


def _ink_map(image: Image.Image, long_side: int = 1000) -> np.ndarray:
    """Binary map of pixels darker than their surroundings (text, rules, drawings)."""
    gray = ImageOps.grayscale(image)
    scale = min(1.0, long_side / max(gray.size))
    if scale < 1.0:
        gray = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.BILINEAR,
        )
    arr = np.asarray(gray, dtype=np.float32)
    local_bg = np.asarray(gray.filter(ImageFilter.GaussianBlur(12)), dtype=np.float32)
    ink = (local_bg - arr) > 18
    h, w = ink.shape
    # Ignore the outer 5% where table edges, shadows and fingers live
    return ink[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)].astype(np.float32)


def _line_structure(profile: np.ndarray, window: int = 31) -> float:
    """How strongly a 1-D ink profile alternates like lines of text (0 = not at all)."""
    kernel = np.ones(window, dtype=np.float32) / window
    trend = np.convolve(profile, kernel, mode="same")
    mean = float(profile.mean()) + 1e-6
    return float((profile - trend).std()) / mean


def detect_text_axis(image: Image.Image) -> str:
    """Return 'horizontal' if text lines run left-right, or 'vertical' if the page is sideways.

    Text lines show up as strong row-by-row alternation of ink in an upright
    photo, and as strong column-by-column alternation in a sideways one.
    Ties and blank images count as 'horizontal' so they are never rotated.
    """
    ink = _ink_map(image)
    rows = _line_structure(ink.sum(axis=1))
    cols = _line_structure(ink.sum(axis=0))
    if cols > 0 and rows < SIDEWAYS_RATIO * cols:
        return "vertical"
    return "horizontal"


def _small_rgb(image: Image.Image, width: int = 700) -> np.ndarray:
    im = image.convert("RGB")
    return np.asarray(im.resize((width, max(1, int(im.height * width / im.width)))), dtype=np.float32)


def _badge_side(image: Image.Image) -> Optional[float]:
    """Mean x position (0..1) of compact, strongly coloured blobs (question number badges)."""
    a = _small_rgb(image) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    mask = ((mx - mn) / (mx + 1e-6) > 0.45) & (mx > 0.35)
    # opening removes thin coloured rules, keeping only blob-shaped badges
    opened = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    ys, xs = np.nonzero(np.asarray(opened) > 0)
    if len(xs) < 30:
        return None
    return float(xs.mean() / mask.shape[1])


def _edge_alignment(image: Image.Image) -> Optional[Tuple[float, float]]:
    """(spread of right line ends, spread of left line ends) over the text lines of a page."""
    g = ImageOps.grayscale(image.resize((700, max(1, int(image.height * 700 / image.width)))))
    arr = np.asarray(g, dtype=np.float32)
    bg = np.asarray(g.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32)
    ink = (bg - arr) > 16
    h, w = ink.shape
    ink = ink[:, int(w * 0.06):int(w * 0.94)]
    row_has_ink = ink.sum(axis=1) > 3
    rights: List[int] = []
    lefts: List[int] = []
    y = 0
    while y < h:
        if not row_has_ink[y]:
            y += 1
            continue
        y2 = y
        while y2 < h and row_has_ink[y2]:
            y2 += 1
        if y2 - y >= 6:
            cols = np.nonzero(ink[y:y2].sum(axis=0) > 1)[0]
            if len(cols) > 40:
                rights.append(int(cols.max()))
                lefts.append(int(cols.min()))
        y = y2
    if len(rights) < 6:
        return None
    return float(np.std(rights)), float(np.std(lefts))


def looks_upside_down(image: Image.Image) -> bool:
    """True when an Arabic (right-to-left) page is turned 180 degrees.

    Two cues vote: the text lines end flush at the RIGHT margin (their left ends
    are ragged), and the coloured question badges sit on the RIGHT side. If
    neither cue is clear the page is left alone.
    """
    vote = 0.0
    align = _edge_alignment(image)
    if align:
        right_spread, left_spread = align
        lo, hi = sorted((right_spread, left_spread))
        if hi > 1.8 * max(lo, 1.0):
            vote += 2.0 if right_spread > left_spread else -2.0
    badge_x = _badge_side(image)
    if badge_x is not None and abs(badge_x - 0.5) > 0.25:
        vote += 1.0 if badge_x < 0.5 else -1.0
    return vote > 0


def rotate_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    """Rotate by a multiple of 90 degrees clockwise, keeping every pixel."""
    degrees %= 360
    if degrees == 0:
        return image
    return image.rotate(-degrees, expand=True)


def orient_image(image: Image.Image, rotate: Optional[int] = None) -> Tuple[Image.Image, float]:
    """Make the page's text run left-right.

    Applies EXIF orientation first.  ``rotate`` is a clockwise multiple of 90;
    when it is None the page is rotated only if it is detected as sideways.
    Returns (image, applied_angle) where the angle uses PIL's convention
    (positive = counter-clockwise), normalised to (-180, 180].
    """
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if rotate is None:
        cw = SIDEWAYS_DEFAULT_CW if detect_text_axis(image) == "vertical" else 0
        # A sideways photo may need either quarter-turn, and any photo may be upside
        # down, so check which way up the page is once the text runs left to right.
        probe = rotate_clockwise(image, cw)
        tilt = estimate_skew_angle(probe)        # tilted lines would blur the line-end cue
        if abs(tilt) >= 0.2:
            probe = probe.rotate(-tilt, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
        if looks_upside_down(probe):
            cw = (cw + 180) % 360
    else:
        cw = int(rotate) % 360

    if cw:
        image = rotate_clockwise(image, cw)

    pil_angle = float((-cw) % 360)
    if pil_angle > 180.0:
        pil_angle -= 360.0
    return image, pil_angle


def estimate_skew_angle(image: Image.Image, max_angle: float = 15.0, step: float = 0.5) -> float:
    """Estimate the skew angle (in degrees) of a page using projection profile variance.

    A positive angle indicates clockwise rotation needed to deskew.
    """
    # Convert to grayscale
    gray = ImageOps.grayscale(image)

    # Downscale for rapid angle detection (max dimension 800px)
    w, h = gray.size
    scale = min(1.0, 800.0 / max(w, h))
    if scale < 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        gray_small = gray.resize((new_w, new_h), Image.BILINEAR)
    else:
        gray_small = gray

    # Enhance horizontal text line edges
    edges = gray_small.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float32)

    # Threshold weak noise / watermarks
    thresh = np.percentile(arr, 75)
    arr = np.where(arr > thresh, 1.0, 0.0)

    best_angle = 0.0
    max_variance = -1.0

    # Coarse search from -max_angle to +max_angle
    angles = np.arange(-max_angle, max_angle + step, step)
    edge_img = Image.fromarray((arr * 255).astype(np.uint8))

    for angle in angles:
        # Rotate image with bilinear interpolation
        rotated = edge_img.rotate(float(angle), resample=Image.BILINEAR, expand=False, fillcolor=0)
        rot_arr = np.asarray(rotated, dtype=np.float32)
        # Horizontal projection profile (sum of edge pixels across each row)
        proj = np.sum(rot_arr, axis=1)
        # When text lines are horizontal, the variance of row sums is maximized
        var = float(np.var(proj))
        if var > max_variance:
            max_variance = var
            best_angle = float(angle)

    # Fine search around best coarse angle
    fine_step = 0.1
    fine_angles = np.arange(best_angle - step, best_angle + step + fine_step, fine_step)
    for angle in fine_angles:
        rotated = edge_img.rotate(float(angle), resample=Image.BILINEAR, expand=False, fillcolor=0)
        rot_arr = np.asarray(rotated, dtype=np.float32)
        proj = np.sum(rot_arr, axis=1)
        var = float(np.var(proj))
        if var > max_variance:
            max_variance = var
            best_angle = float(angle)

    # The angle that maximizes variance is the rotation angle that brings lines horizontal.
    # Therefore, the image's skew tilt angle is -best_angle.
    skew_angle = -best_angle
    return round(skew_angle, 2)


def deskew_image(
    image: Image.Image,
    max_angle: float = 15.0,
    rotate: Optional[int] = None,
) -> Tuple[Image.Image, float]:
    """Orient a page photo upright and correct its small tilt.

    ``rotate`` forces a clockwise quarter-turn count in degrees (0/90/180/270);
    None means auto-detect sideways photos (see orient_image).
    Returns:
        (deskewed_image, total_angle_applied) in PIL's convention
        (positive = counter-clockwise).
    """
    image, coarse_rotated = orient_image(image, rotate=rotate)

    angle = estimate_skew_angle(image, max_angle=max_angle)
    # If the detected skew angle is negligible (< 0.2 degrees), return image
    if abs(angle) < 0.2:
        return image, coarse_rotated

    # Rotate the image to upright position (counteracting skew angle)
    deskewed = image.rotate(-angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    return deskewed, coarse_rotated + angle


def crop_box(
    image: Image.Image,
    box: List[float],
    padding_pct: float = 0.02
) -> Image.Image:
    """Crop a bounding box region from an image.

    Supports:
        [ymin, xmin, ymax, xmax] in normalized (0.0 to 1.0 or 0 to 1000) or pixel coordinates.
    """
    img_w, img_h = image.size
    ymin, xmin, ymax, xmax = [float(v) for v in box]

    # Handle 0..1000 coordinate format (commonly returned by Gemini vision)
    if ymax > 1.0 or xmax > 1.0:
        if ymax <= 1000.0 and xmax <= 1000.0:
            ymin, xmin, ymax, xmax = ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0
        else:
            # Already absolute pixel coordinates
            left = max(0, int(xmin))
            top = max(0, int(ymin))
            right = min(img_w, int(xmax))
            bottom = min(img_h, int(ymax))
            if right <= left or bottom <= top:
                return image.copy()
            return image.crop((left, top, right, bottom))

    # Add margin padding
    pad_h = (ymax - ymin) * padding_pct
    pad_w = (xmax - xmin) * padding_pct
    ymin = max(0.0, ymin - pad_h)
    xmin = max(0.0, xmin - pad_w)
    ymax = min(1.0, ymax + pad_h)
    xmax = min(1.0, xmax + pad_w)

    left = int(xmin * img_w)
    top = int(ymin * img_h)
    right = int(xmax * img_w)
    bottom = int(ymax * img_h)

    # Safety checks
    if right <= left:
        right = min(img_w, left + 10)
    if bottom <= top:
        bottom = min(img_h, top + 10)

    return image.crop((left, top, right, bottom))
