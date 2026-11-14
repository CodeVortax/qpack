"""Image deskewing and region cropping utilities using PIL and NumPy."""

from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter


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


def deskew_image(image: Image.Image, max_angle: float = 15.0) -> Tuple[Image.Image, float]:
    """Deskew an image by detecting and correcting its orientation tilt.
    
    Also handles EXIF orientation and auto-orients landscape documents to portrait.
    Returns:
        (deskewed_image, angle_corrected)
    """
    # 1. Apply EXIF orientation if present
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # 2. Check if a document photo is sideways (landscape where width > height * 1.15)
    # Most exam pages are portrait; if text lines run vertically, orient to portrait
    w, h = image.size
    coarse_rotated = 0.0
    if w > h * 1.15:
        # Check horizontal vs vertical projection variance
        gray = ImageOps.grayscale(image.resize((400, int(400 * h / w))))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(edges, dtype=np.float32)
        var_h = float(np.var(np.sum(arr, axis=1)))
        var_v = float(np.var(np.sum(arr, axis=0)))
        if var_v > var_h:
            # Lines are vertical; rotating by 90 degrees makes them horizontal
            image = image.rotate(-90, expand=True)
            coarse_rotated = -90.0

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
