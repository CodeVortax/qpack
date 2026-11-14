"""Unit tests for image deskewing and cropping."""

import numpy as np
from PIL import Image, ImageDraw
from qpack.deskew import estimate_skew_angle, deskew_image, crop_box


def create_synthetic_skewed_image(angle_deg: float = 5.0) -> Image.Image:
    """Create a synthetic image with horizontal text-like lines rotated by angle_deg."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw several horizontal black text bars
    for y in range(100, 700, 50):
        draw.rectangle([50, y, 550, y + 12], fill=(0, 0, 0))

    # Rotate image by angle_deg
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return rotated


def test_estimate_skew_angle():
    target_angle = 4.0
    skewed = create_synthetic_skewed_image(angle_deg=target_angle)
    detected_angle = estimate_skew_angle(skewed, max_angle=10.0, step=0.5)
    # The detected angle should be close to target_angle
    assert abs(detected_angle - target_angle) <= 1.0


def test_deskew_image_straightens():
    skewed = create_synthetic_skewed_image(angle_deg=3.0)
    deskewed, angle = deskew_image(skewed, max_angle=10.0)
    assert abs(angle - 3.0) <= 1.0
    assert deskewed.size[0] >= skewed.size[0]


def test_crop_box_normalized_0_1():
    img = Image.new("RGB", (500, 400), color=(100, 150, 200))
    # Box: ymin=0.1, xmin=0.2, ymax=0.5, xmax=0.8
    cropped = crop_box(img, [0.1, 0.2, 0.5, 0.8], padding_pct=0.0)
    assert abs(cropped.size[0] - 300) <= 2  # (0.8 - 0.2) * 500 = 300
    assert abs(cropped.size[1] - 160) <= 2  # (0.5 - 0.1) * 400 = 160


def test_crop_box_normalized_0_1000():
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    # Coordinates in 0..1000 range
    cropped = crop_box(img, [100, 200, 600, 700], padding_pct=0.0)
    assert abs(cropped.size[0] - 500) <= 2
    assert abs(cropped.size[1] - 500) <= 2
