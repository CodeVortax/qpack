"""The local reader on synthetic pages (no AI, no real photos)."""

from PIL import Image, ImageDraw

from qpack import vision
from qpack.deskew import looks_upside_down, rotate_clockwise


def bubble_page(rows=3):
    img = Image.new("RGB", (1400, 1900), (235, 235, 232))
    d = ImageDraw.Draw(img)
    for q in range(rows):
        top = 150 + q * 420
        d.ellipse([1240, top, 1275, top + 35], fill=(170, 30, 40))          # number badge
        d.rectangle([700, top + 60, 1180, top + 72], fill=(30, 30, 30))      # question text
        for i, (x, y) in enumerate([(1160, 150), (700, 150), (1160, 215), (700, 215)]):
            cx, cy = x, top + y
            d.ellipse([cx - 17, cy - 17, cx + 17, cy + 17], outline=(20, 20, 20), width=3)
            d.rectangle([cx - 6, cy - 8, cx + 6, cy + 8], fill=(20, 20, 20))  # the letter
            d.rectangle([cx - 200, cy - 8, cx - 40, cy + 8], fill=(20, 20, 20))  # option text
    return img


def test_questions_and_four_options_found_in_reading_order():
    qs, meta = vision.read_page(bubble_page())
    assert meta["split_by"] == "badges" and len(qs) == 3
    for q in qs:
        assert list(q["option_boxes"]) == ["أ", "ب", "ج", "د"]
        a, b, c, d = (q["option_boxes"][k] for k in "أبجد")
        assert a[3] > b[3] and c[3] > d[3] and a[0] < c[0]     # right-to-left, top-to-bottom
        assert q["correct_choice"] is None and q["student_choice"] is None


def test_blank_page_finds_nothing_and_does_not_crash():
    qs, meta = vision.read_page(Image.new("RGB", (800, 1000), "white"))
    assert qs == [] and meta["split_by"] == "none"


def test_framed_page_is_split_by_its_frames():
    img = Image.new("RGB", (1400, 1500), (235, 235, 232))
    d = ImageDraw.Draw(img)
    for y in (100, 400, 700, 1000, 1300):
        d.rectangle([40, y, 1360, y + 4], fill=(200, 70, 40))
    qs, meta = vision.read_page(img)
    assert meta["split_by"] == "frames" and len(qs) == 4


def test_upside_down_check_uses_right_aligned_text():
    img = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(img)
    for i, y in enumerate(range(100, 700, 40)):
        d.rectangle([560 - (i * 37 % 200) - 150, y, 560, y + 10], fill=(0, 0, 0))   # flush right
    assert looks_upside_down(img) is False
    assert looks_upside_down(rotate_clockwise(img, 180)) is True
