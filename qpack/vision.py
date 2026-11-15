"""Local page reader: finds questions, answer options and student marks with plain
image processing. No AI, no network, no API key. Needs only NumPy and Pillow.

What it can do
    * split an upright page photo into questions (number badges, red frames or
      separator lines mark where each one starts);
    * find the four options of each question (lettered bubbles, or the row of
      option texts inside a framed question) and label them أ ب ج د in right-to-left
      reading order;
    * box the question and every option (pictures included) so they can be cropped;
    * spot student ink on the options (circles, ticks, crosses and so on) as hints.

What it cannot do
    * know the right answer (that needs physics knowledge: paste an answer key or tap it);
    * read the Arabic text. Crops carry the content; text stays empty.

All boxes use the normalised [ymin, xmin, ymax, xmax] format the rest of qpack uses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

OPTION_KEYS = ["أ", "ب", "ج", "د"]
WORK_WIDTH = 1400          # every measurement below is relative to a 1400 px wide page


# ---------------------------------------------------------------- connected components
def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected labelling of a boolean mask (run-length union-find)."""
    h, w = mask.shape
    runs: List[Tuple[int, int, int]] = []
    first = np.zeros(h + 1, dtype=np.int64)
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    for y in range(h):
        starts = np.nonzero(edges[y] == 1)[0]
        ends = np.nonzero(edges[y] == -1)[0]
        for a, b in zip(starts, ends):
            runs.append((y, int(a), int(b)))
        first[y + 1] = len(runs)
    parent = list(range(len(runs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(1, h):
        i, iend = first[y - 1], first[y]
        j, jend = first[y], first[y + 1]
        while i < iend and j < jend:
            a0, a1 = runs[i][1], runs[i][2]
            b0, b1 = runs[j][1], runs[j][2]
            if a0 <= b1 and b0 <= a1:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
            if a1 < b1:
                i += 1
            else:
                j += 1
    ids: Dict[int, int] = {}
    labels = np.zeros((h, w), dtype=np.int32)
    for k, (y, a, b) in enumerate(runs):
        labels[y, a:b] = ids.setdefault(find(k), len(ids) + 1)
    return labels, len(ids)


def components(mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    labels, n = label_components(mask)
    if n == 0:
        return labels, []
    ys, xs = np.nonzero(labels)
    ls = labels[ys, xs]
    order = np.argsort(ls, kind="stable")
    ys, xs, ls = ys[order], xs[order], ls[order]
    cuts = np.nonzero(np.diff(ls))[0] + 1
    out = []
    for yy, xx, ll in zip(np.split(ys, cuts), np.split(xs, cuts), np.split(ls, cuts)):
        out.append({"id": int(ll[0]), "x0": int(xx.min()), "x1": int(xx.max()) + 1,
                    "y0": int(yy.min()), "y1": int(yy.max()) + 1, "area": int(len(yy))})
    return labels, out


# ---------------------------------------------------------------- page preparation
class Page:
    """A page resized to WORK_WIDTH with its ink and colour masks."""

    def __init__(self, image: Image.Image):
        rgb = image.convert("RGB")
        self.scale = WORK_WIDTH / rgb.width
        self.rgb = rgb.resize((WORK_WIDTH, max(1, int(round(rgb.height * self.scale)))), Image.BILINEAR)
        self.w, self.h = self.rgb.size
        gray = ImageOps.grayscale(self.rgb)
        a = np.asarray(gray, dtype=np.float32)
        bg = np.asarray(gray.filter(ImageFilter.GaussianBlur(14)), dtype=np.float32)
        self.ink = (bg - a) > 22
        self.dark = (bg - a) > 50          # strong ink only: text, not watermarks
        hsv = np.asarray(self.rgb.convert("HSV")).astype(np.int32)
        sat, val = hsv[..., 1], hsv[..., 2]
        sat_bg = np.asarray(Image.fromarray(sat.astype(np.uint8)).filter(ImageFilter.GaussianBlur(25)), dtype=np.int32)
        self.colour = ((sat - sat_bg) > 45) & (val > 60)
        self._ink_cc: Optional[Tuple[np.ndarray, List[Dict[str, int]]]] = None
        self._col_cc: Optional[Tuple[np.ndarray, List[Dict[str, int]]]] = None

    def ink_components(self):
        if self._ink_cc is None:
            self._ink_cc = components(self.ink)
        return self._ink_cc

    def colour_components(self):
        if self._col_cc is None:
            self._col_cc = components(self.colour)
        return self._col_cc

    def norm(self, x0: float, y0: float, x1: float, y1: float) -> List[float]:
        x0, x1 = sorted((max(0.0, x0), min(float(self.w), x1)))
        y0, y1 = sorted((max(0.0, y0), min(float(self.h), y1)))
        return [round(y0 / self.h, 4), round(x0 / self.w, 4), round(y1 / self.h, 4), round(x1 / self.w, 4)]


# ---------------------------------------------------------------- question segmentation
def find_badges(page: Page) -> List[Dict[str, int]]:
    """Coloured number badges down the right-hand margin."""
    _, comps = page.colour_components()
    found = []
    for c in comps:
        w, h = c["x1"] - c["x0"], c["y1"] - c["y0"]
        if 22 <= w <= 80 and 22 <= h <= 80 and c["area"] > 300 and 0.6 < w / h < 1.6 and c["x0"] > 0.6 * page.w:
            found.append(c)
    found.sort(key=lambda c: c["y0"])
    # drop duplicates stacked on top of each other
    keep: List[Dict[str, int]] = []
    for c in found:
        if keep and c["y0"] - keep[-1]["y0"] < 60:
            continue
        keep.append(c)
    return keep


def find_frame_edges(page: Page) -> List[float]:
    """y positions of the coloured horizontal lines that frame questions.

    A frame edge runs across the whole page, so it shows up in most vertical
    strips of the page at about the same height (this tolerates a slightly tilted
    photo). Coloured pieces of a figure, like a red rod, only cover a few strips.
    A frame's bottom edge and the next frame's top edge are merged."""
    strips = 8
    sw = page.w // strips
    votes = np.zeros(page.h, dtype=np.int32)
    for k in range(strips):
        on = page.colour[:, k * sw:(k + 1) * sw].sum(axis=1) >= 0.55 * sw
        votes += (np.convolve(on.astype(np.int32), np.ones(13, dtype=np.int32), "same") > 0)
    ys = np.nonzero(votes >= 4)[0]
    groups: List[List[int]] = []
    for y in ys:
        if groups and y - groups[-1][-1] <= 45:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])
    return [float(np.mean(g)) for g in groups]


def split_questions(page: Page) -> Tuple[List[Tuple[int, int]], str]:
    """Return ([(y_top, y_bottom), ...], how) for the questions on the page."""
    badges = find_badges(page)
    edges = find_frame_edges(page)
    if len(badges) < 3 and len(edges) >= 4:
        heights = [b - a for a, b in zip(edges, edges[1:]) if b - a >= 90]
        if heights and edges[0] > 0.9 * float(np.median(heights)) * 0.6:
            # the first frame's top edge is faint (the header banner sits above it)
            edges = [max(0.0, edges[0] - float(np.median(heights)))] + edges
        spans = [(int(a), int(b)) for a, b in zip(edges, edges[1:]) if b - a >= 90]
        if len(spans) >= 3:
            return spans, "frames"
    if badges:
        tops = [b["y0"] for b in badges]
        spans = []
        for i, top in enumerate(tops):
            bottom = tops[i + 1] if i + 1 < len(tops) else page.h
            spans.append((top, bottom))
        return spans, "badges"
    return [], "none"


# ---------------------------------------------------------------- option bubbles
def _ring_score(ink: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Fraction of points on a circle of radius r around (cx, cy) that are inked."""
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    hits = 0
    for t in angles:
        best = False
        for dr in (-2, -1, 0, 1, 2):
            x = int(round(cx + (r + dr) * np.cos(t)))
            y = int(round(cy + (r + dr) * np.sin(t)))
            if 0 <= y < ink.shape[0] and 0 <= x < ink.shape[1] and ink[y, x]:
                best = True
                break
        hits += best
    return hits / len(angles)


def find_bubbles(page: Page) -> List[Dict[str, Any]]:
    """Printed circles that carry an option letter."""
    labels, comps = page.ink_components()
    out = []
    for c in comps:
        w, h = c["x1"] - c["x0"], c["y1"] - c["y0"]
        if not (18 <= w <= 52 and 18 <= h <= 52 and 0.75 <= w / h <= 1.3):
            continue
        if c["area"] / (w * h) > 0.55:
            continue
        cx, cy = (c["x0"] + c["x1"]) / 2, (c["y0"] + c["y1"]) / 2
        score = _ring_score(page.ink, cx, cy, min(w, h) / 2 - 2)
        if score >= 0.8:
            out.append({**c, "cx": cx, "cy": cy, "r": (w + h) / 4, "score": score})
    return out


def order_options(bubbles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Right-to-left, top-to-bottom reading order: rows first, then x descending."""
    if not bubbles:
        return []
    size = float(np.median([b["r"] for b in bubbles])) * 2
    rows: List[List[Dict[str, Any]]] = []
    for b in sorted(bubbles, key=lambda b: b["cy"]):
        if rows and abs(b["cy"] - np.mean([x["cy"] for x in rows[-1]])) < 0.9 * size:
            rows[-1].append(b)
        else:
            rows.append([b])
    ordered: List[Dict[str, Any]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: -b["cx"]))
    return ordered


def pick_four(bubbles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Choose the option bubbles of one question: they line up on the right edge of
    their option text, so prefer the strongest rings; keep at most four."""
    if len(bubbles) <= 4:
        return bubbles
    best = sorted(bubbles, key=lambda b: -b["score"])[:4]
    return sorted(best, key=lambda b: b["cy"])


def _text_left_edge(page: Page, b: Dict[str, Any], stop_x: float) -> float:
    """Walk left from a bubble along its text line until a wide empty gap (or stop_x)."""
    y0, y1 = int(max(0, b["cy"] - b["r"])), int(min(page.h, b["cy"] + b["r"]))
    band = page.ink[y0:y1, :].any(axis=0)
    gap_limit = int(0.045 * page.w)
    x = int(b["cx"] - b["r"]) - 2
    left, gap = x, 0
    while x > stop_x:
        if band[x]:
            left, gap = x, 0
        else:
            gap += 1
            if gap > gap_limit:
                break
        x -= 1
    return float(left)


def option_boxes_from_bubbles(page: Page, bubbles: List[Dict[str, Any]], span: Tuple[int, int],
                              left_limit: float) -> Dict[str, List[float]]:
    ordered = order_options(pick_four(bubbles))
    boxes: Dict[str, List[float]] = {}
    for key, b in zip(OPTION_KEYS, ordered):
        same_row = [o for o in ordered if o is not b and abs(o["cy"] - b["cy"]) < 1.3 * b["r"] * 2]
        left_neighbours = [o["cx"] + o["r"] for o in same_row if o["cx"] < b["cx"]]
        stop = (max(left_neighbours) + 4) if left_neighbours else left_limit
        x0 = _text_left_edge(page, b, stop) - 8
        x1 = b["cx"] + b["r"] + 8
        lone = (x1 - x0) < 4.2 * b["r"]          # a bare letter: the option is a picture above it
        if lone:
            row_x = sorted(o["cx"] for o in ordered if abs(o["cy"] - b["cy"]) < 1.3 * b["r"] * 2)
            nb = [abs(o - b["cx"]) for o in row_x if o != b["cx"]]
            half = (min(nb) / 2) if nb else 0.09 * page.w
            half = min(max(half, 3 * b["r"]), 0.16 * page.w)
            x0, x1 = b["cx"] - half, b["cx"] + half
        above = [o["cy"] for o in ordered if o["cy"] < b["cy"] - 1.3 * b["r"]]
        below = [o["cy"] for o in ordered if o["cy"] > b["cy"] + 1.3 * b["r"]]
        up = min((b["cy"] - max(above)) / 2, 2.2 * b["r"]) if above else 1.6 * b["r"]
        down = min((min(below) - b["cy"]) / 2, 2.2 * b["r"]) if below else 1.6 * b["r"]
        if lone:
            up = max(up, min(2.6 * half, b["cy"] - span[0] - 0.3 * (span[1] - span[0])))
            if above:
                up = min(up, b["cy"] - max(above) - 1.3 * b["r"])
        boxes[key] = page.norm(x0, max(b["cy"] - up, span[0]), x1, min(b["cy"] + down, span[1]))
    return boxes


# ---------------------------------------------------------------- framed questions (no bubbles)
def _bands(flags: np.ndarray, merge: int = 7, min_h: int = 13, profile: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    out: List[List[int]] = []
    for y, on in enumerate(flags):
        if on:
            if out and y - out[-1][1] <= merge:
                out[-1][1] = y
            else:
                out.append([y, y])
    bands = [(a, b + 1) for a, b in out]
    if profile is not None:                      # split lines that touch: cut at deep valleys
        split: List[Tuple[int, int]] = []
        sm = np.convolve(profile.astype(float), np.ones(5) / 5, "same")
        for a, b in bands:
            seg = sm[a:b]
            left = np.maximum.accumulate(seg)
            right = np.maximum.accumulate(seg[::-1])[::-1]
            valley = (seg < 0.3 * np.minimum(left, right)) & (np.minimum(left, right) >= 14)
            start = a
            i = 0
            while i < len(seg):
                if valley[i]:
                    j = i
                    while j < len(seg) and valley[j]:
                        j += 1
                    split.append((start, a + i))
                    start = a + j
                    i = j
                else:
                    i += 1
            split.append((start, b))
        bands = split
    bands = [(a, b) for a, b in bands if b - a >= min_h]
    if profile is not None and len(bands) >= 3:  # a band much taller than the rest is two lines stuck together
        med = float(np.median([b - a for a, b in bands]))
        sm = np.convolve(profile.astype(float), np.ones(5) / 5, "same")
        fixed: List[Tuple[int, int]] = []
        for a, b in bands:
            n = int(round((b - a) / med)) if (b - a) >= 1.7 * med and (b - a) >= 40 else 1
            start = a
            for k in range(1, n):
                target = a + (b - a) * k // n
                win = range(max(start + min_h, target - int(0.3 * med)), min(b - min_h, target + int(0.3 * med)) + 1)
                if not len(win):
                    continue
                cut = min(win, key=lambda y: sm[y])
                fixed.append((start, cut))
                start = cut
            fixed.append((start, b))
        bands = fixed
    return bands


def _clusters(cols: np.ndarray, gap: int, min_w: int = 10) -> List[Tuple[int, int]]:
    out: List[List[int]] = []
    for x, on in enumerate(cols):
        if on:
            if out and x - out[-1][1] <= gap:
                out[-1][1] = x
            else:
                out.append([x, x])
    return [(a, b + 1) for a, b in out if b + 1 - a >= min_w]


def _row_left_edge(page: Page, y0: int, y1: int, start_x: int, gap: int) -> int:
    """Walk left from start_x along rows y0..y1 until a wide empty gap."""
    cols = page.dark[y0:y1].any(axis=0)
    x, left, empty = start_x, start_x, 0
    while x > int(0.05 * page.w):
        if cols[x]:
            left, empty = x, 0
        else:
            empty += 1
            if empty > gap:
                break
        x -= 1
    return left


def frame_options(page: Page, span: Tuple[int, int]) -> Tuple[Dict[str, List[float]], str]:
    """Find the four options inside one framed question, the way a person would:

    1. The question text is at the top; the options are what comes after it.
    2. Options stacked as lines (or table rows): the last four text lines.
    3. Options in one row: the last line splits into four groups.
    4. Letters sitting under pictures: four small letters in a row, the picture is
       the column above each letter.
    """
    top, bottom = span[0] + 8, span[1] - 8
    W = page.w
    x_lo, x_hi = int(0.05 * W), int(0.955 * W)
    right0 = int(0.38 * W)
    sub = page.dark[top:bottom].copy()
    tall = np.convolve((sub.mean(axis=0) > 0.45).astype(np.int32), np.ones(7, dtype=np.int32), "same") > 0
    sub[:, tall] = False                                   # frame sides, the book's spine
    labels, comps = page.colour_components()
    for c in comps:                                        # long coloured lines: frame edges, table rules
        w, h = c["x1"] - c["x0"], c["y1"] - c["y0"]
        if w >= 0.2 * W and h <= 40 and c["area"] <= 12 * w and c["y1"] > top and c["y0"] < bottom:
            piece = (labels[c["y0"]:c["y1"], c["x0"]:c["x1"]] == c["id"]).astype(np.uint8) * 255
            thick = np.asarray(Image.fromarray(piece).filter(ImageFilter.MaxFilter(7))) > 0
            ys0, ys1 = max(c["y0"], top), min(c["y1"], bottom)
            sub[ys0 - top:ys1 - top, c["x0"]:c["x1"]] &= ~thick[ys0 - c["y0"]:ys1 - c["y0"]]
    page_dark = sub                                        # local copy, offset by `top`
    flags = sub[:, right0:x_hi].sum(axis=1) >= 5
    rs = sub[:, right0:x_hi].sum(axis=1)
    bands = [(a + top, b + top) for a, b in _bands(flags, profile=rs)]
    if not bands:
        return {}, "none"
    gap_row = int(0.06 * W)

    last = bands[-1]
    h = last[1] - last[0]
    mid0 = last[0] + int(0.3 * h)
    mid1 = last[0] + max(int(0.7 * h), int(0.3 * h) + 2)
    groups: List[Tuple[int, int]] = []
    for r0, r1 in ((mid0, mid1), (last[0], last[1])):        # centre rows first: they dodge hand-drawn rings
        row = sub[r0 - top:r1 - top, x_lo:x_hi].sum(axis=0) >= 1
        for gap in range(int(0.034 * W), int(0.016 * W), -2):   # widest gap that still gives four groups
            found = [(a + x_lo, b + x_lo) for a, b in _clusters(row, gap)]
            if len(found) >= 4:
                groups = found
                break
        if groups:
            break
    if len(groups) >= 4:
        groups = sorted(groups, key=lambda g: -g[1])
        if len(groups) > 4:                               # keep the four widest
            groups = sorted(sorted(groups, key=lambda g: -(g[1] - g[0]))[:4], key=lambda g: -g[1])
        widths = [g[1] - g[0] for g in groups]
        centres = [(g[0] + g[1]) / 2 for g in groups]
        edges = [x_hi] + [(centres[i] + centres[i + 1]) / 2 for i in range(3)] + [x_lo]
        stem_bottom = max([b for a, b in bands[:-1] if b < top + 0.4 * (bottom - top)] or [top])
        gap_h = last[0] - stem_bottom
        filled = 0
        for i in range(4):
            area = sub[stem_bottom - top:last[0] - top, int(edges[i + 1]):int(edges[i])]
            filled += area.size > 0 and area.mean() > 0.015
        if max(widths) < 0.075 * W and gap_h >= 2.2 * (last[1] - last[0]) and filled >= 3:
            # just letters, with pictures standing above them
            return {OPTION_KEYS[i]: page.norm(edges[i + 1], stem_bottom + 4, edges[i], last[1] + 6) for i in range(4)}, "pictures"
        return {OPTION_KEYS[i]: page.norm(g[0] - 8, last[0] - 6, g[1] + 8, last[1] + 6)
                for i, g in enumerate(groups)}, "row"
    if len(bands) >= 4:                                   # stacked lines or table rows
        rows = bands[-4:]
        rights = []
        for a, b in rows:
            cols = np.nonzero(sub[a - top:b - top, :int(0.905 * W)].any(axis=0))[0]
            rights.append(int(cols.max()) if len(cols) else 0)
        if max(rights) - min(rights) > 0.07 * W:          # not a neat right-aligned list
            return {}, "none"
        boxes = {}
        for i, (a, b) in enumerate(rows):
            y0 = (rows[i - 1][1] + a) // 2 if i else a - 6
            y1 = (b + rows[i + 1][0]) // 2 if i < 3 else b + 6
            cols = np.nonzero(sub[a - top:b - top, :int(0.905 * W)].any(axis=0))[0]
            xr = int(cols.max()) if len(cols) else x_hi - 1
            left = _row_left_edge(page, a, b, xr, gap_row)
            boxes[OPTION_KEYS[i]] = page.norm(left - 8, y0, xr + 12, y1)
        return boxes, "lines"
    return {}, "none"


# ---------------------------------------------------------------- student marks (hints)
def detect_marks(page: Page, option_boxes: Dict[str, List[float]]) -> Dict[str, List[str]]:
    """Hand-drawn ink: loops that enclose an option's text read as 'circle'.
    Everything else is left to the person to confirm in the editor."""
    marks: Dict[str, List[str]] = {}
    labels, comps = page.ink_components()
    for key, box in option_boxes.items():
        y0, x0, y1, x1 = box[0] * page.h, box[1] * page.w, box[2] * page.h, box[3] * page.w
        bw, bh = x1 - x0, y1 - y0
        for c in comps:
            w, h = c["x1"] - c["x0"], c["y1"] - c["y0"]
            if w < 0.8 * bw or h < 0.55 * bh or w > 1.8 * bw + 40 or h > 2.2 * bh + 30:
                continue
            fill = c["area"] / (w * h)
            if fill > 0.18:
                continue
            # the loop must wrap the option: its box contains most of the option box
            cx, cy = (c["x0"] + c["x1"]) / 2, (c["y0"] + c["y1"]) / 2
            if x0 - 0.2 * bw <= cx <= x1 + 0.2 * bw and y0 - 0.3 * bh <= cy <= y1 + 0.3 * bh:
                sub = labels[c["y0"]:c["y1"], c["x0"]:c["x1"]] == c["id"]
                ch, cw = sub.shape
                centre = sub[ch // 4: 3 * ch // 4, cw // 4: 3 * cw // 4].mean()
                if centre < 0.12:
                    marks.setdefault(key, []).append("circle")
                    break
    return marks


# ---------------------------------------------------------------- the whole page
def read_page(image: Image.Image) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read an upright page photo. Returns (questions, meta) like the AI reader does."""
    page = Page(image)
    spans, how = split_questions(page)
    bubbles = find_bubbles(page)
    questions: List[Dict[str, Any]] = []
    for idx, (top, bottom) in enumerate(spans, start=1):
        if bottom - top < 40:
            continue
        inside = [b for b in bubbles if top <= b["cy"] < bottom] if how != "frames" else []
        if len(inside) >= 2:
            right = max(b["cx"] + b["r"] for b in inside)
            options = option_boxes_from_bubbles(page, inside, (top, bottom), left_limit=0.04 * page.w)
            last_y = max(b["cy"] + b["r"] for b in inside) + 12
            q_box = page.norm(0.02 * page.w, max(0, top - 16), min(page.w, right + 0.1 * page.w),
                              min(bottom, last_y) if how == "badges" else bottom)
            layout = "bubbles"
        else:
            options, layout = frame_options(page, (top, bottom)) if how == "frames" else ({}, "none")
            q_box = page.norm(0.02 * page.w, top, 0.98 * page.w, bottom)
        if layout == "none" and how == "badges":
            continue                     # a coloured dot that is not a question
        marks = detect_marks(page, options)
        questions.append({
            "id": len(questions) + 1,
            "question_number": str(len(questions) + 1),
            "question_text": "",
            "options": {},
            "question_box": q_box,
            "option_boxes": options,
            "student_marks": marks,
            "answer_sequence": [],
            "student_choice": None,
            "correct_choice": None,
            "local_layout": layout,
        })
    meta = {"local": True, "split_by": how, "questions_found": len(questions)}
    return questions, meta
