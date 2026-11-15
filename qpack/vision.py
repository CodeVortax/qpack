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
    """y positions of the long coloured horizontal lines that frame questions.
    A frame's bottom edge and the next frame's top edge sit close together and
    are merged into one boundary."""
    rows = page.colour.sum(axis=1)
    ys = np.nonzero(rows > 0.12 * page.w)[0]
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
def option_cells_in_frame(page: Page, span: Tuple[int, int]) -> List[Dict[str, int]]:
    """Inside a framed question, the options are the separate word groups on the
    last text line(s). Cluster the ink there by horizontal gaps."""
    top, bottom = span
    x_lo, x_hi = int(0.04 * page.w), int(0.97 * page.w)
    region = page.ink[top + 6:bottom - 6, x_lo:x_hi].copy()
    # remove coloured frame lines so they do not glue everything together
    region &= ~page.colour[top + 6:bottom - 6, x_lo:x_hi]
    if region.size == 0:
        return []
    rows = region.sum(axis=1).astype(float)
    if rows.max() == 0:
        return []
    inked = rows > max(2.0, 0.08 * rows.max())
    bands: List[List[int]] = []
    for y, on in enumerate(inked):
        if on:
            if bands and y - bands[-1][1] <= 6:
                bands[-1][1] = y
            else:
                bands.append([y, y])
    bands = [b for b in bands if b[1] - b[0] >= 12]
    if not bands:
        return []
    last = bands[-1]
    band = region[max(0, last[0] - 8):last[1] + 8]
    # close gaps shorter than an option gap by smearing horizontally
    cols = band.any(axis=0)
    cells: List[List[int]] = []
    gap = int(0.034 * page.w)
    for x, on in enumerate(cols):
        if on:
            if cells and x - cells[-1][1] <= gap:
                cells[-1][1] = x
            else:
                cells.append([x, x])
    cells = [c for c in cells if c[1] - c[0] >= 18]
    y_off = top + 6 + max(0, last[0] - 8)
    return [{"x0": c[0] + x_lo, "x1": c[1] + x_lo + 1, "y0": y_off, "y1": y_off + band.shape[0]} for c in cells]


def option_boxes_from_cells(page: Page, cells: List[Dict[str, int]]) -> Dict[str, List[float]]:
    cells = sorted(cells, key=lambda c: -c["x1"])
    if len(cells) > 4:
        cells = sorted(sorted(cells, key=lambda c: -(c["x1"] - c["x0"]))[:4], key=lambda c: -c["x1"])
    return {key: page.norm(c["x0"] - 6, c["y0"] - 4, c["x1"] + 6, c["y1"] + 4) for key, c in zip(OPTION_KEYS, cells)}


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
        inside = [b for b in bubbles if top <= b["cy"] < bottom]
        if len(inside) >= 2:
            right = max(b["cx"] + b["r"] for b in inside)
            options = option_boxes_from_bubbles(page, inside, (top, bottom), left_limit=0.04 * page.w)
            last_y = max(b["cy"] + b["r"] for b in inside) + 12
            q_box = page.norm(0.02 * page.w, max(0, top - 16), min(page.w, right + 0.1 * page.w),
                              min(bottom, last_y) if how == "badges" else bottom)
            layout = "bubbles"
        else:
            cells = option_cells_in_frame(page, (top, bottom)) if how == "frames" else []
            options = option_boxes_from_cells(page, cells) if len(cells) >= 2 else {}
            q_box = page.norm(0.02 * page.w, top, 0.98 * page.w, bottom)
            layout = "frame" if options else "none"
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
