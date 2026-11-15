"""qpack studio: a small local web app on top of qpack.

Open it in the phone browser, add photos, let Gemini read them or draw the
question / answer boxes yourself, mark what the student chose (including
crossed-out answers), save the page as verified "training" data, and build the
.qpack.  Verified pages are fed back to Gemini as worked examples on later scans.

    python -m qpack.studio            # http://127.0.0.1:8765
    qpack studio --dir ~/qpack-studio

Standard library only, apart from qpack's own dependencies (Pillow, NumPy, requests).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from PIL import Image

from qpack.analyzer import evaluate_question, resolve_question
from qpack.arabic import normalize_option_letter
from qpack.deskew import deskew_image, rotate_clockwise
from qpack.gemini import DEFAULT_MODEL, MAX_EXAMPLES, call_gemini_page
from qpack.vision import read_page
from qpack.packer import create_qpack

OPTION_KEYS = ["أ", "ب", "ج", "د"]
PAGE_ID = re.compile(r"^[0-9a-f]{12}$")
MAX_UPLOAD = 40 * 1024 * 1024
WEB_DIR = Path(__file__).parent / "web"


class StudioError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- helpers ---

def clean_box(box: Any) -> Optional[List[float]]:
    """Return [ymin, xmin, ymax, xmax] in 0..1, or None when it is unusable."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        y0, x0, y1, x1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if max(y0, x0, y1, x1) > 1.0:           # the model may answer in 0..1000
        y0, x0, y1, x1 = y0 / 1000.0, x0 / 1000.0, y1 / 1000.0, x1 / 1000.0
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    if y1 - y0 < 0.004 or x1 - x0 < 0.004:
        return None
    return [round(y0, 4), round(x0, 4), round(y1, 4), round(x1, 4)]


def clean_questions(raw: Any) -> List[Dict[str, Any]]:
    """Validate what the editor (or the model) sent and give every question a stable shape."""
    if not isinstance(raw, list):
        raise StudioError("questions must be a list")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        q = dict(item)
        q["id"] = idx
        q["question_number"] = str(q.get("question_number") or idx)
        q["question_box"] = clean_box(q.get("question_box"))
        if q.get("label_style") == "latin":
            q["label_style"] = "latin"          # left-to-right pages: show A B C D
        else:
            q.pop("label_style", None)
        q["student_working_box"] = clean_box(q.get("student_working_box"))

        options = q.get("options") if isinstance(q.get("options"), dict) else {}
        q["options"] = {
            (normalize_option_letter(k) or k): str(v) for k, v in options.items()
        }

        boxes: Dict[str, List[float]] = {}
        if isinstance(q.get("option_boxes"), dict):
            for k, b in q["option_boxes"].items():
                letter = normalize_option_letter(k) if isinstance(k, str) else None
                cb = clean_box(b)
                if letter and cb:
                    boxes[letter] = cb
        q["option_boxes"] = boxes

        marks: Dict[str, List[str]] = {}
        if isinstance(q.get("student_marks"), dict):
            for k, v in q["student_marks"].items():
                letter = normalize_option_letter(k) if isinstance(k, str) else None
                if letter and isinstance(v, list) and v:
                    marks[letter] = [str(m) for m in v]
        q["student_marks"] = marks

        seq = q.get("answer_sequence")
        q["answer_sequence"] = [
            l for l in (normalize_option_letter(x) for x in seq if isinstance(x, str)) if l
        ] if isinstance(seq, list) else []

        for key in ("correct_choice", "student_choice"):
            v = q.get(key)
            q[key] = normalize_option_letter(v) if isinstance(v, str) and v.strip() else None
        for key in ("concept_tag", "formula", "question_text", "solution_arabic"):
            q[key] = str(q.get(key) or "")
        try:
            q["working_ocr_confidence"] = float(q.get("working_ocr_confidence", 1.0))
        except (TypeError, ValueError):
            q["working_ocr_confidence"] = 1.0
        out.append(q)
    return out


def summarize(question: Dict[str, Any]) -> Dict[str, Any]:
    """What the editor shows live: the resolved answer plus the grading verdict."""
    resolved = resolve_question(question)
    verdict = evaluate_question(question)
    return {
        "final": resolved["final"],
        "status": verdict["status"],
        "method": resolved["method"],
        "candidates": resolved["candidates"],
        "crossed_out": resolved["crossed_out"],
        "changed": resolved["changed"],
        "flags": verdict["review_flags"],
        "notes": resolved["notes"],
        "correct_choice": verdict["correct_choice"],
    }


# ------------------------------------------------------------------ store ---

class Store:
    """Pages, their labels and built packs live in plain files under one folder."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        self.pages = self.root / "pages"
        self.packs = self.root / "packs"
        self.pages.mkdir(parents=True, exist_ok=True)
        self.packs.mkdir(parents=True, exist_ok=True)

    # -- files
    def _json(self, pid: str) -> Path:
        return self.pages / f"{pid}.json"

    def image_path(self, pid: str) -> Path:
        return self.pages / f"{pid}.jpg"

    def _check(self, pid: str) -> str:
        if not PAGE_ID.match(pid or ""):
            raise StudioError("unknown page", HTTPStatus.NOT_FOUND)
        if not self._json(pid).exists():
            raise StudioError("unknown page", HTTPStatus.NOT_FOUND)
        return pid

    def get(self, pid: str) -> Dict[str, Any]:
        self._check(pid)
        return json.loads(self._json(pid).read_text(encoding="utf-8"))

    def _write(self, record: Dict[str, Any]) -> None:
        tmp = self._json(record["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._json(record["id"]))

    def list(self) -> List[Dict[str, Any]]:
        rows = []
        for path in sorted(self.pages.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows.append({
                "id": rec["id"], "name": rec.get("name", ""), "verified": bool(rec.get("verified")),
                "questions": len(rec.get("questions", [])),
                "width": rec.get("width"), "height": rec.get("height"),
            })
        return rows

    # -- pages
    def add_page(self, name: str, data: bytes, rotate: Optional[int] = None) -> Dict[str, Any]:
        try:
            raw = Image.open(io.BytesIO(data))
            raw.load()
        except Exception:
            raise StudioError("that file is not a readable image")
        upright, angle = deskew_image(raw, rotate=rotate)
        pid = uuid.uuid4().hex[:12]
        upright.convert("RGB").save(self.image_path(pid), format="JPEG", quality=90)
        rec = {
            "id": pid, "name": name or f"page-{pid}", "angle": angle, "verified": False,
            "width": upright.width, "height": upright.height,
            "created": time.time(), "questions": [],
        }
        self._write(rec)
        return rec

    def rotate_page(self, pid: str, degrees: int) -> Dict[str, Any]:
        """Turn the stored page by a clockwise quarter-turn. Boxes would no longer fit, so they are cleared."""
        if degrees not in (90, 180, 270):
            raise StudioError("degrees must be 90, 180 or 270")
        rec = self.get(pid)
        img = rotate_clockwise(Image.open(self.image_path(pid)), degrees).convert("RGB")
        img.save(self.image_path(pid), format="JPEG", quality=90)
        rec.update(width=img.width, height=img.height, verified=False)
        rec["angle"] = float(rec.get("angle", 0.0)) - degrees
        for q in rec["questions"]:
            for key in ("question_box", "student_working_box"):
                q[key] = None
            q["option_boxes"] = {}
        self._write(rec)
        return rec

    def save_questions(self, pid: str, questions: Any, verified: bool) -> Dict[str, Any]:
        rec = self.get(pid)
        rec["questions"] = clean_questions(questions)
        rec["verified"] = bool(verified)
        self._write(rec)
        return rec

    def delete(self, pid: str) -> None:
        self._check(pid)
        for path in (self._json(pid), self.image_path(pid), self.packs / f"{pid}.qpack"):
            if path.exists():
                path.unlink()

    # -- AI
    def verified_examples(self, limit: int, exclude: str) -> List[Tuple[Image.Image, List[Dict[str, Any]]]]:
        found = []
        for path in sorted(self.pages.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if rec["id"] == exclude or not rec.get("verified") or not rec.get("questions"):
                continue
            img_path = self.image_path(rec["id"])
            if img_path.exists():
                found.append((Image.open(img_path).convert("RGB"), rec["questions"]))
            if len(found) >= limit:
                break
        return found

    def scan(self, pid: str, api_key: Optional[str], model: str, use_examples: bool,
             allow_mock: bool, engine: str = "local") -> Dict[str, Any]:
        rec = self.get(pid)
        image = Image.open(self.image_path(pid)).convert("RGB")
        if engine == "local" and not allow_mock:
            # Plain image processing on this device: no key, no network.
            questions, meta = read_page(image)
            if not questions:
                note = "No questions were found automatically. Use Fix to draw the boxes."
            else:
                note = "Read on this device. Pick the correct answers; marks are hints, check them."
            rec["questions"] = clean_questions(questions)
            rec["verified"] = False
            self._write(rec)
            return {"page": rec, "mock": False, "examples_used": 0, "note": note, "engine": "local",
                    "split_by": meta.get("split_by")}
        key = api_key or os.environ.get("GEMINI_API_KEY")
        examples = self.verified_examples(MAX_EXAMPLES, pid) if use_examples else []

        questions, meta = call_gemini_page(
            image, api_key=key, model=model or DEFAULT_MODEL,
            force_mock=allow_mock, examples=examples or None,
        )
        note = None
        turn = meta.get("page_rotation_clockwise")
        if turn and not meta.get("mock"):
            rec = self.rotate_page(pid, turn)
            image = Image.open(self.image_path(pid)).convert("RGB")
            questions, meta = call_gemini_page(
                image, api_key=key, model=model or DEFAULT_MODEL, examples=examples or None,
            )
            note = f"The page was turned {turn}° and read again."

        if meta.get("mock") and not allow_mock:
            # Never present canned data as if it were read from the photo.
            raise StudioError(f"The AI reading failed: {meta.get('reason')}", HTTPStatus.BAD_GATEWAY)

        rec["questions"] = clean_questions(questions)
        rec["verified"] = False
        self._write(rec)
        return {"page": rec, "mock": bool(meta.get("mock")), "examples_used": len(examples), "note": note}

    # -- outputs
    def build_pack(self, pid: str, crop_format: str) -> Path:
        rec = self.get(pid)
        if not rec["questions"]:
            raise StudioError("this page has no questions yet")
        image = Image.open(self.image_path(pid)).convert("RGB")
        out = self.packs / f"{pid}.qpack"
        create_qpack(rec["name"], image, float(rec.get("angle", 0.0)), rec["questions"], str(out),
                     crop_format=crop_format)
        return out

    def training_zip(self) -> bytes:
        """Verified pages as training data: images plus one JSON line per page."""
        buf = io.BytesIO()
        lines = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self.pages.glob("*.json")):
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                img_path = self.image_path(rec["id"])
                if not rec.get("verified") or not img_path.exists():
                    continue
                zf.write(img_path, f"images/{rec['id']}.jpg")
                lines.append(json.dumps(
                    {"image": f"images/{rec['id']}.jpg", "name": rec.get("name"),
                     "page_rotation_clockwise": 0, "questions": rec["questions"]},
                    ensure_ascii=False))
            zf.writestr("training.jsonl", "\n".join(lines) + ("\n" if lines else ""))
        return buf.getvalue()


# ------------------------------------------------------------------- http ---

class Handler(BaseHTTPRequestHandler):
    store: Store
    default_api_key: Optional[str] = None
    default_model: str = DEFAULT_MODEL
    server_version = "qpack-studio"

    def log_message(self, fmt: str, *args: Any) -> None:   # keep the terminal quiet
        if os.environ.get("QPACK_STUDIO_LOG"):
            super().log_message(fmt, *args)

    # -- plumbing
    def _send(self, status: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            raise StudioError("file too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        return self.rfile.read(length) if length else b""

    def _body_json(self) -> Dict[str, Any]:
        raw = self._body()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise StudioError("body is not valid JSON")
        if not isinstance(data, dict):
            raise StudioError("body must be a JSON object")
        return data

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        query = parse_qs(url.query)
        try:
            self._route(method, parts, query)
        except StudioError as exc:
            self._json({"error": str(exc)}, int(exc.status))
        except Exception as exc:                                   # never crash the server
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self) -> None: self._dispatch("GET")
    def do_POST(self) -> None: self._dispatch("POST")
    def do_PUT(self) -> None: self._dispatch("PUT")
    def do_DELETE(self) -> None: self._dispatch("DELETE")

    # -- routes
    def _route(self, method: str, parts: List[str], query: Dict[str, List[str]]) -> None:
        st = self.store

        if method == "GET" and not parts:
            return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")

        if parts[:1] != ["api"]:
            raise StudioError("not found", HTTPStatus.NOT_FOUND)
        parts = parts[1:]

        if parts == ["config"] and method == "GET":
            return self._json({"has_key": bool(self.default_api_key or os.environ.get("GEMINI_API_KEY")),
                               "model": self.default_model, "option_keys": OPTION_KEYS})

        if parts == ["pages"]:
            if method == "GET":
                return self._json(st.list())
            if method == "POST":
                name = (query.get("name") or [""])[0]
                rot = (query.get("rotate") or ["auto"])[0]
                if rot not in ("auto", "0", "90", "180", "270"):
                    raise StudioError("rotate must be auto, 0, 90, 180 or 270")
                rotate = None if rot == "auto" else int(rot)
                return self._json(st.add_page(name, self._body(), rotate), 201)

        if parts == ["resolve"] and method == "POST":
            body = self._body_json()
            return self._json(summarize(clean_questions([body.get("question") or {}])[0]))

        if parts == ["training.zip"] and method == "GET":
            return self._send(200, st.training_zip(), "application/zip",
                              {"Content-Disposition": 'attachment; filename="qpack-training.zip"'})

        if len(parts) >= 2 and parts[0] == "pages":
            pid = parts[1]
            tail = parts[2:]
            if tail == [] and method == "GET":
                rec = st.get(pid)
                rec["summaries"] = [summarize(q) for q in rec["questions"]]
                return self._json(rec)
            if tail == [] and method == "PUT":
                body = self._body_json()
                rec = st.save_questions(pid, body.get("questions"), bool(body.get("verified")))
                return self._json(rec)
            if tail == [] and method == "DELETE":
                st.delete(pid)
                return self._json({"ok": True})
            if tail == ["image"] and method == "GET":
                st._check(pid)
                return self._send(200, st.image_path(pid).read_bytes(), "image/jpeg")
            if tail == ["rotate"] and method == "POST":
                return self._json(st.rotate_page(pid, int(self._body_json().get("degrees", 0))))
            if tail == ["scan"] and method == "POST":
                body = self._body_json()
                key = body.get("api_key") or self.default_api_key
                return self._json(st.scan(pid, key, body.get("model") or self.default_model,
                                          body.get("use_examples", True), bool(body.get("mock")),
                                          body.get("engine") or "local"))
            if tail == ["pack"] and method == "POST":
                fmt = "PNG" if self._body_json().get("png") else "JPEG"
                path = st.build_pack(pid, fmt)
                stem = re.sub(r"[^\w.\-]+", "_", Path(st.get(pid)["name"]).stem) or pid
                return self._json({"url": f"/api/pages/{pid}/pack.qpack?name={stem}",
                                   "size": path.stat().st_size})
            if tail == ["pack.qpack"] and method == "GET":
                st._check(pid)
                path = st.packs / f"{pid}.qpack"
                if not path.exists():
                    raise StudioError("build the pack first", HTTPStatus.NOT_FOUND)
                stem = re.sub(r"[^\w.\-]+", "_", (query.get("name") or [pid])[0]) or pid
                return self._send(200, path.read_bytes(), "application/octet-stream",
                                  {"Content-Disposition": f'attachment; filename="{stem}.qpack"'})

        raise StudioError("not found", HTTPStatus.NOT_FOUND)


def make_server(directory: str, host: str = "127.0.0.1", port: int = 8765,
                api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> ThreadingHTTPServer:
    handler = type("StudioHandler", (Handler,), {
        "store": Store(directory), "default_api_key": api_key, "default_model": model,
    })
    return ThreadingHTTPServer((host, port), handler)


def serve(directory: str = "~/qpack-studio", host: str = "127.0.0.1", port: int = 8765,
          api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> int:
    try:
        server = make_server(directory, host, port, api_key, model)
    except OSError as exc:
        print(f"Could not start on {host}:{port}: {exc}", file=sys.stderr)
        return 1
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"qpack studio is running: open http://{shown}:{port} in your browser")
    print(f"Data folder: {Path(directory).expanduser()}   (Ctrl+C to stop)")
    if host == "0.0.0.0":
        print("Warning: 0.0.0.0 shares the studio with everyone on your network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="qpack-studio", description="Local web app for qpack")
    ap.add_argument("--dir", default="~/qpack-studio", help="Folder for pages, labels and packs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    return serve(a.dir, a.host, a.port, a.api_key, a.model)


if __name__ == "__main__":
    sys.exit(main())
