"""End-to-end tests for the qpack studio web app (real HTTP server on a free port)."""

import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from qpack import studio
from qpack.packer import load_qpack


def page_image(sideways=False):
    img = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(img)
    for y in range(100, 700, 50):
        d.rectangle([50, y, 550, y + 12], fill=(0, 0, 0))
    return img.rotate(-90, expand=True) if sideways else img


def jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def app(tmp_path):
    server = studio.make_server(str(tmp_path), port=0, api_key=None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if raw is None else {})
        try:
            with urllib.request.urlopen(req) as r:
                payload = r.read()
                ctype = r.headers.get("Content-Type", "")
                return r.status, (json.loads(payload) if "json" in ctype else payload)
        except urllib.error.HTTPError as e:
            payload = e.read()
            return e.code, (json.loads(payload) if "json" in e.headers.get("Content-Type", "") else payload)

    yield call, server
    server.shutdown()
    server.server_close()


def add_page(call, **kw):
    status, rec = call("POST", "/api/pages?name=p.jpg&rotate=" + kw.get("rotate", "auto"),
                       raw=jpeg_bytes(page_image(kw.get("sideways", False))))
    assert status == 201
    return rec


QUESTION = {
    "question_number": "7",
    "question_box": [0.1, 0.1, 0.5, 0.9],
    "option_boxes": {"أ": [0.5, 0.1, 0.6, 0.4], "ب": [0.5, 0.5, 0.6, 0.9], "ا": [0.7, 0.1, 0.8, 0.4]},
    "student_marks": {"أ": ["tick", "cross"], "ب": ["circle"]},
    "answer_sequence": ["أ", "ب"],
    "correct_choice": "ب",
}


def test_page_is_served_and_config_lists_options(app):
    call, _ = app
    status, html = call("GET", "/")
    assert status == 200 and b"qpack studio" in html
    status, cfg = call("GET", "/api/config")
    assert cfg["option_keys"] == ["أ", "ب", "ج", "د"] and cfg["has_key"] is False


def test_sideways_photo_is_stored_upright(app):
    call, _ = app
    rec = add_page(call, sideways=True)
    assert (rec["width"], rec["height"]) == (600, 800)
    status, img = call("GET", f"/api/pages/{rec['id']}/image")
    assert status == 200 and Image.open(io.BytesIO(img)).size == (600, 800)


def test_bad_upload_and_bad_ids_are_rejected(app):
    call, _ = app
    assert call("POST", "/api/pages?name=x", raw=b"not an image")[0] == 400
    assert call("POST", "/api/pages?rotate=45", raw=jpeg_bytes(page_image()))[0] == 400
    assert call("GET", "/api/pages/../../etc/passwd")[0] == 404
    assert call("GET", "/api/pages/zzzzzzzzzzzz")[0] == 404
    assert call("GET", "/nope")[0] == 404


def test_save_cleans_questions_and_resolve_follows_the_marks(app):
    call, _ = app
    pid = add_page(call)["id"]
    status, rec = call("PUT", f"/api/pages/{pid}", {"questions": [QUESTION, {"question_box": [0.5, 0.5, 0.5, 0.5]}],
                                                    "verified": True})
    assert status == 200 and rec["verified"] is True
    q1, q2 = rec["questions"]
    assert q1["id"] == 1 and set(q1["option_boxes"]) == {"أ", "ب"}      # ا normalised onto أ
    assert q2["question_box"] is None                                    # zero-size box dropped
    assert q2["question_number"] == "2"

    status, live = call("POST", "/api/resolve", {"question": q1})
    assert live["final"] == "ب" and live["status"] == "CORRECT"
    assert live["changed"] is True and live["crossed_out"] == ["أ"]

    status, got = call("GET", f"/api/pages/{pid}")
    assert got["summaries"][0]["final"] == "ب"


def test_pack_contains_a_crop_for_every_option_and_loads_back(app, tmp_path):
    call, _ = app
    pid = add_page(call)["id"]
    call("PUT", f"/api/pages/{pid}", {"questions": [QUESTION], "verified": False})
    status, built = call("POST", f"/api/pages/{pid}/pack", {})
    assert status == 200 and built["url"].endswith("pack.qpack?name=p")

    status, data = call("GET", built["url"])
    assert status == 200
    path = tmp_path / "out.qpack"
    path.write_bytes(data)
    manifest, questions = load_qpack(str(path))
    q = questions[0]
    assert set(q["option_image_paths"]) == {"أ", "ب"}
    assert q["student_choice"] == "ب"
    assert manifest["summary"]["correct"] == 1 and manifest["summary"]["changed_answers"] == 1
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert all(p in names for p in q["option_image_paths"].values())
    assert q["question_image_path"].endswith(".jpg")


def test_pack_needs_questions(app):
    call, _ = app
    pid = add_page(call)["id"]
    assert call("POST", f"/api/pages/{pid}/pack", {})[0] == 400
    assert call("GET", f"/api/pages/{pid}/pack.qpack")[0] == 404


def test_training_export_has_only_verified_pages(app):
    call, _ = app
    a, b = add_page(call)["id"], add_page(call)["id"]
    call("PUT", f"/api/pages/{a}", {"questions": [QUESTION], "verified": True})
    call("PUT", f"/api/pages/{b}", {"questions": [QUESTION], "verified": False})
    status, blob = call("GET", "/api/training.zip")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        lines = zf.read("training.jsonl").decode("utf-8").strip().splitlines()
        names = zf.namelist()
    assert len(lines) == 1 and json.loads(lines[0])["questions"][0]["correct_choice"] == "ب"
    assert f"images/{a}.jpg" in names and f"images/{b}.jpg" not in names


def test_scan_feeds_verified_pages_to_the_model(app):
    call, _ = app
    done, target = add_page(call)["id"], add_page(call)["id"]
    call("PUT", f"/api/pages/{done}", {"questions": [QUESTION], "verified": True})
    seen = {}

    def fake(image, api_key=None, model="m", timeout=60, force_mock=False, examples=None):
        seen["examples"], seen["key"] = examples, api_key
        return [dict(QUESTION)], {}

    with patch("qpack.studio.call_gemini_page", fake):
        status, res = call("POST", f"/api/pages/{target}/scan", {"api_key": "K"})
    assert status == 200 and res["examples_used"] == 1 and res["page"]["verified"] is False
    assert seen["key"] == "K" and len(seen["examples"]) == 1
    assert seen["examples"][0][1][0]["correct_choice"] == "ب"

    with patch("qpack.studio.call_gemini_page", fake):
        _, res = call("POST", f"/api/pages/{target}/scan", {"api_key": "K", "use_examples": False})
    assert res["examples_used"] == 0 and seen["examples"] is None


def test_scan_never_saves_mock_data_as_a_real_reading(app):
    call, _ = app
    pid = add_page(call)["id"]
    with patch("qpack.studio.call_gemini_page", lambda *a, **k: ([{"x": 1}], {"mock": True, "reason": "quota"})):
        status, res = call("POST", f"/api/pages/{pid}/scan", {})
    assert status == 502 and "quota" in res["error"]
    assert call("GET", f"/api/pages/{pid}")[1]["questions"] == []

    # The demo switch is explicit
    status, res = call("POST", f"/api/pages/{pid}/scan", {"mock": True})
    assert status == 200 and res["mock"] is True and len(res["page"]["questions"]) > 0


def test_scan_rotates_and_rereads_when_the_model_says_the_page_is_turned(app):
    call, _ = app
    pid = add_page(call)["id"]
    calls = []

    def fake(image, api_key=None, model="m", timeout=60, force_mock=False, examples=None):
        calls.append(image.size)
        return [dict(QUESTION)], {"page_rotation_clockwise": 90 if len(calls) == 1 else 0}

    with patch("qpack.studio.call_gemini_page", fake):
        status, res = call("POST", f"/api/pages/{pid}/scan", {"api_key": "K"})
    assert status == 200 and calls == [(600, 800), (800, 600)]
    assert "turned 90" in res["note"]


def test_rotate_and_delete(app):
    call, _ = app
    pid = add_page(call)["id"]
    call("PUT", f"/api/pages/{pid}", {"questions": [QUESTION], "verified": True})
    status, rec = call("POST", f"/api/pages/{pid}/rotate", {"degrees": 90})
    assert (rec["width"], rec["height"]) == (800, 600) and rec["verified"] is False
    assert rec["questions"][0]["question_box"] is None and rec["questions"][0]["option_boxes"] == {}
    assert call("POST", f"/api/pages/{pid}/rotate", {"degrees": 45})[0] == 400
    assert call("DELETE", f"/api/pages/{pid}")[0] == 200
    assert call("GET", f"/api/pages/{pid}")[0] == 404


def test_gemini_request_includes_examples_before_the_new_page():
    from qpack.gemini import build_example_parts
    parts = build_example_parts([(Image.new("RGB", (50, 50)), [dict(QUESTION, junk="x")])])
    kinds = ["text" if "text" in p else "image" for p in parts]
    assert kinds == ["text", "image", "text", "text"]
    shown = json.loads(parts[2]["text"])["questions"][0]
    assert shown["correct_choice"] == "ب" and "junk" not in shown
    assert build_example_parts(None) == []
