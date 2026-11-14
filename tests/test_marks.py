"""Tests for resolving the final answer from ticks, circles, crosses and changed answers."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from PIL import Image

from qpack.analyzer import check_exam, evaluate_question
from qpack.gemini import call_gemini_page
from qpack.marks import normalize_mark, resolve_answer
from qpack.packer import create_qpack, load_qpack

OPTS = ["أ", "ب", "ج", "د"]


def _flags(result):
    return " | ".join(result["flags"])


def test_single_tick_is_the_answer():
    r = resolve_answer({"ب": ["tick"]}, OPTS)
    assert r["final"] == "ب"
    assert r["status"] == "ANSWERED"
    assert r["method"] == "mark"
    assert r["changed"] is False
    assert r["flags"] == []


def test_x_beside_wrong_answer_then_another_chosen():
    # The student put a clear X beside one answer and chose another.
    r = resolve_answer({"أ": ["cross"], "ج": ["tick"]}, OPTS)
    assert r["final"] == "ج"
    assert r["crossed_out"] == ["أ"]
    assert r["changed"] is True
    assert r["flags"] == []


def test_ticked_then_crossed_out_then_another_circled():
    r = resolve_answer({"أ": ["tick", "cross"], "ب": ["circle"]}, OPTS)
    assert r["final"] == "ب"
    assert r["crossed_out"] == ["أ"]
    assert r["changed"] is True
    assert r["flags"] == []
    assert any("أ" in n for n in r["notes"])


def test_many_changes_in_a_row():
    r = resolve_answer(
        {
            "أ": ["tick", "cross"],
            "ب": ["circle", "strike"],
            "ج": ["tick", "scribble"],
            "د": ["tick"],
        },
        OPTS,
    )
    assert r["final"] == "د"
    assert r["crossed_out"] == ["أ", "ب", "ج"]
    assert r["changed"] is True
    assert r["flags"] == []


def test_cross_then_tick_on_same_option_is_flagged():
    # The X was drawn first and the tick over it: the answer hinges on the order.
    r = resolve_answer({"ب": ["cross", "tick"]}, OPTS)
    assert r["final"] == "ب"
    assert "order of marks on option ب" in _flags(r)


def test_all_options_crossed_out_is_unanswered_and_flagged():
    r = resolve_answer({k: ["cross"] for k in OPTS}, OPTS)
    assert r["final"] is None
    assert r["status"] == "UNANSWERED"
    assert "Every option is crossed out" in _flags(r)


def test_two_options_marked_without_order_is_ambiguous():
    r = resolve_answer({"أ": ["tick"], "ب": ["tick"]}, OPTS)
    assert r["status"] == "AMBIGUOUS"
    assert r["final"] is None
    assert r["candidates"] == ["أ", "ب"]
    assert r["flags"]


def test_two_options_marked_with_order_takes_latest_but_flags():
    r = resolve_answer({"أ": ["tick"], "ب": ["tick"]}, OPTS, answer_sequence=["أ", "ب"])
    assert r["final"] == "ب"
    assert r["method"] == "latest_marked"
    assert "took the latest" in _flags(r)


def test_elimination_is_inferred_and_flagged():
    r = resolve_answer({"أ": ["cross"], "ب": ["cross"], "د": ["strike"]}, OPTS)
    assert r["final"] == "ج"
    assert r["method"] == "elimination"
    assert "inferred" in _flags(r)
    assert r["changed"] is False


def test_a_single_cross_does_not_pick_an_answer():
    r = resolve_answer({"أ": ["cross"]}, OPTS)
    assert r["final"] is None
    assert r["status"] == "UNANSWERED"
    assert "no final choice" in _flags(r)


def test_mark_aliases_are_understood():
    r = resolve_answer({"أ": ["✗"], "ب": ["check"]}, OPTS)
    assert r["final"] == "ب"
    assert r["crossed_out"] == ["أ"]
    assert normalize_mark("Strikethrough") == "strike"
    assert normalize_mark("crossed out") == "cross"
    assert normalize_mark("zigzag") is None


def test_unrecognized_mark_is_flagged_not_fatal():
    r = resolve_answer({"ب": ["tick", "zigzag"]}, OPTS)
    assert r["final"] == "ب"
    assert "Unrecognized mark 'zigzag'" in _flags(r)


def test_marks_on_unknown_option_are_ignored_and_flagged():
    r = resolve_answer({"هـ": ["tick"], "ب": ["tick"]}, OPTS)
    assert r["final"] == "ب"
    assert "unknown option" in _flags(r)


def test_alef_variants_are_normalised_in_marks():
    assert resolve_answer({"ا": ["tick"]}, OPTS)["final"] == "أ"


def test_reported_choice_is_used_when_there_are_no_marks():
    assert resolve_answer(None, OPTS, reported_choice="ا")["final"] == "أ"
    assert resolve_answer(None, OPTS, reported_choice="ا")["method"] == "reported"
    assert resolve_answer({}, OPTS, reported_choice="none")["final"] is None
    assert resolve_answer({}, OPTS, reported_choice=None)["status"] == "UNANSWERED"


def test_reported_choice_that_disagrees_with_marks_is_flagged():
    r = resolve_answer({"ب": ["tick"]}, OPTS, reported_choice="ج")
    assert r["final"] == "ب"
    assert "Model reported ج" in _flags(r)


def test_evaluate_question_grades_the_final_answer_after_a_change():
    q = {
        "options": {k: "x" for k in OPTS},
        "student_marks": {"أ": ["tick", "cross"], "ب": ["circle"]},
        "correct_choice": "ب",
    }
    e = evaluate_question(q)
    assert e["status"] == "CORRECT"
    assert e["student_choice"] == "ب"
    assert e["changed_answer"] is True
    assert e["needs_review"] is False


def test_check_exam_counts_changed_ambiguous_and_ungraded():
    opts = {k: "x" for k in OPTS}
    questions = [
        {"options": opts, "student_marks": {"أ": ["cross"], "ب": ["tick"]}, "correct_choice": "ب"},
        {"options": opts, "student_marks": {"أ": ["tick"], "ب": ["tick"]}, "correct_choice": "ب"},
        {"options": opts, "student_marks": {"ج": ["tick"]}},
        {"options": opts, "student_marks": {"د": ["tick"]}, "correct_choice": "أ"},
        {"options": opts, "student_choice": None, "correct_choice": "ب"},
    ]
    s = check_exam(questions)
    assert s["total"] == 5
    assert s["correct"] == 1
    assert s["wrong"] == 1
    assert s["unanswered"] == 1
    assert s["ambiguous"] == 1
    assert s["ungraded"] == 1
    assert s["changed_answers"] == 1
    # The question without an answer key is left out of the score
    assert s["score_percent"] == 25.0


def test_pack_stores_the_resolved_choice_and_keeps_the_model_guess():
    question = {
        "id": 1,
        "question_number": "1",
        "options": {k: "x" for k in OPTS},
        "student_marks": {"أ": ["tick", "cross"], "ج": ["circle"]},
        "student_choice": "أ",          # the model's (wrong) guess
        "correct_choice": "ج",
        "question_box": [0.0, 0.0, 1.0, 1.0],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.qpack")
        create_qpack("p.jpg", Image.new("RGB", (100, 100), "white"), 0.0, [question], path)
        manifest, questions = load_qpack(path)
    saved = questions[0]
    assert saved["student_choice"] == "ج"
    assert saved["student_choice_reported"] == "أ"
    assert manifest["summary"]["changed_answers"] == 1
    # Re-checking the saved pack still shows the model/marks mismatch
    assert "Model reported أ" in " ".join(evaluate_question(saved)["review_flags"])


# --- Gemini client: marks, rotation hint, and honest failure reporting ---------

def _gemini_body(payload):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


def test_gemini_page_returns_marks_and_rotation_and_keeps_key_out_of_url():
    payload = {
        "page_rotation_clockwise": 90,
        "questions": [{
            "question_number": "1",
            "options": {"أ": "x", "ب": "y"},
            "student_marks": {"أ": ["cross"], "ب": ["tick"]},
            "student_choice": "ا",
            "correct_choice": "ب",
        }],
    }
    with patch("qpack.gemini.requests.post") as post:
        post.return_value = MagicMock(status_code=200, json=lambda: _gemini_body(payload))
        questions, meta = call_gemini_page(Image.new("RGB", (50, 50)), api_key="SECRETKEY")

    assert meta == {"page_rotation_clockwise": 90}
    assert questions[0]["id"] == 1
    assert questions[0]["student_marks"] == {"أ": ["cross"], "ب": ["tick"]}
    assert questions[0]["student_choice"] == "أ"        # alef variant normalised
    assert "SECRETKEY" not in post.call_args.args[0]
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "SECRETKEY"


def test_gemini_page_accepts_a_plain_list_and_ignores_bad_rotation():
    with patch("qpack.gemini.requests.post") as post:
        post.return_value = MagicMock(
            status_code=200,
            json=lambda: _gemini_body([{"question_number": "1", "options": {"أ": "x"}}]),
        )
        questions, meta = call_gemini_page(Image.new("RGB", (50, 50)), api_key="k")
    assert meta == {}
    assert len(questions) == 1

    with patch("qpack.gemini.requests.post") as post:
        post.return_value = MagicMock(
            status_code=200,
            json=lambda: _gemini_body({"page_rotation_clockwise": 45, "questions": [{"options": {}}]}),
        )
        _, meta = call_gemini_page(Image.new("RGB", (50, 50)), api_key="k")
    assert meta == {"page_rotation_clockwise": None}


def test_gemini_failure_says_why_and_never_leaks_the_key():
    boom = RuntimeError("connection failed for https://example.test/x?key=SECRETKEY")
    with patch("qpack.gemini.requests.post", side_effect=boom):
        questions, meta = call_gemini_page(Image.new("RGB", (50, 50)), api_key="SECRETKEY")
    assert meta["mock"] is True
    assert "connection failed" in meta["reason"]
    assert "SECRETKEY" not in meta["reason"]
    assert questions  # the pipeline still gets the offline dataset


def test_gemini_mock_reasons():
    img = Image.new("RGB", (50, 50))
    assert call_gemini_page(img, force_mock=True)[1]["mock"] is True
    with patch.dict(os.environ, {}, clear=True):
        _, meta = call_gemini_page(img)
    assert meta["mock"] is True
    assert "GEMINI_API_KEY" in meta["reason"]
