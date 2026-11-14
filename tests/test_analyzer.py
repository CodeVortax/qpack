"""Unit tests for answer evaluation, OCR confidence review flagging, and question grouping."""

from qpack.analyzer import evaluate_question, check_exam, group_questions
from qpack.mock_data import MOCK_PHYSICS_QUESTIONS


def test_evaluate_question_correct():
    q = {
        "id": 1,
        "question_number": "١",
        "student_choice": "ب",
        "correct_choice": "ب",
        "working_ocr_confidence": 0.95,
        "concept_tag": "عزم الازدواج المغناطيسي",
        "formula": r"\tau = B I A N \sin\theta"
    }
    eval_res = evaluate_question(q, confidence_threshold=0.70)
    assert eval_res["status"] == "CORRECT"
    assert eval_res["is_correct"] is True
    assert eval_res["needs_review"] is False


def test_evaluate_question_wrong():
    q = {
        "id": 2,
        "question_number": "٢",
        "student_choice": "أ",
        "correct_choice": "ج",
        "working_ocr_confidence": 0.85,
        "concept_tag": "القوة المغناطيسية المتبادلة بين سلكين"
    }
    eval_res = evaluate_question(q, confidence_threshold=0.70)
    assert eval_res["status"] == "WRONG"
    assert eval_res["is_correct"] is False
    assert eval_res["needs_review"] is False


def test_evaluate_question_low_confidence_flagged():
    q = {
        "id": 3,
        "question_number": "٣",
        "student_choice": "ج",
        "correct_choice": "ج",
        "working_ocr_confidence": 0.58,  # Below 0.70 threshold
        "concept_tag": "عزم الازدواج المغناطيسي"
    }
    eval_res = evaluate_question(q, confidence_threshold=0.70)
    assert eval_res["status"] == "CORRECT"
    assert eval_res["needs_review"] is True
    assert any("Low pencil OCR confidence" in r for r in eval_res["review_flags"])


def test_evaluate_question_unanswered():
    q = {
        "id": 4,
        "question_number": "٤",
        "student_choice": None,
        "correct_choice": "د",
        "working_ocr_confidence": 0.90
    }
    eval_res = evaluate_question(q)
    assert eval_res["status"] == "UNANSWERED"
    assert eval_res["needs_review"] is True


def test_check_exam_summary():
    exam_summary = check_exam(MOCK_PHYSICS_QUESTIONS, confidence_threshold=0.70)
    assert exam_summary["total"] == 6
    assert exam_summary["correct"] == 5
    assert exam_summary["wrong"] == 1
    assert exam_summary["unanswered"] == 0
    assert exam_summary["score_percent"] == 83.3
    # Question 37 has confidence 0.62, so 1 question flagged for review
    assert exam_summary["flagged_for_review"] == 1


def test_group_by_concept():
    grouped = group_questions(MOCK_PHYSICS_QUESTIONS, by="concept")
    assert "نقطة التعادل بين سلكين مستقيمين" in grouped
    assert len(grouped["نقطة التعادل بين سلكين مستقيمين"]) == 3  # Q36, Q39, Q40
    assert "معامل النفاذية المغناطيسية" in grouped
    assert len(grouped["معامل النفاذية المغناطيسية"]) == 1


def test_group_by_formula():
    grouped = group_questions(MOCK_PHYSICS_QUESTIONS, by="formula")
    assert any(r"\frac{I_1}{d_1}" in k for k in grouped.keys())
    assert any("B = " in k or r"\mu" in k for k in grouped.keys())
