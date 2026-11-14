"""Question analysis, answer checking, confidence flagging, and concept/formula grouping."""

from __future__ import annotations
from typing import List, Dict, Any, Optional

DEFAULT_CONFIDENCE_THRESHOLD = 0.70


def evaluate_question(
    q: Dict[str, Any],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """Evaluate a single question: student correctness and OCR confidence review flag."""
    student = q.get("student_choice")
    correct = q.get("correct_choice")
    conf = float(q.get("working_ocr_confidence", 1.0))

    if student is None or str(student).strip() == "" or str(student).lower() == "none":
        status = "UNANSWERED"
        is_correct = False
    elif str(student).strip() == str(correct).strip():
        status = "CORRECT"
        is_correct = True
    else:
        status = "WRONG"
        is_correct = False

    review_flags: List[str] = []
    if conf < confidence_threshold:
        review_flags.append(f"Low pencil OCR confidence ({conf:.0%})")
    if student is None:
        review_flags.append("No student answer detected")

    needs_review = len(review_flags) > 0

    return {
        "id": q.get("id"),
        "question_number": q.get("question_number"),
        "concept_tag": q.get("concept_tag", "عام"),
        "formula": q.get("formula", ""),
        "student_choice": student,
        "correct_choice": correct,
        "status": status,
        "is_correct": is_correct,
        "working_ocr_confidence": conf,
        "needs_review": needs_review,
        "review_flags": review_flags,
    }


def check_exam(
    questions: List[Dict[str, Any]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """Check all questions in an exam and generate an evaluation summary."""
    evaluated_questions = [
        evaluate_question(q, confidence_threshold=confidence_threshold)
        for q in questions
    ]

    total = len(evaluated_questions)
    correct_count = sum(1 for e in evaluated_questions if e["status"] == "CORRECT")
    wrong_count = sum(1 for e in evaluated_questions if e["status"] == "WRONG")
    unanswered_count = sum(1 for e in evaluated_questions if e["status"] == "UNANSWERED")
    flagged_count = sum(1 for e in evaluated_questions if e["needs_review"])

    score_pct = (correct_count / total * 100.0) if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "unanswered": unanswered_count,
        "flagged_for_review": flagged_count,
        "score_percent": round(score_pct, 1),
        "results": evaluated_questions
    }


def group_questions(
    questions: List[Dict[str, Any]],
    by: str = "concept"
) -> Dict[str, List[Dict[str, Any]]]:
    """Group questions by 'concept', 'formula', or 'number'."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for q in questions:
        if by == "concept":
            key = q.get("concept_tag") or "بدون تصنيف"
        elif by == "formula":
            key = q.get("formula") or "بدون قانون"
        else:
            key = f"سؤال {q.get('question_number', q.get('id', '?'))}"

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(q)

    return grouped
