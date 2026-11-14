"""Question analysis, answer checking, confidence flagging, and concept/formula grouping."""

from __future__ import annotations
from typing import List, Dict, Any, Optional

from qpack.arabic import normalize_option_letter
from qpack.marks import resolve_answer

DEFAULT_CONFIDENCE_THRESHOLD = 0.70


def resolve_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the student's final answer for one question (see qpack.marks)."""
    options = q.get("options")
    option_keys = list(options.keys()) if isinstance(options, dict) and options else None
    # packer.create_qpack overwrites student_choice with the resolved answer and keeps
    # the model's own guess in student_choice_reported, so prefer that for the cross-check.
    reported = q["student_choice_reported"] if "student_choice_reported" in q else q.get("student_choice")
    return resolve_answer(
        q.get("student_marks"),
        option_keys=option_keys,
        answer_sequence=q.get("answer_sequence"),
        reported_choice=reported,
    )


def evaluate_question(
    q: Dict[str, Any],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """Evaluate a single question: student correctness and OCR confidence review flag."""
    ans = resolve_question(q)
    student = ans["final"]
    raw_correct = q.get("correct_choice")
    correct = normalize_option_letter(raw_correct) if isinstance(raw_correct, str) else None
    conf = float(q.get("working_ocr_confidence", 1.0))

    if ans["status"] == "AMBIGUOUS":
        status = "AMBIGUOUS"
        is_correct = False
    elif student is None:
        status = "UNANSWERED"
        is_correct = False
    elif correct is None:
        status = "UNGRADED"          # no answer key, so we cannot call it right or wrong
        is_correct = False
    elif student == correct:
        status = "CORRECT"
        is_correct = True
    else:
        status = "WRONG"
        is_correct = False

    review_flags: List[str] = []
    if conf < confidence_threshold:
        review_flags.append(f"Low pencil OCR confidence ({conf:.0%})")
    if status == "UNANSWERED":
        review_flags.append("No student answer detected")
    if status == "UNGRADED":
        review_flags.append("No answer key for this question")
    review_flags.extend(ans["flags"])

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
        "answer_method": ans["method"],
        "candidates": ans["candidates"],
        "crossed_out": ans["crossed_out"],
        "changed_answer": ans["changed"],
        "answer_notes": ans["notes"],
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
    ambiguous_count = sum(1 for e in evaluated_questions if e["status"] == "AMBIGUOUS")
    ungraded_count = sum(1 for e in evaluated_questions if e["status"] == "UNGRADED")
    changed_count = sum(1 for e in evaluated_questions if e["changed_answer"])
    flagged_count = sum(1 for e in evaluated_questions if e["needs_review"])

    graded = total - ungraded_count
    score_pct = (correct_count / graded * 100.0) if graded > 0 else 0.0

    return {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "unanswered": unanswered_count,
        "ambiguous": ambiguous_count,
        "ungraded": ungraded_count,
        "changed_answers": changed_count,
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
