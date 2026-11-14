"""Gemini Vision API client using direct REST calls via requests."""

from __future__ import annotations
import base64
import io
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import requests

from qpack.arabic import to_western_digits, normalize_option_letter
from qpack.mock_data import MOCK_PHYSICS_QUESTIONS

DEFAULT_MODEL = "gemini-2.5-flash"
# The key travels in the x-goog-api-key header, never in the URL, so it cannot
# leak through logs or exception messages.
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

EXTRACTION_SYSTEM_PROMPT = """You are an expert Arabic Physics teacher, exam grader, and high-precision OCR vision engine.
You are analyzing a photo of an Arabic (right-to-left) physics exam or workbook page.
The page contains:
- Numbered multiple-choice questions with options labeled (أ, ب, ج, د).
- Physics diagrams, schematics, and graphs.
- Student pencil or pen marks on the options: ticks, circles, crosses, strike-throughs.
- Student handwritten pencil calculations and working in the margins.
- Potential background watermarks (e.g. teacher names, logos, phone numbers, diagonal patterns).

CRITICAL INSTRUCTIONS:
1. IGNORE ALL BACKGROUND WATERMARKS (such as teacher watermark logos, phone numbers, center diagonal text). Focus strictly on printed exam questions, diagrams, student marks, and pencil margin working.
2. Split the page by each numbered question.
3. For each question, extract:
   - question_number: The printed question number (e.g. "1" or "١").
   - question_text: Full Arabic question text, including any descriptions. Keep all math in LaTeX format wrapped in $...$ (e.g. $B = 0.4\\text{ T}$).
   - options: Object mapping Arabic option keys "أ", "ب", "ج", "د" to their printed text and LaTeX math formulas.
   - student_marks: Object mapping each option letter to the list of student ink marks on or next to that option, EARLIEST FIRST. When marks overlap, the one drawn on top is the later one. Allowed mark names:
       "tick"      a check mark, hook or ✓
       "circle"    a hand-drawn ring around the option or its letter
       "underline" a line under the option
       "fill"      the option or its letter shaded in
       "written"   the student wrote the letter or answer beside the question
       "cross"     a clear X on or beside the option
       "strike"    a single line through the option text or letter
       "scribble"  a heavy scribble that blots the option out
     Omit options that carry no student mark. Use {} if the student marked nothing.
     Students often tick or circle an option, then cross it out and pick another, sometimes several times. Report EVERY mark you can see on EVERY option, including the ones that were crossed out. Do NOT decide the final answer yourself; it is computed from your marks.
     Printed bubbles or circles around the option letters are part of the book's layout and are NOT student marks. Only count ink that is hand-drawn, irregular, or in a different colour or pencil/pen.
   - answer_sequence: The option letters in the order the student worked through them, earliest first, when you can tell (for example ["أ", "ب", "د"] if أ was marked first, then ب, and د last). Use [] if you cannot tell.
   - student_choice: Your best guess of the student's final answer ("أ", "ب", "ج", "د", or null if unanswered). This is only a cross-check.
   - question_box: Normalized bounding box [ymin, xmin, ymax, xmax] in range 0.0 to 1.0 (or 0 to 1000) containing the question text, diagram, and options.
   - option_boxes: Object mapping each option letter to the normalized bounding box [ymin, xmin, ymax, xmax] of that single option, including its letter bubble. Options are often pictures (graphs, field-line figures, circuits): box the whole picture with its letter. Use the same coordinate range as question_box.
   - student_working_box: Normalized bounding box [ymin, xmin, ymax, xmax] of the student's pencil working/calculations in the margin or under the question (or null if none).
   - working_ocr_text: Transcribed text of the student's handwritten pencil working (or empty string).
   - working_ocr_confidence: A float between 0.0 and 1.0 indicating OCR confidence in the pencil handwriting (e.g. 0.95 for clear writing, 0.55 for faint/unclear scribbles).
   - concept_tag: The core physics concept in Arabic (e.g. "عزم الازدواج المغناطيسي", "القوة المغناطيسية على سلك", "كثافة الفيض المغناطيسي لملف دائري").
   - formula: Primary formula needed to solve this problem, formatted in LaTeX (e.g. "\\tau = B I A N \\sin\\theta").
   - parameters: Key physical variables and values given in the question as a dictionary (e.g. {"B": "0.4 T", "I": "2 A"}).
   - correct_choice: The scientifically correct option ("أ", "ب", "ج", "د").
   - solution_arabic: Step-by-step Arabic explanation and calculation explaining why correct_choice is correct.
4. Also report page_rotation_clockwise: the number of degrees (0, 90, 180 or 270) this photo must be rotated CLOCKWISE for its text to read upright. Use 0 if the text already reads upright.

OUTPUT REQUIREMENT:
Return ONLY a valid JSON object of the form {"page_rotation_clockwise": 0, "questions": [ ...question objects... ]}. No markdown formatting outside of JSON.
"""


def encode_image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Encode PIL Image to base64 JPEG string."""
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        rgb_image = image.convert("RGB")
    else:
        rgb_image = image
    rgb_image.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _quarter_turn(value: Any) -> Optional[int]:
    """Coerce a model-reported rotation to 0/90/180/270, or None if unusable."""
    try:
        turn = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return None
    return turn if turn in (0, 90, 180, 270) else None


def _mock_result(reason: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return MOCK_PHYSICS_QUESTIONS, {"mock": True, "reason": reason}


# Fields of a verified question that are worth showing the model as an example.
EXAMPLE_FIELDS = (
    "question_number", "options", "question_box", "option_boxes", "student_marks",
    "answer_sequence", "student_choice", "correct_choice", "concept_tag", "formula",
)
MAX_EXAMPLES = 3


def build_example_parts(examples: Optional[List[Tuple[Image.Image, List[Dict[str, Any]]]]]) -> List[Dict[str, Any]]:
    """Turn verified (page image, questions) pairs into few-shot request parts.

    The examples are pages a human has already corrected, so the model sees what
    a right reading looks like: where the boxes go, which marks count as the
    student's, and how crossed-out answers are reported.
    """
    parts: List[Dict[str, Any]] = []
    for n, (img, questions) in enumerate((examples or [])[:MAX_EXAMPLES], start=1):
        small = img.copy()
        small.thumbnail((1000, 1000))
        trimmed = [{k: q[k] for k in EXAMPLE_FIELDS if k in q} for q in questions]
        parts.append({"text": f"EXAMPLE {n}: a page and its human-verified reading. Follow the same conventions."})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": encode_image_to_base64(small, quality=80)}})
        parts.append({"text": json.dumps({"page_rotation_clockwise": 0, "questions": trimmed}, ensure_ascii=False)})
    if parts:
        parts.append({"text": "Now read the following NEW page the same way and return only the JSON object."})
    return parts


def call_gemini_page(
    image: Image.Image,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
    force_mock: bool = False,
    examples: Optional[List[Tuple[Image.Image, List[Dict[str, Any]]]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Send a page photo to Gemini Vision and return (questions, meta).

    meta may contain:
        page_rotation_clockwise  degrees the model says the photo must still be turned
        mock / reason            set when the offline mock dataset was used instead of
                                 a real reading, together with the reason why
    """
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")

    if force_mock:
        return _mock_result("--mock was requested")

    if not effective_key:
        return _mock_result("GEMINI_API_KEY is not set")

    url = API_URL_TEMPLATE.format(model=model)
    base64_data = encode_image_to_base64(image)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": EXTRACTION_SYSTEM_PROMPT},
                    *build_example_parts(examples),
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1
        }
    }

    headers = {"Content-Type": "application/json", "x-goog-api-key": effective_key}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            # If the primary model fails (e.g. 404 on model name), try gemini-1.5-flash
            if model != "gemini-1.5-flash" and response.status_code in (404, 400):
                fallback_url = API_URL_TEMPLATE.format(model="gemini-1.5-flash")
                response = requests.post(fallback_url, headers=headers, json=payload, timeout=timeout)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Gemini API request failed with status {response.status_code}: {response.text}"
                )

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates returned from Gemini API: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"Empty parts returned from Gemini API: {candidates}")

        text = parts[0].get("text", "")
        # Clean any potential markdown code blocks like ```json ... ```
        cleaned_text = re.sub(r'^```json\s*', '', text.strip())
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text.strip())

        parsed = json.loads(cleaned_text)
        meta: Dict[str, Any] = {}
        if isinstance(parsed, dict):
            meta["page_rotation_clockwise"] = _quarter_turn(parsed.get("page_rotation_clockwise"))
            questions = parsed.get("questions", [])
        else:
            questions = parsed

        if not isinstance(questions, list) or len(questions) == 0:
            # If the image was unreadable or returned no questions, fall back
            return _mock_result("Gemini returned no questions")

        # Standardize question IDs and normalized fields
        questions = [q for q in questions if isinstance(q, dict)]
        for idx, q in enumerate(questions, start=1):
            if "id" not in q:
                q["id"] = idx
            for field in ("student_choice", "correct_choice"):
                value = q.get(field)
                q[field] = normalize_option_letter(value) if isinstance(value, str) else None
            if not isinstance(q.get("student_marks"), dict):
                q.pop("student_marks", None)
            if not isinstance(q.get("option_boxes"), dict):
                q.pop("option_boxes", None)

        return questions, meta

    except Exception as exc:
        # If the API call fails (network issue, rate limit, invalid key, or invalid image
        # content) keep the pipeline alive with the mock data, but say why so a wrong
        # result is never mistaken for a real reading.
        return _mock_result(f"{type(exc).__name__}: {str(exc).replace(effective_key, '***')[:300]}")


def call_gemini_vision(
    image: Image.Image,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
    force_mock: bool = False
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper around call_gemini_page that returns only the questions."""
    questions, _meta = call_gemini_page(
        image, api_key=api_key, model=model, timeout=timeout, force_mock=force_mock
    )
    return questions
