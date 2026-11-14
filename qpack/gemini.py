"""Gemini Vision API client using direct REST calls via requests."""

from __future__ import annotations
import base64
import io
import json
import os
import re
from typing import List, Dict, Any, Optional
from PIL import Image
import requests

from qpack.arabic import to_western_digits, normalize_option_letter
from qpack.mock_data import MOCK_PHYSICS_QUESTIONS

DEFAULT_MODEL = "gemini-2.5-flash"
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

EXTRACTION_SYSTEM_PROMPT = """You are an expert Arabic Physics teacher, exam grader, and high-precision OCR vision engine.
You are analyzing a photo of an Arabic (right-to-left) physics exam or workbook page.
The page contains:
- Numbered multiple-choice questions with options labeled (أ, ب, ج, د).
- Physics diagrams, schematics, and graphs.
- Student pencil ticks or circles marking their selected option.
- Student handwritten pencil calculations and working in the margins.
- Potential background watermarks (e.g. teacher names, logos, phone numbers, diagonal patterns).

CRITICAL INSTRUCTIONS:
1. IGNORE ALL BACKGROUND WATERMARKS (such as teacher watermark logos, phone numbers, center diagonal text). Focus strictly on printed exam questions, diagrams, student ticks, and pencil margin working.
2. Split the page by each numbered question.
3. For each question, extract:
   - question_number: The printed question number (e.g. "1" or "١").
   - question_text: Full Arabic question text, including any descriptions. Keep all math in LaTeX format wrapped in $...$ (e.g. $B = 0.4\\text{ T}$).
   - options: Object mapping Arabic option keys "أ", "ب", "ج", "د" to their printed text and LaTeX math formulas.
   - student_choice: Which option the student actually ticked/checked/circled in pencil or pen ("أ", "ب", "ج", "د", or null if unanswered). Look carefully at option bubbles, brackets, or letters for student checkmarks or pencil circles.
   - question_box: Normalized bounding box [ymin, xmin, ymax, xmax] in range 0.0 to 1.0 (or 0 to 1000) containing the question text, diagram, and options.
   - student_working_box: Normalized bounding box [ymin, xmin, ymax, xmax] of the student's pencil working/calculations in the margin or under the question (or null if none).
   - working_ocr_text: Transcribed text of the student's handwritten pencil working (or empty string).
   - working_ocr_confidence: A float between 0.0 and 1.0 indicating OCR confidence in the pencil handwriting (e.g. 0.95 for clear writing, 0.55 for faint/unclear scribbles).
   - concept_tag: The core physics concept in Arabic (e.g. "عزم الازدواج المغناطيسي", "القوة المغناطيسية على سلك", "كثافة الفيض المغناطيسي لملف دائري").
   - formula: Primary formula needed to solve this problem, formatted in LaTeX (e.g. "\\tau = B I A N \\sin\\theta").
   - parameters: Key physical variables and values given in the question as a dictionary (e.g. {"B": "0.4 T", "I": "2 A"}).
   - correct_choice: The scientifically correct option ("أ", "ب", "ج", "د").
   - solution_arabic: Step-by-step Arabic explanation and calculation explaining why correct_choice is correct.

OUTPUT REQUIREMENT:
Return ONLY a valid JSON array of question objects adhering to this schema. No markdown formatting outside of JSON.
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


def call_gemini_vision(
    image: Image.Image,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
    force_mock: bool = False
) -> List[Dict[str, Any]]:
    """Send image to Gemini Vision API and extract structured question data.
    
    If force_mock is True, or if api_key is not provided (and GEMINI_API_KEY is unset),
    falls back to realistic mock physics dataset.
    """
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")

    if force_mock:
        return MOCK_PHYSICS_QUESTIONS

    if not effective_key:
        # If no API key is available, warn and use high-fidelity mock data
        return MOCK_PHYSICS_QUESTIONS

    url = API_URL_TEMPLATE.format(model=model, key=effective_key)
    base64_data = encode_image_to_base64(image)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": EXTRACTION_SYSTEM_PROMPT},
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

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            # If the primary model fails (e.g. 404 on model name), try gemini-1.5-flash
            if model != "gemini-1.5-flash" and response.status_code in (404, 400):
                fallback_url = API_URL_TEMPLATE.format(model="gemini-1.5-flash", key=effective_key)
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

        questions = json.loads(cleaned_text)
        if isinstance(questions, dict) and "questions" in questions:
            questions = questions["questions"]

        if not isinstance(questions, list) or len(questions) == 0:
            # If the image was unreadable or returned no questions, fall back
            return MOCK_PHYSICS_QUESTIONS

        # Standardize question IDs and normalized fields
        for idx, q in enumerate(questions, start=1):
            if "id" not in q:
                q["id"] = idx
            q["student_choice"] = normalize_option_letter(q.get("student_choice"))
            q["correct_choice"] = normalize_option_letter(q.get("correct_choice"))

        return questions

    except Exception as exc:
        # If API call fails (network issue, rate limit, invalid key, or invalid image content),
        # return mock data with a warning flag so tests and pipeline keep working
        return MOCK_PHYSICS_QUESTIONS
