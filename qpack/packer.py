"""Creation and extraction of .qpack archive files (ZIP format)."""

from __future__ import annotations
from datetime import datetime, timezone
import io
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import zipfile
from PIL import Image

from qpack.arabic import normalize_option_letter
from qpack.deskew import crop_box
from qpack.analyzer import check_exam, resolve_question

QPACK_FORMAT_VERSION = "1.0"


def create_qpack(
    source_image_path: str,
    deskewed_image: Image.Image,
    deskew_angle: float,
    questions: List[Dict[str, Any]],
    output_qpack_path: str,
    crop_format: str = "PNG",
) -> str:
    """Package questions, manifest, and cropped images into a .qpack ZIP file.

    ``crop_format`` is "PNG" (lossless, the original behaviour) or "JPEG"
    (about 10x smaller for photos). Every crop in the pack uses it.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_qpack_path)), exist_ok=True)

    fmt = crop_format.upper()
    if fmt not in ("PNG", "JPEG"):
        raise ValueError("crop_format must be 'PNG' or 'JPEG'")
    ext = "png" if fmt == "PNG" else "jpg"

    def encode(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        if fmt == "JPEG":
            img.convert("RGB").save(buf, format="JPEG", quality=85)
        else:
            img.save(buf, format="PNG")
        return buf.getvalue()

    # Process crops and image paths
    updated_questions: List[Dict[str, Any]] = []
    image_entries: Dict[str, bytes] = {}

    # Store full deskewed image
    image_entries[f"images/page_deskewed.{ext}"] = encode(deskewed_image)

    for idx, q in enumerate(questions, start=1):
        q_copy = dict(q)
        q_id = q_copy.get("id", idx)
        q_copy["id"] = q_id

        # Settle the final answer from the per-option marks (crossed-out and changed
        # answers included). The model's own guess is kept next to it for cross-checking.
        if q_copy.get("student_marks"):
            q_copy["student_choice_reported"] = q_copy.get("student_choice")
            q_copy["student_choice"] = resolve_question(q_copy)["final"]

        # 1. Question region crop
        q_box = q_copy.get("question_box")
        q_img_rel_path = f"images/q{q_id}_question.{ext}"
        if q_box and len(q_box) == 4:
            cropped_q = crop_box(deskewed_image, q_box)
        else:
            # Fallback to full page if no box provided
            cropped_q = deskewed_image.copy()

        image_entries[q_img_rel_path] = encode(cropped_q)
        q_copy["question_image_path"] = q_img_rel_path

        # 2. Student pencil working crop
        w_box = q_copy.get("student_working_box")
        if w_box and len(w_box) == 4:
            cropped_w = crop_box(deskewed_image, w_box)
            w_img_rel_path = f"images/q{q_id}_working.{ext}"
            image_entries[w_img_rel_path] = encode(cropped_w)
            q_copy["working_image_path"] = w_img_rel_path
        else:
            q_copy["working_image_path"] = None

        # 3. One crop per answer option. This is what makes answers that are
        #    pictures (graphs, field-line figures, circuits) usable.
        option_paths: Dict[str, str] = {}
        raw_boxes = q_copy.get("option_boxes")
        if isinstance(raw_boxes, dict):
            for raw_key, box in raw_boxes.items():
                opt = normalize_option_letter(raw_key) if isinstance(raw_key, str) else None
                if not opt or not box or len(box) != 4:
                    continue
                rel = f"images/q{q_id}_option_{len(option_paths) + 1}.{ext}"
                image_entries[rel] = encode(crop_box(deskewed_image, box, padding_pct=0.01))
                option_paths[opt] = rel
        q_copy["option_image_paths"] = option_paths

        updated_questions.append(q_copy)

    # Generate exam evaluation summary
    exam_summary = check_exam(updated_questions)

    # Unique concepts and formulas
    unique_concepts = sorted(list({q.get("concept_tag") for q in updated_questions if q.get("concept_tag")}))
    unique_formulas = sorted(list({q.get("formula") for q in updated_questions if q.get("formula")}))

    manifest = {
        "qpack_version": QPACK_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image": os.path.basename(source_image_path),
        "total_questions": len(updated_questions),
        "deskew_angle_degrees": deskew_angle,
        "concepts": unique_concepts,
        "formulas": unique_formulas,
        "summary": {
            "score_percent": exam_summary["score_percent"],
            "correct": exam_summary["correct"],
            "wrong": exam_summary["wrong"],
            "unanswered": exam_summary["unanswered"],
            "ambiguous": exam_summary["ambiguous"],
            "ungraded": exam_summary["ungraded"],
            "changed_answers": exam_summary["changed_answers"],
            "flagged_for_review": exam_summary["flagged_for_review"]
        }
    }

    # Write ZIP archive
    with zipfile.ZipFile(output_qpack_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write manifest.json
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        zf.writestr("manifest.json", manifest_bytes)

        # Write questions.json
        questions_bytes = json.dumps(updated_questions, ensure_ascii=False, indent=2).encode("utf-8")
        zf.writestr("questions.json", questions_bytes)

        # Write image crops
        for rel_path, img_bytes in image_entries.items():
            zf.writestr(rel_path, img_bytes)

    return output_qpack_path


def load_qpack(qpack_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read and validate the manifest and questions from a .qpack file."""
    if not os.path.exists(qpack_path):
        raise FileNotFoundError(f"Pack file not found: {qpack_path}")

    with zipfile.ZipFile(qpack_path, mode="r") as zf:
        namelist = zf.namelist()
        if "manifest.json" not in namelist or "questions.json" not in namelist:
            raise ValueError(f"Invalid .qpack format. Missing manifest.json or questions.json.")

        manifest_raw = zf.read("manifest.json").decode("utf-8")
        manifest = json.loads(manifest_raw)

        questions_raw = zf.read("questions.json").decode("utf-8")
        questions = json.loads(questions_raw)

    return manifest, questions


def extract_image_from_qpack(qpack_path: str, image_rel_path: str) -> Image.Image:
    """Extract and open a specific image from inside a .qpack file."""
    with zipfile.ZipFile(qpack_path, mode="r") as zf:
        if image_rel_path not in zf.namelist():
            raise FileNotFoundError(f"Image '{image_rel_path}' not found in archive.")
        img_bytes = zf.read(image_rel_path)
        return Image.open(io.BytesIO(img_bytes))
