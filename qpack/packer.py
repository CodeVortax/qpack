"""Creation and extraction of .qpack archive files (ZIP format)."""

from __future__ import annotations
from datetime import datetime, timezone
import io
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import zipfile
from PIL import Image

from qpack.deskew import crop_box
from qpack.analyzer import check_exam

QPACK_FORMAT_VERSION = "1.0"


def create_qpack(
    source_image_path: str,
    deskewed_image: Image.Image,
    deskew_angle: float,
    questions: List[Dict[str, Any]],
    output_qpack_path: str
) -> str:
    """Package questions, manifest, and cropped images into a .qpack ZIP file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_qpack_path)), exist_ok=True)

    # Process crops and image paths
    updated_questions: List[Dict[str, Any]] = []
    image_entries: Dict[str, bytes] = {}

    # Store full deskewed image
    deskewed_buffer = io.BytesIO()
    deskewed_image.save(deskewed_buffer, format="PNG")
    image_entries["images/page_deskewed.png"] = deskewed_buffer.getvalue()

    for idx, q in enumerate(questions, start=1):
        q_copy = dict(q)
        q_id = q_copy.get("id", idx)
        q_copy["id"] = q_id

        # 1. Question region crop
        q_box = q_copy.get("question_box")
        q_img_rel_path = f"images/q{q_id}_question.png"
        if q_box and len(q_box) == 4:
            cropped_q = crop_box(deskewed_image, q_box)
        else:
            # Fallback to full page if no box provided
            cropped_q = deskewed_image.copy()

        buf_q = io.BytesIO()
        cropped_q.save(buf_q, format="PNG")
        image_entries[q_img_rel_path] = buf_q.getvalue()
        q_copy["question_image_path"] = q_img_rel_path

        # 2. Student pencil working crop
        w_box = q_copy.get("student_working_box")
        if w_box and len(w_box) == 4:
            cropped_w = crop_box(deskewed_image, w_box)
            w_img_rel_path = f"images/q{q_id}_working.png"
            buf_w = io.BytesIO()
            cropped_w.save(buf_w, format="PNG")
            image_entries[w_img_rel_path] = buf_w.getvalue()
            q_copy["working_image_path"] = w_img_rel_path
        else:
            q_copy["working_image_path"] = None

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
