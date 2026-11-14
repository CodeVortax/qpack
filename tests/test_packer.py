"""Unit tests for .qpack archive creation and loading."""

import os
import tempfile
from PIL import Image
from qpack.packer import create_qpack, load_qpack, extract_image_from_qpack
from qpack.mock_data import MOCK_PHYSICS_QUESTIONS


def test_create_and_load_qpack():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_pack_path = os.path.join(tmpdir, "test.qpack")
        test_img = Image.new("RGB", (800, 1000), color=(255, 255, 255))

        saved_path = create_qpack(
            source_image_path="page1.jpg",
            deskewed_image=test_img,
            deskew_angle=1.5,
            questions=MOCK_PHYSICS_QUESTIONS,
            output_qpack_path=test_pack_path
        )

        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0

        # Load and verify
        manifest, questions = load_qpack(saved_path)
        assert manifest["qpack_version"] == "1.0"
        assert manifest["total_questions"] == len(MOCK_PHYSICS_QUESTIONS)
        assert manifest["deskew_angle_degrees"] == 1.5
        assert manifest["source_image"] == "page1.jpg"
        assert len(manifest["concepts"]) >= 2

        # Verify questions structure
        assert len(questions) == len(MOCK_PHYSICS_QUESTIONS)
        q1 = questions[0]
        assert q1["id"] == 35
        assert "question_image_path" in q1
        assert q1["question_image_path"] == "images/q35_question.png"
        assert "working_image_path" in q1
        assert q1["working_image_path"] == "images/q35_working.png"

        # Verify image extraction
        extracted_q_img = extract_image_from_qpack(saved_path, q1["question_image_path"])
        assert extracted_q_img is not None
        assert extracted_q_img.size[0] > 0
