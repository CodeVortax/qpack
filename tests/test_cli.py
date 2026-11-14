"""End-to-end tests for CLI commands."""

import os
import subprocess
import sys
import tempfile
from PIL import Image


def test_cli_scan_show_check():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test input image
        input_img_path = os.path.join(tmpdir, "page_test.jpg")
        img = Image.new("RGB", (600, 800), color=(240, 240, 240))
        img.save(input_img_path)

        pack_path = os.path.join(tmpdir, "page_test.qpack")

        # 1. Test qpack scan --mock
        cmd_scan = [
            sys.executable, "-m", "qpack.cli", "scan",
            input_img_path, "-o", pack_path, "--mock"
        ]
        res_scan = subprocess.run(cmd_scan, capture_output=True, text=True)
        assert res_scan.returncode == 0
        assert os.path.exists(pack_path)
        assert "Successfully generated .qpack archive" in res_scan.stdout

        # 2. Test qpack show
        cmd_show = [
            sys.executable, "-m", "qpack.cli", "show",
            pack_path, "--group-by", "concept"
        ]
        res_show = subprocess.run(cmd_show, capture_output=True, text=True)
        assert res_show.returncode == 0
        assert "QPACK INSPECTION" in res_show.stdout
        assert "نقطة التعادل بين سلكين مستقيمين" in res_show.stdout

        # 3. Test qpack check
        cmd_check = [
            sys.executable, "-m", "qpack.cli", "check",
            pack_path, "--threshold", "0.70"
        ]
        res_check = subprocess.run(cmd_check, capture_output=True, text=True)
        assert res_check.returncode == 0
        assert "Exam Results & Answer Verification" in res_check.stdout
        # Verify status indicators and review flag appear
        assert "RIGHT" in res_check.stdout
        assert "WRONG" in res_check.stdout
        assert "Low pencil OCR" in res_check.stdout
        assert "flagged for manual review" in res_check.stdout
