"""End-to-end tests for CLI commands."""

import argparse
import os
import subprocess
import sys
import tempfile
from PIL import Image

from qpack.mock_data import MOCK_PHYSICS_QUESTIONS
from qpack.packer import load_qpack


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
        # The mock data contains a changed answer, which show must explain
        assert "Answer changed" in res_show.stdout
        assert "crossed out" in res_show.stdout

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
        assert "changed" in res_check.stdout


def _scan_args(tmpdir, **overrides):
    path = os.path.join(tmpdir, "page.jpg")
    Image.new("RGB", (600, 800), color=(240, 240, 240)).save(path)
    values = dict(
        image=path,
        output=os.path.join(tmpdir, "page.qpack"),
        api_key="test-key",
        model="test-model",
        mock=False,
        ai=True,
        rotate="auto",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_scan_reads_again_when_gemini_says_the_page_is_rotated(monkeypatch):
    from qpack import cli

    calls = []

    def fake_call(image, api_key=None, model="m", timeout=60, force_mock=False):
        calls.append(image.size)
        turn = 180 if len(calls) == 1 else 0
        return MOCK_PHYSICS_QUESTIONS, {"page_rotation_clockwise": turn}

    monkeypatch.setattr(cli, "call_gemini_page", fake_call)
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _scan_args(tmpdir)
        assert cli.handle_scan(args) == 0
        manifest, _ = load_qpack(args.output)

    assert len(calls) == 2
    assert abs(abs(manifest["deskew_angle_degrees"]) - 180.0) < 1.0


def test_scan_does_not_reread_when_rotation_is_forced(monkeypatch):
    from qpack import cli

    calls = []

    def fake_call(image, api_key=None, model="m", timeout=60, force_mock=False):
        calls.append(image.size)
        return MOCK_PHYSICS_QUESTIONS, {"page_rotation_clockwise": 90}

    monkeypatch.setattr(cli, "call_gemini_page", fake_call)
    with tempfile.TemporaryDirectory() as tmpdir:
        assert cli.handle_scan(_scan_args(tmpdir, rotate="0")) == 0
    assert len(calls) == 1


def test_scan_warns_loudly_when_it_falls_back_to_mock_data(monkeypatch, capsys):
    from qpack import cli

    def fake_call(image, api_key=None, model="m", timeout=60, force_mock=False):
        return MOCK_PHYSICS_QUESTIONS, {"mock": True, "reason": "HTTP 429 quota exceeded"}

    monkeypatch.setattr(cli, "call_gemini_page", fake_call)
    with tempfile.TemporaryDirectory() as tmpdir:
        assert cli.handle_scan(_scan_args(tmpdir)) == 0
    out = capsys.readouterr().out
    assert "MOCK" in out
    assert "429" in out


def test_scan_defaults_to_the_local_reader_without_any_key(monkeypatch):
    from qpack import cli
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "call_gemini_page", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no AI")))
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _scan_args(tmpdir)
        args.ai = False
        assert cli.handle_scan(args) == 0
