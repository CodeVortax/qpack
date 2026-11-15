"""CLI Interface for qpack: scan, show, and check commands."""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional

from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from qpack.deskew import deskew_image, rotate_clockwise
from qpack.gemini import call_gemini_page
from qpack.vision import read_page
from qpack.packer import create_qpack, load_qpack
from qpack.analyzer import check_exam, group_questions, resolve_question
from qpack.arabic import to_arabic_indic_digits, to_western_digits, reshape_arabic

console = Console()


def handle_scan(args: argparse.Namespace) -> int:
    """Execute 'qpack scan <image> -o <output.qpack>'."""
    input_path = args.image
    output_path = args.output

    if not os.path.exists(input_path):
        console.print(f"[bold red]Error:[/bold red] Image file not found: [yellow]{input_path}[/yellow]")
        return 1

    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.qpack"

    console.print(f"[bold cyan]🔍 Scanning exam page:[/bold cyan] {input_path}")

    # 1. Load image
    try:
        raw_img = Image.open(input_path)
    except Exception as exc:
        console.print(f"[bold red]Error opening image:[/bold red] {exc}")
        return 1

    # 2. Orient (sideways photos) and deskew
    console.print("[dim]Step 1/4: Orienting and deskewing photo...[/dim]")
    forced_rotation = None if args.rotate == "auto" else int(args.rotate)
    deskewed_img, angle = deskew_image(raw_img, rotate=forced_rotation)
    if abs(angle) >= 0.2:
        console.print(f"  [green]✓[/green] Rotated/deskewed image by [bold]{angle:+.2f}°[/bold]")
    else:
        console.print(f"  [dim]✓ Image orientation is upright (skew angle: {angle:.2f}°)[/dim]")

    # 3. Read the page. Local image processing by default; Gemini only with --ai.
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    used_mock = False
    if not args.ai and not args.mock:
        console.print("[dim]Step 2/4: Finding questions, options and marks on this device (no AI)...[/dim]")
        questions, meta = read_page(deskewed_img)
        console.print(f"  [dim]Split by: {meta.get('split_by')}. Correct answers are not known locally: "
                      "set them with the studio or edit the pack.[/dim]")
    else:
        console.print("[dim]Step 2/4: Reading the page with Gemini...[/dim]")
        try:
            questions, meta = call_gemini_page(deskewed_img, api_key=api_key, model=args.model, force_mock=args.mock)
            turn = meta.get("page_rotation_clockwise")
            if turn and args.rotate == "auto" and not meta.get("mock"):
                console.print(f"  [yellow]Gemini says the page is still {turn}° off; rotating and reading again.[/yellow]")
                turn_as_pil = float((-turn) % 360)
                if turn_as_pil > 180:
                    turn_as_pil -= 360
                deskewed_img, extra_angle = deskew_image(rotate_clockwise(deskewed_img, turn), rotate=0)
                angle += turn_as_pil + extra_angle
                questions, meta = call_gemini_page(deskewed_img, api_key=api_key, model=args.model, force_mock=args.mock)
        except Exception as exc:
            console.print(f"[bold red]Extraction failed:[/bold red] {exc}")
            return 1
        used_mock = bool(meta.get("mock"))
        if used_mock and not args.mock:
            console.print("  [bold red]⚠ The AI reading failed, so MOCK data is shown. Nothing comes from your photo.[/bold red]")
            console.print(f"  [dim]Reason: {meta.get('reason')}[/dim]")

    console.print(f"  [green]✓[/green] Extracted [bold]{len(questions)}[/bold] questions from page.")

    # 4. Package into .qpack
    console.print(f"[dim]Step 3/4: Cropping question regions & packaging archive...[/dim]")
    try:
        saved_pack = create_qpack(
            source_image_path=input_path,
            deskewed_image=deskewed_img,
            deskew_angle=angle,
            questions=questions,
            output_qpack_path=output_path
        )
    except Exception as exc:
        console.print(f"[bold red]Packaging failed:[/bold red] {exc}")
        return 1

    # 5. Summary display
    console.print(f"[dim]Step 4/4: Complete![/dim]\n")
    console.print(Panel(
        f"[bold green]✓ Successfully generated .qpack archive:[/bold green]\n"
        f"[bold white]{saved_pack}[/bold white]\n"
        f"[dim]Total questions: {len(questions)} | Angle: {angle:+.2f}°[/dim]"
        + ("\n[bold red]MOCK DATA: not read from your photo[/bold red]" if used_mock else ""),
        border_style="green"
    ))
    return 0


def handle_show(args: argparse.Namespace) -> int:
    """Execute 'qpack show <file.qpack>'."""
    pack_path = args.qpack

    try:
        manifest, questions = load_qpack(pack_path)
    except Exception as exc:
        console.print(f"[bold red]Error reading .qpack:[/bold red] {exc}")
        return 1

    group_by = args.group_by
    grouped = group_questions(questions, by=group_by)

    # Header Panel
    console.print(Panel(
        f"[bold cyan]📦 QPACK INSPECTION:[/bold cyan] [bold white]{os.path.basename(pack_path)}[/bold white]\n"
        f"[dim]Created:[/dim] {manifest.get('created_at', 'N/A')}  |  "
        f"[dim]Source Image:[/dim] {manifest.get('source_image', 'N/A')}  |  "
        f"[dim]Deskew Angle:[/dim] {manifest.get('deskew_angle_degrees', 0.0):+.2f}°\n"
        f"[dim]Questions:[/dim] [bold]{manifest.get('total_questions', len(questions))}[/bold]  |  "
        f"[dim]Grouped By:[/dim] [bold yellow]{group_by.upper()}[/bold yellow]",
        border_style="cyan"
    ))

    for group_name, q_list in grouped.items():
        console.print(f"\n[bold magenta]━━━ {group_name} ({len(q_list)} question{'s' if len(q_list)>1 else ''}) ━━━[/bold magenta]")
        
        for q in q_list:
            q_num = q.get("question_number", q.get("id"))
            q_text = q.get("question_text", "")
            concept = q.get("concept_tag", "عام")
            formula = q.get("formula", "")
            params = q.get("parameters", {})
            ans = resolve_question(q)
            correct = q.get("correct_choice") or "?"
            conf = q.get("working_ocr_confidence", 1.0)
            solution = q.get("solution_arabic", "")

            if ans["status"] == "AMBIGUOUS":
                student = "?(" + "/".join(ans["candidates"]) + ")"
                status_style, mark = "yellow", "?"
            else:
                student = ans["final"] or "[dim]None[/dim]"
                status_style = "green" if student == correct else "red"
                mark = "✓" if student == correct else "✗"

            console.print(f"\n[bold yellow]► Question {q_num}:[/bold yellow] [bold]{concept}[/bold]  [{status_style}][Student: {student} | Correct: {correct} {mark}][/{status_style}]")
            if formula:
                console.print(f"  [cyan]Formula:[/cyan] [italic]{formula}[/italic]")
            if params:
                param_str = ", ".join(f"{k} = {v}" for k, v in params.items())
                console.print(f"  [cyan]Parameters:[/cyan] {param_str}")

            console.print(f"  [white]{q_text}[/white]")

            # Options
            opts = q.get("options", {})
            for opt_key, opt_val in opts.items():
                is_st = (opt_key == student)
                is_cor = (opt_key == correct)
                flag = ""
                if is_cor:
                    flag = " [bold green](Correct Answer)[/bold green]"
                if is_st:
                    flag += " [bold cyan]◄ Your Choice[/bold cyan]"
                if opt_key in ans["crossed_out"]:
                    flag += " [dim]✗ crossed out[/dim]"
                console.print(f"    [bold]({opt_key})[/bold] {opt_val}{flag}")

            # How the final answer was worked out
            if ans["changed"]:
                console.print(
                    f"  [cyan]Answer changed:[/cyan] crossed out {'، '.join(ans['crossed_out'])} "
                    f"→ final {ans['final']}"
                )
            for note in ans["notes"]:
                console.print(f"  [dim]{note}[/dim]")
            for warning in ans["flags"]:
                console.print(f"  [yellow]⚠ {warning}[/yellow]")

            # Margin working
            working_text = q.get("working_ocr_text", "")
            if working_text:
                conf_color = "green" if conf >= 0.70 else "red"
                console.print(f"  [dim]Pencil Working (Confidence: [{conf_color}]{conf:.0%}[/{conf_color}]):[/dim]")
                for line in working_text.strip().split("\n"):
                    console.print(f"    [italic dim]{line}[/italic dim]")

            # Solution
            if solution:
                console.print(f"  [bold green]💡 Step-by-Step Arabic Solution:[/bold green]")
                for line in solution.strip().split("\n"):
                    console.print(f"    {line}")

    return 0


def handle_check(args: argparse.Namespace) -> int:
    """Execute 'qpack check <file.qpack>'."""
    pack_path = args.qpack

    try:
        manifest, questions = load_qpack(pack_path)
    except Exception as exc:
        console.print(f"[bold red]Error reading .qpack:[/bold red] {exc}")
        return 1

    threshold = args.threshold
    exam_res = check_exam(questions, confidence_threshold=threshold)

    # Render results table
    table = Table(
        title=f"Exam Results & Answer Verification: {os.path.basename(pack_path)}",
        show_header=True,
        header_style="bold cyan",
        border_style="dim"
    )

    table.add_column("Q#", justify="center", style="bold white", width=4)
    table.add_column("Concept", justify="right", style="cyan", width=26)
    table.add_column("My Choice", justify="center", width=11)
    table.add_column("Correct", justify="center", style="bold green", width=9)
    table.add_column("Status", justify="center", width=12)
    table.add_column("OCR Conf", justify="center", width=10)
    table.add_column("Review Flag", justify="left", width=30)

    for item in exam_res["results"]:
        q_num = str(item.get("question_number", item.get("id")))
        concept = str(item.get("concept_tag", "عام"))
        if item.get("status") == "AMBIGUOUS":
            my_choice = "/".join(item.get("candidates", [])) or "?"
        else:
            my_choice = str(item.get("student_choice") or "-")
        correct = str(item.get("correct_choice") or "-")
        status = item.get("status")
        conf = item.get("working_ocr_confidence", 1.0)
        conf_str = f"{conf:.0%}"

        if status == "CORRECT":
            status_text = "[bold green]✓ RIGHT[/bold green]"
        elif status == "WRONG":
            status_text = "[bold red]✗ WRONG[/bold red]"
        elif status == "AMBIGUOUS":
            status_text = "[bold yellow]? AMBIGUOUS[/bold yellow]"
        elif status == "UNGRADED":
            status_text = "[dim]- NO KEY[/dim]"
        else:
            status_text = "[yellow]- SKIPPED[/yellow]"

        conf_styled = f"[green]{conf_str}[/green]" if conf >= threshold else f"[bold red]{conf_str}[/bold red]"

        change_str = ""
        if item.get("changed_answer"):
            change_str = f"[cyan]↺ changed: {'، '.join(item['crossed_out'])} ✗ → {item['student_choice']}[/cyan]"

        if item.get("needs_review"):
            flag_str = "[bold red]⚠️  " + "; ".join(item.get("review_flags", [])) + "[/bold red]"
            if change_str:
                flag_str += " " + change_str
        elif change_str:
            flag_str = "[green]✓[/green] " + change_str
        else:
            flag_str = "[green]✓ Verified[/green]"

        table.add_row(
            q_num,
            concept,
            my_choice,
            correct,
            status_text,
            conf_styled,
            flag_str
        )

    console.print("\n")
    console.print(table)

    # Summary Panel
    score = exam_res["score_percent"]
    color = "green" if score >= 80 else ("yellow" if score >= 50 else "red")

    summary_text = (
        f"Score: [{color}][bold]{score}%[/bold][/{color}]  "
        f"({exam_res['correct']}/{exam_res['total']} Correct)  |  "
        f"[red]{exam_res['wrong']} Wrong[/red]  |  "
        f"[yellow]{exam_res['unanswered']} Unanswered[/yellow]"
    )
    if exam_res["ambiguous"]:
        summary_text += f"  |  [yellow]{exam_res['ambiguous']} Ambiguous[/yellow]"
    if exam_res["ungraded"]:
        summary_text += f"  |  [dim]{exam_res['ungraded']} without answer key[/dim]"
    summary_text += "\n"
    if exam_res["changed_answers"]:
        summary_text += f"[cyan]↺ {exam_res['changed_answers']} answer(s) changed by the student (crossed out, then chose again).[/cyan]\n"
    if exam_res["flagged_for_review"] > 0:
        summary_text += f"[bold red]⚠️  {exam_res['flagged_for_review']} question(s) flagged for manual review due to low OCR confidence (< {threshold:.0%}), unclear marks or missing answers.[/bold red]"
    else:
        summary_text += "[bold green]✓ All pencil handwriting readings passed confidence threshold.[/bold green]"

    console.print(Panel(summary_text, title="📊 Evaluation Summary", border_style=color))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CLI commands."""
    parser = argparse.ArgumentParser(
        prog="qpack",
        description="qpack: Exam & Homework Page Scanner with Arabic Physics Support"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. scan command
    scan_parser = subparsers.add_parser("scan", help="Scan an exam page photo and generate a .qpack archive")
    scan_parser.add_argument("image", help="Path to input photo (e.g. page1.jpg)")
    scan_parser.add_argument("-o", "--output", help="Path to output .qpack file (default: <image_stem>.qpack)")
    scan_parser.add_argument("--api-key", help="Gemini API Key (overrides GEMINI_API_KEY env var)")
    scan_parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini vision model name")
    scan_parser.add_argument("--ai", action="store_true", help="Read the page with Gemini instead of the local reader")
    scan_parser.add_argument("--mock", action="store_true", help="Force offline mock data for testing")
    scan_parser.add_argument(
        "--rotate",
        choices=["auto", "0", "90", "180", "270"],
        default="auto",
        help="Degrees to turn the photo clockwise before reading (default: auto-detect sideways photos)"
    )

    # 2. show command
    studio_parser = subparsers.add_parser("studio", help="Open the local web app (add photos, draw boxes, verify, build packs)")
    studio_parser.add_argument("--dir", default="~/qpack-studio", help="Folder for pages, labels and packs")
    studio_parser.add_argument("--host", default="127.0.0.1")
    studio_parser.add_argument("--port", type=int, default=8765)
    studio_parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    studio_parser.add_argument("--model", default="gemini-2.5-flash")

    show_parser = subparsers.add_parser("show", help="Inspect and display questions from a .qpack file")
    show_parser.add_argument("qpack", help="Path to .qpack file")
    show_parser.add_argument(
        "--group-by",
        choices=["concept", "formula", "number"],
        default="concept",
        help="Group questions display by concept, formula, or number (default: concept)"
    )

    # 3. check command
    check_parser = subparsers.add_parser("check", help="Check answers and flag low-confidence reads")
    check_parser.add_argument("qpack", help="Path to .qpack file")
    check_parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold below which to flag for manual review (default: 0.70)"
    )

    return parser


def main() -> None:
    """CLI Main Entry Point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        sys.exit(handle_scan(args))
    elif args.command == "studio":
        from qpack.studio import serve
        sys.exit(serve(args.dir, args.host, args.port, args.api_key, args.model))
    elif args.command == "show":
        sys.exit(handle_show(args))
    elif args.command == "check":
        sys.exit(handle_check(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
