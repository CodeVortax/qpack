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

from qpack.deskew import deskew_image
from qpack.gemini import call_gemini_vision
from qpack.packer import create_qpack, load_qpack
from qpack.analyzer import check_exam, group_questions
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

    # 2. Deskew photo
    console.print("[dim]Step 1/4: Analyzing skew and deskewing photo...[/dim]")
    deskewed_img, angle = deskew_image(raw_img)
    if abs(angle) >= 0.2:
        console.print(f"  [green]✓[/green] Deskewed image by [bold]{angle:+.2f}°[/bold]")
    else:
        console.print(f"  [dim]✓ Image orientation is upright (skew angle: {angle:.2f}°)[/dim]")

    # 3. Vision / OCR via Gemini API (or mock fallback)
    console.print("[dim]Step 2/4: Running Gemini Vision OCR & Physics Extraction...[/dim]")
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.mock:
        console.print("  [yellow]Notice:[/yellow] GEMINI_API_KEY is not set. Using offline high-fidelity physics mock dataset.")
    elif api_key:
        console.print("  [green]✓[/green] Connected to Gemini API using GEMINI_API_KEY")

    try:
        questions = call_gemini_vision(
            deskewed_img,
            api_key=api_key,
            model=args.model,
            force_mock=args.mock
        )
    except Exception as exc:
        console.print(f"[bold red]Extraction failed:[/bold red] {exc}")
        return 1

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
        f"[dim]Total questions: {len(questions)} | Angle: {angle:+.2f}°[/dim]",
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
            student = q.get("student_choice") or "[dim]None[/dim]"
            correct = q.get("correct_choice") or "?"
            conf = q.get("working_ocr_confidence", 1.0)
            solution = q.get("solution_arabic", "")

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
                console.print(f"    [bold]({opt_key})[/bold] {opt_val}{flag}")

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
        my_choice = str(item.get("student_choice") or "-")
        correct = str(item.get("correct_choice") or "-")
        status = item.get("status")
        conf = item.get("working_ocr_confidence", 1.0)
        conf_str = f"{conf:.0%}"

        if status == "CORRECT":
            status_text = "[bold green]✓ RIGHT[/bold green]"
        elif status == "WRONG":
            status_text = "[bold red]✗ WRONG[/bold red]"
        else:
            status_text = "[yellow]- SKIPPED[/yellow]"

        conf_styled = f"[green]{conf_str}[/green]" if conf >= threshold else f"[bold red]{conf_str}[/bold red]"

        if item.get("needs_review"):
            flag_str = "[bold red]⚠️  " + "; ".join(item.get("review_flags", [])) + "[/bold red]"
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
        f"[yellow]{exam_res['unanswered']} Unanswered[/yellow]\n"
    )
    if exam_res["flagged_for_review"] > 0:
        summary_text += f"[bold red]⚠️  {exam_res['flagged_for_review']} question(s) flagged for manual review due to low OCR confidence (< {threshold:.0%}) or missing answers.[/bold red]"
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
    scan_parser.add_argument("--mock", action="store_true", help="Force offline mock data for testing")

    # 2. show command
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
    elif args.command == "show":
        sys.exit(handle_show(args))
    elif args.command == "check":
        sys.exit(handle_check(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
