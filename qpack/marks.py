"""Resolve a student's final answer from the marks left on each option.

Real workbook pages are messy: a student ticks or circles an option, then
crosses it out and picks another one, sometimes several times in a row.  The
vision model's only job is to report *which marks it sees on which option*;
the final answer is decided here, in plain code, so the rules are explicit
and testable instead of hidden inside a model's judgement.

Input shape (all fields optional)::

    student_marks   {"أ": ["tick", "cross"], "ب": ["circle"]}
                    marks per option, EARLIEST FIRST (a mark drawn on top of
                    another one is later)
    answer_sequence ["أ", "ب"]   options in the order they were worked on,
                    earliest first; [] / None when unknown
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

from qpack.arabic import normalize_option_letter

# A mark that says "this is my answer" / "this is NOT my answer".
POSITIVE = {"tick", "circle", "underline", "fill", "written"}
NEGATIVE = {"cross", "strike", "scribble"}

# Words a model (or a human editing questions.json) might use for the above.
_ALIASES = {
    "check": "tick", "checkmark": "tick", "check_mark": "tick", "ticked": "tick",
    "hook": "tick", "v": "tick", "✓": "tick", "✔": "tick",
    "ring": "circle", "circled": "circle", "o": "circle",
    "underlined": "underline",
    "filled": "fill", "shaded": "fill",
    "letter_written": "written", "written_letter": "written",
    "x": "cross", "✗": "cross", "✘": "cross", "crossed": "cross",
    "cross_out": "cross", "crossed_out": "cross", "x_beside": "cross",
    "strikethrough": "strike", "strike_through": "strike", "struck": "strike",
    "struck_out": "strike", "line_through": "strike",
    "scribbled": "scribble", "scribble_out": "scribble",
    "scratched": "scribble", "overwritten": "scribble",
}


def normalize_mark(mark: Any) -> Optional[str]:
    """Return the canonical mark name, or None if it is not recognised."""
    if mark is None:
        return None
    s = str(mark).strip().lower().replace("-", "_").replace(" ", "_")
    s = _ALIASES.get(s, s)
    return s if (s in POSITIVE or s in NEGATIVE) else None


def _letter(value: Any) -> Optional[str]:
    return normalize_option_letter(value) if isinstance(value, str) else None


def _empty(status: str = "UNANSWERED") -> Dict[str, Any]:
    return {
        "final": None,
        "status": status,        # ANSWERED | UNANSWERED | AMBIGUOUS
        "method": None,          # mark | latest_marked | elimination | reported
        "candidates": [],
        "crossed_out": [],
        "history": [],           # [{"option": "أ", "action": "crossed_out"}, ...]
        "changed": False,        # the student crossed something out and chose again
        "flags": [],             # reasons a human should double-check
        "notes": [],             # informational only
    }


def resolve_answer(
    student_marks: Optional[Dict[str, Any]],
    option_keys: Optional[Sequence[str]] = None,
    answer_sequence: Optional[Sequence[str]] = None,
    reported_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """Work out the student's final choice from per-option marks.

    ``reported_choice`` is the model's own guess (or an old pack's
    ``student_choice``).  It is used only when no marks were reported, and it
    is cross-checked against the marks when both exist.
    """
    res = _empty()
    reported: Optional[str] = None
    if isinstance(reported_choice, str) and reported_choice.strip().lower() not in ("", "none", "null"):
        reported = _letter(reported_choice)

    keys = [k for k in (_letter(k) for k in (option_keys or [])) if k] or None

    # --- No marks reported: fall back to the single reported choice --------
    if not isinstance(student_marks, dict) or not student_marks:
        if reported:
            res.update(final=reported, status="ANSWERED", method="reported")
        return res

    # --- Normalise marks per option ---------------------------------------
    marks_by_opt: Dict[str, List[str]] = {}
    for raw_key, raw_marks in student_marks.items():
        opt = _letter(raw_key)
        if not opt:
            continue
        if keys and opt not in keys:
            res["flags"].append(f"Marks reported on unknown option '{raw_key}'")
            continue
        if isinstance(raw_marks, str):
            raw_marks = [raw_marks]
        names: List[str] = []
        for m in raw_marks or []:
            name = normalize_mark(m)
            if name:
                names.append(name)
            else:
                res["flags"].append(f"Unrecognized mark '{m}' on option {opt}")
        if names:
            marks_by_opt.setdefault(opt, []).extend(names)

    seq = [o for o in (_letter(o) for o in (answer_sequence or [])) if o]

    def order_key(opt: str) -> int:
        return seq.index(opt) if opt in seq else len(seq)

    ordered = sorted(marks_by_opt, key=order_key)  # stable: keeps input order

    # --- Per-option verdict: the LAST mark on an option decides ------------
    selected: List[str] = []
    rejected: List[str] = []
    contested: List[str] = []   # has both kinds of mark, so order matters
    for opt in ordered:
        marks = marks_by_opt[opt]
        has_pos = any(m in POSITIVE for m in marks)
        has_neg = any(m in NEGATIVE for m in marks)
        if has_pos and has_neg:
            contested.append(opt)
            last_positive = marks[-1] in POSITIVE
            (selected if last_positive else rejected).append(opt)
            verdict = "re-chosen" if last_positive else "crossed out"
            res["notes"].append(f"Option {opt}: {' then '.join(marks)} (treated as {verdict})")
        elif has_pos:
            selected.append(opt)
        elif has_neg:
            rejected.append(opt)

    res["crossed_out"] = list(rejected)
    res["history"] = (
        [{"option": o, "action": "crossed_out"} for o in rejected]
        + [{"option": o, "action": "selected"} for o in selected]
    )

    # --- Decide ------------------------------------------------------------
    if len(selected) == 1:
        res.update(final=selected[0], status="ANSWERED", method="mark")
    elif len(selected) > 1:
        latest = next((o for o in reversed(seq) if o in selected), None)
        listing = "، ".join(selected)
        if latest:
            res.update(final=latest, status="ANSWERED", method="latest_marked")
            res["flags"].append(f"Several options marked ({listing}); took the latest ({latest})")
        else:
            res.update(status="AMBIGUOUS", candidates=list(selected))
            res["flags"].append(f"Several options marked ({listing}) and the order is unclear")
    else:
        remaining = [k for k in keys if k not in rejected] if keys else None
        if remaining is not None and len(keys) >= 3 and rejected and len(remaining) == 1:
            res.update(final=remaining[0], status="ANSWERED", method="elimination")
            res["flags"].append(
                f"No positive mark; inferred {remaining[0]} because every other option is crossed out"
            )
        elif remaining is not None and rejected and not remaining:
            res["flags"].append("Every option is crossed out")
        elif rejected:
            res["flags"].append(
                f"Options crossed out ({'، '.join(rejected)}) but no final choice is marked"
            )

    # --- Extra safety checks ----------------------------------------------
    if res["final"] in contested:
        res["flags"].append(
            f"The order of marks on option {res['final']} decides the answer; please confirm"
        )
    if reported and reported != res["final"]:
        shown = res["final"] or "no answer"
        res["flags"].append(f"Model reported {reported} but the marks resolve to {shown}")

    res["changed"] = bool(res["crossed_out"]) and res["method"] in ("mark", "latest_marked")
    return res
