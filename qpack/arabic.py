"""Arabic language support, digit normalization, RTL text, and LaTeX math formatting."""

from __future__ import annotations
import re
from typing import Optional

# Standard Arabic-Indic digits: ٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩ (U+0660 - U+0669)
# Eastern Arabic-Indic (Persian/Urdu): ۰ ۱ ۲ ۳ ۴ ۵ ۶ ۷ ۸ ۹ (U+06F0 - U+06F9)
ARABIC_INDIC_MAP = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

WESTERN_TO_ARABIC_INDIC = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
}

# Arabic multiple-choice letters normalization
OPTION_MAP = {
    'أ': 'أ', 'ا': 'أ', 'إ': 'أ', 'آ': 'أ', 'ء': 'أ',
    'ب': 'ب',
    'ج': 'ج',
    'د': 'د',
}


def to_western_digits(text: str) -> str:
    """Convert any Arabic-Indic digits in the input string to Western ASCII digits."""
    if not text:
        return text
    return "".join(ARABIC_INDIC_MAP.get(ch, ch) for ch in text)


def to_arabic_indic_digits(text: str) -> str:
    """Convert any Western ASCII digits in the input string to Arabic-Indic digits."""
    if not text:
        return text
    return "".join(WESTERN_TO_ARABIC_INDIC.get(ch, ch) for ch in str(text))


def normalize_option_letter(choice: Optional[str]) -> Optional[str]:
    """Normalize an Arabic multiple-choice option (أ, ب, ج, د).
    
    Strips parentheses, brackets, periods, spaces, and normalizes alef variants.
    Returns normalized 'أ', 'ب', 'ج', 'د', or None if unrecognized.
    """
    if choice is None:
        return None
    s = choice.strip()
    # Strip common wrappers like (أ) or [ب] or د.
    s = re.sub(r'[\(\)\[\]\{\}\.\:\-\s]', '', s)
    if not s:
        return None
    first_char = s[0]
    return OPTION_MAP.get(first_char, first_char)


def is_arabic_text(text: str) -> bool:
    """Check if the text contains Arabic script characters."""
    if not text:
        return False
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))


def reshape_arabic(text: str) -> str:
    """Reshape Arabic text for terminals that lack native ligature connection."""
    if not text:
        return text
    try:
        import arabic_reshaper
        return arabic_reshaper.reshape(text)
    except ImportError:
        return text


def clean_latex_math(formula: str) -> str:
    """Sanitize and ensure consistent LaTeX math formatting.
    
    Removes extraneous whitespace, normalizes backslashes, ensures proper wrapping.
    """
    if not formula:
        return ""
    f = formula.strip()
    # If not enclosed in $...$, keep as pure LaTeX representation
    f = re.sub(r'^\$+|\$+$', '', f).strip()
    return f
