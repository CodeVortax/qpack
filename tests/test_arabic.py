"""Unit tests for Arabic support, digit normalization, and math formatting."""

from qpack.arabic import (
    to_western_digits,
    to_arabic_indic_digits,
    normalize_option_letter,
    is_arabic_text,
    clean_latex_math
)


def test_digit_normalization_to_western():
    assert to_western_digits("١٢٣٤٥٦٧٨٩٠") == "1234567890"
    assert to_western_digits("السؤال رقم ٤:") == "السؤال رقم 4:"
    assert to_western_digits("tau = 0.8 N.m") == "tau = 0.8 N.m"
    # Eastern Arabic-Indic digits (Persian/Urdu)
    assert to_western_digits("۱۲۳۴۵۶۷۸۹۰") == "1234567890"


def test_digit_normalization_to_arabic_indic():
    assert to_arabic_indic_digits("1234567890") == "١٢٣٤٥٦٧٨٩٠"
    assert to_arabic_indic_digits(42) == "٤٢"


def test_normalize_option_letter():
    # Standard Arabic choices
    assert normalize_option_letter("أ") == "أ"
    assert normalize_option_letter("ب") == "ب"
    assert normalize_option_letter("ج") == "ج"
    assert normalize_option_letter("د") == "د"

    # Alef variations
    assert normalize_option_letter("ا") == "أ"
    assert normalize_option_letter("إ") == "أ"
    assert normalize_option_letter("آ") == "أ"

    # Brackets, punctuation and spaces
    assert normalize_option_letter("(أ)") == "أ"
    assert normalize_option_letter(" [ب] ") == "ب"
    assert normalize_option_letter("ج.") == "ج"
    assert normalize_option_letter("- د") == "د"

    # None and empty handling
    assert normalize_option_letter(None) is None
    assert normalize_option_letter("") is None


def test_is_arabic_text():
    assert is_arabic_text("فيزياء تالتة ثانوي") is True
    assert is_arabic_text("Physics 101") is False
    assert is_arabic_text("عزم الازدواج: tau = 0.8") is True
    assert is_arabic_text("") is False


def test_clean_latex_math():
    assert clean_latex_math(r"$ \tau = B I A N \sin\theta $") == r"\tau = B I A N \sin\theta"
    assert clean_latex_math(r"F = \frac{\mu I_1 I_2}{2\pi d}") == r"F = \frac{\mu I_1 I_2}{2\pi d}"
