# qpack: Exam & Homework Scanner with Arabic Physics Grader (Phase 1)

**qpack** is an intelligent CLI study tool designed for students and educators. It processes photos of exams and homework workbooks, automatically deskews pages, utilizes Gemini Vision API for high-precision OCR extraction, evaluates multiple-choice answers, groups questions by physical concepts and formulas, and packages everything into portable `.qpack` archive bundles.

---

## ✨ Features

- **Gemini Vision OCR & Extraction**: Directly connects via the lightweight `requests` HTTP library (no heavy SDK needed) to Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash`).
- **Watermark-Resistant Deskewing**: Automatically estimates page tilt using horizontal edge projection variance and straightens rotated captures while filtering out background teacher watermarks.
- **First-Class Arabic (RTL) Support**:
  - Full bidirectional Unicode support.
  - Bidirectional normalization of Arabic-Indic numerals (`٠١٢٣٤٥٦٧٨٩` $\leftrightarrow$ `0123456789`).
  - Standard Arabic multiple-choice letter normalization (`أ`, `ب`, `ج`, `د`).
  - Mathematical equations in physics are parsed and rendered as standard LaTeX (`$ ... $`).
- **Pencil Working & Review Flagging**:
  - Crops both the printed question and the student's handwritten margin notes.
  - Scores pencil OCR confidence and flags ambiguous/faint handwriting ($< 70\%$) for manual review.
- **Concept & Formula Grouping**:
  - Organizes questions not only chronologically by question number, but by physical concepts (e.g. *عزم الازدواج المغناطيسي*, *القوة المغناطيسية*) and underlying formulas (e.g. $\tau = B I A N \sin\theta$).
- **Portable `.qpack` File Format**:
  - Standard ZIP-compressed package containing `manifest.json`, `questions.json`, and all cropped question and working images under `images/`.

---

## 📂 File Structure

```
qpack/
├── page1.jpg                  # Input photo
├── requirements.txt           # Dependencies
├── setup.py                   # Package installer (creates `qpack` executable)
├── README.md                  # This documentation
├── qpack/
│   ├── __init__.py            # Package entry and version
│   ├── cli.py                 # CLI commands: scan, show, check
│   ├── deskew.py              # Photo deskewing and region cropping
│   ├── gemini.py              # Gemini Vision API client (requests)
│   ├── arabic.py              # Arabic digits, normalization & LaTeX math
│   ├── packer.py              # .qpack archive manager
│   ├── analyzer.py            # Answer evaluation & concept grouping
│   └── mock_data.py           # Offline fallback dataset for testing
└── tests/
    ├── __init__.py
    ├── test_arabic.py         # Arabic numeral and text tests
    ├── test_deskew.py         # Deskewing and cropping tests
    ├── test_packer.py         # Archive packaging tests
    ├── test_analyzer.py       # Grader and review flagging tests
    └── test_cli.py            # CLI integration tests
```

---

## 🚀 Installation (Termux / Linux)

### 1. Prerequisites (in Termux)
```bash
pkg install -y python python-pillow python-numpy
```

### 2. Install qpack in editable mode
```bash
cd ~/qpack
pip install -e .
```

### 3. Set your Gemini API Key
```bash
export GEMINI_API_KEY="your-google-gemini-api-key"
```
*(If no API key is provided or offline, qpack gracefully falls back to an authentic Egyptian Secondary Physics Chapter 2 workbook fixture).*

---

## 🛠️ Usage & Commands

### 1. Scan a photo page into a `.qpack` archive
```bash
qpack scan page1.jpg -o page1.qpack
```
This performs:
1. Skew analysis and image straightening.
2. Gemini Vision OCR with background watermark filtering.
3. Cropping of question blocks and margin pencil calculations.
4. Packaging into `page1.qpack`.

### 2. Check answers & flag low-confidence reads
```bash
qpack check page1.qpack
```
Produces a structured table:
- **Q#**: Question number.
- **Concept**: Core physical topic.
- **My Choice**: Student's selected option.
- **Correct**: Verified solution choice.
- **Status**: `✓ RIGHT` / `✗ WRONG` / `- SKIPPED`.
- **OCR Conf**: Confidence score for margin handwriting.
- **Review Flag**: Highlights low-confidence reads ($< 70\%$) for manual human inspection.

### 3. Inspect question details & step-by-step solutions
```bash
# Group questions by concept (default)
qpack show page1.qpack --group-by concept

# Group by formula
qpack show page1.qpack --group-by formula

# Group by chronological question number
qpack show page1.qpack --group-by number
```

---

## 📦 `.qpack` File Specification

A `.qpack` file is an open, standards-compliant ZIP archive with the following contents:

- **`manifest.json`**:
  ```json
  {
    "qpack_version": "1.0",
    "created_at": "2026-11-13T18:15:00Z",
    "source_image": "page1.jpg",
    "total_questions": 4,
    "deskew_angle_degrees": 0.0,
    "concepts": ["عزم الازدواج المغناطيسي", "القوة المغناطيسية المتبادلة بين سلكين"],
    "summary": {
      "score_percent": 75.0,
      "correct": 3,
      "wrong": 1,
      "unanswered": 0,
      "flagged_for_review": 1
    }
  }
  ```
- **`questions.json`**: Full list of questions including Arabic text, LaTeX options, extracted parameters, student choice, correct answer, OCR confidence, and Arabic step-by-step explanation.
- **`images/`**:
  - `page_deskewed.png`: The straightened full page.
  - `q{id}_question.png`: Cropped image of the question, diagram, and options.
  - `q{id}_working.png`: Cropped image of the student's pencil margin notes.

---

## 🧪 Running Tests

To run the complete test suite:
```bash
pytest -v
```
All 18 unit and integration tests run in under 10 seconds in Termux.
