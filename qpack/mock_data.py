"""High-fidelity dataset extracted from Hossam Elshafay Physics Workbook page 1.

Used for offline unit tests, regression tests, and scanning when GEMINI_API_KEY is not set.
"""

from typing import List, Dict, Any

MOCK_PHYSICS_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": 35,
        "question_number": "٣٥",
        "question_text": "في الشكل أربع أسلاك مستقيمة موضوعة عند رؤوس مربع كما بالشكل فإذا كانت التيارات متساوية فإن كثافة الفيض المغناطيسي عند نقطة 1 تكون أكبر ما يمكن في الشكل ..........",
        "options": {
            "أ": "الشكل (أ)",
            "ب": "الشكل (ب)",
            "ج": "الشكل (ج)",
            "د": "الشكل (د)"
        },
        "student_choice": "أ",
        "correct_choice": "أ",
        "concept_tag": "محصلة كثافة الفيض لسلك مستقيم",
        "formula": r"B = \frac{\mu I}{2\pi d}",
        "parameters": {
            "wires": "4",
            "currents": "متساوية",
            "geometry": "رؤوس مربع",
            "center_point": "نقطة 1"
        },
        "student_working_box": [0.03, 0.88, 0.10, 0.98],
        "question_box": [0.01, 0.02, 0.19, 0.98],
        "working_ocr_text": "a = y, m\nB_net = B1 + B2 + B3 + B4",
        "working_ocr_confidence": 0.88,
        "solution_arabic": "بتطبيق قاعدة اليد اليمنى لأمبير على كل سلك عند المركز (نقطة 1): نجد أنه في الشكل (أ) اتجاهات المجالات المغناطيسية الناتجة عن التيارات الأربعة تتجمع وتتفق في نفس الاتجاه فتكون المحصلة أكبر ما يمكن ($B_{net} = 4B$). أما في الأشكال الأخرى فالمجالات تتعامد أو تتلاشى جزئياً فتكون المحصلة أقل.\nلذا الإجابة الصحيحة هي (أ)."
    },
    {
        "id": 36,
        "question_number": "٣٦",
        "question_text": "في الشكل المقابل: إذا انعدمت كثافة الفيض المغناطيسي عند نقطة (أ) فإن مقدار $I_2$ واتجاهه ...... علماً بأن تيار $I_1$ عمودي للخارج",
        "options": {
            "أ": "$I_2 > I_1$ ، داخل الصفحة",
            "ب": "$I_2 < I_1$ ، داخل الصفحة",
            "ج": "$I_2 = I_1$ ، داخل الصفحة",
            "د": "$I_2 = I_1$ ، خارج الصفحة"
        },
        "student_choice": "أ",
        "correct_choice": "أ",
        "concept_tag": "نقطة التعادل بين سلكين مستقيمين",
        "formula": r"\frac{I_1}{d_1} = \frac{I_2}{d_2}",
        "parameters": {
            "point": "أ",
            "I1_direction": "عمودي للخارج",
            "condition": "B_net = 0 (نقطة تعادل)"
        },
        "student_working_box": [0.24, 0.05, 0.32, 0.32],
        "question_box": [0.19, 0.02, 0.40, 0.98],
        "working_ocr_text": "B_net = 0\nI1/d1 = I2/d2\nd2 > d1 ==> I2 > I1\nالاتجاه: داخل الصفحة",
        "working_ocr_confidence": 0.91,
        "solution_arabic": "انعدام كثافة الفيض عند نقطة (أ) يعني أنها نقطة تعادل: $\\frac{I_1}{d_1} = \\frac{I_2}{d_2}$.\nبما أن النقطة (أ) تقع خارج السلكين بجوار السلك (1)، فإن بعدها عن السلك الثاني أكبر ($d_2 > d_1$)، وبالتالي يجب أن يكون تيار الثاني أكبر ($I_2 > I_1$).\nوبما أن نقطة التعادل تقع خارج السلكين، يجب أن يكون التياران في اتجاهين متضادين. وبما أن تيار $I_1$ عمودي للخارج، فإن تيار $I_2$ عمودي لداخل الصفحة.\nلذا الإجابة الصحيحة هي (أ)."
    },
    {
        "id": 37,
        "question_number": "٣٧",
        "question_text": "تقاس النفاذية المغناطيسية بوحدة ............",
        "options": {
            "أ": r"\text{Tm}\cdot\text{sec/C}",
            "ب": r"\text{Wb/A}\cdot\text{m}",
            "ج": r"\text{Tm/A}",
            "د": "جميع ما سبق"
        },
        "student_choice": "د",
        "correct_choice": "د",
        "concept_tag": "معامل النفاذية المغناطيسية",
        "formula": r"\mu = \frac{2\pi d B}{I} \implies [\mu] = \frac{\text{T}\cdot\text{m}}{\text{A}}",
        "parameters": {
            "unit_1": "Tm/A",
            "unit_2": "Wb/(A.m)",
            "unit_3": "Tm.sec/C (حيث A = C/sec)"
        },
        "student_working_box": [0.41, 0.35, 0.48, 0.55],
        "question_box": [0.40, 0.02, 0.48, 0.98],
        "working_ocr_text": "Tm/A = Wb/(A.m)\nA = C/sec ==> Tm.sec/C\nكلهم صح!",
        "working_ocr_confidence": 0.62,  # < 0.70 to trigger manual review flag
        "solution_arabic": "من قانون السلك المستقيم $B = \\frac{\\mu I}{2\\pi d} \\implies \\mu = \\frac{B \\cdot 2\\pi d}{I}$:\n1) الوحدة الأساسية هي $\\text{T}\\cdot\\text{m/A}$ (الاختيار ج).\n2) بما أن ويبر $\\text{Wb} = \\text{T}\\cdot\\text{m}^2$، فإن $\\frac{\\text{T}\\cdot\\text{m}}{\\text{A}} = \\frac{\\text{Wb/m}^2 \\cdot \\text{m}}{\\text{A}} = \\text{Wb/A}\\cdot\\text{m}$ (الاختيار ب).\n3) وبما أن أمبير $\\text{A} = \\text{C/sec}$، فإن $\\frac{\\text{T}\\cdot\\text{m}}{\\text{C/sec}} = \\text{Tm}\\cdot\\text{sec/C}$ (الاختيار أ).\nلذا جميع ما سبق وحدات صحيحة ومكافئة، والإجابة الصحيحة هي (د)."
    },
    {
        "id": 38,
        "question_number": "٣٨",
        "question_text": "أي الأشكال الآتية يعبر بصورة صحيحة عن اتجاهي المجال والتيار ............",
        "options": {
            "أ": "1",
            "ب": "2",
            "ج": "3",
            "د": "لا توجد إجابة صحيحة"
        },
        "student_choice": "ج",
        "correct_choice": "ج",
        "concept_tag": "قاعدة اليد اليمنى لأمبير وتحديد اتجاه المجال",
        "formula": r"\vec{B} \perp \vec{I} \text{ (Right-Hand Grip Rule)}",
        "parameters": {
            "rule": "قاعدة اليد اليمنى لأمبير",
            "thumb": "اتجاه التيار",
            "fingers": "اتجاه المجال"
        },
        "student_working_box": [0.49, 0.03, 0.63, 0.35],
        "question_box": [0.48, 0.02, 0.64, 0.98],
        "working_ocr_text": "قاعدة أمبير لليد اليمنى:\nالشكل 3 الابهام لأعلى والأصابع تلف عكس عقارب الساعة (صحيح)",
        "working_ocr_confidence": 0.92,
        "solution_arabic": "بتطبيق قاعدة اليد اليمنى لأمبير بحيث يشير الإبهام لاتجاه التيار والأصابع لاتجاه خطوط المجال:\n- الشكل 1: اتجاه الدوران مرسوم بالعكس.\n- الشكل 2: السلك أفقي والدوران غير متوافق مع اتجاه التيار.\n- الشكل 3: التيار لأعلى وخطوط الفيض تلتف عكس عقارب الساعة في الاتجاه الصحيح تماماً.\nلذا الإجابة الصحيحة هي (ج)."
    },
    {
        "id": 39,
        "question_number": "٣٩",
        "question_text": "سلكان متوازيان يحملان تيارين متساويين في المقدار في اتجاه واحد فتكون كثافة الفيض المغناطيسي عند منتصف المسافة بينهما ........"
        ,
        "options": {
            "أ": "ضعف كثافة الفيض الناتج من أي من السلكين",
            "ب": "نصف كثافة الفيض الناتج من أي من السلكين",
            "ج": "صفر",
            "د": "مربع كثافة الفيض الناتج من أي من السلكين"
        },
        "student_choice": "ج",
        "correct_choice": "ج",
        "concept_tag": "نقطة التعادل بين سلكين مستقيمين",
        "formula": r"B_{net} = B_1 - B_2 = 0",
        "parameters": {
            "I1": "I",
            "I2": "I",
            "d1": "d",
            "d2": "d",
            "direction": "نفس الاتجاه"
        },
        "student_working_box": [0.70, 0.05, 0.77, 0.30],
        "question_box": [0.64, 0.02, 0.78, 0.98],
        "working_ocr_text": "تياران في نفس الاتجاه: بينهما طرح\nB_net = B1 - B2\nI1 = I2 & d1 = d2 ==> B_net = 0",
        "working_ocr_confidence": 0.96,
        "solution_arabic": "عند مرور تيارين متساويين في نفس الاتجاه في سلكين متوازيين، تكون محصلة كثافة الفيض بينهما هي طرح المجالين ($B_{net} = |B_1 - B_2|$).\nوعند منتصف المسافة بالضبط تكون $d_1 = d_2$، وبما أن $I_1 = I_2$، فإن $B_1 = B_2$.\nبالتالي $B_{net} = B_1 - B_2 = 0$ (نقطة تعادل).\nلذا الإجابة الصحيحة هي (ج)."
    },
    {
        "id": 40,
        "question_number": "٤٠",
        "question_text": "في الشكل المقابل: أي النقاط ينعدم عندها محصلة كثافة الفيض المغناطيسي ..........",
        "options": {
            "أ": "a",
            "ب": "b",
            "ج": "c",
            "د": "d"
        },
        "student_choice": "أ",
        "correct_choice": "د",
        "concept_tag": "نقطة التعادل بين سلكين مستقيمين",
        "formula": r"\frac{I_1}{d_1} = \frac{I_2}{d_2}",
        "parameters": {
            "wire_1": "2I لأعلى",
            "wire_2": "I لأسفل",
            "target": "نقطة انعدام محصلة الفيض (نقطة التعادل)"
        },
        "student_working_box": [0.82, 0.05, 0.95, 0.25],
        "question_box": [0.78, 0.02, 0.99, 0.98],
        "working_ocr_text": "التياران متضادان ==> منطقة طرح في الخارج بجوار التيار الأقل (I)\nالسلك ذو التيار I هو الأيمن، والنقطة d بجواره في الخارج.\nإذن النقطة هي d",
        "working_ocr_confidence": 0.89,
        "solution_arabic": "بما أن التيارين في اتجاهين متضادين ($2I$ لأعلى و $I$ لأسفل)، فإن منطقة الطرح تكون في الخارج خارج السلكين.\nوتقع نقطة التعادل دائماً بجوار السلك الذي يحمل التيار الأضعف ($I$).\nوبما أن السلك الأيمن يحمل التيار $I$، فإن النقطة الخارجية الأقرب إليه هي النقطة (d).\nلذا الإجابة الصحيحة هي (د)."
    }
]
