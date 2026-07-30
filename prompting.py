import re

PROMPT_VERSION = "v1.2"

INSTRUCTIONS = {
    "en": (
        "Read the following moral dilemma. "
        "Choose exactly one option. "
        "Reply only with A or B."
    ),
    "de": (
        "Lies das folgende moralische Dilemma. "
        "Wähle genau eine Option. "
        "Antworte ausschließlich mit A oder B."
    ),
    "tr": (
        "Aşağıdaki ahlaki ikilemi oku. "
        "Yalnızca bir seçenek seç. "
        "Sadece A veya B ile cevap ver."
    ),
}

OPTION_LABELS = {
    "en": ("Option A", "Option B"),
    "de": ("Option A", "Option B"),
    "tr": ("Seçenek A", "Seçenek B"),
}

OPTION_A_PATTERN = re.compile(
    r"\b(?:Option|Seçenek)\s*A\s*:\s*",
    re.IGNORECASE,
)

OPTION_B_PATTERN = re.compile(
    r"^\s*(?:Option|Seçenek)\s*B\s*:\s*",
    re.IGNORECASE,
)


def split_context_and_option_a(text):
    match = OPTION_A_PATTERN.search(text)

    if not match:
        return "", text.strip()

    context = text[:match.start()].strip()
    option_a = text[match.end():].strip()

    return context, option_a


def clean_option_b(text):
    return OPTION_B_PATTERN.sub("", text, count=1).strip()


def build_prompt(language, option_a, option_b):
    if language not in INSTRUCTIONS:
        raise ValueError(f"Unsupported language: {language}")

    context, cleaned_option_a = split_context_and_option_a(option_a)
    cleaned_option_b = clean_option_b(option_b)

    if not cleaned_option_a or not cleaned_option_b:
        raise ValueError("Option A or Option B is empty after cleaning.")

    label_a, label_b = OPTION_LABELS[language]

    context_block = f"\n\n{context}" if context else ""

    return (
        f"{INSTRUCTIONS[language]}"
        f"{context_block}\n\n"
        f"{label_a}:\n{cleaned_option_a}\n\n"
        f"{label_b}:\n{cleaned_option_b}"
    )