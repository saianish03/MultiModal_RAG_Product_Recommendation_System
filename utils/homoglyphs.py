# utils/homoglyphs.py
import re
import unicodedata

# Optional: install unidecode for extra transliteration:
# pip install Unidecode
try:
    from unidecode import unidecode  # type: ignore
except Exception:
    unidecode = None  # fallback later

# -------------------------
# CONFUSABLES MAPPING
# (common homoglyphs -> ASCII)
# Add to this dict as you find new problematic glyphs.
# -------------------------
CONFUSABLES = {
    # ---- Greek (common) ----
    "Α": "A", "Β": "B", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Θ": "TH", "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M", "Ν": "N", "Ξ": "X",
    "Ο": "O", "Π": "P", "Ρ": "P", "Σ": "S", "Τ": "T", "Υ": "Y", "Φ": "F",
    "Χ": "X", "Ψ": "PS", "Ω": "O",
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "h",
    "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "p", "σ": "s", "ς": "s", "τ": "t", "υ": "y",
    "φ": "f", "χ": "x", "ψ": "ps", "ω": "o",
    # lunate sigma / variants
    "Ϲ": "C", "ϲ": "c", "Ϻ": "M", "ϻ": "m",

    # ---- Cyrillic -> visually-similar Latin ----
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ѕ": "s",
    "ԁ": "d", "Ԍ": "G", "з": "3",

    # ---- Armenian common ----
    "Ա": "A", "Բ": "B", "Գ": "G", "Դ": "D", "Ե": "E", "Զ": "Z", "Է": "E",
    "Թ": "T", "Ժ": "ZH", "Ի": "I", "Լ": "L", "Խ": "X", "Ծ": "C", "Կ": "K",
    "Հ": "H", "Ձ": "DZ", "Ղ": "GH", "Ճ": "CH", "Մ": "M", "Յ": "Y", "Ն": "N",
    "Շ": "SH", "Ո": "O", "Չ": "CH", "Պ": "P", "Ջ": "J", "Ռ": "R", "Ս": "S",
    "Վ": "V", "Տ": "T", "Ր": "R", "Ց": "C", "Փ": "P", "Ք": "K", "Օ": "O",
    "Ֆ": "F",
    "ա": "a", "բ": "b", "գ": "g", "դ": "d", "ե": "e", "զ": "z", "է": "e",
    "թ": "t", "ժ": "zh", "ի": "i", "լ": "l", "խ": "x", "ծ": "c", "կ": "k",
    "հ": "h", "ձ": "dz", "ղ": "gh", "ճ": "ch", "մ": "m", "յ": "y", "ն": "n",
    "շ": "sh", "ո": "o", "չ": "ch", "պ": "p", "ջ": "j", "ռ": "r", "ս": "s",
    "վ": "v", "տ": "t", "ր": "r", "ց": "c", "փ": "p", "ք": "k", "օ": "o",
    "ֆ": "f",

    # ---- Runic / other odd letters (common confusables) ----
    "ᛐ": "T", "ᚠ": "F", "ᛒ": "B",

    # ---- Misc useful single-character confusables ----
    "“": "\"", "”": "\"", "‘": "'", "’": "'", "—": "-", "–": "-",
    "﹣": "-", "·": ".", "…": "...", "ª": "a", "º": "o",

    # small special forms that sometimes sneak in
    "Ϲ": "C", "Ϻ": "M", "᧐": "0", "０": "0", "１": "1", "２": "2", "３": "3",
    "４": "4", "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
}

# Build translation table for Python's str.translate (fast)
_TRANSLATION_TABLE = str.maketrans(CONFUSABLES)


# -------------------------
# Main function
# -------------------------
def normalize_homoglyphs(text: str, uppercase: bool = True) -> str:
    """
    Replace Unicode homoglyphs with ASCII equivalents.

    Args:
        text: input string (may contain Greek/Cyrillic/Armenian/etc characters).
        uppercase: if True returns uppercase output (useful for normalization of titles).

    Returns:
        ASCII-only string where confusable characters have been replaced. If a glyph
        has no mapping it will be removed.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: normalize compatibility (fold fullwidth, superscripts, etc.)
    s = unicodedata.normalize("NFKC", text)

    # Step 2: apply explicit confusables mapping (fast)
    s = s.translate(_TRANSLATION_TABLE)

    # Step 3: try unidecode for broad transliteration (optional but useful)
    if unidecode is not None:
        try:
            s = unidecode(s)
        except Exception:
            # if unidecode fails for some reason, skip it
            pass

    # Step 4: remove any remaining non-ASCII characters
    # (this drops characters that have no ASCII equivalent per your requirement)
    s = re.sub(r"[^\x00-\x7F]+", "", s)

    # Step 5: collapse whitespace and fix spacing around punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.,;:!?%])", r"\1", s)

    # Optional small post-processing heuristics (safe fixes)
    # - Convert "VoI." -> "Vol." which is a common OCR/homoglyph issue in titles.
    s = re.sub(r"\b(V|v)oI\.", r"\1ol.", s)

    # Step 6: uppercase if requested
    if uppercase:
        s = s.upper()

    return s
