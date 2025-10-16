import re
import json
import unicodedata
from datasets import Dataset

from utils.homoglyphs import normalize_homoglyphs


def normalize_to_ascii(text: str) -> str:
    """Replace visually similar Unicode homoglyphs with ASCII and strip extras."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """Normalize unicode, remove weird spaces/newlines, and standardize quotes."""
    if not isinstance(text, str):
        return ""
    text = normalize_homoglyphs(text=text)
    text = unicodedata.normalize("NFKC", text)  # normalize font variations
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\s.,!?;:()\"'-]", " ", text)  # remove unknown special chars
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'(\d+)"', r"\1 inch", text)  # 15" → 15 inch
    return text


def clean_parent_asin(asin: str) -> str:
    """Ensure parent_asin is uppercase alphanumeric."""
    if not isinstance(asin, str):
        return ""
    asin = asin.upper()
    asin = re.sub(r"[^A-Z0-9]", "", asin)
    return asin


def clean_numeric(value, dtype="float", max_rating=5.0):
    """Validate and clean numeric fields (no None allowed)."""
    try:
        if dtype == "float":
            val = float(value)
            if val < 0:
                return 0.0
            if max_rating and val > max_rating:
                return max_rating
            return val
        elif dtype == "int":
            val = int(float(value))
            return max(0, val)
    except (ValueError, TypeError):
        return 0 if dtype == "int" else 0.0
    return 0 if dtype == "int" else 0.0


def parse_details(details_str: str):
    """Parse details dict stored as a string and flatten it."""
    try:
        details = json.loads(details_str.replace("'", "\""))
        if not isinstance(details, dict):
            return {}
    except Exception:
        return {}

    flat_details = {}
    for k, v in details.items():
        key = f"detail_{k.strip().lower().replace(' ', '_')}"
        if isinstance(v, str):
            flat_details[key] = normalize_text(v)
        elif isinstance(v, (int, float, bool)):
            flat_details[key] = v
        else:
            flat_details[key] = str(v) if v is not None else ""
    return flat_details


def preprocess_dataset(dataset: Dataset):
    """Main preprocessing pipeline for HuggingFace Dataset. Guarantees no None."""
    processed = []

    for row in dataset:
        processed_row = {
            "parent_asin": clean_parent_asin(row.get("parent_asin", "")),
            "title": normalize_text(row.get("title", "")),
            "description": normalize_text(row.get("description", "")),
            "main_category": normalize_text(row.get("main_category", "")),
            "store": normalize_text(row.get("store", "")),
            "average_rating": clean_numeric(row.get("average_rating"), dtype="float", max_rating=5.0),
            "rating_number": clean_numeric(row.get("rating_number"), dtype="int"),
            "price": clean_numeric(row.get("price"), dtype="float", max_rating=None),
        }

        # Optional: include flattened details
        # details = parse_details(row.get("details", "{}"))
        # processed_row.update(details)

        # Final safety pass: replace any None left behind
        for k, v in processed_row.items():
            if v is None:
                if isinstance(v, (int, float)):
                    processed_row[k] = 0
                else:
                    processed_row[k] = ""

        processed.append(processed_row)

    return Dataset.from_list(processed)
