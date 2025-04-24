from __future__ import annotations

from typing import List, Dict, Any

from transformers import pipeline, Pipeline

# ---------------------------------------------------------------------------
# Load the MNLI zero‑shot pipeline once at import time so that subsequent calls
# are fast.  Use the default device (auto‑detect GPU/CPU).
# ---------------------------------------------------------------------------
_classifier: Pipeline = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device_map="auto"
)

# ---------------------------------------------------------------------------
# Candidate labels – make them *descriptive phrases* rather than bare nouns.
# The MNLI model performs better when the label can slot naturally into the
# hypothesis sentence (“This document is a …”).
# ---------------------------------------------------------------------------
CANDIDATE_LABELS: List[str] = [
    "bank account statement",
    "W‑2 tax form",
    "cell‑phone bill",
    "school‑enrollment certificate",
    "employee payslip",
    "1099 tax form",
]

# Custom hypothesis template tuned for document classification.
HYPOTHESIS_TEMPLATE: str = "This document is a {}."

# Optional: Maximum number of characters from the OCR text to consider.  This
# keeps the prompt inside the model’s context window even for very long PDFs.
MAX_LEN: int = 4000


def classify_text(
    text: str,
    labels: List[str] | None = None,
    return_full: bool = False,
) -> str | Dict[str, Any]:
    """Classify *text* into one of *labels*.

    Args:
        text: Raw text extracted from the document (OCR or plain‑text).
        labels: Optional override of the global ``CANDIDATE_LABELS`` list.
        return_full: If ``True`` return the full HF output dict; otherwise just
            the top‑scoring label.

    Returns:
        str | dict: The best label (default) or the raw HF result object.
    """
    if labels is None:
        labels = CANDIDATE_LABELS

    # Truncate long text blocks – empirically 4 000 chars ≈ 1 000 tokens.
    snippet = text[:MAX_LEN]

    result = _classifier(
        snippet,
        labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    # Optional debug log
    # print("[DEBUG] Classification result:", result)

    return result if return_full else result["labels"][0]


# ---------------------------------------------------------------------------
# Quick self‑test – run `python zero_shot_classifier.py` from the CLI.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _sample_txt = """
    ACME BANK
    Account Summary –  March 2025
    Beginning balance: 7,500.00
    Deposits & Credits: 3,250.00
    Withdrawals & Debits: 1,125.50
    Ending balance: 9,624.50
    """

    print("Predicted label:", classify_text(_sample_txt))
