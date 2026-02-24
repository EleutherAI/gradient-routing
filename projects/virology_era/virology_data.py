"""Data loading for virology ERA unlearning.

Forget: WMDP Bio Remove Dataset (virology/bio papers)
Retain: WikiText-103 (general text)

Produces (text, label) pairs where label=0 is forget, label=1 is retain.
"""

import numpy as np
from datasets import load_dataset


def load_forget_data(max_examples: int | None = None) -> list[tuple[str, int]]:
    """Load WMDP Bio Remove Dataset as forget data (label=0).

    Concatenates title + abstract + text for each document.
    """
    split = f"train[:{max_examples}]" if max_examples else "train"
    ds = load_dataset("Unlearning/WMDP-Bio-Remove-Dataset", split=split)

    forget_data = []
    for row in ds:
        parts = []
        if row["title"]:
            parts.append(row["title"])
        if row["abstract"]:
            parts.append(row["abstract"])
        if row["text"]:
            parts.append(row["text"])
        text = "\n\n".join(parts)
        if text.strip():
            forget_data.append((text, 0))

    return forget_data


def load_retain_data(max_examples: int | None = None) -> list[tuple[str, int]]:
    """Load WikiText-103 as retain data (label=1).

    Uses the document-level version with column 'page'.
    """
    split = f"train[:{max_examples}]" if max_examples else "train"
    ds = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-103-raw-v1",
        split=split,
    )

    retain_data = []
    for row in ds:
        text = row["page"]
        if text and text.strip():
            retain_data.append((text, 1))

    return retain_data


def load_training_data(
    max_forget: int | None = None,
    max_retain: int | None = None,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Load and shuffle combined forget + retain training data."""
    forget = load_forget_data(max_forget)
    retain = load_retain_data(max_retain)

    print(f"Loaded {len(forget)} forget examples, {len(retain)} retain examples")

    combined = forget + retain
    rng = np.random.default_rng(seed)
    rng.shuffle(combined)
    return combined


def load_validation_data(
    n_forget: int = 200,
    n_retain: int = 200,
    seed: int = 123,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Load validation splits for forget and retain.

    Uses the end of the forget dataset and WikiText validation split.
    """
    # Forget validation: take from the end of the training set
    all_forget = load_forget_data()
    rng = np.random.default_rng(seed)
    rng.shuffle(all_forget)
    forget_val = all_forget[-n_forget:]

    # Retain validation: use WikiText validation split
    ds = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-103-raw-v1",
        split=f"validation[:{n_retain}]",
    )
    retain_val = [(row["page"], 1) for row in ds if row["page"] and row["page"].strip()]

    return forget_val, retain_val


if __name__ == "__main__":
    data = load_training_data(max_forget=100, max_retain=100)
    print(f"Total training examples: {len(data)}")
    print(f"First example label: {data[0][1]}, text[:80]: {data[0][0][:80]}")

    fval, rval = load_validation_data(n_forget=10, n_retain=10)
    print(f"Validation: {len(fval)} forget, {len(rval)} retain")
