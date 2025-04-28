#!/usr/bin/env python3
"""rename_danceable_tracks.py

Retro-actively rename already-downloaded MP3 files in
``data/fma/fma_tracks/<genre>`` directories from the plain ``123456.mp3``
format to ``123456_<slugified_title>.mp3``.

We rely on ``data/fma/danceable_genre_samples.csv`` (or a custom CSV) to map
``track_id``→``track_title``.

The script is **fail-fast** – any ambiguity (missing title, duplicate target
file) aborts execution with a descriptive error.

Usage
-----
    python scripts/rename_danceable_tracks.py [SAMPLES_CSV] [DEST_DIR]

Defaults:
    SAMPLES_CSV = data/fma/danceable_genre_samples.csv
    DEST_DIR    = data/fma/fma_tracks
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

DEFAULT_SAMPLES_CSV = Path("data/fma/danceable_genre_samples.csv")
DEFAULT_DEST_DIR = Path("data/fma/fma_tracks")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def slugify(s: str, *, allow_unicode: bool = False, max_length: int = 60) -> str:
    """Create a safe slug for filenames."""

    if allow_unicode:
        value = unicodedata.normalize("NFKC", s)
    else:
        value = (
            unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        )
    value = value.lower()
    value = re.sub(r"[^\w\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")[:max_length]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def main(samples_csv: Path, dest_dir: Path) -> None:
    if not samples_csv.is_file():
        raise FileNotFoundError(samples_csv)
    if not dest_dir.is_dir():
        raise FileNotFoundError(dest_dir)

    # ---------------------------------------------------------------------
    # Build mapping from track_id -> track_title
    # ---------------------------------------------------------------------
    df = pd.read_csv(samples_csv)
    if "track_id" not in df.columns or "track_title" not in df.columns:
        raise RuntimeError(
            "CSV must contain 'track_id' and 'track_title' columns for renaming."
        )

    id_to_title = (
        df[["track_id", "track_title"]]
        .dropna()
        .set_index("track_id")["track_title"]
        .astype(str)
        .to_dict()
    )

    # Pattern for files that still need renaming (no slug part)
    plain_pattern = re.compile(r"^(\d{6})\.mp3$")

    # Iterate over all mp3 files in dest_dir
    for mp3_path in dest_dir.rglob("*.mp3"):
        match = plain_pattern.match(mp3_path.name)
        if not match:
            # Already renamed – skip
            continue

        track_id = int(match.group(1))
        if track_id not in id_to_title:
            raise RuntimeError(
                f"Title for track_id {track_id} not found in {samples_csv}. Cannot rename."
            )

        slug = slugify(id_to_title[track_id])
        new_path = mp3_path.with_name(f"{track_id:06d}_{slug}.mp3")

        if new_path.exists():
            raise RuntimeError(
                f"Target filename already exists: {new_path}. Resolve conflict manually."
            )

        print(f"Renaming {mp3_path} -> {new_path}")
        mp3_path.rename(new_path)

    print("Renaming complete.")


if __name__ == "__main__":
    samples_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLES_CSV
    dest_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DEST_DIR

    try:
        main(samples_arg, dest_arg)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
