#!/usr/bin/env python3
"""download_danceable_tracks.py

Download all tracks listed in *data/fma/danceable_genre_samples.csv* and store them
under *data/fma/fma_tracks/<genre_id>_<slug>/*.mp3*.

The script is **fail-fast**: any problem stops execution with a clear error.

Usage
-----
    python scripts/download_danceable_tracks.py [SAMPLES_CSV] [DEST_DIR]

Both arguments are optional and default to the paths above.

Environment
-----------
Requires an FMA API key in the environment variable ``FMA_KEY`` (or ``FMA_API_KEY``).

Dependencies
------------
Relies on the *fma.utils* module already present in the repository.  No extra
packages are imported beyond the standard library and what *fma.utils*
naturally requires (``requests``).
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Local import – keep the global namespace clean
from fma import utils as fma_utils

# ---------------------------------------------------------------------------
# Default paths – overwrite via command-line arguments if desired
# ---------------------------------------------------------------------------
DEFAULT_SAMPLES_CSV = Path("data/fma/danceable_genre_samples.csv")
DEFAULT_DEST_DIR = Path("data/fma/fma_tracks")

# Name of the file that stores already-downloaded track IDs (one per line)
_COMPLETED_FILE = ".downloaded_ids"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def slugify(s: str, *, allow_unicode: bool = False) -> str:  # copied from Django
    """Return a filesystem-safe slug for *s*.

    – Converts to ASCII when *allow_unicode* is False
    – Replaces non-alnum characters with underscores
    – Strips leading/trailing underscores and collapses runs
    """

    if allow_unicode:
        value = unicodedata.normalize("NFKC", s)
    else:
        value = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    # Replace unwanted characters with underscore, keep alnum and dash
    import re

    value = re.sub(r"[^\w\-]+", "_", value)
    value = re.sub(r"_+", "_", value)  # collapse multiples
    return value.strip("_")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(samples_csv: Path, dest_dir: Path) -> None:
    # --- Validation ---------------------------------------------------------
    if not samples_csv.is_file():
        raise FileNotFoundError(samples_csv)

    api_key = os.getenv("FMA_KEY") or os.getenv("FMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing FMA API key – set the FMA_KEY (or FMA_API_KEY) environment variable."
        )

    # Load CSV – fail if required columns are missing ------------------------
    df = pd.read_csv(samples_csv)

    required_cols = {"genre_id", "genre_title", "track_id"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"{samples_csv} missing required columns: {', '.join(sorted(missing))}"
        )

    # Prepare output directory ----------------------------------------------
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Determine which track IDs were already downloaded (restartability)
    # ---------------------------------------------------------------------
    completed_ids_path = dest_dir / _COMPLETED_FILE

    # First try to read the explicit list (fast).  If it does not exist,
    # fall back to scanning the directory structure once and create it for
    # subsequent restarts.
    completed_ids: set[int] = set()

    if completed_ids_path.exists():
        with completed_ids_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.isdigit():
                    completed_ids.add(int(line))
    else:
        # One-time scan ------------------------------------------------------
        import re

        id_pattern = re.compile(r"(\d+)(?:\.mp3)?$")
        for mp3 in dest_dir.rglob("*.mp3"):
            match = id_pattern.search(mp3.stem)
            if match:
                completed_ids.add(int(match.group(1)))

        # Persist the discovered IDs for the next run
        if completed_ids:
            with completed_ids_path.open("w", encoding="utf-8") as fh:
                fh.writelines(f"{tid}\n" for tid in sorted(completed_ids))

    # Keep the file handle open in append mode for fast updates -------------
    completed_append = completed_ids_path.open("a", encoding="utf-8")
    # Ensure we flush each write so progress is durable even if interrupted
    import atexit

    @atexit.register
    def _close_completed_append():
        completed_append.close()

    # Instantiate FMA helper -------------------------------------------------
    fma = fma_utils.FreeMusicArchive(api_key)

    # Iterate deterministically to ease resumption/debugging -----------------
    for _, row in df.sort_values(["genre_id", "track_id"]).iterrows():
        gid = int(row["genre_id"])
        title = str(row["genre_title"])
        tid = int(row["track_id"])

        genre_slug = slugify(title) or str(gid)
        genre_path = dest_dir / f"{gid}_{genre_slug}"
        genre_path.mkdir(parents=True, exist_ok=True)

        # We fetch track_file via the API – fail early on error --------------
        track_file: str = fma.get_track(tid, "track_file")  # type: ignore[assignment]
        filename = Path(track_file).name
        out_path = genre_path / filename

        if tid in completed_ids or out_path.exists():
            # Already downloaded – skip fast
            continue

        print(f"Downloading {tid} -> {out_path}…", flush=True)
        try:
            fma.download_track(track_file, str(out_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to download track {tid}: {exc}") from exc

        # Mark as completed --------------------------------------------------
        completed_append.write(f"{tid}\n")
        completed_append.flush()

    print("All tracks downloaded successfully.")


if __name__ == "__main__":
    # Resolve optional CLI args ----------------------------------------------
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLES_CSV
    dest_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DEST_DIR

    try:
        main(csv_arg, dest_arg)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1) 