#!/usr/bin/env python3
"""extract_danceable_tracks.py

Extract only the MP3s listed in *data/fma/danceable_genre_samples.csv* from the
remote *fma_full.zip* archive hosted at
https://os.unil.cloud.switch.ch/fma/fma_full.zip.

The script streams the ZIP file via HTTP range requests using *fsspec* so it
never downloads more than what is strictly required.

Files are written to:
    data/fma/fma_tracks/<genre_id>_<slug>/<tid>.mp3

The run is **restartable**: already-extracted track IDs are read from a
".downloaded_ids" file in the destination folder (or inferred by scanning once)
and skipped immediately.  Successful downloads are appended to that file in
real-time, so an interruption will preserve progress.

Usage
-----
    python scripts/extract_danceable_tracks.py [SAMPLES_CSV] [DEST_DIR] [ZIP_URL]

All arguments are optional.  Defaults:
    SAMPLES_CSV = data/fma/danceable_genre_samples.csv
    DEST_DIR    = data/fma/fma_tracks
    ZIP_URL     = https://os.unil.cloud.switch.ch/fma/fma_full.zip

Environment
-----------
No API keys required.

Dependencies
------------
    uv pip install "fsspec[http]" pandas

The rest relies on the Python standard library.
"""

from __future__ import annotations

import atexit
import re
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Set

import fsspec
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults – can be overridden via CLI
# ---------------------------------------------------------------------------
DEFAULT_SAMPLES_CSV = Path("data/fma/danceable_genre_samples.csv")
DEFAULT_DEST_DIR = Path("data/fma/fma_tracks")
DEFAULT_ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"

# Name of the file storing completed IDs for restartability
_COMPLETED_FILE = ".downloaded_ids"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def slugify(s: str, *, allow_unicode: bool = False) -> str:
    """Return a slug suitable for directory names."""

    if allow_unicode:
        value = unicodedata.normalize("NFKC", s)
    else:
        value = (
            unicodedata.normalize("NFKD", s)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = value.lower()
    value = re.sub(r"[^\w\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def track_path_in_zip(track_id: int, root: str = "fma_full") -> str:
    """Return the member path inside the ZIP for *track_id*."""

    tid_str = f"{track_id:06d}"
    return f"{root}/{tid_str[:3]}/{tid_str}.mp3"


# ---------------------------------------------------------------------------
# Restartability helpers
# ---------------------------------------------------------------------------

def load_completed(dest_dir: Path) -> Set[int]:
    """Load already-downloaded track IDs from the marker file *or* scan once."""

    completed_path = dest_dir / _COMPLETED_FILE
    completed: Set[int] = set()

    if completed_path.exists():
        with completed_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.isdigit():
                    completed.add(int(line))
    else:
        pattern = re.compile(r"^(\d{6})(?:_[^\.]+)?\.mp3$")
        for mp3 in dest_dir.rglob("*.mp3"):
            match = pattern.match(mp3.name)
            if match:
                completed.add(int(match.group(1)))

        if completed:
            with completed_path.open("w", encoding="utf-8") as fh:
                fh.writelines(f"{tid}\n" for tid in sorted(completed))

    # Keep file open in append mode for updates
    append_fh = completed_path.open("a", encoding="utf-8")

    def _close():
        append_fh.close()

    atexit.register(_close)

    return completed, append_fh


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def main(samples_csv: Path, dest_dir: Path, zip_url: str) -> None:
    # --- Validate inputs ----------------------------------------------------
    if not samples_csv.is_file():
        raise FileNotFoundError(samples_csv)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load list of target tracks -------------------------------------------
    df = pd.read_csv(samples_csv)
    required_cols = {"genre_id", "genre_title", "track_id"}
    # track_title is optional but preferred for nicer file names
    has_title_col = "track_title" in df.columns
    if has_title_col:
        required_cols_with_title = required_cols | {"track_title"}
    else:
        required_cols_with_title = required_cols

    missing = required_cols.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"{samples_csv} missing required columns: {', '.join(sorted(missing))}"
        )

    # Load / build completed set -------------------------------------------
    completed, completed_append = load_completed(dest_dir)

    # Open remote ZIP via fsspec -------------------------------------------
    print(f"Opening remote ZIP at {zip_url} …")
    fs = fsspec.filesystem("http")
    with fs.open(zip_url, mode="rb") as http_file:
        with zipfile.ZipFile(http_file) as z:
            # Iterate in deterministic order for debuggability --------------
            for _, row in df.sort_values(["genre_id", "track_id"]).iterrows():
                tid = int(row["track_id"])
                if tid in completed:
                    continue

                gid = int(row["genre_id"])
                gtitle = str(row["genre_title"])
                slug = slugify(gtitle) or str(gid)
                genre_dir = dest_dir / f"{gid}_{slug}"
                genre_dir.mkdir(parents=True, exist_ok=True)

                member = track_path_in_zip(tid)

                # Build human-readable filename
                track_slug = slugify(str(row["track_title"]))[:60] if has_title_col else "track"
                out_path = genre_dir / f"{tid:06d}_{track_slug}.mp3"

                # Additional fast path: if file somehow exists but ID not in set
                if out_path.exists():
                    completed.add(tid)
                    completed_append.write(f"{tid}\n")
                    completed_append.flush()
                    continue

                print(f"Extracting {member} -> {out_path}")

                try:
                    with z.open(member) as src, out_path.open("wb") as dst:
                        import shutil

                        shutil.copyfileobj(src, dst)
                except KeyError:
                    raise RuntimeError(f"Track {tid} not found in ZIP archive ({member}).")
                except Exception as exc:
                    raise RuntimeError(f"Failed to extract {tid}: {exc}") from exc

                # Mark completion -----------------------------------------
                completed.add(tid)
                completed_append.write(f"{tid}\n")
                completed_append.flush()

    print("All requested tracks extracted successfully.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLES_CSV
    dest_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DEST_DIR
    url_arg = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_ZIP_URL

    try:
        main(samples_arg, dest_arg, url_arg)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1) 