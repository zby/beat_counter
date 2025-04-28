#!/usr/bin/env python3
"""
sample_tracks_by_genre.py

Create per-genre track samples for genres listed in ``data/fma/danceable_genres_limits.csv``.

For every genre ID in that CSV we look up *all* tracks that belong to the genre or any of
its sub-genres in ``tracks.csv`` from the FMA metadata.  We then pick up to ``limit``
tracks for the genre trying to maximise artist diversity:

1. First we keep at most one track per artist until either we reach the limit or we
   exhausted all artists.
2. If we still need more tracks we perform a second pass, again taking at most one extra
   track per artist, and so on, until the requested number is reached or we run out of
   tracks.

The resulting sample is written to ``data/fma/danceable_genre_samples.csv`` with the
following columns::

    genre_id,genre_title,track_id,track_title,artist_name

The script is *fail-fast*: any unrecoverable problem causes an exception instead of being
silently ignored.
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import sys
import textwrap
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# -----------------------------------------------------------------------------
# Helper loaders – copied (slightly simplified) from find_tracks_by_genre.py
# -----------------------------------------------------------------------------


def load_tracks(tracks_csv: Path) -> pd.DataFrame:
    """Load *tracks.csv* via ``fma.utils.load`` and perform minimal post-processing."""
    if not tracks_csv.is_file():
        raise FileNotFoundError(tracks_csv)

    # Use the canonical loader from the *fma* package -------------------------
    from fma import utils as fma_utils  # local import keeps top-level clean

    tracks = fma_utils.load(str(tracks_csv))

    # Flatten the MultiIndex columns created by fma for easier downstream use
    if isinstance(tracks.columns, pd.MultiIndex):
        tracks.columns = ["_".join(col).strip("_") for col in tracks.columns.values]

    # Sanity-check required columns ------------------------------------------
    required = {"track_title", "artist_name", "track_genres_all"}
    missing = required.difference(tracks.columns)
    if missing:
        raise RuntimeError(
            f"tracks.csv missing required columns after load: {', '.join(sorted(missing))}"
        )

    # ``fma.utils.load`` already converts list-like columns to Python lists,
    # but ensure "track_genres_all" is list-typed just in case.
    if tracks["track_genres_all"].dtype != "O":
        tracks["track_genres_all"] = (
            tracks["track_genres_all"].fillna("[]").apply(ast.literal_eval)
        )

    # Guarantee str types for text columns -----------------------------------
    tracks["track_title"] = tracks["track_title"].astype(str).fillna("[No Title]")
    tracks["artist_name"] = tracks["artist_name"].astype(str).fillna("[No Artist]")

    return tracks


def load_genres(genres_csv: Path) -> pd.DataFrame:
    """Load *genres.csv* via ``fma.utils.load`` and return only parent/title."""
    if not genres_csv.is_file():
        raise FileNotFoundError(genres_csv)

    from fma import utils as fma_utils

    genres_df = fma_utils.load(str(genres_csv))

    # Keep only the columns we need
    for col in ("parent", "title"):
        if col not in genres_df.columns:
            raise RuntimeError(f"Column '{col}' missing from genres.csv after load")
    return genres_df[["parent", "title"]]


# -----------------------------------------------------------------------------
# Descendant genre utility
# -----------------------------------------------------------------------------


def build_parent_map(genres_df: pd.DataFrame) -> Dict[int, List[int]]:
    parent_map: Dict[int, List[int]] = defaultdict(list)
    for gid, row in genres_df.iterrows():
        parent = int(row["parent"])
        if parent != 0:
            parent_map[parent].append(int(gid))
    return parent_map


def collect_descendants(genre_id: int, parent_map: Dict[int, List[int]]) -> Set[int]:
    """Return *genre_id* plus all its descendant ids (depth-first)."""
    result: Set[int] = set()
    queue: deque[int] = deque([genre_id])
    while queue:
        current = queue.popleft()
        if current in result:
            continue
        result.add(current)
        queue.extend(parent_map.get(current, []))
    return result


# -----------------------------------------------------------------------------
# Sampling logic
# -----------------------------------------------------------------------------


def diverse_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Return *limit* rows from *df* maximising artist diversity.

    The dataframe **must** contain an ``artist_name`` column – this function does no
    validation to keep the hot-path tight.
    """
    if len(df) <= limit:
        return df.copy()

    # build per-artist queues --------------------------------------------------
    artist_to_tracks: Dict[str, deque[int]] = defaultdict(deque)
    for idx, row in df.sample(frac=1, random_state=random.randint(0, 1 << 30)).iterrows():
        artist_to_tracks[row["artist_name"]].append(idx)

    sample_ids: List[int] = []
    artists_cycle = deque(artist_to_tracks.keys())

    while artists_cycle and len(sample_ids) < limit:
        artist = artists_cycle.popleft()
        tracks = artist_to_tracks[artist]
        if tracks:
            sample_ids.append(tracks.popleft())
            # re-queue artist if they still have tracks left and we still need more
            if tracks and len(sample_ids) < limit:
                artists_cycle.append(artist)

    return df.loc[sample_ids]


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Create a track sample for each genre in danceable_genres_limits.csv.
            """
        )
    )
    p.add_argument(
        "metadata_dir",
        type=Path,
        nargs="?",
        default=Path("data/fma/fma_metadata"),
        help="Path to directory containing FMA metadata CSVs (genres.csv, tracks.csv). Defaults to 'data/fma/fma_metadata/'",
    )
    p.add_argument(
        "--genres-file",
        default=Path("data/fma/danceable_genres_limits.csv"),
        type=Path,
        help="CSV with columns 'genre_id,title,limit'.  Defaults to data/fma/danceable_genres_limits.csv",
    )
    p.add_argument(
        "--output",
        default=Path("data/fma/danceable_genre_samples.csv"),
        type=Path,
        help="Where to write the resulting CSV.",
    )
    return p.parse_args()



def main() -> None:  # noqa: C901  complex but fine for a script
    args = parse_args()

    genres_csv = args.metadata_dir / "genres.csv"
    tracks_csv = args.metadata_dir / "tracks.csv"

    genres_df = load_genres(genres_csv)
    tracks_df = load_tracks(tracks_csv)

    if not args.genres_file.is_file():
        raise FileNotFoundError(args.genres_file)

    limits_df = pd.read_csv(args.genres_file)
    for col in ("genre_id", "title", "limit"):
        if col not in limits_df.columns:
            raise RuntimeError(
                f"{args.genres_file} must contain a '{col}' column – found {list(limits_df.columns)}"
            )

    parent_map = build_parent_map(genres_df)

    all_samples: List[pd.DataFrame] = []

    for _, row in limits_df.iterrows():
        gid = int(row["genre_id"])
        limit = int(row["limit"])

        if gid not in genres_df.index:
            print(f"WARNING: genre_id {gid} not found in genres.csv – skipping.")
            continue

        descendants = collect_descendants(gid, parent_map)

        # filter tracks belonging to any of the descendant genres -------------
        relevant = tracks_df[tracks_df["track_genres_all"].apply(
            lambda genres: bool(set(map(int, genres)).intersection(descendants))
        )]

        if relevant.empty:
            print(f"NOTE: no tracks found for genre {gid} – skipping.")
            continue

        sample_df = diverse_sample(relevant, limit)
        sample_df = sample_df.assign(
            genre_id=gid,
            genre_title=genres_df.loc[gid, "title"],
        )
        # reorder columns
        sample_df = sample_df.reset_index().rename(columns={"index": "track_id"})[
            [
                "genre_id",
                "genre_title",
                "track_id",
                "track_title",
                "artist_name",
            ]
        ]
        all_samples.append(sample_df)

        print(
            f"Selected {len(sample_df)}/{len(relevant)} tracks for genre {gid} "
            f"({genres_df.loc[gid, 'title']})."
        )

    if not all_samples:
        raise RuntimeError("No samples generated – check input data.")

    result = pd.concat(all_samples, ignore_index=True)

    # ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} total sampled tracks to {args.output}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # fail fast – propagate error after a descriptive message
        print(f"ERROR: {exc}")
        sys.exit(1) 