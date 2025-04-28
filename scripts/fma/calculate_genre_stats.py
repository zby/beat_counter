import pandas as pd
import sys
import os
import ast  # For safely evaluating string representations of lists
from fma import utils  # Import the fma utils

# Define file paths relative to the script's location or a base data directory
# Assuming script is run from the root of the project or paths are adjusted accordingly
BASE_DATA_DIR = "data/fma"
ALL_DANCEABLE_FILE = os.path.join(BASE_DATA_DIR, "all_danceable_genres.csv")
# Assuming tracks data is here, adjust if needed
TRACKS_FILE = os.path.join(BASE_DATA_DIR, "fma_metadata", "tracks.csv")
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "danceable_genres_stats.csv")

# Use tuple column names consistent with fma.utils.load output
# TRACK_ID_COLUMN = ("track", "id") # Removed, using index instead
TRACK_GENRES_COLUMN = ("track", "genres")


def parse_genre_list(genres_str):
    """Safely parse the string representation of a genre list."""
    # fma.utils might return actual lists, not strings, handle this
    if isinstance(genres_str, list):
        return [
            g for g in genres_str if isinstance(g, int)
        ]  # Return list if it's already parsed

    if (
        pd.isna(genres_str)
        or not isinstance(genres_str, str)
        or genres_str.strip() == "[]"
    ):
        return []
    try:
        # Use ast.literal_eval for safe evaluation of Python literals
        genres = ast.literal_eval(genres_str)
        # Ensure it's a list of integers
        if isinstance(genres, list) and all(isinstance(g, int) for g in genres):
            return genres
        else:
            # Handle cases like a single int being parsed, or non-int elements
            if isinstance(genres, int):
                return [genres]  # Convert single int to list
            print(
                f"Warning: Unexpected format in genre list '{genres_str}'. Returning empty list.",
                file=sys.stderr,
            )
            return []
    except (ValueError, SyntaxError, TypeError) as e:
        print(
            f"Warning: Could not parse genre list '{genres_str}': {e}. Returning empty list.",
            file=sys.stderr,
        )
        return []


def main():
    """Loads data using fma.utils, counts tracks for danceable genres, saves stats."""
    # --- Load Data ---
    try:
        # Load danceable genres (derived file, use standard pandas)
        danceable_df = pd.read_csv(ALL_DANCEABLE_FILE)

        # Load tracks.csv using fma.utils
        print(f"Loading {TRACKS_FILE} using fma.utils.load()...")
        tracks_df = utils.load(TRACKS_FILE)
        print("Tracks file loaded.")
        # Set index name explicitly if it's not already set (utils.load might do this)
        if tracks_df.index.name is None:
            tracks_df.index.name = "track_id"  # Assuming the index is the track ID

        # --- Validate Columns from fma.utils output (MultiIndex) ---
        # Removed check for TRACK_ID_COLUMN
        # if TRACK_ID_COLUMN not in tracks_df.columns:
        #     print(f"Error: Track ID column tuple {TRACK_ID_COLUMN} not found in loaded tracks data. Found columns: {tracks_df.columns.tolist()}", file=sys.stderr)
        #     sys.exit(1)
        if TRACK_GENRES_COLUMN not in tracks_df.columns:
            print(
                f"Error: Track genres column tuple {TRACK_GENRES_COLUMN} not found in loaded tracks data. Found columns: {tracks_df.columns.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Select only the necessary column and flatten its name
        # Track ID is now the index
        # Make a copy to avoid SettingWithCopyWarning later and flatten
        tracks_df = tracks_df[[TRACK_GENRES_COLUMN]].copy()
        # Flatten the MultiIndex column to a single-level name 'genres' for simpler processing
        tracks_df.columns = ["genres"]

    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError:
        print(
            f"Error: Failed to import fma.utils. Make sure fma/utils.py exists and is importable.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        # Catch potential errors from utils.load() itself
        print(
            f"Error loading or processing {TRACKS_FILE} using fma.utils: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Validate Columns in danceable_df ---
    required_danceable_cols = {"genre_id", "title"}
    if not required_danceable_cols.issubset(danceable_df.columns):
        missing = required_danceable_cols - set(danceable_df.columns)
        print(
            f"Error: Missing columns in {ALL_DANCEABLE_FILE}: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Data Preparation ---
    try:
        # Convert danceable genre_id to int
        danceable_df["genre_id"] = danceable_df["genre_id"].astype(int)
        # Ensure track_id (index) is integer - this might already be int from utils.load
        # We don't need to convert it explicitly here as it's the index
        # tracks_df[TRACK_ID_COLUMN] = tracks_df[TRACK_ID_COLUMN].astype(int) # Removed
    except ValueError as e:
        print(f"Error converting column to integer: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(
            f"Error: Missing expected column during type conversion: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get the set of target genre IDs
    target_genre_ids = set(danceable_df["genre_id"])

    # Parse the genre lists in tracks_df (accessing via tuple)
    # Note: fma.utils might already parse this, parse_genre_list handles list input
    print(f"Parsing column {TRACK_GENRES_COLUMN}... This might take a while.")
    tracks_df["parsed_genres"] = tracks_df["genres"].apply(parse_genre_list)
    print("Parsing complete.")

    # Drop rows where parsing failed or resulted in empty lists
    tracks_df = tracks_df[tracks_df["parsed_genres"].map(len) > 0]

    # Reset index *before* exploding to handle potential MultiIndex issues
    tracks_df.reset_index(inplace=True)

    # Explode the DataFrame to have one row per track-genre pair
    # We now need 'track_id' (from reset_index) and 'parsed_genres'
    exploded_genres = tracks_df[["track_id", "parsed_genres"]].explode("parsed_genres")
    # Rename the exploded column
    exploded_genres.rename(columns={"parsed_genres": "genre_id"}, inplace=True)
    # No need to reset_index again, as it was done before explode

    # Ensure the exploded genre_id is integer
    exploded_genres["genre_id"] = exploded_genres["genre_id"].astype(int)

    # Ensure track_id is integer (it should be, but safer to check)
    exploded_genres["track_id"] = exploded_genres["track_id"].astype(int)

    # --- Calculate Track Counts ---
    # Filter for target genres
    filtered_genres = exploded_genres[
        exploded_genres["genre_id"].isin(target_genre_ids)
    ]

    # Count unique tracks per genre (using the 'track_id' column)
    # Use nunique() on track_id to count distinct tracks per genre
    genre_counts = (
        filtered_genres.groupby("genre_id")["track_id"]
        .nunique()
        .reset_index(name="track_count")
    )

    # --- Create Output DataFrame ---
    # Merge counts with the original danceable genre titles
    # Use left merge to keep all genres from all_danceable_genres.csv, filling missing counts with 0
    stats_df = pd.merge(
        danceable_df[["genre_id", "title"]], genre_counts, on="genre_id", how="left"
    )

    # Fill NaN counts with 0 (for genres that had no tracks assigned)
    stats_df["track_count"] = stats_df["track_count"].fillna(0).astype(int)

    # Sort by genre_id for consistent output
    stats_df = stats_df.sort_values(by="genre_id").reset_index(drop=True)

    # --- Save Results ---
    try:
        stats_df.to_csv(OUTPUT_FILE, index=False)
        print(
            f"Successfully created {OUTPUT_FILE} with direct track counts for {len(stats_df)} genres."
        )
    except Exception as e:
        print(f"Error writing output file {OUTPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
