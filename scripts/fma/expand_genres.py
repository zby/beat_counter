import pandas as pd
import sys

# Define file paths
GENRES_FILE = "data/fma/fma_metadata/genres.csv"
DANCEABLE_GENRES_FILE = "data/fma/danceable_genres.csv"
OUTPUT_FILE = "data/fma/all_danceable_genres.csv"


def build_genre_tree(genres_df: pd.DataFrame) -> dict[int, list[int]]:
    """Builds the genre hierarchy."""
    tree = {}
    for _, row in genres_df.iterrows():
        parent_id = int(row["parent"])
        genre_id = int(row["genre_id"])
        if parent_id != 0:  # Exclude top-level links for tree building
            if parent_id not in tree:
                tree[parent_id] = []
            tree[parent_id].append(genre_id)

    return tree


def get_all_descendants(
    genre_id: int, tree: dict[int, list[int]]
) -> set[int]:
    """Recursively finds all descendant genres (including the starting one) of a given genre ID."""
    descendants = {genre_id} # Include the current node
    if genre_id in tree:
        for child_id in tree[genre_id]:
            descendants.update(get_all_descendants(child_id, tree))
    return descendants


def main():
    """Main function to load data, process genres, and save output."""
    try:
        # Read genres, explicitly handle potential leading blank lines (though usually default)
        # Removed comment='#' as it might interfere with '#tracks' column header
        genres_df = pd.read_csv(GENRES_FILE, skip_blank_lines=True)
        danceable_genres_df = pd.read_csv(DANCEABLE_GENRES_FILE)
    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Column Validation and Cleaning ---
    # 1. Validate original column names from genres.csv
    original_genre_cols = {"genre_id", "#tracks", "parent", "title", "top_level"}
    if not original_genre_cols.issubset(genres_df.columns):
        # Find missing columns for a clearer error message
        missing_cols = original_genre_cols - set(genres_df.columns)
        print(
            f"Error: Missing expected columns in {GENRES_FILE}. "
            f"Missing: {missing_cols}. Found: {list(genres_df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Clean genre column names (strip leading/trailing spaces and '#')
    genres_df.columns = genres_df.columns.str.strip().str.lstrip("#")
    # 3. Clean danceable genre column names
    danceable_genres_df.columns = danceable_genres_df.columns.str.strip()

    # Ensure required columns exist *after* cleaning
    required_genre_cols = {"genre_id", "parent", "title"}
    # Check using the cleaned column names
    if not required_genre_cols.issubset(genres_df.columns):
        # This check should ideally not fail if the original check passed,
        # but added for extra safety.
        missing_cols = required_genre_cols - set(genres_df.columns)
        print(
            f"Error: Missing required columns after cleaning in {GENRES_FILE}. "
            f"Need: {required_genre_cols}. Missing: {missing_cols}. Current: {list(genres_df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    required_danceable_cols = {"genre_id"}
    if not required_danceable_cols.issubset(danceable_genres_df.columns):
        missing_cols = required_danceable_cols - set(danceable_genres_df.columns)
        print(
            f"Error: Missing required columns in {DANCEABLE_GENRES_FILE}. "
            f"Need: {required_danceable_cols}. Missing: {missing_cols}. Current: {list(danceable_genres_df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Convert genre_id and parent to int for consistency
    try:
        genres_df["genre_id"] = genres_df["genre_id"].astype(int)
        genres_df["parent"] = genres_df["parent"].astype(int)
        danceable_genres_df["genre_id"] = danceable_genres_df["genre_id"].astype(int)
    except ValueError as e:
        print(f"Error converting genre_id or parent to integer: {e}", file=sys.stderr)
        sys.exit(1)

    # Build the genre tree (leaf_nodes no longer needed)
    tree = build_genre_tree(genres_df)

    # Find all descendant genres for the initial danceable genres
    all_danceable_ids = set()
    initial_danceable_ids = set(danceable_genres_df["genre_id"])
    all_valid_genre_ids = set(genres_df["genre_id"]) # For checking existence

    for genre_id in initial_danceable_ids:
        if genre_id not in all_valid_genre_ids:
            print(
                f"Warning: Danceable genre ID {genre_id} not found in {GENRES_FILE}. Skipping.",
                file=sys.stderr,
            )
            continue
        # Use the new function to get all descendants
        all_danceable_ids.update(get_all_descendants(genre_id, tree))

    # Create the final DataFrame using all descendant IDs
    all_descendant_genres_df = genres_df[genres_df["genre_id"].isin(all_danceable_ids)][
        ["genre_id", "title"]
    ].copy()  # Select only relevant columns

    # Sort by genre_id for consistent output
    all_descendant_genres_df = all_descendant_genres_df.sort_values(by="genre_id").reset_index(drop=True)

    # Reorder columns for clarity (optional, but good practice)
    all_descendant_genres_df = all_descendant_genres_df[["genre_id", "title"]]

    # Save the result
    try:
        all_descendant_genres_df.to_csv(OUTPUT_FILE, index=False)
        print(
            f"Successfully created {OUTPUT_FILE} with {len(all_descendant_genres_df)} danceable genres (including descendants)."
        )
    except Exception as e:
        print(f"Error writing output file {OUTPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
