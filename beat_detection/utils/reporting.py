"""
Reporting and statistics utilities.
"""

import numpy as np
import pathlib
import json
from typing import List, Dict, Tuple, Any, Optional, Union

# Import Beats class
from ..core.beats import Beats, BeatStatistics  # Assuming core is sibling to utils


def get_beat_statistics_dict(
    beats: Beats, filename: Optional[str] = None, duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convert beat statistics from a Beats object to a dictionary for JSON serialization.

    Parameters:
    -----------
    beats : Beats
        The Beats object containing the statistics.
    filename : str, optional
        Original filename to include in the report
    duration : float, optional
        Duration of the audio in seconds

    Returns:
    --------
    dict
        Dictionary containing beat statistics formatted for reporting.
    """
    overall_stats = beats.overall_stats
    regular_stats = beats.regular_stats
    irregular_indices = beats.irregular_beat_indices

    stats_dict = {
        "tempo_bpm": round(overall_stats.tempo_bpm, 1),
        "mean_interval": round(overall_stats.mean_interval, 3),
        "median_interval": round(overall_stats.median_interval, 3),
        "std_interval": round(overall_stats.std_interval, 3),
        "min_interval": round(overall_stats.min_interval, 3),
        "max_interval": round(overall_stats.max_interval, 3),
        "irregular_beats_count": len(irregular_indices),
        "total_beats": overall_stats.total_beats,
        # Note: overall_stats.irregularity_percent is based on interval deviation
        # Consider if a different definition based on final irregular_indices is needed here.
        "irregularity_percent": round(overall_stats.irregularity_percent, 1),
        "irregular_beat_indices": irregular_indices,
        # Add regular section stats
        "regular_section_tempo_bpm": round(regular_stats.tempo_bpm, 1),
        "regular_section_mean_interval": round(regular_stats.mean_interval, 3),
        "regular_section_median_interval": round(regular_stats.median_interval, 3),
        "regular_section_std_interval": round(regular_stats.std_interval, 3),
        "regular_section_min_interval": round(regular_stats.min_interval, 3),
        "regular_section_max_interval": round(regular_stats.max_interval, 3),
        "regular_section_irregularity_percent": round(
            regular_stats.irregularity_percent, 1
        ),
        "regular_section_total_beats": regular_stats.total_beats,
        # Add info from Beats object itself
        "beats_per_bar": beats.beats_per_bar,
        "tolerance_percent": beats.tolerance_percent,
        "regular_section_start_idx": beats.start_regular_beat_idx,
        "regular_section_end_idx": beats.end_regular_beat_idx,
    }

    if filename:
        stats_dict["filename"] = filename

    if duration is not None:
        stats_dict["duration"] = round(duration, 3)

    return stats_dict


def save_batch_summary(
    file_results: List[Tuple[str, Optional[Beats]]],
    output_file: Union[str, pathlib.Path],
) -> None:
    """
    Save summary statistics for batch processing results.

    Parameters:
    -----------
    file_results : list of (str, Optional[Beats])
        List of tuples with filename and the corresponding Beats object (or None if processing failed).
    output_file : str or pathlib.Path
        Path to the output summary text file.
    """
    if not file_results:
        return

    # Filter out None results (failed files)
    valid_results = [(fname, b) for fname, b in file_results if b is not None]

    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        if not valid_results:
            f.write("No valid statistics available for any processed files.\n")
            return

        total_processed = len(file_results)
        total_successful = len(valid_results)

        # Calculate summary statistics from Beats objects
        tempos = [b.overall_stats.tempo_bpm for _, b in valid_results]
        avg_tempo = sum(tempos) / total_successful
        min_tempo = min(tempos)
        max_tempo = max(tempos)

        # Find file with highest overall irregularity percentage
        # Using overall_stats.irregularity_percent which is based on intervals
        most_irregular = max(
            valid_results, key=lambda item: item[1].overall_stats.irregularity_percent
        )

        f.write(
            f"Summary Statistics for {total_successful} / {total_processed} Files\n"
        )
        f.write("=" * 50 + "\n\n")
        f.write(f"Average tempo: {avg_tempo:.1f} BPM\n")
        f.write(f"Tempo range: {min_tempo:.1f} - {max_tempo:.1f} BPM\n")
        f.write(
            f"Most irregular file (by interval deviation): {most_irregular[0]} ({most_irregular[1].overall_stats.irregularity_percent:.1f}%)\n\n"
        )
        f.write("Individual File Statistics:\n")
        f.write("-" * 50 + "\n")
        for filename, beats_obj in valid_results:
            f.write(
                f"{filename}: {beats_obj.overall_stats.tempo_bpm:.1f} BPM, {beats_obj.overall_stats.irregularity_percent:.1f}% irregular intervals\n"
            )

        # Add a section for files that failed
        failed_files = [fname for fname, b in file_results if b is None]
        if failed_files:
            f.write("\nFiles that failed processing:\n")
            f.write("-" * 50 + "\n")
            for fname in failed_files:
                f.write(f"{fname}\n")


def print_beat_timestamps(beats: Beats) -> None:
    """
    Print beat timestamps from a Beats object to the console.

    Parameters:
    -----------
    beats : Beats
        The Beats object containing the beat information.
    """
    print("Detected Beats (in seconds):")
    timestamps = beats.timestamps
    irregular_indices = set(beats.irregular_beat_indices)  # Use set for faster lookup
    downbeat_indices = set(beats.downbeat_indices)

    for i, timestamp in enumerate(timestamps):
        is_irregular = i in irregular_indices
        is_downbeat = i in downbeat_indices

        irregular_mark = " (irregular)" if is_irregular else ""
        downbeat_mark = " (DOWNBEAT)" if is_downbeat else ""
        # Assuming 1-based index for printing
        print(
            f"Beat {i+1} (idx {i}): {timestamp:.3f} seconds{irregular_mark}{downbeat_mark}"
        )


def print_statistics(beats: Beats) -> None:
    """
    Print overall beat statistics from a Beats object to the console.

    Parameters:
    -----------
    beats : Beats
        The Beats object containing the statistics.
    """
    stats = beats.overall_stats
    irregular_indices = beats.irregular_beat_indices

    print("\nOverall Beat Statistics:")
    print(f"Tempo (median-based): {stats.tempo_bpm:.1f} BPM")
    print(f"Mean interval: {stats.mean_interval:.3f} seconds")
    print(f"Median interval: {stats.median_interval:.3f} seconds")
    print(f"Standard deviation: {stats.std_interval:.3f} seconds")
    print(f"Min interval: {stats.min_interval:.3f} seconds")
    print(f"Max interval: {stats.max_interval:.3f} seconds")
    print(
        f"Irregular beats (final count): {len(irregular_indices)} ({len(irregular_indices)/stats.total_beats*100:.1f}%)"
    )
    print(
        f"  (Note: Initial interval irregularity was {stats.irregularity_percent:.1f}%)"
    )
    print(f"Beats per Bar: {beats.beats_per_bar}")
    print(
        f"Regular Section: Beats {beats.start_regular_beat_idx} to {beats.end_regular_beat_idx-1}"
    )
    # Optionally print regular stats too
    # print("\nRegular Section Statistics:")
    # ... print regular_stats ...


def print_batch_summary(file_results: List[Tuple[str, Optional[Beats]]]) -> None:
    """
    Print summary statistics for batch processing results to the console.

    Parameters:
    -----------
    file_results : list of (str, Optional[Beats])
        List of tuples with filename and the corresponding Beats object (or None).
    """
    if not file_results:
        return

    valid_results = [(fname, b) for fname, b in file_results if b is not None]

    if not valid_results:
        print("\nNo valid statistics available for any processed files.")
        return

    total_processed = len(file_results)
    total_successful = len(valid_results)

    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {total_processed}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_processed - total_successful}")

    # Calculate average tempo
    tempos = [b.overall_stats.tempo_bpm for _, b in valid_results]
    avg_tempo = sum(tempos) / total_successful
    min_tempo = min(tempos)
    max_tempo = max(tempos)

    print(f"Average tempo: {avg_tempo:.1f} BPM")
    print(f"Tempo range: {min_tempo:.1f} - {max_tempo:.1f} BPM")

    # Show file with most irregular beats (based on final irregular count)
    most_irregular = max(
        valid_results, key=lambda item: len(item[1].irregular_beat_indices)
    )
    irreg_count = len(most_irregular[1].irregular_beat_indices)
    irreg_percent = (
        (irreg_count / most_irregular[1].overall_stats.total_beats * 100)
        if most_irregular[1].overall_stats.total_beats > 0
        else 0
    )
    print(
        f"File with most irregular beats: {most_irregular[0]} ({irreg_count} beats, {irreg_percent:.1f}%)"
    )

    failed_files = [fname for fname, b in file_results if b is None]
    if failed_files:
        print("\nFiles that failed processing:")
        for fname in failed_files:
            print(f"- {fname}")
    print("=" * 80 + "\n")
