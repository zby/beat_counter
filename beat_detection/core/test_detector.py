import numpy as np
import pytest
from beat_detection.core.detector import BeatDetector

def test_align_downbeats_to_beats_provided_case():
    """Test the alignment function with the specific problematic case provided."""
    # Instantiate a dummy detector just to call the private method
    # Parameters like min_bpm don't matter for this specific test
    detector = BeatDetector() 

    beat_timestamps = np.array([
        0.33, 0.77, 1.23, 1.69, 2.14, 2.59, 3.04, 3.49, 3.94, 4.39, 
        4.84, 5.29, 5.75, 6.20, 6.65, 7.10, 7.55, 8.00, 8.45, 8.90, 
        9.36, 9.81
    ])
    downbeat_timestamps = np.array([0.75, 2.58, 4.38, 6.19, 8.00, 9.80])
    
    # Test 1: Check with the old small tolerance that should fail (as originally observed)
    # This confirms the original issue was the small default tolerance.
    small_tolerance = 1e-3 
    expected_indices_small_tolerance = [17] # Only the exact match
    actual_indices_small_tolerance = detector._align_downbeats_to_beats(
        beat_timestamps, 
        downbeat_timestamps,
        search_tolerance=small_tolerance # Explicitly use the small tolerance
    )
    assert actual_indices_small_tolerance == expected_indices_small_tolerance, \
        f"With explicit small tolerance {small_tolerance}, expected {expected_indices_small_tolerance}, but got {actual_indices_small_tolerance}"

    # Test 2: Test with the *current* default tolerance (implicitly used)
    # Default is now 0.02, which should match all cases here.
    # abs(0.77 - 0.75) = 0.02 <= 0.02 -> Match idx 1
    # abs(2.59 - 2.58) = 0.01 <= 0.02 -> Match idx 5
    # abs(4.39 - 4.38) = 0.01 <= 0.02 -> Match idx 9
    # abs(6.20 - 6.19) = 0.01 <= 0.02 -> Match idx 13
    # abs(8.00 - 8.00) = 0.00 <= 0.02 -> Match idx 17
    # abs(9.81 - 9.80) = 0.01 <= 0.02 -> Match idx 21
    expected_indices_default_tolerance = [1, 5, 9, 13, 17, 21] 
    actual_indices_default_tolerance = detector._align_downbeats_to_beats(
        beat_timestamps,
        downbeat_timestamps
        # No search_tolerance parameter passed - uses method default
    )
    assert actual_indices_default_tolerance == expected_indices_default_tolerance, \
        f"With default tolerance, expected {expected_indices_default_tolerance}, but got {actual_indices_default_tolerance}" 