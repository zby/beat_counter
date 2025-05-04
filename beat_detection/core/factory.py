"""
Factory for creating beat detectors.
"""
from typing import Dict, Type, Optional, Any
import inspect  # Add inspect import
import warnings # Add warnings import

from beat_detection.core.detector import BeatDetector
from beat_detection.core.madmom_detector import MadmomBeatDetector
from beat_detection.core.beat_this_detector import BeatThisDetector

# Registry of available detectors
DETECTOR_REGISTRY: Dict[str, Type[BeatDetector]] = {
    "madmom": MadmomBeatDetector,
    "beat_this": BeatThisDetector,
}

def get_beat_detector(algorithm: str = "madmom", **kwargs: Any) -> BeatDetector:
    """
    Factory function to get a beat detector instance based on the algorithm name.
    
    Parameters
    ----------
    algorithm : str
        Name of the beat detection algorithm to use.
        Currently supported: "madmom" (default), "beat_this"
    **kwargs : Any
        Additional keyword arguments to pass to the detector constructor.
        Common parameters:
        - progress_callback: Callback function for progress updates
        
        MadmomBeatDetector specific parameters:
        - min_bpm: Minimum BPM to consider
        - max_bpm: Maximum BPM to consider
        - fps: Frames per second for processing
        
        BeatThisDetector specific parameters:
        - file2beats_processor: Custom processor to use
        
    Returns
    -------
    BeatDetector
        An instance of the requested beat detector.
        
    Raises
    ------
    ValueError
        If the requested algorithm is not supported.
    """
    if algorithm not in DETECTOR_REGISTRY:
        supported = ", ".join(f'"{name}"' for name in DETECTOR_REGISTRY.keys())
        raise ValueError(
            f'Unsupported beat detection algorithm: "{algorithm}". '
            f'Supported algorithms are: {supported}.'
        )
    
    detector_class = DETECTOR_REGISTRY[algorithm]
    
    # Get the signature of the detector's __init__ method
    init_signature = inspect.signature(detector_class.__init__)
    valid_params = {param.name for param in init_signature.parameters.values()}
    
    # Filter kwargs to only include valid parameters for the detector's __init__
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in valid_params
    }
    
    # Check for extraneous arguments and issue a warning
    extraneous_kwargs = {key for key in kwargs if key not in valid_params}
    if extraneous_kwargs:
        warnings.warn(
            f"Ignoring extraneous keyword arguments for {detector_class.__name__}: "
            f"{', '.join(extraneous_kwargs)}",
            UserWarning
        )

    # Debugging: Print filtered kwargs
    # print(f"Initializing {detector_class.__name__} with: {filtered_kwargs}")
    
    return detector_class(**filtered_kwargs)
