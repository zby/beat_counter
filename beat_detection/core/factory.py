"""
Factory for creating beat detectors.
"""
from typing import Dict, Type, Optional, Any

from beat_detection.core.detector import BeatDetector, MadmomBeatDetector
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
    
    # Filter kwargs based on the detector type
    filtered_kwargs = {}
    
    # Common parameters for all detectors
    if "progress_callback" in kwargs:
        filtered_kwargs["progress_callback"] = kwargs["progress_callback"]
    
    # Algorithm-specific parameters
    if algorithm == "madmom":
        # MadmomBeatDetector parameters
        for param in ["min_bpm", "max_bpm", "fps"]:
            if param in kwargs:
                filtered_kwargs[param] = kwargs[param]
    elif algorithm == "beat_this":
        # BeatThisDetector parameters
        if "file2beats_processor" in kwargs:
            filtered_kwargs["file2beats_processor"] = kwargs["file2beats_processor"]
    
    return detector_class(**filtered_kwargs)
