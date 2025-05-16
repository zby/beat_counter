"""
Registry for beat detectors.

This module provides functions to register and retrieve beat detector implementations.
"""
from typing import Dict, Type, Callable, Any

from beat_detection.core.detector_protocol import BeatDetector

# Registry of available detectors
_DETECTORS: Dict[str, Type[BeatDetector]] = {}

def register(name: str) -> Callable:
    """
    Decorator to register a beat detector implementation.
    
    Parameters
    ----------
    name : str
        Name of the detector to register.
        
    Returns
    -------
    Callable
        Decorator function that registers the class.
        
    Examples
    --------
    >>> @register("my_detector")
    >>> class MyDetector(BaseBeatDetector):
    >>>     ...
    """
    def decorator(cls: Type[BeatDetector]) -> Type[BeatDetector]:
        if name in _DETECTORS:
            raise ValueError(f"Detector with name '{name}' is already registered")
        _DETECTORS[name] = cls
        return cls
    return decorator

def get(algorithm: str, **kwargs: Any) -> BeatDetector:
    """
    Get a beat detector instance based on the algorithm name.
    
    Parameters
    ----------
    algorithm : str
        Name of the beat detection algorithm to use.
    **kwargs : Any
        Additional keyword arguments to pass to the detector constructor.
        
    Returns
    -------
    BeatDetector
        An instance of the requested beat detector.
        
    Raises
    ------
    ValueError
        If the requested algorithm is not supported.
    """
    if algorithm not in _DETECTORS:
        supported = ", ".join(f'"{name}"' for name in _DETECTORS.keys())
        raise ValueError(
            f'Unsupported beat detection algorithm: "{algorithm}". '
            f'Supported algorithms are: {supported}.'
        )
    
    detector_class = _DETECTORS[algorithm]
    
    # Filtering kwargs is handled by each detector class
    return detector_class(**kwargs) 