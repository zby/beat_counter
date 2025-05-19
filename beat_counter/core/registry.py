"""
Registry for beat detectors.

This module provides functions to register and retrieve beat detector implementations.
"""
import warnings
from typing import Dict, Type, Callable, Any

from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.detectors.base import DetectorConfig

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

def build(name: str, config: DetectorConfig = None, **kwargs: Any) -> BeatDetector:
    """
    Build a detector using a typed config or keyword arguments.
    
    Parameters
    ----------
    name : str
        Name of the beat detection algorithm to use
    config : DetectorConfig, optional
        Configuration object with detector parameters. If None, one will be created from kwargs.
    **kwargs : Any
        Additional keyword arguments to pass to the detector constructor or create a config.
        
    Returns
    -------
    BeatDetector
        An instance of the requested beat detector.
        
    Raises
    ------
    ValueError
        If the requested algorithm is not supported.
    """
    if name not in _DETECTORS:
        supported = ", ".join(f'"{n}"' for n in _DETECTORS.keys())
        raise ValueError(
            f'Unsupported beat detection algorithm: "{name}". '
            f'Supported algorithms are: {supported}.'
        )
    
    detector_class = _DETECTORS[name]
    
    # If no config provided, build one from kwargs
    if config is None:
        # Extract config params from kwargs
        config_params = {}
        
        # Only pass recognized parameters to the config
        if 'min_bpm' in kwargs:
            config_params['min_bpm'] = kwargs.pop('min_bpm')
        if 'max_bpm' in kwargs:
            config_params['max_bpm'] = kwargs.pop('max_bpm')
        if 'fps' in kwargs:
            config_params['fps'] = kwargs.pop('fps')
        if 'beats_per_bar' in kwargs:
            config_params['beats_per_bar'] = kwargs.pop('beats_per_bar')
            
        config = DetectorConfig(**config_params)
    
    # Pass the config plus any remaining kwargs to the detector
    return detector_class(config, **kwargs)

def get(algorithm: str, **kwargs: Any) -> BeatDetector:
    """
    Get a beat detector instance based on the algorithm name.
    
    DEPRECATED: Use build() instead. This function will be removed in a future release.
    
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
    warnings.warn(
        "get() is deprecated and will be removed in a future release. Use build() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    return build(algorithm, **kwargs) 