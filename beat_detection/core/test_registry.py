"""
Tests for the registry module.
"""
import pytest
from beat_detection.core.registry import register, get, _DETECTORS
from beat_detection.core.detector_protocol import BeatDetector

# Clear registry between tests
@pytest.fixture(autouse=True)
def clear_registry():
    original_detectors = _DETECTORS.copy()
    _DETECTORS.clear()
    yield
    _DETECTORS.clear()
    _DETECTORS.update(original_detectors)

# Create a dummy detector for testing
class _DummyDetector:
    def __init__(self, param1=None):
        self.param1 = param1
    
    def detect_beats(self, audio_path):
        return "dummy_beats"  # Not a real RawBeats, just for testing

def test_register_decorator():
    # Test registering a detector
    @register("dummy")
    class TestDetector(_DummyDetector):
        pass
        
    assert "dummy" in _DETECTORS
    assert _DETECTORS["dummy"] == TestDetector

def test_register_duplicate():
    # Test registering a detector with the same name twice
    @register("dummy")
    class TestDetector1(_DummyDetector):
        pass
        
    with pytest.raises(ValueError, match="is already registered"):
        @register("dummy")
        class TestDetector2(_DummyDetector):
            pass

def test_get_success():
    # Test successfully getting a detector
    @register("dummy")
    class TestDetector(_DummyDetector):
        pass
        
    detector = get("dummy")
    assert isinstance(detector, TestDetector)

def test_get_with_params():
    # Test getting a detector with parameters
    @register("dummy")
    class TestDetector(_DummyDetector):
        pass
        
    detector = get("dummy", param1="test_value")
    assert detector.param1 == "test_value"

def test_get_failure():
    # Test getting a detector that doesn't exist
    with pytest.raises(ValueError, match="Unsupported beat detection algorithm"):
        get("nonexistent") 