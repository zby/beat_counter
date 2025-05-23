"""
Common test fixtures and utilities for all tests in the project.
"""
import re
import os
import pytest
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def assert_raises(exception_type, match=None):
    """
    A cleaner alternative to pytest.raises that doesn't produce nested exception traces.
    
    This helper provides simpler error messages when an exception doesn't match
    the expected pattern, avoiding the "During handling of the above exception..."
    messages that can make test failures harder to understand.
    
    Args:
        exception_type: The exception type that should be raised
        match: Optional regex pattern that the exception message should match
        
    Raises:
        AssertionError: If no exception is raised or if the exception doesn't match
                       the expected type or pattern
    """
    try:
        yield
        pytest.fail(f"Expected {exception_type.__name__} to be raised, but no exception was raised")
    except exception_type as exc:
        if match is not None:
            message = str(exc)
            if not re.search(match, message):
                pytest.fail(
                    f"Expected exception message to match '{match}', but got: '{message}'"
                )
    except Exception as exc:
        pytest.fail(
            f"Expected {exception_type.__name__} to be raised, but got {type(exc).__name__}: {exc}"
        )


@pytest.fixture(scope="session", autouse=True)
def force_cpu_for_tests():
    """
    Force CPU usage for all tests by patching the device parameter used in beat_this_detector.
    
    This fixture sets an environment variable and patches the build function
    to ensure that any BeatThisDetector instances created during tests will use CPU.
    """
    # Set environment variable to indicate CPU usage
    os.environ['BEAT_COUNTER_FORCE_CPU'] = '1'
    
    # Apply patch for BeatThisDetector initialization
    from beat_counter.core.registry import build
    original_build = build
    
    def patched_build(*args, **kwargs):
        # Force 'device' parameter to 'cpu' for any BeatThisDetector
        if 'device' not in kwargs:
            kwargs['device'] = 'cpu'
        return original_build(*args, **kwargs)
    
    with patch('beat_counter.core.registry.build', patched_build):
        yield 

def pytest_collection_modifyitems(items):
    """Mark all tests in the 'tests' directory as slow."""
    for item in items:
        if "/tests/" in str(item.fspath):
            item.add_marker(pytest.mark.slow) 