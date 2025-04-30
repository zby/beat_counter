"""
Common test fixtures and utilities for beat detection tests.
"""
import re
import pytest
from contextlib import contextmanager


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
