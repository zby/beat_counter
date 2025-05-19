"""Utility functions for task management and logging."""

import io
import sys
from typing import Tuple, Dict, Any, Callable
from celery import states
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# Maximum size for log output (10KB)
MAX_LOG_SIZE = 10000


class SafeLogHandler:
    """A wrapper for logger that catches I/O errors."""

    @staticmethod
    def safe_log(log_func, message, *args, **kwargs):
        """Log a message safely, catching any I/O errors."""
        try:
            log_func(message, *args, **kwargs)
        except OSError:
            # Silently ignore I/O errors during logging
            pass
        except Exception as e:
            # For other exceptions, try to log them but don't raise
            try:
                print(f"Logging error: {str(e)}")
            except Exception:
                pass


def safe_info(message, *args, **kwargs):
    """Log an info message safely."""
    SafeLogHandler.safe_log(logger.info, message, *args, **kwargs)


def safe_error(message, *args, **kwargs):
    """Log an error message safely."""
    SafeLogHandler.safe_log(logger.error, message, *args, **kwargs)


def safe_warning(message, *args, **kwargs):
    """Log a warning message safely."""
    SafeLogHandler.safe_log(logger.warning, message, *args, **kwargs)


def safe_debug(message, *args, **kwargs):
    """Log a debug message safely."""
    SafeLogHandler.safe_log(logger.debug, message, *args, **kwargs)


def truncate_output(output: str, max_size: int = MAX_LOG_SIZE) -> str:
    """Truncate output to prevent memory issues.

    Args:
        output: The output string to truncate
        max_size: Maximum size in characters (default: 10KB)

    Returns
    -------
        Truncated output string
    """
    if len(output) > max_size:
        return "[...truncated...]\n" + output[-max_size:]
    return output


def safe_print(message: str) -> None:
    """Print a message, ignoring I/O errors."""
    try:
        print(message)
    except OSError:
        pass


class IOCapture:
    """Context manager for capturing stdout and stderr."""

    def __init__(self):
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def get_output(self) -> Tuple[str, str]:
        """Get the captured stdout and stderr, truncated if necessary."""
        stdout = truncate_output(self.stdout_capture.getvalue())
        stderr = truncate_output(self.stderr_capture.getvalue())
        return stdout, stderr

    def write_stderr(self, message: str) -> None:
        """Write a message to stderr capture."""
        self.stderr_capture.write(message + "\n")


def create_progress_updater(
    celery_task, task_info: dict, output_key: str
) -> Callable[[str, float], None]:
    """Create a progress update function for a task.

    Args:
        celery_task: The Celery task instance
        task_info: Dictionary with task information
        output_key: Key for the output dictionary in the task metadata

    Returns:
        A function that can be called to update progress
    """

    def update_progress(status: str, progress: float) -> None:
        try:
            # Calculate progress percentage
            percent = progress * 100

            # Get current stdout and stderr from celery_task
            if hasattr(celery_task, "_io_capture"):
                current_stdout, current_stderr = celery_task._io_capture.get_output()
            else:
                current_stdout, current_stderr = "", ""

            # Update task state with comprehensive metadata
            update_meta = task_info.copy()
            update_meta.update(
                {
                    "progress": {"status": status, "percent": percent},
                    output_key: {"stdout": current_stdout, "stderr": current_stderr},
                }
            )

            celery_task.update_state(state=states.STARTED, meta=update_meta)

            # Log progress using safe logging
            task_type = output_key.replace("_output", "").upper()
            safe_info(f"{task_type} progress: {status} - {percent:.1f}%")
            safe_print(f"{task_type}: {status} - {percent:.1f}%")

        except Exception as e:
            # If there's an error updating progress, log it but don't fail the task
            safe_error(f"Error updating progress: {str(e)}")
            safe_print(f"Error updating progress: {str(e)}")

    return update_progress
