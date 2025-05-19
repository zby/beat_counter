#!/usr/bin/env python3
"""
Celery Worker Starter Script

This script starts the Celery worker for processing beat detection and video generation tasks,
using the centralized configuration system.
"""

import os
import sys
import argparse
import subprocess
import pathlib
from typing import List, Dict, Optional
from urllib.parse import urlparse
import logging

# Set up basic logging for the script itself
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Determine the application directory dynamically
# Assumes the script is in web_app/ and the root is one level up
APP_ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()

# Add the application root directory to the Python path
sys.path.insert(0, str(APP_ROOT_DIR))

# Import Config after setting up the path
try:
    from beat_counter.web_app.config import Config
except ImportError as e:
    logger.error(
        f"Failed to import Config: {e}. Ensure APP_ROOT_DIR is correct and dependencies are installed."
    )
    sys.exit(1)


def parse_redis_url(url: str) -> Optional[Dict[str, any]]:
    """Parse a Redis URL into connection parameters.

    Args:
        url: Redis URL string (e.g., redis://:password@host:port/db)

    Returns:
        Dictionary with 'host', 'port', 'db', 'password', or None if parsing fails.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme != "redis":
            logger.error(f"Invalid Redis URL scheme: {parsed.scheme}")
            return None

        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 6379,
            "db": (
                int(parsed.path[1:]) if parsed.path and parsed.path[1:].isdigit() else 0
            ),
            "password": parsed.password,
        }
    except Exception as e:
        logger.error(f"Error parsing Redis URL '{url}': {e}")
        return None


def check_redis_connection(redis_params: Dict[str, any]) -> bool:
    """Check if Redis is accessible using provided parameters."""
    try:
        import redis
    except ImportError:
        logger.error("Redis library not found. Please install it: pip install redis")
        return False

    if not redis_params:
        logger.error("No Redis parameters provided for connection check.")
        return False

    try:
        # Create Redis client with connection parameters
        r = redis.Redis(
            host=redis_params.get("host", "localhost"),
            port=redis_params.get("port", 6379),
            db=redis_params.get("db", 0),
            password=redis_params.get("password"),
            socket_connect_timeout=5,  # Add a timeout
            decode_responses=True,  # Optional: helps with ping response
        )
        if r.ping():
            logger.info(
                f"Redis connection successful to {redis_params.get('host')}:{redis_params.get('port')}"
            )
            return True
        else:
            logger.warning("Redis ping failed.")
            return False
    except redis.exceptions.ConnectionError as e:
        logger.error(
            f"Redis connection error to {redis_params.get('host')}:{redis_params.get('port')}: {e}"
        )
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Redis check: {e}")
        return False


def run_worker(worker_args: List[str], app_root_dir: pathlib.Path) -> None:
    """Run the Celery worker command with the given arguments.

    Args:
        worker_args: List of command line arguments for the Celery worker.
        app_root_dir: The root directory of the application.
    """
    # Set the environment variable for the config loader
    os.environ["BEAT_COUNTER_APP_DIR"] = str(app_root_dir)
    logger.info(f"Set BEAT_COUNTER_APP_DIR to: {app_root_dir}")

    # Construct the command using the correct Celery app path
    # The Celery app instance is located in beat_counter/web_app/celery_app.py
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "-m",
        "celery",
        "-A",
        "beat_counter.web_app.celery_app",  # Point to the module containing the Celery app instance
        "worker",
    ] + worker_args

    logger.info(f"Running Celery worker command: {' '.join(cmd)}")

    try:
        # Run the worker as a subprocess
        # check=True will raise CalledProcessError if the command fails
        subprocess.run(cmd, check=True, cwd=str(app_root_dir))
    except subprocess.CalledProcessError as e:
        logger.error(f"Celery worker command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(
            f"Error: '{sys.executable}' or 'celery' command not found. Ensure Python and Celery are installed and in PATH."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while running the worker: {e}")
        sys.exit(1)


def main():
    """Load config, check Redis, parse args, and start the Celery worker."""
    parser = argparse.ArgumentParser(
        description="Start Celery worker for the Beat Detection app"
    )
    parser.add_argument(
        "--queues",
        "-Q",
        type=str,
        default="beat_detection,video_generation",
        help="Comma-separated list of queues to process (default: beat_detection,video_generation)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=None,
        help="Number of worker processes/threads (default: Celery default)",
    )
    parser.add_argument(
        "--loglevel",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Load application configuration
    try:
        config = (
            Config.from_env()
        )  # Assumes BEAT_COUNTER_APP_DIR might be set, or uses CWD
    except FileNotFoundError as e:
        logger.error(
            f"Configuration error: {e}. Make sure config files exist or BEAT_COUNTER_APP_DIR is set correctly."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Get Redis connection parameters from the loaded config
    redis_url = config.celery.broker_url
    redis_params = parse_redis_url(redis_url)

    # Check Redis connection before starting the worker
    if not check_redis_connection(redis_params):
        host = redis_params.get("host", "N/A") if redis_params else "N/A"
        port = redis_params.get("port", "N/A") if redis_params else "N/A"
        logger.error(f"Cannot connect to Redis at {host}:{port}.")
        logger.error("Please ensure Redis is running and accessible.")
        # logger.info("You might need to start Redis, e.g., using: docker-compose up -d redis")
        sys.exit(1)

    # Prepare arguments for the Celery worker command
    worker_args = ["--loglevel", args.loglevel, "-Q", args.queues]
    if args.concurrency is not None:
        worker_args.extend(["-c", str(args.concurrency)])

    # Start the worker
    logger.info(
        f"Starting Celery worker with queues: {args.queues}, loglevel: {args.loglevel}"
    )
    if args.concurrency is not None:
        logger.info(f"Concurrency set to: {args.concurrency}")

    run_worker(worker_args, APP_ROOT_DIR)


if __name__ == "__main__":
    main()
