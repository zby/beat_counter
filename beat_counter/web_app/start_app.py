#!/usr/bin/env python3
"""
FastAPI Application Starter Script

This script starts the FastAPI application for the beat detection web interface.
"""

import os
import sys
import argparse
import uvicorn
import logging # Added for logging
from typing import Optional

# Imports for configuration and Celery initialization
from beat_counter.web_app.config import Config, ConfigurationError
from beat_counter.web_app.celery_app import initialize_celery_app, app as celery_app_instance # Import celery_app_instance for check
from beat_counter.web_app.app import initialize_fastapi_app # Import the new FastAPI app initializer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic logging config for the starter script

def main():
    """Load config, initialize Celery, and start the FastAPI application."""
    parser = argparse.ArgumentParser(
        description="Start the beat detection web application"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    # --- Load Application Configuration ---    
    loaded_config: Optional[Config] = None
    try:
        logger.info("Attempting to load application configuration...")
        app_root_dir = Config.get_app_dir_from_env()
        logger.info(f"Application root directory for config: {app_root_dir}")
        loaded_config = Config.from_dir(app_root_dir)
        logger.info(f"Configuration loaded successfully: App Name - {loaded_config.app.name}")
    except (ConfigurationError, FileNotFoundError) as e:
        logger.critical(f"FATAL: Failed to load essential configuration: {e}", exc_info=True)
        print(f"FATAL: Failed to load essential configuration: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch any other unexpected error during config load
        logger.critical(f"FATAL: An unexpected error occurred while loading configuration: {e}", exc_info=True)
        print(f"FATAL: An unexpected error occurred while loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Initialize Celery Application --- 
    # This makes the web_app.celery_app.app instance fully configured
    if loaded_config: # Should always be true if we haven't exited
        try:
            logger.info("Initializing Celery application...")
            initialize_celery_app(loaded_config)
            logger.info(f"Celery application '{celery_app_instance.main if celery_app_instance else 'UNKNOWN'}' initialized and configured for the web app.")
        except Exception as e:
            # For a web app, failing to init Celery might be non-fatal if task submission is not on all paths
            # However, following "Fail Fast", if Celery is integral, this should be fatal.
            # Let's make it fatal as per the project guidelines for now.
            logger.critical(f"FATAL: Failed to initialize Celery application: {e}", exc_info=True)
            print(f"FATAL: Failed to initialize Celery application: {e}", file=sys.stderr)
            sys.exit(1) # Exit if Celery can't be initialized
    else:
        # This case should ideally not be reached due to sys.exit above if config fails
        logger.critical("FATAL: Configuration was not loaded. Cannot initialize Celery or start web app.")
        print("FATAL: Configuration was not loaded. Cannot initialize Celery or start web app.", file=sys.stderr)
        sys.exit(1)
    
    # --- Initialize FastAPI Application ---
    if loaded_config: # Should still be true
        try:
            logger.info("Initializing FastAPI application instance...")
            initialize_fastapi_app(loaded_config)
            # The global app instance in web_app.app is now created and configured.
            logger.info("FastAPI application instance initialized.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize FastAPI application: {e}", exc_info=True)
            print(f"FATAL: Failed to initialize FastAPI application: {e}", file=sys.stderr)
            sys.exit(1)

    # Start the server
    logger.info(f"Starting FastAPI application on {args.host}:{args.port} (Reload: {args.reload})")
    uvicorn.run(
        "beat_counter.web_app.app:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload, 
        log_level="info" # Align uvicorn log level
    )


if __name__ == "__main__":
    main()
