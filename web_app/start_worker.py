#!/usr/bin/env python3
"""
Celery Worker Starter Script

This script starts the Celery worker for processing beat detection and video generation tasks.
"""

import os
import sys
import argparse
import subprocess
import pathlib
from typing import List

# Add the current directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Celery app and Redis configuration after setting up the path
from web_app.celery_config import get_redis_connection_params, REDIS_HOST, REDIS_PORT

def check_redis_connection():
    """Check if Redis is accessible before starting the worker."""
    import redis
    try:
        # Use centralized Redis configuration
        conn_params = get_redis_connection_params()
        r = redis.Redis(**conn_params)
        
        if r.ping():
            print("Redis connection successful")
            return True
        else:
            print("Redis ping failed")
            return False
    except Exception as e:
        print(f"Redis connection error: {e}")
        return False

def run_worker(args: List[str]) -> None:
    """Run the Celery worker with the given arguments.
    
    Args:
        args: List of command line arguments
    """
    # Get the absolute path of the web_app directory
    app_dir = pathlib.Path(__file__).parent.parent.absolute()
    
    # Set the environment variable
    os.environ['BEAT_COUNTER_APP_DIR'] = str(app_dir)
    
    # Construct the command
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "web_app.celery_app",
        "worker",
        "--loglevel=info",
    ] + args
    
    # Run the worker
    subprocess.run(cmd, check=True)

def main():
    """Start the Celery worker with the specified queues."""
    parser = argparse.ArgumentParser(description='Start Celery worker for beat detection')
    parser.add_argument('--queues', '-Q', type=str, default='beat_detection,video_generation',
                        help='Comma-separated list of queues to process')
    parser.add_argument('--concurrency', '-c', type=int, default=2,
                        help='Number of worker processes/threads')
    parser.add_argument('--loglevel', '-l', type=str, default='INFO',
                        help='Logging level')
    args = parser.parse_args()
    
    # Check Redis connection before starting the worker
    if not check_redis_connection():
        print(f"Cannot connect to Redis. Make sure Redis is running on {REDIS_HOST}:{REDIS_PORT}")
        print("You can start Redis using: docker-compose up -d redis")
        sys.exit(1)
    
    # Start the worker using subprocess
    print(f"Starting Celery worker with queues: {args.queues}")
    run_worker([
        '-Q', args.queues,
        '-c', str(args.concurrency),
        '--loglevel', args.loglevel
    ])

if __name__ == '__main__':
    main()
