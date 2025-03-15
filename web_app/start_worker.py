#!/usr/bin/env python3
"""
Celery Worker Starter Script

This script starts the Celery worker for processing beat detection and video generation tasks.
"""

import os
import sys
import argparse
import subprocess
import time

# Add the current directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Celery app after setting up the path
from web_app.celery_app import app

def check_redis_connection():
    """Check if Redis is accessible before starting the worker."""
    import redis
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        if r.ping():
            print("Redis connection successful")
            return True
        else:
            print("Redis ping failed")
            return False
    except Exception as e:
        print(f"Redis connection error: {e}")
        return False

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
        print("Cannot connect to Redis. Make sure Redis is running on localhost:6379")
        print("You can start Redis using: docker-compose up -d redis")
        sys.exit(1)
    
    # Start the worker using subprocess
    print(f"Starting Celery worker with queues: {args.queues}")
    cmd = [
        'celery', 
        '-A', 'web_app.celery_app', 
        'worker',
        '-Q', args.queues,
        '-c', str(args.concurrency),
        '--loglevel', args.loglevel
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Worker stopped by user")
    except Exception as e:
        print(f"Error starting worker: {e}")

if __name__ == '__main__':
    main()
