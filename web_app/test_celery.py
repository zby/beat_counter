#!/usr/bin/env python3
"""
Test script for Celery integration

This script tests the Celery integration by submitting a simple task and checking its status.
"""

import time
import argparse
from celery_app import app
from tasks import detect_beats_task, generate_video_task

def test_beat_detection():
    """Test the beat detection task."""
    # Replace with a real audio file path for testing
    file_id = "test_file"
    file_path = "/path/to/audio/file.mp3"
    
    print(f"Submitting beat detection task for file: {file_path}")
    task = detect_beats_task.delay(file_id, file_path)
    
    print(f"Task ID: {task.id}")
    print("Task status:", task.status)
    
    # Wait for a moment and check status again
    time.sleep(2)
    print("Task status after 2 seconds:", task.status)
    
    return task.id

def check_task_status(task_id):
    """Check the status of a task by its ID."""
    task = app.AsyncResult(task_id)
    print(f"Task ID: {task_id}")
    print("Task status:", task.status)
    
    if task.status == 'SUCCESS':
        print("Task result:", task.result)
    elif task.status == 'FAILURE':
        print("Task error:", task.result)
    else:
        print("Task info:", task.info)

def main():
    """Run the Celery test."""
    parser = argparse.ArgumentParser(description='Test Celery integration')
    parser.add_argument('--task-id', type=str, help='Task ID to check status')
    args = parser.parse_args()
    
    if args.task_id:
        check_task_status(args.task_id)
    else:
        test_beat_detection()

if __name__ == '__main__':
    main()
