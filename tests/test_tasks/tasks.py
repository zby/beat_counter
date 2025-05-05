"""
Simple test tasks for testing Celery functionality.
"""
from celery import Celery
import time

# Create a simple Celery app for testing
app = Celery('test_tasks')
app.conf.update(
    broker_url='memory://',
    result_backend='rpc://',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    task_always_eager=True,  # Execute tasks immediately for testing
)

@app.task
def add(x, y):
    """Add two numbers together - simple test task."""
    time.sleep(1)  # Simulate some work
    return x + y
