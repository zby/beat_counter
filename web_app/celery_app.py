"""
Celery configuration for the beat detection application.

This module sets up Celery for handling long-running tasks like beat detection
and video generation in a distributed manner.
"""

import os
from celery import Celery

# Create the Celery app with explicit Redis backend
app = Celery(
    'beat_detection',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Explicitly set the result backend again to ensure it's properly configured
app.conf.result_backend = 'redis://localhost:6379/0'

# Ensure task results are stored
app.conf.task_ignore_result = False

# Configure result extended
app.conf.result_extended = True

# Configure task track started
app.conf.task_track_started = True

# Load configuration from a Python module directly
app.config_from_object('web_app.celery_config')

# Auto-discover tasks in the web_app package
app.autodiscover_tasks(['web_app'])

# Optional: Add some debugging info
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
