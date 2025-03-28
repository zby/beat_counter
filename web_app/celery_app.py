"""
Celery configuration for the beat detection application.

This module sets up Celery for handling long-running tasks like beat detection
and video generation in a distributed manner.
"""

import os
from celery import Celery
from web_app.config import Config

# Load configuration
config = Config.from_env()

# Create the Celery app with configuration from config
app = Celery(**config.celery.__dict__)

# Auto-discover tasks in the web_app package
app.autodiscover_tasks(['web_app'])

# Optional: Add some debugging info
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
