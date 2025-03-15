"""
Celery configuration settings.

This module contains all the configuration settings for Celery.
"""

# Broker settings
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

# Ensure connection settings are properly configured
broker_connection_retry = True
broker_connection_retry_on_startup = True
broker_connection_max_retries = 10

# Redis backend settings
redis_max_connections = 20
redis_socket_timeout = 30
redis_socket_connect_timeout = 30

# Note: Using the existing Redis instance running in Docker

# Task serialization format
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'

# Task execution settings
task_track_started = True
task_time_limit = 3600  # 1 hour time limit for tasks
worker_max_tasks_per_child = 50  # Restart worker after 50 tasks to prevent memory leaks

# Task result settings
result_expires = 86400  # Results expire after 1 day
result_extended = True  # Store extended task information
task_store_errors_even_if_ignored = True  # Store error information even if ignored

# Logging
worker_redirect_stdouts = False
worker_redirect_stdouts_level = 'INFO'

# Concurrency
worker_concurrency = 2  # Number of worker processes/threads

# Task routing
task_routes = {
    'detect_beats_task': {'queue': 'beat_detection'},
    'generate_video_task': {'queue': 'video_generation'},
}

# Task default rate limit
task_default_rate_limit = '10/m'  # 10 tasks per minute
