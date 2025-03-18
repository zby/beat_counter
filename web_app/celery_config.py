"""
Celery configuration settings.

This module contains all the configuration settings for Celery and Redis.
"""
import os
from urllib.parse import urlparse

# Redis connection parameters
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)

# Redis URL for Celery
if REDIS_PASSWORD:
    REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
else:
    REDIS_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

# Redis connection settings
REDIS_SETTINGS = {
    'host': REDIS_HOST,
    'port': REDIS_PORT,
    'db': REDIS_DB,
    'password': REDIS_PASSWORD,
    'socket_timeout': 30,
    'socket_connect_timeout': 30,
    'max_connections': 20
}

# Broker settings
broker_url = REDIS_URL
result_backend = REDIS_URL

# Ensure connection settings are properly configured
broker_connection_retry = True
broker_connection_retry_on_startup = True
broker_connection_max_retries = 10

# Redis backend settings
redis_max_connections = REDIS_SETTINGS['max_connections']
redis_socket_timeout = REDIS_SETTINGS['socket_timeout']
redis_socket_connect_timeout = REDIS_SETTINGS['socket_connect_timeout']

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

# Helper functions for Redis connectivity
def get_redis_url():
    """Get the Redis URL for connections."""
    return REDIS_URL

def get_redis_connection_params():
    """Get Redis connection parameters as a dictionary.
    
    Returns:
        dict: Redis connection parameters
    """
    return {
        'host': REDIS_HOST,
        'port': REDIS_PORT,
        'db': REDIS_DB,
        'password': REDIS_PASSWORD
    }

def parse_redis_url(url):
    """Parse a Redis URL into connection parameters.
    
    Args:
        url (str): Redis URL to parse
        
    Returns:
        dict: Redis connection parameters
    """
    parsed = urlparse(url)
    
    # Handle password
    netloc_parts = parsed.netloc.split('@')
    if len(netloc_parts) > 1:
        auth = netloc_parts[0]
        if ':' in auth:
            _, password = auth.split(':')
        else:
            password = auth
    else:
        password = None
    
    # Handle host and port
    host_port = netloc_parts[-1]
    if ':' in host_port:
        host, port = host_port.split(':')
    else:
        host = host_port
        port = 6379
    
    # Handle database
    path = parsed.path
    if path and path.startswith('/'):
        db = int(path[1:]) if path[1:] else 0
    else:
        db = 0
    
    return {
        'host': host,
        'port': int(port),
        'db': db,
        'password': password
    }
