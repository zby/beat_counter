"""ASGI entry point for the Beat Detection Web App."""

from beat_counter.web_app.app import app as application

# This file is the entry point for ASGI servers like Uvicorn, Hypercorn, or Daphne
# Export the ASGI application as 'application'
app = application
