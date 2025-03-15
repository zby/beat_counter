#!/usr/bin/env python3
"""
FastAPI Application Starter Script

This script starts the FastAPI application for the beat detection web interface.
"""

import os
import sys
import argparse
import uvicorn

def main():
    """Start the FastAPI application with the specified host and port."""
    parser = argparse.ArgumentParser(description='Start the beat detection web application')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    args = parser.parse_args()
    
    # Start the server
    print(f"Starting FastAPI application on {args.host}:{args.port}")
    uvicorn.run("web_app.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == '__main__':
    main()
