#!/usr/bin/env python3
"""Script to get file status for a given file_id."""

import asyncio
import argparse
import json
import inspect
from fastapi import Request, Depends
from beat_counter.web_app.app import create_app


async def display_file_status(file_id: str) -> None:
    """Get and display file status for the given file_id by calling the app's endpoint."""
    # Create app instance
    app = create_app()

    # Find the get_file_status route handler
    route_handler = None
    for route in app.routes:
        if route.path == f"/status/{{file_id}}" and "GET" in route.methods:
            route_handler = route.endpoint
            break

    if not route_handler:
        print("Error: Could not find the get_file_status route handler in the app")
        return

    # Get storage and service from app dependencies
    storage = app.dependency_overrides[app.routes[0].dependencies[0].dependency]()
    service = app.dependency_overrides[app.routes[0].dependencies[1].dependency]()

    # Create a mock request
    mock_request = Request(
        {"type": "http", "method": "GET", "path": f"/status/{file_id}"}
    )

    try:
        # Call the get_file_status route handler directly
        response_data = await route_handler(
            file_id=file_id,
            request=mock_request,
            storage=storage,
            service=service,
            user={"username": "cli-user"},  # Mock user
        )

        # Display the status information
        print(f"\nFile Status for {file_id}:")
        print("-" * 50)

        # Format the status data for display
        formatted_data = json.dumps(response_data, indent=2)
        print(formatted_data)

    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main entry point for the script."""
    file_id = "8db77a15-716a-4dd2-8d33-edd462c5594b"
    asyncio.run(display_file_status(file_id))


if __name__ == "__main__":
    main()
