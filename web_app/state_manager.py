"""
State manager for the beat detection and video generation processes.
Handles persistence of state information to disk.
"""

import json
import os
import pathlib
import time
import threading
from typing import Dict, Any, Optional


class StateManager:
    """
    Manages state for processing tasks.
    Uses file-based storage for persistence.
    """

    def __init__(self, state_dir: str):
        """
        Initialize the state manager.

        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = pathlib.Path(state_dir)
        self.state_dir.mkdir(exist_ok=True, parents=True)
        self.lock = threading.Lock()

    def _get_state_file(self, file_id: str) -> pathlib.Path:
        """Get the path to the state file for a given file ID."""
        return self.state_dir / f"{file_id}.json"

    def _get_all_state_files(self) -> list:
        """Get all state files in the state directory."""
        return list(self.state_dir.glob("*.json"))

    def get_state(self, file_id: str, use_lock: bool = True) -> Dict[str, Any]:
        """
        Get the current state for a file.

        Args:
            file_id: ID of the file to get state for
            use_lock: Whether to use lock when reading the file (default: True)

        Returns:
            State dictionary or empty dict if not found
        """
        state_file = self._get_state_file(file_id)
        if not state_file.exists():
            return {}

        # Define a function to read the file to avoid code duplication
        def read_state_file():
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If the file is corrupted or can't be read, return empty state
                return {}

        if use_lock:
            with self.lock:
                return read_state_file()
        else:
            return read_state_file()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get states for all files.

        Returns:
            Dictionary mapping file IDs to their states
        """
        result = {}
        state_files = self._get_all_state_files()

        # For get_all_states, we don't need to lock each individual read
        # since we're just reading the files and not modifying them
        for state_file in state_files:
            file_id = state_file.stem
            # Skip locking for better performance when reading multiple files
            state = self.get_state(file_id, use_lock=False)
            if state:  # Only include non-empty states
                result[file_id] = state

        return result

    def update_state(self, file_id: str, update_dict: Dict[str, Any]) -> None:
        """
        Update the state for a file.

        Args:
            file_id: ID of the file to update state for
            update_dict: Dictionary with updates to apply
        """
        state_file = self._get_state_file(file_id)

        # Read current state outside the lock to minimize lock time
        current_state = {}
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    current_state = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start with empty state
                current_state = {}

        # Update state (this doesn't need a lock since we're working on a local copy)
        self._deep_update(current_state, update_dict)

        # Only lock when writing to the file
        with self.lock:
            with open(state_file, "w") as f:
                json.dump(current_state, f, indent=2)

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Recursively update a dictionary.

        Args:
            target: Dictionary to update
            source: Dictionary with updates
        """
        for key, value in source.items():
            if (
                isinstance(value, dict)
                and key in target
                and isinstance(target[key], dict)
            ):
                # If both are dicts, update recursively
                self._deep_update(target[key], value)
            else:
                # Otherwise, just update the value
                target[key] = value

    def update_progress(self, file_id: str, status: str, percent: float) -> None:
        """
        Update progress for a file.

        Args:
            file_id: ID of the file to update progress for
            status: Status message
            percent: Progress percentage (0-100)
        """
        self.update_state(file_id, {"progress": {"status": status, "percent": percent}})

    def delete_state(self, file_id: str) -> None:
        """
        Delete the state for a file.

        Args:
            file_id: ID of the file to delete state for
        """
        state_file = self._get_state_file(file_id)

        with self.lock:
            if state_file.exists():
                os.remove(state_file)
