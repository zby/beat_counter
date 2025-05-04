"""
File and path utility functions.
"""

import os
import pathlib
from typing import List, Optional, Union, Tuple, Callable

from beat_detection.utils.constants import AUDIO_EXTENSIONS


def find_audio_files(
    paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
    extensions: List[str] = None,
) -> List[pathlib.Path]:
    """
    Find audio files in a list of files/directories, recursively searching subdirectories.

    Parameters:
    -----------
    paths : str, pathlib.Path, or list of str/pathlib.Path
        File or directory paths to search for audio files
    extensions : list of str, optional
        List of file extensions to include. If None, uses all supported audio extensions.

    Returns:
    --------
    list of pathlib.Path
        List of audio file paths
    """
    # Convert single path to list for uniform processing
    if not isinstance(paths, list):
        paths = [paths]

    audio_files = []

    # Use the global constant if no extensions are provided
    ext_list = extensions if extensions is not None else AUDIO_EXTENSIONS

    for path in paths:
        path = pathlib.Path(path)

        # If it's a file, check if it has a valid audio extension
        if path.is_file():
            if any(str(path).lower().endswith(ext.lower()) for ext in ext_list):
                audio_files.append(path)
        # If it's a directory, search recursively
        elif path.is_dir():
            for ext in ext_list:
                # Use rglob for recursive search into subdirectories
                audio_files.extend(list(path.rglob(f"*{ext}")))

    return sorted(audio_files)

