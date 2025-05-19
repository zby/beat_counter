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


def get_output_path(
    input_path: Union[str, pathlib.Path], 
    output_path: Optional[Union[str, pathlib.Path]] = None, 
    extension: str = ".beats"
) -> str:
    """
    Determine the output path for processing results based on input path.
    
    If output_path is provided, it uses that. Otherwise, it replaces the extension
    of the input file path with the specified extension.
    
    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the input file
    output_path : str or pathlib.Path, optional
        Path to the output file. If None, derives from input_path.
    extension : str, optional
        Extension to use when deriving output path, defaults to ".beats"
        
    Returns
    -------
    str
        The determined output path
        
    Notes
    -----
    If an explicit output_path is provided, this function ensures its parent
    directory exists, creating it if necessary.
    """
    # Determine the output path
    if output_path is None:
        # Use the same path as input but change extension
        input_path_obj = pathlib.Path(input_path)
        base, _ = os.path.splitext(str(input_path_obj))
        final_output_path = base + extension
    else:
        # Use the specified output path
        final_output_path = str(output_path)
        # Ensure the parent directory exists if an explicit path was provided
        pathlib.Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    return final_output_path

