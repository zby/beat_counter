"""
File and path utility functions.
"""

import os
import pathlib
from typing import List, Optional, Union, Tuple, Callable

from beat_detection.utils.constants import AUDIO_EXTENSIONS


def ensure_directory(directory: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters:
    -----------
    directory : str or pathlib.Path
        Directory path to ensure exists
    
    Returns:
    --------
    pathlib.Path
        Path object for the directory
    """
    path = pathlib.Path(directory)
    os.makedirs(path, exist_ok=True)
    return path


def find_audio_files(directory: Union[str, pathlib.Path], 
                     extensions: List[str] = None) -> List[pathlib.Path]:
    """
    Find audio files in a directory, recursively searching subdirectories.
    
    Parameters:
    -----------
    directory : str or pathlib.Path
        Directory to search for audio files
    extensions : list of str, optional
        List of file extensions to include. If None, uses all supported audio extensions.
    
    Returns:
    --------
    list of pathlib.Path
        List of audio file paths
    """
    path = pathlib.Path(directory)
    audio_files = []
    
    # Use the global constant if no extensions are provided
    ext_list = extensions if extensions is not None else AUDIO_EXTENSIONS
    
    for ext in ext_list:
        # Use rglob for recursive search into subdirectories
        audio_files.extend(list(path.rglob(f'*{ext}')))
    
    return sorted(audio_files)


def find_files_by_pattern(directory: Union[str, pathlib.Path], pattern: str) -> List[pathlib.Path]:
    """
    Find files in a directory matching a glob pattern, recursively searching subdirectories.
    
    Parameters:
    -----------
    directory : str or pathlib.Path
        Directory to search for files
    pattern : str
        Glob pattern to match (e.g., '*_beats.txt')
    
    Returns:
    --------
    list of pathlib.Path
        List of file paths matching the pattern
    """
    path = pathlib.Path(directory)
    return sorted(list(path.rglob(pattern)))


def resolve_input_path(filename: str, input_dir: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    """
    Resolve an input filename to an absolute path.
    
    Parameters:
    -----------
    filename : str
        Filename or path
    input_dir : str or pathlib.Path, optional
        Default input directory to use if filename is not absolute
    
    Returns:
    --------
    pathlib.Path
        Resolved absolute path
    """
    path = pathlib.Path(filename)
    
    if path.is_absolute():
        return path
    
    if input_dir is not None:
        return pathlib.Path(input_dir) / path
    
    return path.absolute()


def find_related_file(base_name: Union[str, pathlib.Path], 
                      target_dir: Union[str, pathlib.Path],
                      extensions: List[str] = None,
                      suffix: str = None,
                      prefix: str = None,
                      remove_suffix: str = None) -> Optional[pathlib.Path]:
    """
    Find a related file in a target directory.
    
    Parameters:
    -----------
    base_name : str or pathlib.Path
        Base name to search for (with or without extension)
    target_dir : str or pathlib.Path
        Directory to search in
    extensions : list of str, optional
        List of file extensions to search for (e.g., ['.mp3', '.wav'])
    suffix : str, optional
        Suffix to add to the base name (e.g., '_beats')
    prefix : str, optional
        Prefix to add to the base name
    remove_suffix : str, optional
        Suffix to remove from the base name if present
    
    Returns:
    --------
    pathlib.Path or None
        Path to the related file if found, or None if not found
    """
    # Convert to Path object and get stem (filename without extension)
    base_path = pathlib.Path(base_name)
    base_stem = base_path.stem
    
    # Remove any existing suffixes from the stem
    if remove_suffix and base_stem.endswith(remove_suffix):
        base_stem = base_stem[:-len(remove_suffix)]
    
    # Build the search pattern
    target_dir = pathlib.Path(target_dir)
    
    if prefix:
        base_stem = f"{prefix}{base_stem}"
    
    if suffix:
        base_stem = f"{base_stem}{suffix}"
    
    # Search for files with the specified extensions
    if extensions:
        for ext in extensions:
            pattern = f"{base_stem}{ext}"
            matches = list(target_dir.glob(pattern))
            if matches:
                return matches[0]
    else:
        # If no extensions specified, search for any extension
        pattern = f"{base_stem}.*"
        matches = list(target_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def get_output_path(input_file: pathlib.Path, output_dir: pathlib.Path, 
                   suffix: str = '_with_metronome', ext: str = '.wav') -> pathlib.Path:
    """
    Generate an output file path based on input file.
    
    Parameters:
    -----------
    input_file : pathlib.Path
        Input file path
    output_dir : pathlib.Path
        Output directory
    suffix : str
        Suffix to add to the output filename
    ext : str
        File extension for the output file
    
    Returns:
    --------
    pathlib.Path
        Output file path
    """
    return output_dir / f"{input_file.stem}{suffix}{ext}"


def process_input_path(input_path: Union[str, pathlib.Path], 
                       default_dir: Union[str, pathlib.Path],
                       process_file_func: Callable,
                       process_dir_func: Callable,
                       **kwargs) -> None:
    """
    Process an input path that could be a file or directory.
    
    This function handles the common pattern of:
    1. Checking if input_path is a file or directory
    2. Calling the appropriate processing function
    3. Falling back to default directory if path not found
    
    Parameters:
    -----------
    input_path : str or pathlib.Path
        Input path to process
    default_dir : str or pathlib.Path
        Default directory to use if input path not specified or found
    process_file_func : callable
        Function to call for processing a single file
    process_dir_func : callable
        Function to call for processing a directory
    **kwargs : dict
        Additional keyword arguments to pass to processing functions
    """
    path = pathlib.Path(input_path)
    
    if path.is_dir():
        # Process directory
        process_dir_func(path, **kwargs)
    elif path.is_file():
        # Process single file
        process_file_func(path, **kwargs)
    else:
        # Try to find file in default directory
        default_input = pathlib.Path(default_dir) / path
        if default_input.is_file():
            process_file_func(default_input, **kwargs)
        else:
            print(f"Error: Input file or directory '{input_path}' not found")


def find_beats_file_for_audio(audio_file: Union[str, pathlib.Path]) -> Optional[pathlib.Path]:
    """
    Find the beats file for a given audio file.
    
    This function looks for a beats file in the data/beats directory that matches the audio file.
    It first tries to find the beats file in the same subdirectory structure as the audio file,
    and if not found, it searches in all subdirectories.
    
    Parameters:
    -----------
    audio_file : str or pathlib.Path
        Path to the audio file
    
    Returns:
    --------
    pathlib.Path or None
        Path to the beats file if found, None otherwise
    """
    audio_path = pathlib.Path(audio_file)
    
    # Determine the beats directory based on the audio file location
    if "data/original" in str(audio_path):
        # Get the subdirectory structure, if any
        rel_path = audio_path.relative_to(pathlib.Path("data/original"))
        if rel_path.parent != pathlib.Path("."):
            # There is a subdirectory - look in the corresponding beats subdirectory
            beats_dir = pathlib.Path("data/beats") / rel_path.parent
        else:
            beats_dir = pathlib.Path("data/beats")
    else:
        beats_dir = pathlib.Path("data/beats")
    
    # First try to find the beats file in the expected directory
    beats_file = find_related_file(
        audio_path.stem, 
        beats_dir,
        suffix='_beats',
        extensions=['.txt']
    )
    
    # If not found, search recursively in all subdirectories
    if beats_file is None:
        for match in pathlib.Path("data/beats").rglob(f"{audio_path.stem}_beats.txt"):
            beats_file = match
            break
    
    return beats_file


def batch_process(files: List[pathlib.Path], 
                 process_func: Callable, 
                 verbose: bool = True, 
                 **kwargs) -> List:
    """
    Process a batch of files, handling errors for each file individually.
    
    Parameters:
    -----------
    files : list of pathlib.Path
        List of files to process
    process_func : callable
        Function to call for each file
    verbose : bool
        Whether to print progress information
    **kwargs : dict
        Additional keyword arguments to pass to process_func
    
    Returns:
    --------
    list
        List of results from process_func for each file
    """
    if not files:
        if verbose:
            print(f"No files to process")
        return []
    
    if verbose:
        print(f"Processing {len(files)} files...")
    
    results = []
    
    for file_path in files:
        try:
            result = process_func(file_path, verbose=verbose, **kwargs)
            results.append((file_path.name, result))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results