"""
File and path utility functions.
"""

import os
import pathlib
from typing import List, Optional, Union, Tuple, Callable

from beat_detection.utils.constants import AUDIO_EXTENSIONS


def find_audio_files(paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]], 
                     extensions: List[str] = None) -> List[pathlib.Path]:
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
                audio_files.extend(list(path.rglob(f'*{ext}')))
    
    return sorted(audio_files)

def get_input_files(input_path: Union[str, pathlib.Path], 
                   default_dir: str = "data/input",
                   file_pattern: str = "*",
                   extensions: List[str] = None) -> List[pathlib.Path]:
    """
    Get a list of input files based on input path, returning either a single file or
    a list of files from a directory.
    
    Parameters:
    -----------
    input_path : str or pathlib.Path
        Path to file or directory to process
    default_dir : str
        Default directory to use if input path is not found (default: "data/input")
    file_pattern : str
        Glob pattern to match in directories (default: "*")
    extensions : list of str, optional
        List of file extensions to include. If None and searching audio files, 
        uses all supported audio extensions.
    
    Returns:
    --------
    list of pathlib.Path
        List of input file paths to process
    """
    path = pathlib.Path(input_path)
    input_files = []
    
    if path.is_dir():
        # Check if we're looking for audio files specifically
        if extensions is None and file_pattern == "*":
            # This is likely an audio file search
            input_files = find_audio_files(path, extensions=AUDIO_EXTENSIONS)
        else:
            # Use specified pattern and extensions
            if extensions:
                for ext in extensions:
                    input_files.extend(list(path.rglob(f"{file_pattern}{ext}")))
            else:
                input_files.extend(list(path.rglob(file_pattern)))
    elif path.is_file():
        # Just a single file
        input_files = [path]
    else:
        # Try to find file in default directory
        default_input = pathlib.Path(default_dir) / path
        if default_input.is_file():
            input_files = [default_input]
        elif default_input.is_dir():
            # Check if we're looking for audio files specifically
            if extensions is None and file_pattern == "*":
                # This is likely an audio file search
                input_files = find_audio_files(default_input, extensions=AUDIO_EXTENSIONS)
            else:
                # Use specified pattern and extensions
                if extensions:
                    for ext in extensions:
                        input_files.extend(list(default_input.rglob(f"{file_pattern}{ext}")))
                else:
                    input_files.extend(list(default_input.rglob(file_pattern)))
        else:
            print(f"Warning: Input path '{input_path}' not found")
    
    return sorted(input_files)


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


def get_output_directory(input_file: Union[str, pathlib.Path], 
                        input_base_dir: str = "data/input",
                        output_base_dir: str = "data/output") -> pathlib.Path:
    """
    Compute the output directory for a given input file, preserving subdirectory structure.
    
    Parameters:
    -----------
    input_file : str or pathlib.Path
        Path to the input file or directory
    input_base_dir : str
        Base input directory (default: "data/input")
    output_base_dir : str
        Base output directory (default: "data/output")
    
    Returns:
    --------
    pathlib.Path
        Output directory path
    """
    input_path = pathlib.Path(input_file)
    input_base = pathlib.Path(input_base_dir)
    output_base = pathlib.Path(output_base_dir)
    
    # Try to determine if input_file is within the input_base_dir structure
    try:
        # Get the relative path from input_base_dir
        rel_path = input_path.relative_to(input_base)
        
        # If input_path is a file, we want the parent directory's structure
        if input_path.is_file():
            output_dir = output_base / rel_path.parent
        else:
            # Input is a directory, preserve its structure
            output_dir = output_base / rel_path
    except ValueError:
        # File is not in input_base_dir, just use the output_base_dir
        output_dir = output_base
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def get_output_path(input_file: Union[str, pathlib.Path], 
                   suffix: str,
                   ext: str,
                   input_base_dir: str = "data/input",
                   output_base_dir: str = "data/output") -> pathlib.Path:
    """
    Generate an output file path based on input file.
    
    Parameters:
    -----------
    input_file : pathlib.Path
        Input file path
    suffix : str
        Suffix to add to the output filename
    ext : str
        File extension for the output file
    input_base_dir : str
        Base input directory (default: "data/input")
    output_base_dir : str
        Base output directory (default: "data/output")
    
    Returns:
    --------
    pathlib.Path
        Output file path
    """
    input_path = pathlib.Path(input_file)
    
    # Compute output directory based on input path
    output_dir = get_output_directory(input_path, input_base_dir, output_base_dir)
    
    # Ensure extension has a dot prefix
    if ext and not ext.startswith('.'):
        ext = '.' + ext
    
    return output_dir / f"{input_path.stem}{suffix}{ext}"


def find_beats_file_for_audio(audio_file: Union[str, pathlib.Path], 
                               input_base_dir: str = "data/input",
                               output_base_dir: str = "data/output") -> Optional[pathlib.Path]:
    """
    Find the beats file for a given audio file.
    
    This function looks for a beats file in the output directory that matches the audio file.
    It preserves the subdirectory structure from input to output directories.
    
    Parameters:
    -----------
    audio_file : str or pathlib.Path
        Path to the audio file
    input_base_dir : str
        Base input directory (default: "data/input")
    output_base_dir : str
        Base output directory (default: "data/output")
    
    Returns:
    --------
    pathlib.Path or None
        Path to the beats file if found, None otherwise
    """
    audio_path = pathlib.Path(audio_file)
    
    # Get the appropriate output directory
    beats_dir = get_output_directory(audio_path, input_base_dir, output_base_dir)
    
    # First try to find the beats file in the expected directory
    beats_file_path = beats_dir / f"{audio_path.stem}_beats.txt"
    beats_file = beats_file_path if beats_file_path.is_file() else None
    
    # If not found, search recursively in all subdirectories
    if beats_file is None:
        for match in pathlib.Path(output_base_dir).rglob(f"{audio_path.stem}_beats.txt"):
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