"""
Constants used across the beat detection package.
"""

# Audio file formats supported by the library
AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
# Alias for AUDIO_EXTENSIONS to use in file validation
SUPPORTED_AUDIO_EXTENSIONS = AUDIO_EXTENSIONS

# Supported time signatures (upper numeral)
SUPPORTED_BEATS_PER_BAR = [2, 3, 4]  # Corresponds to 2/4, 3/4, and 4/4 time signatures
