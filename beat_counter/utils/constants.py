"""
Constants used across the beat detection package.
"""

# Audio file formats supported by the library
AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
# Alias for AUDIO_EXTENSIONS to use in file validation
SUPPORTED_AUDIO_EXTENSIONS = AUDIO_EXTENSIONS

# Supported time signatures (upper numeral)
SUPPORTED_BEATS_PER_BAR = [3, 4]  # Corresponds to 3/4, and 4/4 time signatures
# 2 is not supported for detection because it overrides 4 - but it is supported when passed alone
# todo - rename this

# The FPS constant below was unused and has been removed.
# # Frames per second for Madmom's activation functions
# FPS = 100
