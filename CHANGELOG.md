# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- **BREAKING CHANGE**: Internal module structure has been significantly refactored:
  - `beat_detection.core.factory` has been removed
  - The primary public API (e.g., `get_beat_detector`, `extract_beats`) is now available directly from `beat_detection.core`
  - Detector implementations have been moved to a dedicated `beat_detection.core.detectors` package
  - A new registry pattern has been implemented for detector registration using the `@register` decorator
  - The pipeline functionality has been extracted to a dedicated module
  - Import paths should be updated accordingly (e.g., `from beat_detection.core import get_beat_detector`)
- Enhanced `process_batch()` function to support genre-specific parameters with the `use_genre_defaults` option
- Refactored duplicated code between pipeline and experiment orchestrator modules
- Improved file path handling with new utility functions
- **BREAKING CHANGE**: Replaced `**kwargs` with typed `DetectorConfig` dataclass:
  - All detector implementations now accept a typed configuration object instead of arbitrary keyword arguments
  - The `build()` function in the registry replaces the deprecated `get()` function
  - `build()` accepts either a DetectorConfig object or kwargs that are transformed into a config
  - Direct instantiation of detectors requires a DetectorConfig object

### Added
- New registry system allowing easier registration of beat detectors
- Better separation of concerns between registry, pipeline, and detector implementations
- `get_output_path()` utility function for standardized output path generation
- New `reproducibility` module for experiment documentation and reproducibility
- Genre-based parameter application directly in the core pipeline module
- New `DetectorConfig` dataclass for strongly-typed configuration with proper validation
- Comprehensive unit tests for all detector implementations with typed configs
- Mypy strict type checking support throughout the codebase

### Removed
- `beat_detection.core.factory` module (functionality moved to `registry.py` and `pipeline.py`)
- Direct references to specific detector implementations from the public API
- Duplicated code between pipeline and experiment orchestrator 