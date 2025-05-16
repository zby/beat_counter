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

### Added
- New registry system allowing easier registration of beat detectors
- Better separation of concerns between registry, pipeline, and detector implementations

### Removed
- `beat_detection.core.factory` module (functionality moved to `registry.py` and `pipeline.py`)
- Direct references to specific detector implementations from the public API 