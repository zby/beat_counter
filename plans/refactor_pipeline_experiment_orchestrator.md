# Refactor Plan – Unify Pipeline and Experiment Orchestrator

This plan addresses code duplication between `beat_detection/core/pipeline.py` and `scripts/experiment_orchestrator.py`. Both files contain similar batch processing logic for audio files, but `experiment_orchestrator.py` has specialized logic for genre detection and experimental workflow.

| Legend |
|--------|
| `+` new file |
| `±` modified file |
| `→` moved/renamed file |
| `×` deleted file |

---

## Step 1 — Enhance Pipeline Module to Support Genre Parameters

| Files | Action |
|-------|--------|
| `beat_detection/core/pipeline.py` `±` | Refactor to generalize `process_batch()` to support genre-based parameter application |
| `beat_detection/core/__init__.py` `±` | Update exports if needed for new parameters |

The primary goal is to enhance the `process_batch()` function to handle genre-specific parameters similar to how `process_batch_with_genre_defaults()` works in the experiment orchestrator. This will:

1. Add an optional `use_genre_defaults` parameter (defaulting to `False`) 
2. Add code to detect genre from file path when enabled
3. Apply appropriate parameter adjustments based on genre

### Unit tests

* Extend `beat_detection/core/test_pipeline.py` to test genre-aware processing:
  ```python
  def test_process_batch_with_genre_defaults():
      # Create mock directory structure with genre paths
      # Verify genre-specific parameters are applied correctly
  ```

---

## Step 2 — Refactor Experiment Orchestrator to Use Enhanced Pipeline

| Files | Action |
|-------|--------|
| `scripts/experiment_orchestrator.py` `±` | Replace `process_batch_with_genre_defaults()` with call to enhanced `process_batch()` |

This step removes the duplicated batch processing code from the experiment orchestrator by replacing:
- Duplicated file finding
- Progress bar handling
- Error handling
- File processing loops

The key modifications:
1. Remove the entire `process_batch_with_genre_defaults()` function
2. Update the `run_experiment()` function to use the enhanced `process_batch()` with `use_genre_defaults=True` parameter

### Unit tests

* Create new test to verify experiment orchestrator still works with refactored code:
  ```python
  def test_experiment_orchestrator_uses_enhanced_pipeline():
      # Mock pipeline.process_batch
      # Verify experiment orchestrator calls it with correct parameters
      # Especially verify genre defaults flag is passed correctly
  ```

---

## Step 3 — Extract Common Output Path Logic

| Files | Action |
|-------|--------|
| `beat_detection/utils/file_utils.py` `±` | Add function for standardized output path generation |
| `beat_detection/core/pipeline.py` `±` | Use the common output path function |
| `scripts/experiment_orchestrator.py` `±` | Use the common output path function |

Currently, both files have logic for determining output paths. Extract this to a common utility:

1. Create `get_output_path(input_path, output_path=None, extension=".beats")` in file_utils.py
2. Update both modules to use this function

### Unit tests

* Add tests for the new utility function:
  ```python
  def test_get_output_path():
      # Test with no output_path provided (default next to input)
      # Test with explicit output_path
      # Test with different extensions
  ```

---

## Step 4 — Extract Common Reproducibility Functions

| Files | Action |
|-------|--------|
| `beat_detection/utils/reproducibility.py` `+` | Create new module for reproducibility utilities |
| `scripts/experiment_orchestrator.py` `±` | Use the common reproducibility functions |

Several functions related to experiment reproducibility should be extracted to a reusable module:

1. Move `get_git_info()` and `save_reproducibility_info()` to the new module
2. Enhance them to be more reusable across different experiment types
3. Update imports in experiment_orchestrator.py

### Unit tests

* Create tests for the reproducibility functions:
  ```python
  def test_get_git_info():
      # Mock subprocess to test Git info extraction
      
  def test_save_reproducibility_info():
      # Verify files are created with correct content
  ```

---

## Step 5 — Documentation & Update Tests

| Files | Action |
|-------|--------|
| `beat_detection/core/pipeline.py` `±` | Update docstrings to reflect new functionality |
| `scripts/experiment_orchestrator.py` `±` | Update docstrings to reflect refactored code |
| `CHANGELOG.md` `±` | Document the refactoring |

1. Update docstrings to clearly document the new parameters and behavior
2. Update CHANGELOG.md to note the refactoring of duplicate code
3. Run all tests to ensure nothing was broken

---

## Benefits of Refactoring

1. **Reduced Code Duplication**: Eliminates duplicate batch processing logic
2. **Improved Maintainability**: Changes to batch processing need to be made in only one place
3. **Better Encapsulation**: Genre-specific logic is properly contained in the pipeline module
4. **Enhanced Reusability**: Extracted utilities can be used in other parts of the codebase
5. **Clearer Responsibility**: Each module has a well-defined purpose without overlap

---

**Milestone complete** when:

1. All tests pass with `pytest`
2. Experiment orchestrator works with the same functionality as before
3. Code duplication is eliminated