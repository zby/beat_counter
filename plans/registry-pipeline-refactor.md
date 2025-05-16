# Refactor Plan № 1 – Extract Registry & Pipeline out of `factory.py`

This refactor breaks the monolithic `factory.py` into small, single‑purpose
modules. The primary public API (`get_beat_detector()` &
`extract_beats()`) will be exposed via `beat_detection.core`.
**This is a breaking change for any code importing directly from `beat_detection.core.factory` as it will be removed.**

| Legend |
|--------|
| `+` new file |
| `±` modified file |
| `→` moved/renamed file |
| `×` deleted file |

---

## Step 0 — Create a protected feature branch

- Branch name suggestion: `refactor/registry‑pipeline`
- Make sure CI runs on the branch.

_No tests – administrative._

---

## Step 1 — Introduce a Detector **registry** ✓

| Files | Action |
|-------|--------|
| `beat_detection/core/registry.py` `+` | add `register()` decorator, `_DETECTORS` dict and `get()` helper |
| `beat_detection/core/__init__.py` `±` | export `registry.get` as `get_beat_detector` to define the public API |

### Unit tests

* `beat_detection/core/test_registry.py`
  ```python
  from beat_detection.core.registry import register, get

  @register("dummy")
  class _Dummy: pass

  def test_lookup_success():
      assert isinstance(get("dummy"), _Dummy)

  def test_lookup_failure():
      import pytest, beat_detection.core.registry as reg
      with pytest.raises(ValueError):
          reg.get("missing")
  ```
* CI runtime ≤ 0.1 s

---

## Step 2 — Move **detector classes** to the detectors package

| Files | Action |
|-------|--------|
| `beat_detection/core/detectors/__init__.py` `+` | re‑export concrete detectors |
| `beat_detection/core/detectors/base.py` `+` | moved from `base_detector.py` |
| `beat_detection/core/madmom_detector.py` `→` | move to `detectors/madmom.py` and add `@register("madmom")` |
| `beat_detection/core/beat_this_detector.py` `→` | move to `detectors/beat_this.py` and add `@register("beat_this")` |
| `beat_detection/core/base_detector.py` `×` | to be deleted after moving to `detectors/base.py` |

### Unit tests

* `beat_detection/core/test_detectors_import.py`  
  Iterates over `beat_detection.core.registry._DETECTORS` and instantiates
  each detector with minimal kwargs, asserting **no import error**.

  ```python
  import inspect, registry
  for name in registry._DETECTORS:
      det = registry.get(name)
      assert inspect.isclass(det.__class__)
  ```

---

## Step 3 — Extract a **pipeline** module

| Files | Action |
|-------|--------|
| `beat_detection/core/pipeline.py` `+` | functions `extract_beats()`, `process_batch()` from factory.py |
| `beat_detection/core/factory.py` `±` | remove pipeline code. This module will be further dismantled. |
| `beat_detection/core/__init__.py` `±` | export `pipeline.extract_beats`, `pipeline.process_batch` |
| CLI scripts (`scripts/*.py`, `beat_detection/cli/*.py` if applicable) `±` | prepare to call `pipeline.extract_beats()` (full update in next step) |

### Unit tests

* `beat_detection/core/test_pipeline.py`
  *Use a mock detector*:

  ```python
  class FakeDet:
      def detect(self, path): return [0.1, 0.2, 0.3]

  from beat_detection.core.pipeline import extract_beats
  def test_extract_uses_detector(monkeypatch, tmp_path):
      (tmp_path/"x.wav").touch()
      assert extract_beats(tmp_path/"x.wav", FakeDet()).beats == [0.1, 0.2, 0.3]
  ```

---

## Step 4 — Refactor **CLI** layer

| Files | Action |
|-------|--------|
| `scripts/*` `±` & `beat_detection/cli/*` `±` | replace direct `factory` imports with imports from `beat_detection.core` (which now exposes `registry.get` as `get_beat_detector` and `pipeline` functions). |
| Migration to **Typer** is deferred and out of scope for this refactor. |

### Unit tests

* `tests/cli/test_help.py` uses `subprocess` to assert `--help` lists all detector names from registry.

---

## Step 5 — Update **import paths**, Remove `factory.py` & Run Static Checks

| Files | Action |
|-------|--------|
| Entire codebase `±` | Review for any remaining imports from the old `beat_detection.core.factory` and update them to use the new `beat_detection.core` public API or direct imports from `registry` and `pipeline` where appropriate internally. |
| `beat_detection/core/factory.py` `×` | Delete this file. |
| Project-wide | Run `isort`, `ruff`, `mypy`. Ensure they pass. |
| CI | Ensure CI pipeline is green. |

_No new tests for this step, focuses on cleanup and verification._

---

## Step 6 — Update **Integration Tests**

| Files | Action |
|-------|--------|
| `tests/*.py` `±` | Review and update existing integration tests (especially those in the root `/tests/` directory, not colocated unit tests) to ensure they align with the new module structure. This includes updating import paths and how detectors or pipeline functions are accessed (e.g., via `beat_detection.core`). |

_No new test files created; existing integration tests are adapted._

---

## Step 7 — Documentation & CHANGELOG

- Update README diagrams & examples to reflect the new module structure.
- Add **UPGRADE** note to CHANGELOG: "Internal module structure has been significantly refactored. `beat_detection.core.factory` has been removed. The primary public API (e.g., `get_beat_detector`, `extract_beats`) is now available directly from `beat_detection.core`. Update your imports accordingly (e.g., `from beat_detection.core import get_beat_detector`)."

---

### Total new/updated tests

| File | Purpose |
|------|---------|
| `test_registry.py` | registry API |
| `test_detectors_import.py` | plug‑in discovery |
| `test_pipeline.py` | extraction logic decoupled |
| `test_help.py` | CLI coherence |
| Existing integration tests (`/tests/`) | Adapt to new module structure |

All tests run in ~3 s, none require audio backends.

---

**Milestone complete** when:

1. `pytest -q` passes.  
2. CLI behaves the same as before (e.g., `python -m beat_detection.cli.main_script ...` or equivalent main CLI entry point).  
3. New PR reviewed and merged into `main`.  
