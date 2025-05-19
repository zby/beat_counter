# Refactor Plan: Standardize Project Structure by Reorganizing `beat_detection` and `web_app`

This refactor aims to improve project organization by renaming the `beat_detection` directory to `beat_counter` and moving the `web_app` directory into the newly named `beat_counter` directory. This change will make the directory structure more standardized and intuitive.

**Pros:**
- **Improved Clarity:** `beat_counter` is a more descriptive name for the core functionality.
- **Better Organization:** Colocating `web_app` within `beat_counter` groups related components, enhancing modularity.
- **Standardization:** Aligns the project structure with common practices, making it easier for new developers to understand.
- **Simplified Imports (Potentially):** Could simplify import paths if `web_app` heavily relies on `beat_counter` components.

**Cons:**
- **Path Updates Required:** All import statements, file references, and configurations (e.g., `PYTHONPATH` modifications, `sys.path` adjustments, CI/CD scripts, `pyproject.toml`) referencing the old paths (`beat_detection/` and `web_app/` at the root) will need to be updated.
- **Risk of Broken Functionality:** If any reference is missed, it could lead to runtime errors or broken builds. Thorough testing is crucial.
- **Merge Conflicts:** This change might cause merge conflicts with other active branches modifying files within these directories or referencing them.

This refactor will involve changes to file paths and potentially import statements across the codebase. No direct API changes are expected, but module paths will change.

| Legend |
|--------|
| `+` new file |
| `±` modified file |
| `→` moved/renamed file |
| `×` deleted file |

---

## Step 0 — Initial setup

- Create a new feature branch: `git checkout -b feat/refactor-beat-counter-structure`
- Ensure all local changes are committed or stashed.
- Verify that CI is green on the main branch.

_No tests – administrative._

---

## Step 1 — Rename `beat_detection` and Move `web_app`

| Files                 | Action |
|-----------------------|--------|
| `beat_detection/`     | `→ beat_counter/` Rename the directory. |
| `web_app/`            | `→ beat_counter/web_app/` Move the directory. |

### Actions
1. Rename the `beat_detection` directory to `beat_counter`.
2. Move the `web_app` directory into the `beat_counter` directory.

_No specific unit tests for this step, as it's a structural change. Subsequent steps will verify the correctness._

---

## Step 2 — Update Core Python Imports for `beat_detection`

| Files                 | Action |
|-----------------------|--------|
| `beat_counter/**/*.py`| `±` Update import statements from `beat_detection.*` to `beat_counter.*`. |

### Actions
1. **Update imports in core modules:**
   * Search for `from beat_detection` and replace with `from beat_counter` in core modules.
   * Search for `import beat_detection` and replace with `import beat_counter` in core modules.
   * Search for paths like `"beat_detection/"` and update to `"beat_counter/"`.

### Unit tests
- No new unit tests specifically for this step. Core module imports will be verified when running tests in a later step.

After updating core module imports, commit changes: `git commit -m "refactor: Update beat_detection imports in core modules"`

---

## Step 3 — Update Core Python Imports for `web_app`

| Files                 | Action |
|-----------------------|--------|
| `beat_counter/**/*.py`| `±` Update import statements from `web_app.*` to `beat_counter.web_app.*`. |

### Actions
1. **Update imports in core modules:**
   * Search for `from web_app` and replace with `from beat_counter.web_app` in core modules.
   * Search for `import web_app` and replace with `import beat_counter.web_app` in core modules.
   * Search for paths like `"web_app/"` and update to `"beat_counter/web_app/"`.

### Unit tests
- No new unit tests specifically for this step. Core module imports will be verified when running tests in a later step.

After updating core module imports, commit changes: `git commit -m "refactor: Update web_app imports in core modules"`

---

## Step 4 — Update Test Imports for `beat_detection`

| Files                 | Action |
|-----------------------|--------|
| `tests/**/*.py`       | `±` Update import statements and file paths related to `beat_detection`. |

### Actions
1. **Update imports in test files:**
   * Search for `from beat_detection` and replace with `from beat_counter` in test files.
   * Search for `import beat_detection` and replace with `import beat_counter` in test files.
   * Update test fixtures that reference the `beat_detection` path.

### Unit tests
- No new unit tests specifically for this step, as we're updating the existing test files themselves.

After updating test imports, commit changes: `git commit -m "refactor: Update beat_detection imports in test files"`

---

## Step 5 — Update Test Imports for `web_app`

| Files                 | Action |
|-----------------------|--------|
| `tests/**/*.py`       | `±` Update import statements and file paths related to `web_app`. |

### Actions
1. **Update imports in test files:**
   * Search for `from web_app` and replace with `from beat_counter.web_app` in test files.
   * Search for `import web_app` and replace with `import beat_counter.web_app` in test files.
   * Update test fixtures that reference the `web_app` path.

### Unit tests
- No new unit tests specifically for this step, as we're updating the existing test files themselves.

After updating test imports, commit changes: `git commit -m "refactor: Update web_app imports in test files"`

---

## Step 6 — Update Configuration Files

| Files                 | Action |
|-----------------------|--------|
| `pyproject.toml`      | `±` Check and update any path-specific configurations (e.g., package includes, script paths). |
| `setup.py` (if exists)| `±` Update package references and paths. |
| `setup.cfg` (if exists)| `±` Update any configuration referencing old paths. |

### Actions
1. **Review and update configuration files:**
   * Check `pyproject.toml` for any references to `beat_detection` or `web_app` paths.
   * Update package discovery settings to reflect the new structure.
   * Update any script definitions that reference the old paths.
   * Check for any tool configurations (pytest, black, etc.) that might reference the old paths.

### Unit tests
- No specific unit tests for this step, as configuration changes will be validated when running the full test suite later.

After updating configuration files, commit changes: `git commit -m "refactor: Update configuration files for beat_counter structure"`

---

## Step 7 — Update CI/CD and Scripts

| Files                 | Action |
|-----------------------|--------|
| `.github/workflows/*` | `±` Check and update CI/CD pipeline configurations and script paths. |
| `scripts/*`           | `±` Check any utility scripts for hardcoded paths. |
| Any files with `sys.path` modifications | `±` Review and update path manipulations. |

### Actions
1. **Review and update CI/CD and scripts:**
   * Update any GitHub Actions workflows or other CI scripts that reference the old paths.
   * Check utility scripts in the `scripts/` directory for hardcoded paths.
   * Review any files that modify `sys.path` and update path references.
   * Update any PYTHONPATH adjustments in scripts or workflow files.

### Unit tests
- No specific unit tests for this step, but CI pipeline will validate changes when run.

After updating CI/CD and scripts, commit changes: `git commit -m "refactor: Update CI/CD workflows and scripts for beat_counter structure"`

---

## Step 8 — Update Documentation

| Files                 | Action |
|-----------------------|--------|
| `**/*.md`             | `±` Update any documentation or examples referencing old paths. |
| `docs/*`              | `±` Update documentation if paths are mentioned. |
| Jupyter notebooks (if any) | `±` Check and update import statements and paths. |

### Actions
1. **Review and update documentation:**
   * Update any markdown files that reference the old paths.
   * Check documentation in `docs/` directory for mentions of the old structure.
   * Update code examples in documentation to use the new import paths.
   * Check Jupyter notebooks if they exist and update imports/paths.

### Unit tests
- No specific unit tests for documentation changes.

After updating documentation, commit changes: `git commit -m "docs: Update documentation with new beat_counter structure"`

---

## Step 9 — Run All Tests

### Unit tests

- Run all existing unit tests to ensure the refactoring didn't break any functionality.
  ```bash
  # (Command to run unit tests, e.g., pytest)
  pytest
  ```
- Address any failing tests. These failures will likely be due to incorrect import paths or references missed in previous steps.

### Integration Tests

- Run all existing integration tests.
  ```bash
  # (Command to run integration tests)
  pytest tests/integration # or similar
  ```
- Address any failing integration tests.

After fixing errors and all tests pass, commit changes: `git commit -m "test: Verify beat_counter refactor with all tests"`

---

## Step 10 — Check for accidental changes

Run `git diff main --stat` and `git diff main` and carefully review if there are no accidental changes and if all changes adhere to our project rules.
Pay close attention to changes in files that were not expected to be modified.

---

## Step 11 — Documentation & CHANGELOG

- **Documentation Updates**:
    - Review all project documentation (`README.md`, files in `docs/`, etc.) for mentions of `beat_detection` or the old `web_app` path and update them.
    - Ensure architectural diagrams or descriptions reflect the new structure.
- **CHANGELOG**:
    - Add an entry to `CHANGELOG.md` (or create it if it doesn't exist) under a new version or "Unreleased" section:
      ```markdown
      ### Changed
      - Renamed `beat_detection` directory to `beat_counter` and moved `web_app` directory into `beat_counter/web_app/` to standardize project structure and improve clarity.
      ```

Commit the changes: `git commit -m "docs: Update documentation and CHANGELOG for beat_counter refactor"`

---

### Total new/updated tests

No new tests are introduced by this refactor. The focus is on ensuring all existing tests pass after the structural changes.

| File        | Purpose                                     |
|-------------|---------------------------------------------|
| `All tests` | Verify correctness after directory renaming and moving. |

Test runtime expectations: Should be similar to existing test suite runtime.

---

**Milestone complete** when:

1. All automated checks (linters, tests) pass in CI on the feature branch.
2. The `beat_detection` directory is successfully renamed to `beat_counter`.
3. The `web_app` directory is successfully moved into `beat_counter/web_app/`.
4. All internal imports and references to the old paths are updated.
5. Documentation and CHANGELOG are updated.
6. The PR is reviewed and approved. 