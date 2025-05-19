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

## Step 2 — Update Imports and References

This is a critical step and requires careful searching and updating across the entire codebase.

| Files                 | Action |
|-----------------------|--------|
| `**/*.py`             | `±` Update import statements from `beat_detection.*` to `beat_counter.*` and `web_app.*` to `beat_counter.web_app.*`. |
| `**/*.md`             | `±` Update any documentation or examples referencing old paths. |
| `pyproject.toml`      | `±` Check and update any path-specific configurations (e.g., package includes, script paths). |
| `.github/workflows/*` | `±` Check and update CI/CD pipeline configurations, script paths, or PYTHONPATH adjustments. |
| `tests/**/*.py`       | `±` Update import statements and file paths used in tests. |
| `scripts/*`           | `±` Check any utility scripts for hardcoded paths. |
| `docs/*`              | `±` Update documentation if paths are mentioned. |

### Actions
1.  **Global Search and Replace:**
    *   Search for `from beat_detection` and replace with `from beat_counter`.
    *   Search for `import beat_detection` and replace with `import beat_counter`.
    *   Search for `from web_app` and replace with `from beat_counter.web_app`.
    *   Search for `import web_app` and replace with `import beat_counter.web_app`.
    *   Search for paths like `"beat_detection/"` and update to `"beat_counter/"`.
    *   Search for paths like `"web_app/"` and update to `"beat_counter/web_app/"`.
2.  **Review `sys.path` modifications:** Check for any scripts or modules that manipulate `sys.path` and update them accordingly.
3.  **Review `pyproject.toml`:** Ensure that package discovery, script definitions, or tool configurations correctly reflect the new structure.
4.  **Review CI/CD workflows:** Update any paths in GitHub Actions workflows or other CI scripts.

### Unit tests

- No new unit tests specifically for this step. Existing unit tests will be run in the next step to verify changes.

After fixing errors and ensuring all references are updated, commit changes: `git commit -m "feat: Update imports and references for beat_counter structure"`

---

## Step 3 — Run All Tests

### Unit tests

- Run all existing unit tests to ensure the refactoring didn't break any functionality.
  ```bash
  # (Command to run unit tests, e.g., pytest)
  pytest
  ```
- Address any failing tests. These failures will likely be due to incorrect import paths or references missed in Step 2.

### Integration Tests

- Run all existing integration tests.
  ```bash
  # (Command to run integration tests)
  pytest tests/integration # or similar
  ```
- Address any failing integration tests.

After fixing errors and all tests pass, commit changes: `git commit -m "test: Verify beat_counter refactor with all tests"`

---

## Check for accidental changes

Run `git diff main --stat` and `git diff main` and carefully review if there are no accidental changes and if all changes adhere to our project rules.
Pay close attention to changes in files that were not expected to be modified.

---

## Final Step — Documentation & CHANGELOG

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