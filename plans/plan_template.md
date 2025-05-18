# Refactor Plan: [Title] - [Brief one-line description]

This refactor [explain purpose and benefits]. 
[Mention any API changes or breaking changes]

| Legend |
|--------|
| `+` new file |
| `±` modified file |
| `→` moved/renamed file |
| `×` deleted file |

---

## Step 0 — [Initial setup step]

- [List administrative tasks like branch creation]
- [CI configuration]

_No tests – administrative._

---

## Step 1 — [First logical step]

| Files | Action |
|-------|--------|
| `[path/to/file]` | `±` [specific action description] |
| `[additional files]` | `+` [specific action description] |

### Unit tests

[Unit tests should be colocated with code]

* `[test file path]`
  ```python
  [sample test code]
  ```
* [Test performance expectation]

Run the new tests and iterate on fixing errors.

After fixing errors commit changes.

---

## Step 2 — [Second logical step]

| Files | Action |
|-------|--------|
| `[path/to/file]` | `±` [specific action description] |
| `[additional files]` | `+` [specific action description] |

### Unit tests

* `[test file path]`
  ```python
  [sample test code]
  ```

Run the new tests and iterate on fixing errors.

After fixing errors commit changes.

---

[Continue with Steps 3-N following the same structure]

---

## Check for accidental changes

Run `git diff main` and check if there are no accidental changes and if all changes adhere to our project rules.

---

## Integration Tests

[Integration tests in this project go into the tests directory]

- [Update existing integration tests to accommodate refactored code]
- [Note: New features may require additional integration tests]
- Run all integration tests and fix errors if encountered

### Example integration test
```python
def test_end_to_end_workflow():
    # Set up test environment as an end user would
    # Execute the main workflow/process
    # Verify outputs and system state
```

_Note: Refactors typically only require updating existing integration tests, not creating new ones._

---

## Check Unit tests

Run all unit tests again - just for check.

---

## Final Step — Documentation & CHANGELOG

- [Documentation updates]
- [CHANGELOG entries]

Commit the changes.

---

### Total new/updated tests

| File | Purpose |
|------|---------|
| `[test file]` | [purpose] |
| `[test file]` | [purpose] |

[Test runtime expectations]

---

**Milestone complete** when:

1. [Verification step 1]
2. [Verification step 2]
3. [Verification step 3]