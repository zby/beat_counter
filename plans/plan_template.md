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
| `[path/to/file]` `[+/±/→/×]` | [specific action description] |
| `[additional files]` `[+/±/→/×]` | [specific action description] |

### Unit tests

[Unit tests should be colocated with code]

* `[test file path]`
  ```python
  [sample test code]
  ```
* [Test performance expectation]

---

[Continue with Steps 2-N following the same structure]

---

## Integration Tests

[Integration tests in this project go into the tests directory]

- [Update existing integration tests to accommodate refactored code]
- [Note: New features may require additional integration tests]

### Example integration test
```python
def test_end_to_end_workflow():
    # Set up test environment as an end user would
    # Execute the main workflow/process
    # Verify outputs and system state
```

_Note: Refactors typically only require updating existing integration tests, not creating new ones._

---

## Final Step — Documentation & CHANGELOG

- [Documentation updates]
- [CHANGELOG entries]

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