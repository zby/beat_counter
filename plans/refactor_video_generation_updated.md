# Updated Refactoring Plan: Video Generation - Improve Testability and Separation of Concerns

This refactor aims to improve the testability of the video generation module (`beat_counter/core/video.py`) by decoupling file I/O operations and object instantiation from the core generation logic. It will also enhance the separation of concerns, making the code easier to understand, maintain, and test.

**Breaking Changes:**
- The API of `beat_counter.core.video.BeatVideoGenerator.generate_video` will change.
- The API of `beat_counter.core.video.generate_single_video_from_files` will remain mostly unchanged.
- A new function for writing video files will be introduced, and `generate_single_video_from_files` will use it.

| Legend |
|--------|
| `+` new file |
| `±` modified file |
| `→` moved/renamed file |
| `×` deleted file |

---

## Step 0 — Initial setup ✅ (Completed)

- Create a new feature branch: `git checkout -b refactor/video-generation`
- Ensure all existing tests pass: `pytest` and `pytest -m slow`.

---

## Step 1 — Modify `BeatVideoGenerator.generate_video` to accept `AudioFileClip` ✅ (Completed)

| Files | Action |
|-------|--------|
| `beat_counter/core/video.py` | `±` 1. Modify `BeatVideoGenerator.generate_video` method:
   - Change its `audio_path: Union[str, pathlib.Path]` parameter to `audio_clip: AudioFileClip`.
   - Remove the internal instantiation of `AudioFileClip`.
2. Update `generate_single_video_from_files` to instantiate `AudioFileClip` and pass it to `BeatVideoGenerator.generate_video`. |
| `beat_counter/core/test_video.py` | `±` Update tests for `BeatVideoGenerator.generate_video` and `generate_single_video_from_files` to reflect the change. Mock `AudioFileClip` where appropriate. |

---

## Step 2 — Use Helper Function to Improve Testing but Keep API Simple

| Files | Action |
|-------|--------|
| `beat_counter/core/video.py` | `±` 1. Keep the `prepare_beats_from_file` helper function we created.
2. Revert the signature of `generate_single_video_from_files` to accept `beats_file: Path` instead of `beats: Beats`.
3. Update the body of `generate_single_video_from_files` to call `prepare_beats_from_file` internally.
4. Update `generate_batch_videos` to call `generate_single_video_from_files` with file paths directly. |
| `beat_counter/cli/generate_video.py` | `±` Revert to calling `generate_single_video_from_files` directly with the beats file path and other parameters. |
| `beat_counter/core/test_video.py` | `±` Keep tests for `prepare_beats_from_file`.
Update tests for `generate_single_video_from_files` to reflect its original signature with file paths.
Update mock configurations in test cases to properly test the refactored code. |

### Unit tests

* Keep all `prepare_beats_from_file` tests
* Revert tests for `generate_single_video_from_files` to match its original signature (accepts `beats_file: Path`)
* Test performance expectation: No significant change.

---

## Step 3 — Decouple Video Writing from `BeatVideoGenerator.generate_video`

| Files | Action |
|-------|--------|
| `beat_counter/core/video.py` | `±` Modify `BeatVideoGenerator.generate_video` to return the `VideoClip` object instead of writing it to a file. Remove the `output_path` parameter and the `video.write_videofile(...)` call. |
| `beat_counter/core/video.py` | `+` Create a new utility function `write_video_clip(video_clip: VideoClip, output_path: Union[str, pathlib.Path], fps: int, codec: str, audio_codec: str, logger: Optional[str] = "bar")` that takes a `VideoClip` and other necessary parameters to write it to a file. |
| `beat_counter/core/video.py` | `±` Update `generate_single_video_from_files` to call the new `write_video_clip` function with the `VideoClip` returned by `BeatVideoGenerator.generate_video`. |
| `beat_counter/core/test_video.py` | `±` Update tests for `BeatVideoGenerator.generate_video` to assert the properties of the returned `VideoClip`. |
| `beat_counter/core/test_video.py` | `+` Create new tests for the `write_video_clip` function, mocking `video_clip.write_videofile` and verifying parameters. |

### Unit tests

* Update tests for `BeatVideoGenerator.generate_video` to verify the returned `VideoClip`
* Create new tests for `write_video_clip`
* Test performance expectation: No significant change.

---

## Check for accidental changes

Run `git diff main` and check if there are no accidental changes and if all changes adhere to our project rules. Review docstrings and type hints.

---

## Integration Tests

- `tests/` directory:
  - Run tests to verify that the API changes work correctly within the broader system.
  - Ensure CLI commands that use video generation still function properly.

Run all integration tests (`pytest -m slow`) and fix errors if encountered.

---

## Check Unit tests

Run all unit tests again: `pytest`.

---

## Final Step — Documentation

- Update docstrings for:
    - `BeatVideoGenerator.generate_video`
    - `prepare_beats_from_file`
    - The new `write_video_clip` function.
- Ensure parameter types and return types are correctly documented.

---

**Milestone complete** when:

1. All three refactoring steps are implemented.
2. All unit tests pass.
3. All integration tests pass.
4. Documentation is updated.
5. The `refactor/video-generation` branch is ready for review and merge. 