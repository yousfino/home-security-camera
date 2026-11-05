# Engineering rules

These rules apply to the entire repository.

- Maintain compatibility with Python 3.11.
- Follow the existing repository architecture unless a refactor is justified.
- Use type hints and modular, testable Python code.
- Keep computer vision processing local-first.
- Never require a physical camera to run automated tests.
- Mock camera and inference dependencies in unit tests.
- Never commit biometric embeddings, face crops, database files, model weights, or secrets.
- Preserve the `familiar`, `unfamiliar`, and `unresolved` classification states.
- Do not treat an unfamiliar person as a confirmed intruder.
- Keep recognition thresholds configurable and document their limitations.
- Add or update tests for every behavioral change.
- Run the existing test suite before completing a task. Use Python 3.11 and run
  `python -m pytest -q` from the repository root with the project environment activated.
- Prefer small, reviewable pull requests.
- Do not introduce Firebase, Android, or remote networking until those milestones
  are explicitly requested.
