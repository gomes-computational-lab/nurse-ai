# Incremental Feature Development Workflow

Use this workflow for every task, bug fix, and feature. The goal is to make one
small, observable change at a time and verify it before adding more behavior.

## Core rule

Do not implement an entire multi-part feature in one pass. Split it into the
smallest useful vertical slices. Each slice must be understandable, testable,
and working before the next slice begins.

A vertical slice should deliver one behavior through all necessary layers. For
example, add and validate one configuration value before implementing the
provider that consumes it.

## 1. Start from a known baseline

Before changing code:

1. Confirm the target branch for the eventual pull request. Use `staging` unless
   another target is explicitly agreed upon.
2. Fetch the target branch and create a new branch from its latest commit.
3. Use a descriptive branch name such as `feat/local-patient-voice` or
   `fix/invalid-provider-fallback`.
4. Check `git status` and preserve unrelated or uncommitted user work.
5. Run the existing test suite. If the baseline is failing, document and resolve
   that before implementing the feature.

## 2. Define the next small behavior

Write down one behavior and its acceptance criteria before coding. A good slice
normally changes one responsibility and can be reviewed independently.

Example:

> Behavior: Reject an unknown TTS provider without making a network request.
>
> Acceptance criteria: Invalid provider values raise a configuration error, Edge
> is never selected implicitly, and a regression test proves both conditions.

If a slice needs several unrelated acceptance criteria, split it again.

## 3. Add the test first or alongside the change

For each slice:

1. Add a focused test that fails for the missing behavior or reproduces the bug.
2. Implement only enough code to satisfy the stated acceptance criteria.
3. Run the focused test immediately.
4. Fix failures before changing another subsystem.
5. Run tests for neighboring behavior to detect local regressions.

Do not defer testing until the complete feature is implemented.

## 4. Verify the slice before continuing

A slice is complete only when all applicable checks pass:

```bash
python -m pytest -q path/to/relevant_test.py
ruff check changed_file.py tests/changed_test.py
ruff format --check changed_file.py tests/changed_test.py
git diff --check
```

Also perform the smallest meaningful integration or manual check when the change
touches audio, browser permissions, model loading, concurrency, or UI behavior.

Before starting the next slice, review the diff and confirm:

- the acceptance criteria are satisfied;
- no unrelated behavior was added;
- errors fail safely and provide useful messages;
- temporary files and resources are cleaned up;
- documentation matches the implemented behavior;
- no text, audio, or other simulation data is sent outside approved local or
  university-controlled systems.

## 5. Commit one completed slice

Commit only after the slice is working and tested. Keep the commit focused and
use a message that describes its observable result, for example:

```text
fix: reject unknown TTS providers
```

Do not combine formatting cleanup, refactors, and new behavior unless they are
required for the same slice. Never begin an unrelated feature on the same
branch.

## 6. Repeat for the next slice

After a successful commit:

1. Select the next smallest behavior.
2. Restate its acceptance criteria.
3. Add its focused test.
4. Implement and verify it.
5. Commit it separately.

If a new failure appears, stop and diagnose it before adding more code. Do not
build additional behavior on top of a failing state.

## 7. Final feature verification

When every slice is complete, run the complete project checks:

```bash
ruff check .
ruff format --check .
python -m pytest -q
git diff --check
```

Then perform the feature's documented manual test plan. For voice features, this
includes real microphone input, local transcription, actual TTS output, fallback
behavior, cancellation, and a run with internet access disabled when fully local
operation is required.

## Pull request checklist

Before opening or updating a pull request:

- [ ] The branch is based on the latest agreed target branch.
- [ ] Each commit represents one tested behavior.
- [ ] Focused tests exist for each new behavior and corrected bug.
- [ ] The complete test suite passes.
- [ ] Lint, formatting, and whitespace checks pass.
- [ ] Required manual tests were completed and recorded.
- [ ] External network use is absent or explicitly disclosed and approved.
- [ ] Setup, troubleshooting, and user-facing instructions are current.
- [ ] Known limitations and unverified hardware paths are documented.

If any required check is incomplete, keep the pull request in draft and do not
begin another feature on that branch.
