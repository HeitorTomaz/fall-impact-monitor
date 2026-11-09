# Claude Code Instructions — Fall Impact Monitor

## Interaction language

- **Speak to the project owner in Brazilian Portuguese (pt-BR)**: planning discussions, questions, progress reports, explanations and summaries.
- **Write all repository artifacts in English**: source code, symbols, comments, docstrings, tests, filenames, configuration descriptions, logs, CLI text, README/docs, commit messages, branches, GitHub issues and pull requests.
- Translate the outcome of a Portuguese discussion into concise English technical artifacts.

## Project identity and privacy

- This is an **independent open-source project**. Never refer to unrelated software projects, their branding, organizations, repositories or infrastructure in public content.
- The Git repository must never contain real household footage, images of visitors, children, credentials, camera addresses, tokens or identifying environment information.
- Local test recordings are stored at the ignored relative path test_videos/. No footage or derived exports may be committed.
- Never paste RTSP credentials, secrets or local absolute paths into commit messages, issues, PRs, logs or public reports.
- Use safe example values, environment variables and synthetic test fixtures.
- Never inspect or upload private test videos to remote services without explicit authorization.

## Architecture, design and scope

- **First task: run the installed brainstorming skill in Claude Code Plan Mode before implementing.** If the skill is not available, explain that in Portuguese and follow a comparable explicit design-review process, without pretending to have run it.
- Read docs/ARCHITECTURE.md and GitHub Issues #1–#9 as **proposed** design, not immutable requirements. Challenge choices and simplify when possible.
- Brainstorming should cover goals/non-goals, alternatives, failure modes, privacy, licensing, evaluation strategy, child-event filtering and realistic deadlines.
- Present recommended design and tradeoffs in Portuguese; record decisions and rationale in English. Ask for approval before starting code.
- Do not start Phase 2 or later phases during Phase 1 implementation.
- Prioritize small, well-tested, portable Python modules over infrastructure complexity.
- Developer workstation is Windows; future server is Linux CPU-first with optional video decode acceleration.
- The current camera is a Tapo C113 over local RTSP; prior FFmpeg/VLC capture smoothness and timing discrepancies remain unproven.

## Licensing

- Original project source is Apache-2.0.
- Audit dataset, pretrained weight, model code and dependency licenses separately.
- Do not use non-commercial, research-only, no-derivatives, ambiguous-license or noncommercial-model-deployment datasets in the default pipeline. Confirm permissions for training and trained-weight distribution independently.
- Record canonical license references, obligations and approval before using any external asset.

## GitHub workflow

- Keep work on a feature branch and open a PR against main; do not push implementation directly to main.
- Maintain automated tests and documentation alongside each change.
- Do not execute untrusted PR code with repository secrets.
- Do not mark any safety-critical model clinically validated without evidence.
