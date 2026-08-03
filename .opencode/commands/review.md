# /review

Run the Code Reviewer agent on the current working changes.

## Steps
1. Read `AGENTS.md`, `docs/architecture/`, and `docs/context.md`.
2. Run `git diff --stat` and `git diff`.
3. Delegate to `.opencode/agents/reviewer.md` with the diff context.
4. Output: bullet list of blockers, risks, and one-line approval verdict.
