# Riyadh Natural-20 knowledge contract

Treat this vault as compiled project memory. Keep raw evidence immutable and
distinguish three scopes: a single segment, a pilot subset, and the official
5000-segment evaluation.

Before answering project-status questions:

1. Run `python knowledge/brain.py ingest --repo-root . --github-repo HadiRx/wasfa-project`.
2. Read `knowledge/Home.md` and the linked evidence note.
3. Cite the note and its underlying checkpoint or GitHub run.
4. Never label a per-segment or pilot mean as the leaderboard score.

After a completed experiment, ingest again and run
`python knowledge/brain.py audit --strict`. Preserve rejected experiments and
record why they were rejected in `Decisions/`.
