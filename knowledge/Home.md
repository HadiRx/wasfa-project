---
type: dashboard
project: Riyadh Natural-20
updated_at: 2026-07-18T07:30:08+00:00
official_submission_score: null
---

# Riyadh Natural-20 / الرياض — الهدف الطبيعي 20

> [!info] Current truth
> The best verified checkpoint for segment `00000` is **19.691418**.
> The mean across the **11 available per-segment checkpoints** is
> **31.846522**, with **5** below 20. This is not a leaderboard result.

## Navigation

- [[Experiments/Auto-Results|Generated experiment results]]
- [[Experiments/Segment-00000|Segment 00000 history]]
- [[Experiments/Pilot-00001-00010|Ten-segment pilot]]
- [[Experiments/General-Controller-V1|General controller V1]]
- [[Decisions/Scale-Gate|Scale decision]]
- [[Decisions/General-Controller-Track|General controller decision]]
- [[Evidence/Verified-Scores|Verified scores and evidence rules]]
- [[Evidence/GitHub-Runs|GitHub execution history]]
- [[Submission/Controller-Design|Controller design]]
- [[Submission/Checklist|Official submission checklist]]
- [[Project-Discovery-Answers|Repository-backed discovery answers]]
- [[Sources/Comma-Official|Official comma sources]]
- [[Templates/Experiment|New experiment template]]

## Three commands

```bash
python knowledge/brain.py ingest --repo-root . --github-repo HadiRx/wasfa-project
python knowledge/brain.py query "why did we not scale to 5000"
python knowledge/brain.py audit --strict
```
