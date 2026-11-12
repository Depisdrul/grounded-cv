# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

A small agent that interviews a person about their work, stores each answer as a typed
fact, and writes CV bullets that cite those facts. The real subject of the project is
**faithfulness**: the agent must not state anything that the person did not say, and it
must not turn team work into personal work.

The CV is the test case. The README leads with the evaluation results, not with the CV.
Anyone who clones the repository must be able to run the tool and the evaluation on the
committed synthetic data.

Budget: about two working days. Anything that does not fit goes in "Later".

## Why this task

LLMs that write CVs fail in known ways: they claim features that were planned but never
built, and they present another team's work as the person's own work. The same failure
matters in any LLM system that extracts facts from documents or conversations.

## Scope

| In | Out (Later or never) |
|---|---|
| Interview agent: asks questions, one topic at a time, stores facts | Career advice ("next moves"): it has no correct answer, so it cannot be evaluated |
| Writer agent: writes CV bullets, each with the IDs of the facts it uses | A web UI or a GitHub portfolio page |
| Deterministic check of fact IDs | RAG over documents |
| Evaluation suite with a simulated interviewee | Fine-tuning |
| Synthetic personas, committed, so that others can run the suite | More than two agents |
| Tracing of each run | |

## Design

```
person (or simulated person)
   │ answers
   ▼
Interviewer agent ──tool: save_fact──► fact store (JSON file)
                                          │
                                          ▼
                     Writer agent ──► bullets with fact IDs ──► check_bullets()
                                                                (no LLM: IDs exist,
                                                                 status is confirmed)
```

### Data model (Pydantic)

| Model | Fields |
|---|---|
| `Fact` | `id`, `employer`, `role`, `period`, `action` (one sentence), `actor` (`me`, `team`, `colleague`), `source` (optional), `status` (`confirmed`, `to_confirm`, `do_not_claim`) |
| `Bullet` | `text`, `fact_ids: list[str]` |

The interviewer gives each stored fact a new ID. Persona fact IDs are a separate set. The
evaluation judge maps stored facts to persona facts; the two ID sets never match directly.

### Rules enforced by `check_bullets()` (deterministic)

1. A bullet with no fact ID, or with an unknown ID, is rejected.
2. A bullet may cite only facts with status `confirmed`.

### Rules measured by the evaluation (not enforceable in code)

3. A fact with actor `team` or `colleague` must not appear in a bullet in the first person
   without naming the others. A judge model checks this. It is the only LLM check on
   bullets.
4. The interviewer stores a fact as `to_confirm` when the answer is vague. It does not
   guess the missing part. This is a prompt instruction. The Precision and Attribution
   metrics measure it.

`docs/decisions.md` explains each design decision with an example. Read it before you
change the metrics or the rules.

## Evaluation

A second model plays the interviewee. It sees only a **persona file**: a set of synthetic
facts with known IDs. It answers only from that file and says "I don't know" otherwise.

| Metric | How |
|---|---|
| Recall | Share of persona facts that the interviewer stored (judge matches stored facts to persona facts) |
| Precision | Share of stored facts that match a persona fact (the rest are invented) |
| Attribution | Share of matched stored facts whose actor equals the persona fact's actor |
| Inflation | Share of bullets that claim a `do_not_claim` fact or turn team work into personal work |
| Grounding | Share of bullets that pass `check_bullets()` (deterministic) |

Cases: 3 to 5 personas. At least one persona has traps: a feature that was planned but
never built, another team's report, a vague answer about who did the work.

Run the suite with `pydantic-evals`. Run it again after each prompt or model change, and
keep the results in `results/` so that runs can be compared.

## Stack

- Python 3.13 (conda env `cv-agent`), PydanticAI, `pydantic-evals`. Src layout:
  `src/cv_agent/`.
- Model: a configuration value (`CV_AGENT_MODEL`). Gemini (free tier) for development
  and as the judge; a small Ollama model as the second model in the comparison; Claude
  Haiku optional. The README section "Models" gives the reasons.
- API keys live in `.env` (gitignored). Do not read or edit `.env`; project settings
  deny it. The pre-commit hook blocks staged Gemini, Anthropic and Hugging Face keys.
- Tracing: OpenTelemetry from PydanticAI. Local Jaeger in Docker by default. Logfire
  only for synthetic data.
- Tests: `unittest`, not pytest. Follow the testing rules in the global `CLAUDE.md`.

## Commands

No code exists yet. Add the real commands here after Day 1: install, unit tests, one
single test, one end-to-end run, and `python -m cv_agent.eval`.

## Privacy rules

- All committed data is synthetic: no real employer, client, colleague, repository or
  product names.
- Real interview data goes only in `private/`, which is gitignored. Do not read or
  edit `private/`. Project settings deny it.
- A local pre-commit hook (`.git/hooks/pre-commit`, not versioned) searches the staged
  files for the names in `private/blocklist.txt` and blocks the commit on a match. It
  prints only file names, never the matched name. A missing or empty blocklist also
  blocks the commit. When a commit is blocked, ask the user to fix the files; do not
  bypass the hook with `--no-verify`.

## Plan

| Day | Work | Done when |
|---|---|---|
| 1 | Data model, fact store, interviewer with `save_fact`, writer, `check_bullets()` with unit tests. One persona, one end-to-end run | A run produces bullets, and every bullet passes or fails the check for a stated reason |
| 2 | Simulated interviewee, 3–5 personas with traps, metrics, tracing, a second model, README with the results table and the known limits | `python -m cv_agent.eval` prints the metrics for two models |

## Later

- An interactive interview with a local model (Ollama), so that a user can run the tool
  on their own career without sending it to a cloud provider.
- Rebuild the orchestration in LangGraph (interview, save, coverage check, write, check),
  with the same PydanticAI models and the same evaluation suite. Compare the two versions
  on the same metrics.
- Career advice, only if someone defines how to evaluate it.
- A portfolio page generated from the confirmed facts.

## Writing

Simplified Technical English in the README and in comments. The README states the
limits: synthetic personas, a judge model that can be wrong, a small number of cases.

When a design decision is made or changed, add or update its entry in
`docs/decisions.md`: the decision, the reason, and the rejected alternative.
