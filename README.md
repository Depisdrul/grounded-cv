# grounded-cv

An agent that interviews a person about their work and writes CV bullets that cite only
what the person said. The project measures one thing: **does the agent stay faithful to
its source?**

## Results

_To fill after the first evaluation run._

| Model | Recall | Precision | Attribution | Inflation | Grounding |
|---|---|---|---|---|---|
| | | | | | |

## How it works

The product has two agents and one deterministic check. The evaluation adds two more
model roles.

```
                    PRODUCT                                   EVALUATION ONLY
 ┌───────────────────────────────────────────┐
 │                                           │       persona file
 │  Interviewer agent ◄──── answers ─────────┼──── Simulated interviewee
 │        │          ───── questions ────────┼───►
 │        │ save_fact                        │
 │        ▼                                  │
 │   fact store ──► Writer agent ──► bullets │
 │                                     │     │
 │                          check_bullets()  │──► Grounding
 └───────────────────────────────────────────┘
       fact store + bullets + persona file ──► Judge ──► Recall, Precision,
                                                         Attribution, Inflation
```

| Role | Kind | Sees | Produces |
|---|---|---|---|
| Interviewer agent | LLM, product | The conversation | Facts in the fact store |
| Writer agent | LLM, product | The confirmed facts | Bullets, each with fact IDs |
| `check_bullets()` | Code, product | Bullets and fact store | Pass or fail for each bullet, with the reason |
| Simulated interviewee | LLM, evaluation only | One persona file and the questions | Answers. "I don't know" when the file has no answer |
| Judge | LLM, evaluation only | Persona file, fact store, bullets | Matches between facts, and a verdict for each bullet |

The interviewer never sees the persona file. The persona file is the answer key.

## Worked example

One persona, two facts, from interview to metrics. All data is synthetic.

**1. Persona file** (seen only by the simulated interviewee and the judge)

| ID | Persona fact | Actor |
|---|---|---|
| P2 | The platform team moved the company's servers to the cloud. | `team` |
| P3 | I wrote the scripts that moved the customer database. | `me` |

**2. Interview**

> **Interviewer:** What was your part in the cloud move?
> **Simulated interviewee:** The platform team moved the servers. I wrote the scripts
> for the customer database.

**3. Stored facts** (the interviewer creates new IDs)

| ID | Action | Actor | Status |
|---|---|---|---|
| F1 | Moved the company's servers to the cloud | `team` | `confirmed` |
| F2 | Wrote the scripts that moved the customer database | `me` | `confirmed` |

**4. Bullets** (the writer cites fact IDs)

| Bullet | Cites |
|---|---|
| B1: Wrote the migration scripts for the customer database during a cloud move. | F2 |
| B2: Moved the company's servers to the cloud. | F1 |

**5. Scores**

| Metric | Result | Why |
|---|---|---|
| Grounding | 2 of 2 | Both bullets cite facts that exist and are `confirmed`. Code checks this. |
| Recall | 2 of 2 | The judge matches F1 to P2 and F2 to P3. |
| Precision | 2 of 2 | Both stored facts match a persona fact. Nothing is invented. |
| Attribution | 2 of 2 | Both matched facts have the same actor as the persona fact. |
| Inflation | 1 of 2 | B2 has no subject, so on a CV it reads as personal work. F1 is team work. |

B2 passes the code check but fails the judge. This is the failure that the project
measures.

If the interviewer had stored F1 with actor `me`, Precision would still be 2 of 2: the
fact is not invented. Attribution would drop to 1 of 2.

## Metrics

| Metric | Question | Checked by |
|---|---|---|
| Recall | Did the interviewer collect the persona's facts? | Judge |
| Precision | Did the interviewer invent facts? | Judge |
| Attribution | Did the interviewer record who did the work correctly? | Judge |
| Inflation | Do the bullets claim more than the facts allow? | Judge |
| Grounding | Does each bullet cite facts that exist and are `confirmed`? | Code |

## Models

The model is a configuration value (`GROUNDED_CV_MODEL`). The code does not change between
models.

| Model | Used for | Why |
|---|---|---|
| Gemini (cloud, free tier) | Development runs, and the judge in every evaluation | Fast, and reliable with tool calls. The judge must be a strong model: a weak judge makes every metric noise. |
| Ollama (local, small model) | The second model in the results table | Free, runs offline, needs no API key. The comparison shows how a small local model differs from a cloud model on faithfulness. |
| Ollama (local) | A real interview about your own career | Real career data must not go to a cloud provider. A local model keeps it on your machine. |
| Claude Haiku (cloud, paid API) | Optional: a third model, or a substitute if the Gemini free-tier rate limits slow the runs | Low cost per run. Not needed to run the project. |

The judge stays the same model in all runs, so that results from different models can be
compared.

Limits of this choice:

- The Gemini free tier has rate limits. A full evaluation run can be slow.
- Free-tier data can be used by the provider to train its models. The committed data is
  synthetic, so this is acceptable here. Do not send real career data to it.
- A small local model may fail tool calls more often. That failure is part of what the
  comparison measures.

## Glossary

| Term | Meaning |
|---|---|
| Fact | One sentence about one piece of work, with its employer, role, period, actor and status. |
| Stored fact | A fact that the interviewer saved. IDs start with `F`. |
| Persona | A synthetic person used for evaluation. |
| Persona fact | A fact in a persona file: the ground truth. IDs start with `P`. |
| Actor | Who did the work: `me`, `team` or `colleague`. |
| Status | `confirmed`: the person stated it clearly. `to_confirm`: the answer was vague. `do_not_claim`: it must not appear on the CV, for example a feature that was planned but never built. |
| Bullet | One CV line. It lists the IDs of the facts that support it. A bullet with no subject reads as first person. |
| Match | The judge's decision that a stored fact and a persona fact describe the same work. |
| Trap | A persona fact that invites inflation: a cancelled feature, another team's work, a vague answer about who did the work. |
| Judge | A model that compares texts by meaning. It can be wrong. |

The reasons for each design choice are in [docs/decisions.md](docs/decisions.md).

## Limits

- The personas are synthetic and few.
- Recall, precision, attribution and inflation use a judge model, which can be wrong.
- _Add the limits found during the runs._

## Run

_To fill: setup, model configuration, tracing, evaluation command._
