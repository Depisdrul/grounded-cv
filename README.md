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
| Actor | Who did the work: `me`; `team`, a group that includes the person; `colleague`, another person or group, without the person. |
| Status | `confirmed`: the person stated it clearly. `to_confirm`: the answer was vague. `do_not_claim`: it must not appear on the CV, for example a feature that was planned but never built. |
| Bullet | One CV line. It lists the IDs of the facts that support it. A bullet with no subject reads as first person. |
| Match | The judge's decision that a stored fact and a persona fact describe the same work. |
| Trap | A persona fact that invites inflation: a cancelled feature, another team's work, a vague answer about who did the work, or a confirmed fact with an exact limit (released but not in production, estimated but not measured). |
| Judge | A model that compares texts by meaning. It can be wrong. |

The reasons for each design choice are in [docs/decisions.md](docs/decisions.md).

## Limits

- The personas are synthetic and few.
- Recall, precision, attribution and inflation use a judge model, which can be wrong.
- The simulated interviewee can reveal hidden details too early or invent answers.
  Check the saved transcript before drawing conclusions.
- This first suite has one persona. It is not a representative benchmark.
- A saved transcript is not a full OpenTelemetry trace.

## Run

Install Python 3.13 and the project:

```bash
python -m pip install -e ".[dev]"
```

Start Ollama with `qwen3.5:4b` available. Put `GEMINI_API_KEY` in the
environment or in a local `.env` file. Do not commit the key. Then run:

```bash
python -m grounded_cv.eval
```

The default product model is `ollama:qwen3.5:4b`. The simulated interviewee
and judge use `google:gemini-3.1-flash-lite`. Override them with
`--product-model`, `--interviewee-model` and `--judge-model`. Ollama uses
`OLLAMA_BASE_URL` when set and otherwise `http://localhost:11434/v1`.
Use only synthetic personas: the simulated interviewee and the judge can send their input
to a cloud provider.

Flash Lite was available for the first run, but it is a smaller judge. Its
verdicts need particular care when reviewed. Use a stronger judge when one is
available, and keep the judge fixed when comparing product models.

The command runs each JSON persona in `personas/` through Pydantic Evals and
saves a JSON result in `results/`. Each result includes the persona, model
names, prompt hashes, interview transcript, stored facts, bullets, check
results, judge explanations and scores. The question limit is 30 by default;
`--max-questions` changes it. Runs with different limits, personas, prompts
or judge models are not directly comparable.

Recall counts unique persona facts with a match. Precision and attribution
count stored facts; inflation and grounding count bullets. An empty
denominator is reported as `null`, not as a perfect score. Inflation counts
each failed bullet once; the breakdown counts every type the judge assigned.
The evaluation exits with an error if a case fails or the judge returns
missing, duplicate or unknown references.
