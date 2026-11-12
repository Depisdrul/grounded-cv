# cv-agent

An agent that interviews a person about their work and writes CV bullets that cite only
what the person said. The project measures one thing: **does the agent stay faithful to
its source?**

## Results

_To fill after the first evaluation run._

| Model | Recall | Precision | Inflation | Grounding |
|---|---|---|---|---|
| | | | | |

## How it works

1. The interviewer agent asks questions and stores each answer as a typed fact.
2. The writer agent writes CV bullets. Each bullet lists the IDs of the facts it uses.
3. A deterministic check rejects bullets with missing or unknown fact IDs, or with facts
   that are not confirmed.

## How it is evaluated

A second model plays the interviewee, from a persona file with known facts. Some personas
contain traps: a feature that was planned but never built, or another team's work.

| Metric | Meaning |
|---|---|
| Recall | Share of the persona's facts that the interviewer collected |
| Precision | Share of the collected facts that are true for the persona |
| Inflation | Share of bullets that claim work that the persona did not do |
| Grounding | Share of bullets that pass the fact-ID check |

## Limits

- The personas are synthetic and few.
- Recall, precision and inflation use a judge model, which can be wrong.
- _Add the limits found during the runs._

## Run

_To fill: setup, model configuration, tracing, evaluation command._
