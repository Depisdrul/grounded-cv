# Lessons learned

This project tests one property of an LLM system: it must not state more than
its source says. The lessons below come from building it and from its first
runs with a small local model (Qwen3.5 4B on Ollama, on a laptop CPU). Each
lesson names what happened, what changed, and the general rule. The decision
entries in [decisions.md](decisions.md) give the details.

## 1. A real model finds bugs that scripted tests do not

The unit tests script the model's responses. They pass on every change, and
they still missed three problems that the first real runs found:

| Problem | Found by | Change |
|---|---|---|
| Two `save_fact` calls in one model response ran in parallel threads, and one write failed on the other's temporary file | A run with Qwen | A lock in the fact store (entry 12) |
| The actor values had no clear place for another team's work | A run with Qwen | `colleague` covers another group too (entry 13) |
| The interviewee's thinking filled the model's context before the first answer | A run with Qwen | No thinking for the interviewee (entry 19) |

**Rule:** run a short real interview after each change to the agent loop. It
costs minutes and finds a class of bugs that scripted tests cannot find.

## 2. Silent failures are the dangerous ones

Three settings looked correct and had no effect:

- A commit-message hook searched for private names with `grep -i -F -f`. The
  grep version in Git for Windows aborts on that combination. In a test, the
  hook read the abort as "no match" and let a blocked name through. It now
  blocks on any grep status other than "match" or "no match".
- PydanticAI's provider-neutral `thinking=False` setting does not reach Ollama.
  The model kept thinking: 1092 output tokens for a one-sentence answer. The
  Ollama-specific key gives 41 tokens (entry 19).
- The `OLLAMA_CONTEXT_LENGTH` variable was saved, and Ollama was restarted. The
  context stayed at 4096 tokens. A committed Modelfile now sets the context, and
  the model name records it (entry 20).

A fourth problem hid the progress log. Under `python -m grounded_cv.eval`, the
module's `__name__` is `__main__`, so its logger was outside the `grounded_cv`
logger that the command sets to INFO. The tests import the module by its real
name and did not see it. The logger now has a fixed name.

**Rule:** check a safety or resource setting by its effect (a token count, a
context size, a blocked test commit), not by its presence in the
configuration.

## 3. An evaluation can leak its answer key

The simulated interviewee first saw the full persona file, including labels
such as `"status": "do_not_claim"`. A model that reads such a label can say
the truth in its first answer, and then the trap no longer tests the
interviewer. The interviewee now sees only the facts, without the labels
(entry 17).

**Rule:** treat each model in the evaluation as a reader that uses everything
it sees. Give it only what the role needs.

## 4. Metrics must cover the failures that actually happen

The first metrics checked only that a bullet cites existing, confirmed facts.
The failures that people correct in LLM-written CVs are different: a feature
that was planned but never built, another team's work in the first person, a
release to testers shown as a release to production. The judge now checks
every claim of a bullet against its cited facts, and reports the failure type
(entry 14).

**Rule:** derive the metrics from a list of real failures, not from what is
easy to check.

## 5. Keep the expensive step when a cheap step fails

With a local model, one interview takes much longer than the judge call that
follows it. The first version raised on a judge failure, for example an
exhausted quota, and lost the interview. Now each case is saved with the error,
and `--rejudge` judges saved interviews again without a new interview
(entry 18). In the example run, this saved both interviews: one judge call was
interrupted, the small local judge failed the other, and a later re-judge
scored both.

**Rule:** order the pipeline so that a failure in a later, cheaper step does
not destroy the output of an earlier, expensive step.

## 6. Small local models change the engineering

A small model on a laptop is not a slower copy of a cloud model:

- Thinking is on by default and costs more tokens than the answer.
- The default context is small, and the interview history grows with each
  question.
- On a CPU, one interview takes tens of minutes, so a full evaluation must run
  overnight.

**Rule:** measure tokens and time per call before you plan the evaluation. A
local model is the right choice for private data (a real person's career), but
its limits set the size of the experiment.

## Status and next steps

The results table in the [README](../README.md) shows one example run. It shows
that the pipeline works end to end. It is not a benchmark. The next steps are:

| Step | Purpose |
|---|---|
| Personas 2 to 5: team-heavy, vague talker, writer bait, clean control | Each trap type in at least two personas |
| A strong, fixed judge model | Verdicts that the comparison can trust. `--rejudge` applies it to saved interviews |
| Interviewer thinking on and off | Does thinking improve faithfulness? |
| A second product model | How a small local model differs from a cloud model |
| Three runs per case | A small model gives a different interview on each run |
