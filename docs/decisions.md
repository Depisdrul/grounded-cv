# Design decisions

Each entry states the decision, the reason, and the alternative that was rejected.

## 1. The subject is faithfulness, not the CV

- **Decision:** The project measures whether the agent states only what the person said.
  The CV is the test case.
- **Why:** Faithfulness has a ground truth, so it can be measured. "A good CV" has no
  ground truth.
- **Rejected:** Career advice. It has no correct answer, so no metric can score it.

## 2. Code enforces only what code can check

- **Decision:** `check_bullets()` enforces two rules without an LLM: every cited fact ID
  exists, and every cited fact is `confirmed`. The evaluation measures the other two
  rules (team work as personal work, guessing on vague answers) with a judge model.
- **Why:** A deterministic check cannot be wrong about what it checks. A judge model can
  be wrong, so its results are metrics, not gates.
- **Rejected:** One LLM judge for all rules. That hides which failures are certain and
  which are estimates.

## 3. Stored facts and persona facts have separate IDs

- **Decision:** The interviewer creates new IDs. The judge maps stored facts to persona
  facts.
- **Why:** The interviewer must not see the persona file. Shared IDs would leak the
  answer key into the system under test.

## 4. Personas contain traps

- **Decision:** At least one persona has a feature that was planned but never built,
  another team's report, and a vague answer about who did the work.
- **Why:** These are the known ways that LLMs inflate a CV. A suite without traps
  measures only the easy cases.

## 5. The evaluation suite is the fixed point

- **Decision:** The same suite runs on every model and every prompt change. Results go
  to `results/` and are compared across runs.
- **Why:** A change is an improvement only when the metrics show it. The same suite will
  also compare the PydanticAI version with a later LangGraph version.

## 6. Privacy: an instruction is not enforcement

- **Decision:** Real career data stays outside the repository and outside the coding
  assistant's reach. Permission deny rules block the file tools. A shell hook blocks
  shell commands that name the private paths.
- **Why:** An instruction in a file such as `CLAUDE.md` is a request to the model, not a
  control. Deny rules cover only the file tools: a shell command such as `cat` can still
  read the file. Any instruction file that the assistant loads is sent to the model
  provider, so such a file can hold paths and rules but never the private facts.
- **Rejected:** Rules in `CLAUDE.md` only.
