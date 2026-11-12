# Design decisions

Each entry states the decision, the reason, an example, and the cost or the rejected
alternative. The examples use synthetic data.

## 1. The subject is faithfulness, not the CV

- **Decision:** The project measures whether the agent states only what the person said.
  The CV is the test case.
- **Why:** Faithfulness has a ground truth: the facts that the person gave. A claim is
  supported by those facts or it is not. "A good CV" has no ground truth, so two
  reviewers can disagree and neither is wrong.
- **Example:** The person says "I wrote the data export." The agent writes "Designed
  and built the reporting platform." The CV reads better, but it is unfaithful. This
  project scores that bullet as a failure.
- **Rejected:** Career advice. It has no correct answer, so no metric can score it.

## 2. Code enforces only what code can check

- **Decision:** `check_bullets()` enforces two rules without an LLM: every cited fact ID
  exists, and every cited fact is `confirmed`. The evaluation measures the other two
  rules (team work as personal work, guessing on vague answers) with a judge model.
- **Why:** A deterministic check cannot be wrong about what it checks. A judge model can
  be wrong, so its results are metrics, not gates.
- **Example:**
  - A bullet cites `F7`, and `F7` has status `to_confirm`. The code rejects the bullet.
    No model is involved, so the result is certain.
  - A bullet says "I migrated the database" and cites `F3`, where the actor is `team`.
    The IDs are valid, so the code passes it. Only a model can see that "I" conflicts
    with `team`. That result is an estimate.
- **Cost:** Two rules depend on a judge model that can be wrong. The README must report
  this.
- **Rejected:** One LLM judge for all rules. That hides which failures are certain and
  which are estimates.

## 3. Stored facts and persona facts have separate IDs

- **Decision:** The interviewer agent creates new IDs. A judge model maps stored facts to
  persona facts by meaning.
- **Why:** The interviewer must not see the persona file: the persona file is the answer
  key. So the interviewer cannot reuse persona IDs, and the evaluation must match facts
  by content.
- **Example:** The persona has `P2: "The platform team moved the servers to the
  cloud."` The interviewer stores `F5: "Moved the servers to the cloud"`. The judge
  reads both sentences, decides that they describe the same event, and records the
  match `F5 → P2`. The metrics use this match (see entry 6).
- **Cost:** Recall, precision and attribution depend on the judge's matching.

## 4. Personas contain traps

- **Decision:** At least one persona has a feature that was planned but never built,
  another team's report, and a vague answer about who did the work.
- **Why:** These are the known ways that LLMs inflate a CV. A suite without traps
  measures only the easy cases, where any model scores well.
- **Example:** The persona file says: "The mobile app was planned for Q3 but was
  cancelled." A faithful agent stores this as `do_not_claim`, or does not store it. An
  unfaithful agent writes "Delivered the mobile app." The Inflation metric counts that
  bullet.

## 5. The evaluation suite is the fixed point

- **Decision:** The same suite runs on every model and every prompt change. Results go
  to `results/` and are compared across runs.
- **Why:** A change is an improvement only when the metrics show it. A prompt change
  that makes bullets read better can also raise Inflation. Without a fixed suite, that
  trade-off is not visible. The same suite will also compare the PydanticAI version with
  a later LangGraph version.
- **Cost:** Each run calls the models many times. A small suite (3 to 5 personas) keeps
  the cost low, but small differences between runs can be noise.

## 6. Attribution is a separate metric

- **Decision:** The evaluation scores a stored fact on two separate questions:
  - **Precision:** does the stored fact match any persona fact? If not, the interviewer
    invented it.
  - **Attribution:** for a stored fact that matches, is the actor (`me`, `team`,
    `colleague`) the same as in the persona fact?
- **Why:** A wrong actor and an invented fact are different failures. A wrong actor is
  the failure that this project is about: team work turned into personal work. One
  precision number would mix the two failures, and a reader could not tell which one
  happened.
- **Example:** The persona has `P2: "The platform team moved the servers to the
  cloud."`, actor `team`. The interviewer stores `F5: "Moved the servers to the
  cloud"`, actor `me`.
  - Recall: `P2` has a match, so it counts as collected.
  - Precision: `F5` has a match, so it is not invented.
  - Attribution: the actors differ, so `F5` fails.
- **Rejected:**
  - Wrong actor counts against precision: the number then mixes two failures.
  - Ignore the actor in the fact metrics: an actor error is then visible only when it
    reaches a bullet.

## 7. Privacy: an instruction is not enforcement

- **Decision:** Real career data stays outside the repository and outside the coding
  assistant's reach. Three controls apply:
  - Permission deny rules block the assistant's file tools.
  - A shell hook blocks shell commands that name the private paths.
  - A pre-commit hook blocks a commit that contains a name from a private list.
- **Why:** An instruction in a file such as `CLAUDE.md` is a request to the model, not a
  control. Each control covers a gap that the one before it leaves open:
  - Deny rules cover only the file tools. A shell command such as `cat` can still read
    the file, so a hook checks shell commands.
  - Neither control checks the content of a commit. The pre-commit hook does.
  - Any instruction file that the assistant loads is sent to the model provider. So
    such a file can hold paths and rules, but never the private facts.
- **Example:** The pre-commit hook prints only the file names, never the matched name.
  If the assistant runs the commit, it sees which file to fix but not the private name.
- **Cost:** The hooks block some harmless commands, for example a commit message that
  contains a blocked word.
- **Rejected:** Rules in `CLAUDE.md` only.

## 8. The writer sees only confirmed facts

- **Decision:** `write_bullets()` sends only the facts with status `confirmed` to the
  writer agent. `check_bullets()` still checks the bullets against all stored facts.
- **Why:** A fact that the writer does not see cannot leak into a bullet. A prompt rule
  such as "cite only confirmed facts" is a request to the model. The filter is code.
- **Example:** The store holds `F1` (confirmed) and `F2` (do_not_claim, a feature that
  was planned but not built). The writer receives only `F1`. If a bullet cites `F2`,
  the writer guessed the ID, and `check_bullets()` rejects the bullet with
  `not_confirmed`.
- **Cost:** The evaluation no longer tests whether the writer obeys a status. A planned
  feature can still reach the CV in one way: the interviewer stores it as `confirmed`.
  Only the Inflation metric can see this, and only when the judge compares the bullet
  with the persona fact. Day 2 decides which fact the judge uses.
- **Rejected:** Send all facts with their status, and rely on the prompt and the check.

## 9. Code ends the interview when the answers stop adding facts

- **Decision:** `run_interview()` ends when `max_answers_without_facts` answers in a
  row add no fact. The agent can still end earlier with Done. `max_questions` is the
  hard limit. The result records which condition ended the interview.
- **Why:** An end rule that is only in the prompt ("end when the person says I don't
  know") leaves the count to the model. The model can stop too early, which lowers
  Recall, or never stop, which uses all the questions. The store size is a fact
  that code can count.
- **Example:** With a limit of 2, the answers "I don't know", "I built the export
  job", "I don't know", "I don't know" end the interview after the fourth answer.
  The second answer adds a fact and restarts the count.
- **Cost:** An answer that the agent stores as no fact counts as empty, even when it
  carries context, for example the list of employers.
- **Rejected:** Match "I don't know" in the answer text. Human answers vary too much.

## 10. Prompts are Markdown files, with a test that keeps them in sync

- **Decision:** The agent instructions are in `src/cv_agent/prompts/*.md`.
  `tests/test_prompts.py` checks that the prompts name every `Status` and `Actor`
  value, the `save_fact` tool and the `fact_ids` field.
- **Why:** A prompt change is an experiment. The evaluation runs again after each
  one, so a prompt change must be its own diff. Later, the results can record which
  prompt file produced them.
- **Example:** Rename `Status.TO_CONFIRM` to `"unconfirmed"` in code only. The model
  still reads "to_confirm" in the prompt and stores a value that fails validation.
  The sync test fails first.
- **Cost:** The test checks that each name appears, not that its definition is
  right. A deleted definition passes when the name appears elsewhere in the prompt.
- **Rejected:** Prompts as Python strings next to the agent: the code identifiers
  are close, but a prompt change mixes with code changes in one file.
