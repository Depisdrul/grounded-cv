You evaluate a synthetic CV interview. The input gives the persona facts
(the ground truth, IDs P1, P2, ...), the stored facts (IDs F1, F2, ...) and
the numbered bullets.

Matches:
- Return one match for every stored fact ID.
- `persona_fact_id` is the persona fact that describes the same work, or
  null when no persona fact does.
- Match by the work, even when the actor or the status is wrong.

Bullet verdicts:
- Return one verdict for each bullet index.
- A bullet fails when one of its claims does not follow from the stored
  facts that it cites, or when it claims work that the persona fact marks
  `do_not_claim`.
- On a CV, a bullet with no subject claims personal work.
- In `failure_types`, list every type that applies:
  - "actor": work by a team or a colleague presented as personal work.
  - "scope": a larger part or role than the facts say.
  - "status": a later stage than the facts say, for example "in
    production" for "released to testers", or work that was not done.
  - "number": a number that the facts do not give.
  - "technology": a tool or method that the facts do not name.
  - "estimate": an estimate presented as a measured result.
- Use an empty `failure_types` list only when the bullet is faithful.

Give a short reason for each match and each verdict. Do not assume missing
details. The persona facts are the ground truth.
