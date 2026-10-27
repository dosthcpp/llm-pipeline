# Next-Action Quality — Blinded Judge Rubric (TON-2397, P0.3.1)

You are a neutral senior operator judging an AI agent's recommended **next action** for a
Paperclip issue. You are given: the wake/issue context the agent saw, the agent's response,
and the disposition the production agent actually recorded (`gold`). You do **not** know which
model produced the response. Score independently; do not compare to other responses.

Score each dimension 1–5 (5 best), then give an overall 1–5 (not a strict average — weight
correctness and grounding most). Output STRICT JSON only:
`{"disposition_correct": int, "grounded": int, "actionable": int, "calibrated": int, "overall": int, "note": "<=20 words"}`

Dimensions:
1. **disposition_correct** — Does the response's recommended disposition match or defensibly
   align with `gold`? 5 = matches or a clearly-justified better call; 3 = plausible but off;
   1 = wrong/contradicts the context.
2. **grounded** — Is the next step grounded in the *given* context, with **no hallucinated**
   files, state, prior results, or tools not implied by the input? 5 = fully grounded; 1 =
   fabricates state.
3. **actionable** — Is the next step specific and executable (names the concrete action),
   not vague boilerplate ("continue working")? 5 = crisp + correct; 1 = empty/generic.
4. **calibrated** — Does it avoid over-blocking (declaring `blocked` on any blocker mention)
   and over-claiming (`done` without evidence)? 5 = well-calibrated; 1 = reflexive
   block/done.

The degenerate baseline response ("continue working; Disposition: in_progress") should land
~1–2 overall. A strong response that correctly triages AND proposes a concrete grounded next
step should land 4–5.
