# Matched-scale Direct LLM results

This directory records the aggregate results of the Direct LLM rerun used for
the revised Table 5. Wilson and University use their first 50,000 physical log
lines in original order; HS uses all 36,449 available lines. Each Direct LLM
request processes 40 lines, as described in the manuscript.

`summary.csv` preserves the unrounded F1, FEA, and total token counts supplied
for the rerun. Table 5 displays F1 and FEA to two decimal places. Elapsed time
was not recorded and is therefore left blank rather than estimated.

For the University dataset, 5,920,450 total tokens over 50,000 lines correspond
to 118,409 tokens per 1,000 lines. A 40-line request batch corresponds to 25
LLM calls per 1,000 lines. These normalized values are the Direct LLM cost
reference used in Fig. 7.
