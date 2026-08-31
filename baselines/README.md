# Template-baseline reproduction

This directory defines the reproducible LILAC+ and LogBatcher+ pipelines used
for the WiFiLogParser revision. It does not vendor either upstream parser.
`upstream.json` pins the exact public source revisions, and
`scripts/prepare_template_baselines.sh` checks them out under the ignored
`.baseline-work/` directory.

Both upstream methods are left responsible for their original task: assigning
a generic log template to every input line. The `+` layer is shared by both
methods and consists of two template-level LLM calls:

1. classify each template as `connect`, `disconnect`, or `other`;
2. for an event template, generate one anchored Python regular expression with
   the named groups `ap_id` and `client_id`.

The prompts are versioned in `prompts/`. Representative raw lines are supplied
only to interpret placeholders already produced by the baseline. No labels or
ground-truth values are included in either prompt. The learned label and regex
are propagated to every line assigned to that template.

The default experiment uses the first 2,000 physical lines of each dataset.
This deterministic prefix rule matches the historical subset-construction
script and permits a genuinely matched comparison with Direct LLM and
WiFiLogParser. Raw datasets, credentials, upstream checkouts, detailed logs,
and line-level outputs remain local and are not committed.

Run:

```bash
./scripts/prepare_template_baselines.sh
.venv/bin/python scripts/run_template_baselines.py \
  --config configs/template_baselines_2000.json
```

The runner records input hashes, upstream commits, model settings, exact LLM
requests/responses, normalized per-line predictions, and both literal
paper-definition and legacy implementation metrics under `outputs/`.

Important metric note: reporting policy is intentionally not hidden in the
runner. It emits three-class macro F1, the historical event-set F1, strict FEA
as defined in the manuscript, and a diagnostic line-index FEA that includes
event accuracy and partial matching. The latter is not presented as the HS
repository's special record-matching metric. Which columns appear in the
revised manuscript must be decided explicitly rather than inferred from an old
table.
