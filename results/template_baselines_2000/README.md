# Matched 2,000-line LILAC+ and LogBatcher+ results

These are the aggregate results of the reproducible template-baseline rerun for
the IEEE Access revision. All datasets use the deterministic first 2,000
physical lines. Both methods use their pinned official implementation for
generic template induction and the same versioned two-stage semantic
post-processing in `baselines/prompts/`.

`summary.csv` intentionally reports two event metrics and two field diagnostics:

- `macro_f1_3class` is the definition written in the submitted manuscript:
  one-vs-rest F1 averaged over connect, disconnect, and other.
- `event_set_f1` is the binary event-set metric implemented by the existing
  main-experiment evaluator. It is retained to expose, not hide, the mismatch.
- `strict_fea` is the submitted manuscript definition: exact normalized AP and
  client accuracy over every ground-truth event line, averaged over the two
  fields. Event-label accuracy is not part of this value.
- `line_index_style_fea` is a diagnostic that adds event accuracy and permits
  partial identifier matching, matching the line-index evaluator's structure.
  It is not the special HS record-matching configuration.

No number in this directory has yet been approved for the manuscript. The
metric definition and the matched WiFiLogParser/Direct LLM rows must be resolved
first.

Detailed outputs stay under ignored `outputs/` because they contain raw log
content or identifiers and the datasets are not redistributable. The evidence
manifest records their paths and hashes for local verification. A future public
release should include only artifacts permitted by the data-use terms.

The first LILAC+/Wilson pilot used an event definition that incorrectly treated
connection requests as `other`; that pilot is excluded. The final runs align
the event semantics with WiFiLogParser and the historical annotation rules.
