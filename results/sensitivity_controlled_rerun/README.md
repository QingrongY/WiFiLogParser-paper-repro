# Controlled Table 8 sensitivity rerun

This directory records the controlled sensitivity experiments run on
2026-09-03 UTC for the IEEE Access revision. Every configuration used the first
50,000 physical lines of the University dataset, the current released
WiFiLogParser code, temperature 0, a maximum completion length of 8,192 tokens,
and a fresh parser cache. Each configuration was run once in a new process.
Only the parameter named by a row was changed from the default configuration.

The default configuration was `k=8`, chunk size 50,000, reasoning off, and
`gemini-3.1-flash-lite`. The other runs changed one of the following:

- diversity sample size: 2, 4, 16, or 32;
- chunk size: 5,000, 10,000, or 100,000;
- reasoning: medium;
- model: `gpt-4o-mini-2024-07-18` or `gpt-4o-2024-08-06`.

`summary.csv` contains the configuration, run ID, UTC start/end time, full
model ID, all controlled settings, F1, FEA, LLM calls, total LLM tokens,
runtime, input size, input hashes, and the local evidence directory. Detailed
records, parser outputs, and raw logs remain under ignored `outputs/` paths
because they can contain identifiers and are not distributed. The aggregate
results here contain no raw log lines or identifiers.

The default run produced F1 0.894345, FEA 1.000000, 82 LLM calls, 154,075
tokens, and a runtime of 91.392 s. This is the canonical University 50,000-line
WiFiLogParser result used in Tables 5 and 8 and at N=50,000 in Fig. 7.

Run the complete suite with:

```bash
.venv/bin/python scripts/run_sensitivity_controlled.py
```

If a provider interruption occurs, completed rows can be retained with:

```bash
.venv/bin/python scripts/run_sensitivity_controlled.py --resume
```
