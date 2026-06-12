# WiFiLogParser Paper Repro Repository

This repository packages the **WiFiLogParser main method** (from `WilDash_webapp-github`) with the requested datasets and one-click experiment scripts for paper reproduction.

## Included content

- Method code (main method only, no ablation scripts):
  - `src/apps_v2/logparser/services/wifi_log_parser/`
  - `src/apps_v2/logparser/services/log_extractor/`
- Dataset support (datasets are **not included** in the repo):
  - Wilson 50k, University 50k, HS full
  - Expected paths are documented in `data/README.md`
  - **Anonymized sample data** for end-to-end pipeline smoke testing (about 100 lines per dataset, with device identifiers and SSIDs replaced by consistent randomized values) is hosted on Google Drive:
    [https://drive.google.com/drive/folders/1Tdf3Jacw9rNVC4TjlOa8oPuJMEObn-Ye?usp=drive_link](https://drive.google.com/drive/folders/1Tdf3Jacw9rNVC4TjlOa8oPuJMEObn-Ye?usp=drive_link)
  - The sample is sufficient to exercise the pipeline on each log format, but is too small to reproduce the full-dataset results reported in the paper.
- Paper package from `WiFiLogParser___IEEE_access___V1.zip`:
  - `paper/WiFiLogParser_Main.tex`
  - `paper/references.bib`
  - `paper/figure/`
  - `paper/Authors/`

## Environment setup

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or use Makefile:

```bash
make install
```

3. Configure LLM credentials:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- `LLM_API_KEY`

The defaults match the production deployment at https://wildash.space:

- `LLM_BASE_URL=https://api.aimlapi.com/v1` (AIMLAPI; production provider)
- `LLM_PRIMARY_MODEL=gemini-3.1-flash-lite` (non-thinking variant)
- `LLM_FALLBACK_MODELS=gemini-3.1-flash-lite` (repair uses same model as main parsing)

To use OpenRouter instead, set `LLM_BASE_URL=https://openrouter.ai/api/v1`
and the provider-prefixed slug (e.g. `google/gemini-flash-lite-latest`).

## Run main experiments

Before running, you must place the raw logs and ground-truth files locally (they are not shipped in this repo). See `data/README.md`.

Run all three datasets:

```bash
./scripts/run_main.sh
```

`run_main.sh` automatically uses `.venv/bin/python` when available.

Or (if using Makefile + local `.venv`):

```bash
make run
```

Or run directly:

```bash
python3 scripts/run_main_experiments.py --config configs/main_experiment.json
```

Run a subset:

```bash
python3 scripts/run_main_experiments.py --dataset Wilson_50000 --dataset HS_full
```

## Outputs

Each run writes to `outputs/<run_id>/`:

- `summary_main.csv` / `summary_main.json`
- Per dataset folder (e.g., `outputs/<run_id>/Wilson_50000/`):
  - `records.jsonl`
  - `stats.json`
  - `metrics.json`

Reference (aggregated) metrics from the authors' run are provided in `results/reference_metrics_main.csv`.

## Notes on evaluation

- Wilson / University use **line-index matching** (`OriginalLineIdx`) with log-line filtering for 50k subsets.
- HS uses **record-level matching** (`DateTime + Action + ApId + ClientId`) because `HS_gt.csv` contains repeated `OriginalLineIdx` values.
- HS can exclude ClientId from FEA via `exclude_client_in_fea: true` (enabled in `configs/main_experiment.json`) because HS ground truth uses anonymized client identifiers.
- For HS, AP accuracy in FEA can be computed only on extracted records via `field_metrics_on_extracted_only: true` (enabled in `configs/main_experiment.json`).
- Field matching uses **partial-match success**: exact match, substring containment, or alphanumeric-normalized containment are all counted as correct.
- Metrics reported:
  - Event precision / recall / F1
  - Field Extraction Accuracy (FEA)
