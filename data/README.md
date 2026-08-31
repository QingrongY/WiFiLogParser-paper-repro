# Datasets (Not Included)

This repository **does not** ship the raw datasets or ground-truth files.

Reason: dataset distribution/licensing and repository size limits.

## Expected layout

Place the datasets in the following paths so the default config works:

```
data/
  raw/
    Wilson/
      Wilson_50000.log
      Wilson_full.log
    University/
      University_2000.log
      University_50000.log
    HS/
      HS_full.log
  ground_truth/
    Wilson/
      Wilson_gt.csv
    University/
      University_gt.csv
    HS/
      HS_gt.csv
```

The main-experiment config uses Wilson/University 50k and HS full. The matched
template-baseline config uses the first 2,000 physical lines of `Wilson_full.log`
and `HS_full.log`, plus the exact prefix file `University_2000.log`.

If you use different filenames/paths, update `configs/main_experiment.json`.

## Run

After placing the files:

```bash
./scripts/run_main.sh
```
