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
    University/
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

If you use different filenames/paths, update `configs/main_experiment.json`.

## Run

After placing the files:

```bash
./scripts/run_main.sh
```
