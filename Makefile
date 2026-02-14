.PHONY: venv install run

venv:
	python3 -m venv .venv

install: venv
	.venv/bin/python -m pip install -r requirements.txt

run:
	.venv/bin/python scripts/run_main_experiments.py --config configs/main_experiment.json
