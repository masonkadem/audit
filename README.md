# wav2sleep time-of-night shortcut audit (MESA)

Pre-registered audit of whether [wav2sleep](https://github.com/joncarter1/wav2sleep)
(`hf://joncarter/wav2sleep`) uses position in the night as a shortcut.
See [`prereg.md`](prereg.md) for the design and [`report.md`](report.md) for results.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate        # Windows; use bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/joncarter1/wav2sleep.git lightgbm scipy scikit-learn matplotlib pytest
```

## Data (NSRR access required; nothing is redistributed here)

```bash
nsrr download mesa/polysomnography            # EDFs + annotations-events-nsrr XML
python wav2sleep/preprocessing/1_ingest.py --folder /path/to/mesa --output-folder $WAV2SLEEP_DATA
python wav2sleep/preprocessing/2a_create_splits_paper.py --folder $WAV2SLEEP_DATA/mesa
```

## Run order

```bash
python -m audit.rf                                             # step 0: receptive field
python -m audit.rf --data $WAV2SLEEP_DATA/mesa/test            # R, on unaltered nights only
python -m audit.analysis fit-prior --source ingest --data $WAV2SLEEP_DATA/mesa/train
python -m audit.run_model --data $WAV2SLEEP_DATA/mesa/test     # steps 3 and 5
python -m audit.controls --features F.parquet --labels L.parquet   # step 6
python -m audit.analysis slopes --runs results/w2s --models ppg ecg all --primary-dist <R> --out results/w2s_slopes
python -m audit.analysis slopes --runs results/controls --models lgbm_notime lgbm_time \
    --min-dist 5 --primary-dist 5 --out results/controls_slopes
python -m audit.report
pytest -q
```

`results/` stays local: it holds per-participant outputs.
