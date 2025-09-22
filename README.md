# F1 ML Project

Reproducible pipeline for practice/qualifying features → position prediction.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put raw data
# data/raw/results.csv, data/raw/races.csv

python scripts/train.py --config configs/train.yaml
```
