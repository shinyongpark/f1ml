from pathlib import Path
import json, yaml

def read_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
