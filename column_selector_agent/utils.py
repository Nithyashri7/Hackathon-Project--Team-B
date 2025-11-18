import pandas as pd
import numpy as np
from pathlib import Path
import json

def read_table(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except Exception:
            # try line-delimited JSON
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            records = [json.loads(l) for l in lines if l.strip()]
            return pd.DataFrame.from_records(records)
    raise ValueError(f"Unsupported file type: {suffix}")