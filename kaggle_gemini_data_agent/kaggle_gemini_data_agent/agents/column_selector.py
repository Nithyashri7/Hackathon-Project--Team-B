# Simple heuristic-based column selector that reads CSV files in a folder and suggests candidate columns.
import os, pandas as pd
from pathlib import Path

def suggest_columns_from_dir(dirpath, top_n=5):
    files = list(Path(dirpath).glob('*.csv'))
    suggestions = {}
    for f in files:
        try:
            df = pd.read_csv(f, nrows=2000)
        except Exception:
            continue
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique = df[col].nunique(dropna=True)
            # heuristic: columns with moderate uniqueness and numeric types are often useful
            score = 0
            if 'int' in dtype or 'float' in dtype:
                score += 2
            if 2 < unique < min(1000, len(df)):
                score += 1
            suggestions.setdefault(f.name, []).append((col, score))
    # collapse and sort
    out = {}
    for k,v in suggestions.items():
        out[k] = sorted(v, key=lambda x: x[1], reverse=True)[:top_n]
    return out
