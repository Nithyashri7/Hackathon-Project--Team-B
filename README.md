# Column Selector Agent

**Purpose:** Analyze one or more dataset files (CSV, Excel, JSON) and automatically select the most useful columns to feed into a synthetic data generator agent. The agent produces a JSON list of selected columns per file and a merged CSV containing only selected columns (for downstream synthetic data generation).

## What you get in this package
- `column_selector_agent.py` — main CLI script (Python).
- `utils.py` — helper functions for file reading and basic encodings.
- `config_example.json` — example configuration for thresholds and options.
- `requirements.txt` — Python packages required.
- `sample_input_instructions.txt` — how to provide inputs to the agent.
- `README.md` — this file.
- `LICENSE.txt` — MIT license.

## Quick setup (Linux / Mac / Windows WSL / Windows with Python 3.8+)
1. Create a virtual environment and activate it:
```bash
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\\Scripts\\activate
```
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Run the agent on a directory or single file:
```bash
python column_selector_agent.py --input /path/to/downloaded/datasets --output ./output --config config_example.json
```
4. Output files will be in `./output`:
- `selected_columns.json` — columns chosen per file plus combined list.
- `selected_combined.csv` — merged CSV of selected columns from all files (where possible).
- `report.txt` — short human-readable report explaining the choices and metrics per file.

## Notes
- Supports CSV, .xls/.xlsx, and JSON files (list of records or table-like JSON).
- If your data agent has downloaded multiple files into a folder, point `--input` to that folder.
- Tune thresholds in `config_example.json` to change behavior (e.g., allow more columns, stricter missingness).