# Kaggle Data Agent (project scaffold)

**Purpose:** An "agentic" tool that, given a topic (e.g. "Heart Disease"), searches Kaggle for relevant datasets,
ranks them, and downloads the selected dataset (CSVs / files). This scaffold also includes stubs to call
Google Gemini (you provided an API key) for dataset-quality evaluation / summarization.

## What I created for you
- `agents/data_agent.py` — main agent that searches Kaggle and downloads the best dataset.
- `agents/column_selector.py` — helper to inspect dataset files and propose useful columns (simple heuristics).
- `ai/gemini_client.py` — placeholder for calls to Google Gemini; use your Google AI Studio / Gemini endpoint here.
- `run_agent.py` — CLI to run the agent: `python run_agent.py --topic "Heart Disease" --download`.
- `.env.example` — shows environment variables (KAGGLE_USERNAME, KAGGLE_KEY, GEMINI_API_KEY).
- `requirements.txt` — Python packages required.
- `Dockerfile` — simple Dockerfile to containerize the agent.
- `LICENSE` — MIT license.

## Quick setup (local, VSCode)
1. Install Python 3.9+ and pip.
2. Create a virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (windows: .\.venv\Scripts\activate)
   pip install -r requirements.txt
   ```
3. Place your `kaggle.json` file in this directory (or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`).
   If you have a kaggle.json, copy it like:
   ```bash
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Create a `.env` file or export `GEMINI_API_KEY` with your Google API key.
5. Run:
   ```bash
   python run_agent.py --topic "Heart Disease" --download
   ```

## Notes & Caveats
- The Kaggle client used is the official `kaggle` Python package. It requires your `kaggle.json` or
  `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars to be configured.
- Google Gemini usage in `ai/gemini_client.py` is a placeholder. Replace the endpoint and request
  format with the one matching your Google AI Studio / Gemini API.
- The code contains fallback logic (uses Kaggle CLI via subprocess if the Python client fails).

## Project workflow (high level)
1. User provides topic (CLI, UI, or via orchestrator).
2. Data Agent searches Kaggle for datasets (via Kaggle API).
3. Agent ranks datasets (downloads count, file types, relevance in title/description).
4. Agent downloads the selected dataset (zip or files).
5. Column selector agent inspects CSVs and returns useful columns and a score.
6. (Optional) Gemini client evaluates dataset quality or summarizes content and suggests best dataset.
7. Dashboard/visualizer agent (not included) would show results and allow selection.

## Where to put secrets
- `~/.kaggle/kaggle.json`  -> recommended for Kaggle CLI/API.
- `.env` or environment variables -> `GEMINI_API_KEY`.

Enjoy — the project scaffold is packaged with this repo zip.
