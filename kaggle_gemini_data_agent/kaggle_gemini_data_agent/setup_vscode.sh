# Optional helper: sets up a python venv and installs requirements
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
echo "Done. Activate the venv with: source .venv/bin/activate"
