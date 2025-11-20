import os
import json
import tempfile
import subprocess
from pathlib import Path
from kaggle import KaggleApi
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class DataAgent:
    def __init__(self):
        # Attempt to authenticate using ~/.kaggle/kaggle.json or env vars
        self.api = KaggleApi()
        try:
            self.api.authenticate()
        except Exception as e:
            print('Kaggle API auth failed:', e)
            print('Make sure ~/.kaggle/kaggle.json exists or KAGGLE_USERNAME/KAGGLE_KEY are set.')
            raise

    def search(self, topic: str, limit: int = 10):
        """Search Kaggle for datasets related to `topic`. Returns a list of candidate dicts with metadata and score."""
        print(f'Searching Kaggle for topic: {topic}')
        # Try several possible KaggleApi method names (different kaggle package versions use different names)
        results = None
        for fn_name in ("datasets_list", "dataset_list", "datasets", "dataset_list_search"):
            api_fn = getattr(self.api, fn_name, None)
            if callable(api_fn):
                try:
                    # try common signatures
                    try:
                        results = api_fn(search=topic, page=1, max_results=limit)
                    except TypeError:
                        results = api_fn(search=topic, page=1)
                    break
                except Exception:
                    # if this method exists but errors for some other reason, continue to try others
                    results = None
                    continue

        if not results:
            # fallback: try the CLI via subprocess and parse its output minimally (best-effort)
            print("Kaggle API method not found or failed — falling back to kaggle CLI for search.")
            cmd = ["kaggle", "datasets", "list", "-s", topic, "-p", str(limit)]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                # CLI prints a table — we won't parse it fully; return empty (caller can still attempt manual CLI download)
                print(out)
                return []
            except Exception as e:
                raise RuntimeError("Kaggle search failed (API + CLI). Error: " + str(e))

        candidates = []
        for r in results:
            # `r` may be a dict-like object. Convert to dict safely.
            try:
                meta = {
                    'ref': getattr(r, 'ref', None) or (r.get('ref') if isinstance(r, dict) else None),
                    'title': getattr(r, 'title', None) or (r.get('title') if isinstance(r, dict) else None),
                    'subtitle': getattr(r, 'subtitle', None) or (r.get('subtitle') if isinstance(r, dict) else None),
                    'url': getattr(r, 'url', None) or (r.get('url') if isinstance(r, dict) else None),
                    'downloadCount': getattr(r, 'downloadCount', 0) or (r.get('downloadCount', 0) if isinstance(r, dict) else 0),
                    'lastUpdated': getattr(r, 'lastUpdated', None) or (r.get('lastUpdated') if isinstance(r, dict) else None),
                }
            except Exception:
                try:
                    meta = dict(r)
                except Exception:
                    meta = {'ref': str(r), 'title': str(r)}

            # Basic scoring: prioritize downloadCount, then title match
            score = (meta.get('downloadCount') or 0) * 1.0
            if topic.lower() in (meta.get('title') or '').lower():
                score += 1000
            meta['score'] = score
            candidates.append(meta)

        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        return candidates

    def download_dataset(self, dataset_meta: dict, outdir: str = 'downloads'):
        """Download chosen dataset. Uses KaggleApi.dataset_download_files or CLI fallback."""
        Path(outdir).mkdir(parents=True, exist_ok=True)
        ref = dataset_meta.get('ref') or dataset_meta.get('url')
        if not ref:
            raise ValueError('No reference found for dataset')
        # ref may look like 'username/dataset-slug'
        print('Downloading dataset', ref, 'to', outdir)
        try:
            # KaggleApi expects dataset = 'username/dataset'
            # Some kaggle versions use dataset_download_files, others may have dataset_download_cli method names;
            # We try the typical method and fall back to CLI if needed.
            download_fn = getattr(self.api, "dataset_download_files", None) or getattr(self.api, "dataset_download_cli", None)
            if callable(download_fn):
                # call with commonly accepted signature
                try:
                    download_fn(ref, path=outdir, unzip=True, quiet=False)
                except TypeError:
                    # signature variant
                    self.api.dataset_download_files(ref, path=outdir, unzip=True)
                return os.path.abspath(outdir)
            else:
                raise Exception("No suitable Kaggle API download method found")
        except Exception as e:
            print('KaggleApi download failed:', e)
            # Fallback to Kaggle CLI if installed
            try:
                cmd = ['kaggle', 'datasets', 'download', '-d', ref, '-p', outdir, '--unzip']
                subprocess.check_call(cmd)
                return os.path.abspath(outdir)
            except Exception as e2:
                raise RuntimeError('Download failed: ' + str(e2))
