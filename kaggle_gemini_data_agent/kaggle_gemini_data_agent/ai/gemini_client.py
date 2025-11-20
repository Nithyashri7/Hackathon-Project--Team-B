# Placeholder Gemini client. Replace endpoint, payload and auth as required by your Google AI Studio / Gemini setup.
import os
import requests
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def summarize_text(text: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError('GEMINI_API_KEY not set in environment')
    # Example placeholder: send `text` to Gemini and return summary.
    # Replace with actual endpoint and request format.
    endpoint = 'https://api.example.com/v1/generate'  # TODO: replace with real endpoint
    payload = {'prompt': 'Summarize:\n' + text, 'max_tokens': 256}
    headers = {'Authorization': f'Bearer {GEMINI_API_KEY}'}
    r = requests.post(endpoint, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get('summary') or r.text
