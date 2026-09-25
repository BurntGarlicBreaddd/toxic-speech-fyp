#quick manual test for the /analyze endpoint (toxicity + sentiment)
import requests

URL = "http://127.0.0.1:5000/analyze"

samples = [
    "You are so bad at this game, uninstall.",
    "Nice shot, well played!",
    "gg ez noob",                       # gaming slang - VADER likely scores near 0
    "wow great job, really impressive",  # could read as sincere OR sarcastic
]

for text in samples:
    response = requests.post(URL, json={"text": text})
    print(f"Input: {text!r}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")