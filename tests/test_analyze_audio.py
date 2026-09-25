#quick manual test for the /analyze-audio endpoint
import requests

URL = "http://127.0.0.1:5000/analyze-audio"
AUDIO_FILE_PATH = "audio/Pekinwoof.mp3"  

with open(AUDIO_FILE_PATH, "rb") as f:
    files = {"audio": f}
    response = requests.post(URL, files=files)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")