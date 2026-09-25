import os
import tempfile
import requests
import whisper
import flagging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv() #load api key from .env

app = Flask(__name__)

PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY")
PERSPECTIVE_URL = (
    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
)

#load models on startup
vader_analyzer = SentimentIntensityAnalyzer()
whisper_model = whisper.load_model("base")

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm"}

#which flagging rule in use
ACTIVE_FLAGGING_RULE = flagging.baseline


def get_toxicity_score(text):
    # send text to perspective api, returns toxicity score (0-1)
    payload = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}},
    }
    response = requests.post(
        PERSPECTIVE_URL,
        params={"key": PERSPECTIVE_API_KEY},
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()
    return result["attributeScores"]["TOXICITY"]["summaryScore"]["value"]


def get_sentiment_score(text):
    #run vader locally, return sentiment score (-1 to +1)
    scores = vader_analyzer.polarity_scores(text)
    return scores["compound"]


def get_transcript(audio_path,  language="en"):
    #transcribe audio with whisper, forced english due to mistranslation, misread my accented english to indonesian/malay
    result = whisper_model.transcribe(audio_path,  language=language)
    return result["text"].strip()


@app.route("/analyze", methods=["POST"])
def analyze():
    #json api route, takes text and returns scores
    if not PERSPECTIVE_API_KEY:
        return jsonify({"error": "PERSPECTIVE_API_KEY is not set on the server"}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Request body must include a non-empty 'text' field"}), 400

    sentiment_score = get_sentiment_score(text)

    try:
        toxicity_score = get_toxicity_score(text)
    except requests.exceptions.RequestException as e:
        #catch network/api errors instead of crashing
        detail = None
        if e.response is not None:
            try:
                detail = e.response.json()
            except ValueError:
                detail = e.response.text
        return jsonify({"error": "Perspective API request failed", "detail": detail}), 502

    return jsonify({
        "text": text,
        "toxicity_score": toxicity_score,
        "sentiment_score": sentiment_score,
    })


@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    #json api route, take audio file and returns transcripts + scores
    if not PERSPECTIVE_API_KEY:
        return jsonify({"error": "PERSPECTIVE_API_KEY is not set on the server"}), 500

    if "audio" not in request.files:
        return jsonify({"error": "Request must include a file under the 'audio' field"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'",
            "allowed": sorted(ALLOWED_AUDIO_EXTENSIONS),
        }), 400

    #save temp file since whisper cant use raw upload data
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = get_transcript(tmp_path)
    finally:
        os.remove(tmp_path) #always deletes the file for privacy issues

    if not transcript:
        return jsonify({"error": "Whisper returned an empty transcript for this audio"}), 422

    sentiment_score = get_sentiment_score(transcript)

    try:
        toxicity_score = get_toxicity_score(transcript)
    except requests.exceptions.RequestException as e:
        detail = None
        if e.response is not None:
            try:
                detail = e.response.json()
            except ValueError:
                detail = e.response.text
        return jsonify({"error": "Perspective API request failed", "detail": detail}), 502

    return jsonify({
        "transcript": transcript,
        "toxicity_score": toxicity_score,
        "sentiment_score": sentiment_score,
    })

@app.route("/", methods=["GET"])
def upload_page():
    #shows the upload page
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    #handles the browser form upload
    #then renders a results page
    if "audio" not in request.files or request.files["audio"].filename == "":
        return render_template("upload.html", error="Please choose an audio file first.")

    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename)[1].lower()

    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return render_template(
            "upload.html",
            error=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = get_transcript(tmp_path)
    finally:
        os.remove(tmp_path)

    if not transcript:
        return render_template("upload.html", error="Could not transcribe this audio — try a different clip.")

    sentiment_score = get_sentiment_score(transcript)

    try:
        toxicity_score = get_toxicity_score(transcript)
    except requests.exceptions.RequestException:
        return render_template("upload.html", error="Perspective API request failed — check your API key and try again.")

    return render_template(
        "results.html",
        transcript=transcript,
        toxicity_score=toxicity_score,
        sentiment_score=sentiment_score,
        flagged=ACTIVE_FLAGGING_RULE(toxicity_score, sentiment_score),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)