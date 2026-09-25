# Automated Toxic Speech Detection for Online Gaming

CM3070 Final Year Project
- A pipeliine that transcribes gaming voice chat and scores it for toxicity and sentiment, accessible via JSON API or a simple browser interface.

# How it works

1. Whisper (OpenAI): Transcribes uploaded audio into text
2. Perpective API (Google): Scores the transcript for toxicity (0 - 1)
3. VADER: Scores the transcript's sentiment (-1 to +1, negative to positive)
4. Flask: Orchestrates the models, and serves as a JSON API and browser-based upload + results interface

Flagging logic (whether a clip is marked "toxic") is kept within its own seperate module, (flagging.py)

# Project Structure
app.py Flask App: routes + scoring function
flagging.py: Flagging rule definitions
compare_thresholds.py: Batch testing audio clips against flagging rules
whisper_wer.py: Whisper transcription accuracy evaluation (WER using jiwer)
tests: Manual test scripts for /analyze and /analyze-audio
templates: upload and results html
audio/: Sample clips and manually written ground truth transcriptions
results/: saved comparisons and transcript outputs

# API Routes
- 'POST /analyze' - text in, toxicity scores + Sentiment Score out (JSON)
- 'POST /analyze-audio' - audio file in, transcript + both scores out (JSON)
- 'GET /' / 'POST /upload' - browser upload + results pages
