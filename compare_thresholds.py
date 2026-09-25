#compare the 3 flagging options on a batch of audio clips
import os
import csv

#reuse the same functions app.py uses
from app import get_transcript, get_toxicity_score, get_sentiment_score
from flagging import FLAGGING_OPTIONS

AUDIO_DIR = "audio"
AUDIO_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm"}
SNIPPET_LENGTH = 60  #how many characters of transcript to show in table


def find_audio_clips(folder):
    #get all audio files in the folder
    clips = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            clips.append(os.path.join(folder, filename))
    return clips


def evaluate_clip(path):
    #runs one clip through the entire pipeline + all flagging options
    transcript = get_transcript(path)
    toxicity = get_toxicity_score(transcript)
    sentiment = get_sentiment_score(transcript)

    flags = {
        name: rule(toxicity, sentiment)
        for name, rule in FLAGGING_OPTIONS.items()
    }

    snippet = transcript[:SNIPPET_LENGTH]
    if len(transcript) > SNIPPET_LENGTH:
        snippet += "..."

    return {
        "clip": os.path.basename(path),
        "transcript_snippet": snippet,
        "full_transcript": transcript,
        "toxicity": round(toxicity, 3),
        "vader": round(sentiment, 3),
        **{f"{name}_flag": flag for name, flag in flags.items()},
    }


def print_table(results):
    #prints a table to terminal
    columns = ["clip", "toxicity", "vader", "baseline_flag", "option_a_flag", "option_b_flag"]
    widths = {"clip": 22, "toxicity": 10, "vader": 10, "baseline_flag": 10, "option_a_flag": 10, "option_b_flag": 10}

    header = "".join(col.replace("_flag", "").upper().ljust(widths[col]) for col in columns)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = "".join(str(r[col]).ljust(widths[col]) for col in columns)
        print(row)
    print()


def save_csv(results, path="results/threshold_comparison.csv"):
    #save everything to csv for inspection (transcript too)
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved full results (with transcript snippets) to {path}")


def main():
    clips = find_audio_clips(AUDIO_DIR)

    if not clips:
        print(f"No audio files found in '{AUDIO_DIR}/' — nothing to compare.")
        return

    results = []
    for path in clips:
        print(f"Processing {os.path.basename(path)}...")
        results.append(evaluate_clip(path))

    print_table(results)
    save_csv(results)

    #print full transcripts for viewing
    print("Full transcripts:\n")
    for r in results:
        print(f"--- {r['clip']} ---")
        print(r["full_transcript"])
        print()


if __name__ == "__main__":
    main()