import whisper
import jiwer

model = whisper.load_model("base")

#transcribe both clips using whisper
result1 = model.transcribe("audio/Pekinwoof.mp3")
print("Transcript 1: ")
print(result1["text"])
with open("results/transcript_clip1.txt", "w") as f:
    f.write(result1["text"])

result2 = model.transcribe("audio/Valragebait.mp4")
print("Transcript 2: ")
print(result2["text"])
with open("results/transcript_clip2.txt", "w") as f:
    f.write(result2["text"])

print("Transcripts saved")


# preprocess text (lowercase, remove punctuation etc)
wer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

#using jiwer to calculate WER
def calculate_wer(reference, hypothesis):
    return jiwer.wer(
        reference,
        hypothesis,
        reference_transform=wer_transform,
        hypothesis_transform=wer_transform,
    ) * 100


#load ground truth transcripts (manually written by me)
with open("audio/pwoofgroundtruth.txt", "r") as f:
    pwoofgroundtruth = f.read()

with open("audio/ragebaitgroundtruth.txt", "r") as f:
    ragebaitgroundtruth = f.read()

#compare whisper's output against the ground truth
wer1 = calculate_wer(pwoofgroundtruth, result1["text"])
wer2 = calculate_wer(ragebaitgroundtruth, result2["text"])

print(f"Clip 1 WER: {wer1:.2f}%")
print(f"Clip 2 WER: {wer2:.2f}%")