#checks how accurate each flagging rule is against my own labels
#reads results/threshold_comparison.csv (run compare_thresholds.py first)
import csv

CSV_PATH = "results/threshold_comparison.csv"
OUTPUT_PATH = "results/baseline_metrics.txt"

#my own labels for each clip (true = toxic, false = not toxic)
GROUND_TRUTH = {
    "LOLbroxah(non-toxic).mp3": False,
    "LOLCROWNIE'SCRASHOUT(toxic).mp3": True,
    "LOLDoubleliftPobelter(non-toxic).mp3": False,
    "LOLPekinwoof(non-toxic).mp3": False,
    "LOLvtuber(toxic).mp3": True,
    "LOLYamikazetilts(toxic).mp3": True,
    "MarvelRivalracist(toxic).mp3": True,
    "MarvelRivals(toxic).mp3": True,
    "MarvelRivalsfunny(non-toxic).mp3": False,
    "MarvelRivalsWhoelsome(non-toxic).mp4": False,
    "Marvelrivaltoxic2(toxic).mp3": True,
    "marverivalgirl(non-toxic).mp3": False,
    "Robloxcurse(toxic).mp3": True,
    "Robloxdad(non-toxic).mp3": False,
    "Robloxmad(toxic).mp3": True,
    "RobloxObama(non-toxic).mp3": False,
    "Robloxragebait(toxic).mp3": True,
    "Robloxwholesome(non-toxic).mp3": False,
    "Valchilllobby(non-toxic).mp3": False,
    "Valcrashout(toxic).mp3": True,
    "Valfunny(non-toxic).mp3": False,
    "Valrage(toxic).mp3": True,
    "Valragebait(toxic).mp4": True,
    "Valsinging(non-toxic).mp3": False,
}

#which flagging rules to test - matches the columns in the csv
RULES = {
    "Baseline": "baseline_flag",
    "Option A": "option_a_flag",
    "Option B": "option_b_flag",
}


def str_to_bool(value):
    #csv stores true/false as text, convert back to real bool
    return value.strip().lower() == "true"


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def evaluate(results, ground_truth, flag_column):
    #go through each clip and count tp/tn/fp/fn for one flagging rule
    tp = tn = fp = fn = 0
    rows = []

    for r in results:
        clip = r["clip"]
        if clip not in ground_truth:
            continue  # warning already printed once elsewhere

        actual_toxic = ground_truth[clip]
        predicted_toxic = str_to_bool(r[flag_column])

        if predicted_toxic and actual_toxic:
            tp += 1
            outcome = "TP"
        elif not predicted_toxic and not actual_toxic:
            tn += 1
            outcome = "TN"
        elif predicted_toxic and not actual_toxic:
            fp += 1
            outcome = "FP"
        else:
            fn += 1
            outcome = "FN"

        rows.append({
            "clip": clip,
            "ground_truth": "toxic" if actual_toxic else "not_toxic",
            "predicted": "toxic" if predicted_toxic else "not_toxic",
            "outcome": outcome,
        })

    return tp, tn, fp, fn, rows


def compute_metrics(tp, tn, fp, fn):
    #standard precision/recall/f1/accuracy formulas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return precision, recall, f1, accuracy


def check_missing_labels(results, ground_truth):
    #check once for any clips with no matching ground truth label
    for r in results:
        if r["clip"] not in ground_truth:
            print(f"Warning: no label for '{r['clip']}', skipping")


def build_rule_report(rule_name, rows, tp, tn, fp, fn, precision, recall, f1, accuracy):
    #builds the text block for one flagging rule
    lines = []
    lines.append(f"=== {rule_name} ===\n")

    lines.append("Per-clip results:")
    lines.append(f"{'CLIP':<35}{'GROUND TRUTH':<15}{'PREDICTED':<15}{'OUTCOME':<10}")
    lines.append("-" * 75)
    for r in rows:
        lines.append(f"{r['clip']:<35}{r['ground_truth']:<15}{r['predicted']:<15}{r['outcome']:<10}")

    lines.append("")
    lines.append("Confusion matrix:")
    lines.append(f"  True Positives:  {tp}")
    lines.append(f"  True Negatives:  {tn}")
    lines.append(f"  False Positives: {fp}")
    lines.append(f"  False Negatives: {fn}")

    lines.append("")
    lines.append("Metrics:")
    lines.append(f"  Precision: {precision:.3f}")
    lines.append(f"  Recall:    {recall:.3f}")
    lines.append(f"  F1 Score:  {f1:.3f}")
    lines.append(f"  Accuracy:  {accuracy:.3f}")
    lines.append("")

    return "\n".join(lines)


def build_summary_table(rule_metrics):
    #short side-by-side table comparing all 3 rules at a glance
    lines = ["=== Summary (all rules side by side) ===\n"]
    lines.append(f"{'RULE':<12}{'PRECISION':<12}{'RECALL':<12}{'F1':<12}{'ACCURACY':<12}")
    lines.append("-" * 60)
    for rule_name, (precision, recall, f1, accuracy) in rule_metrics.items():
        lines.append(f"{rule_name:<12}{precision:<12.3f}{recall:<12.3f}{f1:<12.3f}{accuracy:<12.3f}")
    lines.append("")
    return "\n".join(lines)


def main():
    results = load_results(CSV_PATH)
    check_missing_labels(results, GROUND_TRUTH)

    full_output = []
    rule_metrics = {}

    for rule_name, flag_column in RULES.items():
        tp, tn, fp, fn, rows = evaluate(results, GROUND_TRUTH, flag_column)
        precision, recall, f1, accuracy = compute_metrics(tp, tn, fp, fn)
        rule_metrics[rule_name] = (precision, recall, f1, accuracy)
        full_output.append(build_rule_report(rule_name, rows, tp, tn, fp, fn, precision, recall, f1, accuracy))

    #summary table goes at the very top for a quick glance
    output = build_summary_table(rule_metrics) + "\n" + "\n".join(full_output)

    print(output)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()