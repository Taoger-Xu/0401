"""
Build eval_tool/answers/<experiment>/<category>.txt files for MME scoring.

Unlike playground/data/eval/MME/convert_answer_to_mme.py (which reads ground
truth from MME_Benchmark_release_version/<category>/questions_answers_YN/),
this reads ground truth from eval_tool/Your_Results/<category>.txt, which is
the ground-truth reference that ships with the official MME eval_tool for ALL
14 categories. This sidesteps the fact that the artwork/celebrity/landmark/
posters/scene categories in this machine's MME_Benchmark_release_version copy
are missing their questions_answers_YN/ folders (see docs/exp.md).
"""
import argparse
import json
import os
from collections import defaultdict

CATEGORIES = [
    "existence", "count", "position", "color", "posters", "celebrity", "scene",
    "landmark", "artwork", "OCR",
    "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning",
]


def load_gt(your_results_dir):
    gt = {}
    for cat in CATEGORIES:
        path = os.path.join(your_results_dir, f"{cat}.txt")
        with open(path) as f:
            for line in f:
                img, question, ans = line.rstrip("\n").split("\t")
                gt[(cat, img, question)] = ans
    return gt


def to_gt_question(prompt):
    prompt = prompt.replace(
        "\nAnswer the question using a single word or phrase.", ""
    ).strip()
    if "Please answer yes or no." not in prompt:
        prompt = prompt + " Please answer yes or no."
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--answers-dir", default="answers")
    ap.add_argument("--your-results-dir", default="eval_tool/Your_Results")
    ap.add_argument("--out-dir", default="eval_tool/answers")
    args = ap.parse_args()

    gt = load_gt(args.your_results_dir)

    answers = [json.loads(l) for l in open(os.path.join(args.answers_dir, f"{args.experiment}.jsonl"))]
    result_dir = os.path.join(args.out_dir, args.experiment)
    os.makedirs(result_dir, exist_ok=True)

    by_cat = defaultdict(list)
    missing = []
    for a in answers:
        category = a["question_id"].split("/")[0]
        img = a["question_id"].split("/")[-1]
        question = to_gt_question(a["prompt"])
        key = (category, img, question)
        if key not in gt:
            alt = (category, img, question.replace(" Please answer yes or no.", "  Please answer yes or no."))
            key = alt if alt in gt else None
        if key is None:
            missing.append((category, img, question))
            continue
        response = a["text"].replace("\n", " ").replace("\t", " ")
        by_cat[category].append((img, question, gt[key], response))

    for category, rows in by_cat.items():
        with open(os.path.join(result_dir, f"{category}.txt"), "w") as f:
            for img, q, g, resp in rows:
                f.write("\t".join([img, q, g, resp]) + "\n")

    total = sum(len(v) for v in by_cat.values())
    print(f"[{args.experiment}] wrote {total} rows across {len(by_cat)} categories, missing={len(missing)}")
    if missing:
        for m in missing[:5]:
            print("  missing example:", m)


if __name__ == "__main__":
    main()
