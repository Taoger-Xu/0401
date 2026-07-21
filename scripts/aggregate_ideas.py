#!/usr/bin/env python
"""Aggregate idea1/idea2/idea3 lmms-eval results (+ VisionZip / vanilla baselines)
into one comparison table. Reads docs/idea*/logs/k*/**/*results.json and
logs/lmms-eval/{vanilla,vz192,vz128,vz64}/**/*results.json.

Usage: python scripts/aggregate_ideas.py [--md docs/idea_summary.md]
"""
import argparse
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["gqa", "mmbench_en_dev", "mme", "pope", "scienceqa_img", "textvqa_val"]
COLS = ["GQA", "MMB-EN", "MME(tot)", "POPE-acc", "SQA", "TextVQA"]


def metrics_from(results):
    r = results.get("results", results)
    out = {}
    if "gqa" in r:
        out["GQA"] = r["gqa"].get("exact_match,none", float("nan")) * 100
    if "mmbench_en_dev" in r:
        out["MMB-EN"] = r["mmbench_en_dev"].get("gpt_eval_score,none", float("nan"))
    if "mme" in r:
        m = r["mme"]
        out["MME(tot)"] = m.get("mme_perception_score,none", 0) + m.get("mme_cognition_score,none", 0)
    if "pope" in r:
        out["POPE-acc"] = r["pope"].get("pope_accuracy,none", float("nan")) * 100
    if "scienceqa_img" in r:
        out["SQA"] = r["scienceqa_img"].get("exact_match,none", float("nan")) * 100
    if "textvqa_val" in r:
        out["TextVQA"] = r["textvqa_val"].get("exact_match,none", float("nan")) * 100
    return out


def latest_results_json(dir_):
    files = glob.glob(os.path.join(dir_, "**", "*results.json"), recursive=True)
    files = [f for f in files if "samples" not in os.path.basename(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load(dir_):
    f = latest_results_json(dir_)
    if not f:
        return None
    try:
        return metrics_from(json.load(open(f)))
    except Exception as e:
        return {"_err": str(e)}


def fmt(v):
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    rows = []  # (label, metrics)
    # baselines
    for name, label in [("vanilla", "vanilla(576)"), ("vz192", "VisionZip-192"),
                         ("vz128", "VisionZip-128"), ("vz64", "VisionZip-64")]:
        d = os.path.join(REPO, "logs", "lmms-eval", name)
        m = load(d) if os.path.isdir(d) else None
        if m:
            rows.append((label, m))
    # ideas
    idea_name = {"idea1": "idea1-spectral", "idea2": "idea2-localvar",
                 "idea3": "idea3-clsmmr", "idea4": "idea4-anchorcov"}
    for idea in ["idea1", "idea2", "idea3", "idea4"]:
        for K in [192, 128, 64, 346, 288]:
            d = os.path.join(REPO, "docs", idea, "logs", f"k{K}")
            if not os.path.isdir(d):
                continue
            m = load(d)
            if m:
                rows.append((f"{idea_name[idea]}-{K}", m))

    # build table
    header = "| Config | " + " | ".join(COLS) + " | Avg%vanilla |"
    sep = "|" + "---|" * (len(COLS) + 2)
    van = dict(rows).get("vanilla(576)", {})
    lines = [header, sep]
    for label, m in rows:
        vals = [fmt(m.get(c)) for c in COLS]
        # avg retention vs vanilla over available cols
        rr = [m[c] / van[c] for c in COLS if c in m and c in van and van[c]]
        avg = f"{100*sum(rr)/len(rr):.1f}%" if rr else "-"
        lines.append(f"| {label} | " + " | ".join(vals) + f" | {avg} |")
    table = "\n".join(lines)
    print(table)
    if args.md:
        with open(args.md, "w") as f:
            f.write("# 三方案 vs VisionZip vs vanilla 对比（6 benchmark）\n\n")
            f.write("指标：GQA/SQA/TextVQA=exact_match×100，MMB-EN=gpt_eval_score，"
                    "MME=perception+cognition，POPE=accuracy×100。Avg%vanilla=各列相对 vanilla 保留率均值。\n\n")
            f.write(table + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
