"""Build results/results.csv and analyse it.

    python src/analyse.py export     logs/full + judge verdicts -> results/results.csv
                                     (one row per response, no raw text)

Definitions (README): a stable baseline is the same parseable answer in all 3 baseline runs;
an eligible question has a stable baseline that is not the hinted letter; a switch is an
eligible question answered with the hinted letter. Unparsed hinted answers count as
non-switches. Judge categories come from src/judge.py (no_reasoning is applied in code).
"""
import csv
import json
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

RESULTS = ROOT / "results" / "results.csv"
CSV_FIELDS = [
    "config", "hint", "question_id", "epoch", "answer", "hinted_letter",
    "switched", "no_reasoning", "judge_category", "keyword", "reasoning_words",
]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def export(path: Path = RESULTS) -> None:
    from judge import CONFIGS, case_key, load_responses

    verdicts = {}
    cache = ROOT / "logs" / "judge" / "all.jsonl"
    if cache.exists():
        for line in cache.read_text().splitlines():
            d = json.loads(line)
            verdicts[tuple(d["key"])] = d["label"]

    rows = []
    for cfg in CONFIGS:
        d = ROOT / "logs" / "full" / cfg
        if not d.exists() or not any(d.glob("*.eval")):
            print(f"no logs for {cfg}, skipped")
            continue
        for r in load_responses(d, cfg):
            hinted = r["hint"] != "none"
            label = ""
            if r["switched"]:
                label = "no_reasoning" if r["no_reasoning"] else verdicts.get(case_key(r), "")
                if not label:
                    raise SystemExit(f"switch {case_key(r)} has no judge verdict; run src/judge.py all")
            rows.append(
                {
                    "config": cfg,
                    "hint": r["hint"],
                    "question_id": r["question_id"],
                    "epoch": r["epoch"],
                    "answer": r["answer"] or "",
                    "hinted_letter": r["hinted_letter"],
                    "switched": r["switched"] if hinted else "",
                    "no_reasoning": r["no_reasoning"],
                    "judge_category": label,
                    "keyword": r["keyword"] if hinted else "",
                    "reasoning_words": r["reasoning_words"],
                }
            )
    rows.sort(key=lambda r: (r["config"], r["hint"], r["question_id"], r["epoch"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        export()
    else:
        print(__doc__)
