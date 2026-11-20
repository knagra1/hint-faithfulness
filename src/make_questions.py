"""Freeze the 300-question MMLU sample used by every run.

Run once from the project root:  python src/make_questions.py
Writes data/questions.json. The Hugging Face cache goes to ./.cache (gitignored),
which can be deleted afterwards.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache"))  # must be set before importing datasets

from datasets import load_dataset  # noqa: E402

N_QUESTIONS = 300
SEED = 0


def main() -> None:
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED).select(range(N_QUESTIONS))
    questions = [
        {
            "id": i,
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer": int(row["answer"]),  # index 0-3 of the correct choice
            "subject": row["subject"],
        }
        for i, row in enumerate(ds)
    ]
    assert all(len(q["choices"]) == 4 for q in questions), "every question needs 4 choices"

    out = ROOT / "data" / "questions.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"Wrote {len(questions)} questions to {out}")


if __name__ == "__main__":
    main()
