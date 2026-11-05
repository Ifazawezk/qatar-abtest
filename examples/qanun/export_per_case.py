"""Turn qanun's retrieval evaluation into one per-case file per method.

qanun (github.com/Ifazawezk/qanun) saves each method's ranking per question in
eval/results.json and the gold articles in eval/questions.jsonl. The metrics use
qanun's own definitions: recall@k is the share of gold articles in the top k, and
mrr@10 the reciprocal rank of the first gold article in the top 10. hit@k is 1
when any gold article is in the top k, which makes a right-or-wrong score for
McNemar's test.

    python examples/qanun/export_per_case.py path/to/qanun
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent


def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def reciprocal_rank(ranked, gold, cutoff=10):
    for rank, doc in enumerate(ranked[:cutoff], start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main(qanun: Path) -> None:
    questions = [json.loads(line) for line in (qanun / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    results = json.loads((qanun / "eval/results.json").read_text(encoding="utf-8"))
    for method in results["methods"]:
        rows = []
        for q in questions:
            ranked, gold = method["rankings"][q["id"]], q["gold"]
            row = {"id": q["id"], "lang": q["lang"]}
            row.update({f"recall@{k}": recall_at_k(ranked, gold, k) for k in (1, 3, 4, 5, 10)})
            row["mrr@10"] = reciprocal_rank(ranked, gold)
            row.update({f"hit@{k}": int(bool(set(ranked[:k]) & set(gold))) for k in (1, 5)})
            rows.append(row)
        path = HERE / f"{slug(method['method'])}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
        print(f"{path.name}: {method['method']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
