
import csv
import json
import statistics
import time
from itertools import combinations
from pathlib import Path

from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.semantic import SemanticRetriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.mmr import MMRRetriever


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "dataset.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

K = 5
REPEATS = 5


def percentile(values, percent):
    """Calculate percentile using linear interpolation."""
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def jaccard(set_a, set_b):
    """Measure overlap between two sets of chunk IDs."""
    union = set_a | set_b

    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


def save_csv(filename, records):
    if not records:
        return

    path = RESULTS / filename

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {path}")


def main():
    questions = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    retrievers = {
        "BM25": BM25Retriever(),
        "Semantic": SemanticRetriever(),
        "Hybrid": HybridRetriever(),
        "MMR": MMRRetriever(lambda_mult=0.5)
    }

    measurements = []
    overlap_records = []

    # Keep the latest result set for each query/method.
    retrieved_sets = {}

    print("\nWarming up retrievers...")

    first_question = questions[0]["question"]

    for name, retriever in retrievers.items():
        retriever.retrieve(first_question, k=K)
        print(f"Warm-up completed: {name}")

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        print(f"\nQUESTION: {question}")

        for name, retriever in retrievers.items():
            for run in range(1, REPEATS + 1):

                start = time.perf_counter()

                results = retriever.retrieve(
                    question,
                    k=K
                )

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                ids = {result.id for result in results}

                measurements.append({
                    "question_id": question_id,
                    "retriever": name,
                    "run": run,
                    "latency_ms": round(latency_ms, 3),
                    "results_count": len(results),
                    "unique_chunks": len(ids)
                })

                retrieved_sets[(question_id, name)] = ids

                print(
                    f"{name:10} "
                    f"Run {run}: {latency_ms:.2f} ms"
                )

        # Compare retrieved chunk IDs across methods.
        for method_a, method_b in combinations(
            retrievers.keys(), 2
        ):
            set_a = retrieved_sets[(question_id, method_a)]
            set_b = retrieved_sets[(question_id, method_b)]

            overlap_records.append({
                "question_id": question_id,
                "method_a": method_a,
                "method_b": method_b,
                "jaccard": round(
                    jaccard(set_a, set_b), 4
                ),
                "shared_chunks": len(set_a & set_b)
            })

    summary = []

    for name in retrievers:
        latencies = [
            row["latency_ms"]
            for row in measurements
            if row["retriever"] == name
        ]

        summary.append({
            "retriever": name,
            "runs": len(latencies),
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2)
        })

    summary.sort(key=lambda row: row["mean_ms"])

    save_csv("benchmark_v2_runs.csv", measurements)
    save_csv("benchmark_v2_summary.csv", summary)
    save_csv("retrieval_overlap.csv", overlap_records)

    print("\n========== PERFORMANCE SUMMARY ==========")

    for row in summary:
        print(
            f"{row['retriever']:10} "
            f"Mean: {row['mean_ms']:8.2f} ms | "
            f"Median: {row['median_ms']:8.2f} ms | "
            f"P95: {row['p95_ms']:8.2f} ms"
        )

    print("\n========== AVERAGE RETRIEVAL OVERLAP ==========")

    pairs = sorted({
        (row["method_a"], row["method_b"])
        for row in overlap_records
    })

    for method_a, method_b in pairs:
        scores = [
            row["jaccard"]
            for row in overlap_records
            if row["method_a"] == method_a
            and row["method_b"] == method_b
        ]

        print(
            f"{method_a:10} vs {method_b:10} "
            f"Jaccard: {statistics.mean(scores):.3f}"
        )


if __name__ == "__main__":
    main()
