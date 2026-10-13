
import csv
import json
import time
from pathlib import Path

from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.semantic import SemanticRetriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.mmr import MMRRetriever


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "dataset.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


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

    records = []
    retrieved = {}

    for item in questions:
        question = item["question"]
        question_id = item["id"]

        print(f"\nQUESTION: {question}")

        for name, retriever in retrievers.items():
            start = time.perf_counter()
            results = retriever.retrieve(question, k=5)
            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            ids = [result.id for result in results]

            unique_sources = len({
                result.metadata.get("filename")
                for result in results
            })

            retrieved[(question_id, name)] = set(ids)

            records.append({
                "question_id": question_id,
                "question": question,
                "retriever": name,
                "latency_ms": round(latency_ms, 2),
                "results_count": len(results),
                "unique_chunks": len(set(ids)),
                "source_diversity": unique_sources
            })

            print(
                f"{name:10} "
                f"{latency_ms:8.2f} ms | "
                f"{len(set(ids))} chunks | "
                f"{unique_sources} sources"
            )

    output = RESULTS / "benchmark.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nBenchmark saved to {output}")


if __name__ == "__main__":
    main()
