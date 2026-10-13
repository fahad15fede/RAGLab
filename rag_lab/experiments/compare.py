
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

INPUT_FILE = RESULTS / "benchmark.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    print("\n========== BENCHMARK RESULTS ==========")
    print(df.to_string(index=False))

    summary = (
        df.groupby("retriever")
        .agg(
            avg_latency_ms=("latency_ms", "mean"),
            min_latency_ms=("latency_ms", "min"),
            max_latency_ms=("latency_ms", "max"),
            avg_source_diversity=("source_diversity", "mean"),
            avg_chunks=("unique_chunks", "mean")
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "avg_latency_ms"
    )

    summary = summary.round(2)

    print("\n========== SUMMARY ==========")
    print(summary.to_string(index=False))

    summary.to_csv(
        RESULTS / "leaderboard.csv",
        index=False
    )

    # Chart 1: Average latency
    plt.figure(figsize=(9, 5))

    plt.bar(
        summary["retriever"],
        summary["avg_latency_ms"]
    )

    plt.title("VIPER RAGLab — Average Retrieval Latency")
    plt.xlabel("Retrieval Method")
    plt.ylabel("Latency (ms)")

    plt.tight_layout()

    plt.savefig(
        RESULTS / "latency_comparison.png",
        dpi=200
    )

    plt.close()

    # Chart 2: Average source diversity
    plt.figure(figsize=(9, 5))

    plt.bar(
        summary["retriever"],
        summary["avg_source_diversity"]
    )

    plt.title("VIPER RAGLab — Source Diversity")
    plt.xlabel("Retrieval Method")
    plt.ylabel("Average Distinct PDF Sources")

    plt.ylim(0, 3)

    plt.tight_layout()

    plt.savefig(
        RESULTS / "source_diversity.png",
        dpi=200
    )

    plt.close()

    print("\nFiles generated:")
    print(" - leaderboard.csv")
    print(" - latency_comparison.png")
    print(" - source_diversity.png")


if __name__ == "__main__":
    main()
