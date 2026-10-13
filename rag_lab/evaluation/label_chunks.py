
import json
from pathlib import Path

from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.semantic import SemanticRetriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.mmr import MMRRetriever


DATASET_PATH = Path(__file__).parent / "dataset.json"


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    retrievers = {
        "BM25": BM25Retriever(),
        "Semantic": SemanticRetriever(),
        "Hybrid": HybridRetriever(),
        "MMR": MMRRetriever(lambda_mult=0.5)
    }

    for item in dataset:
        question = item["question"]

        print("\n" + "=" * 75)
        print("QUESTION:", question)

        candidates = {}

        for name, retriever in retrievers.items():
            results = retriever.retrieve(question, k=5)

            for result in results:
                if result.id not in candidates:
                    candidates[result.id] = {
                        "result": result,
                        "found_by": []
                    }

                candidates[result.id]["found_by"].append(name)

        chunks = list(candidates.values())

        print(f"\nUnique candidate chunks: {len(chunks)}")

        for index, entry in enumerate(chunks, start=1):
            result = entry["result"]

            print(f"\n[{index}] {result.metadata.get('filename')}")
            print("Page:", result.metadata.get("page", 0) + 1)
            print("Found by:", ", ".join(entry["found_by"]))
            print("ID:", result.id)
            print("Content:")
            print(result.text[:650])
            print("-" * 60)

        print("\nEnter relevant chunk numbers separated by commas.")
        print("Example: 1,3,5,8")
        print("Enter 'q' to save and quit.")

        while True:
            selection = input("\nRelevant chunks: ").strip()

            if selection.lower() == "q":
                DATASET_PATH.write_text(
                    json.dumps(dataset, indent=2),
                    encoding="utf-8"
                )
                return

            try:
                selected_numbers = [
                    int(number.strip())
                    for number in selection.split(",")
                    if number.strip()
                ]

                if any(
                    number < 1 or number > len(chunks)
                    for number in selected_numbers
                ):
                    raise ValueError("Number out of range")

                break

            except ValueError:
                print("Invalid selection. Try again.")

        item["relevant_chunk_ids"] = [
            chunks[number - 1]["result"].id
            for number in sorted(set(selected_numbers))
        ]

        DATASET_PATH.write_text(
            json.dumps(dataset, indent=2),
            encoding="utf-8"
        )

        print("Labels saved!")

    print("\nAll questions labeled!")


if __name__ == "__main__":
    main()
