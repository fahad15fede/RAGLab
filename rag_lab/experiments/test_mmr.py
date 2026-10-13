
from rag_lab.retrievers.mmr import MMRRetriever

question = "How do I declare a fang variable?"

for lambda_value in [1.0, 0.5, 0.2]:

    print("\n" + "=" * 60)
    print("LAMBDA:", lambda_value)

    retriever = MMRRetriever(
        lambda_mult=lambda_value,
        fetch_k=20
    )

    results = retriever.retrieve(question, k=5)

    for rank, result in enumerate(results, start=1):
        print(f"\nRESULT {rank}")
        print("Similarity:", round(result.score, 4))
        print("Chunk ID:", result.id[:12])
        print("Source:", result.metadata.get("filename"))
        print("Page:", result.metadata.get("page", 0) + 1)
        print("Content:", result.text[:250])
