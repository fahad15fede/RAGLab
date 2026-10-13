
from rag_lab.retrievers.hybrid import HybridRetriever

retriever = HybridRetriever()

questions = [
    "How do I declare a fang variable?",
    "Which VIPER data type stores whole numbers?",
    "What is an illegal identifier?",
    "How do I write a conditional statement?"
]

for question in questions:
    print("\n" + "=" * 60)
    print("QUESTION:", question)

    results = retriever.retrieve(question, k=5)

    for rank, result in enumerate(results, 1):
        print(f"\nRESULT {rank}")
        print("RRF score:", round(result.score, 5))
        print("Source:", result.metadata.get("filename"))
        print("Page:", result.metadata.get("page", 0) + 1)
        print("Content:", result.text[:250])
