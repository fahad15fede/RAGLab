
from rag_lab.retrievers.semantic import SemanticRetriever

retriever = SemanticRetriever()

questions = [
    "How do I declare a fang variable?",
    "Which VIPER data type stores whole numbers?",
    "What is an illegal identifier?",
    "How do I write a conditional statement?"
]

for question in questions:

    print("\n" + "=" * 65)
    print("QUESTION:", question)

    results = retriever.retrieve(question, k=5)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- RESULT {rank} ---")
        print("Distance:", round(result.score, 4))
        print("Source:", result.metadata.get("filename"))
        print("Page:", result.metadata.get("page", 0) + 1)
        print("Content:", result.text[:350])
