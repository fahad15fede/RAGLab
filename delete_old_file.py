
from chroma_client import get_chroma_collection

collection = get_chroma_collection()

print("Total vectors before:", collection.count())

# Show all indexed PDF filenames
data = collection.get(include=["metadatas"])

filenames = sorted({
    metadata.get("filename", "Unknown")
    for metadata in data["metadatas"]
    if metadata
})

print("\nIndexed PDF files:")
for filename in filenames:
    print("-", filename)

# Change this to the EXACT old filename shown above
OLD_FILENAME = "VIPER_Lexical_Errors_Detailed_RAG_Guide.pdf"

matches = collection.get(
    where={"filename": OLD_FILENAME},
    include=["metadatas"]
)

print("\nMatching vectors:", len(matches["ids"]))

if not matches["ids"]:
    print("No matching vectors found.")
else:
    confirmation = input(
        f"Delete {len(matches['ids'])} vectors? Type DELETE: "
    )

    if confirmation == "DELETE":
        collection.delete(ids=matches["ids"])
        print("Old vectors deleted.")
    else:
        print("Deletion cancelled.")

print("\nTotal vectors after:", collection.count())
