from chroma_client import get_chroma_collection

collection = get_chroma_collection()

print("Connected to Chroma Cloud!")
print("Collection:", collection.name)
print("Stored vectors:", collection.count())