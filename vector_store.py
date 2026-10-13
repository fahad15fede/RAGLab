
import os
import re
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 1. Find all PDFs
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in {DATA_DIR}"
    )

# 2. Load and clean documents
def clean_text(text):
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


documents = []

for pdf_path in pdf_files:
    print(f"Loading: {pdf_path.name}")

    loader = PyPDFLoader(
        str(pdf_path),
        extraction_mode="layout"
    )

    pages = loader.load()

    for page in pages:
        page.page_content = clean_text(page.page_content)
        page.metadata["filename"] = pdf_path.name

        if page.page_content:
            documents.append(page)

# 3. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)

# 4. Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    output_dimensionality=768
)

# 5. Connect to Chroma Cloud
client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"]
)

vector_store = Chroma(
    collection_name=os.getenv(
        "CHROMA_COLLECTION",
        "viper_knowledge_base"
    ),
    embedding_function=embeddings,
    client=client
)

# 6. Generate stable IDs
ids = []

for chunk in chunks:
    filename = chunk.metadata.get("filename", "")
    page = chunk.metadata.get("page", 0)

    unique_content = (
        f"{filename}:{page}:{chunk.page_content}"
    )

    chunk_id = hashlib.sha256(
        unique_content.encode("utf-8")
    ).hexdigest()

    ids.append(chunk_id)

# 7. Upload in batches
batch_size = 20

for start in range(0, len(chunks), batch_size):
    batch_chunks = chunks[start:start + batch_size]
    batch_ids = ids[start:start + batch_size]

    vector_store.add_documents(
        documents=batch_chunks,
        ids=batch_ids
    )

    print(
        f"Indexed {min(start + batch_size, len(chunks))}"
        f"/{len(chunks)} chunks"
    )

# 8. Verify
collection = client.get_collection(
    os.getenv(
        "CHROMA_COLLECTION",
        "viper_knowledge_base"
    )
)

print("\nIndexing complete!")
print("PDF files:", len(pdf_files))
print("Pages loaded:", len(documents))
print("Chunks processed:", len(chunks))
print("Stored vectors:", collection.count())

# 9. Test retrieval
question = "How do I declare an integer variable in VIPER?"

results = vector_store.similarity_search(
    question,
    k=3
)

print("\nQuestion:", question)

for i, doc in enumerate(results, start=1):
    print(f"\n--- RESULT {i} ---")
    print("File:", doc.metadata.get("filename"))
    print("Page:", doc.metadata.get("page", 0) + 1)
    print(doc.page_content[:600])
