import os
import time
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 🔹 NEW: memory-aware imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -------------------- Config --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
FORCE_RECREATE = os.getenv("FORCE_RECREATE_INDEX", "0") == "1"

EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
EMBED_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise SystemExit("Please set OPENAI_API_KEY and PINECONE_API_KEY in .env")

# -------------------- PDF -> Text --------------------
def pdf_to_documents(pdf_path: Path) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    docs: List[Document] = []

    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            chunks = splitter.split_text(page_text)
            for i, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num + 1,  # 1-based
                            "chunk_idx": i,
                        },
                    )
                )
    return docs

def load_local_docs() -> List[Document]:
    """Load PDFs/TXTs from ./docs. No fallback content."""
    root = Path("docs")
    all_docs: List[Document] = []

    if root.exists():
        for p in root.iterdir():
            if p.suffix.lower() == ".pdf":
                all_docs.extend(pdf_to_documents(p))
            elif p.suffix.lower() == ".txt":
                text = p.read_text(encoding="utf-8", errors="ignore")
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                for i, chunk in enumerate(splitter.split_text(text)):
                    all_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": p.name, "page": None, "chunk_idx": i},
                        )
                    )
    return all_docs

# -------------------- Pinecone Setup (v5) --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

def wait_ready(name: str):
    while True:
        desc = pc.describe_index(name)
        if getattr(desc.status, "ready", False):
            break
        time.sleep(1)

def ensure_index(name: str, want_dim: int):
    try:
        existing = [idx.name for idx in pc.list_indexes().indexes]
    except Exception:
        existing = []
    if name not in existing:
        print(f"Creating Pinecone index '{name}' (dim={want_dim}) ...")
        pc.create_index(
            name=name,
            dimension=want_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        wait_ready(name)
        print("Index ready.")
        return

    desc = pc.describe_index(name)
    have_dim = getattr(desc, "dimension", None) or getattr(desc.status, "dimension", None)
    if have_dim and have_dim != want_dim:
        msg = (f"Index '{name}' has dimension {have_dim}, but embeddings are {want_dim}.")
        if FORCE_RECREATE:
            print(f"[WARN] {msg} Recreating due to FORCE_RECREATE_INDEX=1 ...")
            pc.delete_index(name)
            pc.create_index(
                name=name,
                dimension=want_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            )
            wait_ready(name)
            print("Index recreated and ready.")
        else:
            raise SystemExit(
                msg
                + " Delete the index in console or set a new PINECONE_INDEX, "
                  "or set FORCE_RECREATE_INDEX=1 in .env to auto-recreate."
            )
    else:
        wait_ready(name)
        print("Index ready.")

ensure_index(INDEX_NAME, want_dim=EMBED_DIM)

# -------------------- Vector Store --------------------
emb = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=emb)

def ingest() -> int:
    """Ingest docs into Pinecone. Returns number of chunks added (approx)."""
    docs = load_local_docs()
    if not docs:
        print("No documents found in ./docs (PDF/TXT).")
        return 0
    print(f"Ingesting {len(docs)} chunks ...")
    vectorstore.add_documents(docs)
    print("Ingestion complete.")
    return len(docs)

# -------------------- Retriever & LLM --------------------
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

SYSTEM = (
    "You are a helpful assistant. Use ONLY the provided context to answer. "
    "If the answer is not in context, say you don't know. Cite sources as [source: name p.X]."
)

def docs_to_context(ctx_docs: List[Document]) -> str:
    return "\n\n".join(
        f"- {d.page_content.strip()}\n  [source: {d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]"
        for d in ctx_docs
    ) or "(no relevant context found)"

# -------------------- 🔹 Memory-enabled chain --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    # This placeholder will be filled by the history wrapper automatically
    MessagesPlaceholder("history"),
    ("user", "Question:\n{question}\n\nContext:\n{context}")
])

base_chain = prompt | llm

# simple per-session history store (in-memory dict)
_history_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = InMemoryChatMessageHistory()
    return _history_store[session_id]

chat_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",   # which key to treat as the user message
    history_messages_key="history",  # where to inject/retrieve history
)

def ask_with_memory(session_id: str, q: str):
    # Retrieve fresh context every turn (memory is for *conversation*, not facts)
    docs = retriever.get_relevant_documents(q)
    ctx = docs_to_context(docs)

    # call the chain with history bound to session_id
    resp = chat_with_memory.invoke(
        {"question": q, "context": ctx},
        config={"configurable": {"session_id": session_id}},
    )
    return resp, docs

# -------------------- CLI --------------------
def chat_loop():
    print("🔹 RAG over local docs (Pinecone + OpenAI) with chat memory. Type 'exit' to quit.")
    session_id = "cli-session"  # you can swap this for a user id in a web app

    while True:
        q = input("\nAsk: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        resp, docs = ask_with_memory(session_id, q)

        print("\n--- Retrieved Chunks (brief) ---")
        for d in docs:
            snippet = d.page_content.replace("\n", " ")[:140]
            print(f"[{d.metadata.get('source')} p.{d.metadata.get('page')} c{d.metadata.get('chunk_idx')}] {snippet}...")

        print("\n--- Answer ---")
        print(resp.content)

if __name__ == "__main__":
    added = ingest()
    if added == 0:
        raise SystemExit("Add PDFs/TXTs to ./docs and rerun.")
    chat_loop()
