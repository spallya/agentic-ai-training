"""
Hybrid Confluence RAG with full fallback (no DB):
- Step 1: Try CQL search (fast keyword-ish)
- Step 2: If CQL empty → fetch *all accessible pages* (paginated) and semantic-rank them
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from atlassian import Confluence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os

# ========= CONFIG =========
BASE_URL   = "https://yeramareddyranafsd.atlassian.net"
EMAIL      = ""
API_TOKEN = os.getenv("API_TOKEN")
MAX_CQL_RESULTS   = 20      # how many from CQL
MAX_ALL_PAGES     = 300     # limit for full fallback
CHUNK_SIZE        = 900
CHUNK_OVERLAP     = 120
TOP_K_CHUNKS      = 6
EMBED_MODEL       = "text-embedding-3-small"
CHAT_MODEL        = "gpt-4o-mini"
# ==========================

load_dotenv()

def html_to_text(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text("\n").strip()


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks, n, start = [], len(text), 0
    while start < n:
        end = min(n, start + size)
        ch = text[start:end]
        if ch.strip():
            chunks.append(ch)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def confluence_client() -> Confluence:
    return Confluence(url=BASE_URL, username=EMAIL, password=API_TOKEN, cloud=True)


def normalize_item(content: Dict[str, Any]) -> Tuple[str, str, str]:
    title = (content.get("title") or "Untitled").strip()
    space = (content.get("space") or {}).get("key") or ""
    pid   = str(content.get("id") or "")
    html  = (((content.get("body") or {}).get("storage") or {}).get("value")) or ""
    text  = html_to_text(html)
    url   = f"{BASE_URL}/wiki/spaces/{space}/pages/{pid}/{title.replace(' ', '+')}"
    return title, url, text


# ---------- Retrieval ----------
def cql_search(conf: Confluence, query: str, limit: int) -> List[Dict[str, Any]]:
    try:
        res = conf.cql(f'text ~ "{query}" AND type=page',
                       limit=limit, expand="content.body.storage,content.space") or {}
        return res.get("results", [])
    except Exception as e:
        print("⚠️ CQL failed:", e)
        return []


def fetch_all_pages(conf: Confluence, max_pages=300) -> List[Dict[str, Any]]:
    """
    Page through all Confluence pages up to max_pages.
    """
    all_pages = []
    start = 0
    batch_size = 50
    while len(all_pages) < max_pages:
        try:
            res = conf.get("rest/api/content",
                           params={"type": "page",
                                   "limit": batch_size,
                                   "start": start,
                                   "expand": "body.storage,space"}) or {}
            results = res.get("results", [])
            if not results:
                break
            all_pages.extend(results)
            start += batch_size
            time.sleep(0.2)
        except Exception as e:
            print("⚠️ Fetch all pages failed:", e)
            break
    return all_pages[:max_pages]


# ---------- Semantic Ranking ----------
def rank_chunks(question: str, items: List[Tuple[str, str, str]]) -> List[Tuple[float, Dict[str, str]]]:
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    qv = np.array(emb.embed_query(question), dtype="float32")
    dv = np.array(emb.embed_documents([it[2] for it in items]), dtype="float32")

    qn = qv / (np.linalg.norm(qv) + 1e-8)
    dn = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-8)
    sims = (dn @ qn).tolist()

    ranked = sorted(
        [(sims[i], {"title": items[i][0], "url": items[i][1], "chunk": items[i][2]})
         for i in range(len(items))],
        key=lambda x: x[0],
        reverse=True
    )
    return ranked


def build_context(ranked: List[Tuple[float, Dict[str, str]]], k: int) -> str:
    parts = []
    for i, (_, meta) in enumerate(ranked[:k], start=1):
        parts.append(f"[{i}] {meta['title']} — {meta['url']}\n{meta['chunk']}\n")
    return "\n".join(parts)


# ---------- Orchestrator ----------
def answer_question(question: str) -> str:
    conf = confluence_client()

    # Step 1: CQL
    cql_results = cql_search(conf, question, limit=MAX_CQL_RESULTS)
    if cql_results:
        items = [normalize_item(r["content"]) for r in cql_results]
    else:
        # Step 2: Full fallback
        print("ℹ️ CQL returned nothing, fetching all pages…")
        all_pages = fetch_all_pages(conf, max_pages=MAX_ALL_PAGES)
        items = [normalize_item(r) for r in all_pages]

    if not items:
        return "❌ No pages accessible."

    # Chunk
    candidates = []
    for title, url, text in items:
        for ch in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            candidates.append((title, url, ch))

    if not candidates:
        return "⚠️ Pages had no readable text."

    # Rank
    ranked = rank_chunks(question, candidates)
    context = build_context(ranked, TOP_K_CHUNKS)

    # LLM
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    prompt = f"""Answer using the context. If not in context, say "I don't know".
Cite sources like [1], [2].

Question:
{question}

Context:
{context}

Answer:"""
    return llm.invoke(prompt).content


# ---------- CLI ----------
if __name__ == "__main__":
    print("Ask Confluence (type 'exit' to quit):")
    while True:
        q = input("> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        try:
            print("\n" + answer_question(q) + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
