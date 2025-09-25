# semantic_kv_cache_with_ids.py
import itertools
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import initialize_agent, AgentType, tool

load_dotenv()

def norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

# ---- Embeddings + Vector DB (QUESTION vectors; ANSWER & ID in metadata) ----
emb = OpenAIEmbeddings(model="text-embedding-3-small")
kb = Chroma(collection_name="qa_kv_with_ids", embedding_function=emb)  # add persist_directory="kb_store" to persist

REL_THRESHOLD = 0.40                                  # relevance ∈ [0..1], higher=better
exact_cache_q2a = {}                                   # normalized_question -> answer
id_cache = {}                                          # id_str -> answer
id_counter = itertools.count(10000)                    # simple numeric IDs (10000, 10001, ...)

# ---------- Tools ----------
@tool
def kb_get_by_id(id_str: str) -> str:
    """Return the stored answer by numeric/string ID, or '__MISS__' if not found."""
    key = id_str.strip()
    ans = id_cache.get(key)
    if ans:
        return ans
    # Fallback: scan Chroma by filtering metadata (lightweight for small KBs)
    hits = kb.similarity_search_with_relevance_scores("dummy", k=1)  # we won't use this result
    # NOTE: Chroma Python API does not filter by metadata directly in similarity helpers.
    # If needed, keep a parallel dict (id_cache) as above, which we do.
    return "__MISS__"

@tool
def kb_lookup(query: str) -> str:
    """Return cached answer if a close semantic match to a stored QUESTION exists; else '__MISS__'."""
    qn = norm(query)
    if qn in exact_cache_q2a:
        print("[EXACT HIT]")
        return exact_cache_q2a[qn]
    hits = kb.similarity_search_with_relevance_scores(qn, k=1)
    if not hits:
        return "__MISS__"
    doc, rel = hits[0]
    print(f"[KB CHECK] relevance={rel:.3f}  matched_question='{doc.page_content}'")
    if rel is not None and rel >= REL_THRESHOLD:
        md = (doc.metadata or {})
        ans = md.get("answer")
        return ans if ans else "__MISS__"
    return "__MISS__"

@tool
def kb_upsert(payload: str) -> str:
    """Store (ID optional) a QUESTION→ANSWER pair.
    Accepts either:
      'question|||answer'              -> auto-generates numeric ID (e.g., 10000)
      'id|||question|||answer'         -> uses your provided ID
    Returns: '✅ Stored with id=<ID>'
    """
    parts = payload.split("|||")
    if len(parts) == 2:
        # no ID provided
        q, a = parts
        id_str = str(next(id_counter))
    elif len(parts) == 3:
        id_str, q, a = parts
        id_str = id_str.strip()
        if not id_str:
            id_str = str(next(id_counter))
    else:
        return "❌ Expected 'question|||answer' or 'id|||question|||answer'"

    q_clean = q.strip()
    a_clean = a.strip()
    q_norm = norm(q_clean)

    # Update in-memory caches
    exact_cache_q2a[q_norm] = a_clean
    id_cache[id_str] = a_clean

    # Store in Chroma: we vectorize the QUESTION text; stash ID & ANSWER in metadata
    kb.add_texts(
        [q_norm],
        metadatas=[{"answer": a_clean, "id": id_str}]
        # NOTE: we could also pass ids=... here, but Chroma's internal id is separate
    )
    return f"✅ Stored with id={id_str}"

@tool
def llm_answer(query: str) -> str:
    """Use the LLM to generate a concise answer (only used on KB misses)."""
    return llm.invoke(f"Answer succinctly in 3–5 lines.\n\nQ: {query}\nA:").content

# ---------- LLM (only for misses) ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Agent (ReAct) ----------
agent = initialize_agent(
    tools=[kb_get_by_id, kb_lookup, llm_answer, kb_upsert],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

ID_REGEX = re.compile(r"^\d{3,}$")  # treat pure 3+ digit strings as possible IDs

def ask_loop():
    print("\n--- Semantic KV Cache with IDs (ReAct on MISS) ---")
    print("• Upsert: auto-id via 'question|||answer'  OR custom-id via 'id|||question|||answer'")
    print("• Ask by ID (e.g., '10000') or by semantic question (typos allowed). Type 'q' to quit.")

    while True:
        raw = input("\nYou: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Bye! 👋"); break

        # 1) If input looks like a numeric ID → return by ID (zero LLM)
        if ID_REGEX.match(raw):
            ans = kb_get_by_id.invoke({"id_str": raw})
            if ans != "__MISS__":
                print("[KB ID HIT] (no LLM)\n" + ans + "\n")
                continue
            else:
                print("[KB ID MISS] (no answer for that id)\n")
                continue

        # 2) Zero-LLM fast path: exact/semantic lookup over QUESTIONS
        qn = norm(raw)
        if qn in exact_cache_q2a:
            print("[EXACT HIT] (no LLM)\n")
            print(exact_cache_q2a[qn], "\n")
            continue

        pre = kb.similarity_search_with_relevance_scores(qn, k=1)
        if pre:
            doc, rel = pre[0]
            print(f"[KB PRECHECK] relevance={rel:.3f}")
            if rel is not None and rel >= REL_THRESHOLD:
                print("[KB HIT] (no LLM)\n")
                print((doc.metadata or {}).get("answer", ""), "\n")
                continue

        # 3) MISS → let ReAct agent do: kb_lookup -> llm_answer -> kb_upsert
        print("[KB MISS] using agent (LLM once, then cache)\n")
        guidance = (
            "First call kb_get_by_id if the input looks like an ID (digits). "
            "If not, call kb_lookup. If it returns '__MISS__', call llm_answer to get the answer, "
            "then call kb_upsert with 'question|||answer'. Return only the final answer.\n"
            f"Question: {raw}"
        )
        out = agent.invoke({"input": guidance})
        ans = out["output"]

        # Safety upsert if agent forgot
        if norm(raw) not in exact_cache_q2a:
            # auto-ID upsert
            kb_upsert.invoke({"payload": f"{raw}|||{ans}"})

        print(ans, "\n")

if __name__ == "__main__":
    ask_loop()
