import os
import re
import json
from typing import List, Dict, Any
import PyPDF2
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# PDF Parsing
def parse_section_from_pdf(path: str) -> Dict[str, List[str]]:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text += "\n " + (p.extract_text() or " ")
    
    sections = {"ticket": [], "refund": [], "tourism": [], "packages": []}
    current = None
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        if "E-Ticket" in line:
            current = "ticket"
        elif "Refund Policy" in line:
            current = "refund"
        elif "Tourism Guide" in line:
            current = "tourism"
        elif "Budget-Friendly Packages" in line:
            current = "packages"
        elif current:
            sections[current].append(line)
    
    return sections


# Ticket Parser
def parse_ticket_details(lines: List[str]) -> Dict[str, Any]:
    text = " ".join(lines)
    res = {}
    m = re.search(r"PNR[:\s]*([A-Za-z0-9]{4,15})", text)
    if m: res["pnr"] = m.group(1)
    m = re.search(r"Train No[:\s]*([0-9]{3,6})", text)
    if m: res["train_no"] = m.group(1)
    m = re.search(r"Train Name[:\s]*([A-Za-z0-9 \-&]+)", text)
    if m: res["train_name"] = m.group(1).strip()
    m = re.search(r"Date[:\s]*([0-9\-\/]{6,10})", text)
    if m: res["date"] = m.group(1)
    m = re.search(r"From[:\s]*([A-Za-z]+)\s*To[:\s]*([A-Za-z]+)", text)
    if m:
        res["source"] = m.group(1)
        res["destination"] = m.group(2)
    m = re.search(r"Passenger[:\s]*([A-Za-z ]+)", text)
    if m: res["passenger"] = m.group(1)
    return res


# Vector DB Setup
def build_vectorstores(sections: Dict[str, List[str]]):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs_refund = [Document(page_content=line) for line in sections["refund"]]
    docs_tourism = [Document(page_content=line) for line in sections["tourism"]]
    docs_packages = [Document(page_content=line) for line in sections["packages"]]

    refund_store = Chroma.from_documents(docs_refund, embedding, persist_directory="./refund_chroma")
    tourism_store = Chroma.from_documents(docs_tourism, embedding, persist_directory="./tourism_chroma")
    packages_store = Chroma.from_documents(docs_packages, embedding, persist_directory="./packages_chroma")

    print("✅ Vector stores built with HuggingFace embeddings in Chroma")
    return refund_store, tourism_store, packages_store


# Query with Ollama
def query_agent(store, query: str, model: str = "llama2"):
    retriever = store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([d.page_content for d in docs])
    prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}\nAnswer:"

    llm = Ollama(model=model)
    return llm(prompt)


if __name__ == "__main__":
    pdf_path = "irctc_resources.pdf"
    if not os.path.exists(pdf_path):
        print("❌ PDF not found. Place irctc_resources.pdf in working directory.")
        exit()

    sections = parse_section_from_pdf(pdf_path)

    print("===== Ticket Details =====")
    ticket = parse_ticket_details(sections["ticket"])
    print(json.dumps(ticket, indent=2))

    # Build vector DBs
    refund_store, tourism_store, packages_store = build_vectorstores(sections)

    # Example queries
    print("\n===== Query: Refund Policy =====")
    ans = query_agent(refund_store, "What are the IRCTC Refund Policies?", model="gemma:2b")
    print(ans)

    print("\n===== Query: Tourism Guide =====")
    ans = query_agent(tourism_store, "Tell me about tourist attractions in Goa", model="llama2:7b")
    print(ans)

    print("\n===== Query: Budget-Friendly Packages =====")
    ans = query_agent(packages_store, "What are budget-friendly packages?", model="llama2:7b")
    print(ans)