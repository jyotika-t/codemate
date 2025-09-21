import os
import json
import uuid
from typing import List, Dict, Any, Tuple
import math

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

import markdown2
import pdfkit

def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for pg in reader.pages:
        pages.append(pg.extract_text() or "")
    return "\n".join(pages)


class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: int = -1):
        print("Loading embedder:", model_name)
        self.model = SentenceTransformer(model_name, device="cuda" if device == 0 else "cpu")
    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True))

class FAISSIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  
        self.ids = [] 
    def add(self, vectors: np.ndarray, ids: List[str]):
        self.index.add(vectors.astype('float32'))
        self.ids.extend(ids)
    def search(self, qvec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        qvec = qvec.astype('float32')
        D, I = self.index.search(qvec, top_k)  
        results = []
        for sims, inds in zip(D, I):
            for sim, ind in zip(sims, inds):
                if ind == -1:
                    continue
                results.append((self.ids[int(ind)], float(sim)))
        return results

class MetadataStore:
    """Simple in-memory metadata store; persist to disk with JSON for demo"""
    def __init__(self, path="meta_store.json"):
        self.path = path
        self.store = {}  
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}
    def add(self, id: str, meta: Dict[str,Any]):
        self.store[id] = meta
    def get(self, id: str) -> Dict[str,Any]:
        return self.store.get(id)
    def persist(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2)

class LocalReasoner:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        print("Loading summarization model:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if self._cuda_available() else -1)
    def _cuda_available(self):
        import torch
        return torch.cuda.is_available()
    def summarize(self, texts: List[str], max_length=200) -> str:
        joined = "\n\n".join(texts)
        out = self.pipe(joined, max_length=max_length, truncation=True)
        return out[0]['summary_text']
    def generate_answer(self, context: str, question: str, max_length=256) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely with reasoning steps."
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        gen = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)

# ---------- Query Planner (multi-step) ----------
class QueryPlanner:
    """
    Very simple planner: splits the user query into subtasks:
    - find definition/explanation
    - find methods/approaches
    - extract comparisons/ pros-cons
    This is rule-based; you can replace with an LLM planner.
    """
    def plan(self, query: str) -> List[str]:
        subtasks = []
        subtasks.append(f"Definition/explanation of: {query}")
        subtasks.append(f"Main methods/approaches for: {query}")
        subtasks.append(f"Comparison and pros/cons of approaches for: {query}")
        subtasks.append(f"Recent advances or notable findings relating to: {query}")
        return subtasks

# ---------- DeepResearcher Agent ----------
class DeepResearcher:
    def __init__(self, embedder_model="sentence-transformers/all-MiniLM-L6-v2", summarize_model="sshleifer/distilbart-cnn-12-6"):
        self.embedder = LocalEmbedder(embedder_model)
        self.embed_dim = self.embedder.model.get_sentence_embedding_dimension()
        self.index = FAISSIndex(self.embed_dim)
        self.meta = MetadataStore()
        self.reasoner = LocalReasoner(summarize_model)
        self.planner = QueryPlanner()
    def ingest_document(self, text: str, source: str):
        chunks = chunk_text(text, max_tokens=300, overlap=50)
        vecs = self.embedder.embed(chunks)
        ids = []
        for i, chunk in enumerate(chunks):
            id = str(uuid.uuid4())
            meta = {"source": source, "chunk_index": i, "text": chunk}
            self.meta.add(id, meta)
            ids.append(id)
        self.index.add(vecs, ids)
        self.meta.persist()
        print(f"Ingested {len(chunks)} chunks from {source}")
    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        # Planner breaks query into subtasks
        subtasks = self.planner.plan(user_query)
        subresults = []
        qvecs = self.embedder.embed([user_query])
        for st in subtasks:
            st_vec = self.embedder.embed([st])
            hits = self.index.search(st_vec, top_k=top_k)
            # collect texts
            texts = []
            for hid, score in hits:
                meta = self.meta.get(hid)
                if meta:
                    texts.append({"text": meta["text"], "source": meta["source"], "score": score})
            # summarize top texts for that subtask
            top_texts = [t["text"] for t in texts][:6]
            if top_texts:
                summary = self.reasoner.summarize(top_texts, max_length=150)
            else:
                summary = "No relevant documents found."
            subresults.append({"subtask": st, "summary": summary, "evidence": texts})
        # final synthesis: combine summaries
        synth = self.reasoner.summarize([r["summary"] for r in subresults], max_length=250)
        return {"query": user_query, "subtasks": subresults, "synthesis": synth}
    def export_markdown(self, report: Dict[str,Any], filename: str = "report.md"):
        md = []
        md.append(f"# Research Report: {report['query']}\n")
        md.append("## Synthesis\n")
        md.append(report["synthesis"] + "\n")
        md.append("## Subtasks and Evidence\n")
        for s in report["subtasks"]:
            md.append(f"### {s['subtask']}\n")
            md.append(s["summary"] + "\n")
            md.append("Sources:\n")
            for e in s["evidence"]:
                txt = e['text'][:200].replace("\n"," ") + ("..." if len(e['text'])>200 else "")
                md.append(f"- {e['source']} (score: {e['score']:.3f}) — `{txt}`\n")
        md_text = "\n".join(md)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(md_text)
        print("Wrote", filename)
        return filename
    def export_pdf(self, md_file: str, pdf_file: str = "report.pdf"):
        html = markdown2.markdown_path(md_file)
        # requires wkhtmltopdf installed, or swap to weasyprint
        pdfkit.from_string(html, pdf_file)
        print("Wrote", pdf_file)
        return pdf_file

# ---------- Example usage ----------
def demo():
    agent = DeepResearcher()
    # Ingest sample documents (replace with your local files)
    sample_txt = "Transformer models are neural networks widely used in NLP. They use self-attention. Applications include summarization and question answering."
    agent.ingest_document(sample_txt, source="sample1.txt")
    # Optionally ingest a real PDF: uncomment & set path
    # pdf_text = read_pdf("papers/attention.pdf")
    # agent.ingest_document(pdf_text, source="attention.pdf")

    # Query
    q = "What are transformer models and their main approaches?"
    report = agent.query(q, top_k=4)
    print("Synthesis:\n", report["synthesis"])
    md = agent.export_markdown(report, filename="research_report.md")
    # export to pdf (requires wkhtmltopdf)
    try:
        agent.export_pdf(md, pdf_file="research_report.pdf")
    except Exception as e:
        print("PDF export failed (wkhtmltopdf required). Error:", e)

if __name__ == "__main__":
    demo()
