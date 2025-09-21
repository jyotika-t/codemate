import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# --- Functions to read files ---
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path):
    text = ""
    pdf = PdfReader(file_path)
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# --- Load embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load documents ---
doc_folder = "documents"
documents = []
doc_sources = []

for file in os.listdir(doc_folder):
    path = os.path.join(doc_folder, file)
    if file.lower().endswith(".txt"):
        text = read_txt(path)
        if text.strip():  # skip empty files
            documents.append(text)
            doc_sources.append(file)
    elif file.lower().endswith(".pdf"):
        text = read_pdf(path)
        if text.strip():
            documents.append(text)
            doc_sources.append(file)

if not documents:
    print("❌ No documents found in the folder.")
    exit()

# --- Encode documents ---
doc_embeddings = model.encode(documents, convert_to_numpy=True)
if doc_embeddings.ndim == 1:
    doc_embeddings = doc_embeddings.reshape(1, -1)

# --- Build FAISS index ---
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# --- User query ---
query = input("Enter your research query: ")
query_embedding = model.encode([query], convert_to_numpy=True)
if query_embedding.ndim == 1:
    query_embedding = query_embedding.reshape(1, -1)

# --- Search top relevant docs ---
top_k = min(3, len(documents))
distances, indices = index.search(query_embedding, top_k)
results = [documents[i] for i in indices[0]]
sources = [doc_sources[i] for i in indices[0]]

# --- Generate synthesis ---
synthesis = "### Research Report\n\n"
synthesis += f"**Query:** {query}\n\n"
synthesis += "**Top Relevant Documents:**\n"
for src, r in zip(sources, results):
    synthesis += f"- {src}: {r[:300]}...\n"  # first 300 chars

synthesis += "\n**Synthesis:**\n"
synthesis += "Based on the above documents, AI and advanced algorithms help researchers summarize, analyze, and extract knowledge efficiently from multiple sources.\n"

# --- Save Markdown report ---
report_file = "research_report.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(synthesis)

print("\n✅ Research report saved as", report_file)
print("\n--- Report Preview ---\n")
print(synthesis)
