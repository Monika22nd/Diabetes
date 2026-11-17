from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge_graph" / "pdf_kb_run.json"
PDF_FILES = [
    Path(r"D:\Download\intel_sys_dev_slide_02_final.pdf"),
    Path(r"D:\Download\intel_sys_dev_slide_03_deepLeaning_1.pdf"),
    Path(r"D:\Download\intel_sys_dev_slide_04_compare_models_CNN.pdf"),
    Path(r"D:\Download\knowled_graph RAG-lectureNote.pdf"),
    Path(r"D:\Download\project_architecture_multiagent_diabet_v1.pdf"),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk(text: str, size: int = 850, overlap: int = 120) -> list[str]:
    if not text:
        return []
    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + size // 2:
                end = boundary + 1
        value = text[start:end].strip()
        if len(value) >= 80:
            parts.append(value)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return parts


records = []
per_file = []
for pdf_path in PDF_FILES:
    reader = PdfReader(str(pdf_path))
    file_chunks = 0
    extracted_pages = 0
    for page_no, page in enumerate(reader.pages, 1):
        text = clean(page.extract_text() or "")
        if text:
            extracted_pages += 1
        for index, value in enumerate(chunk(text), 1):
            records.append({
                "id": f"{pdf_path.stem}-p{page_no}-c{index}",
                "file": pdf_path.name,
                "page": page_no,
                "text": value,
            })
            file_chunks += 1
    per_file.append({
        "file": pdf_path.name,
        "pages": len(reader.pages),
        "pages_with_text": extracted_pages,
        "chunks": file_chunks,
    })

texts = [row["text"] for row in records]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, stop_words="english")
tfidf = vectorizer.fit_transform(texts)
n_components = min(128, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=76)
normalizer = Normalizer(copy=False)
embeddings = normalizer.fit_transform(svd.fit_transform(tfidf))


def embed_query(question: str) -> np.ndarray:
    return normalizer.transform(svd.transform(vectorizer.transform([question])))[0]


def retrieve(question: str, k: int = 3) -> list[dict]:
    query = embed_query(question)
    scores = embeddings @ query
    top = np.argsort(scores)[::-1][:k]
    return [
        {"id": records[i]["id"], "file": records[i]["file"], "page": records[i]["page"], "score": round(float(scores[i]), 4)}
        for i in top
    ]


tests = [
    ("How should tabular data be represented for machine learning?", "intel_sys_dev_slide_02_final.pdf"),
    ("What is function composition in a deep neural network?", "intel_sys_dev_slide_03_deepLeaning_1.pdf"),
    ("How are Keras and PyTorch CNN implementations compared?", "intel_sys_dev_slide_04_compare_models_CNN.pdf"),
    ("What is retrieval augmented generation?", "knowled_graph RAG-lectureNote.pdf"),
    ("Which agents are used in the diabetes decision support architecture?", "project_architecture_multiagent_diabet_v1.pdf"),
]
evaluation = []
for question, expected_file in tests:
    hits = retrieve(question)
    evaluation.append({
        "question": question,
        "expected_file": expected_file,
        "retrieved": hits,
        "hit_at_3": int(any(hit["file"] == expected_file for hit in hits)),
    })

payload = {
    "seed": 76,
    "chunk_size": 850,
    "chunk_overlap": 120,
    "embedding": f"TF-IDF followed by TruncatedSVD dense LSA ({n_components} dimensions)",
    "files": per_file,
    "total_pages": sum(x["pages"] for x in per_file),
    "pages_with_text": sum(x["pages_with_text"] for x in per_file),
    "chunks": len(records),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
    "evaluation": evaluation,
    "hit_at_3": round(float(np.mean([x["hit_at_3"] for x in evaluation])), 4),
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
