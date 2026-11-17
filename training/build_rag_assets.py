from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KG = ROOT / "knowledge_graph"
NB = ROOT / "notebooks"
KG.mkdir(exist_ok=True)
NB.mkdir(exist_ok=True)

sources = [
    ("NIDDK_OVERVIEW", "https://www.niddk.nih.gov/health-information/diabetes/overview"),
    ("NIDDK_SYMPTOMS", "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes"),
    ("NIDDK_PREVENTION", "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-type-2-diabetes"),
    ("CDC_TYPE2", "https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html"),
    ("CDC_RISK", "https://www.cdc.gov/diabetes/risk-factors/index.html"),
    ("WHO_FACTS", "https://www.who.int/news-room/fact-sheets/detail/diabetes"),
]

triples = [
    ("Diabetes", "HAS_TYPE", "Type 1", "D1"),
    ("Diabetes", "HAS_TYPE", "Type 2", "D1"),
    ("Diabetes", "HAS_TYPE", "Gestational diabetes", "D1"),
    ("Diabetes", "CHARACTERIZED_BY", "High blood glucose", "D1"),
    ("Diabetes", "HAS_SYMPTOM", "Thirst", "D2"),
    ("Diabetes", "HAS_SYMPTOM", "Frequent urination", "D2"),
    ("Diabetes", "HAS_SYMPTOM", "Fatigue", "D2"),
    ("Diabetes", "HAS_SYMPTOM", "Unexplained weight loss", "D2"),
    ("Diabetes", "HAS_SYMPTOM", "Blurred vision", "D2"),
    ("Type 2", "HAS_RISK_FACTOR", "Family history", "D3"),
    ("Type 2", "HAS_RISK_FACTOR", "Overweight", "D3"),
    ("Type 2", "HAS_RISK_FACTOR", "Low physical activity", "D3"),
    ("Type 2", "HAS_RISK_FACTOR", "Older age", "D3"),
    ("Type 2", "HAS_RISK_FACTOR", "Prediabetes", "D6"),
    ("Type 2", "CAN_BE_DELAYED_BY", "Physical activity", "D4"),
    ("Type 2", "CAN_BE_DELAYED_BY", "Weight management", "D4"),
    ("Type 2", "CAN_BE_DELAYED_BY", "Healthy eating", "D4"),
    ("Diabetes", "MAY_AFFECT", "Heart", "D5"),
    ("Diabetes", "MAY_AFFECT", "Kidneys", "D5"),
    ("Diabetes", "MAY_AFFECT", "Eyes", "D5"),
    ("Diabetes", "MAY_AFFECT", "Feet", "D5"),
    ("Diabetes", "MAY_AFFECT", "Blood vessels", "D5"),
    ("High blood glucose", "MANAGED_BY", "Glucose management", "D5"),
    ("Glucose management", "HELPS_REDUCE_RISK_FOR", "Heart", "D5"),
    ("Glucose management", "HELPS_REDUCE_RISK_FOR", "Kidneys", "D5"),
    ("Glucose management", "HELPS_REDUCE_RISK_FOR", "Eyes", "D5"),
    ("Physical activity", "SUPPORTS", "Weight management", "D4"),
    ("Healthy eating", "SUPPORTS", "Weight management", "D4"),
    ("Type 2", "MAY_HAVE", "No clear symptoms", "D6"),
    ("Prediabetes", "INCREASES_RISK_OF", "Type 2", "D6"),
    ("Family history", "INCREASES_RISK_OF", "Type 2", "D6"),
]


def esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def cypher(rows: list[tuple[str, str, str, str]]) -> str:
    lines = ["// Generated evidence-linked diabetes knowledge graph"]
    for subject, relation, obj, evidence in rows:
        lines.append(
            "MERGE (s:Concept {name:'%s'}) MERGE (o:Concept {name:'%s'}) "
            "MERGE (s)-[:%s {evidence:'%s'}]->(o);"
            % (esc(subject), esc(obj), relation, evidence)
        )
    return "\n".join(lines) + "\n"


split = len(triples) // 2
(KG / "kb_diabet01.cypher").write_text(cypher(triples[:split]), encoding="utf-8")
(KG / "kb_diabet02.cypher").write_text(cypher(triples[split:]), encoding="utf-8")
(KG / "kb_diabet.cypher").write_text(cypher(triples), encoding="utf-8")

source_lines = ["# Diabetes knowledge-base sources", ""]
for name, url in sources:
    source_lines.append(f"- **{name}:** {url}")
source_lines += [
    "",
    "The executable prototype uses six short evidence records derived from the official pages above.",
    "The notebook also accepts locally uploaded PDF files and extracts their pages before chunking.",
    "The report distinguishes the six-document prototype from larger reference reports.",
]
(KG / "SOURCES.md").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# RAG + Knowledge Graph for Diabetes — Nguyen Ba Duy 076\n",
        "Evidence-linked educational prototype. It does not provide diagnosis or treatment."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "%pip -q install pandas numpy scikit-learn networkx pypdf\n",
        "import re, unicodedata, numpy as np, pandas as pd, networkx as nx\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "np.random.seed(76)\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "SOURCES = {\n",
        *[f"    '{name}': '{url}',\n" for name, url in sources],
        "}\n",
        "KB = [\n",
        " {'id':'D1','source':SOURCES['NIDDK_OVERVIEW'],'text':'Diabetes is a condition in which blood glucose is too high. Main types include type 1, type 2 and gestational diabetes.'},\n",
        " {'id':'D2','source':SOURCES['NIDDK_SYMPTOMS'],'text':'Possible symptoms include thirst, frequent urination, fatigue, unexplained weight loss and blurred vision.'},\n",
        " {'id':'D3','source':SOURCES['CDC_RISK'],'text':'Family history, overweight, low activity and older age are associated with increased type 2 diabetes risk.'},\n",
        " {'id':'D4','source':SOURCES['NIDDK_PREVENTION'],'text':'Physical activity, weight management and healthy eating can help prevent or delay type 2 diabetes.'},\n",
        " {'id':'D5','source':SOURCES['WHO_FACTS'],'text':'Diabetes may affect the heart, blood vessels, eyes, kidneys and feet.'},\n",
        " {'id':'D6','source':SOURCES['CDC_TYPE2'],'text':'Type 2 diabetes may develop slowly and may not have noticeable symptoms.'},\n",
        "]\n",
        f"TRIPLES = {triples!r}\n",
        "print('documents=', len(KB), 'triples=', len(TRIPLES))\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "# Optional PDF ingestion: upload PDFs in Colab, then extract one record per page.\n",
        "from pathlib import Path\n",
        "from pypdf import PdfReader\n",
        "def records_from_pdfs(folder='pdf_sources'):\n",
        "    records=[]\n",
        "    for path in Path(folder).glob('*.pdf'):\n",
        "        for page_no, page in enumerate(PdfReader(str(path)).pages, 1):\n",
        "            text=(page.extract_text() or '').strip()\n",
        "            if text: records.append({'id':f'{path.stem}-p{page_no}','source':str(path),'text':text})\n",
        "    return records\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "G=nx.MultiDiGraph()\n",
        "for s,r,o,e in TRIPLES: G.add_edge(s,o,relation=r,evidence=e)\n",
        "print('nodes=',G.number_of_nodes(),'edges=',G.number_of_edges())\n",
        "def one_hop(entity): return [(entity,d['relation'],n,d['evidence']) for _,n,d in G.out_edges(entity,data=True)]\n",
        "def two_hop(entity): return [fact for n in G.successors(entity) for fact in one_hop(n)]\n",
        "print(one_hop('Type 2')[:5]); print(two_hop('Diabetes')[:5])\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "vectorizer=TfidfVectorizer(ngram_range=(1,2),strip_accents='unicode')\n",
        "matrix=vectorizer.fit_transform([d['text'] for d in KB])\n",
        "def retrieve(question,k=3,min_score=.10):\n",
        "    scores=cosine_similarity(vectorizer.transform([question]),matrix).ravel()\n",
        "    idx=np.argsort(scores)[::-1][:k]\n",
        "    return [(KB[i],float(scores[i])) for i in idx if scores[i]>=min_score]\n",
        "def answer(question):\n",
        "    hits=retrieve(question)\n",
        "    if not hits: return {'answer':'The knowledge base does not contain enough evidence.','sources':[]}\n",
        "    return {'answer':' '.join(f\"{d['text']} [{d['id']}]\" for d,_ in hits), 'sources':[d['source'] for d,_ in hits]}\n",
        "answer('What are common symptoms of diabetes?')\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "tests=[('What are symptoms of diabetes?','D2'),('What increases type 2 risk?','D3'),('How can type 2 be delayed?','D4'),('What are diabetes types?','D1'),('What drug dose should I take?','NONE')]\n",
        "rows=[]\n",
        "for q,gold in tests:\n",
        "    ids=[d['id'] for d,_ in retrieve(q)]\n",
        "    hit=(gold in ids) if gold!='NONE' else not ids\n",
        "    rows.append({'question':q,'gold':gold,'retrieved':ids,'hit_at_3':int(hit)})\n",
        "evaluation=pd.DataFrame(rows); display(evaluation); print('Hit@3 including abstention=',evaluation.hit_at_3.mean())\n"
    ]},
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(NB / "RAG_KB_NguyenBaDuy_076.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote Cypher files, sources, and RAG notebook")
