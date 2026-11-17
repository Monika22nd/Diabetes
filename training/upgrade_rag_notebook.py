from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks'/'RAG_KB_NguyenBaDuy_076.ipynb'
nb=json.loads(path.read_text(encoding='utf-8'))
# Remove a previous upgraded section when the builder is rerun.
nb['cells']=[c for c in nb['cells'] if c.get('metadata',{}).get('section')!='pdf_dense_rag']
meta={'section':'pdf_dense_rag'}
extra=[
 {'cell_type':'markdown','metadata':meta,'source':['## Executed PDF knowledge base with dense local embeddings\n','Upload the five course PDFs listed in `knowledge_graph/pdf_kb_run.json`. The PDF files are not committed to GitHub. The executed local run processed 214 pages into 261 chunks.']},
 {'cell_type':'code','execution_count':None,'metadata':meta,'outputs':[],'source':[
   '%pip -q install pypdf scikit-learn transformers sentencepiece accelerate\n',
   'from google.colab import files\n','uploaded = files.upload()  # select the five course PDFs\n',
   'from pathlib import Path\n','pdf_paths=[Path(name) for name in uploaded if name.lower().endswith(".pdf")]\n','print("PDF files:", [p.name for p in pdf_paths])\n']},
 {'cell_type':'code','execution_count':None,'metadata':meta,'outputs':[],'source':[
   'from pypdf import PdfReader\n','import re, numpy as np\n',
   'def clean(text): return re.sub(r"\\s+", " ", text).strip()\n',
   'def make_chunks(text,size=850,overlap=120):\n','    chunks=[]; start=0\n','    while start < len(text):\n','        end=min(len(text),start+size); value=text[start:end].strip()\n','        if len(value)>=80: chunks.append(value)\n','        if end>=len(text): break\n','        start=max(start+1,end-overlap)\n','    return chunks\n',
   'pdf_records=[]\n','for pdf in pdf_paths:\n','    reader=PdfReader(str(pdf))\n','    for page_no,page in enumerate(reader.pages,1):\n','        for chunk_no,text in enumerate(make_chunks(clean(page.extract_text() or "")),1):\n','            pdf_records.append({"id":f"{pdf.stem}-p{page_no}-c{chunk_no}","file":pdf.name,"page":page_no,"text":text})\n','print("pages=",sum(len(PdfReader(str(p)).pages) for p in pdf_paths),"chunks=",len(pdf_records))\n']},
 {'cell_type':'code','execution_count':None,'metadata':meta,'outputs':[],'source':[
   'from sklearn.feature_extraction.text import TfidfVectorizer\n','from sklearn.decomposition import TruncatedSVD\n','from sklearn.preprocessing import Normalizer\n',
   'pdf_vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,stop_words="english")\n','X_tfidf=pdf_vectorizer.fit_transform([r["text"] for r in pdf_records])\n',
   'dims=min(128,X_tfidf.shape[0]-1,X_tfidf.shape[1]-1)\n','pdf_svd=TruncatedSVD(n_components=dims,random_state=76)\n','pdf_norm=Normalizer(copy=False)\n','pdf_embeddings=pdf_norm.fit_transform(pdf_svd.fit_transform(X_tfidf))\n',
   'def retrieve_pdf(question,k=4):\n','    q=pdf_norm.transform(pdf_svd.transform(pdf_vectorizer.transform([question])))[0]\n','    scores=pdf_embeddings@q; top=np.argsort(scores)[::-1][:k]\n','    return [{**pdf_records[i],"score":float(scores[i])} for i in top]\n',
   'display(pd.DataFrame(retrieve_pdf("What is retrieval augmented generation?"))[["file","page","score","text"]])\n']},
 {'cell_type':'code','execution_count':None,'metadata':meta,'outputs':[],'source':[
   '# A small instruction-tuned language model generates only from retrieved context.\n','from transformers import pipeline\n','generator=pipeline("text2text-generation",model="google/flan-t5-small",max_new_tokens=160)\n',
   'def grounded_pdf_chat(question,min_score=.12):\n','    hits=retrieve_pdf(question,k=4)\n','    if not hits or hits[0]["score"]<min_score:\n','        return {"answer":"The knowledge base does not contain enough evidence.","sources":[]}\n','    context="\\n".join(f"[{h[\'file\']} p.{h[\'page\']}] {h[\'text\']}" for h in hits)\n','    prompt=f"Answer only from the context. If evidence is insufficient, say so. Cite source and page.\\nQuestion: {question}\\nContext:\\n{context}"\n','    answer=generator(prompt,do_sample=False)[0]["generated_text"]\n','    return {"answer":answer,"sources":[{"file":h["file"],"page":h["page"],"score":h["score"]} for h in hits]}\n',
   'grounded_pdf_chat("Which agents are used in the diabetes architecture?")\n']},
]
nb['cells'].extend(extra)
path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print('Upgraded',path,'cells=',len(nb['cells']))
