# Diabetes MAS Lab

End-to-end educational diabetes model training and a source-linked multi-agent web demonstration by **Nguyen Ba Duy - 076**.

> This repository is an educational prototype. It is not a medical diagnosis, treatment recommendation, or triage service.

## Run the complete Google Colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Monika22nd/Diabetes/blob/main/notebooks/Diabetes_End_to_End_Training_NguyenBaDuy_076.ipynb)

The notebook contains every required code cell in execution order:

1. Install dependencies.
2. Import libraries and set deterministic seeds.
3. Download the 100,000-row public dataset.
4. Audit schema, missing values, duplicates, and class balance.
5. Clean and validate records.
6. Explore feature and target distributions.
7. Create stratified train, validation, and test partitions.
8. Build leakage-safe preprocessing.
9. Train logistic regression and random forest baselines.
10. Train a weighted PyTorch MLP with early stopping.
11. Select the operating threshold on validation data.
12. Evaluate once on the held-out test set.
13. Save the checkpoint, preprocessor, metadata, browser model, and knowledge base.
14. Run raw-record inference and a small coordinated assessment.

The same cells are available in [`COLAB_CELLS.md`](COLAB_CELLS.md) for manual copy and paste.

## Knowledge graph and RAG notebook

[Open the source-linked RAG + Knowledge Graph notebook in Colab](https://colab.research.google.com/github/Monika22nd/Diabetes/blob/main/notebooks/RAG_KB_NguyenBaDuy_076.ipynb). The executable prototype contains six official-source evidence records, 24 graph nodes, 31 evidence-linked relations, one-hop and two-hop traversal, TF-IDF retrieval, source citations, and an abstention path. Neo4j import files are provided as [`kb_diabet01.cypher`](knowledge_graph/kb_diabet01.cypher), [`kb_diabet02.cypher`](knowledge_graph/kb_diabet02.cypher), and the merged [`kb_diabet.cypher`](knowledge_graph/kb_diabet.cypher). The source inventory contains six official health-information links and the notebook includes optional PDF ingestion.

## Verified local training result

The included browser model was trained from the same pipeline with seed 76 on 96,146 records after exact duplicate removal.

| Metric | Test result |
|---|---:|
| Accuracy | 0.9513 |
| Precision | 0.6939 |
| Recall | 0.8019 |
| F1 | 0.7440 |
| ROC-AUC | 0.9752 |
| PR-AUC | 0.8753 |
| Selected threshold | 0.7100 |

These values are technical results for this dataset and split. They are not measures of clinical validity.

## Static web demo

The files under `docs/` implement preprocessing and MLP inference directly in JavaScript. No patient record is sent to a backend. GitHub Pages deployment is configured in `.github/workflows/pages.yml`.

## Repository structure

```text
Diabetes/
├── .github/workflows/pages.yml
├── COLAB_CELLS.md
├── README.md
├── docs/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── assets/
├── model/
│   ├── model_bundle.json
│   ├── metadata.json
│   ├── diabetes_mlp.pt
│   ├── preprocessor.joblib
│   └── knowledge_base.json
├── knowledge_graph/
│   ├── SOURCES.md
│   ├── kb_diabet01.cypher
│   ├── kb_diabet02.cypher
│   └── kb_diabet.cypher
├── notebooks/
│   ├── Diabetes_End_to_End_Training_NguyenBaDuy_076.ipynb
│   └── RAG_KB_NguyenBaDuy_076.ipynb
└── training/
    ├── build_notebook.py
    └── run_local_training.py
```

## Data source

The notebook downloads the public 100,000-row Diabetes Prediction Dataset from the Hugging Face dataset repository `marianeft/diabetes_prediction_dataset`. The raw data is not committed here.

## Medical safety boundary

- The model has no external or prospective clinical validation.
- The threshold was selected for the supplied dataset, not by a clinical protocol.
- Predicted probabilities are not clinically calibrated.
- Rule flags are deterministic educational checks.
- Evidence links provide general information from WHO, CDC, and NIDDK.
- Medical decisions must be made with qualified health professionals.
