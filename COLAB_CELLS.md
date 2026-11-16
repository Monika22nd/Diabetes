# Google Colab Cells - End-to-End Diabetes Training

Student: **Nguyen Ba Duy**  
Identifier: **076**

Copy each numbered code block into a separate Colab code cell in the same order. Markdown blocks may be copied into text cells.

# End-to-End Diabetes Model Training

**Student:** Nguyen Ba Duy  
**Identifier:** 076

This notebook downloads the public Diabetes Prediction Dataset, audits and cleans it, creates leakage-safe train/validation/test partitions, trains classical baselines and a PyTorch multilayer perceptron, selects the operating threshold only on validation data, evaluates once on the test set, and exports all artifacts needed by the static web demo.

**Important:** This is an educational prototype. It is not a diagnostic, treatment, or triage system.

## Cell 1 - Install dependencies

Google Colab already includes PyTorch. The following command installs or updates the remaining packages used by the notebook.

### Code cell 1

```python
!pip -q install "pandas>=2.0" "scikit-learn>=1.4" "joblib>=1.3" "seaborn>=0.13"
```

## Cell 2 - Imports and reproducibility

### Code cell 2

```python
import copy
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 76
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)
```

## Cell 3 - Download the dataset

The dataset contains 100,000 rows and the fields `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`, and `diabetes`.

### Code cell 3

```python
DATA_URL = (
    "https://huggingface.co/datasets/marianeft/"
    "diabetes_prediction_dataset/resolve/main/diabetes_prediction_dataset.csv"
)

raw_df = pd.read_csv(DATA_URL)
print("Raw shape:", raw_df.shape)
display(raw_df.head())
```

## Cell 4 - Audit schema, missing values, duplicates, and target balance

### Code cell 4

```python
print(raw_df.info())
display(raw_df.describe(include="all").T)

audit = pd.DataFrame({
    "dtype": raw_df.dtypes.astype(str),
    "missing": raw_df.isna().sum(),
    "unique": raw_df.nunique(dropna=False),
})
display(audit)

print("Duplicate rows:", raw_df.duplicated().sum())
print("Target distribution:")
display(raw_df["diabetes"].value_counts(dropna=False).rename("count").to_frame())
display(raw_df["diabetes"].value_counts(normalize=True).rename("ratio").to_frame())
```

## Cell 5 - Clean and validate the data

Exact duplicates are removed before splitting. `No Info` is renamed to `Unknown`; it is not interpreted as a smoking behavior. Rows with unsupported gender labels or impossible numeric ranges are rejected.

### Code cell 5

```python
df = raw_df.copy()
df.columns = [c.strip() for c in df.columns]
df["gender"] = df["gender"].astype(str).str.strip().str.title()
df["smoking_history"] = (
    df["smoking_history"].astype(str).str.strip().replace({"No Info": "Unknown"})
)
df = df.drop_duplicates().reset_index(drop=True)

required_columns = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level",
    "blood_glucose_level", "diabetes",
]
assert set(required_columns).issubset(df.columns)
assert df[required_columns].isna().sum().sum() == 0
assert df["gender"].isin(["Female", "Male", "Other"]).all()
assert df["hypertension"].isin([0, 1]).all()
assert df["heart_disease"].isin([0, 1]).all()
assert df["diabetes"].isin([0, 1]).all()
assert df["age"].between(0, 120).all()
assert df["bmi"].between(5, 100).all()
assert df["HbA1c_level"].between(2, 20).all()
assert df["blood_glucose_level"].between(40, 500).all()

print("Clean shape:", df.shape)
print("Removed duplicates:", len(raw_df) - len(df))
display(df.head())
```

## Cell 6 - Exploratory analysis

These plots are descriptive. They do not prove causal relationships.

### Code cell 6

```python
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sns.countplot(data=df, x="diabetes", ax=axes[0, 0], color="#0F8B8D")
axes[0, 0].set_title("Target distribution")

sns.histplot(data=df, x="age", hue="diabetes", bins=30,
             stat="density", common_norm=False, ax=axes[0, 1])
axes[0, 1].set_title("Age by target")

sns.boxplot(data=df.sample(min(15000, len(df)), random_state=SEED),
            x="diabetes", y="HbA1c_level", ax=axes[1, 0], color="#9ED9CC")
axes[1, 0].set_title("HbA1c by target")

sns.boxplot(data=df.sample(min(15000, len(df)), random_state=SEED),
            x="diabetes", y="blood_glucose_level", ax=axes[1, 1], color="#F47B3A")
axes[1, 1].set_title("Blood glucose by target")

plt.tight_layout()
plt.show()
```

## Cell 7 - Define features and create train, validation, and test partitions

The split is stratified. The test set is not used for model or threshold selection.

### Code cell 7

```python
TARGET = "diabetes"
NUMERIC_FEATURES = [
    "age", "hypertension", "heart_disease", "bmi",
    "HbA1c_level", "blood_glucose_level",
]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET].astype(np.int64).copy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

split_summary = pd.DataFrame({
    "rows": [len(X_train), len(X_val), len(X_test)],
    "positive_rate": [y_train.mean(), y_val.mean(), y_test.mean()],
}, index=["train", "validation", "test"])
display(split_summary)
```

## Cell 8 - Build a leakage-safe preprocessor

### Code cell 8

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
], remainder="drop", verbose_feature_names_out=False)

print(preprocessor)
```

## Cell 9 - Train classical baselines

Logistic regression provides an interpretable linear baseline. Random forest provides a nonlinear tree ensemble baseline.

### Code cell 9

```python
logistic_pipeline = Pipeline([
    ("preprocess", copy.deepcopy(preprocessor)),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )),
])

forest_pipeline = Pipeline([
    ("preprocess", copy.deepcopy(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )),
])

logistic_pipeline.fit(X_train, y_train)
forest_pipeline.fit(X_train, y_train)
print("Classical baselines trained.")
```

## Cell 10 - Evaluate baselines on validation data

### Code cell 10

```python
def probability_metrics(y_true, probability, threshold=0.50):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }

baseline_results = {}
for name, pipeline in {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": forest_pipeline,
}.items():
    probability = pipeline.predict_proba(X_val)[:, 1]
    baseline_results[name] = probability_metrics(y_val, probability)

display(pd.DataFrame(baseline_results).T.sort_values("f1", ascending=False))
```

## Cell 11 - Fit the neural-network preprocessor on training data only

### Code cell 11

```python
nn_preprocessor = copy.deepcopy(preprocessor)
X_train_nn = nn_preprocessor.fit_transform(X_train).astype(np.float32)
X_val_nn = nn_preprocessor.transform(X_val).astype(np.float32)
X_test_nn = nn_preprocessor.transform(X_test).astype(np.float32)

y_train_nn = y_train.to_numpy(dtype=np.float32)
y_val_nn = y_val.to_numpy(dtype=np.float32)
y_test_nn = y_test.to_numpy(dtype=np.float32)

feature_names = nn_preprocessor.get_feature_names_out().tolist()
print("Processed shapes:", X_train_nn.shape, X_val_nn.shape, X_test_nn.shape)
print("Features:", feature_names)
```

## Cell 12 - Create PyTorch datasets and loaders

### Code cell 12

```python
BATCH_SIZE = 512

train_dataset = TensorDataset(
    torch.from_numpy(X_train_nn),
    torch.from_numpy(y_train_nn),
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val_nn),
    torch.from_numpy(y_val_nn),
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test_nn),
    torch.from_numpy(y_test_nn),
)

generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    generator=generator, num_workers=0
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train batches:", len(train_loader))
```

## Cell 13 - Define the multilayer perceptron

### Code cell 13

```python
class DiabetesMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

model = DiabetesMLP(X_train_nn.shape[1]).to(DEVICE)
print(model)
```

## Cell 14 - Configure weighted loss and optimizer

`pos_weight` gives the minority positive class more influence during optimization. Validation and test distributions remain unchanged.

### Code cell 14

```python
negative_count = int((y_train_nn == 0).sum())
positive_count = int((y_train_nn == 1).sum())
positive_weight = torch.tensor(
    [negative_count / positive_count], dtype=torch.float32, device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

print("Negative rows:", negative_count)
print("Positive rows:", positive_count)
print("Positive class weight:", positive_weight.item())
```

## Cell 15 - Define training and evaluation helpers

### Code cell 15

```python
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    probabilities = []
    targets = []

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * len(features)
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())

    probability = np.concatenate(probabilities)
    target = np.concatenate(targets).astype(int)
    return total_loss / len(loader.dataset), probability, target
```

## Cell 16 - Train with early stopping

The best checkpoint is selected only by validation loss. Training stops after six epochs without improvement.

### Code cell 16

```python
MAX_EPOCHS = 40
PATIENCE = 6
best_state = None
best_val_loss = float("inf")
epochs_without_improvement = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_probability, train_target = run_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_probability, val_target = run_epoch(
        model, val_loader, criterion
    )
    scheduler.step(val_loss)

    train_auc = roc_auc_score(train_target, train_probability)
    val_auc = roc_auc_score(val_target, val_probability)
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    print(
        f"Epoch {epoch:02d} | train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f} | val AUC {val_auc:.4f}"
    )

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break

model.load_state_dict(best_state)
history_df = pd.DataFrame(history)
print("Best validation loss:", best_val_loss)
```

## Cell 17 - Plot training history

### Code cell 17

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
history_df.plot(x="epoch", y=["train_loss", "val_loss"], ax=axes[0])
axes[0].set_title("Loss")
history_df.plot(x="epoch", y=["train_auc", "val_auc"], ax=axes[1])
axes[1].set_title("ROC-AUC")
plt.tight_layout()
plt.show()
```

## Cell 18 - Select the classification threshold on validation data

The candidate with the highest validation F1 among thresholds whose recall is at least 0.80 is selected. If no candidate meets that recall constraint, the notebook selects the highest validation F1.

### Code cell 18

```python
_, val_probability, val_target = run_epoch(model, val_loader, criterion)

threshold_rows = []
for threshold in np.linspace(0.05, 0.95, 181):
    row = probability_metrics(val_target, val_probability, threshold)
    row["threshold"] = float(threshold)
    threshold_rows.append(row)

threshold_df = pd.DataFrame(threshold_rows)
eligible = threshold_df[threshold_df["recall"] >= 0.80]
selection_pool = eligible if len(eligible) else threshold_df
best_row = selection_pool.sort_values(
    ["f1", "precision"], ascending=False
).iloc[0]
SELECTED_THRESHOLD = float(best_row["threshold"])

print("Selected threshold:", SELECTED_THRESHOLD)
display(best_row.to_frame("validation_value"))
```

## Cell 19 - Evaluate once on the held-out test set

### Code cell 19

```python
test_loss, test_probability, test_target = run_epoch(model, test_loader, criterion)
test_prediction = (test_probability >= SELECTED_THRESHOLD).astype(int)
test_metrics = probability_metrics(
    test_target, test_probability, SELECTED_THRESHOLD
)
test_metrics["loss"] = test_loss
test_metrics["threshold"] = SELECTED_THRESHOLD

display(pd.Series(test_metrics, name="test_value").to_frame())
print(classification_report(test_target, test_prediction, digits=4))
```

## Cell 20 - Confusion matrix

### Code cell 20

```python
cm = confusion_matrix(test_target, test_prediction)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Test confusion matrix at threshold {SELECTED_THRESHOLD:.3f}")
plt.show()
print(cm)
```

## Cell 21 - ROC and precision-recall curves

### Code cell 21

```python
fpr, tpr, _ = roc_curve(test_target, test_probability)
precision_curve, recall_curve, _ = precision_recall_curve(
    test_target, test_probability
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(fpr, tpr, label=f"AUC = {test_metrics['roc_auc']:.4f}")
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve")
axes[0].legend()

axes[1].plot(recall_curve, precision_curve,
             label=f"AP = {test_metrics['pr_auc']:.4f}")
axes[1].axhline(test_target.mean(), linestyle="--", color="gray",
                label="Positive-rate baseline")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-recall curve")
axes[1].legend()

plt.tight_layout()
plt.show()
```

## Cell 22 - Inspect feature names and preprocessing parameters

### Code cell 22

```python
display(pd.Series(feature_names, name="model_feature"))

fitted_numeric = nn_preprocessor.named_transformers_["numeric"]
fitted_categorical = nn_preprocessor.named_transformers_["categorical"]
scaler = fitted_numeric.named_steps["scaler"]
encoder = fitted_categorical.named_steps["onehot"]

print("Numeric means:", scaler.mean_)
print("Numeric scales:", scaler.scale_)
for name, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
    print(name, list(categories))
```

## Cell 23 - Save PyTorch and preprocessing artifacts

### Code cell 23

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": X_train_nn.shape[1],
    "threshold": SELECTED_THRESHOLD,
    "seed": SEED,
}, ARTIFACT_DIR / "diabetes_mlp.pt")

joblib.dump(nn_preprocessor, ARTIFACT_DIR / "preprocessor.joblib")

metadata = {
    "student": "Nguyen Ba Duy",
    "identifier": "076",
    "seed": SEED,
    "dataset_url": DATA_URL,
    "raw_rows": int(len(raw_df)),
    "clean_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "feature_names": feature_names,
    "selected_threshold": SELECTED_THRESHOLD,
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "limitations": [
        "Educational prototype only",
        "No external or prospective clinical validation",
        "Threshold is selected for this dataset and is not a clinical threshold",
        "Predicted probabilities are not clinically calibrated",
    ],
}
(ARTIFACT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("Saved artifacts:")
for path in sorted(ARTIFACT_DIR.iterdir()):
    print(path, path.stat().st_size, "bytes")
```

## Cell 24 - Export a browser-compatible model bundle

The static web demo implements the same standardization, one-hot encoding, dense layers, ReLU activations, sigmoid output, and selected threshold in JavaScript.

### Code cell 24

```python
def linear_layer_to_json(layer):
    return {
        "weights": layer.weight.detach().cpu().numpy().tolist(),
        "bias": layer.bias.detach().cpu().numpy().tolist(),
    }

network_layers = [module for module in model.network if isinstance(module, nn.Linear)]
browser_bundle = {
    "format": "diabetes-mlp-browser-v1",
    "student": "Nguyen Ba Duy",
    "identifier": "076",
    "threshold": SELECTED_THRESHOLD,
    "preprocessing": {
        "numeric_features": NUMERIC_FEATURES,
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": [values.tolist() for values in encoder.categories_],
        "output_features": feature_names,
    },
    "network": {
        "activation": "relu",
        "output_activation": "sigmoid",
        "layers": [linear_layer_to_json(layer) for layer in network_layers],
    },
    "metrics": {k: float(v) for k, v in test_metrics.items()},
    "disclaimer": "Educational prototype only. Not a medical diagnosis.",
}

(ARTIFACT_DIR / "model_bundle.json").write_text(
    json.dumps(browser_bundle), encoding="utf-8"
)
print("Browser bundle saved.")
```

## Cell 25 - Define raw-record inference

### Code cell 25

```python
def predict_record(record):
    row = pd.DataFrame([record], columns=FEATURE_COLUMNS)
    processed = nn_preprocessor.transform(row).astype(np.float32)
    tensor = torch.from_numpy(processed).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()
    return {
        "probability": probability,
        "threshold": SELECTED_THRESHOLD,
        "predicted_label": int(probability >= SELECTED_THRESHOLD),
    }
```

## Cell 26 - Run a sample inference

### Code cell 26

```python
sample_record = {
    "age": 52,
    "hypertension": 1,
    "heart_disease": 0,
    "bmi": 31.4,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 150,
    "gender": "Male",
    "smoking_history": "former",
}

print(json.dumps(predict_record(sample_record), indent=2))
```

## Cell 27 - Create a small source-linked knowledge base

This knowledge base is deliberately separate from the prediction model. It provides general educational evidence and must not be interpreted as personalized medical advice.

### Code cell 27

```python
KNOWLEDGE_BASE = [
    {
        "id": "WHO_DIABETES",
        "topic": "overview",
        "title": "WHO Diabetes Questions and Answers",
        "url": "https://www.who.int/news-room/questions-and-answers/item/diabetes",
        "text": "Diabetes occurs when the body cannot regulate blood glucose properly.",
    },
    {
        "id": "CDC_RISK",
        "topic": "risk factors",
        "title": "CDC Diabetes Risk Factors",
        "url": "https://www.cdc.gov/diabetes/risk-factors/",
        "text": "Risk factors include age, family history, overweight or obesity, and high blood pressure.",
    },
    {
        "id": "NIDDK_OVERVIEW",
        "topic": "management",
        "title": "NIDDK Diabetes Overview",
        "url": "https://www.niddk.nih.gov/health-information/diabetes",
        "text": "Diabetes information should support, not replace, care from qualified health professionals.",
    },
]

(ARTIFACT_DIR / "knowledge_base.json").write_text(
    json.dumps(KNOWLEDGE_BASE, indent=2), encoding="utf-8"
)
print("Knowledge items:", len(KNOWLEDGE_BASE))
```

## Cell 28 - Add deterministic rules and a coordinator

### Code cell 28

```python
def rule_flags(record):
    flags = []
    if record["bmi"] >= 30:
        flags.append("BMI is in a high range in this educational rule set")
    if record["HbA1c_level"] >= 6.5:
        flags.append("HbA1c value requires professional interpretation")
    if record["blood_glucose_level"] >= 200:
        flags.append("Blood glucose value requires professional interpretation")
    if record["hypertension"] == 1:
        flags.append("Hypertension is present in the submitted record")
    return flags

def retrieve_evidence(question, limit=3):
    words = set(question.lower().split())
    ranked = []
    for item in KNOWLEDGE_BASE:
        searchable = f"{item['topic']} {item['title']} {item['text']}".lower()
        score = sum(word in searchable for word in words)
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[:limit]]

def assess(record, question="general diabetes information"):
    prediction = predict_record(record)
    return {
        "prediction": prediction,
        "rule_flags": rule_flags(record),
        "evidence": retrieve_evidence(question),
        "warning": "Educational prototype only. Not a medical diagnosis.",
    }

display(assess(sample_record, "type 2 diabetes risk factors"))
```

## Cell 29 - Download all generated artifacts

Run this final cell in Colab to create one ZIP containing the trained model, preprocessor, metadata, browser bundle, and knowledge base.

### Code cell 29

```python
import shutil
from google.colab import files

archive_path = shutil.make_archive(
    "NguyenBaDuy_076_diabetes_artifacts",
    "zip",
    root_dir=ARTIFACT_DIR,
)
print("Created:", archive_path)
files.download(archive_path)
```

## Completion checklist

- Dataset downloaded and audited.
- Duplicates removed before splitting.
- Train, validation, and test sets remain separate.
- Preprocessor fitted on training data only.
- Logistic regression and random forest baselines trained.
- PyTorch MLP trained with weighted loss and early stopping.
- Threshold selected on validation data only.
- Test set evaluated once.
- Preprocessor, checkpoint, metadata, browser bundle, and knowledge base exported.
- Raw-record inference and coordinator demonstrated.
