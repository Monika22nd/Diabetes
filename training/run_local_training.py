import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Diabetes_End_to_End_Training_NguyenBaDuy_076.ipynb"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"

os.chdir(ROOT)
namespace = {"display": print}
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

skip = {1, 6, 17, 20, 21, 29}
for index, cell in enumerate(code_cells, start=1):
    if index in skip:
        print(f"Skipping presentation-only cell {index}")
        continue
    source = cell["source"]
    if index == 16:
        source = source.replace("MAX_EPOCHS = 40", "MAX_EPOCHS = 24")
    print(f"Running code cell {index}")
    exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)

MODEL_DIR.mkdir(exist_ok=True)
for path in ARTIFACTS.iterdir():
    if path.is_file():
        shutil.copy2(path, MODEL_DIR / path.name)

print("Local verification complete.")
for path in sorted(MODEL_DIR.iterdir()):
    print(path.name, path.stat().st_size)
