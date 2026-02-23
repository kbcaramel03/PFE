from pathlib import Path
from ultralytics import YOLO

# === 1. Trouver automatiquement le dernier run ===
runs_dir = Path("runs/detect")
train_dirs = sorted(
    [p for p in runs_dir.glob("train*") if p.is_dir()],
    key=lambda x: x.stat().st_mtime
)

if not train_dirs:
    raise RuntimeError("Aucun dossier train trouvé")

last_run = train_dirs[-1]
weights = last_run / "weights" / "best.pt"

print("Using model:", weights)

# === 2. Charger le modèle ===
model = YOLO(str(weights))

# === 3. CALCULER LES METRIQUES ===
metrics = model.val(
    data="thermal.yaml",   # ton YAML dataset
    split="val",               # validation set
    device=0,                  # GPU
    imgsz=640
)

print("\n=== METRIQUES ===")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# === 4. PREDICTION SUR QUELQUES IMAGES ===
images = list(Path("dataset_mix/images/val").glob("*"))[:5]

results = model.predict(
    source=[str(p) for p in images],
    imgsz=640,
    conf=0.25,
    device=0,
    save=True
)

print("\nRésultats sauvegardés dans:", results[0].save_dir)
