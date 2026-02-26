from ultralytics import YOLO
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

model = YOLO(cfg["base_model"])

results = model.train(
    data="data/dataset/data.yaml",
    epochs=30,
    imgsz=cfg["detection"]["imgsz"],
    batch=8,
    name="surgical_instrument_detector",
    project="models",
    device="cpu",
    patience=10,
    save=True
)

print("Training complete. Model saved to models/")