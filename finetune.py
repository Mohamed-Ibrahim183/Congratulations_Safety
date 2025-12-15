import os
from ultralytics import YOLO

models = {
    "yolo11n": "models/best11n.pt",
    "yolo11x": "models/best11x.pt",
    "yolov8n": "models/bestv8n.pt",
    "yolov8x": "models/bestv8x.pt",
}

# Fine-tune each model
for name, path in models.items():
    # Load the trained model
    model = YOLO(path)
    name = f"{name}_finetuned"

    # Fine-tune on new dataset
    results = model.train(
        data="data1.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name=name,
        patience=10,
    )
