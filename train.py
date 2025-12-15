from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt", pretrained=True)

# Train the model
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo11x_custom1PPE",
    workers=0,
    device="0,1",
    patience=10,
)
