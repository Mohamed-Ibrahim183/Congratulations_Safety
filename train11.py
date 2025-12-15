from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="/kaggle/working/data.yaml",
    epochs=100,
    imgsz=640,
    batch=128,
    name="yolo11x_custom3PPE",
    workers=0,
    device="0,1",
    patience=15,
)
