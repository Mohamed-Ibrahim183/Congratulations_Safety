from ultralytics import YOLO

model = YOLO("./models/best11x.pt")  # path to your trained model
results = model.predict(
    source="./Construction Site Safety Basics.mp4",
    show=True,
    save=True,
    conf=0.6,
    line_width=2,
    show_labels=True,
    show_conf=True,
)
