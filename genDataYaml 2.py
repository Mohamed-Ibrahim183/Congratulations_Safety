# make this code to generate data.yaml file for yolov8 training and save it as data.yaml
import yaml

data = {
    "train": "/kaggle/input/construction-site-safety-image-dataset-roboflow/css-data/train",  # path to training images
    "val": "/kaggle/input/construction-site-safety-image-dataset-roboflow/css-data/valid",  # path to validation images
    "test": "/kaggle/input/construction-site-safety-image-dataset-roboflow/css-data/test",  # path to test images
    "nc": 10,  # number of classes
    "names": [
        "Hardhat",
        "Mask",
        "NO-Hardhat",
        "NO-Mask",
        "NO-Safety Vest",
        "Person",
        "Safety Cone",
        "Safety Vest",
        "machinery",
        "vehicle",
    ],  # list of class names
}

with open("data.yaml", "w") as f:
    yaml.dump(data, f)
