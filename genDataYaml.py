# make this code to generate data.yaml file for yolov8 training and save it as data.yaml
import yaml

data = {
    "train": "path/to/train/images",  # path to training images
    "val": "path/to/val/images",  # path to validation images
    "test": "path/to/test/images",  # path to test images
    "nc": 11,  # number of classes
    "names": [
        "Helmet",
        "Gloves",
        "Vest",
        "Boots",
        "Goggles",
        "none",
        "Person",
        "no_helmet",
        "no_goggle",
        "no_gloves",
        "no_boots",
    ],  # list of class names
}

with open("data.yaml", "w") as f:
    yaml.dump(data, f)
