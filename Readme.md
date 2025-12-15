# Construction Site Safety PPE Detection

## Description

This project implements a real-time Personal Protective Equipment (PPE) detection system using YOLO (You Only Look Once) object detection model. The system is designed to identify safety compliance on construction sites by detecting various PPE items such as helmets, gloves, vests, boots, and goggles, as well as identifying violations when these items are missing.

The project includes:

- A Streamlit web dashboard for interactive detection
- Model training scripts for custom YOLO models
- Prediction scripts for video and image analysis
- Data generation utilities for preparing training datasets

## Features

- **Real-time Detection**: Detect PPE items and safety violations in real-time
- **Web Dashboard**: User-friendly Streamlit interface for uploading and analyzing images/videos
- **Custom Training**: Scripts to train YOLO models on custom datasets
- **Video Analysis**: Process video files for continuous monitoring
- **Data Preparation**: Utilities to generate YAML configuration files for training

## Classes Detected

The model detects the following classes:

- Helmet
- Gloves
- Vest
- Boots
- Goggles
- Person
- no_helmet (violation)
- no_goggle (violation)
- no_gloves (violation)
- no_boots (violation)
- none

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Congratulations_Safety
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained YOLO model weights (if not included):
   - Place your trained model weights in the `models/` directory as `best11x.pt`

## Usage

### Running the Web Dashboard

To start the Streamlit dashboard:

```bash
streamlit run app.py
```

This will launch a web interface where you can:

- Upload images or videos for PPE detection
- View detection results with bounding boxes
- Analyze safety compliance statistics

### Training a Custom Model

To train the YOLO model on your dataset:

1. Prepare your dataset with images and annotations
2. Update the paths in `data.yaml` (or use `genDataYaml.py` to generate it)
3. Run the training script:

```bash
python train.py
```

### Making Predictions

To run predictions on a video file:

```bash
python predict.py
```

Make sure to update the `source` path in `predict.py` to point to your video file.

## Data Preparation

Use `genDataYaml.py` to generate the data configuration file for YOLO training:

```bash
python genDataYaml.py
```

This creates a `data.yaml` file with the necessary paths and class information.

## Requirements

- Python 3.8+
- Streamlit
- Ultralytics YOLO
- PIL (Pillow)
- NumPy

See `requirements.txt` for exact versions.

## Project Structure

```
├── app.py                 # Streamlit web dashboard
├── train.py               # Model training script
├── predict.py             # Prediction script
├── finetune.py            # Fine-tuning script
├── genDataYaml.py         # Data YAML generator
├── data.yaml              # Dataset configuration
├── requirements.txt       # Python dependencies
├── test/                  # Test files
└── models/                # Trained model weights (create this directory)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
