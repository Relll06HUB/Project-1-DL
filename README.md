# Vehicle Detection Using YOLOv10 and YOLOv11

## Project Overview
This project aims to detect large vehicles (cars, buses, and trucks) on the road as potential traffic hazards using Deep Learning-based Object Detection methods.

Two models are implemented and compared:
- YOLOv10  
- YOLOv11  

The primary goal is to analyze the performance of both models based on overall detection quality and computational efficiency.

## Objectives
- Implement YOLOv10 and YOLOv11 for object detection  
- Compare model performance using relevant evaluation metrics  
- Improve model performance through fine-tuning  

## Dataset
- **Dataset:** Traffic Vehicles Object Detection (Kaggle)  
- **Classes used:** `car`, `bus`, `truck`  
- **Total images after preprocessing:** 1,784  
- **Dataset Access Links:** [Kaggle Dataset](https://kaggle.com) or via [Google Drive](https://google.com) (`TrafficDataset` & `filtered_dataset` folders)

## How to Use the Dataset
1. Download the dataset from either of the links provided above.
2. Perform preprocessing as described in the **Data Preprocessing** section (filtering and oversampling).  
3. Place the preprocessed dataset into the following directory structure:

```text
project/
│── filtered_dataset/

filtered_dataset/
│── images/
│   ├── train/
│   ├── val/
│   └── test/
│── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

4. Ensure the `data.yaml` file points to the `filtered_dataset` folder.

**Notes:** 
- The dataset used for training is the preprocessed data stored inside the `filtered_dataset` folder.
- The dataset shown in the demo video is the same dataset used during model training.
- Access links and setup instructions are included to improve project reproducibility.
- The README has been updated to improve documentation clarity.
- Documentation updates do not affect the dataset, trained models, or experimental results shown in the demo video.
- Annotations have been converted to standard YOLO format (`.txt`).

## Data Preprocessing
- Filtered only relevant vehicle classes (`car`, `bus`, `truck`)  
- Applied *oversampling* to address class imbalance  
- Data split:
  - Training: 70%  
  - Validation: 20%  
  - Testing: 10%  

## Models Used
- YOLOv10 (Baseline)  
- YOLOv11 (Baseline)  
- YOLOv11 (Fine-Tuned)  

## Training Configuration
- **Epochs:** 40 (*baseline*), 50 (*fine-tuned*)  
- **Image size:** 640x640  
- **Batch size:** 4  
- **Hardware:** NVIDIA GTX 1650 Ti  
- **Framework:** Ultralytics YOLO  
- **Random seed:** 42  

## Fine-Tuning Strategy

For the YOLOv11 model, fine-tuning was conducted to boost detection performance. The adjustments included:

- Tuning the learning rate to accelerate model convergence  
- Increasing the number of training epochs (from 40 to 50)  
- Applying partial layer freezing to preserve foundational features from the pre-trained backbone  

This approach improves the model's ability to recognize vehicle classes on the preprocessed dataset and enhances generalization across the filtered data distribution.

## Evaluation Results

| Model | Precision | Recall | mAP@50 | mAP@50-95 | FPS |
|------|----------|--------|--------|-----------|-----|
| YOLOv10 | 0.7637 | 0.8864 | 0.8894 | 0.5864 | 60.89 |
| YOLOv11 | 0.7593 | 0.9232 | 0.8961 | 0.5903 | 59.43 |
| YOLOv11 (Fine-Tuned) | 0.7479 | 0.9316 | 0.9032 | 0.5933 | 60.83 |

**Evaluation Summary:** 
The **YOLOv11 (Fine-Tuned)** model achieved the highest overall detection quality (highest Recall, mAP@50, and mAP@50-95). Meanwhile, the **YOLOv10** model delivered the fastest inference speed (highest FPS among all tested models). These results highlight the trade-off between detection accuracy and real-time inference speed.

## Sample Detection Results

The sample image below illustrates vehicle detection results across YOLOv10, YOLOv11, and YOLOv11 (fine-tuned with adjusted learning rate, epochs, and partial layer freezing), accurately identifying cars, buses, and trucks with bounding boxes.

![Detection Results](results/Sample_Detection.jpg)

## Training Logs

Below are the training performance curves for each model:

### YOLOv10 Baseline
![YOLOv10](results/train10_results.png)

### YOLOv11 Baseline
![YOLOv11](results/train11_results.png)

### YOLOv11 Fine-Tuned
![YOLOv11 FT](results/train11_finetuned_results.png)

**Note:** Experiments were run across multiple scenarios—baseline (YOLOv10 and YOLOv11) and fine-tuned YOLOv11 with adjusted hyperparameters (learning rate, epochs, and partial layer freezing). Training generated performance curves (loss, precision, recall, and mAP) used to analyze model convergence.

## Key Findings
- YOLOv11 delivers stronger overall detection quality (higher Recall and mAP)  
- YOLOv10 achieves slightly faster inference speed (higher FPS)  
- Fine-tuning measurably improves YOLOv11 detection metrics  

## Model Weights
Trained weights are available in the `/weights/` directory:
- `YOLOv10_baseline_best.pt`   
- `YOLOv11_baseline_best.pt`  
- `YOLOv11_finetuned_best.pt`  

## Environment Setup

This project was tested using the following environment:
- **Python:** 3.10  
- **OS:** Windows  
- **GPU:** NVIDIA GTX 1650 Ti (optional)

**Dependencies:**
Developed with the following library versions:
- `ultralytics` 8.4.33
- `torch` 2.11.0
- `torchvision` 0.26.0
- `numpy` 2.2.6
- `opencv-python` 4.13.0.92
- `matplotlib` 3.10.8
- `PyYAML` 6.0.3

## How to Run

Follow these steps to run the project locally:

1. **Clone the repository:**
```bash
git clone https://github.com
cd Project-1-DL
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Alternative (Optional):
```bash
pip install ultralytics
```

3. **Run inference via CLI:**
```bash
yolo task=detect mode=predict model=weights/YOLOv11_finetuned_best.pt source=images/Test1.jpg
```

4. **Run inference via Python:**
```python
from ultralytics import YOLO

model = YOLO("weights/YOLOv11_finetuned_best.pt")
results = model("images/Test1.jpg", save=True)
```

**Notes:**
- Detection outputs are automatically saved inside the `runs/detect/` folder.
- You can replace the input `source` path with any custom image.
- To run batch inference on multiple images at once, pass a list of file paths:

```python
results = model(["images/Test1.jpg", "images/Test2.jpg"], save=True) 
```

## License
- This project was developed for academic purposes and is not intended for commercial use.
- The dataset follows the license terms of its original source (Kaggle).

## Demo Video

Watch a demonstration of the model inference pipeline here:  
[View Demo Video on Google Drive](https://google.com)

## Project Structure

```text
project/
│── weights/            # Trained model weights
│   ├── YOLOv10_baseline_best.pt
│   ├── YOLOv11_baseline_best.pt
│   └── YOLOv11_finetuned_best.pt
│
│── results/            # Sample detections & training logs
│   ├── Sample_Detection.jpg
│   ├── train10_results.png
│   ├── train11_results.png
│   └── train11_finetuned_results.png
│
│── runs/               # Training & inference outputs (excluded from repo)
│── Project1.ipynb      # Main Jupyter Notebook
│── data.yaml           # Dataset configuration
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation
```

**Notes:**
- The `results/` directory contains sample detection visualizations and training metric plots.
- The `runs/` directory is auto-generated during training and inference runs.
- Successful inference outputs are saved in `runs/detect/` for visual inspection.

