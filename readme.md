# Classification-Resnet50

**Classification-Resnet50** is a Python-based project for image classification using a pre-trained ResNet50 model with TensorFlow/Keras. It supports training, evaluation, and exporting the model to ONNX format for cross-platform deployment. The project is designed for tasks such as object recognition or custom image classification, leveraging transfer learning for efficient model training on custom datasets.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Testing Environment](#testing-environment)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Exporting Model to ONNX](#exporting-model-to-onnx)
- [Running Inference with ONNX Model](#running-inference-with-onnx-model)
- [Results](#results)
- [File Structure](#file-structure)
- [Notes](#notes)
- [Deliverables](#deliverables)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
The **Classification-Resnet50** project provides a robust framework for image classification using a pre-trained ResNet50 model, fine-tuned for custom datasets. It includes scripts for data preprocessing, model training, evaluation, and exporting to ONNX format for efficient inference across various platforms. The project is optimized for CPU-based environments and is suitable for researchers and developers working on computer vision tasks.

## Features
- **Pre-trained ResNet50**: Utilizes a ResNet50 model pre-trained on ImageNet for robust feature extraction.
- **Transfer Learning**: Fine-tunes the model on custom datasets for specific classification tasks.
- **ONNX Export**: Converts the trained model to ONNX format for cross-platform compatibility.
- **Flexible Dataset Support**: Handles multi-class image datasets organized by class folders.
- **Evaluation Metrics**: Computes accuracy, precision, recall, and other metrics for model performance.
- **Visualization**: Generates plots for training/validation loss and accuracy.

## Testing Environment
The project has been tested in the following environment:
- **OS**: Linux
- **Python**: 3.10
- **CPU**: Intel i5 8th Gen
- **GPU**: None (CPU-based processing)

## Prerequisites
To use this project, ensure you have the following installed:
- Python >= 3.10
- Dependencies listed in `requirements.txt`:
  ```bash
  tensorflow
  onnx
  onnxruntime
  numpy
  opencv-python
  matplotlib
  ```
- A dataset of images organized by class in supported formats (e.g., `.jpg`, `.png`).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Udit0495/Classification-Resnet50.git
   cd Classification-Resnet50
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the required packages directly:
   ```bash
   pip install tensorflow onnx onnxruntime numpy opencv-python matplotlib
   ```

## Dataset Preparation
Prepare your image dataset for classification:
1. **Directory Structure**:
   Organize images into subdirectories, each representing a class:
   ```
   dataset/
   ├── train/
   │   ├── class1/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── class2/
   │   │   ├── image3.jpg
   │   │   ├── image4.jpg
   │   │   └── ...
   │   └── ...
   ├── val/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   └── test/
       ├── class1/
       ├── class2/
       └── ...
   ```
   - `train/`: Training images, split by class.
   - `val/`: Validation images, split by class.
   - `test/`: Test images, split by class.
   - Supported formats include `.jpg`, `.png`, and other common image types.

2. **Data Cleaning**:
   - The dataset should be manually cleaned to ensure high quality (e.g., remove corrupted or irrelevant images).

3. **Update Configuration**:
   - Modify `model_training_onnx_export.ipynb` to specify:
     - Path to the dataset directories (`train/`, `val/`, `test/`).
     - Image size (default: 180x180).
     - Number of classes (based on subdirectories).
     - Hyperparameters (e.g., batch size, learning rate, epochs).

## Model Training
To train the model, execute the Jupyter Notebook:
```bash
jupyter notebook model_training_onnx_export.ipynb
```
The training process includes:
1. **Data Loading & Preprocessing**: Images are resized to 180x180 and normalized.
2. **Model Architecture**: Uses a pre-trained ResNet50 model (or a custom CNN) with a modified classification head.
3. **Training**: Trains using categorical cross-entropy loss and an optimizer (e.g., Adam).
4. **Evaluation**: Validates the model on the test dataset, computing metrics like accuracy.
5. **ONNX Export**: Exports the trained model to ONNX format for inference.

## Exporting Model to ONNX
After training, the model is saved and exported to ONNX format using the following steps in the notebook:
```python
model.save("model.keras")
model = tf.keras.models.load_model("model.keras")
model.export("saved_model")

import subprocess
subprocess.run(["python", "-m", "tf2onnx.convert", "--saved-model", "saved_model", "--output", "model.onnx"])
```
- This generates `model.onnx` in the project directory.

## Running Inference with ONNX Model
To perform inference using the exported ONNX model:
```python
import onnxruntime
import numpy as np
import cv2

# Load ONNX model
session = onnxruntime.InferenceSession("model.onnx")

# Load and preprocess image
image = cv2.imread("test.jpg")
image = cv2.resize(image, (180, 180)).astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# Perform inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image})
print("Predicted class:", np.argmax(output))
```
- Ensure the image is preprocessed to match the training input size (180x180) and normalization.

## Results
The model achieves high accuracy on the test dataset after fine-tuning. Example output for a test image:
```
Image: test.jpg
Predicted Class: class1 (Confidence: 0.94)
```
- Training results, including loss and accuracy curves, are saved as plots in the `results/` directory (e.g., `results/train_val_metrics.png`).
- Evaluation metrics (e.g., accuracy, precision, recall) are computed during training and logged in the notebook output.

## File Structure
```
Classification-Resnet50/
├── model_training_onnx_export.ipynb  # Jupyter Notebook for training and ONNX export
├── main.py                           # Optional script for inference (if implemented)
├── requirements.txt                  # Python dependencies
├── dataset/                          # Directory for input images (train/val/test)
├── models/                           # Directory for saved models
│   ├── model.keras                   # Saved Keras model
│   ├── saved_model/                  # Exported TensorFlow SavedModel
│   ├── model.onnx                    # Exported ONNX model
├── results/                          # Directory for output visualizations (e.g., plots)
├── classes.txt                       # File containing class labels
└── README.md                         # Project documentation
```

## Notes
- **Dataset Quality**: The dataset was manually cleaned to improve model performance and ensure high-quality inputs.
- **ONNX Compatibility**: The ONNX export ensures the model can be deployed in various environments supporting ONNX Runtime.
- **ONNX Runtime**: Ensure ONNX Runtime is installed for efficient inference execution.
- **Flexibility**: The notebook can be adapted for other architectures (e.g., custom CNN) by modifying the model definition.

## Deliverables
- `model_training_onnx_export.ipynb`: Jupyter Notebook for training and ONNX export.
- `model.onnx`: Exported ONNX model for inference.
- `classes.txt`: File containing class labels.
- Example inference script (embedded in the notebook or as `main.py`, if implemented).

## Performance Optimization
- **Data Augmentation**: Apply transformations (e.g., rotation, flipping) in the notebook to improve model robustness.
- **Batch Size**: Adjust batch size in the notebook (e.g., 32) to balance memory usage and training speed on CPU.
- **Early Stopping**: Implement early stopping to prevent overfitting (add to the training loop in the notebook).
- **GPU Support**: For GPU-enabled systems, install TensorFlow with GPU support:
  ```bash
  pip install tensorflow-gpu
  ```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

Please ensure your code follows the project's coding standards and includes appropriate documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [TensorFlow](https://www.tensorflow.org/) and [ONNX](https://onnx.ai/) for deep learning and model export.
- Inspired by image classification projects and ResNet50 implementations.
- Thanks to the open-source community for providing robust tools and pre-trained models.
