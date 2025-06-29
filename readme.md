# Image Classification Model

This repository contains a deep learning-based image classification model, including scripts for training, exporting to ONNX, and performing inference.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install tensorflow onnx onnxruntime numpy opencv-python matplotlib
```

## Model Training

To train the model, execute the Jupyter Notebook `model_training_onnx_export.ipynb`. The training process includes:

1. **Data Loading & Preprocessing**: Images are resized and normalized.
2. **Model Architecture**: A deep learning model (ResNet or a custom CNN) is used.
3. **Training**: The model is trained using categorical cross-entropy loss.
4. **Evaluation**: The model is validated against a test dataset.
5. **ONNX Export**: The trained model is converted to ONNX format.

## Exporting Model to ONNX

After training, the model is saved and exported using the following steps:

```model.save("model.keras")
model = tf.keras.models.load_model("model.keras")
model.export("saved_model")

import subprocess

subprocess.run(["python", "-m", "tf2onnx.convert", "--saved-model", "saved_model", "--output", "model.onnx"])
```

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

## Notes
- The dataset used was cleaned manually to improve quality.
- ONNX export ensures compatibility with various deployment environments.
- Ensure ONNX Runtime is installed for efficient inference execution.

## Deliverables
- `model_training_onnx_export.ipynb` (Training and ONNX export script)
- `model.onnx` (Exported model for inference)
- `classes.txt` (Class labels)
- Example inference script


