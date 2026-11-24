# Handwritten Digit Recognition CNN (MNIST)

A high-performance, lightweight Convolutional Neural Network (CNN) built with **TensorFlow / Keras** for classifying handwritten digits from the **MNIST** dataset.

---

## 📌 Architecture Overview

The network is engineered specifically with **3 Convolutional layers** for hierarchical spatial feature extraction followed by **2 Fully Connected (Dense) layers** for high-level reasoning and classification:

```
[Input: 28x28x1]
       │
       ▼
[Conv2D: 32 filters, 3x3, ReLU]  ──> Feature Map: 26x26x32  (Params: 320)
       │
[MaxPooling2D: 2x2]              ──> Feature Map: 13x13x32
       │
       ▼
[Conv2D: 64 filters, 3x3, ReLU]  ──> Feature Map: 11x11x64  (Params: 18,496)
       │
[MaxPooling2D: 2x2]              ──> Feature Map: 5x5x64
       │
       ▼
[Conv2D: 64 filters, 3x3, ReLU]  ──> Feature Map: 3x3x64    (Params: 36,928)
       │
       ▼
[Flatten]                        ──> Vector: 576
       │
       ▼
[Dense: 64 units, ReLU]          ──> Feature Representation (Params: 36,928)
       │
[Dropout: 0.25]                  ──> Regularization against co-adaptation
       │
       ▼
[Dense: 10 units, Softmax]       ──> Class Probabilities 0-9 (Params: 650)
```

### Layer Summary Table

| # | Layer Name | Layer Type | Output Shape | Parameters | Function / Purpose |
|---|---|---|---|---|---|
| 1 | `conv_layer_1` | Conv2D (3x3) | `(None, 26, 26, 32)` | 320 | Low-level feature extraction (edges, strokes, gradients) |
| - | `maxpool_1` | MaxPooling2D (2x2) | `(None, 13, 13, 32)` | 0 | Downsampling & spatial translation invariance |
| 2 | `conv_layer_2` | Conv2D (3x3) | `(None, 11, 11, 64)` | 18,496 | Mid-level feature extraction (curves, intersections, angles) |
| - | `maxpool_2` | MaxPooling2D (2x2) | `(None, 5, 5, 64)` | 0 | Downsampling & receptive field expansion |
| 3 | `conv_layer_3` | Conv2D (3x3) | `(None, 3, 3, 64)` | 36,928 | High-level spatial digit parts (loops, stems, digit topology) |
| - | `flatten` | Flatten | `(None, 576)` | 0 | Unrolls 3D feature tensor into 1D vector |
| 4 | `fc_layer_1` | Dense (ReLU) | `(None, 64)` | 36,928 | **Fully Connected Layer 1**: non-linear combination of spatial features |
| - | `dropout` | Dropout (0.25) | `(None, 64)` | 0 | Prevents overfitting and feature co-dependency |
| 5 | `fc_layer_2_output` | Dense (Softmax) | `(None, 10)` | 650 | **Fully Connected Layer 2**: Normalized class probabilities for digits 0-9 |

- **Total Parameters:** `93,322` (~364.5 KB memory footprint)
- **Trainable Parameters:** `93,322`

---

## ⚡ Model Capabilities & Power

### 1. Near-Human Accuracy
- **Test Accuracy:** **`98.90%`** evaluated on 10,000 completely unseen images.
- **Validation Accuracy:** **`98.32%`** on a dedicated 10,000-sample validation split.
- **Test Loss:** `0.0373` (Sparse Categorical Crossentropy).

### 2. High Parameter Efficiency
- With only **93,322 parameters**, this model achieves state-of-the-art grade digit classification performance while remaining under **400 KB** in size.
- Avoids over-parameterization, making training virtually instantaneous even on modest consumer CPUs without requiring dedicated GPUs.

### 3. Spatial Translation & Distortion Invariance
- Combining three convolutions with pooling enables the model to recognize digits regardless of:
  - Slight shifts / centering variations.
  - Variations in stroke thickness and slant.
  - Individual handwriting styles (loops vs. crossed digits like `7` or `0`).

### 4. Low Latency & Edge / Mobile Deployment Ready
- Average CPU inference latency: **< 1 millisecond** per image.
- Easily exportable to **TensorFlow Lite (`.tflite`)**, **ONNX**, or **TF.js** for deployment on microcontrollers, Raspberry Pi, mobile devices, and browser applications.

---

## 📊 Dataset Partitioning & Validation Strategy

The original MNIST dataset (70,000 images total) was rigorously partitioned to prevent data leakage and provide true validation tracking:

- **Training Set:** `50,000` samples (used strictly for gradient updates)
- **Validation Set:** `10,000` samples (used strictly to monitor generalization during epochs)
- **Test Set:** `10,000` samples (used strictly for final unbiased benchmark evaluation)

---

## 📁 Repository Structure

```
├── index.html            # Web Studio: Touchpad canvas, multi-digit detector & neural visualizer
├── app.py                # Web backend server (serves web UI and CNN inference API)
├── segmentation.py       # Multi-digit bounding-box extraction & MNIST centering engine
├── aidemo.py             # Script to inspect and visualize raw dataset samples
├── train_cnn.py          # Script to build, train, validate, and evaluate the CNN
├── predict.py            # CLI script to predict single or multiple digits from any image
├── sample_digit.png      # Single digit test sample
├── multi_digits_sample.png# Multi-digit test sample (4, 2, 7)
├── mnist_cnn.keras       # Serialized trained model weights and architecture
├── mnist_samples.png     # Visual sample of MNIST dataset digits
├── training_curves.png   # Training vs. Validation Loss and Accuracy curves
├── predictions_sample.png# Model test predictions with ground-truth validation
└── README.md             # Model documentation and capability specification
```

---

## 🚀 How to Run

### 1. Launch Interactive Web Studio (Touchpad Canvas + Neural Visualizer)
```bash
python3 app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser:
- **Touchpad/Mouse drawing pad** supporting multi-digit input with real-time bounding boxes.
- **Neural Brain Scan tab** showing live filter activations across all 3 Conv layers, FC1, and Softmax output.
- **Model Capabilities tab** with architectural breakdowns and performance benchmarks.
- **Surprise Features**:
  1. *AI Voice Readout*: Speaks the detected sequence and confidence aloud.
  2. *Smart Math & Number Insights*: Instant prime detection, parity, sum/product of digits, hex, and binary conversions.
  3. *Adversarial Noise Slider*: Inject real-time sensor noise into your drawings to test CNN robustness!

### 2. Run Multi-Digit Prediction from CLI
```bash
# Predict single or multiple numbers from any image file (supports spaces in path)
python3 predict.py multi_digits_sample.png
python3 predict.py image.png
```

### 3. Visualize Raw Dataset Samples
```bash
python3 aidemo.py
```

### 4. Train the Model from Scratch
```bash
python3 train_cnn.py
```
```python
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("mnist_cnn.keras")

# Run inference on any (28, 28) or (N, 28, 28, 1) normalized image
# test_images normalized to [0, 1]
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
```
