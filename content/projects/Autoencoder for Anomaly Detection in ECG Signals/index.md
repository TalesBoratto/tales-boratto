---
title: Autoencoder for Anomaly Detection in ECG Signals
# date: 2023-10-26
summary: Detecting anomalies in electrocardiogram signals using an autoencoder and reconstruction error.
tags:
  - Machine Learning
  - Anomaly Detection
  - Autoencoder
  - TensorFlow
featured: true
draft: false
---

## Introduction

This project explores anomaly detection in electrocardiogram (ECG)
signals using an autoencoder and reconstruction error. The objective
is to investigate whether a model trained on normal signals can
identify patterns that differ from those observed during training.

Rather than learning a direct mapping from signals to class labels,
the autoencoder learns to reconstruct normal ECG patterns. The
reconstruction error is then used as an anomaly score: signals with
errors above a selected threshold are flagged as potentially anomalous.

The workflow covers data exploration, preprocessing, model training,
threshold selection, and performance evaluation. Available labels
are used to identify normal training samples and evaluate the
resulting anomaly predictions.

This project is an experimental machine learning study and is not
intended for clinical diagnosis.

### Imports

The following libraries support data manipulation, visualization,
model development, and performance evaluation.

'''python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
'''

## Dataset

This project uses a simplified, binary-labeled version of ECG5000,
distributed through the TensorFlow autoencoder tutorial. ECG5000
contains approximately 5,000 univariate heartbeat sequences, each
represented by 140 time points.

While the original dataset contains five heartbeat classes, the
CSV used here already provides binary labels: `1` for normal
heartbeats and `0` for anomalous heartbeats. Each row contains
140 signal measurements followed by its corresponding label.

The task is formulated as anomaly detection. Normal samples are
selected for autoencoder training, while both normal and anomalous
samples are included in the evaluation. Labels are used to identify
normal training samples and assess detection performance rather
than as reconstruction targets.

### Load Dataset

The dataset is loaded directly from the TensorFlow data repository.
Because the CSV does not contain a header row, `header=None`
ensures that the first sample is read as data.

```python
dataset_url = (
    "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
)

dataframe = pd.read_csv(dataset_url, header=None)
```

The last column contains the binary labels. The remaining columns
contain the ECG measurements.

```python
raw_data = dataframe.values

# The last column contains the labels: 1 = normal, 0 = anomalous.
labels = raw_data[:, -1]

# The remaining columns contain the ECG signal measurements.
signals = raw_data[:, :-1]
```

### Initial Data Inspection

Before preprocessing, the dataset dimensions, missing values,
and class distribution are inspected.

```python
print(f"Number of samples: {signals.shape[0]}")
print(f"Time points per signal: {signals.shape[1]}")
print(f"Missing values: {dataframe.isna().sum().sum()}")

class_counts = (
    pd.Series(labels, name="label")
    .value_counts()
    .sort_index()
    .rename(index={0: "Anomalous", 1: "Normal"})
)

print("\nClass distribution:")
print(class_counts.to_string())
```
```text
Number of samples: 4998
Time points per signal: 140
Missing values: 0

Class distribution:
label
Anomalous    2079
Normal       2919
```

### Train–Test Split

The signals are divided into training and test sets, using 80% of the
samples for training and 20% for testing. A fixed random seed makes
the split reproducible, while stratification approximately preserves
the proportion of normal and anomalous samples in both subsets.

The test set is reserved for the final performance evaluation.
Although the initial training partition contains both classes,
only normal samples are subsequently used to train the autoencoder.

```python
train_data, test_data, train_labels, test_labels = train_test_split(
    signals,
    labels,
    test_size=0.2,
    random_state=21,
    stratify=labels,
)

print(f"Training samples: {train_data.shape[0]}")
print(f"Test samples: {test_data.shape[0]}")
```
```text
Training samples: 3998
Test samples: 1000
```

### Signal Normalization

A global min–max transformation is applied using the minimum and
maximum values observed across all signals in the training partition.

These training-derived parameters map the training values to the
[0, 1] interval and are reused unchanged to transform the test set.
The transformation therefore does not use test-set statistics.

A single pair of scaling parameters is used for all signals, rather
than normalizing each heartbeat independently. Test values outside
the training range may consequently fall below 0 or above 1.

In this implementation, scaling parameters are estimated from the
full training partition, including both classes. Normal samples
are selected afterward for autoencoder training.

```python
# Estimate global scaling parameters from the training partition only.
min_val = float(np.min(train_data))
max_val = float(np.max(train_data))
scale_range = max_val - min_val

# Apply the same transformation to training and test signals.
train_data = tf.convert_to_tensor(
    (train_data - min_val) / scale_range,
    dtype=tf.float32,
)

test_data = tf.convert_to_tensor(
    (test_data - min_val) / scale_range,
    dtype=tf.float32,
)

print(f"Training minimum before scaling: {min_val:.4f}")
print(f"Training maximum before scaling: {max_val:.4f}")

print(
    "Normalized training range: "
    f"[{tf.reduce_min(train_data).numpy():.4f}, "
    f"{tf.reduce_max(train_data).numpy():.4f}]"
)

print(
    "Normalized test range: "
    f"[{tf.reduce_min(test_data).numpy():.4f}, "
    f"{tf.reduce_max(test_data).numpy():.4f}]"
)
```
```text
Training minimum before scaling: -7.0904
Training maximum before scaling: 7.4021
Normalized training range: [0.0000, 1.0000]
Normalized test range: [0.0559, 0.8319]
```

### Selecting Normal Training Samples

Normal heartbeats are labeled as `1`, while anomalous heartbeats
are labeled as `0`. Boolean masks are used to separate the classes
within each partition.

Only normal training signals are used to optimize the autoencoder.
The test signals from both classes are retained to evaluate whether
reconstruction error can distinguish normal from anomalous patterns.

```python
normal_train_mask = train_labels == 1
normal_test_mask = test_labels == 1

normal_train_data = tf.boolean_mask(train_data, normal_train_mask)
anomalous_train_data = tf.boolean_mask(train_data, ~normal_train_mask)

normal_test_data = tf.boolean_mask(test_data, normal_test_mask)
anomalous_test_data = tf.boolean_mask(test_data, ~normal_test_mask)

print(f"Normal training samples: {normal_train_data.shape[0]}")
print(f"Anomalous training samples: {anomalous_train_data.shape[0]}")
print(f"Normal test samples: {normal_test_data.shape[0]}")
print(f"Anomalous test samples: {anomalous_test_data.shape[0]}")
```
```text
Normal training samples: 2335
Anomalous training samples: 1663
Normal test samples: 584
Anomalous test samples: 416
```

### Normal and Anomalous ECG Examples

One normal and one anomalous training signal are shown below.
Both panels use the same axis limits to support a direct visual
comparison of waveform shape and normalized amplitude.

These are individual examples and should not be interpreted as
representative of every signal within their respective classes.

```python
time_points = np.arange(train_data.shape[1])

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

examples = [
    (normal_train_data[0].numpy(), "Normal heartbeat", "#2563EB"),
    (anomalous_train_data[0].numpy(), "Anomalous heartbeat", "#DC2626"),
]

for ax, (signal, title, color) in zip(axes, examples):
    ax.plot(time_points, signal, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Time-point index")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Normalized amplitude")

plt.show()
```
![Normal and anomalous ECG signals plotted side by side](normal-vs-anomalous-ecg.png)

These individual examples illustrate waveform differences, but do not
represent the full variability within each class.

## Building the Model

A fully connected autoencoder is used to learn a compact representation
of normal ECG signals. The encoder progressively reduces each
140-point sequence to an eight-dimensional latent representation,
while the decoder reconstructs the signal from this representation.

The encoder uses dense layers with 32, 16, and 8 units. The decoder
mirrors this expansion with 16 and 32 units, followed by an output
layer with 140 units. Hidden layers use ReLU activation, while the
output layer uses a sigmoid activation to produce values between
0 and 1.

The model is optimized using Adam and mean absolute error (MAE),
which measures the average absolute difference between each input
signal and its reconstruction.

```python
class AnomalyDetector(Model):
    def __init__(self):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(140,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(8,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid"),
        ])

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        return self.decoder(encoded, training=training)


tf.keras.utils.set_random_seed(21)

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer="adam", loss="mae")
```

## Train the Autoencoder

Only normal ECG signals are used to train the autoencoder. Each
signal serves as both the input and the reconstruction target.

The normal training samples are further divided into fitting and
validation subsets. The validation subset monitors reconstruction
performance on normal signals not used for gradient updates,
while the test set remains reserved for subsequent evaluation.

The model is trained for 20 epochs with a batch size of 512.
Training samples are shuffled between epochs.

```python
normal_fit_data, normal_val_data = train_test_split(
    normal_train_data.numpy(),
    test_size=0.2,
    random_state=21,
)

history = autoencoder.fit(
    normal_fit_data,
    normal_fit_data,
    epochs=20,
    batch_size=512,
    validation_data=(normal_val_data, normal_val_data),
    shuffle=True,
    verbose=1,
)
```

### Training and Validation Loss

Training and validation MAE are tracked across epochs to inspect
optimization progress and the difference between fitting and
held-out reconstruction performance.

Both curves describe reconstruction of normal signals. They do not,
by themselves, measure anomaly detection performance.

```python
epochs = np.arange(1, len(history.history["loss"]) + 1)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    epochs,
    history.history["loss"],
    label="Training MAE",
    linewidth=2,
)

ax.plot(
    epochs,
    history.history["val_loss"],
    label="Validation MAE",
    linewidth=2,
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean absolute reconstruction error")
ax.set_title("Autoencoder Training")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("training-loss.png", dpi=300, bbox_inches="tight")
plt.show()
```
![Training Loss](training-loss.png)


