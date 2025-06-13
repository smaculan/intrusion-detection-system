# Model Architecture for IDS with Deep Learning

This document details the CNN architecture used for intrusion detection.

## Architecture
- **Input Layer**: Shape `(num_features, 1)` where `num_features` depends on the dataset.
- **Conv1D (32 filters, kernel size 3)**: Extracts local patterns from traffic data.
- **MaxPooling1D (pool size 2)**: Reduces dimensionality.
- **Conv1D (64 filters, kernel size 3)**: Captures higher-level features.
- **MaxPooling1D (pool size 2)**: Further reduces size.
- **Flatten**: Converts to 1D vector.
- **Dense (128 units, ReLU)**: Fully connected layer for decision-making.
- **Output Layer (2 units, softmax)**: Binary classification (normal or malicious).

## Training
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: EarlyStopping (patience=5) to prevent overfitting.

## Rationale
- CNNs excel at detecting spatial patterns in sequential data, suitable for network traffic.
- Binary classification simplifies the task for demonstration; multi-class is possible with dataset adjustments.
