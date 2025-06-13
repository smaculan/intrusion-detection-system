# Intrusion Detection System with Deep Learning

## Description
This project implements an IDS using a CNN in TensorFlow to classify network traffic as normal or malicious, demonstrating deep learning in cybersecurity.

## Prerequisites
- Python 3.8+
- TensorFlow 2.x

## Installation
1. **Clone the Repository**: Extract this ZIP file.
2. **Install Dependencies**: Run `pip install -r requirements.txt`.
3. **Prepare Dataset**: Download NSL-KDD (or similar) and place CSV files in `data/`.

## Usage
### Training Mode
Train the model:
```bash
python src/ids_deep_learning.py --train data/train.csv --val data/val.csv --test data/test.csv --epochs 50 --batch_size 32 --model model.h5
