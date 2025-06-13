#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data preprocessing
def preprocess_data(train_path, val_path, test_path):
    """Load and preprocess network traffic data."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Assuming 'label' is the target column, rest are features
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_val = val_df.drop('label', axis=1)
    y_val = val_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    # Encode categorical features if any (simplified here)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Reshape for CNN (assuming 1D sequence data)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, le

# Build CNN model
def build_model(input_shape):
    """Define the CNN architecture."""
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Assuming binary classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training function
def train_model(train_path, val_path, test_path, epochs, batch_size, model_path):
    """Train the IDS model."""
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = preprocess_data(train_path, val_path, test_path)
    
    model = build_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

# Testing function
def test_model(test_path, model_path):
    """Test the trained model on new data."""
    model = load_model(model_path)
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    
    loss, acc = model.evaluate(X_test, y_test)
    logging.info(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Intrusion Detection System with Deep Learning")
    parser.add_argument('--train', help="Path to training CSV")
    parser.add_argument('--val', help="Path to validation CSV")
    parser.add_argument('--test', help="Path to testing CSV")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--model', default='model.h5', help="Path to save/load model")
    parser.add_argument('--load_model', action='store_true', help="Test with pre-trained model")
    
    args = parser.parse_args()
    
    if args.train and args.val and args.test and not args.load_model:
        train_model(args.train, args.val, args.test, args.epochs, args.batch_size, args.model)
    elif args.test and args.load_model:
        test_model(args.test, args.model)
    else:
        logging.error("Provide --train, --val, --test for training or --test and --load_model for testing.")

if __name__ == "__main__":
    main()
