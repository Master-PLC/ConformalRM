import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file
from sklearn.model_selection import train_test_split


data2levels = {
    'hs': [0., 1., 2., 3., 4.],
    'ufb': [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.3, 8.5, 9.0, 9.3, 9.5, 9.8, 10.0],
    'saferlhf': [0., 1., 2., 3.],
}


def binarize_labels(labels, data_name):
    """
    Binarize labels based on data2levels.
    
    Args:
        labels: Original continuous labels
        data_name: Name of the dataset
    
    Returns:
        binary_labels: Binarized labels (0 or 1)
    """
    levels = data2levels[data_name]
    median_level = np.mean(levels)
    
    # Convert to binary: 0 for below or equal to median, 1 for above median
    binary_labels = (labels >= median_level).astype(float)
    
    print(f"Binarization threshold (median): {median_level}")
    print(f"Binary distribution: {np.sum(binary_labels == 0)} negative, {np.sum(binary_labels == 1)} positive")
    
    return binary_labels


if __name__ == "__main__":
    parser = ArgumentParser(description="Data preparation and splitting for Conformal RM")
    parser.add_argument("--model_name", type=str, default="FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--data_name", type=str, default="hs")
    parser.add_argument("--data_root", type=str, default="./embeddings/normal")
    parser.add_argument("--output_dir", type=str, default="./embeddings/clean")
    parser.add_argument("--cal_ratio", type=float, default=0.2, help="Ratio of calibration set")
    args = parser.parse_args()

    # Load training data
    data = load_file(f"{args.data_root}/{args.model_name}_{args.data_name}_train.safetensors")
    embeddings = data["embeddings"].float().numpy()
    labels = data["labels"].float().numpy()
    print(f"Total embeddings loaded: {embeddings.shape[0]}")
    print(f"Total labels loaded: {labels.shape[0]}")

    # Filter data where target_label is nan
    embeddings_filtered = embeddings[~np.isnan(labels)]
    target_labels_filtered = labels[~np.isnan(labels)]
    unique_labels = np.unique(target_labels_filtered)
    print(f"Original data size: {embeddings.shape[0]}")
    print(f"Data size after filtering NaN values: {embeddings_filtered.shape[0]}")
    print(f"Unique labels: {unique_labels}")

    # Data split: Train and Calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        embeddings_filtered, target_labels_filtered, 
        test_size=args.cal_ratio, random_state=42
    )
    print(f"\nData Split:")
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")
    print(f"Calibration set size: {X_cal.shape[0]} ({X_cal.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")

    # Binarize labels
    print(f"\nBinarizing training labels...")
    y_train_binary = binarize_labels(y_train, args.data_name)
    print(f"\nBinarizing calibration labels...")
    y_cal_binary = binarize_labels(y_cal, args.data_name)

    # Load and binarize test data
    print(f"\nLoading test data...")
    data_test = load_file(f"{args.data_root}/{args.model_name}_{args.data_name}_test.safetensors")
    X_test = data_test["embeddings"]
    y_test = data_test["labels"].float().numpy()
    print(f"Binarizing test labels...")
    y_test_binary = binarize_labels(y_test, args.data_name)

    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"{args.model_name}_{args.data_name}.safetensors"
    
    save_file({
        # Training set
        "X_train": torch.from_numpy(X_train),
        "y_train": torch.from_numpy(y_train),
        "y_train_binary": torch.from_numpy(y_train_binary),
        # Calibration set (renamed from val)
        "X_cal": torch.from_numpy(X_cal),
        "y_cal": torch.from_numpy(y_cal),
        "y_cal_binary": torch.from_numpy(y_cal_binary),
        # Test set
        "X_test": X_test,
        "y_test": torch.from_numpy(y_test),
        "y_test_binary": torch.from_numpy(y_test_binary),
    }, f"{args.output_dir}/{output_filename}")
    
    print(f"\nSaved processed data to {args.output_dir}/{output_filename}")
    print(f"\nData keys saved:")
    print(f"  - Training: X_train, y_train, y_train_binary")
    print(f"  - Calibration: X_cal, y_cal, y_cal_binary")
    print(f"  - Test: X_test, y_test, y_test_binary")
