# utils/data_utils.py

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple

def load_and_split_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                   List[str], List[str]]:
    """
    Load the breast cancer dataset and split it into train and test sets.

    Returns:
        Tuple containing:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Test labels
            feature_names (List[str]): Names of features
            class_names (List[str]): Names of classes ['malignant', 'benign']
    """
    print("Loading Breast Cancer dataset...")
    try:
        cancer = sklearn.datasets.load_breast_cancer()
    except Exception as e:
        print(f"Error loading scikit-learn dataset: {e}", file=sys.stderr)
        print("Please ensure scikit-learn is installed correctly.", file=sys.stderr)
        sys.exit(1)

    X, y = cancer.data, cancer.target
    feature_names = list(cancer.feature_names) # Ensure it's a list
    class_names = list(cancer.target_names)  # Ensure it's a list

    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, feature_names, class_names