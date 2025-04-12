# utils/model_utils.py

import sklearn
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier  # Keep if you might use XGBoost later
import numpy as np
import sys

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a RandomForest model on the given training data.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        RandomForestClassifier: Trained model
    """
    print("Training RandomForestClassifier model...")
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Alternative: XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error during model training: {e}", file=sys.stderr)
        sys.exit(1)