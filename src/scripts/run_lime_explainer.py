# scripts/run_lime_explainer.py

"""
Terminal-friendly script for ML model explainability with LIME.

Uses shared utility functions for data loading and model training.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import sys
from typing import List

from utils.data_utils import load_and_split_data
from utils.model_utils import train_model


import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for LIME explainer."""
    parser = argparse.ArgumentParser(
        description="Terminal-friendly ML explainability script using LIME")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./explanations_lime",
        help="Directory to save LIME explanation plots"
    )
    parser.add_argument(
        "--instance-idx",
        type=int,
        default=0,
        help="Index of test instance to explain (0-based)"
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
        help="Number of features to include in LIME explanations"
    )
    return parser.parse_args()


def explain_with_lime(
    X_train: np.ndarray,
    X_test: np.ndarray,
    instance_idx: int,
    model: RandomForestClassifier,
    feature_names: List[str],
    class_names: List[str],
    y_test: np.ndarray,
    num_features: int,
    output_dir: str
) -> None:
    """Generate and save LIME explanations for a test instance."""
    print(f"\n--- Generating LIME Explanation for Instance Index: {instance_idx} ---")

    try:
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
    except ImportError:
        print("\nError: LIME library not found.", file=sys.stderr)
        print("Install LIME using: pip install lime", file=sys.stderr)
        return
    except Exception as e:
        print(f"\nError creating LIME explainer: {e}", file=sys.stderr)
        return

    instance_to_explain = X_test[instance_idx]
    true_label = y_test[instance_idx]
    model_prediction = model.predict(instance_to_explain.reshape(1, -1))[0]
    model_proba = model.predict_proba(instance_to_explain.reshape(1, -1))[0]

    print(f"True Label:         {class_names[true_label]} ({true_label})")
    print(f"Model Prediction:   {class_names[model_prediction]} ({model_prediction})")
    print(f"Model Probability [{class_names[0]}, {class_names[1]}]: [{model_proba[0]:.4f}, {model_proba[1]:.4f}]")

    try:
        explanation_lime = explainer_lime.explain_instance(
            data_row=instance_to_explain,
            predict_fn=model.predict_proba,
            num_features=num_features
        )
    except Exception as e:
        print(f"\nError generating LIME explanation: {e}", file=sys.stderr)
        return

    print(f"\nLIME Explanation (Top {num_features} Features for Predicted Class '{class_names[model_prediction]}'):")
    for feature, weight in explanation_lime.as_list(label=model_prediction):
        print(f"{feature:<25}: {weight:.4f}")

    try:
        fig = explanation_lime.as_pyplot_figure(label=model_prediction)
        plt.title(f'LIME Instance {instance_idx} (Pred: {class_names[model_prediction]}, True: {class_names[true_label]})')
        plt.tight_layout()
        lime_output_path = os.path.join(output_dir, f"lime_explanation_instance_{instance_idx}.png")
        plt.savefig(lime_output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"LIME plot saved to: {lime_output_path}")
    except Exception as e:
        print(f"Error saving LIME plot: {e}", file=sys.stderr)
        plt.close()

def main() -> None:
    """Main function to run the LIME explainability pipeline."""
    print("--- Starting LIME Explainability Script ---")
    args = parse_arguments()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory set to: {os.path.abspath(args.output_dir)}")
    except OSError as e:
        print(f"Error creating output directory '{args.output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    X_train, X_test, y_train, y_test, feature_names, class_names = load_and_split_data()
    print(f"Data loaded: {len(X_train)} train samples, {len(X_test)} test samples.")

    if not (0 <= args.instance_idx < len(X_test)):
        print(f"\nError: Instance index {args.instance_idx} is out of bounds.", file=sys.stderr)
        print(f"Please choose an index between 0 and {len(X_test) - 1}.", file=sys.stderr)
        sys.exit(1)

    model = train_model(X_train, y_train)

    try:
        accuracy = model.score(X_test, y_test)
        print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    except Exception as e:
        print(f"Error evaluating model: {e}", file=sys.stderr)

    explain_with_lime(
        X_train=X_train,
        X_test=X_test,
        instance_idx=args.instance_idx,
        model=model,
        feature_names=feature_names,
        class_names=class_names,
        y_test=y_test,
        num_features=args.num_features,
        output_dir=args.output_dir
    )

    print("\n--- LIME Explanation generation complete! ---")
    print(f"Plot saved to directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        plt.switch_backend('Agg')
    except Exception as e:
        print(f"Could not switch matplotlib backend: {e}", file=sys.stderr)

    main()