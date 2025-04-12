# scripts/run_shap_explainer.py

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from utils.data_utils import load_and_split_data
from utils.model_utils import train_model

# Specific imports for this script
import numpy as np
import sklearn
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for SHAP explainer script."""
    parser = argparse.ArgumentParser(
        description="Terminal-friendly ML explainability script using SHAP")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./explanations_shap", 
        help="Directory to save SHAP explanation plots"
    )
    parser.add_argument(
        "--instance-idx",
        type=int,
        default=0,
        help="Index of test instance for local SHAP explanation (0-based)"
    )
    parser.add_argument(
        "--max-display", 
        type=int,
        default=15,
        help="Max features to display in SHAP summary plots"
    )

    return parser.parse_args()



def explain_with_shap(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    instance_idx: int,
    feature_names: List[str],
    class_names: List[str],
    output_dir: str,
    max_display: int
) -> None:
    """Generate and save SHAP explanations."""
    print("\n--- Generating SHAP Explanations (this may take a while) ---")

    shap_values_pos_class = None
    expected_value_pos_class = None

    try:
        try:
            shap_version = shap.__version__
            print(f"Detected SHAP version: {shap_version}")
        except AttributeError:
            print("Could not determine SHAP version.")

        explainer_shap = shap.TreeExplainer(model)
        print("Calculating SHAP values for the test set...")
        shap_values_output = explainer_shap.shap_values(X_test)
        expected_value_output = explainer_shap.expected_value

        positive_class_index = 1
        n_classes = len(class_names)
        print(f"Using positive class '{class_names[positive_class_index]}' at index {positive_class_index} for plots.")

        # --- Correctly Handle SHAP Values Structure ---
        if isinstance(shap_values_output, list):
            if len(shap_values_output) == n_classes:
                shap_values_pos_class = shap_values_output[positive_class_index]
                print("SHAP values interpreted as list [class0, class1]. Using index 1.")
            else:
                 print(f"Error: SHAP values list expected length {n_classes}, but got {len(shap_values_output)}", file=sys.stderr)
                 return
        elif isinstance(shap_values_output, np.ndarray):
            if len(shap_values_output.shape) == 3 and shap_values_output.shape[2] == n_classes:
                print("SHAP values interpreted as ndarray (samples, features, classes). Slicing for positive class.")
                shap_values_pos_class = shap_values_output[:, :, positive_class_index]
            elif len(shap_values_output.shape) == 2:
                 print("Warning: SHAP values are a single 2D ndarray. Assuming it corresponds to the positive class.")
                 shap_values_pos_class = shap_values_output
            else:
                 print(f"Error: Unexpected SHAP values array shape: {shap_values_output.shape}", file=sys.stderr)
                 return
        else:
             print(f"Error: Unexpected type for SHAP values: {type(shap_values_output)}", file=sys.stderr)
             return

        # --- Correctly Handle Expected Value Structure ---
        if isinstance(expected_value_output, list) or isinstance(expected_value_output, np.ndarray) and expected_value_output.ndim == 1:
            if len(expected_value_output) == n_classes:
                 expected_value_pos_class = expected_value_output[positive_class_index]
                 print("SHAP expected_value interpreted as list/array [class0, class1]. Using index 1.")
            else:
                 print(f"Error: SHAP expected_value list/array expected length {n_classes}, but got {len(expected_value_output)}", file=sys.stderr)
                 return
        elif isinstance(expected_value_output, (int, float, np.number)):
             print("Warning: SHAP expected_value is a single number. Assuming it corresponds to the positive class.")
             expected_value_pos_class = expected_value_output
        else:
             print(f"Error: Unexpected structure for SHAP expected_value. Type: {type(expected_value_output)}", file=sys.stderr)
             return

        if shap_values_pos_class is None or expected_value_pos_class is None:
            print("Error: Could not determine SHAP values or expected value for the positive class.", file=sys.stderr)
            return

        print(f"Shape of SHAP values (positive class): {shap_values_pos_class.shape}")
        print(f"Expected value (positive class): {expected_value_pos_class}")

        # --- Try creating Explanation object ---
        explanation_object = None
        use_new_api = False
        try:
            explanation_object = shap.Explanation(
                values=shap_values_pos_class,
                base_values=expected_value_pos_class,
                data=X_test,
                feature_names=feature_names
            )
            print("Successfully created SHAP Explanation object.")
            use_new_api = True
        except Exception as e:
            print(f"Could not create SHAP Explanation object (may indicate older SHAP version or issue): {e}")
            print("Will attempt to use legacy plotting functions.")

    except ImportError:
         print("\nError: SHAP library not found. Cannot generate SHAP explanations.", file=sys.stderr)
         print("Install SHAP using: pip install shap", file=sys.stderr)
         return
    except Exception as e:
        print(f"\nAn unexpected error occurred during SHAP value calculation or processing: {e}", file=sys.stderr)
        print("Consider updating SHAP: pip install -U shap", file=sys.stderr)
        return


    # --- Generate Plots ---

    print(f"\nGenerating SHAP Force Plot for Instance Index: {instance_idx}")
    try:
        plt.figure()
        shap.plots.force(
            expected_value_pos_class,
            shap_values_pos_class[instance_idx,:],
            X_test[instance_idx,:],
            feature_names=feature_names,
            matplotlib=True, show=False
        )
        force_plot_path = os.path.join(output_dir, f"shap_force_plot_instance_{instance_idx}.png")
        plt.savefig(force_plot_path, bbox_inches='tight'); plt.close()
        print(f"SHAP force plot saved to: {force_plot_path}")
    except Exception as e:
        print(f"Error generating force plot: {e}", file=sys.stderr); plt.close()

    print("\nGenerating SHAP Summary Plot (Bar)...")
    try:
        plt.figure()
        if use_new_api and explanation_object is not None:
            shap.plots.bar(explanation_object, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values_pos_class, X_test, feature_names=feature_names,
                              plot_type="bar", max_display=max_display, show=False)
        plt.title("SHAP Feature Importance (Mean Absolute Value)")
        summary_bar_path = os.path.join(output_dir, "shap_summary_bar_plot.png")
        plt.savefig(summary_bar_path, bbox_inches='tight'); plt.close()
        print(f"SHAP summary bar plot saved to: {summary_bar_path}")
    except Exception as e:
        print(f"Error generating summary bar plot: {e}", file=sys.stderr); plt.close()

    print("\nGenerating SHAP Beeswarm Summary Plot...")
    try:
        plt.figure()
        if use_new_api and explanation_object is not None:
            shap.plots.beeswarm(explanation_object, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values_pos_class, X_test, feature_names=feature_names,
                              max_display=max_display, show=False)
        plt.title("SHAP Beeswarm Summary Plot")
        beeswarm_plot_path = os.path.join(output_dir, "shap_beeswarm_plot.png")
        plt.savefig(beeswarm_plot_path, bbox_inches='tight'); plt.close()
        print(f"SHAP beeswarm plot saved to: {beeswarm_plot_path}")
    except Exception as e:
        print(f"Error generating beeswarm plot: {e}", file=sys.stderr); plt.close()

    feature_to_plot = "worst concave points"
    interaction_feature = "worst perimeter"
    print(f"\nGenerating SHAP Dependence Plot for '{feature_to_plot}'...")
    try:
        try:
            feature_idx = feature_names.index(feature_to_plot)
            interaction_idx = feature_names.index(interaction_feature)
        except ValueError:
            print(f"Could not find features '{feature_to_plot}' or '{interaction_feature}'. Skipping dependence plot.", file=sys.stderr); return

        plt.figure()
        if use_new_api and explanation_object is not None:
            shap.plots.scatter(
                explanation_object[:, feature_idx],
                color=explanation_object[:, interaction_idx], show=False
            )
            plt.ylabel(f"SHAP value for\n{feature_to_plot}")
        else:
            shap.dependence_plot(
                feature_to_plot,
                shap_values_pos_class,
                X_test,
                feature_names=feature_names,
                interaction_index=interaction_feature,
                show=False
            )
        plt.title(f"SHAP Dependence: {feature_to_plot} (Color: {interaction_feature} SHAP Value)")
        dependence_plot_path = os.path.join(output_dir, f"shap_dependence_{feature_to_plot.replace(' ','_')}.png")
        plt.savefig(dependence_plot_path, bbox_inches='tight'); plt.close()
        print(f"SHAP dependence plot saved to: {dependence_plot_path}")
    except Exception as e:
        print(f"Error generating dependence plot: {e}", file=sys.stderr); plt.close()

def main() -> None:
    """Main function to run the SHAP explainability pipeline."""
    print("--- Starting SHAP Explainability Script ---")
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

    
    explain_with_shap(
        model=model, X_test=X_test, instance_idx=args.instance_idx,
        feature_names=feature_names, class_names=class_names,
        output_dir=args.output_dir, max_display=args.max_display
    )

    print("\n--- SHAP Explanation generation complete! ---")
    print(f"Plots saved to directory: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception as e:
        print(f"Could not switch matplotlib backend: {e}", file=sys.stderr)

    main()