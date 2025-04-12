# ML Interpretability Toolbox

This repository provides command-line scripts for explaining machine learning models using popular interpretability techniques like LIME and SHAP. It's designed to be a simple toolkit for generating local and global explanations for classification models.

## Features

*   **LIME Explanations:** Generate local instance explanations using LIME (Local Interpretable Model-agnostic Explanations).
*   **SHAP Explanations:** Generate local and global explanations using SHAP (SHapley Additive exPlanations).
    *   Local Force Plots for individual instances.
    *   Global Summary Plots (Bar, Beeswarm) for overall feature importance.
    *   Dependence Plots to understand feature interactions.
*   **Command-Line Interface:** Easy-to-use scripts run directly from the terminal.
*   **Modular Utilities:** Shared functions for data loading (`data_utils.py`) and model training (`model_utils.py`). The current implementation uses the sklearn breast cancer dataset and a RandomForestClassifier by default within these utilities.
*   **Testing:** Includes unit tests (`pytest`) for the explanation scripts.


## Interpretability Methods Explained

This toolbox implements two popular techniques for model interpretability:

### LIME (Local Interpretable Model-agnostic Explanations)

*   **What it is:** LIME focuses on **local interpretability**. It explains the prediction of a specific instance by approximating the complex, potentially black-box model with a simpler, interpretable model (like linear regression) in the local neighborhood of that instance. It perturbs the instance's features, gets predictions from the original model for these perturbations, and then weights these perturbed samples based on their proximity to the original instance to train the simple local model. The coefficients or feature importances of this simple model then serve as the explanation for the original instance's prediction.
*   **Scope:** Local (explains individual predictions).
*   **Model Agnostic:** Yes (can be applied to any classification or regression model).
*   **Reliable Sources:**
    *   **Original Paper:** "Why Should I Trust You?": Explaining the Predictions of Any Classifier (Ribeiro, Singh, Guestrin, 2016) - [arXiv Link](https://arxiv.org/abs/1602.04938)
    *   **GitHub Repository:** [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)
    *   **Book Chapter:** Interpretable Machine Learning by Christoph Molnar - [LIME Chapter](https://christophm.github.io/interpretable-ml-book/lime.html)

### SHAP (SHapley Additive exPlanations)

*   **What it is:** SHAP is based on **Shapley values**, a concept from cooperative game theory used to fairly distribute the "payout" (the model's prediction difference from the baseline) among the "players" (the features). SHAP values quantify the marginal contribution of each feature to the prediction for a specific instance, considering all possible combinations of features. It provides a unified framework that connects LIME and other methods, offering both local explanations (how features contributed to a single prediction) and global explanations (overall feature importance derived by aggregating local SHAP values).
*   **Scope:** Local and Global.
*   **Model Agnostic:** Yes (with model-specific optimizations available for certain types, like tree-based models, which are faster).
*   **Reliable Sources:**
    *   **Original Paper:** A Unified Approach to Interpreting Model Predictions (Lundberg, Lee, 2017) - [arXiv Link](https://arxiv.org/abs/1705.07874)
    *   **GitHub Repository:** [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
    *   **Documentation:** [https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)
    *   **Book Chapter:** Interpretable Machine Learning by Christoph Molnar - [SHAP Chapter](https://christophm.github.io/interpretable-ml-book/shap.html)
  
## Project Structure

```
ML-INTERPRETABILITY-TOOLBOX/
├── .gitignore
├── .mypy_cache/             # mypy type checking cache (ignore)
├── .pytest_cache/          # pytest cache (ignore)
├── explanations_lime/        # Default output directory for LIME plots
├── explanations_shap/        # Default output directory for SHAP plots
├── README.md               # This file
├── requirements.txt        # Project dependencies
├── src/
│   ├── __init__.py
│   ├── scripts/              # Executable scripts
│   │   ├── __init__.py
│   │   ├── run_lime_explainer.py # Script for LIME explanations
│   │   └── run_shap_explainer.py # Script for SHAP explanations
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── data_utils.py     # Data loading and splitting logic
│       └── model_utils.py    # Model training logic
└── tests/                  # Unit and integration tests
    ├── __init__.py
    ├── test_run_lime_explainer.py # Tests for LIME script
    └── test_run_shap_explainer.py # Tests for SHAP script
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ML-INTERPRETABILITY-TOOLBOX
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    source venv/Scripts/activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure `requirements.txt` includes `lime`, `shap`, `scikit-learn`, `numpy`, `matplotlib`, and `pytest`.*

## Usage

The primary way to use this toolbox is through the command-line scripts located in `src/scripts/`.

### LIME Explanations

The `run_lime_explainer.py` script generates a LIME explanation plot for a specific instance in the test set.

**Basic Usage:**
```bash
python src/scripts/run_lime_explainer.py
```
This will explain the instance with index `0` and save the plot to `./explanations_lime/lime_explanation_instance_0.png`.

**Command-Line Arguments:**

*   `--output-dir <path>`: Specify the directory to save the LIME plot (default: `./explanations_lime`).
*   `--instance-idx <int>`: Specify the 0-based index of the test instance to explain (default: `0`).
*   `--num-features <int>`: Specify the number of top features to display in the LIME explanation (default: `10`).

**Example:** Explain instance 5 and save to a custom directory:
```bash
python src/scripts/run_lime_explainer.py --instance-idx 5 --output-dir ./my_lime_results --num-features 8
```

### SHAP Explanations

The `run_shap_explainer.py` script generates several SHAP plots: a local force plot for a specific instance, and global summary (bar, beeswarm) and dependence plots based on the entire test set. *Note: SHAP value calculation for the entire test set can take some time.*

**Basic Usage:**
```bash
python src/scripts/run_shap_explainer.py
```
This will generate plots for instance `0` (force plot) and the entire test set (summary/dependence plots), saving them to `./explanations_shap/`.

**Command-Line Arguments:**

*   `--output-dir <path>`: Specify the directory to save the SHAP plots (default: `./explanations_shap`).
*   `--instance-idx <int>`: Specify the 0-based index of the test instance for the *local* force plot (default: `0`).
*   `--max-display <int>`: Specify the maximum number of features to display in the summary plots (default: `15`).

**Example:** Generate SHAP plots for instance 10, saving to a custom directory, showing top 12 features:
```bash
python src/scripts/run_shap_explainer.py --instance-idx 10 --output-dir ./my_shap_results --max-display 12
```

**Output Files (SHAP):**

*   `shap_force_plot_instance_<idx>.png`
*   `shap_summary_bar_plot.png`
*   `shap_beeswarm_plot.png`
*   `shap_dependence_<feature_name>.png` (currently hardcoded for 'worst concave points')

## Running Tests

Unit tests are implemented using `pytest`. To run the tests:

```bash
pytest
```
or
```bash
pytest tests/
```

This will execute the tests defined in the `tests/` directory, checking argument parsing, output file creation, and handling of invalid inputs for both LIME and SHAP scripts.

## Dependencies

*   Python 3.x
*   LIME (`lime`)
*   SHAP (`shap`)
*   Scikit-learn (`scikit-learn`)
*   NumPy (`numpy`)
*   Matplotlib (`matplotlib`)
*   Pytest (`pytest`) for testing