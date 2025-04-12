# tests/test_run_shap_explainer.py

import sys
import os
import pytest
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') 


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root)) 

from src.scripts import run_shap_explainer
from src.utils.data_utils import load_and_split_data
from src.utils.model_utils import train_model

# --- Fixtures ---

@pytest.fixture(scope="module")
def trained_model_and_data():
    """Fixture to load data and train a model once per test module."""
    X_train, X_test, y_train, y_test, feature_names, class_names = load_and_split_data()
    model = train_model(X_train, y_train)

    n_test_samples = 5 
    if len(X_test) > n_test_samples:
        X_test = X_test[:n_test_samples]
        y_test = y_test[:n_test_samples] 
    return {
        "X_train": X_train, 
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_names": class_names,
        "model": model
    }

# --- Tests for Argument Parsing ---

def test_parse_arguments_defaults_shap(monkeypatch):
    """Test that default SHAP arguments are parsed correctly."""
    monkeypatch.setattr(sys, 'argv', ['run_shap_explainer.py'])
    args = run_shap_explainer.parse_arguments()

    assert args.output_dir == "./explanations_shap"
    assert args.instance_idx == 0
    assert args.max_display == 15

def test_parse_arguments_custom_shap(monkeypatch):
    """Test that custom SHAP arguments are parsed correctly."""
    test_output = "./custom_shap_output"
    test_idx = 3
    test_max_display = 10
    monkeypatch.setattr(sys, 'argv', [
        'run_shap_explainer.py',
        '--output-dir', test_output,
        '--instance-idx', str(test_idx),
        '--max-display', str(test_max_display)
    ])
    args = run_shap_explainer.parse_arguments()

    assert args.output_dir == test_output
    assert args.instance_idx == test_idx
    assert args.max_display == test_max_display



# --- Tests for main function execution ---

def test_main_creates_output_dir_shap(monkeypatch, tmp_path):
    """Test if main function creates the SHAP output directory."""
    output_dir = tmp_path / "main_shap_output"
    monkeypatch.setattr(sys, 'argv', [
        'run_shap_explainer.py',
        '--output-dir', str(output_dir)
    ])

    run_shap_explainer.main()

    assert output_dir.is_dir(), f"Output directory {output_dir} was not created by main()."


def test_main_generates_plot_files_shap(monkeypatch, tmp_path):
    """Test if main function execution results in the expected SHAP plot files."""
    output_dir = tmp_path / "main_shap_plot_test"
    test_idx = 0
    monkeypatch.setattr(sys, 'argv', [
        'run_shap_explainer.py',
        '--output-dir', str(output_dir),
        '--instance-idx', str(test_idx)
    ])

    run_shap_explainer.main()

    
    expected_files = [
        output_dir / f"shap_force_plot_instance_{test_idx}.png",
        output_dir / "shap_summary_bar_plot.png"
    ]
    for expected_file in expected_files:
        assert expected_file.is_file(), f"Output file {expected_file} was not created by main()."
        assert expected_file.stat().st_size > 0, f"Output file {expected_file} is empty."


def test_main_invalid_instance_idx_exits_shap(monkeypatch, tmp_path):
    """Test if main function exits correctly with invalid instance index for SHAP."""
    output_dir = tmp_path / "main_shap_invalid_idx"
    invalid_idx = 99999
    monkeypatch.setattr(sys, 'argv', [
        'run_shap_explainer.py',
        '--output-dir', str(output_dir),
        '--instance-idx', str(invalid_idx)
    ])

    with pytest.raises(SystemExit) as e:
        run_shap_explainer.main()

    assert e.value.code != 0, "Script should exit with non-zero status for invalid index."
    
    expected_files = [
        output_dir / f"shap_force_plot_instance_{invalid_idx}.png",
        output_dir / "shap_summary_bar_plot.png"
    ]
    for expected_file in expected_files:
        assert not expected_file.exists(), f"Plot file {expected_file} should not be created for invalid index."