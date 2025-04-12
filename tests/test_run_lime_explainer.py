# tests/test_run_lime_explainer.py

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

from src.scripts import run_lime_explainer
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
        "X_train": X_train, "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names, "class_names": class_names,
        "model": model
    }

# --- Tests for Argument Parsing ---

def test_parse_arguments_defaults(monkeypatch):
    """Test that default arguments are parsed correctly."""
    monkeypatch.setattr(sys, 'argv', ['run_lime_explainer.py'])
    args = run_lime_explainer.parse_arguments()

    assert args.output_dir == "./explanations_lime"
    assert args.instance_idx == 0
    assert args.num_features == 10

def test_parse_arguments_custom(monkeypatch):
    """Test that custom arguments are parsed correctly."""
    test_output = "./custom_output"
    test_idx = 5
    test_num_features = 8
    monkeypatch.setattr(sys, 'argv', [
        'run_lime_explainer.py',
        '--output-dir', test_output,
        '--instance-idx', str(test_idx),
        '--num-features', str(test_num_features)
    ])
    args = run_lime_explainer.parse_arguments()

    assert args.output_dir == test_output
    assert args.instance_idx == test_idx
    assert args.num_features == test_num_features


# --- Tests for explain_with_lime function ---

def test_explain_with_lime_runs_and_saves_plot(trained_model_and_data, tmp_path):
    """Test if explain_with_lime runs and saves the plot file."""
    data = trained_model_and_data
    test_idx = 0
    num_features = 5
    output_dir = tmp_path / "lime_output"
    output_dir.mkdir()

    run_lime_explainer.explain_with_lime(
        X_train=data["X_train"],
        X_test=data["X_test"],
        instance_idx=test_idx,
        model=data["model"],
        feature_names=data["feature_names"],
        class_names=data["class_names"],
        y_test=data["y_test"],
        num_features=num_features,
        output_dir=str(output_dir)
    )

    expected_file = output_dir / f"lime_explanation_instance_{test_idx}.png"
    assert expected_file.is_file(), f"Output file {expected_file} was not created."
    assert expected_file.stat().st_size > 0, "Output file is empty."


# --- Tests for main function execution ---

def test_main_creates_output_dir(monkeypatch, tmp_path):
    """Test if main function creates the output directory."""
    output_dir = tmp_path / "main_lime_output"
    monkeypatch.setattr(sys, 'argv', [
        'run_lime_explainer.py',
        '--output-dir', str(output_dir)
    ])

    run_lime_explainer.main()

    assert output_dir.is_dir(), f"Output directory {output_dir} was not created by main()."


def test_main_generates_plot_file(monkeypatch, tmp_path):
    """Test if main function execution results in the plot file."""
    output_dir = tmp_path / "main_lime_plot_test"
    test_idx = 0

    monkeypatch.setattr(sys, 'argv', [
        'run_lime_explainer.py',
        '--output-dir', str(output_dir),
        '--instance-idx', str(test_idx)
    ])


    run_lime_explainer.main()


    expected_file = output_dir / f"lime_explanation_instance_{test_idx}.png"
    assert expected_file.is_file(), f"Output file {expected_file} was not created by main()."
    assert expected_file.stat().st_size > 0, "Output file is empty."


def test_main_invalid_instance_idx_exits(monkeypatch, tmp_path):
    """Test if main function exits correctly with invalid instance index."""
    output_dir = tmp_path / "main_lime_invalid_idx"
    invalid_idx = 99999 

    monkeypatch.setattr(sys, 'argv', [
        'run_lime_explainer.py',
        '--output-dir', str(output_dir),
        '--instance-idx', str(invalid_idx)
    ])


    with pytest.raises(SystemExit) as e:
        run_lime_explainer.main()


    assert e.value.code != 0, "Script should exit with non-zero status for invalid index."

    expected_file = output_dir / f"lime_explanation_instance_{invalid_idx}.png"
    assert not expected_file.exists(), "Plot file should not be created for invalid index."