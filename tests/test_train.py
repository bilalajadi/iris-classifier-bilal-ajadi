"""Basic checks for src/train.py. Run from the project root with:  python -m pytest"""

import sys
from pathlib import Path

# Let Python find train.py inside the src/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import save_confusion_matrix, train_and_evaluate  # noqa: E402


def test_accuracy_is_at_least_90_percent():
    results = train_and_evaluate(test_size=0.2, random_state=42)
    assert results["accuracy"] >= 0.9


def test_split_sizes_are_correct():
    results = train_and_evaluate(test_size=0.2, random_state=42)
    assert len(results["y_test"]) == 30          # 20% of 150 flowers
    assert len(results["y_pred"]) == len(results["y_test"])


def test_confusion_matrix_image_is_saved(tmp_path):
    # tmp_path is a throwaway folder pytest provides, so the test never touches outputs/
    results = train_and_evaluate()
    image = tmp_path / "confusion_matrix.png"
    save_confusion_matrix(results["y_test"], results["y_pred"], results["class_names"], image)
    assert image.exists()

def test_max_depth_limits_tree():
    results = train_and_evaluate(max_depth=2)
    assert results["model"].get_depth() <= 2