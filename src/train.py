"""Train a decision-tree classifier on the Iris dataset and evaluate it.

Usage (from the project root folder):
    python src/train.py --test-size 0.2 --random-state 42
    python src/train.py --max-depth 2      # limit the tree's depth

Prints the test accuracy and a classification report, and saves the
confusion matrix to outputs/confusion_matrix.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures straight to file, never open a window
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# outputs/ sits next to src/, whichever folder the script is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    """Read the command-line options (e.g. --test-size 0.2)."""
    parser = argparse.ArgumentParser(description="Train a decision tree on the Iris dataset.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of the data held out for testing (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed for the train/test shuffle and the tree (default: 42)")
    parser.add_argument("--max-depth", type=int, default=None,  # NEW
                        help="Maximum tree depth; omit for a fully grown tree (default: None)")
    return parser.parse_args()


def train_and_evaluate(test_size=0.2, random_state=42, max_depth=None):  # NEW: max_depth
    """Run the full workflow: load, split, train, predict, score.

    Returns a dictionary so other code (and the tests) can use the results.
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)  # NEW
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "class_names": iris.target_names,
    }


def save_confusion_matrix(y_test, y_pred, class_names, path):
    """Draw the confusion matrix and save it as an image, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, cmap="Blues"
    )
    display.ax_.set_title("Iris decision tree: confusion matrix")
    display.figure_.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(display.figure_)


def main():
    args = parse_args()
    results = train_and_evaluate(test_size=args.test_size, random_state=args.random_state,
                                 max_depth=args.max_depth)  # NEW

    print(f"Test size: {args.test_size}   Random state: {args.random_state}   "
          f"Max depth: {args.max_depth}")  # NEW
    print(f"Test samples: {len(results['y_test'])}")
    print(f"Accuracy: {results['accuracy']:.3f}\n")
    print(classification_report(results["y_test"], results["y_pred"],
                                target_names=results["class_names"]))

    figure_path = OUTPUT_DIR / "confusion_matrix.png"
    save_confusion_matrix(results["y_test"], results["y_pred"], results["class_names"], figure_path)
    print(f"Confusion matrix saved to {figure_path}")


if __name__ == "__main__":
    main()