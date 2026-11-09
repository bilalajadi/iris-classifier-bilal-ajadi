"""Iris flower classifier (decision tree).

Run from the project folder: python src/train.py --test-size 0.2 --random-state 42

It should print the test accuracy and save a confusion matrix picture to outputs/confusion_matrix.png."""

# --- Tools we need ---
import argparse                  # reads options typed after the command, e.g. --test-size 0.2
from pathlib import Path         # builds file paths that work on Windows

import matplotlib
matplotlib.use("Agg")            # save charts straight to a file, doesnt open a window
import matplotlib.pyplot as plt  # drawing commands
from sklearn.datasets import load_iris                    # 150-flower dataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report  # scoring tools
from sklearn.model_selection import train_test_split      # splits data into training and test sets
from sklearn.tree import DecisionTreeClassifier           # the model

# Saving the Chart
# Go up from src/train.py to the project folder, then into outputs/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    """Read the options typed on the command line."""
    parser = argparse.ArgumentParser(description="Train a decision tree on the Iris dataset.")
    # How much of the data to hide for testing (0.2 = 20%)
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of the data held out for testing (default: 0.2)")
    # Fixes the random shuffle so every run gives the same result
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed for the train/test shuffle and the tree (default: 42)")
    return parser.parse_args()


def train_and_evaluate(test_size=0.2, random_state=42):
    """Load the data, split it, train the tree, predict and score."""
    # 1. Load the 150 flowers: data = measurements (X), target = species (y)
    iris = load_iris()

    # 2. Split: most flowers for training, the rest hidden for the final test
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )

    # 3. Build the tree and let it learn from the training flowers
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Ask the trained tree to name the species of the hidden test flowers
    y_pred = model.predict(X_test)

    # 5. Hand back everything needed for printing and the chart
    return {
        "y_test": y_test,                             # the true answers
        "y_pred": y_pred,                             # the model's answers
        "accuracy": accuracy_score(y_test, y_pred),   # fraction it got right
        "class_names": iris.target_names,             # species names, for labels
    }


def save_confusion_matrix(y_test, y_pred, class_names, path):
    """Draw the confusion matrix and save it as a picture."""
    path.parent.mkdir(parents=True, exist_ok=True)    # create outputs/ if it doesn't exist yet
    
    # Rows = true species, columns = predicted species; the diagonal = correct answers
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, cmap="Blues"
    )
    display.ax_.set_title("Iris decision tree: confusion matrix")
    display.figure_.savefig(path, dpi=150, bbox_inches="tight")   # save the picture
    plt.close(display.figure_)                                    # tidy up


def main():
    """Run everything in order: read options, train, print results, save the chart."""
    args = parse_args()
    results = train_and_evaluate(test_size=args.test_size, random_state=args.random_state)

    # Print the settings and how well the model did
    print(f"Test size: {args.test_size}   Random state: {args.random_state}")
    print(f"Test samples: {len(results['y_test'])}")
    print(f"Accuracy: {results['accuracy']:.3f}\n")    # 3 decimal places
    # Precision, recall and F1-score for each species
    print(classification_report(results["y_test"], results["y_pred"],
                                target_names=results["class_names"]))

    # Save the confusion matrix picture into outputs/
    figure_path = OUTPUT_DIR / "confusion_matrix.png"
    save_confusion_matrix(results["y_test"], results["y_pred"], results["class_names"], figure_path)
    print(f"Confusion matrix saved to {figure_path}")

if __name__ == "__main__":
    main()