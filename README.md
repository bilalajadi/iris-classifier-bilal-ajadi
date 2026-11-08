# Iris Classifier (Decision Tree)

## Overview

End-to-end machine learning example from the IT Career Switch **AI Fundamentals** course. It builds a decision-tree classifier on the classic Iris dataset using scikit-learn: load the data, split it into training and test sets, train the model, predict on unseen flowers, and evaluate with accuracy and a confusion matrix.

The dataset has 150 flowers, 4 measurements each (sepal length, sepal width, petal length, petal width, in cm) and 3 species (setosa, versicolor, virginica). It ships with scikit-learn, so no download is needed.

## Quick start

```bash
git clone https://github.com/bilalajadi/iris-classifier.git
cd iris-classifier
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows Command Prompt
pip install -r requirements.txt
python src/train.py --test-size 0.2 --random-state 42
```

Run the tests:

```bash
python -m pytest
```

Open the walk-through notebook:

```bash
jupyter notebook notebooks/iris_model.ipynb
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--test-size` | 0.2 | Fraction of data held out for testing |
| `--random-state` | 42 | Seed for the split and the tree |
| `--max-depth` | None | Maximum tree depth (None = fully grown) |

```bash
python src/train.py --max-depth 2
```

## Results

With `--test-size 0.2 --random-state 42` (120 training flowers, 30 test flowers):

| Metric | Value |
| --- | --- |
| Test accuracy | 1.000 (30 / 30 correct) |
| Confusion matrix | All predictions on the diagonal (see `outputs/confusion_matrix.png`) |

![Confusion matrix](outputs/confusion_matrix.png)

The tree's first split is on petal length (≤ 2.45 cm → setosa), which separates setosa perfectly. The remaining splits use petal length and width to tell versicolor from virginica, the only pair of species that overlap.

### Limitations

- 30 test samples is a small test set: each mistake moves accuracy by 3.3 percentage points, and other random splits score lower (5-fold cross-validation gives 90–100% per fold). Cross-validation would give a more reliable estimate.
- - By default the tree is grown to full depth, which fits the training data perfectly and would overfit noisier data. The `--max-depth` option limits it: `--max-depth 2` scores 0.967 (29/30), with one versicolor misclassified as virginica.

## Project structure

```
iris-classifier/
├── data/                   # empty: Iris is loaded from scikit-learn
├── notebooks/
│   └── iris_model.ipynb    # walk-through notebook
├── src/
│   └── train.py            # reproducible command-line script
├── tests/
│   └── test_train.py       # basic pytest checks
├── outputs/                # created by train.py (confusion matrix figure)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## License

MIT. See [LICENSE](LICENSE).
