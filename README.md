# Machine Learning Project

This repository contains multiple machine learning implementations built for IITJ coursework: from-scratch Linear Regression, KNN, Decision Tree, and a from-scratch Random Forest, plus small scikit-learn examples.

## 📁 Project Structure

```
MachineLearning/
├── KNN/
│   ├── Knn.py                       # From-scratch KNN on Iris (classification)
│   └── knn_run.py                   # scikit-learn KNN example (regression)
│   └── Visulaization/               # Generated plots (accuracy vs K, decision boundaries)
├── DecisionTree/
│   ├── DecisionTree.py              # From-scratch Decision Tree on Iris
│   ├── model_evaluation.txt         # Depth vs accuracy table
│   ├── decision_tree.png            # Graphviz tree visualization
│   └── decision_boundary_plot.png   # 2D boundary (petal features)
├── LinearRegresssion/
│   ├── linearRegression.py          # From-scratch Linear Regression on California Housing
│   ├── linearRegression_run.py      # scikit-learn Linear Regression on Salary dataset
│   ├── Training DataSet/
│   │   └── Salary_Data.csv          # Dataset for salary prediction examples
│   └── Visulaizations/              # Generated plots and metrics
├── K_Validation/
│   └── K_Validation.py         # From-scratch K-Fold CV with KNN (Iris)
│   ├── kfold_boxplot.png            # Accuracy distribution across folds
│   └── traintest_scatter.png        # Repeated train-test accuracies
├── Random Forest/
│   └── RandomForest.py              # From-scratch Random Forest vs Decision Tree (Wine Quality)
│   ├── accuracy_vs_trees.png        # Accuracy vs number of trees
│   ├── feature_importance.png       # RF feature importances
│   └── comparison_report.txt        # Metrics comparison summary
├── README.md
└── requirments.py                   # Dependency list (typo in name kept as-is)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### Install dependencies

You can install the packages listed below directly (since the repo has a `requirments.py` file instead of a `requirements.txt`).

```bash
pip install -r requirments.py
```

## 📊 Projects and How to Run

### 1) Linear Regression — from scratch (`LinearRegresssion/linearRegression.py`)

- Uses the California Housing dataset (downloaded via scikit-learn).
- Saves learning curve and actual-vs-predicted plots.

Run from the project root or from inside the `LinearRegresssion` folder:

```bash
cd LinearRegresssion
python linearRegression.py
```

Generated files will be saved in your current working directory (recommended to run from `LinearRegresssion` so outputs align with that folder).

### 2) Linear Regression — scikit-learn on Salary dataset (`LinearRegresssion/linearRegression_run.py`)

- Reads `Training DataSet/Salary_Data.csv`.
- Trains a simple linear model and can generate plots/metrics.

Because the script reads `Salary_Data.csv` from the current directory, run it from the dataset folder and reference the script relatively:

```bash
cd "LinearRegresssion/Training DataSet"
python ../linearRegression_run.py
```

### 3) KNN — from scratch on Iris (`KNN/Knn.py`)

- Downloads the Iris dataset via `ucimlrepo`.
- Can generate decision boundary plots for selected features.

Run from the `KNN` folder:

```bash
cd KNN
python Knn.py
```

Outputs (accuracy vs K plot, decision boundaries) are saved in the current directory; recommended to run from `KNN` so they land in that folder. Existing images are stored under `KNN/Visulaization`.

### 4) KNN — scikit-learn example on Salary dataset (`KNN/knn_run.py`)

- Uses `KNeighborsRegressor` on the Salary dataset.
- Expects `Salary_Data.csv` in the current working directory.

Run it from the dataset folder and reference the script relatively:

```bash
cd "LinearRegresssion/Training DataSet"
python ../../KNN/knn_run.py
```

### 5) Decision Tree — from scratch on Iris (`DecisionTree/DecisionTree.py`)

- Trains a Gini-based tree, evaluates depth vs accuracy, and saves a Graphviz tree and 2D decision boundary.

Run from the `DecisionTree` folder:

```bash
cd DecisionTree
python DecisionTree.py
```

Generates `model_evaluation.txt`, `decision_tree.png`, and `decision_boundary_plot.png` in the folder.

### 6) Random Forest — from scratch on Wine Quality (`Random Forest/RandomForest.py`)

- Compares Random Forest vs single Decision Tree; saves accuracy vs trees and feature importance plots.

Run from the `Random Forest` folder:

```bash
cd "Random Forest"
python RandomForest.py
```

Generates `accuracy_vs_trees.png`, `feature_importance.png`, and `comparison_report.txt`.

### 7) K-Fold Validation — KNN on Iris (`K_Validation/K_Validation copy.py`)

- Implements K-Fold CV from scratch, compares with train-test splits, and saves plots.

Run from the project root (paths assume root as CWD):

```bash
python K_Validation/"K_Validation copy.py"
```

Outputs are saved to `K_Validation/kfold_boxplot.png` and `K_Validation/traintest_scatter.png`.

## 🛠️ Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
ucimlrepo
tabulate
graphviz
```

## 🔍 Notes and Tips

- If you get file-not-found errors for `Salary_Data.csv`, ensure your working directory matches the commands above.
- For from-scratch implementations, outputs are saved to the current working directory. Run from the respective subfolder to keep files organized.

## 📚 Resources

- Pandas Documentation: https://pandas.pydata.org/
- Scikit-learn Documentation: https://scikit-learn.org/
- Matplotlib Documentation: https://matplotlib.org/

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created for Machine Learning coursework at IITJ.
