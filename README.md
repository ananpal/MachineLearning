# Machine Learning Project

This repository contains multiple machine learning implementations built for IITJ coursework: a from-scratch Linear Regression, and a from-scratch KNN classifier (plus a scikit-learn KNN example).

## 📁 Project Structure

```
MachineLearning/
├── KNN/
│   ├── Knn.py                       # From-scratch KNN on Iris (classification)
│   └── knn_run.py                   # scikit-learn KNN example (regression)
│   └── Visulaization/               # Generated plots (accuracy vs K, decision boundaries)
│       ├── Accuracy_vs_K_value.png
│       ├── Accuracy_vs_K_value.txt
│       ├── decision_boundry_1.png
│       └── decision_boundry_15.png
├── LinearRegresssion/
│   ├── linearRegression.py          # From-scratch Linear Regression on California Housing
│   ├── linearRegression_run.py      # scikit-learn Linear Regression on Salary dataset
│   ├── Training DataSet/
│   │   └── Salary_Data.csv          # Dataset for salary prediction examples
│   └── Visulaizations/              # Generated plots and metrics
│       ├── actual_vs_predicted.png
│       ├── learning_curve.png
│       ├── r2_and_error.txt
│       └── report.txt
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

## 🛠️ Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
ucimlrepo
tabulate
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
