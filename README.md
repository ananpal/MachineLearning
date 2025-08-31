# Machine Learning Project

This repository contains machine learning implementations and data analysis projects.

## 📁 Project Structure

```
MachineLearning/
├── linearRegression.py          # Linear regression implementation
├── requirments.py              # Project dependencies
├── Salary_Data.csv             # Dataset
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Required packages (see `requirments.py`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MachineLearning
```

2. Install dependencies:
```bash
pip install -r requirments.py
```

## 📊 Projects

### 1. Linear Regression (`linearRegression.py`)

Implementation of linear regression for salary prediction using:
- Pandas for data manipulation
- Scikit-learn for machine learning
- One-hot encoding for categorical variables
- Matplotlib for visualization

**Features:**
- Data preprocessing with One-Hot encoding
- Missing value handling
- Train-test split
- Model training and prediction
- Visualization of results

## 🛠️ Dependencies

```
matplotlib
numpy
pandas
seaborn
scikit-learn
```

## 📈 Usage

### Running Linear Regression

```bash
python linearRegression.py
```

## 📝 Key Learning Points

### Data Preprocessing
- **One-Hot Encoding**: Converting categorical variables to numeric
- **Missing Value Handling**: Proper handling of NaN values
- **Data Reshaping**: Converting 1D arrays to 2D for ML algorithms

### Machine Learning
- **Linear Regression**: Basic regression implementation
- **Train-Test Split**: Proper data splitting for model evaluation
- **Feature Engineering**: Creating meaningful features from raw data

### Common Issues & Solutions
- **Shape Mismatch**: Using `.reshape(-1, 1)` for single features
- **Categorical Data**: One-hot encoding vs label encoding
- **Missing Values**: Proper filtering and handling

## 🔧 Troubleshooting

### Common Errors

1. **"Expected 2D array, got 1D array"**
   - Solution: Use `.reshape(-1, 1)` for single features

2. **"divide by zero encountered in matmul"**
   - Solution: Check for NaN values and handle missing data

3. **"RuntimeWarning: overflow encountered"**
   - Solution: Check data types and scale features if needed

## 📚 Resources

- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created for Machine Learning coursework at IITJ.

---

**Note**: This project is part of the Machine Learning curriculum and serves as a learning resource for data preprocessing and basic machine learning implementations.
