from ucimlrepo import fetch_ucirepo
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def k_fold_cross_validation(model, X, y, k):
    """
    Implement K-Fold Cross-Validation from scratch
    
    Args:
        model: The model to train and evaluate
        X: Feature data
        y: Target data  
        k: Number of folds
        
    Returns:
        List of accuracy scores from each fold
    """
    # Shuffle the data randomly
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Calculate fold size
    fold_size = len(X) // k
    accuracies = []
    
    # Perform k-fold cross validation
    for i in range(k):
        # Define validation indices for current fold
        start_idx = i * fold_size
        if i == k - 1:  # Last fold gets remaining data
            end_idx = len(X)
        else:
            end_idx = (i + 1) * fold_size
            
        val_indices = np.arange(start_idx, end_idx)
        train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, len(X))])
        
        # Split data
        X_train = X_shuffled[train_indices]
        X_val = X_shuffled[val_indices]
        y_train = y_shuffled[train_indices]
        y_val = y_shuffled[val_indices]
        
        # Train model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)
        
    return accuracies

# Load and prepare Iris dataset
iris = fetch_ucirepo(id=53)
X = iris.data.features.to_numpy()
y = iris.data.targets.to_numpy().ravel()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model
model = KNeighborsClassifier(n_neighbors=5)

# Run 5-fold cross validation
accuracies = k_fold_cross_validation(model, X_scaled, y, k=5)

print("K-Fold Cross-Validation Results:")
print("=" * 40)
for i, acc in enumerate(accuracies, 1):
    print(f"Fold {i}: {acc:.4f}")

# 1. Model Performance Evaluation
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f"\n1. Model Performance Evaluation:")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation: {std_accuracy:.4f}")

# 2. Comparison with Train-Test Split
print(f"\n2. Comparison with Train-Test Split:")
print("=" * 50)

# Single 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_single = KNeighborsClassifier(n_neighbors=5)
model_single.fit(X_train, y_train)
single_accuracy = model_single.score(X_test, y_test)
print(f"Single 80/20 split accuracy: {single_accuracy:.4f}")

# Repeated train-test splits (5 times with different random states)
repeated_accuracies = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i)
    model_repeat = KNeighborsClassifier(n_neighbors=5)
    model_repeat.fit(X_train, y_train)
    repeat_accuracy = model_repeat.score(X_test, y_test)
    repeated_accuracies.append(repeat_accuracy)
    print(f"Split {i+1} (random_state={i}): {repeat_accuracy:.4f}")

# 3. Visualization
print(f"\n3. Creating Visualizations...")

# Box plot for K-fold results
plt.figure(figsize=(8, 6))
plt.boxplot(accuracies, labels=['K-Fold CV'])
plt.title('K-Fold Cross-Validation Accuracy Distribution')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.savefig('K_Validation/kfold_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot for repeated train-test splits
plt.figure(figsize=(8, 6))
plt.scatter(range(1, 6), repeated_accuracies, color='red', s=100, alpha=0.7)
plt.title('Repeated Train-Test Split Accuracies')
plt.xlabel('Split Number')
plt.ylabel('Accuracy')
plt.xticks(range(1, 6))
plt.grid(True, alpha=0.3)
plt.savefig('K_Validation/traintest_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved as separate files:")
print(f"- K-Fold box plot: 'kfold_boxplot.png'")
print(f"- Train-test scatter plot: 'traintest_scatter.png'")
print(f"\nSummary:")
print(f"K-Fold CV - Mean: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}")
print(f"Single Split: {single_accuracy:.4f}")
print(f"Repeated Splits - Mean: {np.mean(repeated_accuracies):.4f}, Std: {np.std(repeated_accuracies):.4f}")
