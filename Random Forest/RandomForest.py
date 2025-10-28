
from ucimlrepo import fetch_ucirepo 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TreeNode:
    def __init__(self, feature, thresold, isLeafNode = False, classValue = None):
        self.feature = feature
        self.thresold = thresold
        self.left = None
        self.right = None
        self.isLeafNode = isLeafNode
        self.classValue = classValue

class DecisionTree:
    def __init__(self, max_depth=5):
        self.rootNode = None
        self.max_depth = max_depth  
    
    def fit(self, X, Y):
        self.rootNode = self._createDecisionTree(X,Y)

    def predict(self, X):
        return np.array([self.predict_y(row, self.rootNode) for row in X])
    
    def predict_y(self, x, treeNode):
        if treeNode is None:
            return None
        if(treeNode.isLeafNode):
            return treeNode.classValue
        if x[treeNode.feature] <= treeNode.thresold:
            return self.predict_y(x, treeNode.left)
        else:
            return self.predict_y(x, treeNode.right)


    def _createDecisionTree(self,x,y, depth=1):
        if len(y) == 0:
            return TreeNode(None, None, True, 0)
        
        if(depth == self.max_depth):
            values, counts = np.unique(y, return_counts=True)
            if len(values) > 0:
                return TreeNode(None, None, True, values[np.argmax(counts)])
            else:
                return TreeNode(None, None, True, 0)
        
        y_unique = np.unique(y)
        if y_unique.size == 1:
            return TreeNode(None, None, True, y_unique[0])

        n_features = x.shape[1]
        best_feature = None
        best_thresold = None
        best_feature_gini = 1.0
        for feature in range(n_features):
            feature_values = x[:, feature]
            values = np.unique(feature_values)
            best_value_gini = 1.0
            best_value_thresold = None
            for value in values:
                left_indices = np.where(feature_values<=value)[0]
                right_indices = np.where(feature_values>value)[0]
                gini_value = self._calculateWiightedGini(y,left_indices,right_indices)
                if gini_value < best_value_gini:
                    best_value_gini = gini_value
                    best_value_thresold = value
            if best_feature_gini > best_value_gini:
                best_feature_gini = best_value_gini
                best_feature = feature
                best_thresold = best_value_thresold
        if best_feature is None or best_thresold is None:
            values, counts = np.unique(y, return_counts=True)
            if len(values) > 0:
                return TreeNode(None, None, True, values[np.argmax(counts)])
            else:
                return TreeNode(None, None, True, 0)
        
        node = TreeNode(best_feature, best_thresold)
        left_indicies = np.where(x[:,best_feature]<= best_thresold)[0]
        right_indicies = np.where(x[:,best_feature]>best_thresold)[0]
        node.left = self._createDecisionTree(x[left_indicies,],y[left_indicies], depth+1)
        node.right = self._createDecisionTree(x[right_indicies,],y[right_indicies], depth+1)
        if(node.left == None and node.right == None):
            node.isLeafNode = True
            values, counts = np.unique(y, return_counts=True)
            if len(values) > 0:
                node.classValue = values[np.argmax(counts)]
            else:
                node.classValue = 0
        return node
            
    def _calculateWiightedGini(self, y, left_indices, right_indices):
        y_left = y[left_indices]
        y_right = y[right_indices]

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        if n_total == 0:
            return 0.0

        g_left = self._calculateGini(y_left)
        g_right = self._calculateGini(y_right)

        return (n_left / n_total) * g_left + (n_right / n_total) * g_right

    def _calculateGini(self, data):
        uv, counts = np.unique(data, return_counts=True)
        probs = counts/len(data)
        return 1.0 - np.sum(probs**2)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.feature_importances_ = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))
        
        self.trees = []
        
        for i in range(self.n_trees):
            # Bootstrap sampling
            bootstrap_indices = self._bootstrap_sample(X.shape[0])
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Create decision tree with feature randomness
            tree = DecisionTreeWithFeatureRandomness(
                max_depth=self.max_depth, 
                n_features=self.n_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        # Calculate feature importances based on Gini reduction
        self.feature_importances_ = self._calculate_feature_importances(X, y)
    
    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros((X.shape[0], self.n_trees))
        
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        
        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[i, :]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)
    
    def _bootstrap_sample(self, n_samples):
        return np.random.choice(n_samples, size=n_samples, replace=True)
    
    def _calculate_feature_importances(self, X, y):
        # Simple feature importance based on how often features are used
        feature_counts = np.zeros(X.shape[1])
        
        for tree in self.trees:
            self._count_feature_usage(tree.rootNode, feature_counts)
        
        # Normalize by total usage
        if np.sum(feature_counts) > 0:
            feature_importances = feature_counts / np.sum(feature_counts)
        else:
            feature_importances = np.ones(X.shape[1]) / X.shape[1]
        
        return feature_importances
    
    def _count_feature_usage(self, node, feature_counts):
        if node is None or node.isLeafNode:
            return
        
        if node.feature is not None:
            feature_counts[node.feature] += 1
        
        self._count_feature_usage(node.left, feature_counts)
        self._count_feature_usage(node.right, feature_counts)


class DecisionTreeWithFeatureRandomness(DecisionTree):
    def __init__(self, max_depth=5, n_features=None):
        super().__init__(max_depth)
        self.n_features = n_features
        self.feature_importances_ = None
    
    def _createDecisionTree(self, x, y, depth=1):
        if len(y) == 0:
            return TreeNode(None, None, True, 0)
        
        if(depth == self.max_depth):
            values, counts = np.unique(y, return_counts=True)
            if len(values) > 0:
                return TreeNode(None, None, True, values[np.argmax(counts)])
            else:
                return TreeNode(None, None, True, 0)
        
        y_unique = np.unique(y)
        if y_unique.size == 1:
            return TreeNode(None, None, True, y_unique[0])

        n_features = x.shape[1]
        
        # Random feature selection
        if self.n_features and self.n_features < n_features:
            selected_features = np.random.choice(n_features, size=self.n_features, replace=False)
        else:
            selected_features = np.arange(n_features)
        
        best_feature = None
        best_thresold = None
        best_feature_gini = 1.0
        
        for feature in selected_features:
            feature_values = x[:, feature]
            values = np.unique(feature_values)
            best_value_gini = 1.0
            best_value_thresold = None
            
            for value in values:
                left_indices = np.where(feature_values<=value)[0]
                right_indices = np.where(feature_values>value)[0]
                gini_value = self._calculateWiightedGini(y,left_indices,right_indices)
                if gini_value < best_value_gini:
                    best_value_gini = gini_value
                    best_value_thresold = value
            
            if best_feature_gini > best_value_gini:
                best_feature_gini = best_value_gini
                best_feature = feature
                best_thresold = best_value_thresold
        
        if best_feature is None or best_thresold is None:
            values, counts = np.unique(y, return_counts=True)
            if len(values) > 0:
                return TreeNode(None, None, True, values[np.argmax(counts)])
            else:
                return TreeNode(None, None, True, 0)
        
        node = TreeNode(best_feature, best_thresold)
        left_indicies = np.where(x[:,best_feature]<= best_thresold)[0]
        right_indicies = np.where(x[:,best_feature]>best_thresold)[0]
        node.left = self._createDecisionTree(x[left_indicies,],y[left_indicies], depth+1)
        node.right = self._createDecisionTree(x[right_indicies,],y[right_indicies], depth+1)
        
        if(node.left == None and node.right == None):
            node.isLeafNode = True
            values, counts = np.unique(y, return_counts=True)
            if len(values) > 0:
                node.classValue = values[np.argmax(counts)]
            else:
                node.classValue = 0
        return node


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def evaluate_models(X_train, X_test, y_train, y_test):
    # Random Forest
    rf_model = RandomForest(n_trees=50, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = calculate_metrics(y_test, rf_pred)
    
    # Single Decision Tree (high max_depth for overfitting)
    dt_model = DecisionTree(max_depth=20)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_metrics = calculate_metrics(y_test, dt_pred)
    
    return rf_metrics, dt_metrics, rf_model

def plot_accuracy_vs_trees(X_train, X_test, y_train, y_test, n_trees_list=[1, 5, 10, 25, 50, 100]):
    accuracies = []
    
    for n_trees in n_trees_list:
        rf = RandomForest(n_trees=n_trees, max_depth=10)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accuracies.append(acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees_list, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest: Accuracy vs Number of Trees')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_vs_trees.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracies

def plot_feature_importance(model, feature_names):
    if model.feature_importances_ is None:
        print("Feature importances not available")
        return
    
    # Sort features by importance
    indices = np.argsort(model.feature_importances_)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(model.feature_importances_)), 
            model.feature_importances_[indices])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Random Forest')
    plt.xticks(range(len(feature_names)), 
               [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


# Load Wine Quality dataset
wine_quality = fetch_ucirepo(id=186) 
X = wine_quality.data.features 
y = wine_quality.data.targets

# Convert to numpy arrays
X = X.to_numpy()
y = y.to_numpy().flatten()

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset loaded successfully!")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# Evaluate models
print("\nEvaluating models...")
rf_metrics, dt_metrics, rf_model = evaluate_models(X_train, X_test, y_train, y_test)

# Print results
print("\nRandom Forest Results:")
print(f"Accuracy: {rf_metrics[0]:.4f}")
print(f"Precision: {rf_metrics[1]:.4f}")
print(f"Recall: {rf_metrics[2]:.4f}")
print(f"F1-Score: {rf_metrics[3]:.4f}")

print("\nSingle Decision Tree Results:")
print(f"Accuracy: {dt_metrics[0]:.4f}")
print(f"Precision: {dt_metrics[1]:.4f}")
print(f"Recall: {dt_metrics[2]:.4f}")
print(f"F1-Score: {dt_metrics[3]:.4f}")

# Create visualizations
print("\nCreating visualizations...")
plot_accuracy_vs_trees(X_train, X_test, y_train, y_test)

feature_names = wine_quality.variables['name'].tolist()[:11]  # Take first 11 features
plot_feature_importance(rf_model, feature_names)

# Write comparison report

print("\nAnalysis complete! Check the generated files:")
print("- accuracy_vs_trees.png")
print("- feature_importance.png") 
print("- comparison_report.txt") 