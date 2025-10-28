from ucimlrepo import fetch_ucirepo
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from graphviz import Digraph 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
# # metadata 
# print(iris.metadata) 
  
# variable information 
#print(iris.variables) 

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
        if(depth == self.max_depth):
            values, counts = np.unique(y, return_counts=True)
            return TreeNode(None, None, True, values[np.argmax(counts)])
        y_unique = np.unique(y)
        if y_unique.size ==1:
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
            return TreeNode(None, None, True, values[np.argmax(counts)])
        
        node = TreeNode(best_feature, best_thresold)
        left_indicies = np.where(x[:,best_feature]<= best_thresold)[0]
        right_indicies = np.where(x[:,best_feature]>best_thresold)[0]
        node.left = self._createDecisionTree(x[left_indicies,],y[left_indicies], depth+1)
        node.right = self._createDecisionTree(x[right_indicies,],y[right_indicies], depth+1)
        if(node.left == None and node.right == None):
            node.isLeafNode = True
            values, counts = np.unique(y, return_counts=True)
            node.classValue = values[np.argmax(counts)]
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


def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true.flatten() == y_pred.flatten()) / len(y_true)

def evaluate_model_performance(X_train, X_test, y_train, y_test, max_depths=[1, 2, 3, 5, 10]):
    results = []
    
    # Prepare evaluation data
    for depth in max_depths:
        # Train model with current depth
        model = DecisionTree(max_depth=depth)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate accuracies
        train_acc = calculate_accuracy(y_train, y_train_pred)
        test_acc = calculate_accuracy(y_test, y_test_pred)
        
        # Determine overfitting
        overfitting = "Yes" if train_acc - test_acc > 0.1 else "No"
        
        results.append({
            'depth': depth,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting': overfitting
        })
    
    return results

def write_evaluation_to_file(results, X, Y, x_train, x_test, filename="model_evaluation.txt"):
    """
    Write evaluation results to a file in table format
    """
    with open(filename, 'w') as f:
        
        # Model evaluation table
        f.write("MODEL EVALUATION: Effect of Tree Depth on Overfitting\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Max Depth':<12} {'Training Accuracy':<20} {'Testing Accuracy':<19} {'Overfitting':<12}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"{result['depth']:<12} {result['train_acc']:<20.4f} {result['test_acc']:<19.4f} {result['overfitting']:<12}\n")
        
        f.write("-" * 80 + "\n\n")

def create_tree_graphviz(node, feature_names=None, filename="tree"):
    """Create tree visualization using graphviz"""
    dot = Digraph(comment='Decision Tree')
    
    def add_nodes(node, parent_id=None, edge_label=""):
        if node.isLeafNode:
            node_id = f"leaf_{id(node)}"
            dot.node(node_id, f"Class {node.classValue}", shape='box', style='filled', fillcolor='lightgreen')
        else:
            feature_name = feature_names[node.feature] if feature_names else f"Feature_{node.feature}"
            node_id = f"node_{id(node)}"
            dot.node(node_id, f"{feature_name}\n<= {node.thresold:.3f}", shape='box', style='filled', fillcolor='lightblue')
            
            if node.left:
                add_nodes(node.left, node_id, "Yes")
            if node.right:
                add_nodes(node.right, node_id, "No")
        
        if parent_id:
            dot.edge(parent_id, node_id, label=edge_label)
    
    add_nodes(node)
    dot.render(filename, format='png', cleanup=True)

def plot_decision_boundary_2d(model, X_2d, y_2d, feature_names, class_names, title="Decision Boundary Plot"):
    """
    Create a decision boundary plot for 2D data
    """
    # Create a mesh grid
    x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
    y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Make predictions on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Convert string predictions to numeric values for plotting
    unique_classes = np.unique(y_2d.flatten())
    class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
    Z_numeric = np.array([class_to_num[cls] for cls in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_numeric, alpha=0.8, cmap=plt.cm.RdYlBu, levels=len(unique_classes))
    
    # Plot the data points
    y_numeric = np.array([class_to_num[cls] for cls in y_2d.flatten()])
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_numeric, cmap=plt.cm.RdYlBu, edgecolors='black')
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    
    # Add colorbar with class labels
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(len(unique_classes)))
    cbar.set_ticklabels(unique_classes)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('decision_boundary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Load and prepare data
X = iris.data.features 
y = iris.data.targets 
  
X = X.to_numpy()
Y = y.to_numpy()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the default model (max_depth=5)
model = DecisionTree(max_depth=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluate models with different max_depth values
evaluation_results = evaluate_model_performance(x_train, x_test, y_train, y_test)

# Write evaluation results to file
write_evaluation_to_file(evaluation_results, X, Y, x_train, x_test, "model_evaluation.txt")

# Create tree visualization
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
create_tree_graphviz(model.rootNode, feature_names, "decision_tree")

# Decision Boundary Plot (using petal length and petal width)
print("\nCreating Decision Boundary Plot...")
X_2d = X[:, [2, 3]]  # Petal length and petal width
Y_2d = Y

# Train model on 2D data with max_depth=3
model_2d = DecisionTree(max_depth=3)
model_2d.fit(X_2d, Y_2d)

# Create decision boundary plot
feature_names_2d = ['petal length (cm)', 'petal width (cm)']
class_names = ['setosa', 'versicolor', 'virginica']
plot_decision_boundary_2d(model_2d, X_2d, Y_2d, feature_names_2d, class_names, "Decision Tree Boundary (Petal Length vs Petal Width)")

