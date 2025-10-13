import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# print(dataset.feature_names)
# print(dataset.data)
# print (dataset.target_names)
# print(dataset.target)

class LinearRegression:
    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.cost_history = []

    def fit( self, x, y):
        x = self.preprocess_data(x, is_training = True)
        features = x.shape[1]
        inputCount = x.shape[0]
        self.weights = np.zeros(x.shape[1])
        self.bias = 0.0
        epsillon = 1e-7
        prev_cost = float('inf')
        for i in range(self.iterations):
            y_pred = np.dot(x, self.weights) + self.bias
            cost = np.mean((y_pred - y)**2)

            #check for convergance
            #if(cost <= epsillon):
            if(abs(cost - prev_cost) < epsillon):
                print("convergance yay---y")
                break;
            cost_change = np.dot(x.T, (y_pred -y)) / inputCount
            self.cost_history.append(cost)
            self.weights -= self.learning_rate * cost_change
            self.bias -= self.learning_rate * np.mean(y_pred - y)


    def predict(self, x):
        x = self.preprocess_data(x)
        return np.dot(x, self.weights) + self.bias
    
    def preprocess_data(self, data, is_training = False):
        if is_training:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        return (data - self.mean) / self.std
    
def create_and_run_model(iterations, learning_rate, generate_visualization = True, log_metrics_to_file = False):  
    #model = LinearRegression(1000,.001)
    dataset = fetch_california_housing()
    model = LinearRegression(iterations, learning_rate)
    x = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    pred_mse = np.mean((y_pred - y_test) ** 2)

    y_test_mean = np.mean(y_test)
    sst = np.sum((y_test - y_test_mean)**2)
    sse = np.sum((y_pred - y_test)**2)
    r2 = 1 - sse/sst
    print(f"error is:" , pred_mse)
    print(f"r2 score is:" , r2)
    
    if log_metrics_to_file:
        # write the iteration, r2 and error to a file
        with open('r2_and_error.txt', 'a') as f:
            if f.tell() == 0:
                f.write(f"{'Iterations':<12}{'Learning Rate':<15}{'R2 Score':<12}{'MSE':<12}\n")
            # Write values in aligned columns
            f.write(f"{model.iterations:<12}{model.learning_rate:<15}{r2:<12.6f}{pred_mse:<12.6f}\n")
    

    if(generate_visualization):
        plt.plot( range(model.iterations), model.cost_history)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost(MSE)')
        plt.title('Learning curve vs Iterations')
        plt.legend()

        plt.savefig('learning_curve.png')

        plt.close()
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2)

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True)
        plt.savefig('actual_vs_predicted.png')
# create_and_run_model(1000,.001, False,log_metrics_to_file = True)
# create_and_run_model(1000,.01, False,log_metrics_to_file = True)
# create_and_run_model(10000,.01, False,log_metrics_to_file = True)
# create_and_run_model(10000,.02, False,log_metrics_to_file = True)
# create_and_run_model(1000,.02, False,log_metrics_to_file = True)
# create_and_run_model(10000,.03, False,log_metrics_to_file = True)
# create_and_run_model(1000,.025, False,log_metrics_to_file = True)
# create_and_run_model(1000,.026, False,log_metrics_to_file = True)
# create_and_run_model(1000,.024, False,log_metrics_to_file = True)
# create_and_run_model(1000,.027, False,log_metrics_to_file = True)

create_and_run_model(1000,.026, True, log_metrics_to_file = False)


