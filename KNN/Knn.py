from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
  

class Knn :
    def __init__(self, k):
        self.k = k
        self.mean =  None
        self.std = None
        self.x_train = None
        self.y_train = None


    def fit(self, x_train, y_train):
        x_standarized = self._standarize_data(x_train, True)
        self.x_train = x_standarized
        self.y_train = y_train
        
    def predict(self, x_input):
        x_standarized = self._standarize_data(x_input)
        #for each x_standarized calculate distance from all x_train and sum all of those 
        # as the dimension of x_standarized and x_train are different
        # use brodcasting trick 
        x_standarized_mod = x_standarized[:,np.newaxis,:]
        x_train_mod = self.x_train[np.newaxis,:,:]
        diff_mat = x_standarized_mod - x_train_mod
        euclidean_distance =  np.sqrt(np.sum(diff_mat**2, axis=2))

        k_min_indicies = np.argpartition(euclidean_distance, self.k, axis=1)[:,:self.k]
        
        #k_min_reshaped = k_min_indicies[:,np.newaxis,:]
        y_train = self.y_train.ravel()
        k_min_y_train = y_train[k_min_indicies]

        y_predict = np.array([np.bincount(k_min.astype(int)).argmax() for k_min in k_min_y_train])

        return y_predict

    def _standarize_data(self, x, is_training_data=False):
        if(is_training_data):
            self.mean=np.mean(x,axis=0)
            self.std = np.std(x,axis=0)
        return (x - self.mean)/self.std

def run_model(k, use_less_feature=False, draw_decision_boundry_graph = False):
    # fetch dataset 
    iris = fetch_ucirepo(id=53) 
    x = iris.data.features if not use_less_feature else iris.data.features[['petal length', 'petal width']]
    y = iris.data.targets

    le = LabelEncoder()
    y_encoder = le.fit_transform(y)

    x_np = x.to_numpy()
    y_np = np.array(y_encoder)
    model = Knn(k)
    x_train, x_test, y_train, y_test = train_test_split(x_np,y_np,test_size=0.2)
    model.fit(x_train, y_train)
    #draw decision boundry graph
    if draw_decision_boundry_graph:
        x_petal_length_min, x_petal_length_max = x_np[:,0].min()-1 , x_np[:,0].max()+1
        x_petal_width_min, x_petal_width_max = x_np[:,1].min()-1 , x_np[:,1].max()+1
        x_points, y_points = np.meshgrid(np.arange(x_petal_length_min, x_petal_length_max, 0.01), np.arange(x_petal_width_min, x_petal_width_max,0.01))
        model_out = model.predict(np.c_[x_points.ravel(), y_points.ravel()])
        model_out_trans = le.fit_transform(model_out)
        model_out = model_out.reshape(x_points.shape)
        plt.contourf(x_points, y_points, model_out, alpha=0.3, cmap=plt.cm.RdYlBu)
        plt.scatter(x_np[:, 0], x_np[:, 1], c=y_np, s=40, edgecolor='k', cmap=plt.cm.RdYlBu)
        plt.title('KNN Decision Boundary')
        plt.xlabel('Petal length')
        plt.ylabel('Petal width')
        plt.savefig(f'decision_boundry_{k}.png')
        plt.close()
    else: 
        y_pred = model.predict(x_test)
        accuracy = np.mean(y_pred==y_test)
        return accuracy


def main():
    k_test = [1,3,5,7,11]
    accuracy = [run_model(k) for k in k_test]
    plt.plot(k_test, accuracy)
    plt.title('Accuracy vs K-value')
    plt.xlabel('K_Value')
    plt.ylabel('Accuracy')
    plt.savefig('Accuracy_vs_K_value.png')
    plt.close()
    headers = ['K_Value','Accuracy']
    table = list(zip(k_test,accuracy))
    with open('Accuracy_vs_K_value.txt', 'a') as f:
        f.write(tabulate(table,headers,tablefmt="grid"))

    run_model(k=1, use_less_feature=True, draw_decision_boundry_graph = True)
    run_model(k=15, use_less_feature=True, draw_decision_boundry_graph = True)



if __name__ == "__main__":
    main()