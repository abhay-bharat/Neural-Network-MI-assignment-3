import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_LBW_Dataset.csv")
#create features and target variable dataframe
target = df.iloc[:, 9].to_numpy()
features = df.iloc[:, 0:9].to_numpy()

#Perform train-test split, with random state introduced
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 7) 

###Creating neural network with no hidden layers
def initialise_weight(c, r): # r-row, c-column
    wt = np.array(np.random.rand(r, c)[0])
    return wt

def forward_propagation(X, weight, b):
    y_list = []
    for ftr in X:
        y_list.append(np.dot(ftr, weight))
    y = np.array(y_list)
    y = y+b
    return y

def sigmoid_activation(y):
    Oy = 1//(1 + np.exp(-y))
    return Oy

def compute_error(true_y, y):
    e  = (true_y - y)**2
    return (e.sum())

#initialise weight matrix
weight = initialise_weight(X_train.shape[1], 1) #because binary classification, so 2nd parameter = 1
#weight matrix of dimension 1x9, here
# bias = np.random.rand(1, 1) #creating a random bias array of dimension 2x1 here
bias = 1 #bias = 1
iterations = 0
while(iterations < 10):
    y_hat = forward_propagation(X_train, weight, bias) #y_hat will have same dimension as y_train
    Oy_hat = sigmoid_activation(y_hat)
    error = compute_error(y_train, Oy_hat) #batch training, error computed as summation of all data element errors(i.e. batch gradient descent)
    number_of_train = X_train.shape[0]
    accuracy = (number_of_train-error)/number_of_train
    print("Training accuracy is ",accuracy)
    ###now perform backPropagation 
    back_propagation(weight, bias)
    iterations += 1