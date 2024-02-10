# mas mercedes company a chces zistit, aky ludia najpravdepodobnejsie kupia auto od teba, aby si vedel zostavit reklamu

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("Car_Purchases.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# equation of the Hyperplane
# y = wx - b
# b, w are model parameters

# gradient descent (optimalization)
# w = w - learning rate*dw
# b = b - learning rate*db

class SVM_classifier():
    # initiating the hyperparameters
    def __init__(self, learning_rate, num_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_parameter = lambda_parameter

    # fitting the dataset to SVM Classifier
    def fit(self, X, y):
        # m number of rows in training dataset
        # n number of columns in training dataset

        self.m, self.n = X_train.shape

        # initiating the weights values and bias value

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X
        self.y = y


        # implementing Gradient Descent algorithm for Optimalization 

        for i in range(self.num_iterations):
            self.update_weights()
    
    # function for updating the weights and bias value
    def update_weights(self):
        y_label = np.where(self.y <= 0, -1, 1)

        # gradient
        for index, x_i in enumerate(self.X):

            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

            if(condition == True):
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]

        
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
    
    # predict the label for a given input value
    def predict(self, X):
        output = np.dot(X, self.w) - self.b

        # predicted_labels will be +1 or -1
        predicted_labels = np.sign(output)

        # true labels 1 or 0
        y_hat = np.where(predicted_labels <= -1, 0, 1)

        return y_hat

model = SVM_classifier(learning_rate=0.001, num_iterations=1000, lambda_parameter=0.01)

print(y)

# Data Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X.shape, X_train.shape, X_test.shape)

# Support Vector Machine Classifier
classifier = SVM_classifier(learning_rate=0.001, num_iterations=1000, lambda_parameter=0.01)

# training the SVM classifier with training data
classifier.fit(X_train, y_train)

# 0 - wont buy a mercedes
print(classifier.predict(scaler.transform([[35,108000]])))

# Accuracy test
y_pred = classifier.predict(X_test)
test_data_accuracy = accuracy_score(y_test, y_pred)
print(f'This is test_data_accuracy {test_data_accuracy}')
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))