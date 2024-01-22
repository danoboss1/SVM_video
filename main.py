# basic intuition

# https://www.youtube.com/watch?v=_YPScrckx28
# inspirovat sa prvou minutou tohto videa

# CODE

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('Car_Purchase.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC 
classifier = SVC(kernel = 'linear', random_state=0)
classifier.fit(X_train, y_train)

# vysledok 0
# print(classifier.predict(sc.transform([[31,74000]])))

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)),1))