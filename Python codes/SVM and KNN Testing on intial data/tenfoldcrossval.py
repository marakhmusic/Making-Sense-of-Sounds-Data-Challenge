from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# matplotlib inline
# read in the iris data


def KNN_classifier(X_train, y_train, X_test, y_test):

    k_range = list(range(1, 10))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
        k_scores.append(scores.mean())
    print("The scores for different values of neighbours is", k_scores)

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
# print (k_scores.index(max(k_scores)))
    knn = KNeighborsClassifier(n_neighbors=11)
    # final_scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy').mean()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    c = confusion_matrix(y_test, y_pred)
    d = accuracy_score(y_test, y_pred)

    return c, d
