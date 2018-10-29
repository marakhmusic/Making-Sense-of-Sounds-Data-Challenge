# SVM
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


dataset_train = pd.read_csv('audio_dataset_training.csv')
dataset_test = pd.read_csv('audio_dataset_testing.csv')
traindataX = dataset_train.iloc[:, 2:43].values
testdataX = dataset_test.iloc[:, 2:42].values
print(np.shape(newdataX))
np.random.shuffle(newdataX)
# np.random.permutation(newdataX)
train_rows = int(0.8 * newdataX.shape[0])
test_rows = newdataX.shape[0] - train_rows

train_data = newdataX[:train_rows]
test_data = newdataX[train_rows:train_rows+test_rows]

# print ("Training data", train_data)
# print ("Testing data", test_data)
np.random.shuffle(train_data)
print(np.shape(train_data))
# X = train_data[:, 0:40]
# y = train_data[:, -1]
original = train_data


def split_the_data_and_zscorenormalize(train_cv, test_cv):
    feature_train_cv = train_cv[:, 0:40]
    labels_train_cv = train_cv[:, -1]
    feature_test_cv = test_cv[:, 0:40]
    labels_test_cv = test_cv[:, -1]
    feature_train_cv_normalized = stats.zscore(feature_train_cv, axis=0, ddof=0)
    feature_train_cv_mean = np.mean(feature_train_cv, axis=0)
    feature_train_cv_std = np.std(feature_train_cv, axis=0)
    feature_test_cv_normalized = np.divide(np.subtract(
        feature_test_cv, feature_train_cv_mean), (feature_train_cv_std))
    return feature_train_cv_normalized, feature_test_cv_normalized, labels_train_cv, labels_test_cv


def split_the_data_and_normalize(train_cv, test_cv):
    feature_train_cv = train_cv[:, 0:40]
    labels_train_cv = train_cv[:, -1]
    feature_test_cv = test_cv[:, 0:40]
    labels_test_cv = test_cv[:, -1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(feature_train_cv)
    feature_train_cv_normalized = scaler.transform(feature_train_cv)
    feature_train_cv_max = scaler.data_max_
    feature_train_cv_min = scaler.data_min_
    feature_test_cv_normalized = np.divide(np.subtract(
        feature_test_cv, feature_train_cv_min), (feature_train_cv_max - feature_train_cv_min))
    return feature_train_cv_normalized, feature_test_cv_normalized, labels_train_cv, labels_test_cv


def accuracy_calculator(labels_test_cv, predict_test_cv):
    c = confusion_matrix(labels_test_cv, predict_test_cv)
    accuracy = np.divide(np.sum(np.diag(c)), np.sum(c))
    return accuracy


def cross_validation(k, train_data):
    train_size_prop = int(np.divide(train_data.shape[0], k))
    k_range = list(range(1, 31))
    z_range = list(range(1, 31))
    k_scores = []
    min_test_cv = 0
    max_test_cv = train_size_prop
    Accuracy_matrix_knn = np.zeros((10, 31))
    Accuracy_matrix_svm = np.zeros((10, 31))
    for i in range(k):
        scores = []
        scores_svm = []
        test_cv = train_data[min_test_cv:max_test_cv, :]
        train_cv = np.delete(train_data, np.s_[min_test_cv:max_test_cv], 0)
        feature_train_cv_normalized, feature_test_cv_normalized, labels_train_cv, labels_test_cv = split_the_data_and_zscorenormalize(
            train_cv, test_cv)
        for k in k_range:
            neigh = KNeighborsClassifier(
                n_neighbors=k, weights='distance', p=2)
            neigh.fit(feature_train_cv_normalized, labels_train_cv)
            predict_test_cv = neigh.predict(feature_test_cv_normalized)
            Accuracy = accuracy_calculator(labels_test_cv, predict_test_cv)
            scores = np.append(scores, Accuracy)
            Accuracy_matrix_knn[i, k] = Accuracy

        for z in z_range:
            # print (z)
            clf = SVC(C=z, cache_size=200, class_weight='balanced', coef0=0.0,
                      decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
            clf.fit(feature_train_cv_normalized, labels_train_cv)
            predict_test_cv = clf.predict(feature_test_cv_normalized)
            Accuracy = accuracy_calculator(labels_test_cv, predict_test_cv)
            scores_svm = np.append(scores, Accuracy)
            Accuracy_matrix_svm[i, z] = Accuracy
        # print ("%d", i, scores)
        min_test_cv += train_size_prop
        max_test_cv += train_size_prop
    # print (min_test_cv)
        train_data = original
    return Accuracy_matrix_knn, Accuracy_matrix_svm
    # k_scores = np.append(k_scores, scores.mean())
    # print (k_scores)


if __name__ == "__main__":
    k = 10
    Accuracy_matrix_knn, Accuracy_matrix_svm = cross_validation(k, train_data)
    #Accuracy_matrix_knn = np.delete(Accuracy_matrix_knn, 0, 1)
    #Accuracy_matrix_svm = np.delete(Accuracy_matrix_svm, 0, 1)
    print(np.shape(Accuracy_matrix_knn))
    plt.plot(np.transpose(Accuracy_matrix_knn), '.')
    lab = np.array(["Fold1", "Fold2", "Fold3", "Fold4", "Fold5",
                    "Fold6", "Fold7", "Fold8", "Fold9", "Fold10"])
    plt.legend(labels=lab)
    plt.show()
    plt.plot(np.transpose(Accuracy_matrix_svm), '.')
    plt.legend(labels=lab)
    plt.show()

    Accuracy_knn_mean = np.mean(Accuracy_matrix_knn, axis=0)
    Accuracy_knn_std = np.std(Accuracy_matrix_knn, axis=0)

    Accuracy_svm_mean = np.mean(Accuracy_matrix_svm, axis=0)
    Accuracy_svm_std = np.std(Accuracy_matrix_svm, axis=0)
    print(Accuracy_knn_std)
    print(Accuracy_svm_std)
    #print (Accuracy_knn_mean)
    # print(Accuracy_svm_knn)
    plt.plot(Accuracy_knn_mean, '.', label='knn')
    plt.plot(Accuracy_svm_mean, '*', label='svm')
    plt.xlabel('Hyperparameter')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('KNN vs SVM')
    plt.show()

    # plt.plot(, '.')
    # plt.show()

    Xtrain, Xtest, ytrain, ytest = split_the_data_and_zscorenormalize(train_data, test_data)
    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)
    neigh.fit(Xtrain, ytrain)
    ypred = neigh.predict(Xtest)
    total_accuracy = accuracy_calculator(ytest, ypred)
    print(total_accuracy)
    c = confusion_matrix(ytest, ypred)
    print(c)
    n = c / c.astype(np.float).sum(axis=1)
    print(n)

    clf = SVC(C=2, cache_size=200, class_weight='balanced', coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    Accuracy = accuracy_calculator(ytest, ypred)
    print(Accuracy)
    c = confusion_matrix(ytest, ypred)
    print(c)
    n = c / c.astype(np.float).sum(axis=1)
    print(n)

#    print (np.corrcoef(ypred,ytest))

#    test_size = 1 - train_size
#    train_fold = int(train_size_prop*k)

#    np.random.shuffle(train_data)
#    for i in range(k):


"""
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train_normalized = scaler.transform(X_train)
X_max = scaler.data_max_
X_min = scaler.data_min_
X_den = X_max - X_min
X_test_scaled = np.subtract(X_test, X_min)
X_test_normalized = np.divide(X_test_scaled, X_den)


def KNN(X_train_normalized, y_train):
        k_range = list(range(1, 31))
        k_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train_normalized, y_train, cv=10, scoring='accuracy')
            k_scores.append(scores.mean())
        print("The scores for different values of neighbours is", k_scores)

        plt.plot(k_range, k_scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Accuracy')
        plt.show()


# print (k_scores.index(max(k_scores)))
clf = SVC()
clf.fit(X_train_normalized, y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

y_pred = clf.predict(X_test_normalized)
c = confusion_matrix(y_test, y_pred)
d = accuracy_score(y_test, y_pred)
print(np.sum(c))
print(c)
print(d)
accuracy = np.zeros(5)
temp = np.zeros(5)
i = 0
j = 0
while i < 5:
    while j < 5:
        if i == j:
            temp[i] = c[i, j]
            accuracy[i] = c[i, j]/np.sum(c[i])
            j += 1
        i += 1
Complete_Accuracy = np.sum(temp)/np.sum(c)
print ("The number of 0,1,2,3,4", accuracy)
print ("The complete accuracy is", Complete_Accuracy)


from Comparison_Code import split_the_data_and_zscorenormalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
dataset = pd.read_csv('audio_dataset.csv')
newdataX = dataset.iloc[:, 2:42].values





Xtrain, Xtest, ytrain, ytest = split_the_data_and_zscorenormalize(train_data, test_data)
pca = PCA(.95)
pca.fit(Xtrain)
print(pca.components_)
print(pca.explained_variance_)

np.random.shuffle(newdataX)
train_rows = int(0.8 * newdataX.shape[0])
test_rows = newdataX.shape[0] - train_rows
train_data = newdataX[:train_rows]
test_data = newdataX[train_rows:train_rows+test_rows]
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(train_data)
print(train_data.shape)
print(projected.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax1 = Axes3D.scatter(xs=projected[:, 0], ys=projected[:, 1], zs=0, zdir='z',
                     s=20, c=None, depthshade=True)

plt.scatter(projected[:, 0], projected[:, 1],
            c=train_data[:, -1], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Accent', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()
"""
