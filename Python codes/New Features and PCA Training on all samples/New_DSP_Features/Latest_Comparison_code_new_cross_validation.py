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
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

newdataX = np.load(r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/Python codes/New Features and PCA Training on all samples/New_DSP_Features/new_features_enhanced.npy')


print(np.shape(newdataX))
#np.random.shuffle(newdataX)
# np.random.permutation(newdataX)
#original = train_data
l = np.shape(newdataX)
print (l)
number = int((l[0]/5))
print (number)
y = np.repeat(np.arange(5), number)

y = np.reshape(y,[1500,1])
print(y)
print (np.shape(y))
newdataX = np.concatenate((newdataX,y),axis=1)
print (newdataX)
print (newdataX.shape)

#train_rows = int(0.8 * newdataX.shape[0])
#test_rows = newdataX.shape[0] - train_rows

#train_data = newdataX[:train_rows]
#test_data = newdataX[train_rows:train_rows+test_rows]

# print ("Training data", train_data)
# print ("Testing data", test_data)
#np.random.shuffle(train_data)

# X = train_data[:, 0:40]
# y = train_data[:, -1]
Effects_train = newdataX[0:250,:]
Human_train = newdataX[300:550,:]
Music_train = newdataX[600:850,:]
Nature_train = newdataX[900:1150,:]
Urban_train = newdataX[1200:1450,:]

Effects_test = newdataX[250:300,:]
Human_test = newdataX[550:600,:]
Music_test = newdataX[850:900,:]
Nature_test = newdataX[1150:1200,:]
Urban_test = newdataX[1450:1500,:]


train_data = np.concatenate((Effects_train,Human_train,Music_train,Nature_train,Urban_train),axis=0)
test_data = np.concatenate((Effects_test,Human_test,Music_test,Nature_test,Urban_test),axis=0)
print(np.shape(train_data))
print(np.shape(test_data))
print(np.max(train_data))
print(np.max(test_data))
print(np.isnan(train_data).any())
print(np.isnan(test_data).any())
original = train_data



def split_the_data_and_zscorenormalize(train_cv, test_cv):
    feature_train_cv = train_cv[:, 0:129]
    labels_train_cv = train_cv[:, -1]
    feature_test_cv = test_cv[:, 0:129]
    labels_test_cv = test_cv[:, -1]
    feature_train_cv_normalized = stats.zscore(feature_train_cv, axis=0, ddof=0)
    feature_train_cv_mean = np.mean(feature_train_cv, axis=0)
    feature_train_cv_std = np.std(feature_train_cv, axis=0)
    feature_test_cv_normalized = np.divide(np.subtract(
        feature_test_cv, feature_train_cv_mean), (feature_train_cv_std))
    x = np.where(np.isnan(feature_train_cv_normalized))
    y = np.where(np.isnan(feature_test_cv_normalized))
    feature_train_cv_normalized[x[0],x[1]] = 10**-5
    feature_test_cv_normalized[y[0],y[1]] = 10**-5
    return feature_train_cv_normalized, feature_test_cv_normalized, labels_train_cv, labels_test_cv


def split_the_data_and_normalize(train_cv, test_cv):
    feature_train_cv = train_cv[:, 0:129]
    labels_train_cv = train_cv[:, -1]
    feature_test_cv = test_cv[:, 0:129]
    labels_test_cv = test_cv[:, -1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(feature_train_cv)
    feature_train_cv_normalized = scaler.transform(feature_train_cv)
    feature_train_cv_max = scaler.data_max_
    feature_train_cv_min = scaler.data_min_
    feature_test_cv_normalized = np.divide(np.subtract(
        feature_test_cv, feature_train_cv_min), (feature_train_cv_max - feature_train_cv_min))
    x = np.where(np.isnan(feature_train_cv_normalized))
    y = np.where(np.isnan(feature_test_cv_normalized))
    feature_train_cv_normalized[x[0],x[1]] = 10**-5
    feature_test_cv_normalized[y[0],y[1]] = 10**-5
    return feature_train_cv_normalized, feature_test_cv_normalized, labels_train_cv, labels_test_cv


def accuracy_calculator(labels_test_cv, predict_test_cv):
    c = confusion_matrix(labels_test_cv, predict_test_cv)
    accuracy = np.divide(np.sum(np.diag(c)), np.sum(c))
    return accuracy


def cross_validation(k, Effects_train, Human_train, Music_train, Nature_train, Urban_train):
    train_size_prop = int(np.divide(Effects_train.shape[0], k))
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
        Effects_test_cv = Effects_train[min_test_cv:max_test_cv, :]
        Human_test_cv = Human_train[min_test_cv:max_test_cv, :]
        Music_test_cv = Music_train[min_test_cv:max_test_cv, :]
        Nature_test_cv = Nature_train[min_test_cv:max_test_cv, :]
        Urban_test_cv = Urban_train[min_test_cv:max_test_cv, :]
        test_cv = np.concatenate((Effects_test_cv, Human_test_cv, Music_test_cv, Nature_test_cv, Urban_test_cv), axis=0)

        Effects_train_cv = np.delete(Effects_train, np.s_[min_test_cv:max_test_cv],0)
        Human_train_cv = np.delete(Human_train, np.s_[min_test_cv:max_test_cv], 0)
        Music_train_cv = np.delete(Music_train, np.s_[min_test_cv:max_test_cv], 0)
        Nature_train_cv = np.delete(Nature_train, np.s_[min_test_cv:max_test_cv], 0)
        Urban_train_cv = np.delete(Urban_train, np.s_[min_test_cv:max_test_cv], 0)
        train_cv = np.concatenate((Effects_train_cv, Human_train_cv, Music_train_cv, Nature_train_cv, Urban_train_cv), axis=0)
        feature_train_cv_normalized, feature_test_cv_normalized, labels_train_cv, labels_test_cv = split_the_data_and_zscorenormalize(train_cv, test_cv)
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
    Accuracy_matrix_knn, Accuracy_matrix_svm = cross_validation(k, Effects_train, Human_train, Music_train, Nature_train, Urban_train)
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

    clf = SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
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