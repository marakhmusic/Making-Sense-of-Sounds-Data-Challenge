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

dataset = pd.read_csv('test_comparison_code.csv')
newdataX = dataset.iloc[:, 1:6].values
np.random.shuffle(newdataX)
# np.random.permutation(newdataX)
train_rows = int(0.8 * newdataX.shape[0])
test_rows = newdataX.shape[0] - train_rows

train_data = newdataX[:train_rows]
test_data = newdataX[train_rows:train_rows+test_rows]


if __name__ == "__main__":
    feature_train_cv = train_data[:, 0:4]
    labels_train_cv = train_data[:, -1]
    feature_test_cv = test_data[:, 0:4]
    labels_test_cv = test_data[:, -1]
    feature_train_cv_normalized = stats.zscore(feature_train_cv, axis=0, ddof=1)
    feature_train_cv_mean = np.mean(feature_train_cv, axis=0)
    feature_train_cv_std = np.std(feature_train_cv, axis=0)
    feature_test_cv_normalized = np.divide(np.subtract(
        feature_test_cv, feature_train_cv_mean), (feature_train_cv_std))
    print (feature_train_cv_normalized)
    print (feature_test_cv_normalized)

    feature_train_cv = train_data[:, 0:4]
    labels_train_cv = train_data[:, -1]
    feature_test_cv = test_data[:, 0:4]
    labels_test_cv = test_data[:, -1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(feature_train_cv)
    feature_train_cv_normalized = scaler.transform(feature_train_cv)
    feature_train_cv_max = scaler.data_max_
    feature_train_cv_min = scaler.data_min_
    feature_test_cv_normalized = np.divide(np.subtract(
        feature_test_cv, feature_train_cv_min), (feature_train_cv_max - feature_train_cv_min))
    print (feature_train_cv_normalized)
    print (feature_test_cv_normalized)
