# SVM
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
from scipy import stats
import numpy as np

newdataX = np.load(r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/Python codes/New Features and PCA Training on all samples/VGG Features/vgg_features_raw__44100.npy')


print(np.shape(newdataX))
#np.random.shuffle(newdataX)
# np.random.permutation(newdataX)
#original = train_data
l = np.shape(newdataX)
print (l)
number = int((l[0]/5))
print (number)
y = np.repeat(np.arange(5), number)

y = np.reshape(y,[7500,1])
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
Effects_train = newdataX[0:1200,:]
Human_train = newdataX[1500:2700,:]
Music_train = newdataX[3000:4200,:]
Nature_train = newdataX[4500:5700,:]
Urban_train = newdataX[6000:7200,:]

Effects_test = newdataX[1200:1500,:]
Human_test = newdataX[2700:3000,:]
Music_test = newdataX[4200:4500,:]
Nature_test = newdataX[5700:6000,:]
Urban_test = newdataX[7200:7500,:]


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
    feature_train_cv = train_cv[:, 0:128]
    labels_train_cv = train_cv[:, -1]
    feature_test_cv = test_cv[:, 0:128]
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
    feature_train_cv = train_cv[:, 0:128]
    labels_train_cv = train_cv[:, -1]
    feature_test_cv = test_cv[:, 0:128]
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

    min_test_cv = 0
    max_test_cv = train_size_prop

    val_loss_array = []
    val_accuracy_array = []

    num_of_epochs = 3
    train_loss_array = np.empty([0, num_of_epochs])
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
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        history_callback = model.fit(feature_train_cv_normalized, labels_train_cv, epochs=num_of_epochs)
        loss_history = history_callback.history["loss"]
        print(loss_history)
        val_loss, val_acc = model.evaluate(feature_test_cv_normalized, labels_test_cv)
        print(val_loss)
        print(val_acc)
        loss_history = np.reshape(loss_history, (1, num_of_epochs))
        train_loss_array = np.concatenate((train_loss_array,loss_history),axis=0)
        val_loss_array.append(val_loss)
        val_accuracy_array.append(val_acc)

        min_test_cv += train_size_prop
        max_test_cv += train_size_prop
    # print (min_test_cv)
        train_data = original
    return train_loss_array, val_loss_array, val_accuracy_array
    # k_scores = np.append(k_scores, scores.mean())
    # print (k_scores)


if __name__ == "__main__":
    k = 10
    train_loss_array, val_loss_array, val_accuracy_array = cross_validation(k, Effects_train, Human_train, Music_train, Nature_train, Urban_train)
    Xtrain, Xtest, ytrain, ytest = split_the_data_and_zscorenormalize(train_data, test_data)
    print(train_loss_array)
    print(val_loss_array)
    print(val_accuracy_array)
    print(np.shape(train_loss_array))
    print(np.shape(val_loss_array))
    print(np.shape(val_accuracy_array))
    print(type(train_loss_array))
    print(type(val_loss_array))
    print(type(val_accuracy_array))
    #val_loss_array = np.reshape(val_loss_array,(1,10))
    #val_accuracy_array = np.reshape(val_accuracy_array, (1,10))
    lab = np.array(["Fold1", "Fold2", "Fold3", "Fold4", "Fold5",
                    "Fold6", "Fold7", "Fold8", "Fold9", "Fold10"])

    plt.plot(train_loss_array)
    plt.legend(labels=lab)
    plt.show()

    train_loss_array_mean = np.mean(train_loss_array, axis= 1)



    plt.plot(val_accuracy_array)
    plt.legend(labels=lab)
    plt.show()

    plt.plot(train_loss_array_mean, label='train_loss')
    plt.plot(val_loss_array,label='val_loss')
    plt.xlabel('Folds')
    plt.ylabel('Loss Function')
    plt.legend(loc='lower right')
    plt.title('train_loss vs val_loss')
    plt.show()

#    print (np.corrcoef(ypred,ytest))

#    test_size = 1 - train_size
#    train_fold = int(train_size_prop*k)

#    np.random.shuffle(train_data)
#    for i in range(k):