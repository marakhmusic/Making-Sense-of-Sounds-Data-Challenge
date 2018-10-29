import tensorflow.keras as keras
import tensorflow as tf
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

print(tf.__version__)

newdataX = np.load(r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/Python codes/New Features and PCA Training on all samples/VGG Features/vgg_features_postprocessed.npy')
testing_dataX = np.load(r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/Python codes/New Features and PCA Training on all samples/VGG Features/features_to_test_on_postprocessed.npy')
l = np.shape(newdataX)
number = int((l[0]/5))
y = np.repeat(np.arange(5), number)
labels = np.reshape(y,[7500,1])
data = preprocessing.scale(newdataX)

testing_data = preprocessing.scale(testing_dataX)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, epochs=30)
predictions = model.predict(testing_data)
print(predictions.astype(int))