import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Comparison_Code import split_the_data_and_zscorenormalize
import pandas as pd

newdataX = np.load(r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/vggish/models/research/audioset/vgg_features.npy')
print (a.shape)


"""
newdataX = dataset.iloc[:, 2:43].values
train_rows = int(0.8 * newdataX.shape[0])
test_rows = newdataX.shape[0] - train_rows

train_data = newdataX[:train_rows]
test_data = newdataX[train_rows:train_rows+test_rows]

x_train, x_test, y_train, y_test = split_the_data_and_zscorenormalize(train_data, test_data)

#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

"""