import tensorflow.keras as keras
import tensorflow as tf
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt

print(tf.__version__)

newdataX = np.load(r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/Python codes/New Features and PCA Training on all samples/VGG Features/vgg_features_postprocessed.npy')
l = np.shape(newdataX)
number = int((l[0]/5))
y = np.repeat(np.arange(5), number)
y = np.reshape(y,[7500,1])
newdataX = np.concatenate((newdataX,y),axis=1)
print (newdataX)
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
#train_data = np.random.shuffle(train_data,axis =0)
test_data = np.concatenate((Effects_test,Human_test,Music_test,Nature_test,Urban_test),axis=0)
#test_data = np.random.shuffle(test_data,axis=0)
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



x_train, x_test, y_train, y_test = split_the_data_and_zscorenormalize(train_data, test_data)


#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history_callback = model.fit(x_train, y_train, epochs=10)
train_loss, train_accuracy = model.evaluate(x_train, y_train)

loss_history = history_callback.history["loss"]
print(np.shape(loss_history))
print(type(loss_history))
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

"""
writer_val = tf.summary.FileWriter('./logs/plot_val')
writer_train = tf.summary.FileWriter('./logs/plot_train')
loss_var = tf.Variable(0.0)
tf.summary.scalar("loss", loss_var)
write_op = tf.summary.merge_all()
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
for i in range(100):
 # loss validation
 summary = session.run(write_op, {loss_var: np.random.rand()})
 writer_val.add_summary(summary, i)
 writer_val.flush()
 # loss train
 summary = session.run(write_op, {loss_var: np.random.rand()})
 writer_train.add_summary(summary, i)
 writer_train.flush()
"""