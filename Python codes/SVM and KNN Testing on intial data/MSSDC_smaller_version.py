"""Making Sense of Sounds Data Challenge

Task Description:
The task is to classify audio data as belonging to one of five broad categories,
which were derived from human classification. In a psychological experiment at
the University of Salford, participants were asked to categorise 60 sound types,
chosen so as to represent the most commonly used search terms on Freesound.org.
Five principal categories were identified by correspondence analysis and
hierarchical cluster analysis of the human data:
Nature
Human
Music
Effects
Urban
"""
# import required libraris Pandas, Numpy and Scikit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os
import os.path
import librosa
import librosa.display
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import tenfoldcrossval as tf
import SVM_classifier_program as svm
DIRNAME = r'/Users/Mansoor/Desktop/MUSI 7100 Project/MSoS_challenge_2018_Development_v1-00/Testing_learningpart'
# OUTPUTFILE = r'/Users/Mansoor/Desktop/MUSI 7100 Project/MSoS_challenge_2018_Development_v1-00/Development'
# path


def get_file_paths(dirname):                        # function to access file in the directory
    file_paths = []
    # os.walk generates names in the
    for root_dirpath, directories, files in os.walk(dirname):
        for filename in files:                                            # directory tree, dirpath, dirnames, filenames
            # os.path.join converts a relative path to an absolute one
            filepath = os.path.join(root_dirpath, filename)
            file_paths.append(filepath)
    print ("The file paths will be", file_paths)
    return file_paths


def input_sound(file):  # function to calculate the FS and time series input
    y = []
    sr = []
    y, sr = librosa.load(file, 16000)
    return y, sr


def MFCC_calculation():
    files = get_file_paths(DIRNAME)                 # function called to get the file paths
    file_count = len(files)                         # count the number of files in the directory
    a = np.zeros(file_count)
    File_names = []
    feature_vector = []
    # 40 features to be calculated Mean and Std of 20 MFCCs
    full_feature_vector = np.empty([0, 40])
    i = 0
    print (file_count)
    for file in sorted(files):          # loop to access each file
        # print (file)
        (filepath, ext) = os.path.splitext(file)  # get extension of the file
        file_name = os.path.basename(file)  # get the file name
        if ext == '.wav':
            File_names.append(file_name)
            a = input_sound(file)       # calculate features on only wav files
            filedata = a[0]         # file data stores time series data
            nos = a[1]               # nos stores sampling frequency
            # S = librosa.feature.melspectrogram(filedata, sr=nos, n_mels=128)
            # log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(y=filedata, sr=nos, S=None,             # calculate 20 mfccs
                                        n_mfcc=20, dct_type=2, norm='ortho')
            mean_of_mfcc = np.mean(mfcc, axis=1)                # calculate mean of mfccs
            # normalized_mean_of_mfcc = librosa.util.normalize(mean_of_mfcc)
            std_of_mfcc = np.std(mfcc, axis=1)              # calculate standard deviations of mfccs
            # normalized_std_of_mfcc = librosa.util.normalize(std_of_mfcc)
            # combine the two features in one array feature vector
            feature_vector = np.hstack((mean_of_mfcc, std_of_mfcc))
            # reshape them in  (sample, feature) format
            transposed_feature_vector = np.reshape(feature_vector, (1, 40))
            # print (np.shape(transposed_feature_vector))
            # full_feature_vector = np.append(full_feature_vector,transposed_feature_vector, axis=0)
            full_feature_vector = np.concatenate(
                (full_feature_vector, transposed_feature_vector), axis=0)
            # print (np.shape(full_feature_vector))
    print (type(File_names))
    return full_feature_vector, File_names


if __name__ == "__main__":
    X = []
    M = MFCC_calculation()                  # call the function in main()
    X = M[0]
    # print (np.shape(X))
    File_names = np.asarray(M[1])
    # print (File_names)
    l = np.shape(X)                             # get dimensions of X
    df1 = pd.DataFrame(data=File_names)
    df2 = pd.DataFrame(data=X)
    df1 = df1.join(df2, lsuffix="File_names", rsuffix="File_features")
    number = int((l[0])/5)
    # use number of samples to find number of classes/labels (0-4)
    y = np.repeat(np.arange(5), number)
    df3 = pd.DataFrame(data=y, columns=['Labels'])
    df1 = df1.join(df3, how='left', lsuffix='_train_test', rsuffix='_labels')
    df1.to_csv('audio_dataset.csv')
    print (df1)
    # print (df1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=6)   # split the data into train and test
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_max = scaler.data_max_
    X_min = scaler.data_min_
    X_den = X_max - X_min
    X_test_scaled = np.subtract(X_test, X_min)
    X_test_normalized = np.divide(X_test_scaled, X_den)
    # print ("The norm", X_test_normalized)
    # print ("Hey", X_test[0, :])
    # X_scaled = preprocessing.scale(X_train)
    # print("Test Values", X_test)
    # print("Min Values", X_min)
    # print("Scaled Values", X_test_scaled)
    # print (y_train)
    print ("Test data", np.shape(X_test_scaled))
    print ("Train data", np.shape(X_train_scaled))
    print ("Train labels", np.shape(y_train))
    print ("Test labels", np.shape(y_test))
    print ("The accuracy score for a KNN classifier is", tf.KNN_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test))
    print ("The accuracy score for a SVM classifier is", svm.SVM_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test))
    test = tf.KNN_classifier(X_train_scaled, y_train, X_test_scaled, y_test)
    confusion = test[0]
    accuracy = np.zeros(5)
    print("Type", type(confusion))
    i = 0
    j = 0
    while i < 5:
        while j < 5:
            if i == j:
                accuracy[i] = confusion[i, j]/np.sum(confusion[i])
            j += 1
        i += 1
    print ("The number of 0,1,2,3,4", accuracy)

    # print(np.shape(X_train))
