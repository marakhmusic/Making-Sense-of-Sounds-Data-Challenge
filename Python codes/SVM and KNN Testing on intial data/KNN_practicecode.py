import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os
import os.path
import librosa
import librosa.display
from vggish_input import waveform_to_examples
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import tenfoldcrossval as tf




DIRNAME = r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/MSoS_challenge_2018_Development_v1-00/Development'
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
    print("The file paths will be", file_paths)
    return file_paths


def input_sound(file):  # function to calculate the FS and time series input
    y, sr = librosa.load(file)
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
    print(file_count)
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
    print(type(File_names))
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
