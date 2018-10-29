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

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import csv
import os, os.path
import librosa
import librosa.display
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

DIRNAME = r'/Users/Mansoor/Desktop/MUSI 7100 Project/MSoS_challenge_2018_Development_v1-00/Development'
#OUTPUTFILE = r'/Users/Mansoor/Desktop/MUSI 7100 Project/MSoS_challenge_2018_Development_v1-00/Development'
def get_file_paths(dirname):                        # function to access file in the directory
    file_paths = [ ]
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

def input_sound(file):              #function to calculate the FS and time series input
    y = [ ]
    sr = [ ]
    y, sr = librosa.load(file,16000)
    return y,sr

def MFCC_calculation():
    files = get_file_paths(DIRNAME)
    file_count = len(files)
    a = np.zeros(file_count)
    feature_vector = []
    full_feature_vector = np.empty([0,40])
    i = 0
    print (file_count)
    for file in sorted(files):
        #print (file)
        (filepath, ext) = os.path.splitext(file)
        file_name = os.path.basename(file)
        if ext == '.wav':
            a = input_sound(file)
            filedata = a[0]
            nos = a[1]
            #S = librosa.feature.melspectrogram(filedata, sr=nos, n_mels=128)
            #log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(y = filedata, sr = nos, S = None, n_mfcc = 20, dct_type = 2, norm = 'ortho')
            mean_of_mfcc = np.mean(mfcc, axis=1)
            #normalized_mean_of_mfcc = librosa.util.normalize(mean_of_mfcc)
            std_of_mfcc = np.std(mfcc, axis=1)
            #normalized_std_of_mfcc = librosa.util.normalize(std_of_mfcc)
            feature_vector = np.hstack((mean_of_mfcc,std_of_mfcc))
            transposed_feature_vector = np.reshape(feature_vector, (1,40))
            #print (np.shape(transposed_feature_vector))
            #full_feature_vector = np.append(full_feature_vector,transposed_feature_vector, axis=0)
            full_feature_vector = np.concatenate((full_feature_vector,transposed_feature_vector), axis=0)
            #print (np.shape(full_feature_vector))
    return full_feature_vector            

if __name__ == "__main__":
    A = []
    B = []
    A = MFCC_calculation()
    scaler = MinMaxScaler()
    scaler.fit(A)
    B = scaler.transform(A)
    X = B
    l = np.shape(X)
    size = l[0]
    number = int((l[0])/5)
    y = np.repeat(np.arange(5),number)
    print (np.shape(X))
    print (np.shape(y))
    #lasso = linear_model.Lasso()
    #cv_results = cross_validate(lasso, X, y, return_train_score=True)
    #print (X_train.shape, y_train.shape)
    #print (X_test.shape, y_test.shape)
    #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    #clf = svm.SVC(kernel='linear', C=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
    print (np.shape(X_train))
    print (np.shape(X_test))
    kf = KFold(n_splits = 10)
    print (kf.get_n_splits(X_train))
 
    clf = SVC()
    clf.fit(X_train, y_train) 
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    
    scores = cross_val_score(clf, X, y, cv=5)
    y_pred = clf.predict(X_test)
    #b = clf.score(X_train, y_train, sample_weight=None)
    #print (clf.score(X_test, y_test) )
    print (y_pred)
    #print (b)
    #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    #print (clf.score(X_test,y_test))
    c = confusion_matrix(y_test, y_pred)
    print (c)
    d = accuracy_score(y_test, y_pred)
    print (d)
    """from sklearn import svm
    X = A
    y = [0, 1, 2, 3, 4]
    print (X)
    print (np.shape(X))
    print (y)
    clf = svm.SVC()
    clf.fit(X, y)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    
    #print (std_of_mfcc)
    #print (normalized_mean_of_mfcc)
    #print (normalized_std_of_mfcc)
            
    #        with open(OUTPUTFILE, 'a') as f:
     #            writer = csv.writer(f)
      #           writer.writerow(['file_name','mfcc_values'])
       #          writer.writerrow([file_name,f])

           

    audio_path = ('/Users/Mansoor/Desktop/MUSI 7100 Project/MSoS_challenge_2018_Development_v1-00/Development/Music/0AT.wav')
    y, sr = librosa.load(audio_path)
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc = 20)
    print (mfcc)
    mean_mfcc = np.mean(mfcc)
    std_mfcc = np.std(mfcc)

    plt.figure(figsize =(12, 6))
    plt.subplot(3,1,1)
    librosa.display.specshow(mfcc)
    plt.ylabel('MFCC')
    plt.colorbar()

    plt.subplot(3,1,2)
    plt.plot(mean_mfcc)
    plt.ylabel('Mean')
    plt.colorbar()

    plt.subplot(3,1,3)
    plt.plot(std_mfcc)
    plt.ylabel('Standard Deviation')
    plt.colorbar()

    plt.tight_layout()
    x= np.shape(mfcc)
    print (x)
    y= np.shape(mean_mfcc)
    print (y)
    z= np.shape(std_mfcc)
    print (z)
    #plt.show()
   # M = np.vstack([mfcc, mean_mfcc, std_mfcc]) """
