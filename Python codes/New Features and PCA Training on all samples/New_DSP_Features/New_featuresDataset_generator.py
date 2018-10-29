import numpy as np
import pandas as pd
import csv
import os
import os.path
import librosa
import librosa.display
import matplotlib.pyplot as plt



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
    y, sr = librosa.load(file,sr=None)
    return y, sr


def MFCC_calculation():
    files = get_file_paths(DIRNAME)
    print (files)# function called to get the file paths
    file_count = len(files)                         # count the number of files in the directory
    a = np.zeros(file_count)
    File_names = []
    feature_vector = []
    # 40 features to be calculated Mean and Std of 20 MFCCs
    full_feature_vector = np.empty([0, 129])
    i = 0
    print(file_count)
    for file in sorted(files):          # loop to access each file
        # print (file)
        (filepath, ext) = os.path.splitext(file)  # get extension of the file
        file_name = os.path.basename(file)  # get the file name
        #feature_vector = np.empty(0,130)
        if ext == '.wav':
            File_names.append(file_name)
            a = input_sound(file)       # calculate features on only wav files
            filedata = a[0]         # file data stores time series data
            nos = a[1]               # nos stores sampling frequency
            # S = librosa.feature.melspectrogram(filedata, sr=nos, n_mels=128)
            # log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(y=filedata, sr=nos, S=None,             # calculate 20 mfccs for 2048 FFT and 512 hop
                                        n_mfcc=20, dct_type=2, norm='ortho')

            mean_of_mfcc = np.mean(mfcc, axis=1)    #first 20 features of featurevector (20 means of MFCC( over 431 blocks))(index 0-19)
            std_of_mfcc = np.std(mfcc, axis=1)      #Next 20 features of featurevector (20 std of MFCC)(index 20-39)

            delta_mfcc= librosa.feature.delta(mfcc, width=21, order=1)     # delta MFCCs Check with Sid
            mean_of_delta_mfcc = np.mean(delta_mfcc, axis=1) #index 40-59
            std_of_delta_mfcc = np.std(delta_mfcc, axis=1) #index 60-79
            double_delta_mfcc = librosa.feature.delta(mfcc, width=21, order=2)  # double delta MFCCs
            mean_of_double_delta_mfcc = np.mean(double_delta_mfcc, axis=1) #index 80-99 feature
            std_of_double_delta_mfcc = np.std(double_delta_mfcc, axis=1) #index 100-119 feature

            stft_of_file = np.abs(librosa.stft(y=filedata, n_fft=2048, hop_length=512) ) # Short time fourier transform
            rms = librosa.feature.rmse(S=stft_of_file)

            mean_of_rms = np.mean(rms) #index 120 feature
            std_of_rms = np.std(rms) #index 121 feature
            rms_upper_band = np.sum(librosa.core.amplitude_to_db(S = stft_of_file[512:1024]),axis=0)
            rms_lower_band = np.sum(librosa.core.amplitude_to_db(S = stft_of_file[0:512]),axis=0)
            rms_whole_upper_band = np.sum(rms_upper_band)
            rms_whole_lower_band = np.sum(rms_lower_band)
            Energy_whole_clip = rms_whole_upper_band/rms_whole_lower_band   #feature
            Energy_ratio_per_frame = rms_upper_band/rms_lower_band

            Energy_ratio_per_frame_mean = np.mean(Energy_ratio_per_frame) #feature
            Energy_ratio_per_frame_std = np.std(Energy_ratio_per_frame) #feature

            onset_env = librosa.onset.onset_strength(y=filedata, sr=nos, hop_length = 512, aggregate = np.median)
            peaks_of_spectra = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 10)
            peaks_of_spectra = peaks_of_spectra.astype(int)
            sum_of_peaks_of_spectra = np.sum((stft_of_file[:,peaks_of_spectra]))
            sum_of_spectra = np.sum((stft_of_file))

            peak_to_stft_ratio = sum_of_peaks_of_spectra/sum_of_spectra #feature

            each_peak_to_stft_ratio = (stft_of_file[:,peaks_of_spectra])/np.sum((stft_of_file))


            mean_of_ratio_of_peak_to_spectra = np.mean(each_peak_to_stft_ratio) #feature
            std_of_ratio_of_peak_to_spectra = np.std(each_peak_to_stft_ratio) #feature
            #rms_silence_vs_non = np.copy(rms)
            #thresh = 0.5
            #super_threshold_indices = rms_silence_vs_non < thresh
            #rms_silence_vs_non[super_threshold_indices] = 0
            #sper_threshold_indices = rms_silence_vs_non > thresh
            #rms_silence_vs_non[sper_threshold_indices] = 1
            #Num_of_1_and_0 = np.unique(rms_silence_vs_non,return_counts=True)
            #num_of_0 = Num_of_1_and_0[1][0]
            #num_of_1 = Num_of_1_and_0[1][1]
            #Silence_no_silence_ratio = num_of_0/num_of_1    #feature


            ACF = librosa.core.autocorrelate(stft_of_file, max_size=None, axis=0)
            Max_ACF =  np.max(ACF)   #feature
            feature_vector = np.hstack((mean_of_mfcc, std_of_mfcc, mean_of_delta_mfcc,std_of_delta_mfcc,mean_of_double_delta_mfcc,std_of_double_delta_mfcc,mean_of_rms,std_of_rms,Energy_whole_clip,Energy_ratio_per_frame_mean,Energy_ratio_per_frame_std,peak_to_stft_ratio,mean_of_ratio_of_peak_to_spectra,std_of_ratio_of_peak_to_spectra,Max_ACF))
            transposed_feature_vector = np.reshape(feature_vector, (1, 129))
            full_feature_vector = np.concatenate((full_feature_vector, transposed_feature_vector), axis=0)
            print(np.shape(full_feature_vector))
            #full_feature_vector = np.concatenate((full_feature_vector, feature_vector), axis=0)
            # tempo, beat_frames = librosa.beat.beat_track(y=filedata, sr=nos)
    print(type(File_names))
    return full_feature_vector


if __name__ == "__main__":
    X = []
    M = MFCC_calculation()                  # call the function in main()
    np.save("new_features_enhanced", M)