from __future__ import print_function
import numpy as np
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

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
flags = tf.app.flags
flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')
FLAGS = flags.FLAGS

DIRNAME = r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/MSoS_challenge_2018_Development_v1-00/Evaluation'
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


def input_sound():  # function to calculate the FS and time series input
    files = get_file_paths(DIRNAME)  # function called to get the file paths
    File_names = []
    full_feature_vector = np.empty([0,128])
    for file in sorted(files):  # loop to access each file
        # print (file)
        (filepath, ext) = os.path.splitext(file)  # get extension of the file
        file_name = os.path.basename(file)  # get the file name
        if ext == '.wav':
            File_names.append(file_name)
            y, sr = librosa.load(file,sr=None)
            print (sr)
            examples_batch = waveform_to_examples(y,sr)
            pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
            with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
                features_tensor = sess.graph.get_tensor_by_name(
                    vggish_params.INPUT_TENSOR_NAME)
                embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
                postprocessed_batch = pproc.postprocess(embedding_batch)
                print(np.shape(postprocessed_batch))
                full_feature_vector = np.concatenate((full_feature_vector, postprocessed_batch), axis=0)
                print (np.shape(full_feature_vector))
    return full_feature_vector



if __name__ == "__main__":
    X = []
    M = input_sound()                 # call the function in main()
    np.save("features_to_test_on",M)
