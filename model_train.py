import cv2
import numpy as np
import os
import mediapipe as mp
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# import custom model functions
from custom_model import *


# test: works great!
# test1: missing sequence in actions 'idle/30' missing
# test2: missing numpy array in sequence 'swing/30/29.npy' missing
# test3: missing actions 'data/test3/' is empty
# test4: missing all actions in 'swing'
# test5: missing all sequences in 'idle'
# test6: missing all arrays in 'swing/29/'
# test7: doesn't exist

# setup information
PROJECT = 'test'

# initialize DATA_PATH object
DATA_PATH = os.path.join('data', PROJECT)

# initialize important variables
# Create Actions numpy array that we try to detect
actions = []

# How many "video" sequence we would like to track
no_sequences = -1

# How long (how many frames) each "video" sequence will be
sequence_length = -1

# Folder start number
start_folder = -1

my_frame_path_list = []
my_list = []
nick_frame_path_list = []
nick_list = []

# Path for exported data, numpy arrays
try:
    if os.path.exists(DATA_PATH):
        # get actions list
        actions = sorted(os.listdir(DATA_PATH))
        if not actions:
            raise Exception("{} does not contain any actions".format(DATA_PATH))
        
        # create label_map
        label_map = {label:num for num, label in enumerate(actions)}
        print(actions)

        # testing my hypothesis
        my_seq, my_labels = [], []
        for action in actions:
            ACTION_PATH = os.path.join(DATA_PATH, action)
            str_seq = os.listdir(ACTION_PATH)

            sequences = sorted(np.array(str_seq).astype(int))
            if not sequences:
                raise Exception("{} does not contain any sequences".format(ACTION_PATH))
            
            seq_count = len(sequences) 
            start_folder = int(sequences[0])
            if no_sequences == -1 or seq_count == no_sequences:
                no_sequences = seq_count
            else:
                raise Exception("{} does not have consistent number of sequences, expecting {}, found {}".format(ACTION_PATH, no_sequences, seq_count))
            
            my_list = sequences
            print(sequences)
            for sequence in sequences:
                # for the real thing, since when we generate data we will save frames for each sequence as well, specify 'arrays' directory
                # SEQ_PATH = os.path.join(ACTION_PATH, sequence, 'arrays')
                SEQ_PATH = os.path.join(ACTION_PATH, str(sequence))
                np_count = 0
                my_window = []
                frames = sorted(os.listdir(SEQ_PATH))
                for frame_num in os.listdir(SEQ_PATH):
                    FRAME_PATH = os.path.join(SEQ_PATH, frame_num)
                    if os.path.isfile(FRAME_PATH):
                        np_count += 1
                    
                    # my_res = np.load(FRAME_PATH)
                    my_res = FRAME_PATH
                    my_window.append(my_res)
                    my_frame_path_list.append(FRAME_PATH)

                if sequence_length == -1 or np_count == sequence_length:
                    sequence_length = np_count
                elif np_count == 0:
                    raise Exception("{} does not contain any arrays".format(SEQ_PATH))
                else:
                    raise Exception("{} does not have consistent number of numpy arrays, expecting {}, found {}".format(SEQ_PATH, sequence_length, np_count))

                my_seq.append(my_window)
                my_labels.append(label_map[action])

        print("Actions: ", actions)
        print("no_sequences: ", no_sequences)
        print("sequence_length: ", sequence_length)
        print("start_folder: ", start_folder)

        ## OLD CODE Preprocess Data and Create Labels and Features

        new_sequences, labels = [], []
        for action in actions:
            nick_list = np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int)
            for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(sequence_length):
                    # res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    res = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
                    # print(type(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))))
                    window.append(res)
                    nick_frame_path_list.append(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                new_sequences.append(window)
                labels.append(label_map[action])


        # Create training and testing sets
        # X = np.array(new_sequences)
        # y = to_categorical(labels).astype(int)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

        # log_dir = os.path.join('Logs')
        # tb_callback = TensorBoard(log_dir=log_dir)

        # # Build LSTM RNN model
        # model = build_model(actions)

        # # Train the model
        # model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

        # # Save the weights
        # model.save(PROJECT + '.h5')




    else:
        raise Exception("{} does not exist".format(DATA_PATH))
except Exception as e:
    print("ERROR: ", e)








