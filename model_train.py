import numpy as np
import os
import mediapipe as mp
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# import custom model functions
from custom_model import *


# setup information
PROJECT = 'Clubbell_nohold_noidle'

# initialize DATA_PATH object
DATA_PATH = os.path.join('data', PROJECT)

# initialize important variables

# How many "video" sequence we would like to track
no_sequences = -1

# How long (how many frames) each "video" sequence will be
sequence_length = -1

# Folder start number
start_folder = -1

try:
    # Check if this project has data, \data\PROJECT
    if os.path.exists(DATA_PATH):

        # get actions list
        actions = sorted(os.listdir(DATA_PATH))
        
        # Check if actions in the project directory are missing
        if not actions:
            raise Exception("{} does not contain any actions".format(DATA_PATH))
        
        # create label_map from the actions list, i.e. 'swing':0, 'hold': 1, etc.
        label_map = {label:num for num, label in enumerate(actions)}

        # Create lists for training and testing data
        all_data, labels = [], []

        # Open each action and check the directories inside
        for action in actions:

            # Create new path to actions, i.e. \data\PROJECT\ACTION
            ACTION_PATH = os.path.join(DATA_PATH, action)

            # create a list of the directories in this action
            str_seq = os.listdir(ACTION_PATH)

            # check if sequences in the action directory are missing
            if not str_seq:
                raise Exception("{} does not contain any sequences".format(ACTION_PATH))

            # Turn the list into a sorted list of integers
            sequences = sorted(np.array(str_seq).astype(int))

            # get the amount of sequences
            seq_count = len(sequences) 
            start_folder = int(sequences[0])

            # check to see if the data is consisten across actions, update no_sequences variable
            if no_sequences == -1 or seq_count == no_sequences:
                no_sequences = seq_count
            else:
                raise Exception("{} does not have consistent number of sequences, expecting {}, found {}".format(ACTION_PATH, no_sequences, seq_count))
            
            # Open each sequence and check the files inside
            for sequence in sequences:
                # There are two directory in each sequence directory, we need to specify 'arrays' directory as the appropriate path
                SEQ_PATH = os.path.join(ACTION_PATH, str(sequence), 'arrays')

                # initialize a variable to count the numpy arrays in the sequence directory
                np_count = 0

                # initialize a "video" list that will contain each frames landmarks in this particular sequence
                video = []
                
                # get list of numpy arrays in the sequence directory
                str_np_arrays = os.listdir(SEQ_PATH)

                # check if numpy arrays in the sequence directory are missing
                if not str_np_arrays:
                    raise Exception("{} does not contain any files or directories".format(SEQ_PATH))
                
                # Count the number of numpy arrays in the directory
                for np_array in str_np_arrays:
                    FRAME_PATH = os.path.join(SEQ_PATH, np_array)
                    if os.path.isfile(FRAME_PATH):
                        np_count += 1
                
                # check to see if the number of arrays is consistent across all sequences, update the sequence_length variable
                if sequence_length == -1 or np_count == sequence_length:
                    sequence_length = np_count
                elif len(str_np_arrays) > 0 and np_count == 0:
                    raise Exception("{} does not contain any numpy array files".format(SEQ_PATH))
                else:
                    raise Exception("{} does not have consistent number of numpy arrays, expecting {}, found {}".format(SEQ_PATH, sequence_length, np_count))
                
                for frame_num in range(sequence_length):
                    # Load each numpy array and unpack the file into the original landmark keypoints array
                    frame_keypoints = np.load(os.path.join(SEQ_PATH, '{}.npy'.format(frame_num)))

                    # add them to the video list
                    video.append(frame_keypoints)

                all_data.append(video)
                labels.append(label_map[action])

        print(label_map)

        # Nicholas Renotte's code:
        # new_sequences, labels = [], []
        # for action in actions:
        #     for sequence in range(no_sequences):
        #         window = []
        #         for frame_num in range(sequence_length):
        #             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
        #             window.append(res)
        #         new_sequences.append(window)
        #         labels.append(label_map[action])

        # Create training and testing sets
        X = np.array(all_data)
        y = to_categorical(labels).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # # Create logging and callbacks for TensorBoard
        # log_dir = os.path.join('Logs')
        # tb_callback = TensorBoard(log_dir=log_dir)

        # Ensure actions is an np.array for the RNN model
        actions = np.array(actions)

        # Build LSTM RNN model
        model = build_model(actions)

        # Train the model
        model.fit(X_train, y_train, epochs=100)

        # Save the weights
        model.save('pretrained/' + PROJECT + '_weights.h5')

    else:
        raise Exception("{} does not exist".format(DATA_PATH))
except Exception as e:
    print("ERROR: ", e)
