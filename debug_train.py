import cv2
import numpy as np
import os
import mediapipe as mp
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# import custom model functions
from custom_model import *


# setup information
PROJECT = 'Testing'
MOVEMENTS = ['swing']
# MOVEMENTS = ['swing', 'hold', 'open', 'idle']
# MOVEMENTS = ['swing', 'idle', 'hold'] - for simple mace swings

# initialize DATA_PATH object
DATA_PATH = os.path.join('data', PROJECT)

# Path for exported frames, check to see if it exists
if os.path.exists(DATA_PATH):
    PROJECT += '_new'
    DATA_PATH = os.path.join('data', PROJECT)

# Actions that we try to detect
actions = np.array(MOVEMENTS)

# Thirty videos worth of data
no_sequences = 10

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 1

# create folders
for action in actions: 
    os.makedirs(os.path.join(DATA_PATH,action))
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Generate and Collect Keypoint Values for Training and Testing
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Boolean to check if this is a new data collecting session
new_video = True





# Set mediapipe model 
mp_holistic = mp.solutions.holistic # Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # font variable used in directions text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks (optional step)
                # draw_pose_landmarks(image, results)
                
                # check if we're starting a new collection and stall so we can get ready
                if new_video:
                    # no longer check for new video
                    new_video = False
                    cv2.putText(image, 'STARTING IN 5 SECONDS', (200,150), 
                               font, 1, (0,255, 0), 4, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(5000)
                    

                # Apply different text depending on the frame so we know what to do
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING {}'.format(action.upper()), (250,200), font, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)
                elif frame_num > 0 and frame_num < 11:
                    cv2.putText(image, 'GO!', (300,200), font, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action.upper(), sequence), (15,12), 
                               font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1)
                
                # Export keypoints
                # keypoints = extract_keypoints(results)
                # Save the keypoints
                # npy_path = os.path.join(DATA_PATH, action, str(sequence), 'array', str(frame_num))
                # np.save(npy_path, keypoints)

                # To verify the data, save the frames as jpg
                imgname = os.path.join(DATA_PATH, action, str(sequence), 'frames', '{}.jpg'.format(frame_num))
                cv2.imwrite(imgname, frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Reset position before starting the next movement
            cv2.putText(image, 'RESET MOVEMENT', (250,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(3000)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()