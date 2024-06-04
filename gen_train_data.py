import cv2
import numpy as np
import os
import mediapipe as mp

# import custom model functions
from custom_model import *

"""
USE THIS FILE TO GENERATE DATA FOR ACTIONS THAT ARE SEPARATE 
FROM EACH OTHER, AND CAN BE IN ANY ORDER, i.e.
ACTION 1 WILL BE REPEATED no_sequences TIMES, FOLLOWED BY
ACTION 2 for no_sequences TIMES, etc.
"""

# setup information
gen_data = False
PROJECT = 'Clubbell'
# MOVEMENTS = ['swing']
MOVEMENTS = ['swing', 'open', 'hold', 'idle']
# MOVEMENTS = ['swing', 'idle', 'hold'] - for simple mace swings

# Create Actions numpy array that we try to detect
actions = np.array(MOVEMENTS)

# How many "video" sequence we would like to track
no_sequences = 5

# How long (how many frames) each "video" sequence will be
sequence_length = 30

# Folder start number
start_folder = 1

# initialize the path for data
DATA_PATH = os.path.join('data', PROJECT)

# If we are generating data, we need someplace to store it
if gen_data:

    # Path for exported frames, check to see if it exists
    while os.path.exists(DATA_PATH):
        PROJECT += '_new'
        DATA_PATH = os.path.join('data', PROJECT)

    # create folders
    for action in actions: 

        # Make directories for each action
        ACTION_PATH = os.path.join(DATA_PATH, action)
        os.makedirs(ACTION_PATH)
        
        for sequence in range(start_folder, start_folder+no_sequences):
        
            # Make directories for the number of sequences we're collecting
            SEQ_PATH = os.path.join(ACTION_PATH, str(sequence))
            os.makedirs(SEQ_PATH)
        
            try: 
                # Create the paths for arrays and for frames
                ARRAY_PATH = os.path.join(SEQ_PATH, 'arrays')
                FRAMES_PATH = os.path.join(SEQ_PATH, 'frames')

                # Create the directories to hold the arrays and the frames
                os.makedirs(ARRAY_PATH)
                os.makedirs(FRAMES_PATH)
            except:
                pass

# Generate and Collect Keypoint Values for Training and Testing
cap = cv2.VideoCapture(0)

# font variable used in directions text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# The actual text used in the directions
collect_text = 'Collecting frames for {} Video Number {} of ' + str(start_folder+no_sequences-1)
new_text = 'GET READY STARTING SOON'
action_text = 'NEW ACTION! PREPARE {}'
start_text = 'COLLECTING {}'
go_text = 'GO!'
reset_text = 'RESET MOVEMENT'

# lengths for new video, sequence countdown, and reset pause lengths
new_length = 200
countdown_length = 30
reset_length = 50

# Set mediapipe model 
mp_holistic = mp.solutions.holistic

# Start collecting data by intializing MediaPipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Need a countdown before starting!
    for i in range(new_length):
        # Update feed
        ret, image = cap.read()

        # "GET READY STARTING SOON"
        text_width, text_height = get_size(new_text, font, 1, 4)
        textX, textY = get_center(text_width, text_height, cap)
        cv2.putText(image, new_text, (textX , textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)

        # Progress bar
        progress = i/new_length
        image = progress_bar(cap, image, progress)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        cv2.waitKey(30)

    # Loop through actions
    for action in actions:

        # Quick countdown for new action
        for i in range(countdown_length):

            # Update feed
            ret, image = cap.read()
            
            # Top bar with status of sequence and action
            current_collect_text = collect_text.format(action, 0)
            text_size = cv2.getTextSize(current_collect_text, font, 0.5, 1)[0]
            cv2.rectangle(image, (0,0), (text_size[0]+8, text_size[1]+12), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, current_collect_text, (5, 17), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # "NEW ACTION: PREPARE {action}"
            current_action_text = action_text.format(action.upper())
            text_width, text_height = get_size(current_action_text, font, 1, 4)
            textX, textY = get_center(text_width, text_height, cap)
            cv2.putText(image, current_action_text, (textX , textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)

            # Progress Bar
            progress = i/countdown_length
            image = progress_bar(cap, image, progress)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(30)

        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            
            # Which Action are we doing with a countdown to prepare
            for i in range(countdown_length):
                # get new feed
                ret, image = cap.read()

                # Top bar with status of sequence and action
                current_collect_text = collect_text.format(action, sequence)
                text_size = cv2.getTextSize(current_collect_text, font, 0.5, 1)[0]
                cv2.rectangle(image, (0,0), (text_size[0]+8, text_size[1]+12), (0, 0, 0), cv2.FILLED)
                cv2.putText(image, current_collect_text, (5, 17), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

                # "COLLECTING {action}"
                current_start_text = start_text.format(action.upper())
                text_width, text_height = get_size(current_start_text, font, 1, 4)
                textX, textY = get_center(text_width, text_height, cap)
                cv2.putText(image, current_start_text, (textX,textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)

                # Progress bar
                progress = i/countdown_length
                image = progress_bar(cap, image, progress)

                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(30)

            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks (optional step)
                # draw_pose_landmarks(image, results)

                # Set collecting text (every frame gets this)
                current_collect_text = collect_text.format(action, sequence)
                text_size = cv2.getTextSize(current_collect_text, font, 0.5, 1)[0]
                cv2.rectangle(image, (0,0), (text_size[0]+8, text_size[1]+12), (0, 0, 0), cv2.FILLED)
                cv2.putText(image, current_collect_text, (5, 17), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

                # "GO!" - do the movement as we're collecting the frames
                if frame_num >= 0 and frame_num < 11:
                    text_width, text_height = get_size(go_text, font, 1, 4)
                    textX, textY = get_center(text_width, text_height, cap)
                    cv2.putText(image, go_text, (textX, textY), font, 2, (0,255, 0), 5, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(30)

                # Check to see if we're generating data to save
                if gen_data:
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    # Save the keypoints
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), 'arrays', str(frame_num))
                    np.save(npy_path, keypoints)

                    # To verify the data, save the frames as jpg
                    imgname = os.path.join(DATA_PATH, action, str(sequence), 'frames', '{}.jpg'.format(frame_num))
                    cv2.imwrite(imgname, frame)

            # Reset position before starting the next movement
            for i in range(reset_length):
                # get new feed
                ret, image = cap.read()

                # Top bar with status of sequences and action
                current_collect_text = collect_text.format(action, sequence)
                text_size = cv2.getTextSize(current_collect_text, font, 0.5, 1)[0]
                cv2.rectangle(image, (0,0), (text_size[0]+8, text_size[1]+12), (0, 0, 0), cv2.FILLED)
                cv2.putText(image, current_collect_text, (5, 17), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

                # "RESET MOVEMENT"
                text_width, text_height = get_size(reset_text, font, 1, 4)
                textX, textY = get_center(text_width, text_height, cap)
                cv2.putText(image, reset_text, (textX,textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)

                # Progress bar
                progress = i/reset_length
                image = progress_bar(cap, image, progress)

                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(30)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()