import cv2
import numpy as np
import os
import mediapipe as mp

# import custom model functions
from custom_model import *

def get_size(text, fontFace, fontScale, thickness):
    (width, height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    return width, height

# Get x,y coordinates for centering text on screen
def get_center(text_width, text_height, video):
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_pos = (video_width - text_width) // 2
    y_pos = (video_height + text_height) // 2
    return x_pos, y_pos

def progress_bar(text_width, text_height, image, progress):
    pass

"""
USE THIS FILE IF THE ACTIONS ARE SEQUENTIAL, 
i.e. END OF ACTION 1 IS BEGINNING OF ACTION 2, 
END OF ACTION 2 IS BEGINNING OF ACTION 3, etc.
"""

# setup information
gen_data = True
PROJECT = 'Clubbell'
# MOVEMENTS = ['swing']
MOVEMENTS = ['swing', 'open', 'hold', 'idle']
# MOVEMENTS = ['swing', 'idle', 'hold'] - for simple mace swings

# Create Actions numpy array that we try to detect
actions = np.array(MOVEMENTS)

# How many "video" sequence we would like to track
no_sequences = 15

# How long (how many frames) each "video" sequence will be
sequence_length = 30

# Folder start number, change if you already collected data, i.e. you have 15, start_folder should be 16
start_folder = 16

# initialize the path for data
DATA_PATH = os.path.join('data', PROJECT)

# If we are generating data, we need someplace to store it
if gen_data:

    # Check to see if the path already exists, if so modify the project name
    while os.path.exists(DATA_PATH):
        PROJECT += '_new'
        DATA_PATH = os.path.join('data', PROJECT)

    # Create folders used to store data
    for action in actions:

        # Make directories for each action
        ACTION_PATH = os.path.join(DATA_PATH, action)
        os.makedirs(ACTION_PATH)


        for sequence in range(start_folder, start_folder+no_sequences):
            
            # Make directories for the number of sequences we're collecting
            SEQ_PATH = os.path.join(ACTION_PATH, str(sequence))
            os.makedirs(SEQ_PATH)

            # Finally, make separate directories for the data and the frames we collect
            try: 
                # Directory for training data
                ARRAY_PATH = os.path.join(SEQ_PATH, 'arrays')
                os.makedirs(ARRAY_PATH)

                # Directory for visually verifying collected data
                FRAMES_PATH = os.path.join(SEQ_PATH, 'frames')
                os.makedirs(FRAMES_PATH)
            except:
                pass

# Initialize where we are getting the video, for webcam use 0
cap = cv2.VideoCapture(0)

# font variable used in directions text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# The actual text used in the directions
collect_text = 'Collecting frames for {} video number {} of ' + str(start_folder+no_sequences-1)
new_text = 'GET READY STARTING SOON'
actions_string = " ".join(actions)
action_text = 'ORDER: {}'.format(actions_string.upper())
start_text = 'COLLECTING {}'
go_text = 'GO!'

# lengths for new video, sequence countdown, and reset pause lengths
new_length = 100
countdown_length = 30
ok = False
frame_wait = 30
break_wait = 10

# Initialize MediaPipe model, in this case we're using the Holistic model
mp_holistic = mp.solutions.holistic

# Start collecting data
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # We start with the start_folder number (usually 1, not zero)
    sequence = start_folder
    
    # Need a countdown before starting!
    for i in range(new_length):
        # Update feed
        ret, image = cap.read()

        # "GET READY STARTING SOON"
        text_width, text_height = get_size(new_text, font, 1, 4)
        textX, textY = get_center(text_width, text_height, cap)
        cv2.putText(image, new_text, (textX, textY-40), font, 1, (0,255, 0), 4, cv2.LINE_AA)

        # "ORDER: {actions}""
        text_width, text_height = get_size(action_text, font, 1, 4)
        textX, textY = get_center(text_width, text_height, cap)
        cv2.putText(image, action_text, (textX , textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)

        # Progress bar
        progress = i/new_length
        rect_topleft = (textX, textY + 10)
        rect_bottomright = (textX + text_width, textY+5 + text_height)
        rect_progress_bottomright = (textX + int(text_width-(text_width*progress)), textY+5 + text_height)
        cv2.rectangle(image, rect_topleft,  rect_bottomright, (0,0,0), cv2.FILLED)
        cv2.rectangle(image, rect_topleft,  rect_progress_bottomright, (0,255,0), cv2.FILLED)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        cv2.waitKey(frame_wait)

        if cv2.waitKey(break_wait) & 0xFF == ord('q'):
            break

    # loop through the amount of videos(sequences) we want to collect
    while sequence < start_folder+no_sequences:
        # Loop through actions
        for action in actions:
            # Countdown to prepare for next action
            for i in range(countdown_length):
                # Get current frame
                ret, image = cap.read()

                # "Collecting frames for {ACTION} video number {sequence} of {no_sequences}"
                current_collect_text = collect_text.format(action.upper(), sequence)
                text_size = cv2.getTextSize(current_collect_text, font, 0.5, 1)[0]
                cv2.rectangle(image, (0,0), (text_size[0]+8, text_size[1]+12), (0, 0, 0), cv2.FILLED)
                cv2.putText(image, current_collect_text, (5, 17), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

                # COLLECTING {action}
                current_start_text = start_text.format(action.upper())
                text_width, text_height = get_size(current_start_text, font, 1, 4)
                textX, textY = get_center(text_width, text_height, cap)
                cv2.putText(image, current_start_text, (textX,textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)

                # Progress bar
                progress = i/countdown_length
                rect_topleft = (textX, textY + 10)
                rect_bottomright = (textX + text_width, textY+5 + text_height)
                rect_progress_bottomright = (textX + int(text_width-(text_width*progress)), textY+5 + text_height)
                cv2.rectangle(image, rect_topleft,  rect_bottomright, (0,0,0), cv2.FILLED)
                cv2.rectangle(image, rect_topleft,  rect_progress_bottomright, (0,255,0), cv2.FILLED)
                
                # Show updated frame
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(frame_wait)

                if cv2.waitKey(break_wait) & 0xFF == ord('q'):
                    ok = False
                    break
            else:
                ok = True
            
            if not ok:
                break
            
            # Loop through video length aka sequence_length
            for frame_num in range(sequence_length):
                # Get current frame
                ret, frame = cap.read()

                # Make detections on current frame
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks (optional step)
                # draw_pose_landmarks(image, results)

                # "Collecting frames for {ACTION} video number {sequence} of {no_sequences}"
                current_collect_text = collect_text.format(action.upper(), sequence)
                text_size = cv2.getTextSize(current_collect_text, font, 0.5, 1)[0]
                cv2.rectangle(image, (0,0), (text_size[0]+8, text_size[1]+12), (0, 0, 0), cv2.FILLED)
                cv2.putText(image, current_collect_text, (5, 17), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

                # Started collecting: "GO!"
                if frame_num >= 0 and frame_num < 11:
                    text_width, text_height = get_size(go_text, font, 1, 4)
                    textX, textY = get_center(text_width, text_height, cap)
                    cv2.putText(image, go_text, (textX, textY), font, 2, (0,255, 0), 5, cv2.LINE_AA)
                
                # Show updated frame
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(frame_wait)

                # If we are generating data
                if gen_data:
                    # Export keypoints we made during mediapipe_detection
                    keypoints = extract_keypoints(results)
                    # Save the keypoints as numpy array
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), 'arrays', str(frame_num))
                    np.save(npy_path, keypoints)

                    # For data verification, save the frame as jpg (not the updated image frame we show to feed)
                    imgname = os.path.join(DATA_PATH, action, str(sequence), 'frames', '{}.jpg'.format(frame_num))
                    cv2.imwrite(imgname, frame)
                
                if cv2.waitKey(break_wait) & 0xFF == ord('q'):
                    break
            else:
                continue

            # if cv2.waitKey(60) & 0xFF == ord('q'):
            break
        else:
            sequence += 1
            continue

            # Iterate the sequence until we have the desired number of videos (sequences)
            

        break
    
    cap.release()
    cv2.destroyAllWindows()