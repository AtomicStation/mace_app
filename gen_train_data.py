import cv2
import numpy as np
import os
import mediapipe as mp

# import custom model functions
from custom_model import *

# Get x,y coordinates for centering text on screen
def get_center(text, fontFace, fontScale, thickness, cap):
    text_size = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_pos = (width - text_size[0]) // 2
    y_pos = (height + text_size[1]) // 2
    return x_pos, y_pos

# setup information
gen_data = False
PROJECT = 'Testing'
MOVEMENTS = ['swing']
# MOVEMENTS = ['swing', 'hold', 'open', 'idle']
# MOVEMENTS = ['swing', 'idle', 'hold'] - for simple mace swings

# Create Actions numpy array that we try to detect
actions = np.array(MOVEMENTS)

# How many "video" sequence we would like to track
no_sequences = 5

# How long (how many frames) each "video" sequence will be
sequence_length = 30

# Folder start number
start_folder = 1

if gen_data:
    # Path for exported frames, check to see if it exists
    while os.path.exists(DATA_PATH):
        PROJECT += '_new'
        DATA_PATH = os.path.join('data', PROJECT)

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

# font variable used in directions text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# The actual text used in the directions
collect_text = 'Collecting frames for {} Video Number {} of ' + str(no_sequences)
new_text = 'STARTING IN 5 SECONDS'
start_text = 'STARTING {}'
go_text = 'GO!'
reset_text = 'RESET MOVEMENT'

# Set mediapipe model 
mp_holistic = mp.solutions.holistic # Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Need a countdown before starting!
    for i in range(100):
        # Update feed
        ret, image = cap.read()
        # no longer check for new video
        textX, textY = get_center(new_text, font, 1, 4, cap)
        cv2.putText(image, new_text, (textX , textY - 40), font, 1, (0,255, 0), 4, cv2.LINE_AA)
        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        cv2.waitKey(30)
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            # Which
            for i in range(100):
                # get new feed
                ret, image = cap.read()
                current_start_text = start_text.format(action.upper())
                textX, textY = get_center(current_start_text, font, 1, 4, cap)
                cv2.putText(image, current_start_text, (textX,textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)
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

                if frame_num >= 0 and frame_num < 11:
                    textX, textY = get_center(go_text, font, 2, 5, cap)
                    cv2.putText(image, go_text, (textX, textY), font, 2, (0,255, 0), 5, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(30)

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
            for i in range(100):
                # get new feed
                ret, image = cap.read()
                textX, textY = get_center(reset_text, font, 1, 4, cap)
                cv2.putText(image, reset_text, (textX,textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(30)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()