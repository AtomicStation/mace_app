import cv2
import numpy as np
import os
import mediapipe as mp

# import custom model functions
from custom_model import *


# setup information
PROJECT = 'text'
MOVEMENTS = ['swing']

# Actions that we try to detect
actions = np.array(MOVEMENTS)

# Thirty videos worth of data
no_sequences = 5

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 1

# Boolean to check if this is a new data collecting session
new_video = True

# Generate and Collect Keypoint Values for Training and Testing
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

collect_text = 'Collecting frames for {} Video Number {}'
new_text = 'STARTING IN 5 SECONDS'
start_text = 'STARTING {}'
go_text = 'GO!'
reset_text = 'RESET MOVEMENT'

# putText(img, text, position, fontFace, fontScale, color, thickness, lineType)
def get_center(text, fontFace, fontScale, thickness, cap):
    text_size = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_pos = (width - text_size[0]) // 2
    y_pos = (height + text_size[1]) // 2
    return x_pos, y_pos

# Set mediapipe model 
mp_holistic = mp.solutions.holistic # Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # font variable used in directions text overlay
    
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
                    textX, textY = get_center(new_text, font, 1, 4, cap)
                    cv2.putText(image, new_text, (textX , textY - 40), font, 1, (0,255, 0), 4, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(5000)
                    
                # Set collecting text (every frame get's this)
                cv2.putText(image, collect_text.format(action, sequence), (15, 12), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

                # Apply different text depending on the frame so we know what to do
                if frame_num == 0: 
                    start_text = start_text.format(action.upper())
                    textX, textY = get_center(start_text, font, 1, 4, cap)
                    cv2.putText(image, start_text, (textX,textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)
                elif frame_num > 0 and frame_num < 11:
                    textX, textY = get_center(go_text, font, 1, 4, cap)
                    cv2.putText(image, go_text, (textX, textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1)
                else:
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1)

            # Reset position before starting the next movement
            textX, textY = get_center(reset_text, font, 1, 4, cap)
            cv2.putText(image, reset_text, (textX,textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(3000)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()