import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from custom_model import *

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities



# Actions that we try to detect
actions = np.array(['swing', 'idle', 'hold'])


model = build_model(actions)
model.load_weights('pretrained/maceCV_weights_kelley.h5')

# 1. New detection variables
sequence = [] # collect 30 frames to make a prediction
# sentence = [] # history of detection prediction
predictions = []
counter = 0
current_stage = ''
threshold = 0.5 # confidence

cap = cv2.VideoCapture("videos/5_mace_swings.mp4")
# new_video = 'videos/counted_reps.mp4'
new_video = 'videos/cropped.mp4'
video_fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
h, w, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


ratio = w/h
new_w = int(h/ratio)
center_hor = w/2 + 75
crop_left = int(center_hor - (new_w/2))
crop_right = int(center_hor + (new_w/2))
new_width = crop_right - crop_left

writer = cv2.VideoWriter(new_video, fourcc, video_fps, (new_width,h))
# # Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while ret:
        cropped = frame[0:h, crop_left:crop_right]

        # Read feed
        # ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(cropped, holistic)
        # print(results)
        
        # Draw landmarks
        draw_pose_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            body_language_class = actions[np.argmax(res)]
            body_language_prob = res[np.argmax(res)]

                        
            if body_language_class == 'swing' and body_language_prob >= 0.7:
                current_stage = 'swing'
            elif current_stage == 'swing' and body_language_class == 'hold' and body_language_prob >= 0.7:
                current_stage = 'hold'
                counter += 1



            # Status Box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), cv2.FILLED)
            
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob,2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Count
            cv2.putText(image, 'Count'
                        , (200,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter)
                        , (200,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        image = cv2.resize(image, (new_width, h))
        writer.write(image)
        # ret, frame = cap.read()
    
        # Show to screen
        # cv2.imshow('OpenCV Feed', cropped)
        # cv2.waitKey(10)
        ret, frame = cap.read()

        # Break gracefully
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

print("ratio w/h: " + str(ratio))
writer.release()
cap.release()
cv2.destroyAllWindows()