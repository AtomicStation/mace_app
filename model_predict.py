import cv2
import numpy as np
import mediapipe as mp
# import custom model functions
from custom_model import *

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Are we demoing the clubbell model?
demo = False

actions = []

PROJECT = 'Clubbell'
MOVEMENTS = ['swing', 'open', 'hold', 'idle']

# New detection variables
sequence = [] # collect 30 frames to make a prediction
predictions = []
counter = 0
current_stage = ''
threshold = 0.5 # confidence

if demo:
    actions = np.array(['hold', 'swing'])
    model = build_model(actions)
    model.load_weights('clubbell_weights.h5')
    cap = cv2.VideoCapture(0)

else:
    actions = np.array(MOVEMENTS)
    model = build_model(actions)
    model.load_weights(PROJECT + '_weights.h5')
    cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture("videos/MaceCV/IMG_0020.MOV")

#cap = cv2.VideoCapture("videos/webcam_test.mp4")

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_pose_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            body_language_class = actions[np.argmax(res)]
            body_language_prob = res[np.argmax(res)]

            # Counter logic
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
    
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()