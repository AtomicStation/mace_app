import cv2
import numpy as np
import mediapipe as mp
# import custom model functions
from custom_model import *

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

PROJECT = 'Clubbell_nohold_noidle'
MOVEMENTS = ['swing', 'open']

# PROJECT = 'Mace_Demo'
# MOVEMENTS = ['swing', 'hold', 'idle']

# New detection variables
# list of the last keypoints
sequence = []

# list of predictions
predictions = []

# predicted action
pred_action = ''

# predicted probability
pred_prob = 0

# rep counter
counter = 0

# Confidence threshold that the action is what we think it is
threshold = 0.8

# Variable to check if we changed actions
check_action = ''

# Debug stuff
confirmed = []
test_counter = 0

# TO DO: make functions have this font already built in
font = cv2.FONT_HERSHEY_SIMPLEX

# Sort the movement actions for when we build the model
sorted_actions = sorted(MOVEMENTS)

# Model requires the actions to be a numpy array
actions = np.array(sorted_actions)

# Models have inconsistent reset action names 'open' vs 'hold'
not_swing_index = np.where(actions != 'swing')[0]
not_swing_actions = actions[not_swing_index]

# If its a 2 category model, reset action is the not swing action
if actions.size == 2:
    reset_action = not_swing_actions[0]
else:
    # For 3 category models, the other action is 'idle'
    reset_index = np.where(not_swing_actions != 'idle')[0]
    reset_action = not_swing_actions[reset_index][0]

# Build the model using the actions array
model = build_model(actions)

# Load the projects pre-trained weights
model.load_weights(PROJECT + '_weights.h5')

# Check if this is a live demo or using previously recorded video
demo = True

if demo:
    # OpenCV uses webcam (value 0)
    cap = cv2.VideoCapture(0)

else:
    # OpenCV needs to find the video
    video_name = PROJECT.lower()
    video_path = 'videos/' + video_name + '.mp4'
    cap = cv2.VideoCapture(video_path)

# Start using the MediaPipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Give countdown timer before starting to capture frames for detection
    for i in range(100):
        # Update feed
        ret, image = cap.read()

        # "GET READY STARTING SOON"
        new_text = "GET READY STARTING SOON"
        text_width, text_height = get_size(new_text, font, 1, 4)
        textX, textY = get_center(text_width, text_height, cap)
        cv2.putText(image, new_text, (textX, textY), font, 1, (0,255, 0), 4, cv2.LINE_AA)

        # Progress bar
        image = progress_bar(cap, image, i/100)

        # Show modified frame to screen
        cv2.imshow('OpenCV Feed', image)
        cv2.waitKey(30)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Start capturing frames for detection
    while cap.isOpened():

        # Get current frame and whether or not there is a current frame (ret)
        ret, frame = cap.read()

        # If ret is False, no longer capturing frames and need to break
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_pose_landmarks(image, results)
        
        # Extract the keypoints from the current frame
        keypoints = extract_keypoints(results)

        # Add these to the sequence list
        sequence.append(keypoints)

        # take the last 30 sequences and keep track of them
        last_sequence = sequence[-30:]


        # When we have 30 frames of data, we can make a prediction
        if len(last_sequence) == 30:
            # get prediction, but first make sure the shape of the array is correct so expand
            res = model.predict(np.expand_dims(last_sequence, axis=0), verbose=0)[0]

            # The prediction will be an array of probability values, we want the highest probability index
            index_res = np.argmax(res)

            # add this prediction to a list
            predictions.append(index_res)
            
            # Use this index to get the name of the action
            pred_action = actions[index_res]

            # Use the index to get the probability of the action
            pred_prob = res[index_res]

            # if the last -N predictions equal the current prediction, there's a high confidence in that prediction
            if np.unique(predictions[-10:])[0] == index_res:

                # Check the probability versus our confidence threshold
                if pred_prob > threshold:
                    # if the previous action was swing and current action is the reset_action
                    if check_action == 'swing' and pred_action == reset_action:
                        # count the rep
                        counter += 1
                    
                    # after the check, the current action becomes the new check action for next time
                    check_action = pred_action

                    # Debugging and tracking the sequence of confirmed actions
                    if len(confirmed) > 0:
                        if pred_action != confirmed[-1]:
                            confirmed.append(actions[index_res])
                    else:
                        confirmed.append(actions[index_res])

            # image = prob_viz(res, actions, image)

            # Display the counter to the screen
            image = display_count(image, counter)
    
        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        cv2.waitKey(20)

        # Break gracefully
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()