# Import dependencies
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from scipy import stats
from keras import Sequential
from keras.layers import LSTM, Dense

# Get the MediaPipe landmark locations
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# Draw the MediaPipe Pose landmarks to the screen
def draw_pose_landmarks(image, results):
    # MediaPipe Holistic Model and Drawing utilities
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic
    # Cyberpunk colors in BGR
    light_blue = (255, 247, 209)
    purple = (255, 0, 214)
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=purple, thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=light_blue, thickness=2, circle_radius=2)
                             ) 
    

# Draw all MediaPipe Holistic landmarks (and connections) to the screen
def draw_all_landmarks(image, results):
    # MediaPipe Holistic Model and Drawing utilities
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic
    # Cyberpunk colors in BGR
    fusia = (109, 42, 255)
    light_blue = (255, 247, 209)
    green = (159, 255, 0)
    des_blue = (120, 86, 0)
    purple = (255, 0, 214)
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=light_blue, thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=green, thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=purple, thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=light_blue, thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=fusia, thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=light_blue, thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=fusia, thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=light_blue, thickness=2, circle_radius=2)
                             ) 


# Extract the MediaPipe landmarks and make one giant array with the information, used by the LSTM RNN model
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# Visualize the probabilities of the actions to scren
def prob_viz(res, actions, input_frame):
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[0], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# Builds the LSTM RNN Model
def build_model(actions_list):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions_list.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Return the width and height of text
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

# Display a progress bar to the screen to know how much time is left before data is gathered
def progress_bar(video, image, progress):
    # get width and height of the video
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # position the bar on screen
    x_pos = video_width // 4
    y_pos = video_height * 3 // 4

    # set up the size 
    bar_width = video_width // 2
    bar_height = video_height // 20

    # Set up the topleft corner and bottomright corner of the bars
    rect_topleft = (x_pos, y_pos)
    rect_bottomright = (x_pos + bar_width, y_pos + bar_height)
    rect_progress_bottomright = (x_pos + int(bar_width-(bar_width*progress)), y_pos + bar_height)

    # draw the bars
    cv2.rectangle(image, rect_topleft,  rect_bottomright, (0,0,0), cv2.FILLED)
    cv2.rectangle(image, rect_topleft,  rect_progress_bottomright, (0,255,0), cv2.FILLED)

    return image

# Display the rep count in the upper left corner
def display_count(image, count):
    cv2.putText(image, str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return image

# Dispaly Probability and Action in upper left corner -- also works with display_count()
def display_stats(image, prob, action):
    build_stats = action + ": " + str(int(prob*100)) + "%"
    cv2.putText(image, build_stats, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return image

# Display the Probability, Action, and Count in a nice bold box in upper left corner
def bold_stats(image, probability, action, count):
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    color_white = (255,255,255)
    color_black = (0,0,0)
    # Status Box
    cv2.rectangle(image, (0,0), (250, 60), color_black, 1)
    
    # Display Class
    cv2.putText(image, 'CLASS', (95,12), text_font, 0.5, color_white, 1, cv2.LINE_AA)
    cv2.putText(image, action, (90,40), text_font, 1, color_white, 2, cv2.LINE_AA)
    
    # Display Probability
    cv2.putText(image, 'PROB', (15,12), text_font, 0.5, color_white, 1, cv2.LINE_AA)
    cv2.putText(image, probability, (10,40), text_font, 1, color_white, 2, cv2.LINE_AA)
    
    # Display Count
    cv2.putText(image, 'REPS', (180,12), text_font, 0.5, color_white, 1, cv2.LINE_AA)
    cv2.putText(image, str(count), (175,40), text_font, 1, color_white, 2, cv2.LINE_AA)
    return image

# Changes the frame to be completely white with only the MediaPipe landmarks showing
def only_mediapipe(image, results):
    img_white = np.empty(image.shape)
    img_white.fill(255)
    draw_all_landmarks(img_white, results)