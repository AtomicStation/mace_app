import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from scipy import stats
from keras import Sequential
from keras.layers import LSTM, Dense


mp_holistic = mp.solutions.holistic # Holistic model


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_pose_landmarks(image, results):
    # Drawing utilities
    mp_drawing = mp.solutions.drawing_utils 
    # Cyberpunk colors in BGR
    light_blue = (255, 247, 209)
    purple = (255, 0, 214)
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=purple, thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=light_blue, thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame):
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[0], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

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

def display_count(image, count):
    cv2.putText(image, str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return image

def display_stats(image, prob, action):
    build_stats = action + ": " + str(int(prob*100)) + "%"
    cv2.putText(image, build_stats, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return image