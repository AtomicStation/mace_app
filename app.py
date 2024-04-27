# Import dependencies for Kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty, BooleanProperty, ListProperty

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger


# import computer vision dependencies
import cv2

import tensorflow as tf
import numpy  as np
import os

# mediapipe
import mediapipe as mp
mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# import custom model functions
from custom_model import *

# Builder.load_file("menu.kv")
# doesn't work, need to use screenmanager for a proper screen rather than this one which is just a crappy overlay

class FirstWindow(Screen):
    pass

class SecondWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

PROJECT = 'Clubbell_nohold_noidle'
MOVEMENTS = ['swing', 'open']

# Build the app
class MaceCVApp(App):
    counter = 0
    toggle_button_text = StringProperty("Start Counting")
    mediapipe_enabled = BooleanProperty(False)
    rep_label_text = StringProperty('Null')
    prob_label_text = StringProperty('Null')
    class_label_text = StringProperty('Class')
    actions = np.array(sorted(MOVEMENTS))
    sequence = ListProperty([])
    predictions = ListProperty([])
    check_action = ''
    

    def build(self):
        # create super box
        superBox = BoxLayout(orientation='vertical')

        # create top Box layout
        topBox = BoxLayout(orientation='horizontal', size_hint=(1,0.2))

        # Create and build individual label boxes
        label_box_1 = BoxLayout(orientation='vertical')
        prob_label = Label(text='PROBABILITY')
        self.prob_label_value = Label(text=self.prob_label_text,color=(1,1,1,0))
        label_box_1.add_widget(prob_label)
        label_box_1.add_widget(self.prob_label_value)

        label_box_2 = BoxLayout(orientation='vertical')
        class_label = Label(text='ACTION')
        self.class_label_value = Label(text=self.class_label_text,color=(1,1,1,0))
        label_box_2.add_widget(class_label)
        label_box_2.add_widget(self.class_label_value)
        
        label_box_3 = BoxLayout(orientation='vertical')
        rep_label = Label(text='REPS')
        self.rep_label_value = Label(text='Null',color=(1,1,1,0))
        label_box_3.add_widget(rep_label)
        label_box_3.add_widget(self.rep_label_value)
        
        # build top box
        topBox.add_widget(label_box_1)
        topBox.add_widget(label_box_2)
        topBox.add_widget(label_box_3)

        # build bottom box
        bottomBox = BoxLayout(orientation='horizontal', size_hint=(1,0.2))
        self.play_button = ToggleButton(text=self.toggle_button_text, on_press=self.on_toggle_button_state)
        reset_button = Button(text='Reset', on_press=self.reset_button)
        bottomBox.add_widget(self.play_button)
        bottomBox.add_widget(reset_button)

        # Camera components
        self.web_cam = Image(size_hint=(1,0.6))
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1/33)

        # LSTM RNN Model
        self.model = build_model(self.actions)
        self.model.load_weights(PROJECT + '_weights.h5')

        # build superBox
        superBox.add_widget(topBox)
        superBox.add_widget(self.web_cam)
        superBox.add_widget(bottomBox)

        # Return superBox
        return superBox
        

    def update(self, *args):
        action_seq = []
        current_stage = ''

        # bring capture object into function
        cap = self.capture
        
        # Read feed
        ret, image = cap.read()

        # MediaPipe is enabled, start MediaPipe
        if self.mediapipe_enabled:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                # Make MediaPipe detections
                image, results = mediapipe_detection(image, holistic)

                # Draw landmarks on image
                draw_pose_landmarks(image, results)

                # Prediction logic
                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                self.last_sequence = self.sequence[-30:]

                if len(self.last_sequence) == 30:
                    res = self.model.predict(np.expand_dims(self.last_sequence, axis=0), verbose=0)[0]
                    index_res = np.argmax(res)
                    self.predictions.append(index_res)
                    pred_action = self.actions[index_res]
                    pred_prob = res[index_res]

                    # # Counter logic
                    if np.unique(self.predictions[-7:])[0] == index_res:
                        if pred_prob >= 0.8:
                            image = display_stats(image, pred_prob, pred_action)
                            self.prob_label_value.text = str(int(pred_prob*100))+"%"
                            self.prob_label_value.color = (1,1,1,1)
                            self.class_label_value.text = pred_action
                            self.class_label_value.color = (1,1,1,1)


                            if self.check_action == 'swing' and pred_action == 'open':
                                self.counter += 1
                                self.rep_label_value.text = str(self.counter)
                                self.rep_label_value.color = (1,1,1,1)

                            self.check_action = pred_action
                    
                    image = display_count(image, self.counter)
                    

                    # # Update Probability and Class labels
                    
                    # if prob_text >= 0.9:
                    #     self.prob_label_value.color=(0,1,0)
                    #     self.class_label_value.color=(0,1,0)
                    # else:
                    #     self.prob_label_value.color=(1,1,1)
                    #     self.class_label_value.color=(1,1,1)
                    

        # Convert raw OpenCV image array into a texture for rendering        
        #     # put image into buffer        
        buf1 = cv2.flip(image, 0)
        buf = buf1.tobytes()

        # convert image into kivy texture -- OpenGL texture
        img_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.web_cam.texture = img_texture

    def prediction(self, *args):
        pass

    def on_toggle_button_state(self, widget):
        if widget.state == "normal":
            widget.text = "Start NOW"
            self.mediapipe_enabled = False
        else:
            widget.text = "STOP"
            self.mediapipe_enabled = True
    
    def reset_button(self, *args):
        self.prob_label_value.text = '0'
        self.class_label_value.text = 'Idle'
        self.counter = 0
        # self.rep_label_value.text = '0'


# run the app, close openCV after app is closed
MaceCVApp().run()
cv2.destroyAllWindows()