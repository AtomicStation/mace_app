# Import dependencies for Kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty, BooleanProperty

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
mp_drawing = mp.solutions.drawing_utils

# Builder.load_file("menu.kv")
# doesn't work, need to use screenmanager for a proper screen rather than this one which is just a crappy overlay

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def cyberpunk_landmarks(image, results):
    # Cyberpunk colors in BGR
    light_blue = (255, 247, 209)
    purple = (255, 0, 214)
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=purple, thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=light_blue, thickness=2, circle_radius=2)
                             ) 

class FirstWindow(Screen):
    pass

class SecondWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

# 7. Build layout
# class AppLayout(BoxLayout):
#     pass

# Build the app
class MaceCVApp(App):

    toggle_button_text = StringProperty("Start Counting")
    mediapipe_enabled = BooleanProperty(False)

    def build(self):
        # create super box
        superBox = BoxLayout(orientation='vertical')

        # create top Box layout
        topBox = BoxLayout(orientation='horizontal', size_hint=(1,0.2))

        # Create and build individual label boxes
        label_box_1 = BoxLayout(orientation='vertical')
        prob_label = Label(text='PROBABILITY')
        prob_label_value = Label(text='0.84')
        label_box_1.add_widget(prob_label)
        label_box_1.add_widget(prob_label_value)

        label_box_2 = BoxLayout(orientation='vertical')
        class_label = Label(text='CLASS')
        class_label_value = Label(text='Swing')
        label_box_2.add_widget(class_label)
        label_box_2.add_widget(class_label_value)
        
        label_box_3 = BoxLayout(orientation='vertical')
        rep_label = Label(text='REPS')
        rep_label_value = Label(text='0')
        label_box_3.add_widget(rep_label)
        label_box_3.add_widget(rep_label_value)
        
        # build top box
        topBox.add_widget(label_box_1)
        topBox.add_widget(label_box_2)
        topBox.add_widget(label_box_3)

        # build bottom box
        bottomBox = BoxLayout(orientation='horizontal', size_hint=(1,0.2))
        self.play_button = ToggleButton(text=self.toggle_button_text, on_press=self.on_toggle_button_state)
        reset_button = Button(text='Reset')
        bottomBox.add_widget(self.play_button)
        bottomBox.add_widget(reset_button)

        # Camera components
        self.web_cam = Image(size_hint=(1,0.6))
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1/33)

        # build superBox
        superBox.add_widget(topBox)
        superBox.add_widget(self.web_cam)
        superBox.add_widget(bottomBox)

        # Return superBox
        return superBox
        

    def update(self, *args):
        # bring capture object into function
        cap = self.capture

        if self.mediapipe_enabled:
            # MediaPipe is enabled, start MediaPipe
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                # Read feed
                ret, frame = cap.read()

                # Make MediaPipe detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                cyberpunk_landmarks(image, results)

                # Convert raw OpenCV image array into a texture for rendering

                # put image into buffer
                buf1 = cv2.flip(image, 0)
                buf = buf1.tostring()

                # convert image into kivy texture -- OpenGL texture
                img_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                self.web_cam.texture = img_texture
        else:
            # MediaPipe is not enabled
            # Read feed
            ret, image = cap.read()
            # put image into buffer
            buf1 = cv2.flip(image, 0)
            buf = buf1.tostring()

            # convert image into kivy texture -- OpenGL texture
            img_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            self.web_cam.texture = img_texture

    def on_toggle_button_state(self, widget):
        if widget.state == "normal":
            widget.text = "Start NOW"
            self.mediapipe_enabled = False
        else:
            widget.text = "STOP"
            self.mediapipe_enabled = True





# run the app, close openCV after app is closed
MaceCVApp().run()
cv2.destroyAllWindows()