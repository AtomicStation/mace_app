# Import dependencies for Kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import computer vision dependencies
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy  as np
import os


# Builder.load_file("menu.kv")
# doesn't work, need to use screenmanager for a proper screen rather than this one which is just a crappy overlay



# 7. Build layout
class AppLayout(BoxLayout):


    def build(self):
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
        class_label_value = Label(text='0.84')

        label_box_1.add_widget(class_label)
        label_box_1.add_widget(class_label_value)
        
        label_box_3 = BoxLayout(orientation='vertical')
        rep_label = Label(text='REPS')
        rep_label_value = Label(text='0.84')

        label_box_1.add_widget(rep_label)
        label_box_1.add_widget(rep_label_value)
        
        # build top box

        topBox.add_widget(label_box_1)
        topBox.add_widget(label_box_2)
        topBox.add_widget(label_box_3)

        bottomBox = BoxLayout(orientation='horizontal', size_hint=(1,0.2))
        play_buttom = Button(text="Start/Stop")
        reset_button = Button(text='Reset')

        bottomBox.add_widget(play_buttom)
        bottomBox.add_widget(reset_button)






        # components
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
        # todo: add Mediapipe layer over the top of the openCV layer
        # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        ret, frame = self.capture.read()

        # convert raw OpenCV image array into a texture for rendering

        # put image into buffer
        buf = cv2.flip(frame, 0).tostring()

        # need image to be a kivy texture -- OpenGL texture
        img_texture = Texture.create(colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.web_cam.texture = img_texture


class MaceCVApp(App):
    pass






MaceCVApp().run()

# 8. Build update function



# 9. Bring over preprocessing function




# 10. bring over verification function
# 11. update verification function to handle new paths and save current frame
# 12. update verification function to set verified text
# 13. link verification function to button
# 14. setup logger