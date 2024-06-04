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

# import Numpy array tools
import numpy  as np

# Import MediaPipe dependencies
import mediapipe as mp
mp_holistic = mp.solutions.holistic

# import custom LSTM RNN model functions
from custom_model import *

# Project constant variables
# Clubbell LIVE demo
PROJECT = 'Clubbell_nohold_noidle'
MOVEMENTS = ['swing', 'open']
RESET_ACTION = 'open'
CAP_VIDEO = 0 # webcam object
WEIGHT_PATH = 'pretrained/' + PROJECT + '_weights.h5'

# # Mace video test - Doesn't work properly yet
# PROJECT = 'Mace_Demo'
# MOVEMENTS = ['swing', 'hold', 'idle']
# RESET_ACTION = 'hold'
# CAP_VIDEO = 'videos/diff_mace_test.mp4'

# Main App Class
class MaceCVApp(App):

    # Initialize variables to use later
    counter = 0
    toggle_button_text = StringProperty("Start Counting")
    mediapipe_enabled = BooleanProperty(False)
    
    rep_label_text = StringProperty('Null')
    prob_label_text = StringProperty('Null')
    class_label_text = StringProperty('Class')
    
    sequence = ListProperty([])
    predictions = ListProperty([])
    check_action = ''

    actions = np.array(sorted(MOVEMENTS))

    # Main app page
    def build(self):

        # create super box - all parts go in here
        superBox = BoxLayout(orientation='vertical')

        # create top Box layout - Top subsection of app
        topBox = BoxLayout(orientation='horizontal', size_hint=(1,0.2))

        # Create and build individual label boxes for topBox
        label_box_1 = BoxLayout(orientation='vertical')
        prob_label = Label(text='PROBABILITY', size_hint=(1,0.3))
        self.prob_label_value = Label(text=self.prob_label_text,font_size='30dp',color=(1,1,1,0))
        label_box_1.add_widget(prob_label)
        label_box_1.add_widget(self.prob_label_value)

        label_box_2 = BoxLayout(orientation='vertical')
        class_label = Label(text='ACTION', size_hint=(1,0.3))
        self.class_label_value = Label(text=self.class_label_text,font_size='30dp',color=(1,1,1,0))
        label_box_2.add_widget(class_label)
        label_box_2.add_widget(self.class_label_value)
        
        label_box_3 = BoxLayout(orientation='vertical')
        rep_label = Label(text='REPS',size_hint=(1,0.3))
        self.rep_label_value = Label(text='Null',font_size='60dp',color=(1,1,1,0))
        label_box_3.add_widget(rep_label)
        label_box_3.add_widget(self.rep_label_value)
        
        # build top box
        topBox.add_widget(label_box_1)
        topBox.add_widget(label_box_2)
        topBox.add_widget(label_box_3)

        # build bottom box - Bottom section of app
        bottomBox = BoxLayout(orientation='horizontal', size_hint=(1,0.2))
        self.play_button = ToggleButton(text=self.toggle_button_text, on_press=self.on_toggle_button_state)
        reset_button = Button(text='Reset', on_press=self.reset_button)
        bottomBox.add_widget(self.play_button)
        bottomBox.add_widget(reset_button)

        # Initialize Camera components - Middle section of app
        self.web_cam = Image(size_hint=(1,0.6))
        self.capture = cv2.VideoCapture(CAP_VIDEO)
        
        # Update the camera every at a framerate of 33 per second 
        Clock.schedule_interval(self.update, 1/33)

        # Build the LSTM RNN Model using Project actions
        self.model = build_model(self.actions)
        self.model.load_weights(WEIGHT_PATH)

        # build super Box
        superBox.add_widget(topBox)
        superBox.add_widget(self.web_cam)
        superBox.add_widget(bottomBox)

        # Return superBox
        return superBox
        
    # Function that runs every frame
    def update(self, *args):

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

                # Need at least 30 frames to make a prediction
                if len(self.last_sequence) == 30:
                    # Use the model to make a prediction with the last 30 frames of data
                    res = self.model.predict(np.expand_dims(self.last_sequence, axis=0), verbose=0)[0]

                    # Variable to hold the location of the highest predicted action
                    index_res = np.argmax(res)

                    # Keep track of all predictions
                    self.predictions.append(index_res)
                    
                    # Get the actual Predicted action label and the probability
                    pred_action = self.actions[index_res]
                    pred_prob = res[index_res]

                    # Counter logic - Needs to be made more robust in the future
                    # Check if last 7 predictions match our current prediction
                    if np.unique(self.predictions[-7:])[0] == index_res:
                        
                        # Check if Probability is higher than 80%
                        if pred_prob >= 0.8:
                            
                            # Used for debugging kivy app, display stats directly on frame
                            # image = display_stats(image, pred_prob, pred_action)
                            
                            # Update Kivy objects with predictions
                            self.prob_label_value.text = str(int(pred_prob*100))+"%"
                            self.prob_label_value.color = (1,1,1,1)
                            self.class_label_value.text = pred_action
                            self.class_label_value.color = (1,1,1,1)

                            # Check if action was previously 'swing', if so count a rep
                            if self.check_action == 'swing' and pred_action == RESET_ACTION:
                                # Update Counter variable
                                self.counter += 1

                                # Update Kivy objects
                                self.rep_label_value.text = str(self.counter)
                                self.rep_label_value.color = (1,1,1,1)

                            # Update the action to check next time with current action
                            self.check_action = pred_action
                    
                    # Used for debugging kivy app, display the counter directly on the fram
                    # image = display_count(image, self.counter)
                    

        # Convert raw OpenCV image array into a texture for rendering to Kivy GUI        
        # put image into buffer        
        buf1 = cv2.flip(image, 0)
        buf = buf1.tobytes()

        # convert image into kivy OpenGL texture
        img_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        
        # update web_cam texture with the new image
        self.web_cam.texture = img_texture

    # Unused, will be updated with the 
    def prediction(self, *args):
        pass

    # Function to enable MediaPipe by clicking a Toggle Button
    def on_toggle_button_state(self, widget):
        if widget.state == "normal":
            widget.text = "Start NOW"
            self.mediapipe_enabled = False
        else:
            widget.text = "STOP"
            self.mediapipe_enabled = True
    
    # Function to reset the values from the previous rep counting session
    def reset_button(self, *args):
        # Probability text and color (r,g,b,0) is transparent
        self.prob_label_value.text = 'Null'
        self.prob_label_value.color = (0,0,0,0)
        # Action class text and color
        self.class_label_value.text = 'Null'
        self.class_label_value.color = (0,0,0,0)
        # Rep count text and color
        self.rep_label_value.text = 'Null'
        self.rep_label_value.color = (0,0,0,0)
        # Reset Counter variable used by prediction
        self.counter = 0

# run the app, close openCV after app is closed
MaceCVApp().run()
cv2.destroyAllWindows()