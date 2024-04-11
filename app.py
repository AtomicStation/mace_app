# Import dependencies for Kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import computer vision dependencies
import cv2
import mediapipe
import tensorflow as tf
import numpy  as np
import os


# 7. Build layout
class AppLayout(BoxLayout):
    pass

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