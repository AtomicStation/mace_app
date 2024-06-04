import cv2
import time
import mediapipe as mp
from custom_model import *


prev_video = 'videos/5_mace_swings.mp4'
# prev_video = 'videos/clubbell.mp4'
new_video = 'videos/test.mp4'
cap = cv2.VideoCapture(prev_video)
pTime = 0
frameTime = 33

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_rate = int(video_fps)//4

ret,frame = cap.read()
h, w, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(new_video, fourcc, video_fps, (w,h))

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while ret:
        
        # cTime = time.time()
        image, results = mediapipe_detection(frame, holistic)

        draw_pose_landmarks(image, results)


        
        # fps = 1/(cTime - pTime)
        # pTime = cTime
        
        writer.write(image)
        # cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 3)
        # cv2.imshow("Image",image)
        
        
        # if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
        #     break
        ret, frame = cap.read()












# while ret:
#     writer.write(frame)
#     cv2.imshow("frame", frame)
    
    

writer.release()
cap.release()
cv2.destroyAllWindows()
print(video_fps)
print(type(video_fps))