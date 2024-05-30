import cv2

import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize Kalman filter
dt = 1.0  # Time step
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)



        

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    #print(landmark_points)

    if landmark_points:
        landamarks = landmark_points[0].landmark
        print(len(landamarks))
        for id,landmark in enumerate(landamarks[474:478]):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            if id == 1:
                screen_x = int(x * screen_width / frame.shape[1])
                screen_y = int(y * screen_height / frame.shape[0])
                pyautogui.moveTo(screen_x, screen_y)
        left = [landamarks[145],landamarks[159]]
        for landmark in left:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        if (left[0].y-left[1].y) < 0.018:
            print(left[0].y-left[1].y)
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

