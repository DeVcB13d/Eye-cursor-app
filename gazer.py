import cv2

import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Load the calibration points
calib_points_right = np.load('calib_points_right.npy')
calib_points_left = np.load('calib_points_left.npy')



while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    #print(landmark_points)

    if landmark_points:
        landmarks = landmark_points[0].landmark
        right_eye_landmarks = [landmarks[474], landmarks[476]]
        left_eye_landmarks = [landmarks[145], landmarks[159]]

        for landmark in right_eye_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        for landmark in left_eye_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Compute the average x and y coordinates of the right eye
        x_avg_right = np.mean([landmark.x * frame.shape[1] for landmark in right_eye_landmarks])
        y_avg_right = np.mean([landmark.y * frame.shape[0] for landmark in right_eye_landmarks])

        # Map the average x and y coordinates using the calibration points
        screen_x_right = np.interp(x_avg_right, calib_points_right[0], calib_points_right[1])
        screen_y_right = np.interp(y_avg_right, calib_points_right[2], calib_points_right[3])


        # Move the mouse to the average x and y coordinates of the right eye
        pyautogui.moveTo(screen_x_right, screen_y_right)

        #     cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        #     if id == 1:
        #         screen_x = int(x * screen_width / frame.shape[1])
        #         screen_y = int(y * screen_height / frame.shape[0])
        #         pyautogui.moveTo(screen_x, screen_y)
        # left = [landamarks[145],landamarks[159]]
        # for landmark in left:
        #     x = int(landmark.x * frame.shape[1])
        #     y = int(landmark.y * frame.shape[0])
        #     cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

