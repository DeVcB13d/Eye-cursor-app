import cv2

import mediapipe as mp
import pyautogui
import numpy as np

'''

This file contains the calibration function which is used to calibrate the eye tracker. 
The calibration function takes the face_mesh object, the camera object, and the x and y 
coordinates of the calibration point as input and returns the average x and y coordinates 
of the eye as output. The calibration function works by taking 100 samples of the x and y 
coordinates of the eye and then calculating the average of these coordinates. 
'''

def get_calib_points(face_mesh:mp.solutions.face_mesh.FaceMesh,
                cam:cv2.VideoCapture,
                dot_x:int,
                dot_y:int,
                n_samples:int=500
            ):
    # Compute the screen width and height
    screen_width, screen_height = pyautogui.size()
    # Create a white background
    blank = np.zeros((screen_height, screen_width, 3), np.uint8)
    # Display the white backgroun
    print('Calibration started, look into the red dots')


    cv2.circle(blank, (dot_x - 5, dot_y - 5), 20, (255, 0, 255), -1)
    cv2.imshow('blank', blank)

    # Store the x and y coordinates of the eye
    x_avg_list_right, y_avg_list_right = [], []
    x_avg_list_left, y_avg_list_left = [], []
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Right Eye
            x_avg, y_avg = [], []

            right_eye_landmarks = [landmarks[474], landmarks[476]]
            for landmark in right_eye_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                x_avg.append(x)
                y_avg.append(y)
            # Append the centroid of the eye to the list
            x_avg_list_right.append(np.mean(x_avg))
            y_avg_list_right.append(np.mean(y_avg))
            

            # Left Eye
            left_eye_landmarks = [landmarks[145], landmarks[159]]
            x_avg, y_avg = [], []
            for landmark in left_eye_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                x_avg.append(x)
                y_avg.append(y)
            x_avg_list_left.append(np.mean(x_avg))
            y_avg_list_left.append(np.mean(y_avg))
            if len(x_avg_list_right) == n_samples:
                break

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Remove outliers
    x_avg_list_right = np.array(x_avg_list_right)
    y_avg_list_right = np.array(y_avg_list_right)
    x_avg_list_right = x_avg_list_right[abs(x_avg_list_right - np.mean(x_avg_list_right)) < 2 * np.std(x_avg_list_right)]
    y_avg_list_right = y_avg_list_right[abs(y_avg_list_right - np.mean(y_avg_list_right)) < 2 * np.std(y_avg_list_right)]

    x_avg_list_left = np.array(x_avg_list_left)
    y_avg_list_left = np.array(y_avg_list_left)
    x_avg_list_left = x_avg_list_left[abs(x_avg_list_left - np.mean(x_avg_list_left)) < 2 * np.std(x_avg_list_left)]
    y_avg_list_left = y_avg_list_left[abs(y_avg_list_left - np.mean(y_avg_list_left)) < 2 * np.std(y_avg_list_left)]

    x_avg_right = int(np.mean(x_avg_list_right))
    y_avg_right = int(np.mean(y_avg_list_right))

    x_avg_left = int(np.mean(x_avg_list_left))
    y_avg_left = int(np.mean(y_avg_list_left))

    return x_avg_right, y_avg_right, x_avg_left, y_avg_left

def calibrate():
    # Calibrate the eye for 5 points 
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    # Get screen width and height
    screen_width, screen_height = pyautogui.size()

    # REduce the screen width and height by 100 pixels
    screen_width -= 100
    screen_height -= 100
    # Calibration points
    points = [
        (0,0), # Top left
        (screen_width - 1,0), # Top right
        (screen_width - 1, screen_height - 1), # Bottom right
        (0, screen_height - 1), # Bottom left
        (screen_width // 2, screen_height // 2) # Center
    ]

    # Shift the calibration points by 5 pixels
    points = [(point[0] + 5, point[1] + 5) for point in points]



    calib_points_right, calib_points_left = [], []
    for point in points:
        x_right, y_right, x_left, y_left = get_calib_points(face_mesh, cam, point[0], point[1])
        calib_points_right.append((x_right, y_right))
        calib_points_left.append((x_left, y_left))
    cam.release()

    print("Calibration completed successfully!")
    print("Calibration points for the right eye: ", calib_points_right)
    print("Calibration points for the left eye: ", calib_points_left)
    # Save the calibration points
    np.save('calib_points_right.npy', np.array(calib_points_right))
    np.save('calib_points_left.npy', np.array(calib_points_left))

if __name__ == '__main__':
    calibrate()



        
