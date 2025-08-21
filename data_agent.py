import cv2
import numpy as np
import mss
import pyautogui

monitor = {"top": 35, "left": 0, "width": 900, "height": 400}
sct = mss.mss()

while True:
    frame = np.array(sct.grab(monitor))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(frame, 50, 50)

    # Region in front of character (adjust as needed)
    roi = edges[400:500, 200:400]

    # Find contours (possible obstacles)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacle_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:   # filter small noise
            obstacle_detected = True
            break                            

    if obstacle_detected:
        pyautogui.keyDown("space")
        pyautogui.keyUp("space")

    cv2.imshow("Edges", roi)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break
