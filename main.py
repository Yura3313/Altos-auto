import mss
import cv2
import numpy as np

sct = mss.mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}


while True:
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 50)
    roi = edges[200:1080, 350:1920]
    
    cv2.imshow("Alto's auto", roi)
    


    if cv2.waitKey(1) == 27:  # ESC
        break

