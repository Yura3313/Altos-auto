import mss
import cv2
import numpy as np
import keyboard
from time import sleep, time
from termcolor import colored


space_pressed = False
last_press_time = 0.0
cooldown_i = 3
fps = 20
ROI = (230, 100, 1310, 1060)

now = time()
x1, y1, x2, y2 = ROI
w = x2 - x1
h = y2 - y1


with mss.mss() as sct:
    i = 0
    monitor_number = 2
    mon = sct.monitors[monitor_number]
    print(f"[INFO] Capturing monitor: {monitor_number}, ROI size: {w}x{h}")
    monitor = {
    "top": mon["top"] + y1,
    "left": mon["left"] + x1,
    "width": w,
    "height": h,
    "mon": monitor_number}       



    while True:
        now = time()

        sct_img = np.array(sct.grab(monitor))
        resized = cv2.resize(sct_img, (384, 384))

        cv2.imshow("Preview", resized)

        if keyboard.is_pressed('space') or keyboard.is_pressed('c'):
            cv2.imwrite(f'obstacle-detection/images/obstacle/{i}.png', resized)
            obstacle = True
            space_pressed = False
            last_press_time = i
        else:
            cv2.imwrite(f'obstacle-detection/images/no_obstacle/{i}.png', resized)
            obstacle = False

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

        i += 1
        if obstacle is True:
            print(colored(f'image {i} saved as obstacle', 'red'))
        else:
            print(colored(f'image {i} saved as not obstacle', 'green'))
        sleep(1 / fps)
