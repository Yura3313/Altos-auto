import mss
import cv2
import numpy as np
import keyboard
from time import sleep, time
from termcolor import colored

i = 0
space_pressed = False
last_press_time = 0.0
COOLDOWN_S = 0.3
fps = 20

now = time()

# This runs asynchronously and sets a flag
def on_space(e):
    global space_pressed
    space_pressed = True

keyboard.on_press_key("space", on_space)

with mss.mss() as sct:
    monitor_number = 2
    mon = sct.monitors[monitor_number]

    monitor = {
        "top": mon["top"] + 100,
        "left": mon["left"] + 350,
        "width": 940,
        "height": 940,
        "mon": monitor_number,
    }

    while True:
        now = time()

        sct_img = np.array(sct.grab(monitor))

        cv2.imshow("Preview", sct_img)

        if space_pressed and (now - last_press_time) >= COOLDOWN_S:
            cv2.imwrite(f'obstacle-detection/images/obstacle/{i}.png', sct_img)
            obstacle = True
            space_pressed = False
            last_press_time = now
        else:
            cv2.imwrite(f'obstacle-detection/images/no_obstacle/{i}.png', sct_img)
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
