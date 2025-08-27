import mss
import cv2
import numpy as np
import keyboard
from time import sleep, time

i = 0
space_pressed = False
last_press_time = 0.0
COOLDOWN_S = 0.1

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
        "top": mon["top"] + 30,
        "left": mon["left"] + 88,
        "width": 512,
        "height": 512,
        "mon": monitor_number,
    }

    while True:
        now = time()

        sct_img = np.array(sct.grab(monitor))
        #gray = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2GRAY)

        cv2.imshow("Preview", sct_img)

        if space_pressed and (now - last_press_time) >= COOLDOWN_S:
            cv2.imwrite(f'obstacle-detection/images/obstacle/{i}.png', sct_img)
            obstacle = 'obstacle'
            space_pressed = False
            last_press_time = now
        else:
            cv2.imwrite(f'obstacle-detection/images/no_obstacle/{i}.png', sct_img)
            obstacle = 'no obstacle'

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

        i += 1
        print(f'image {i} saved as {obstacle}')
        sleep(0.2)
