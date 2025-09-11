import time, datetime
import collections
import numpy as np
import torch
import mss
import cv2
import pyautogui
from torchvision import models
import torch.nn as nn
from torch.amp.autocast_mode import autocast

# ----------------- CONFIG -----------------
model_name = "v2_28-08"
MODEL_PATH = f"obstacle-detection/models/{model_name}.pth"
THRESH_OBSTACLE = 0.55
monitor_number = 2
ROI = (350, 100, 1290, 1040)
IMG_SIZE = 384
status_check_freq = 20
# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[INFO] Device: {device}")
if device.type == "cuda":
    status_check_freq = 50
    torch.backends.cudnn.benchmark = True

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
model.to(device)

# if CUDA -> allow half precision for speed
use_fp16 = (device.type == "cuda")
if use_fp16:
    model.half()
softmax = nn.Softmax(dim=1)

last_obstacle_t = 0.0
frame_counter = 0
playing = True
scheduled_actions = []
t0 = time.time()
fps_history = collections.deque(maxlen=30)

sct = mss.mss()
mon = sct.monitors[monitor_number]
x1, y1, x2, y2 = ROI
w = x2 - x1
h = y2 - y1
print(f"[INFO] Capturing monitor: {monitor_number}, ROI size: {w}x{h}")

monitor = {
    "top": mon["top"] + 100,
    "left": mon["left"] + 350,
    "width": 940,
    "height": 940,
    "mon": monitor_number,
}

stats_frame = {
    'top': mon['top'] + 30,
    'left': mon['left'] + 1630,
    'width': 270,
    'height': 140,
    'mon': monitor_number
}


print(f"[INFO] Starting using {model_name}. Press ESC on preview window to quit.")
while True:
    loop_start = time.time()

    raw = np.array(sct.grab(monitor))
    rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
    

    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    tensor = tensor.float().div(255.0)
    if use_fp16:
        tensor = tensor.half()

    # inference
    with torch.no_grad():
        if use_fp16:
            with autocast('cuda', dtype=torch.float16):
                logits = model(tensor)
        else:
            logits = model(tensor)
        probs = softmax(logits)
        p_obstacle = float(probs[0, 0].cpu().numpy())

    # cooldown & action
    now = time.time()
    if p_obstacle < THRESH_OBSTACLE:
        pyautogui.press("space")
        last_obstacle_t = now
        status = ("OBSTACLE", (255, 0, 0))
        print(f'[INFO] Obstacle! at {datetime.datetime.now().strftime("%H:%M:%S")}, p={p_obstacle:.3f}')
    else:
        status = ("NO OBSTACLE", (0, 255, 0))

    preview = raw
    label = f"{status[0]}  p={p_obstacle:.2f}  fps={np.mean(fps_history) if fps_history else 0:.1f} state={'in game' if playing else 'not in game'}"
    cv2.putText(raw, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status[1], 2)
    cv2.imshow("Game (cropped preview)", raw)

    # fps calc
    loop_dt = time.time() - loop_start
    fps_history.append(1.0 / loop_dt if loop_dt > 0 else 0.0)



    # state detection
    if frame_counter % status_check_freq == 0:
        stats_img = np.array(sct.grab(stats_frame))
        m1 = stats_img[39, 215]
        m2 = stats_img[51, 256]
        if m1[0] == m2[0] and m1[1] == m2[1] and m1[2] == m2[2]:
            print(f'[INFO] In game detected at {datetime.datetime.now().strftime("%H:%M:%S")}, frame {frame_counter}')
            playing = True
        else:
            if playing:
                now = time.time()
                scheduled_actions.append((now + 2, "space"))
                scheduled_actions.append((now + 3, "space"))
                print(f'[INFO] Out of game detected at {datetime.datetime.now().strftime("%H:%M:%S")}, frame {frame_counter}, scheduling respawn')
            playing = False

    if scheduled_actions and now >= scheduled_actions[0][0]:
        _, key = scheduled_actions.pop(0)
        pyautogui.press(key)



    frame_counter += 1
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
print("[INFO] Stopped.")
