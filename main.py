import time, numpy as np, torch, mss, cv2, pyautogui, torch.nn as nn
from termcolor import colored
from torchvision import transforms, models  
from torch.amp import autocast
from PIL import Image

# ==== SETTINGS ====   
def build_model(name, num_classes=2):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenetv3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenetv3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model

model_name = 'mobilenetv3_small'
MODEL_PATH = "obstacle-detection/models/1756297986.726654.pth"
THRESH_obstacle = 0.6 # !!!
COOLDOWN_S  = 0.35
monitor_number = 1

ROI = (30, 88, 542, 600)

# ==== DEVICE ====
device = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[INFO] Device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ==== MODEL ====
model = build_model(model_name, num_classes=2)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval().to(device)

pre = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
 
softmax = nn.Softmax(dim=1)

sct = mss.mss()
mon = sct.monitors[monitor_number]

y1, x1, y2, x2 = ROI

monitor = {
    "top": mon["top"] + y1,
    "left": mon["left"] + x1,
    "width": x2 - x1,
    "height": y2 - y1,
    "mon": monitor_number,
}


print(f"[INFO] Capturing monitor: {monitor}")

last_obstacle_t = 0.0

print("[INFO] Running. Press ESC on preview window to quit.")
while True:

    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    pil_ready = Image.fromarray(frame)

    # preprocess -> tensor on GPU (ensure 1 channel)
    tensor = pre(pil_ready).unsqueeze(0).to(device, non_blocking=True)  # shape [1,1,H,W]

    with torch.no_grad():
        with autocast('cuda'):
            logits = model(tensor)
            probs = softmax(logits)[0].detach().cpu().numpy()  # [p_obstacle, p_dont]

    p_obstacle = float(probs[0])

    
    # action with cooldown
    now = time.time()
    if p_obstacle <= THRESH_obstacle and (now - last_obstacle_t) >= COOLDOWN_S:
        pyautogui.keyDown("space")
        pyautogui.keyUp("space")
        last_obstacle_t = now
        print(colored(f'Obstacle found! Confidance {p_obstacle:.2f}', 'blue'))    
    else:
        print(colored(f'No obstacle. Confidance {p_obstacle:.2f}', 'green'))

    def preview():
        disp = frame.copy()
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"confidance: {p_obstacle:.2f}"
        color = (0,0,255) if last_obstacle_t == now else (0,255,0)
        cv2.putText(disp, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Screen", disp)
    preview()
    # exit
    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
print("[INFO] Stopped.")
