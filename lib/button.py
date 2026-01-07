import cv2

buttons = []   # เก็บปุ่มทั้งหมด

def add_button(x, y, button_img_path, callback):
    btn = cv2.imread(button_img_path, cv2.IMREAD_UNCHANGED)
    btn = cv2.resize(btn, (80,80))
    h, w = btn.shape[:2]
    buttons.append({
        "x1": x, "y1": y, "x2": x+w, "y2": y+h,
        "img": btn,
        "callback": callback
    })

def draw_buttons(frame):
    for b in buttons:
        btn = b["img"]

        if btn.shape[2] == 4:
            bh, bw = btn.shape[:2]

            roi = frame[b["y1"]:b["y2"], b["x1"]:b["x2"]]

            bgr = btn[:,:,:3]
            alpha = btn[:,:,3] / 255.0

            alpha = alpha[..., None]

            roi[:] = alpha * bgr + (1 - alpha) * roi
        else:
            frame[b["y1"]:b["y2"], b["x1"]:b["x2"]] = btn

        # frame[b["y1"]:b["y2"], b["x1"]:b["x2"]] = b["img"]
    return frame

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for b in buttons:
            if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
                b["callback"]()

# def btn1_action():
#     print("กดปุ่ม 1 แล้ว")

# def btn2_action():
#     print("กดปุ่ม 2 แล้ว")