from ultralytics import YOLO
import cv2
import time
import tkinter as tk
import numpy as np

root = tk.Tk()
root.withdraw()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

splash = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)

if splash is None:
    print("ไม่พบไฟล์ splash.png")
else:
    h, w = splash.shape[:2]

    scale = min(screen_w / w, screen_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    splash_resized = cv2.resize(splash, (new_w, new_h), interpolation=cv2.INTER_AREA)

    b, g, r, a = cv2.split(splash_resized)

    alpha = a.astype(float) / 255.0
    alpha_3 = cv2.merge([alpha, alpha, alpha])

    fg = cv2.merge([b, g, r]).astype(float)

    background_color = (255, 255, 255)  # ← เปลี่ยนสีพื้นหลังที่นี่ (B,G,R)
    background = np.full((new_h, new_w, 3), background_color, dtype=np.uint8).astype(float)

    blended = (alpha_3 * fg + (1 - alpha_3) * background).astype(np.uint8)

    cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Detection Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Detection Result", blended)
    cv2.waitKey(2000)   # แสดง 2 วินาที
    # cv2.destroyWindow("Splash")

# โหลดโมเดล YOLO
model = YOLO("/home/jiji/Documents/CountStick/trainedModel/best-v2-1.pt")

# เปิดเว็บแคม
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: ไม่สามารถเปิดกล้องเว็บแคมได้")
    exit()

# cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Detection Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
freeze_frame = None

print("กด SPACE เพื่อถ่ายภาพและตรวจจับ | กด Q เพื่อออก")

while True:
    ret, frame = cap.read()
    cv2.imshow("Detection Result", frame)

    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        continue
    
    # --- รอการกดปุ่ม ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' '):   # SPACE BAR ถูกกด
        
        # --- YOLO Detection ---
        results = model(frame, conf=0.4, verbose=False)
        if results:
            result = results[0]
            boxes = result.boxes
            count = len(boxes)

        # วาด bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # แสดงจำนวนที่นับได้
        cv2.putText(
            frame,
            f"Count: {count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 255),
            5,
            cv2.LINE_AA
        )

        # --- แสดงภาพแบบค้าง (freeze frame) ---
        cv2.imshow("Detection Result", frame)

        while cv2.waitKey(1) & 0xFF != ord(' '):
            cv2.imshow("Detection Result", frame)
            time.sleep(0.2)

# cleanup
cap.release()
cv2.destroyAllWindows()
