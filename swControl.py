from ultralytics import YOLO
import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox
import serial.tools.list_ports
import numpy as np
import sys, os

def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def list_webcams(max_test=6):
    """คืน list ของ id กล้อง เช่น [0,1,2]"""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def start_program():
    sel_port = combo_serial.get()
    sel_cam_text = combo_cam.get()

    if not sel_port:
        messagebox.showerror("Error", "กรุณาเลือก Serial Port")
        return
    if not sel_cam_text:
        messagebox.showerror("Error", "กรุณาเลือก Webcam")
        return

    # ======= วิธี A: ถ้าใช้ label "Camera 0" และมี mapping =======
    if sel_cam_text in cam_label_to_id:
        sel_cam_id = cam_label_to_id[sel_cam_text]
    else:
        # ======= วิธี B: ถ้า combobox เก็บเป็นตัวเลขเป็น string เช่น "0" =======
        try:
            sel_cam_id = int(sel_cam_text)
        except ValueError:
            messagebox.showerror("Error", "รูปแบบ id กล้องไม่ถูกต้อง")
            return

    root.selected_port = sel_port
    root.selected_cam  = sel_cam_id
    root.destroy()

# ============================
#   UI WINDOW
# ============================
root = tk.Tk()
root.title("เลือก Serial Port และ Webcam")
root.geometry("380x230")

# ==== Serial Port ====
tk.Label(root, text="Serial Port:", font=("Arial", 11)).pack(pady=(10,4))
serial_ports = list_serial_ports()
combo_serial = ttk.Combobox(root, values=serial_ports, state="readonly")
if serial_ports:
    combo_serial.current(0)
combo_serial.pack(fill="x", padx=20)

# ==== Webcam ====
tk.Label(root, text="Webcam:", font=("Arial", 11)).pack(pady=(10,4))
webcam_ids = list_webcams(max_test=6)  # ปรับถ้าต้องการสแกนมากกว่า

# วิธี A: สร้าง label สวย ๆ แต่เก็บ mapping -> แปลงกลับเป็น int
cam_label_to_id = {f"Camera {i}": i for i in webcam_ids}
cam_labels = list(cam_label_to_id.keys())

# วิธี B: แสดงเป็น "0", "1", "2" (string) แล้วแปลงเป็น int เมื่ออ่านค่า
cam_id_strings = [str(i) for i in webcam_ids]

# --- เลือกวิธีการแสดงผลใน combobox ให้ uncomment หนึ่งบรรทัดด้านล่าง ---
# combo_cam = ttk.Combobox(root, values=cam_labels, state="readonly")   # วิธี A: แสดง "Camera 0"
combo_cam = ttk.Combobox(root, values=cam_id_strings, state="readonly")  # วิธี B: แสดง "0", "1", ...

if webcam_ids:
    combo_cam.current(0)
combo_cam.pack(fill="x", padx=20)

# ==== Start Button ====
tk.Button(root, text="เริ่มโปรแกรม", font=("Arial", 11), command=start_program).pack(pady=18)

root.mainloop()

selected_serial = getattr(root, "selected_port", None)
selected_camera = getattr(root, "selected_cam", None)
# -------------------------------------------------------


root = tk.Tk()
root.withdraw()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

# โหลดโมเดล YOLO
model = YOLO("/home/jiji/Documents/CountStick/trainedModel/best-v3.pt")

try:
    ser = serial.Serial(selected_serial, 115200, timeout=1)
except serial.serialutil.SerialException:
    messagebox.showerror("Error", "Cannot connect to Port "+selected_serial)
    sys.exit()

try:
    if not ser.isOpen():
        ser.open()
except serial.SerialException as e:
    print(f"Serial port error: {e}")

# เปิดเว็บแคม
cap = cv2.VideoCapture(selected_camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Error: ไม่สามารถเปิดกล้องเว็บแคมได้")
    exit()

logo = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
scale_logo = 150 / logo.shape[1]
new_w = int(logo.shape[1] * scale_logo)
new_h = int(logo.shape[0] * scale_logo)
logo = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

def overlay_png(background, overlay, x, y):    
    h, w = overlay.shape[:2]

    # แยก channel
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        alpha = a.astype(float) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])
        overlay_rgb = cv2.merge([b, g, r]).astype(float)
    else:
        # PNG ไม่มี alpha
        alpha = np.ones((h, w, 3), dtype=float)
        overlay_rgb = overlay.astype(float)

    # พื้นหลังเฉพาะตำแหน่งที่จะวาง
    roi = background[y:y+h, x:x+w].astype(float)

    # blend
    blended = alpha * overlay_rgb + (1 - alpha) * roi
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background

def imProcess():
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        return
    results = model(frame, conf=0.5, verbose=False)
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
    h, w = frame.shape[:2]
    x = w - logo.shape[1] - 20   # ขยับจากขอบขวา 20px
    y = h - logo.shape[0] - 20   # ขยับจากขอบล่าง 20px

    frame = overlay_png(frame, logo, x, y)
    cv2.imshow("Detection Result", frame)
    return frame

while True:
    frame = imProcess()
    if ser.in_waiting:  # ถ้ามีข้อมูลอยู่ในบัฟเฟอร์
        data = ser.readline().decode('utf-8', errors='ignore').strip()
        if data:
            if data == 'c':
                frame = imProcess()
            else:
                pass
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()