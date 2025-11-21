from ultralytics import YOLO
import cv2
import os, sys
import tkinter as tk
from tkinter import ttk, messagebox

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def list_webcams(max_test=6):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def start_program():
    sel_cam_text = combo_cam.get()
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

    root.selected_cam  = sel_cam_id
    root.destroy()

root = tk.Tk()
root.title("เลือก Webcam")
root.geometry("380x230")
tk.Label(root, text="Webcam:", font=("Arial", 11)).pack(pady=(10,4))
webcam_ids = list_webcams(max_test=6) 
cam_label_to_id = {f"Camera {i}": i for i in webcam_ids}
cam_labels = list(cam_label_to_id.keys())
cam_id_strings = [str(i) for i in webcam_ids]
combo_cam = ttk.Combobox(root, values=cam_id_strings, state="readonly")  # วิธี B: แสดง "0", "1", ...
if webcam_ids:
    combo_cam.current(0)
combo_cam.pack(fill="x", padx=20)
tk.Button(root, text="เริ่มโปรแกรม", font=("Arial", 11), command=start_program).pack(pady=18)

root.mainloop()

selected_camera = getattr(root, "selected_cam", None)

model = YOLO(resource_path(os.path.join('trainedModel', 'best-v3.pt')))

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("ข้อผิดพลาด: ไม่สามารถเปิดกล้องเว็บแคมได้")
    exit()
    
cv2.namedWindow("Realtime Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Realtime Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # ไม่จำเป็นเสมอไป

while True:
    ret, frame = cap.read()

    if not ret:
        print("ไม่สามารถรับเฟรมจากกล้องได้ (สิ้นสุดสตรีม?)")
        break

    results = model(frame, conf=0.5, verbose=False) # verbose=False เพื่อให้แสดงข้อความน้อยลง

    if results:
        result = results[0]
        boxes = result.boxes
        count = len(boxes)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # ตำแหน่งกล่อง
            conf = box.conf[0]                      # ความมั่นใจ
            cls = int(box.cls[0])                   # คลาสวัตถุ

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
  
        cv2.putText(
            frame,
            f"Count: {count}",
            (20, 80),  
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,       
            (0, 0, 255),  # สีแดง 
            5,         
            cv2.LINE_AA
        )

    cv2.imshow("Realtime Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()