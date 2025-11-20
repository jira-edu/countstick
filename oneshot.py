from ultralytics import YOLO
import cv2
import time

# โหลดโมเดล YOLO
model = YOLO("/home/jiji/Documents/CountStick/trainedModel/best-v2-1.pt")

# เปิดเว็บแคม
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: ไม่สามารถเปิดกล้องเว็บแคมได้")
    exit()

cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
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
