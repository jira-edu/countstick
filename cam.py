from ultralytics import YOLO
import cv2

# --- 1. การกำหนดค่าเริ่มต้น ---
# โหลดโมเดล YOLOv8 ของคุณ
model = YOLO("/home/jiji/Documents/CountStick/trainedModel/best-v2.pt")

# เปิดกล้องเว็บแคม (0 คือ ID ของกล้องหลัก)
cap = cv2.VideoCapture(2)

# ตรวจสอบว่ากล้องเปิดสำเร็จหรือไม่
if not cap.isOpened():
    print("ข้อผิดพลาด: ไม่สามารถเปิดกล้องเว็บแคมได้")
    exit()
    
# กำหนดหน้าต่างแสดงผล
cv2.namedWindow("Realtime Detection", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Realtime Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # ไม่จำเป็นเสมอไป

# --- 2. ลูปการตรวจจับแบบเรียลไทม์ ---
while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()

    if not ret:
        print("ไม่สามารถรับเฟรมจากกล้องได้ (สิ้นสุดสตรีม?)")
        break

    # --- 3. การประมวลผลด้วย YOLOv8 ---
    # ใช้โมเดลกับเฟรมปัจจุบัน
    # conf=0.5 คือค่าความมั่นใจขั้นต่ำที่คุณกำหนดไว้
    results = model(frame, conf=0.4, verbose=False) # verbose=False เพื่อให้แสดงข้อความน้อยลง

    # ดึงผลลัพธ์จากเฟรมเดียว
    if results:
        result = results[0]
        boxes = result.boxes
        count = len(boxes)
        
        # --- 4. วาดผลลัพธ์บนเฟรม ---
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # ตำแหน่งกล่อง
            conf = box.conf[0]                      # ความมั่นใจ
            cls = int(box.cls[0])                   # คลาสวัตถุ

            # วาดกรอบสี่เหลี่ยม
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # (ทางเลือก) เพิ่มป้ายกำกับคลาสและความมั่นใจ
            # label = f"{model.names[cls]}: {conf:.2f}"
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # แสดงจำนวนวัตถุที่ตรวจพบ
        # ตำแหน่งที่เหมาะสม: มุมซ้ายบนของวิดีโอ (x=20, y=80) 
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

    # --- 5. การแสดงผลและการควบคุม ---
    # แสดงเฟรมที่มีการตรวจจับแล้ว
    cv2.imshow("Realtime Detection", frame)

    # กด 'q' เพื่อออกจากลูป
    # cv2.waitKey(1) จะรอ 1 มิลลิวินาที ทำให้วิดีโอเคลื่อนไหวแบบเรียลไทม์
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. การทำความสะอาด ---
cap.release()
cv2.destroyAllWindows()