from ultralytics import YOLO
import cv2

model = YOLO("/home/jiji/Desktop/best2.pt")


image_path = "/home/jiji/Desktop/2025-11-17-130814.jpg"
img = cv2.imread(image_path)

results = model(img, conf=0.5)

result = results[0]  # ภาพเดียว
boxes = result.boxes  # กล่องตรวจจับทั้งหมด

count = len(boxes)

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # ตำแหน่งกล่อง
    conf = box.conf[0]                      # ความมั่นใจ
    cls = int(box.cls[0])                   # คลาสวัตถุ

    # วาดกรอบ
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


cv2.putText(
    img,
    str(count),
    (80, 250),  # ตำแหน่ง (x, y)
    cv2.FONT_HERSHEY_SIMPLEX,
    7.0,       # ขนาดฟอนต์
    (0, 0, 255),  # สีแดง (B,G,R)
    12,         # ความหนาเส้น
    cv2.LINE_AA
)

print("จำนวนไม้:", count)

cv2.namedWindow("Detected Sticks", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Detected Sticks", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Detected Sticks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()