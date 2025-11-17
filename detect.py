from ultralytics import YOLO
import cv2

model = YOLO("/home/jiji/Desktop/best.pt")


image_path = "/home/jiji/Desktop/dataset/LINE_ALBUM_14112025_251115_473.jpg"
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
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

print("จำนวนไม้:", count)
cv2.imshow("Detected Sticks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()