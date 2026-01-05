from gpiozero import Button
from time import sleep

import cv2
from picamera2 import Picamera2
from datetime import datetime
from ultralytics import YOLO

model = YOLO("trainedModel/best-v3.pt")

from tkinter import messagebox

greenCount, redCount, purpleCount, blueCount = 0

greenPrice = 5
redPrice = 10
purplePrice = 20
bluePrice = 30
promptPayNumber = "0926582873"
from lib import receipt, dominateColor

# Initialize camera
picam = Picamera2()
camera_config = picam.create_preview_configuration(main={"size": (1920, 1080),"format": "RGB888"})
picam.configure(camera_config)
picam.start()
picam.set_controls({"AfMode": 2, "AfTrigger": 0})

# Initialize proximity sensor
prox = Button(25, pull_up=True)

def time_stamp():
    now = datetime.now()
    return now.strftime("%y%m%d-%H%M%S")

def detectStick(frame):
    global greenCount, redCount, purpleCount, blueCount
    greenCount, redCount, purpleCount, blueCount = 0

    results = model(frame, conf=0.5, verbose=False)
    if results:
        result = results[0]
        boxes = result.boxes
        count = len(boxes)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

        cropped_image = frame[y1:y2, x1:x2]
        # cv2.imwrite("temp.jpg", cropped_image)
        stickColor = dominateColor.getColor(cropped_image)
        if stickColor == 'เขียว':
            greenCount += 1
            boxColor = (0,255,0)
        elif stickColor == 'ม่วง':
            purpleCount += 1
            boxColor = (128,0,128)
        elif stickColor == 'แดง':
            redCount += 1
            boxColor = (0,0,255)
        elif stickColor == 'น้ำเงิน':
            blueCount += 1
            boxColor = (255,0,0)
        else:
            boxColor = (0,0,0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, 2)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # show stick count each color
    cv2.putText(
        frame,
        f"green:{greenCount}\npurple:{purpleCount}\nred:{redCount}\nblue:{blueCount}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 255),
        5,
        cv2.LINE_AA
    )
    return frame

while True:
    frame = picam.capture_array()
    cv2.imshow("CountStick", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Check proximity sensor
    if prox.is_pressed:
        # wait a moment to capture stable frames
        count_down = 6
        for i in range(50):
            if (i % 10 == 0):
                count_down-=1
            frame = picam.capture_array()
            cv2.putText(
                frame,
                f"Countdown: {count_down}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 0, 0),
                5,
                cv2.LINE_AA
            )
            cv2.imshow("CountStick", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            sleep(0.1)
        
        print("มาแล้ว")
        frame = picam.capture_array()
        frame = detectStick(frame)
        print("เสร็จแล้ว")

        while prox.is_pressed:
            cv2.imshow("CountStick", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            sleep(0.1)
        print("วัตถุหายไป!")
        sleep(1)
    sleep(0.1)

picam.stop()
cv2.destroyAllWindows()