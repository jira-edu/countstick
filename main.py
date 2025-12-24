from gpiozero import Button
from time import sleep

import cv2
from picamera2 import Picamera2
from datetime import datetime

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

while True:
    frame = picam.capture_array()
    cv2.imshow("CountStick", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Check proximity sensor
    if prox.is_pressed:
        # wait a moment to capture stable frames
        for i in range(50):
            frame = picam.capture_array()
            cv2.imshow("CountStick", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            sleep(0.1)
        
        print("เจอวัตถุ!", time_stamp())
        
        while prox.is_pressed:
            sleep(0.2)
        print("วัตถุหายไป!")
        sleep(1)
    sleep(0.1)

picam.stop()
cv2.destroyAllWindows()