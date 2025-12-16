import cv2
from picamera2 import Picamera2
import time
from datetime import datetime

picam = Picamera2()
camera_config = picam.create_preview_configuration(main={"size": (1920, 1080),"format": "RGB888"})
picam.configure(camera_config)
picam.start()
picam.set_controls({"AfMode": 2, "AfTrigger": 0})

while True:
    frame = picam.capture_array()
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera Module 3 OpenCV Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        now = datetime.now()
        timestamp = now.strftime("%y%m%d-%H%M%S")
        cv2.imwrite('/home/countstick/Desktop/Images/'+timestamp+'.jpg', frame)
        print('Captured')

picam.stop()
cv2.destroyAllWindows()