from gpiozero import Button, LED
from time import sleep

button = Button(25, pull_up=True)
beep = LED(18)

while True:
    if button.is_pressed:
        print("เจอวัตถุ!")
        beep.on()
        while button.is_pressed:
            sleep(0.1)
        print("วัตถุหายไป!")
        beep.off()
    sleep(0.1)