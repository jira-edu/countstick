#include <Arduino.h>
#include <ESP32Servo.h> 
Servo myservo;
int servoPin = 18;

#define sw 2
#define lampRed 3
#define lampGreen 4
#define sensor 5

void setup() {
  pinMode(sw, INPUT_PULLUP);
  pinMode(sensor, INPUT);
  pinMode(lampGreen, OUTPUT);
  pinMode(lampRed, OUTPUT);
  Serial.begin(115200);
  ESP32PWM::allocateTimer(0);
	ESP32PWM::allocateTimer(1);
	ESP32PWM::allocateTimer(2);
	ESP32PWM::allocateTimer(3);
  myservo.setPeriodHertz(50);// Standard 50hz servo
  myservo.attach(servoPin);
}

void loop() {
  if (digitalRead(sw) == LOW) {
    digitalWrite(lampRed, HIGH);
    digitalWrite(lampGreen, LOW);
    myservo.write(120); 
    delay(500);
    while (digitalRead(sensor) == HIGH) {
      delay(10);
    }
    myservo.write(90); // stop motor 
    delay(1000);
    digitalWrite(lampRed, LOW);
    digitalWrite(lampGreen, HIGH);
    Serial.println('c');
    while (digitalRead(sw) == LOW) {
      delay(100);
    }
  }
}

