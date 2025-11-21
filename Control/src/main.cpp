#include <Arduino.h>
#include <ESP32Servo.h> 
Servo motor;
int servoPin = 25;

#define sw 21
#define lampRed 18
#define lampGreen 19
#define sensor 23

void setup() {
  pinMode(sw, INPUT_PULLUP);
  pinMode(sensor, INPUT);
  pinMode(lampGreen, OUTPUT);
  pinMode(lampRed, OUTPUT);
  digitalWrite(lampGreen, LOW);
  digitalWrite(lampRed, HIGH);
  Serial.begin(115200);
  ESP32PWM::allocateTimer(0);
	ESP32PWM::allocateTimer(1);
	ESP32PWM::allocateTimer(2);
	ESP32PWM::allocateTimer(3);
  motor.setPeriodHertz(50);// Standard 50hz servo
  motor.attach(servoPin);
}

void loop() {
  if (digitalRead(sw) == LOW) {
    digitalWrite(lampRed, HIGH);
    digitalWrite(lampGreen, LOW);
    motor.write(100); 
    delay(150);
    motor.write(110); 
    delay(150);
    motor.write(120); 
    delay(150);
    motor.write(130); 
    delay(150);
    motor.write(140); 
    delay(150);

    while (digitalRead(sensor) == HIGH) {
      delay(10);
    }

    motor.write(90); // stop motor 
    delay(1000);
    digitalWrite(lampRed, LOW);
    digitalWrite(lampGreen, HIGH);
    Serial.println('c');
    while (digitalRead(sw) == LOW) {
      delay(100);
    }
  }
  delay(100);
}

