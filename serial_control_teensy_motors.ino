#include <Servo.h>

// Motor A
const int INA1 = 5;
const int INA2 = 6;
const int PWMA = 7;

// Motor B
const int INB1 = 8;
const int INB2 = 9;
const int PWMB = 10;

// Motor C
const int INC1 = 20;
const int INC2 = 19;
const int PWMC = 18;

// Motor D
const int IND1 = 17;
const int IND2 = 16;
const int PWMD = 15;

const int servoPin1 = 36;
const int servoPin2 = 37;

const int pump = 25;

const int seed1 = 2;
const int seed2 = 3;

Servo servo1;
Servo servo2;

int speedVal = 255;

void setup() {
  pinMode(INA1, OUTPUT);
  pinMode(INA2, OUTPUT);
  pinMode(PWMA, OUTPUT);

  pinMode(INB1, OUTPUT);
  pinMode(INB2, OUTPUT);
  pinMode(PWMB, OUTPUT);

  pinMode(INC1, OUTPUT);
  pinMode(INC2, OUTPUT);
  pinMode(PWMC, OUTPUT);

  pinMode(IND1, OUTPUT);
  pinMode(IND2, OUTPUT);
  pinMode(PWMD, OUTPUT);
  
  pinMode(pump, OUTPUT);

  digitalWrite(pump, LOW);


  pinMode(seed1, OUTPUT);
  pinMode(seed2, OUTPUT);
  digitalWrite(seed1, HIGH);
  digitalWrite(seed2, HIGH);


  servo1.attach(servoPin1);
  servo2.attach(servoPin2);

  Serial.begin(9600);
  Serial.println("Ready: f/b/l/r/s");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    switch (cmd) {
      case 'f':
        forward();
        break;
      case 'b':
        backward();
        break;
      case 'l':
        turnLeft();
        break;
      case 'r':
        turnRight();
        break;
      case 's':
        stopMotors();
        break;
      case 'p':
        digitalWrite(pump, HIGH);
        delay(1000);
        digitalWrite(pump, LOW);
        break;
      case 'm':
        digitalWrite(seed1, LOW);
        digitalWrite(seed2, HIGH);
        break;
      case 'n':
        digitalWrite(seed1, HIGH);
        digitalWrite(seed2, LOW);
        break;
      case 'v':
        digitalWrite(seed1, LOW);
        digitalWrite(seed2, LOW);
        break;

    }
  }
}

// ================= FUNCTIONS =================

void forward() {
  servo1.write(85);
  servo2.write(85);

  digitalWrite(INA1, HIGH); digitalWrite(INA2, LOW);
  digitalWrite(INB1, HIGH); digitalWrite(INB2, LOW);
  digitalWrite(INC1, HIGH); digitalWrite(INC2, LOW);
  digitalWrite(IND1, HIGH); digitalWrite(IND2, LOW);

  setSpeed(speedVal);
}

void backward() {
  servo1.write(85);
  servo2.write(85);

  digitalWrite(INA1, LOW);  digitalWrite(INA2, HIGH);
  digitalWrite(INB1, LOW);  digitalWrite(INB2, HIGH);
  digitalWrite(INC1, LOW);  digitalWrite(INC2, HIGH);
  digitalWrite(IND1, LOW);  digitalWrite(IND2, HIGH);

  setSpeed(speedVal);
}

void turnLeft() {
  servo1.write(60);
  servo2.write(60);

  // Left motors backward, right motors forward
  digitalWrite(INA1, HIGH); digitalWrite(INA2, LOW);
  digitalWrite(INB1, HIGH); digitalWrite(INB2, LOW);
  digitalWrite(INC1, HIGH); digitalWrite(INC2, LOW);
  digitalWrite(IND1, HIGH); digitalWrite(IND2, LOW);

  setSpeed(speedVal);
}

void turnRight() {
  servo1.write(110);
  servo2.write(110);

  // Opposite of left
  digitalWrite(INA1, HIGH); digitalWrite(INA2, LOW);
  digitalWrite(INB1, HIGH); digitalWrite(INB2, LOW);
  digitalWrite(INC1, HIGH); digitalWrite(INC2, LOW);
  digitalWrite(IND1, HIGH); digitalWrite(IND2, LOW);

  setSpeed(speedVal);
}

void stopMotors() {
  digitalWrite(INA1, LOW); digitalWrite(INA2, LOW);
  digitalWrite(INB1, LOW); digitalWrite(INB2, LOW);
  digitalWrite(INC1, LOW); digitalWrite(INC2, LOW);
  digitalWrite(IND1, LOW); digitalWrite(IND2, LOW);

  setSpeed(0);
}

void setSpeed(int spd) {
  analogWrite(PWMA, spd);
  analogWrite(PWMB, spd);
  analogWrite(PWMC, spd);
  analogWrite(PWMD, spd);
}