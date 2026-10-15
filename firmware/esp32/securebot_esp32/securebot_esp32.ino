#include <Arduino.h>

// ==================== MOTOR PINS ====================
// Left motor
const int ENA = 25;
const int IN1 = 26;
const int IN2 = 27;

// Right motor
const int ENB = 14;
const int IN3 = 12;
const int IN4 = 13;

// Encoder pins - not used yet
const int RIGHT_ENCODER_BLUE = 33;
const int RIGHT_ENCODER_WHITE = 32;
const int LEFT_ENCODER_BLUE = 35;
const int LEFT_ENCODER_WHITE = 34;

// ==================== ULTRASONIC SENSOR PINS ====================
#define UPPER_TRIG 5
#define UPPER_ECHO 18
#define LOWER_TRIG 17
#define LOWER_ECHO 16

// ==================== SETTINGS ====================
int speedPWM = 95;

const float STOP_DISTANCE_FEET = 1.0;    // 12 inches
const float CLEAR_DISTANCE_FEET = 1.5;   // 18 inches

String mode = "stop";

unsigned long avoidStartTime = 0;
int avoidState = 0;

unsigned long lastDistancePrint = 0;

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(UPPER_TRIG, OUTPUT);
  pinMode(UPPER_ECHO, INPUT);

  pinMode(LOWER_TRIG, OUTPUT);
  pinMode(LOWER_ECHO, INPUT);

  stopMotors();

  Serial.println("ESP32 SecureBot controller ready");
}

// ==================== LOOP ====================
void loop() {
  readJetsonCommand();

  float upperFeet = readUltrasonicFeet(UPPER_TRIG, UPPER_ECHO);
  float lowerFeet = readUltrasonicFeet(LOWER_TRIG, LOWER_ECHO);

  printDistances(upperFeet, lowerFeet);

  bool obstacleDetected =
    (validDistance(upperFeet) && upperFeet < STOP_DISTANCE_FEET) ||
    (validDistance(lowerFeet) && lowerFeet < STOP_DISTANCE_FEET);

  if (mode == "stop") {
    stopMotors();
    avoidState = 0;
    return;
  }

  if (mode == "forward" || mode == "patrol") {
    if (obstacleDetected || avoidState != 0) {
      avoidObstacle();
    } else {
      forward();
    }
  }
  else if (mode == "left") {
    left();
  }
  else if (mode == "right") {
    right();
  }
  else if (mode == "backward") {
    backward();
  }

  delay(30);
}

// ==================== SERIAL COMMANDS ====================
void readJetsonCommand() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "forward") {
      mode = "forward";
      Serial.println("Mode: forward");
    }
    else if (cmd == "patrol") {
      mode = "patrol";
      Serial.println("Mode: patrol");
    }
    else if (cmd == "backward") {
      mode = "backward";
      Serial.println("Mode: backward");
    }
    else if (cmd == "left") {
      mode = "left";
      Serial.println("Mode: left");
    }
    else if (cmd == "right") {
      mode = "right";
      Serial.println("Mode: right");
    }
    else if (cmd == "stop") {
      mode = "stop";
      avoidState = 0;
      stopMotors();
      Serial.println("Mode: stop");
    }
    else if (cmd.startsWith("speed ")) {
      int newSpeed = cmd.substring(6).toInt();
      speedPWM = constrain(newSpeed, 0, 255);

      Serial.print("Speed set to ");
      Serial.println(speedPWM);
    }
  }
}

// ==================== OBSTACLE AVOIDANCE ====================
void avoidObstacle() {
  unsigned long now = millis();

  float upperFeet = readUltrasonicFeet(UPPER_TRIG, UPPER_ECHO);
  float lowerFeet = readUltrasonicFeet(LOWER_TRIG, LOWER_ECHO);

  bool pathClear =
    (!validDistance(upperFeet) || upperFeet > CLEAR_DISTANCE_FEET) &&
    (!validDistance(lowerFeet) || lowerFeet > CLEAR_DISTANCE_FEET);

  if (avoidState == 0) {
    avoidState = 1;
    avoidStartTime = now;
    stopMotors();
    Serial.println("Obstacle detected: stopping");
  }

  else if (avoidState == 1) {
    backward();

    if (now - avoidStartTime > 800) {
      avoidState = 2;
      avoidStartTime = now;
      stopMotors();
      Serial.println("Backing up complete, rotating until clear");
    }
  }

  else if (avoidState == 2) {
    right();

    if (pathClear) {
      avoidState = 3;
      avoidStartTime = now;
      stopMotors();
      Serial.println("Clear path found");
    }
  }

  else if (avoidState == 3) {
    stopMotors();

    if (now - avoidStartTime > 300) {
      avoidState = 0;
      Serial.println("Resuming forward movement");
    }
  }
}

// ==================== ULTRASONIC ====================
float readUltrasonicFeet(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 25000);

  if (duration == 0) {
    return -1;
  }

  float distanceInches = duration * 0.0135 / 2.0;
  float distanceFeet = distanceInches / 12.0;

  return distanceFeet;
}

bool validDistance(float distanceFeet) {
  return distanceFeet > 0 && distanceFeet < 16.0;
}

void printDistances(float upperFeet, float lowerFeet) {
  if (millis() - lastDistancePrint >= 500) {
    Serial.print("Upper sensor: ");

    if (validDistance(upperFeet)) {
      Serial.print(upperFeet, 2);
      Serial.print(" ft");
    } else {
      Serial.print("No reading");
    }

    Serial.print(" | Lower sensor: ");

    if (validDistance(lowerFeet)) {
      Serial.print(lowerFeet, 2);
      Serial.println(" ft");
    } else {
      Serial.println("No reading");
    }

    lastDistancePrint = millis();
  }
}

// ==================== MOTOR CONTROL ====================
void setMotorSpeed() {
  analogWrite(ENA, speedPWM);
  analogWrite(ENB, speedPWM);
}

void forward() {
  setMotorSpeed();

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void backward() {
  setMotorSpeed();

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void left() {
  setMotorSpeed();

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void right() {
  setMotorSpeed();

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}