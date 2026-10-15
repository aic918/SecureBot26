// SecureBot ESP32 low-level controller, firmware 2.0
//
// Serial protocol, 115200 baud, newline-terminated.
//
// Jetson -> ESP32
//   M,<left>,<right>   signed wheel PWM, clamped to +/-MAX_PWM (used by the ROS bridge)
//   stop | forward | patrol | backward | left | right   (legacy text commands)
//   speed <0-MAX_PWM>  PWM used by the legacy commands
//
// ESP32 -> Jetson
//   U,<upper_ft>,<lower_ft>   every TELEMETRY_MS; -1 means "no reading"
//   S,<mode>,<leftPWM>,<rightPWM>,<avoidState>   every STATUS_MS
//   I,<text>   information     E,<text>   error / safety event
//
// Safety:
//   * Every motor output goes through applyMotors(), which enforces MAX_PWM.
//   * Watchdog: if no valid command arrives for WATCHDOG_MS the robot stops, so
//     the Jetson must resend its command at least every ~200 ms.
//   * Net-forward motion from M,l,r is blocked while an ultrasonic sensor sees an
//     obstacle inside STOP_DISTANCE_FEET (rotating in place and reversing still work).
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
const int MAX_PWM = 150;              // hard cap on any wheel PWM, every command path
const int DEFAULT_SPEED_PWM = 95;     // legacy-command speed until "speed N" is received

const unsigned long WATCHDOG_MS = 500;           // stop if no valid command this long
const unsigned long SENSOR_PERIOD_MS = 40;       // one sensor per tick, alternating
const unsigned long TELEMETRY_MS = 100;
const unsigned long STATUS_MS = 500;

const float STOP_DISTANCE_FEET = 1.0;    // 12 inches
const float CLEAR_DISTANCE_FEET = 1.5;   // 18 inches
const int OBSTACLE_CONFIRM_READS = 2;    // consecutive close readings before reacting

const unsigned long AVOID_REVERSE_MS = 800;
const unsigned long AVOID_ROTATE_TIMEOUT_MS = 4000;  // give up if never clear
const unsigned long AVOID_SETTLE_MS = 300;

enum Mode { MODE_STOP, MODE_FORWARD, MODE_PATROL, MODE_BACKWARD, MODE_LEFT, MODE_RIGHT, MODE_DRIVE };
const char *MODE_NAMES[] = {"stop", "forward", "patrol", "backward", "left", "right", "drive"};

// ==================== STATE ====================
Mode mode = MODE_STOP;
int speedPWM = DEFAULT_SPEED_PWM;
int driveLeft = 0;                 // requested by M,l,r
int driveRight = 0;
int appliedLeft = 0;               // what is actually on the motors
int appliedRight = 0;
unsigned long lastCommandMs = 0;

int avoidState = 0;                // 0 idle, 1 reverse, 2 rotate, 3 settle
unsigned long avoidStartMs = 0;

float upperFeet = -1;
float lowerFeet = -1;
int upperCloseReads = 0;
int lowerCloseReads = 0;
bool sampleUpperNext = true;
unsigned long lastSensorMs = 0;

unsigned long lastTelemetryMs = 0;
unsigned long lastStatusMs = 0;

char lineBuf[48];
size_t lineLen = 0;
bool lineOverflow = false;

// ==================== PROTOTYPES ====================
void readSerial();
void handleCommand(char *cmd);
void setMode(Mode newMode);
void checkWatchdog();
void updateSensors();
void controlMotors();
void runAvoidance(int &left, int &right);
void applyMotors(int left, int right);
void setMotor(int en, int inA, int inB, int pwm);
void sendTelemetry();
float readUltrasonicFeet(int trigPin, int echoPin);
bool validDistance(float distanceFeet);
bool obstacleAhead();
bool pathClear();

// ==================== SETUP ====================
void setup() {
  // Motors first, so nothing can spin while the rest of setup runs.
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  applyMotors(0, 0);

  pinMode(UPPER_TRIG, OUTPUT);
  pinMode(UPPER_ECHO, INPUT);
  pinMode(LOWER_TRIG, OUTPUT);
  pinMode(LOWER_ECHO, INPUT);

  Serial.begin(115200);
  Serial.println("I,SecureBot ESP32 fw 2.0 ready");
}

// ==================== LOOP ====================
void loop() {
  readSerial();
  checkWatchdog();
  updateSensors();
  controlMotors();
  sendTelemetry();
  delay(1);
}

// ==================== SERIAL COMMANDS ====================
// Non-blocking: collects bytes into lineBuf and runs a command per full line.
void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      if (lineOverflow) {
        Serial.println("E,line too long");
      } else if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleCommand(lineBuf);
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (lineOverflow) {
      // drop the rest of an oversized line
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineOverflow = true;
    }
  }
}

void handleCommand(char *cmd) {
  while (*cmd == ' ') {
    cmd++;
  }
  size_t len = strlen(cmd);
  while (len > 0 && cmd[len - 1] == ' ') {
    cmd[--len] = '\0';
  }

  bool valid = true;

  if (strcmp(cmd, "stop") == 0) {
    setMode(MODE_STOP);
  } else if (strcmp(cmd, "forward") == 0) {
    setMode(MODE_FORWARD);
  } else if (strcmp(cmd, "patrol") == 0) {
    setMode(MODE_PATROL);
  } else if (strcmp(cmd, "backward") == 0) {
    setMode(MODE_BACKWARD);
  } else if (strcmp(cmd, "left") == 0) {
    setMode(MODE_LEFT);
  } else if (strcmp(cmd, "right") == 0) {
    setMode(MODE_RIGHT);
  } else if (strncmp(cmd, "speed ", 6) == 0) {
    speedPWM = constrain(atoi(cmd + 6), 0, MAX_PWM);
    Serial.print("I,speed ");
    Serial.println(speedPWM);
  } else if (strncmp(cmd, "M,", 2) == 0) {
    int l = 0;
    int r = 0;
    if (sscanf(cmd, "M,%d,%d", &l, &r) == 2) {
      driveLeft = constrain(l, -MAX_PWM, MAX_PWM);
      driveRight = constrain(r, -MAX_PWM, MAX_PWM);
      setMode(MODE_DRIVE);
    } else {
      valid = false;
    }
  } else {
    valid = false;
  }

  if (valid) {
    lastCommandMs = millis();   // any valid command counts as a heartbeat
  } else {
    Serial.print("E,bad command: ");
    Serial.println(cmd);
  }
}

void setMode(Mode newMode) {
  if (newMode != mode || newMode == MODE_STOP) {
    avoidState = 0;
  }
  mode = newMode;
}

// ==================== WATCHDOG ====================
void checkWatchdog() {
  if (mode != MODE_STOP && millis() - lastCommandMs > WATCHDOG_MS) {
    setMode(MODE_STOP);
    Serial.println("E,watchdog: no command, stopped");
  }
}

// ==================== SENSORS ====================
// Reads one sensor per SENSOR_PERIOD_MS, alternating, so a loop pass blocks for
// at most one pulseIn and the two sensors never fire at the same time.
void updateSensors() {
  unsigned long now = millis();
  if (now - lastSensorMs < SENSOR_PERIOD_MS) {
    return;
  }
  lastSensorMs = now;

  if (sampleUpperNext) {
    upperFeet = readUltrasonicFeet(UPPER_TRIG, UPPER_ECHO);
    bool close = validDistance(upperFeet) && upperFeet < STOP_DISTANCE_FEET;
    upperCloseReads = close ? min(upperCloseReads + 1, 100) : 0;
  } else {
    lowerFeet = readUltrasonicFeet(LOWER_TRIG, LOWER_ECHO);
    bool close = validDistance(lowerFeet) && lowerFeet < STOP_DISTANCE_FEET;
    lowerCloseReads = close ? min(lowerCloseReads + 1, 100) : 0;
  }
  sampleUpperNext = !sampleUpperNext;
}

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

bool obstacleAhead() {
  return upperCloseReads >= OBSTACLE_CONFIRM_READS || lowerCloseReads >= OBSTACLE_CONFIRM_READS;
}

bool pathClear() {
  return (!validDistance(upperFeet) || upperFeet > CLEAR_DISTANCE_FEET) &&
         (!validDistance(lowerFeet) || lowerFeet > CLEAR_DISTANCE_FEET);
}

// ==================== MOTION ====================
void controlMotors() {
  int left = 0;
  int right = 0;

  switch (mode) {
    case MODE_STOP:
      break;

    case MODE_FORWARD:
    case MODE_PATROL:
      if (obstacleAhead() || avoidState != 0) {
        runAvoidance(left, right);
      } else {
        left = speedPWM;
        right = speedPWM;
      }
      break;

    case MODE_BACKWARD:
      left = -speedPWM;
      right = -speedPWM;
      break;

    case MODE_LEFT:
      left = -speedPWM;
      right = speedPWM;
      break;

    case MODE_RIGHT:
      left = speedPWM;
      right = -speedPWM;
      break;

    case MODE_DRIVE:
      left = driveLeft;
      right = driveRight;
      // Failsafe: never drive net-forward into a confirmed close obstacle.
      if (left + right > 0 && obstacleAhead()) {
        left = 0;
        right = 0;
      }
      break;
  }

  applyMotors(left, right);
}

// Same sequence as the original firmware (stop, back up, rotate until clear,
// settle), plus a timeout on the rotation so it can never spin forever.
void runAvoidance(int &left, int &right) {
  unsigned long now = millis();

  if (avoidState == 0) {
    avoidState = 1;
    avoidStartMs = now;
    Serial.println("I,obstacle detected: stopping");
  } else if (avoidState == 1) {
    left = -speedPWM;
    right = -speedPWM;

    if (now - avoidStartMs > AVOID_REVERSE_MS) {
      avoidState = 2;
      avoidStartMs = now;
      Serial.println("I,backed up, rotating until clear");
    }
  } else if (avoidState == 2) {
    left = speedPWM;
    right = -speedPWM;

    if (pathClear()) {
      avoidState = 3;
      avoidStartMs = now;
      Serial.println("I,clear path found");
    } else if (now - avoidStartMs > AVOID_ROTATE_TIMEOUT_MS) {
      left = 0;
      right = 0;
      setMode(MODE_STOP);
      Serial.println("E,avoid timeout: no clear path, stopped");
    }
  } else if (avoidState == 3) {
    if (now - avoidStartMs > AVOID_SETTLE_MS) {
      avoidState = 0;
      Serial.println("I,resuming forward movement");
    }
  }
}

// ==================== MOTOR CONTROL ====================
// The only place that touches the motor pins. Enforces MAX_PWM.
void applyMotors(int left, int right) {
  left = constrain(left, -MAX_PWM, MAX_PWM);
  right = constrain(right, -MAX_PWM, MAX_PWM);

  setMotor(ENA, IN1, IN2, left);
  setMotor(ENB, IN3, IN4, right);

  appliedLeft = left;
  appliedRight = right;
}

void setMotor(int en, int inA, int inB, int pwm) {
  if (pwm > 0) {
    digitalWrite(inA, HIGH);
    digitalWrite(inB, LOW);
    analogWrite(en, pwm);
  } else if (pwm < 0) {
    digitalWrite(inA, LOW);
    digitalWrite(inB, HIGH);
    analogWrite(en, -pwm);
  } else {
    digitalWrite(inA, LOW);
    digitalWrite(inB, LOW);
    analogWrite(en, 0);
  }
}

// ==================== TELEMETRY ====================
void sendTelemetry() {
  unsigned long now = millis();

  if (now - lastTelemetryMs >= TELEMETRY_MS) {
    lastTelemetryMs = now;
    Serial.print("U,");
    Serial.print(validDistance(upperFeet) ? upperFeet : -1.0f, 2);
    Serial.print(",");
    Serial.println(validDistance(lowerFeet) ? lowerFeet : -1.0f, 2);
  }

  if (now - lastStatusMs >= STATUS_MS) {
    lastStatusMs = now;
    Serial.print("S,");
    Serial.print(MODE_NAMES[mode]);
    Serial.print(",");
    Serial.print(appliedLeft);
    Serial.print(",");
    Serial.print(appliedRight);
    Serial.print(",");
    Serial.println(avoidState);
  }
}
