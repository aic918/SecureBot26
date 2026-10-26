// SecureBot ESP32 low-level controller, firmware 2.0
//
// Serial protocol, 115200 baud, newline-terminated.
//
// Jetson -> ESP32
//   M,<left>,<right>   signed wheel PWM, 10-bit duty (0-1023), clamped to +/-MAX_PWM
//   stop | forward | patrol | backward | left | right   (legacy text commands)
//   speed <0-MAX_PWM>  PWM used by the legacy commands
//   freq <Hz>          motor PWM frequency, MIN_PWM_FREQ_HZ..MAX_PWM_FREQ_HZ, until reboot
//                      (for tuning; not a heartbeat)
//
// ESP32 -> Jetson
//   U,<left_ft>,<right_ft>   every TELEMETRY_MS; -1 means "no reading"
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

// ==================== PINOUT ====================
// Motor driver (two channels). The RIGHT motor is on channel A and the LEFT motor
// on channel B. "A"/"B" below is the driver's per-motor direction input pair.
// Right motor (driver channel A)
const int RIGHT_EN = 26;
const int RIGHT_IN_A = 25;    // IN1
const int RIGHT_IN_B = 33;    // IN2

// Left motor (driver channel B)
const int LEFT_EN = 27;
const int LEFT_IN_A = 32;     // IN3
const int LEFT_IN_B = 14;     // IN4

// Wiring polarity, as verified with the wheels off the ground using the senior1typetest
// sketch (RIGHT_FORWARD_IN1_HIGH = true, LEFT_FORWARD_IN3_HIGH = false): forward is IN1 high
// for the right motor and IN3 low for the left one, so only the left motor is reversed.
// If a wheel ever turns the wrong way for a positive command, flip its flag here.
const bool LEFT_REVERSED = true;
const bool RIGHT_REVERSED = false;

// Encoder pins - not used yet. GPIO 34/35 are input-only with no internal pull-ups.
// Never use a boot-strapping pin here (0, 2, 5, 12, 15): an encoder holding GPIO 12 high at
// reset made the flash unreadable and the board would not boot or accept uploads.
const int RIGHT_ENCODER_BLUE = 13;
const int RIGHT_ENCODER_WHITE = 18;
const int LEFT_ENCODER_BLUE = 34;
const int LEFT_ENCODER_WHITE = 35;

// Ultrasonic 1 = left, ultrasonic 2 = right (this is also the telemetry order).
#define ULTRA_LEFT_TRIG 23
#define ULTRA_LEFT_ECHO 19
#define ULTRA_RIGHT_TRIG 4
#define ULTRA_RIGHT_ECHO 17

// ==================== SETTINGS ====================
// Motor PWM: 10-bit duty (0-1023) at 10 kHz. Chosen with tools/motor_tune.py on the floor
// (2026-10-26): the robot needs ~650 to start from standstill (at 500 it needed a push).
// `freq <Hz>` still changes the frequency at run time for further tuning.
const int PWM_FREQUENCY_HZ = 10000;
const int MIN_PWM_FREQ_HZ = 50;
const int MAX_PWM_FREQ_HZ = 25000;
const int PWM_BITS = 10;
const int LEFT_PWM_CHANNEL = 1;       // LEDC channels (Arduino-ESP32 2.x API)
const int RIGHT_PWM_CHANNEL = 0;
const int MAX_PWM = 655;              // hard cap on any wheel PWM, every command path
const int DEFAULT_SPEED_PWM = 650;    // legacy-command speed until "speed N" is received

const unsigned long WATCHDOG_MS = 250;           // stop if no valid command this long
                                                 // (the ROS bridge resends every 100 ms)
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
int pwmFrequencyHz = PWM_FREQUENCY_HZ;
int driveLeft = 0;                 // requested by M,l,r
int driveRight = 0;
int appliedLeft = 0;               // what is actually on the motors
int appliedRight = 0;
unsigned long lastCommandMs = 0;

int avoidState = 0;                // 0 idle, 1 reverse, 2 rotate, 3 settle
unsigned long avoidStartMs = 0;

float leftUltraFeet = -1;
float rightUltraFeet = -1;
int leftUltraCloseReads = 0;
int rightUltraCloseReads = 0;
bool sampleLeftNext = true;
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
void setMotor(int channel, int inA, int inB, int pwm);
void sendTelemetry();
float readUltrasonicFeet(int trigPin, int echoPin);
bool validDistance(float distanceFeet);
bool obstacleAhead();
bool pathClear();

// ==================== SETUP ====================
void setup() {
  // Motors first, so nothing can spin while the rest of setup runs.
  pinMode(LEFT_EN, OUTPUT);
  pinMode(LEFT_IN_A, OUTPUT);
  pinMode(LEFT_IN_B, OUTPUT);
  pinMode(RIGHT_EN, OUTPUT);
  pinMode(RIGHT_IN_A, OUTPUT);
  pinMode(RIGHT_IN_B, OUTPUT);
  ledcSetup(LEFT_PWM_CHANNEL, PWM_FREQUENCY_HZ, PWM_BITS);
  ledcSetup(RIGHT_PWM_CHANNEL, PWM_FREQUENCY_HZ, PWM_BITS);
  ledcAttachPin(LEFT_EN, LEFT_PWM_CHANNEL);
  ledcAttachPin(RIGHT_EN, RIGHT_PWM_CHANNEL);
  applyMotors(0, 0);

  pinMode(ULTRA_LEFT_TRIG, OUTPUT);
  pinMode(ULTRA_LEFT_ECHO, INPUT);
  pinMode(ULTRA_RIGHT_TRIG, OUTPUT);
  pinMode(ULTRA_RIGHT_ECHO, INPUT);

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
  bool heartbeat = true;          // configuration commands are valid but do not feed the watchdog

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
  } else if (strncmp(cmd, "freq ", 5) == 0) {
    int hz = constrain(atoi(cmd + 5), MIN_PWM_FREQ_HZ, MAX_PWM_FREQ_HZ);
    if (ledcChangeFrequency(LEFT_PWM_CHANNEL, hz, PWM_BITS) != 0 &&
        ledcChangeFrequency(RIGHT_PWM_CHANNEL, hz, PWM_BITS) != 0) {
      pwmFrequencyHz = hz;
      Serial.print("I,freq ");
      Serial.println(pwmFrequencyHz);
    } else {
      Serial.println("E,freq not supported");
    }
    heartbeat = false;
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
    if (heartbeat) {
      lastCommandMs = millis();   // any valid motion/speed command counts as a heartbeat
    }
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

  if (sampleLeftNext) {
    leftUltraFeet = readUltrasonicFeet(ULTRA_LEFT_TRIG, ULTRA_LEFT_ECHO);
    bool close = validDistance(leftUltraFeet) && leftUltraFeet < STOP_DISTANCE_FEET;
    leftUltraCloseReads = close ? min(leftUltraCloseReads + 1, 100) : 0;
  } else {
    rightUltraFeet = readUltrasonicFeet(ULTRA_RIGHT_TRIG, ULTRA_RIGHT_ECHO);
    bool close = validDistance(rightUltraFeet) && rightUltraFeet < STOP_DISTANCE_FEET;
    rightUltraCloseReads = close ? min(rightUltraCloseReads + 1, 100) : 0;
  }
  sampleLeftNext = !sampleLeftNext;
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
  return leftUltraCloseReads >= OBSTACLE_CONFIRM_READS || rightUltraCloseReads >= OBSTACLE_CONFIRM_READS;
}

bool pathClear() {
  return (!validDistance(leftUltraFeet) || leftUltraFeet > CLEAR_DISTANCE_FEET) &&
         (!validDistance(rightUltraFeet) || rightUltraFeet > CLEAR_DISTANCE_FEET);
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

  setMotor(LEFT_PWM_CHANNEL, LEFT_IN_A, LEFT_IN_B, LEFT_REVERSED ? -left : left);
  setMotor(RIGHT_PWM_CHANNEL, RIGHT_IN_A, RIGHT_IN_B, RIGHT_REVERSED ? -right : right);

  appliedLeft = left;
  appliedRight = right;
}

void setMotor(int channel, int inA, int inB, int pwm) {
  if (pwm > 0) {
    digitalWrite(inA, HIGH);
    digitalWrite(inB, LOW);
    ledcWrite(channel, pwm);
  } else if (pwm < 0) {
    digitalWrite(inA, LOW);
    digitalWrite(inB, HIGH);
    ledcWrite(channel, -pwm);
  } else {
    digitalWrite(inA, LOW);
    digitalWrite(inB, LOW);
    ledcWrite(channel, 0);
  }
}

// ==================== TELEMETRY ====================
void sendTelemetry() {
  unsigned long now = millis();

  if (now - lastTelemetryMs >= TELEMETRY_MS) {
    lastTelemetryMs = now;
    Serial.print("U,");
    Serial.print(validDistance(leftUltraFeet) ? leftUltraFeet : -1.0f, 2);
    Serial.print(",");
    Serial.println(validDistance(rightUltraFeet) ? rightUltraFeet : -1.0f, 2);
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
