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
//   G,<yaw_deg>,<yaw_rate_dps>   every IMU_TELEMETRY_MS, if the IMU answered at boot
//   W,<left_ticks>,<right_ticks> every IMU_TELEMETRY_MS: wheel encoder counts since boot
//                                (quadrature, both edges of both channels; + = that wheel forward)
//                                (yaw: gyro Z integrated since boot, CCW positive, -180..180)
//   I,<text>   information     E,<text>   error / safety event
//
// Safety:
//   * Every motor output goes through applyMotors(), which enforces MAX_PWM.
//   * Watchdog: if no valid command arrives for WATCHDOG_MS the robot stops, so
//     the Jetson must resend its command at least every ~200 ms.
//   * Net-forward motion from M,l,r is blocked while an ultrasonic sensor sees an
//     obstacle inside STOP_DISTANCE_FEET (rotating in place and reversing still work).
#include <Arduino.h>
#include <Wire.h>

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

// Encoders: the two channels of each wheel are decoded together (x4 quadrature), so which
// wire is A and which is B only decides the sign, set by *_ENCODER_REVERSED.
const bool LEFT_ENCODER_REVERSED = false;
const bool RIGHT_ENCODER_REVERSED = false;

// IMU: MPU-6050 board (HW-123 / GY-521) on the ESP32's default I2C pins, AD0 low
const int IMU_SDA = 21;
const int IMU_SCL = 22;

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
const unsigned long BRAKE_MS = 150;   // short brake after a motor stops (see outputMotor)

const unsigned long WATCHDOG_MS = 250;           // stop if no valid command this long
                                                 // (the ROS bridge resends every 100 ms)
const unsigned long SENSOR_PERIOD_MS = 40;       // one sensor per tick, alternating
const unsigned long TELEMETRY_MS = 100;
const unsigned long STATUS_MS = 500;

const uint8_t IMU_ADDRS[] = {0x68, 0x69};        // AD0 low / high
const float IMU_LSB_PER_DPS = 65.5;              // gyro range +/-500 deg/s
const unsigned long IMU_PERIOD_US = 5000;        // sample the gyro at 200 Hz
const unsigned long IMU_TELEMETRY_MS = 50;       // G line at 20 Hz
const int IMU_CALIBRATE_SAMPLES = 400;           // ~1.2 s at boot, motors stopped
const int IMU_MAX_FAILS = 20;                    // consecutive bad reads -> IMU off
const unsigned long IMU_STILL_MS = 1000;         // motors off this long -> refine the bias
const float IMU_STILL_DPS = 1.0;                 // ... while the gyro reads below this
const float IMU_BIAS_RATE = 0.002;               // bias update per sample when still

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

bool imuOk = false;
uint8_t imuAddr = 0x68;
int imuFails = 0;
float imuBiasDps = 0;
float yawDeg = 0;
float yawRateDps = 0;
unsigned long lastImuUs = 0;
unsigned long lastImuTelemetryMs = 0;
unsigned long motorsStillSinceMs = 0;

volatile long leftTicks = 0;
volatile long rightTicks = 0;
volatile uint8_t leftEncState = 0;
volatile uint8_t rightEncState = 0;
// x4 quadrature: index = previous state (2 bits) << 2 | new state; value = step
const int8_t QUAD_STEP[16] = {0, 1, -1, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, -1, 1, 0};

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
void outputMotor(int m, int channel, int inA, int inB, int pwm);
void setMotor(int channel, int inA, int inB, int pwm);
void sendTelemetry();
void imuWireCheck(int &sda, int &scl);
void setupImu();
void setupEncoders();
void encoderScope();
void printI2cScan();
void updateImu();
bool imuWrite(uint8_t reg, uint8_t value);
bool imuReadGyroZ(int16_t &raw);
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
  Serial.println("I,booting");     // before the IMU: a stuck I2C bus shows up as silence after it
  setupImu();                      // motors are already stopped: safe to calibrate
  setupEncoders();
  Serial.println("I,SecureBot ESP32 fw 2.1 ready");
}

// ==================== LOOP ====================
void loop() {
  readSerial();
  checkWatchdog();
  updateSensors();
  updateImu();
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
  } else if (strcmp(cmd, "encscope") == 0) {
    encoderScope();                           // diagnostic: stops the motors, takes ~3 s
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

// ==================== IMU (MPU-6050 gyro Z) ====================
bool imuWrite(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(imuAddr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool imuReadGyroZ(int16_t &raw) {
  Wire.beginTransmission(imuAddr);
  Wire.write(0x47);                          // GYRO_ZOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)imuAddr, 2) != 2) return false;
  int hi = Wire.read();
  int lo = Wire.read();
  raw = (int16_t)((hi << 8) | lo);
  return true;
}

// Wake the MPU-6050, set +/-500 deg/s and a 44 Hz filter, then average the gyro while the
// robot stands still to get its bias. Leaves imuOk false (and says so) if it does not answer.
// Print every I2C address that answers on the current pins, in decimal (104 = 0x68).
void printI2cScan() {
  bool any = false;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.print(" ");
      Serial.print((int)a);
      any = true;
    }
  }
  if (!any) Serial.print(" none");
}

// The module's pull-up resistors hold SDA and SCL high; against the ESP32's weak pull-down a
// connected wire reads 1, a loose or missing one (or no shared GND) reads 0.
void imuWireCheck(int &sda, int &scl) {
  pinMode(IMU_SDA, INPUT_PULLDOWN);
  pinMode(IMU_SCL, INPUT_PULLDOWN);
  delay(5);
  sda = digitalRead(IMU_SDA);
  scl = digitalRead(IMU_SCL);
}

void setupImu() {
  // Check the wires before starting I2C: with a wire missing the bus has no pull-up and the
  // ESP32's I2C driver can hang for good, so the board would never finish booting.
  int sda, scl;
  imuWireCheck(sda, scl);
  if (!sda || !scl) {
    Serial.print("E,imu not found (MPU-6050 on SDA 21 / SCL 22); wire check (1 = connected): SDA=");
    Serial.print(sda);
    Serial.print(" SCL=");
    Serial.println(scl);
    return;
  }
  Wire.begin(IMU_SDA, IMU_SCL);
  Wire.setClock(100000);                     // standard speed: robust on longer wires
  Wire.setTimeOut(10);                       // ms: a stuck bus must not stall the boot
  // After a power-on the MPU-6050 needs up to ~100 ms before it answers: keep trying ~1 s.
  bool found = false;
  for (int attempt = 0; attempt < 20 && !found; attempt++) {
    for (uint8_t a : IMU_ADDRS) {
      imuAddr = a;
      if (imuWrite(0x6B, 0x01)) {            // PWR_MGMT_1: awake, clock from the X gyro
        found = true;
        break;
      }
    }
    if (!found) delay(50);
  }
  if (!found || !imuWrite(0x1A, 0x03) ||     // CONFIG: digital low-pass ~44 Hz
      !imuWrite(0x1B, 0x08)) {               // GYRO_CONFIG: +/-500 deg/s
    imuWireCheck(sda, scl);
    Serial.print("E,imu not found (MPU-6050 on SDA 21 / SCL 22); wire check (1 = connected): SDA=");
    Serial.print(sda);
    Serial.print(" SCL=");
    Serial.println(scl);
    Wire.begin(IMU_SDA, IMU_SCL);
    if (sda && scl) {                        // both wires there: list what answers (slow)
      Serial.print("E,imu i2c devices (decimal):");
      printI2cScan();
      Serial.println();
    }
    return;
  }
  delay(100);                                // gyro start-up
  long sum = 0;
  int good = 0;
  for (int i = 0; i < IMU_CALIBRATE_SAMPLES; i++) {
    int16_t raw;
    if (imuReadGyroZ(raw)) {
      sum += raw;
      good++;
    }
    delay(3);
  }
  if (good < IMU_CALIBRATE_SAMPLES / 2) {
    Serial.println("E,imu not answering, gyro off");
    return;
  }
  imuBiasDps = (float)sum / good / IMU_LSB_PER_DPS;
  imuOk = true;
  imuFails = 0;
  yawDeg = 0;
  lastImuUs = micros();
  Serial.print("I,imu ready, gyro bias ");
  Serial.print(imuBiasDps, 3);
  Serial.println(" deg/s");
}

// Integrate gyro Z into yawDeg. While the motors have been off for a while and the gyro
// reads almost nothing, nudge the bias towards the reading so slow drift is cancelled.
void updateImu() {
  if (!imuOk) return;
  unsigned long nowUs = micros();
  if (nowUs - lastImuUs < IMU_PERIOD_US) return;
  float dt = (nowUs - lastImuUs) * 1e-6f;
  lastImuUs = nowUs;
  int16_t raw;
  if (!imuReadGyroZ(raw)) {
    if (++imuFails >= IMU_MAX_FAILS) {
      imuOk = false;
      Serial.println("E,imu stopped answering, gyro off");
    }
    return;
  }
  imuFails = 0;
  yawRateDps = raw / IMU_LSB_PER_DPS - imuBiasDps;

  unsigned long nowMs = millis();
  if (appliedLeft != 0 || appliedRight != 0) {
    motorsStillSinceMs = nowMs;
  } else if (nowMs - motorsStillSinceMs >= IMU_STILL_MS && fabs(yawRateDps) < IMU_STILL_DPS) {
    imuBiasDps += IMU_BIAS_RATE * yawRateDps;
    yawRateDps *= 1.0f - IMU_BIAS_RATE;
  }
  yawDeg += yawRateDps * dt;
  if (yawDeg > 180.0f) yawDeg -= 360.0f;
  if (yawDeg < -180.0f) yawDeg += 360.0f;
}

// ==================== WHEEL ENCODERS ====================
void IRAM_ATTR leftEncoderIsr() {
  uint8_t now = (digitalRead(LEFT_ENCODER_BLUE) << 1) | digitalRead(LEFT_ENCODER_WHITE);
  leftTicks += QUAD_STEP[(leftEncState << 2) | now];
  leftEncState = now;
}

void IRAM_ATTR rightEncoderIsr() {
  uint8_t now = (digitalRead(RIGHT_ENCODER_BLUE) << 1) | digitalRead(RIGHT_ENCODER_WHITE);
  rightTicks += QUAD_STEP[(rightEncState << 2) | now];
  rightEncState = now;
}

// GPIO 13 and 18 have internal pull-ups; 34 and 35 are input-only without them, so the left
// encoder relies on external pull-ups. The boot line shows each wire's idle level.
void setupEncoders() {
  pinMode(RIGHT_ENCODER_BLUE, INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_WHITE, INPUT_PULLUP);
  pinMode(LEFT_ENCODER_BLUE, INPUT);
  pinMode(LEFT_ENCODER_WHITE, INPUT);
  leftEncState = (digitalRead(LEFT_ENCODER_BLUE) << 1) | digitalRead(LEFT_ENCODER_WHITE);
  rightEncState = (digitalRead(RIGHT_ENCODER_BLUE) << 1) | digitalRead(RIGHT_ENCODER_WHITE);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_BLUE), leftEncoderIsr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_WHITE), leftEncoderIsr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_BLUE), rightEncoderIsr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_WHITE), rightEncoderIsr, CHANGE);
  Serial.print("I,encoders idle levels L34/L35/R13/R18: ");
  Serial.print(digitalRead(LEFT_ENCODER_BLUE));
  Serial.print(digitalRead(LEFT_ENCODER_WHITE));
  Serial.print(digitalRead(RIGHT_ENCODER_BLUE));
  Serial.println(digitalRead(RIGHT_ENCODER_WHITE));
}

// Diagnostic "oscilloscope" for the encoder lines (command "encscope"): with the motors
// stopped, sample the voltage of the encoder pins that have an ADC (GPIO 34, 35, 13) for
// ENC_SCOPE_MS and report min / average / max in mV and the share of samples below the
// ESP32's logic-low level. Turn a wheel by hand meanwhile. A working line swings to ~0 V.
const unsigned long ENC_SCOPE_MS = 3000;
const int ENC_LOW_MV = 800;

void encoderScope() {
  setMode(MODE_STOP);
  applyMotors(0, 0);
  const int pins[3] = {LEFT_ENCODER_BLUE, LEFT_ENCODER_WHITE, RIGHT_ENCODER_BLUE};
  long sum[3] = {0, 0, 0};
  int lo[3] = {9999, 9999, 9999}, hi[3] = {0, 0, 0};
  long low[3] = {0, 0, 0}, n = 0;
  unsigned long start = millis();
  while (millis() - start < ENC_SCOPE_MS) {
    for (int i = 0; i < 3; i++) {
      int mv = analogReadMilliVolts(pins[i]);
      sum[i] += mv;
      lo[i] = min(lo[i], mv);
      hi[i] = max(hi[i], mv);
      if (mv < ENC_LOW_MV) low[i]++;
    }
    n++;
  }
  for (int i = 0; i < 3; i++) {
    Serial.print("I,encscope GPIO");
    Serial.print(pins[i]);
    Serial.print(" min/avg/max mV ");
    Serial.print(lo[i]);
    Serial.print("/");
    Serial.print(n ? (int)(sum[i] / n) : 0);
    Serial.print("/");
    Serial.print(hi[i]);
    Serial.print(" low ");
    Serial.print(n ? (int)(100 * low[i] / n) : 0);
    Serial.print("% of ");
    Serial.print(n);
    Serial.println(" samples");
  }
  lastCommandMs = millis();
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

  outputMotor(0, LEFT_PWM_CHANNEL, LEFT_IN_A, LEFT_IN_B, LEFT_REVERSED ? -left : left);
  outputMotor(1, RIGHT_PWM_CHANNEL, RIGHT_IN_A, RIGHT_IN_B, RIGHT_REVERSED ? -right : right);

  appliedLeft = left;
  appliedRight = right;
}

// Brake after a stop: a motor that was turning and is told 0 gets both direction inputs high
// with the enable on for BRAKE_MS (the driver shorts the motor: short brake), then is
// released. Without it the motors free-wheel and the robot coasts ~25 deg after a spin.
unsigned long brakeUntilMs[2] = {0, 0};
int lastMotorPwm[2] = {0, 0};

void outputMotor(int m, int channel, int inA, int inB, int pwm) {
  unsigned long now = millis();
  if (pwm == 0 && lastMotorPwm[m] != 0) brakeUntilMs[m] = now + BRAKE_MS;
  if (pwm != 0) brakeUntilMs[m] = 0;
  lastMotorPwm[m] = pwm;
  if (pwm == 0 && brakeUntilMs[m] != 0 && (long)(brakeUntilMs[m] - now) > 0) {
    digitalWrite(inA, HIGH);
    digitalWrite(inB, HIGH);
    ledcWrite(channel, MAX_PWM);
    return;
  }
  brakeUntilMs[m] = 0;
  setMotor(channel, inA, inB, pwm);
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

  if (now - lastImuTelemetryMs >= IMU_TELEMETRY_MS) {
    lastImuTelemetryMs = now;
    if (imuOk) {
      Serial.print("G,");
      Serial.print(yawDeg, 2);
      Serial.print(",");
      Serial.println(yawRateDps, 2);
    }
    noInterrupts();
    long left = leftTicks;
    long right = rightTicks;
    interrupts();
    Serial.print("W,");
    Serial.print(LEFT_ENCODER_REVERSED ? -left : left);
    Serial.print(",");
    Serial.println(RIGHT_ENCODER_REVERSED ? -right : right);
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
