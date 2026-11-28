// SecureBot ESP32 low-level controller, firmware 2.2
//
// Serial protocol, 115200 baud, newline-terminated.
//
// Jetson -> ESP32
//   M,<left>,<right>   signed wheel PWM, 10-bit duty (0-1023), clamped to +/-MAX_PWM
//   V,<left>,<right>   signed wheel SPEEDS in encoder ticks/s (+ = forward): each wheel's PWM
//                      is adjusted every SPEED_PERIOD_MS until its encoder matches
//   stop | forward | patrol | backward | left | right   (legacy text commands)
//   speed <0-MAX_PWM>  PWM used by the legacy commands
//   freq <Hz>          motor PWM frequency, MIN_PWM_FREQ_HZ..MAX_PWM_FREQ_HZ, until reboot
//                      (for tuning; not a heartbeat)
//   tamper on | tamper off   arm / disarm tamper detection (not a heartbeat; off at boot)
//   tamper clear             acknowledge a tamper alarm: motors may run again (stays armed)
//
// ESP32 -> Jetson
//   U,<left_ft>,<right_ft>   every TELEMETRY_MS; -1 means "no reading"
//   S,<mode>,<leftPWM>,<rightPWM>,<avoidState>   every STATUS_MS
//   G,<yaw_deg>,<yaw_rate_dps>   every IMU_TELEMETRY_MS, if the IMU answered at boot
//   W,<left_ticks>,<right_ticks> every IMU_TELEMETRY_MS: wheel encoder counts since boot
//                                (quadrature, both edges of both channels; + = that wheel forward)
//                                (yaw: gyro Z integrated since boot, CCW positive, -180..180)
//   B,<mV>,<mA>  every POWER_PERIOD_MS: battery pack voltage and current (+ = discharging)
//                from the INA226, if it answered at boot
//   T,<state>[,<reason>]  every STATUS_MS and at once on a change: off | armed |
//                tripped,<tilt|lift> | unavailable (no accelerometer)
//   I,<text>   information     E,<text>   error / safety event
//
// Safety:
//   * Every motor output goes through applyMotors(), which enforces MAX_PWM.
//   * Watchdog: if no valid command arrives for WATCHDOG_MS the robot stops, so
//     the Jetson must resend its command at least every ~200 ms.
//   * Net-forward motion from M,l,r is blocked while an ultrasonic sensor sees an
//     obstacle inside STOP_DISTANCE_FEET (rotating in place and reversing still work).
//   * Tamper: while armed, the robot tipped past TAMPER_TILT_DEG or lifted (vertical
//     acceleration off by TAMPER_LIFT_G) stops the motors at once and keeps them stopped,
//     whatever is commanded, until "tamper clear".
#include <Arduino.h>
#include <Wire.h>

// ==================== PINOUT ====================
// Motor driver: TI DRV8262 (dual H-bridge). It has no EN pins: speed and direction both go
// on IN1..IN4, so each IN pin gets its own PWM channel. GPIO 26 / 27 (the L298N's ENA / ENB)
// are not connected on this board and are left alone. The RIGHT motor is on H-bridge 1
// (IN1/IN2) and the LEFT motor on H-bridge 2 (IN3/IN4).
const int RIGHT_IN_A = 32;    // IN1
const int RIGHT_IN_B = 26;    // IN2
const int LEFT_IN_A = 25;     // IN3
const int LEFT_IN_B = 33;     // IN4

// The board's MODE2 pin (DRV8262 pin 29) picks the input scheme, latched at power-up:
//   floating or high -> PWM mode:   IN1/IN2 = 1/0 forward, 0/1 reverse, 1/1 brake, 0/0 coast
//   grounded         -> PH/EN mode: IN1 = enable (0 = brake), IN2 = direction (1 = forward)
// Set this to match the board (false = PWM mode, the default with MODE2 floating).
const bool DRIVER_PH_EN = false;

// Wiring polarity, as verified with the wheels off the ground using the senior1typetest
// sketch (RIGHT_FORWARD_IN1_HIGH = true, LEFT_FORWARD_IN3_HIGH = false): forward is IN1 high
// for the right motor and IN3 low for the left one, so only the left motor is reversed.
// If a wheel ever turns the wrong way for a positive command, flip its flag here.
const bool LEFT_REVERSED = true;
const bool RIGHT_REVERSED = false;

// Encoder pins (see ENCODERS_SWAPPED). GPIO 34/35 are input-only with no internal pull-ups.
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
// wire is A and which is B only decides the sign. Measured 2026-11-22 (driving and spinning,
// checked against the gyro): the counter on the pins named LEFT_* (GPIO 34/35) follows the
// PHYSICAL RIGHT wheel and counts backwards, the one on RIGHT_* (13/18) the physical left
// wheel. The pin names stay with the pins (34/35 need external pull-ups); the W line swaps
// them so it reports the physical wheels. *_REVERSED apply to the physical wheels, so that
// + = that wheel forward.
const bool ENCODERS_SWAPPED = true;

// Which end is the robot's front. 2026-11-22: "forward" drove the robot towards its back (away
// from the ultrasonics, which face the real front). Turning the robot's frame round makes the
// other end the front: the robot's left wheel is then the old right one, run backwards, and
// the other way round, for the motors (applyMotors) and the encoders (readWheelTicks) alike, so
// forward is forward again and a left turn still turns left. Everything else (M / V commands,
// the speed loop, patrol, the obstacle gate on the ultrasonics) works in this robot frame.
const bool FRONT_FLIPPED = true;
const bool LEFT_ENCODER_REVERSED = false;
const bool RIGHT_ENCODER_REVERSED = true;

// IMU: MPU-6050 board (HW-123 / GY-521) on the ESP32's default I2C pins, AD0 low
const int IMU_SDA = 21;
const int IMU_SCL = 22;

// Battery monitor: INA226 on the same I2C bus (SDA 21 / SCL 22), its IN+ to the battery +
// and IN- to the robot's supply (high side), VBS (VBUS) to IN+. Its address (0x40..0x4F, set
// by A0 / A1; 0x40 on boards without A0 / A1 pads) is found at boot. The 24 V pack can read up
// to ~29.4 V, over the INA219's 26 V limit: use the INA226 (36 V).
// SHUNT_OHMS must match the shunt resistor on the module: "R100" = 0.1 ohm (most INA226
// boards; only up to ~0.8 A, too little for the motors), "R010" = 0.01 ohm (up to ~8 A).
const uint8_t INA226_FIRST_ADDR = 0x40;   // the INA226's 16 possible addresses
const uint8_t INA226_LAST_ADDR = 0x4F;
const float SHUNT_OHMS = 0.1;      // the robot's module: R100 (measured 2026-11-28)

// ==================== SETTINGS ====================
// Motor PWM: 10-bit duty (0-1023) at 10 kHz. Chosen with tools/motor_tune.py on the floor
// (2026-10-26): the robot needs ~650 to start from standstill (at 500 it needed a push).
// `freq <Hz>` still changes the frequency at run time for further tuning.
const int PWM_FREQUENCY_HZ = 10000;
const int MIN_PWM_FREQ_HZ = 50;
const int MAX_PWM_FREQ_HZ = 25000;
const int PWM_BITS = 10;
const int RIGHT_A_CHANNEL = 0;        // LEDC channels (Arduino-ESP32 2.x API), one per IN pin
const int RIGHT_B_CHANNEL = 1;
const int LEFT_A_CHANNEL = 2;
const int LEFT_B_CHANNEL = 3;
const int FULL_DUTY = (1 << 10) - 1;  // 1023: an IN pin held high (100 % duty)
const int MAX_PWM = 800;              // hard cap on any wheel PWM, every command path
                                      // (655 until 2026-11-22: laps at 1.25 mph = 0.56 m/s need ~730
                                      //  on the DRV8262, plus headroom for the speed loop)
const int DEFAULT_SPEED_PWM = 650;    // legacy-command speed until "speed N" is received

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
const unsigned long IMU_WARMUP_MS = 1000;        // the gyro reads off for a while after waking
const float IMU_BIAS_MAX_DPS = 5.0;              // healthy boots: +0.3..+0.5 deg/s on this board
const float IMU_CAL_MAX_SPREAD_DPS = 1.0;        // std dev of the samples: more = robot moving
const int IMU_CAL_TRIES = 5;                     // then give up (gyro off) rather than run wrong
const int IMU_MAX_FAILS = 20;                    // consecutive bad reads -> IMU off
const unsigned long IMU_STILL_MS = 1000;         // motors off this long -> refine the bias
const float IMU_STILL_DPS = 1.0;                 // ... while the gyro reads below this
const float IMU_BIAS_RATE = 0.002;               // bias update per sample when still

const unsigned long POWER_PERIOD_MS = 500;       // B line at 2 Hz

// Tamper detection with the MPU-6050's accelerometer (+/-2 g). "Down" is measured at boot
// with the robot standing (any board orientation works). Tipped: the filtered acceleration
// points more than TAMPER_TILT_DEG away from "down" for TAMPER_TILT_MS. Lifted: the size of
// the filtered acceleration is off 1 g by more than TAMPER_LIFT_G (picked up: >1 g, then
// dropped or swung: <1 g) for TAMPER_LIFT_MS; tilting alone leaves it at 1 g. Driving bumps
// are shorter than that.
const float ACCEL_LSB_PER_G = 16384.0;
const unsigned long ACCEL_PERIOD_MS = 10;        // 100 Hz
const float ACCEL_TAU_MS = 60.0;                 // low-pass filter time constant
const float TAMPER_TILT_DEG = 35.0;
const unsigned long TAMPER_TILT_MS = 300;
const float TAMPER_LIFT_G = 0.30;
const unsigned long TAMPER_LIFT_MS = 150;

// Wheel speed control (V command). Measured 2026-11-22 (DRV8262, robot on the floor): a wheel
// at rest needs ~550 PWM to break free, but once rolling it needs far less: ~230 + 0.75 PWM per
// tick/s (300 -> ~95 t/s, 400 -> ~200, 500 -> ~365). The two motors differ by up to 50 %, and
// differently forward and backward, so each wheel has its own PI loop on top of that
// feed-forward, and a stopped wheel that should turn gets at least SPEED_KICK_PWM to start.
const unsigned long SPEED_PERIOD_MS = 40;
const float SPEED_FF_START = 230.0;      // PWM at which a ROLLING wheel just keeps turning
const float SPEED_FF_SLOPE = 0.75;       // PWM per tick/s above that
const float SPEED_KP = 0.5;              // PWM per tick/s of speed error
const float SPEED_KI = 1.5;              // PWM per tick of accumulated error
const float SPEED_I_LIMIT = 150.0;       // PWM the integral may add or take away
const float SPEED_KICK_PWM = 580.0;      // at least this while a wheel that should turn ...
const float SPEED_KICK_BELOW_TPS = 40.0; // ... is slower than this (static friction)
const int MAX_SPEED_TPS = 1000;          // ticks/s (~0.9 m/s)

const float STOP_DISTANCE_FEET = 1.0;    // 12 inches
const float CLEAR_DISTANCE_FEET = 1.5;   // 18 inches
const int OBSTACLE_CONFIRM_READS = 2;    // consecutive close readings before reacting

const unsigned long AVOID_REVERSE_MS = 800;
const unsigned long AVOID_ROTATE_TIMEOUT_MS = 4000;  // give up if never clear
const unsigned long AVOID_SETTLE_MS = 300;

enum Mode { MODE_STOP, MODE_FORWARD, MODE_PATROL, MODE_BACKWARD, MODE_LEFT, MODE_RIGHT, MODE_DRIVE,
            MODE_SPEED };
const char *MODE_NAMES[] = {"stop", "forward", "patrol", "backward", "left", "right", "drive",
                            "speed"};

// ==================== STATE ====================
Mode mode = MODE_STOP;
int speedPWM = DEFAULT_SPEED_PWM;
int pwmFrequencyHz = PWM_FREQUENCY_HZ;
int driveLeft = 0;                 // requested by M,l,r
int targetTps[2] = {0, 0};         // requested by V,l,r: [physical left, physical right]
float measuredTps[2] = {0, 0};     // encoder speeds, filtered
float speedInteg[2] = {0, 0};      // PI integral, ticks
long lastSpeedTicks[2] = {0, 0};
unsigned long lastSpeedMs = 0;
int speedOut[2] = {0, 0};          // PWM the speed loop chose
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

bool wireStarted = false;          // I2C bus up (the IMU's wires are there)
bool powerOk = false;              // INA226 answered at boot
uint8_t inaAddr = 0x40;            // where it answered
unsigned long lastPowerMs = 0;

bool accelOk = false;              // accelerometer works and "down" is known
float gravity[3] = {0, 0, 1};      // unit vector "down" in the board frame, from boot
float accelF[3] = {0, 0, 1};       // filtered acceleration, g
unsigned long lastAccelMs = 0;
bool tamperArmed = false;
bool tamperTripped = false;
const char *tamperReason = "";
unsigned long tiltSinceMs = 0;     // 0 = not tilted
unsigned long liftSinceMs = 0;     // 0 = not lifted

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
void readWheelTicks(long &left, long &right);
void resetSpeedControl();
void updateSpeedControl();
void runAvoidance(int &left, int &right);
void applyMotors(int left, int right);
void outputMotor(int m, int chA, int chB, int pwm);
void setMotor(int chA, int chB, int pwm);
void sendTelemetry();
void imuWireCheck(int &sda, int &scl);
bool imuBusRecover();
void setupImu();
void setupEncoders();
void encoderScope();
void printI2cScan();
void updateImu();
bool imuWrite(uint8_t reg, uint8_t value);
bool imuReadGyroZ(int16_t &raw);
bool imuReadAccel(float a[3]);
void setupAccel();
void updateTamper();
void tripTamper(const char *reason);
void sendTamper();
void setupPowerMonitor();
bool inaRead(uint8_t reg, uint16_t &value);
bool inaWrite(uint8_t reg, uint16_t value);
void sendPower();
float readUltrasonicFeet(int trigPin, int echoPin);
bool validDistance(float distanceFeet);
bool obstacleAhead();
bool pathClear();

// ==================== SETUP ====================
void setup() {
  // Motors first, so nothing can spin while the rest of setup runs.
  const int inPins[4] = {RIGHT_IN_A, RIGHT_IN_B, LEFT_IN_A, LEFT_IN_B};
  const int inChannels[4] = {RIGHT_A_CHANNEL, RIGHT_B_CHANNEL, LEFT_A_CHANNEL, LEFT_B_CHANNEL};
  for (int i = 0; i < 4; i++) {
    ledcSetup(inChannels[i], PWM_FREQUENCY_HZ, PWM_BITS);
    ledcWrite(inChannels[i], 0);
    ledcAttachPin(inPins[i], inChannels[i]);
  }
  applyMotors(0, 0);

  pinMode(ULTRA_LEFT_TRIG, OUTPUT);
  pinMode(ULTRA_LEFT_ECHO, INPUT);
  pinMode(ULTRA_RIGHT_TRIG, OUTPUT);
  pinMode(ULTRA_RIGHT_ECHO, INPUT);

  Serial.begin(115200);
  Serial.println("I,booting");     // before the IMU: a stuck I2C bus shows up as silence after it
  setupImu();                      // motors are already stopped: safe to calibrate
  setupAccel();
  setupPowerMonitor();
  setupEncoders();
  Serial.println("I,SecureBot ESP32 fw 2.2 ready");
}

// ==================== LOOP ====================
void loop() {
  readSerial();
  checkWatchdog();
  updateSensors();
  updateImu();
  updateTamper();
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
  } else if (strcmp(cmd, "tamper on") == 0) {
    tamperArmed = true;
    tiltSinceMs = liftSinceMs = 0;
    sendTamper();
    heartbeat = false;
  } else if (strcmp(cmd, "tamper off") == 0) {
    tamperArmed = false;
    tamperTripped = false;
    sendTamper();
    heartbeat = false;
  } else if (strcmp(cmd, "tamper clear") == 0) {
    if (tamperTripped) Serial.println("I,tamper alarm cleared");
    tamperTripped = false;
    tiltSinceMs = liftSinceMs = 0;
    sendTamper();
    heartbeat = false;
  } else if (strcmp(cmd, "encscope") == 0) {
    encoderScope();                           // diagnostic: stops the motors, takes ~3 s
  } else if (strncmp(cmd, "freq ", 5) == 0) {
    int hz = constrain(atoi(cmd + 5), MIN_PWM_FREQ_HZ, MAX_PWM_FREQ_HZ);
    if (ledcChangeFrequency(RIGHT_A_CHANNEL, hz, PWM_BITS) != 0 &&
        ledcChangeFrequency(RIGHT_B_CHANNEL, hz, PWM_BITS) != 0 &&
        ledcChangeFrequency(LEFT_A_CHANNEL, hz, PWM_BITS) != 0 &&
        ledcChangeFrequency(LEFT_B_CHANNEL, hz, PWM_BITS) != 0) {
      pwmFrequencyHz = hz;
      Serial.print("I,freq ");
      Serial.println(pwmFrequencyHz);
    } else {
      Serial.println("E,freq not supported");
    }
    heartbeat = false;
  } else if (strncmp(cmd, "V,", 2) == 0) {
    int l = 0;
    int r = 0;
    if (sscanf(cmd, "V,%d,%d", &l, &r) == 2) {
      l = constrain(l, -MAX_SPEED_TPS, MAX_SPEED_TPS);
      r = constrain(r, -MAX_SPEED_TPS, MAX_SPEED_TPS);
      if (mode != MODE_SPEED) resetSpeedControl();
      for (int m = 0; m < 2; m++) {
        int t = m == 0 ? l : r;
        if ((t > 0) != (targetTps[m] > 0) || t == 0) speedInteg[m] = 0;   // new direction
        targetTps[m] = t;
      }
      setMode(MODE_SPEED);
    } else {
      valid = false;
    }
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

// An MPU-6050 cut off mid-transfer (the ESP32 reset while it was sending) keeps holding SDA
// low until it gets the rest of its clock pulses; it stays powered through a reset, so only a
// power cycle freed it (2026-11-07: "SDA=0" after some reboots). Clock SCL until SDA is
// released (at most 16 pulses), then send a STOP. Returns true if SDA was held and is free now.
bool imuBusRecover() {
  pinMode(IMU_SDA, INPUT_PULLUP);
  pinMode(IMU_SCL, INPUT_PULLUP);
  delayMicroseconds(10);
  if (digitalRead(IMU_SDA) == HIGH || digitalRead(IMU_SCL) == LOW) return false;
  pinMode(IMU_SCL, OUTPUT_OPEN_DRAIN);
  for (int i = 0; i < 16 && digitalRead(IMU_SDA) == LOW; i++) {
    digitalWrite(IMU_SCL, LOW);
    delayMicroseconds(5);
    digitalWrite(IMU_SCL, HIGH);
    delayMicroseconds(5);
  }
  // STOP: SDA low -> high while SCL is high
  pinMode(IMU_SDA, OUTPUT_OPEN_DRAIN);
  digitalWrite(IMU_SDA, LOW);
  delayMicroseconds(5);
  digitalWrite(IMU_SCL, HIGH);
  delayMicroseconds(5);
  digitalWrite(IMU_SDA, HIGH);
  delayMicroseconds(5);
  pinMode(IMU_SDA, INPUT_PULLUP);
  pinMode(IMU_SCL, INPUT_PULLUP);
  delayMicroseconds(10);
  return digitalRead(IMU_SDA) == HIGH;
}

void setupImu() {
  // A gyro left holding the bus by a reset: free it first (see imuBusRecover).
  if (imuBusRecover()) Serial.println("I,imu bus freed (SDA was held low after a reset)");
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
  wireStarted = true;
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
  // Bias calibration. Right after waking, the MPU-6050 here read several deg/s (once 41)
  // off, and that offset was then subtracted for the whole run: the robot saw itself turning
  // while still and spun in place correcting it. So: let it settle first, and only accept a
  // bias that is plausible and measured while still (small spread); otherwise retry.
  Serial.println("I,imu calibrating: keep the robot still");
  delay(IMU_WARMUP_MS);
  bool calibrated = false;
  for (int attempt = 1; attempt <= IMU_CAL_TRIES && !calibrated; attempt++) {
    double sum = 0, sumSq = 0;
    int good = 0;
    for (int i = 0; i < IMU_CALIBRATE_SAMPLES; i++) {
      int16_t raw;
      if (imuReadGyroZ(raw)) {
        double dps = raw / IMU_LSB_PER_DPS;
        sum += dps;
        sumSq += dps * dps;
        good++;
      }
      delay(3);
    }
    if (good < IMU_CALIBRATE_SAMPLES / 2) {
      Serial.println("E,imu not answering, gyro off");
      return;
    }
    double mean = sum / good;
    double spread = sqrt(max(0.0, sumSq / good - mean * mean));
    if (fabs(mean) <= IMU_BIAS_MAX_DPS && spread <= IMU_CAL_MAX_SPREAD_DPS) {
      imuBiasDps = (float)mean;
      calibrated = true;
    } else {
      Serial.print("E,imu calibration rejected (try ");
      Serial.print(attempt);
      Serial.print("): bias ");
      Serial.print(mean, 3);
      Serial.print(" deg/s, spread ");
      Serial.print(spread, 3);
      Serial.println(" deg/s");
      delay(300);
    }
  }
  if (!calibrated) {
    Serial.println("E,imu calibration failed (robot moving or gyro faulty), gyro off");
    return;
  }
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

  if (tamperTripped) {                     // stays stopped until "tamper clear"
    applyMotors(0, 0);
    return;
  }

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

    case MODE_SPEED:
      updateSpeedControl();
      left = speedOut[0];
      right = speedOut[1];
      // Failsafe: never drive net-forward into a confirmed close obstacle.
      if (targetTps[0] + targetTps[1] > 0 && obstacleAhead()) {
        left = 0;
        right = 0;
        speedInteg[0] = speedInteg[1] = 0;
      }
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
// Physical wheel encoder counts (+ = forward), see ENCODERS_SWAPPED.
void readWheelTicks(long &left, long &right) {
  noInterrupts();
  long a = leftTicks;
  long b = rightTicks;
  interrupts();
  long l = ENCODERS_SWAPPED ? b : a;
  long r = ENCODERS_SWAPPED ? a : b;
  l = LEFT_ENCODER_REVERSED ? -l : l;
  r = RIGHT_ENCODER_REVERSED ? -r : r;
  left = FRONT_FLIPPED ? -r : l;                     // see FRONT_FLIPPED
  right = FRONT_FLIPPED ? -l : r;
}

void resetSpeedControl() {
  readWheelTicks(lastSpeedTicks[0], lastSpeedTicks[1]);
  lastSpeedMs = millis();
  for (int m = 0; m < 2; m++) {
    measuredTps[m] = 0;
    speedInteg[m] = 0;
    speedOut[m] = 0;
  }
}

// PI speed loop per wheel, every SPEED_PERIOD_MS: PWM = feed-forward for the target speed
// + KP * error + KI * integral. A target of 0 brakes (outputMotor) and clears the integral.
void updateSpeedControl() {
  unsigned long now = millis();
  unsigned long dtMs = now - lastSpeedMs;
  if (dtMs < SPEED_PERIOD_MS) return;
  float dt = dtMs / 1000.0f;
  long ticks[2];
  readWheelTicks(ticks[0], ticks[1]);
  lastSpeedMs = now;
  for (int m = 0; m < 2; m++) {
    float rate = (ticks[m] - lastSpeedTicks[m]) / dt;
    lastSpeedTicks[m] = ticks[m];
    measuredTps[m] = 0.5f * measuredTps[m] + 0.5f * rate;
    int target = targetTps[m];
    if (target == 0) {
      speedInteg[m] = 0;
      speedOut[m] = 0;
      continue;
    }
    float sign = target > 0 ? 1.0f : -1.0f;
    float err = target - measuredTps[m];
    speedInteg[m] = constrain(speedInteg[m] + err * dt, -SPEED_I_LIMIT / SPEED_KI,
                              SPEED_I_LIMIT / SPEED_KI);
    float pwm = sign * (SPEED_FF_START + SPEED_FF_SLOPE * fabs((float)target))
                + SPEED_KP * err + SPEED_KI * speedInteg[m];
    if (pwm * sign < 0) pwm = 0;                       // never push the wrong way
    if (fabs(measuredTps[m]) < SPEED_KICK_BELOW_TPS && fabs(pwm) < SPEED_KICK_PWM) {
      pwm = sign * SPEED_KICK_PWM;                     // break static friction
    }
    speedOut[m] = constrain((int)lroundf(pwm), -MAX_PWM, MAX_PWM);
  }
}

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
  int motorLeft = FRONT_FLIPPED ? -right : left;     // see FRONT_FLIPPED
  int motorRight = FRONT_FLIPPED ? -left : right;

  outputMotor(0, LEFT_A_CHANNEL, LEFT_B_CHANNEL, LEFT_REVERSED ? -motorLeft : motorLeft);
  outputMotor(1, RIGHT_A_CHANNEL, RIGHT_B_CHANNEL, RIGHT_REVERSED ? -motorRight : motorRight);

  appliedLeft = left;
  appliedRight = right;
}

// A wheel told 0 is braked (PWM mode: IN1 = IN2 = high; PH/EN mode: enable low) for as long as
// it is told 0, not released to coast. With the DRV8262 the robot turns ~145 deg/s at full
// PWM, and a 150 ms brake followed by coasting let it turn on another ~60 deg after a stop.
void outputMotor(int m, int chA, int chB, int pwm) {
  (void)m;
  if (pwm == 0) {
    if (DRIVER_PH_EN) {
      ledcWrite(chA, 0);                      // enable low = brake
      ledcWrite(chB, 0);
    } else {
      ledcWrite(chA, FULL_DUTY);              // both high = brake
      ledcWrite(chB, FULL_DUTY);
    }
    return;
  }
  setMotor(chA, chB, pwm);
}

// Speed is the PWM duty on the IN pins (DRV8262: a steady high on an IN pin is 100 %).
// PWM mode: the driving input is pulsed and the other held low, so the motor coasts between
//   pulses, like the L298N's EN-pin PWM the speeds were tuned with.
// PH/EN mode: IN1 (enable) is pulsed and IN2 holds the direction; the motor brakes between
//   pulses, and 0 is a brake (this mode cannot coast).
void setMotor(int chA, int chB, int pwm) {
  int duty = min(abs(pwm), FULL_DUTY);
  if (DRIVER_PH_EN) {
    ledcWrite(chA, duty);
    ledcWrite(chB, pwm > 0 ? FULL_DUTY : 0);
  } else if (pwm > 0) {
    ledcWrite(chB, 0);
    ledcWrite(chA, duty);
  } else if (pwm < 0) {
    ledcWrite(chA, 0);
    ledcWrite(chB, duty);
  } else {
    ledcWrite(chA, 0);
    ledcWrite(chB, 0);
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
    long left, right;
    readWheelTicks(left, right);
    Serial.print("W,");
    Serial.print(left);
    Serial.print(",");
    Serial.println(right);
  }

  if (powerOk && now - lastPowerMs >= POWER_PERIOD_MS) {
    lastPowerMs = now;
    sendPower();
  }

  if (now - lastStatusMs >= STATUS_MS) {
    lastStatusMs = now;
    sendTamper();
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

// ==================== TAMPER (MPU-6050 accelerometer) ====================
bool imuReadAccel(float a[3]) {
  Wire.beginTransmission(imuAddr);
  Wire.write(0x3B);                          // ACCEL_XOUT_H .. ACCEL_ZOUT_L
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)imuAddr, 6) != 6) return false;
  for (int i = 0; i < 3; i++) {
    int hi = Wire.read();
    int lo = Wire.read();
    a[i] = (int16_t)((hi << 8) | lo) / ACCEL_LSB_PER_G;
  }
  return true;
}

// Measure "down" with the robot standing (right after the gyro calibration, which already
// made sure it is still). Without it tamper detection is unavailable.
void setupAccel() {
  if (!imuOk) return;
  float sum[3] = {0, 0, 0};
  int good = 0;
  for (int i = 0; i < 50; i++) {
    float a[3];
    if (imuReadAccel(a)) {
      for (int k = 0; k < 3; k++) sum[k] += a[k];
      good++;
    }
    delay(2);
  }
  float n = good ? sqrtf(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]) / good : 0;
  if (good < 25 || n < 0.7f || n > 1.3f) {
    Serial.println("E,accelerometer not usable: tamper detection unavailable");
    return;
  }
  for (int k = 0; k < 3; k++) {
    gravity[k] = sum[k] / good / n;
    accelF[k] = gravity[k];
  }
  accelOk = true;
  lastAccelMs = millis();
  Serial.println("I,accelerometer ready (tamper detection: send \"tamper on\")");
}

void updateTamper() {
  if (!accelOk || !imuOk) return;
  unsigned long now = millis();
  if (now - lastAccelMs < ACCEL_PERIOD_MS) return;
  float k = (now - lastAccelMs) / (ACCEL_TAU_MS + (now - lastAccelMs));
  lastAccelMs = now;
  float a[3];
  if (!imuReadAccel(a)) return;              // the gyro side counts I2C failures
  for (int i = 0; i < 3; i++) accelF[i] += k * (a[i] - accelF[i]);
  if (!tamperArmed || tamperTripped) return;

  float down = accelF[0] * gravity[0] + accelF[1] * gravity[1] + accelF[2] * gravity[2];
  float mag = sqrtf(accelF[0] * accelF[0] + accelF[1] * accelF[1] + accelF[2] * accelF[2]);
  float cosTilt = mag > 0.2f ? down / mag : 1.0f;
  bool tilted = cosTilt < cosf(TAMPER_TILT_DEG * (float)M_PI / 180.0f);
  bool lifted = fabsf(mag - 1.0f) > TAMPER_LIFT_G;     // tilting alone keeps 1 g

  if (!tilted) tiltSinceMs = 0;
  else if (tiltSinceMs == 0) tiltSinceMs = now;
  if (!lifted) liftSinceMs = 0;
  else if (liftSinceMs == 0) liftSinceMs = now;

  if (tiltSinceMs && now - tiltSinceMs >= TAMPER_TILT_MS) tripTamper("tilt");
  else if (liftSinceMs && now - liftSinceMs >= TAMPER_LIFT_MS) tripTamper("lift");
}

void tripTamper(const char *reason) {
  tamperTripped = true;
  tamperReason = reason;
  setMode(MODE_STOP);
  applyMotors(0, 0);
  Serial.print("E,TAMPER: ");
  Serial.println(reason);
  sendTamper();
}

void sendTamper() {
  if (!accelOk || !imuOk) {               // no accelerometer, or the IMU stopped answering
    Serial.println("T,unavailable");
  } else if (tamperTripped) {
    Serial.print("T,tripped,");
    Serial.println(tamperReason);
  } else {
    Serial.println(tamperArmed ? "T,armed" : "T,off");
  }
}

// ==================== BATTERY (INA226) ====================
bool inaWrite(uint8_t reg, uint16_t value) {
  Wire.beginTransmission(inaAddr);
  Wire.write(reg);
  Wire.write((uint8_t)(value >> 8));
  Wire.write((uint8_t)(value & 0xFF));
  return Wire.endTransmission() == 0;
}

bool inaRead(uint8_t reg, uint16_t &value) {
  Wire.beginTransmission(inaAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)inaAddr, 2) != 2) return false;
  int hi = Wire.read();
  int lo = Wire.read();
  value = (uint16_t)((hi << 8) | lo);
  return true;
}

void setupPowerMonitor() {
  if (!wireStarted) {                        // the IMU part did not start the bus
    int sda, scl;
    imuWireCheck(sda, scl);
    if (!sda || !scl) {
      Serial.println("E,power monitor not found (INA226 on SDA 21 / SCL 22: no I2C wires)");
      return;
    }
    Wire.begin(IMU_SDA, IMU_SCL);
    Wire.setClock(100000);
    Wire.setTimeOut(10);
    wireStarted = true;
  }
  bool found = false;
  for (uint8_t a = INA226_FIRST_ADDR; a <= INA226_LAST_ADDR && !found; a++) {
    inaAddr = a;
    uint16_t id = 0, die = 0;
    // manufacturer ID "TI" and the INA226's die ID (an INA219 / INA260 there is not used)
    found = inaRead(0xFE, id) && id == 0x5449 && inaRead(0xFF, die) && (die >> 4) == 0x226;
  }
  if (!found) {
    Serial.println("E,power monitor not found (INA226 at 0x40-0x4F): battery level unknown");
    return;
  }
  // CONFIG: average 16 samples, 1.1 ms bus and shunt conversions, continuous: a new value
  // every ~35 ms, smooth enough for the motors' PWM ripple
  if (!inaWrite(0x00, 0x4527)) {
    Serial.println("E,power monitor did not accept its settings: battery level unknown");
    return;
  }
  powerOk = true;
  char addr[24];
  snprintf(addr, sizeof addr, "INA226 at 0x%02X)", inaAddr);
  Serial.print("I,power monitor ready (");
  Serial.println(addr);
}

void sendPower() {
  uint16_t bus, shunt;
  if (!inaRead(0x02, bus) || !inaRead(0x01, shunt)) return;
  long mv = lroundf(bus * 1.25f);                                  // 1.25 mV per bit
  long ma = lroundf((int16_t)shunt * 2.5e-3f / SHUNT_OHMS);       // 2.5 uV per bit
  Serial.print("B,");
  Serial.print(mv);
  Serial.print(",");
  Serial.println(ma);
}
