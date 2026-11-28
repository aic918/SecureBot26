// Host-side tests for securebot_esp32.ino. Build/run: see test/run_tests.sh
#include <Arduino.h>
#include <string>
#include "../securebot_esp32/securebot_esp32.ino"

// ---------- fake hardware ----------
static unsigned long g_ms = 0;
static int g_pin[64];
static int g_pwm[64];
static long g_echo_us[64];          // pulseIn result per echo pin (0 = no echo)
static std::string g_in, g_out;
static size_t g_in_pos = 0;

unsigned long millis() { return g_ms; }
unsigned long micros() { return g_ms * 1000UL; }

// Fake MPU-6050 on the I2C bus: registers written, a gyro Z reading tests can set.
static bool g_imu_present = true;
static bool g_imu_wired = true;      // module pull-ups on SDA/SCL: the pins read 1
static int16_t g_gyro_z_raw = 0;
static uint8_t g_imu_reg[128];
static int g_wire_sda = -1, g_wire_scl = -1;
static uint8_t g_tx[8];
static int g_tx_len = 0, g_rx_pos = 0, g_rx_len = 0;
static uint8_t g_rx[8];
static bool g_tx_ok = false;
TwoWire Wire;
bool TwoWire::begin(int sda, int scl) { g_wire_sda = sda; g_wire_scl = scl; return true; }
void TwoWire::setClock(uint32_t) {}
void TwoWire::setTimeOut(uint16_t) {}
static int g_wake_after_writes = 0;          // >0: the IMU starts answering after this many tries
void TwoWire::beginTransmission(uint8_t addr) {
  g_tx_len = 0;
  if (g_wake_after_writes > 0 && --g_wake_after_writes == 0) g_imu_present = true;
  g_tx_ok = g_imu_present && addr == 0x68;
}
size_t TwoWire::write(uint8_t b) { if (g_tx_len < 8) g_tx[g_tx_len++] = b; return 1; }
uint8_t TwoWire::endTransmission(bool) {
  if (!g_tx_ok) return 2;                                  // NACK on address
  if (g_tx_len == 2) g_imu_reg[g_tx[0] & 127] = g_tx[1];   // register write
  return 0;
}
uint8_t TwoWire::requestFrom(int addr, int n) {
  if (!g_imu_present || addr != 0x68) return 0;
  if (g_tx_len >= 1 && g_tx[0] == 0x47 && n == 2) {        // GYRO_ZOUT_H, _L
    g_rx[0] = (uint8_t)((uint16_t)g_gyro_z_raw >> 8); g_rx[1] = (uint8_t)(g_gyro_z_raw & 0xFF);
    g_rx_len = 2; g_rx_pos = 0; return 2;
  }
  return 0;
}
int TwoWire::available() { return g_rx_len - g_rx_pos; }
int TwoWire::read() { return g_rx_pos < g_rx_len ? g_rx[g_rx_pos++] : -1; }
void delay(unsigned long) {}
void delayMicroseconds(unsigned int) {}
long pulseIn(int pin, int, unsigned long) { return g_echo_us[pin]; }
void pinMode(int, int) {}
void digitalWrite(int pin, int v) { g_pin[pin] = v; }
int digitalRead(int pin) { return g_pin[pin]; }
static int g_mv[64];
int analogReadMilliVolts(int pin) { g_ms++; return g_mv[pin]; }
static void (*g_isr[64])();
int digitalPinToInterrupt(int pin) { return pin; }
void attachInterrupt(int irq, void (*isr)(), int) { g_isr[irq] = isr; }
void interrupts() {}
void noInterrupts() {}
// drive one encoder's two wires through `steps` quadrature steps (+ forward, - backward)
static void turnEncoder(int pinA, int pinB, int steps) {
  static const int seq[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
  int k = 0;
  for (int i = 0; i < 4; i++) if (g_pin[pinA] == seq[i][0] && g_pin[pinB] == seq[i][1]) k = i;
  for (int n = 0; n < abs(steps); n++) {
    k = (k + (steps > 0 ? 1 : 3)) % 4;
    int changed = g_pin[pinA] != seq[k][0] ? pinA : pinB;
    g_pin[pinA] = seq[k][0]; g_pin[pinB] = seq[k][1];
    g_isr[changed]();
  }
}
void analogWrite(int pin, int v) { g_pwm[pin] = v; }
static int g_chan_pin[16];
static int g_chan_bits[16];
static double g_chan_freq[16];
double ledcSetup(int ch, double f, int bits) { g_chan_freq[ch] = f; g_chan_bits[ch] = bits; return f; }
void ledcAttachPin(int pin, int ch) { g_chan_pin[ch] = pin; }
void ledcWrite(int ch, int duty) { g_pwm[g_chan_pin[ch]] = duty; }  // recorded by IN pin
uint32_t ledcChangeFrequency(int ch, uint32_t f, int bits) { g_chan_freq[ch] = f; g_chan_bits[ch] = bits; return f; }
SerialClass Serial;
int SerialClass::available() { return (int)(g_in.size() - g_in_pos); }
int SerialClass::read() { return g_in_pos < g_in.size() ? (unsigned char)g_in[g_in_pos++] : -1; }
void SerialClass::print(const char *s) { g_out += s; }
void SerialClass::print(char c) { g_out += c; }
void SerialClass::print(int v) { g_out += std::to_string(v); }
void SerialClass::print(unsigned long v) { g_out += std::to_string(v); }
void SerialClass::print(long v) { g_out += std::to_string(v); }
void SerialClass::print(double v, int d) { char b[32]; snprintf(b, sizeof b, "%.*f", d, v); g_out += b; }
void SerialClass::println() { g_out += "\n"; }
void SerialClass::println(const char *s) { g_out += s; g_out += "\n"; }
void SerialClass::println(int v) { print(v); g_out += "\n"; }
void SerialClass::println(long v) { print(v); g_out += "\n"; }
void SerialClass::println(double v, int d) { print(v, d); g_out += "\n"; }

// ---------- helpers ----------
static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("  FAIL line %d: %s\n", __LINE__, #cond); failures++; } } while (0)

// DRV8262 in PWM mode: the duty on the driving IN pin is the speed, the other pin low.
// Both high = brake, both low = coast (both read as 0 here).
static int bridgePWM(int inA, int inB) {
  if (g_pwm[inA] > 0 && g_pwm[inB] == 0) return g_pwm[inA];
  if (g_pwm[inB] > 0 && g_pwm[inA] == 0) return -g_pwm[inB];
  return 0;
}
static bool braked(int inA, int inB) { return g_pwm[inA] == FULL_DUTY && g_pwm[inB] == FULL_DUTY; }
static int motorLeftPWM() { int v = bridgePWM(LEFT_IN_A, LEFT_IN_B); return LEFT_REVERSED ? -v : v; }
static int motorRightPWM() { int v = bridgePWM(RIGHT_IN_A, RIGHT_IN_B); return RIGHT_REVERSED ? -v : v; }
// the robot-frame command (undoes FRONT_FLIPPED, see applyMotors)
static int leftPWM() { return FRONT_FLIPPED ? -motorRightPWM() : motorLeftPWM(); }
static int rightPWM() { return FRONT_FLIPPED ? -motorLeftPWM() : motorRightPWM(); }
static void send(const std::string &s) { g_in += s; }
static void step(int ms) { for (int i = 0; i < ms; i++) { g_ms++; loop(); } }
// step while resending `cmd` every 100 ms like the Jetson's bridge does
static void stepHB(int ms, const std::string &cmd) {
  for (int i = 0; i < ms; i++) { if (g_ms % 100 == 0) send(cmd + "\n"); g_ms++; loop(); }
}
static bool outHas(const char *s) { return g_out.find(s) != std::string::npos; }
static void boot() {
  memset(g_pin, 0, sizeof g_pin); memset(g_pwm, 0, sizeof g_pwm);
  g_pin[IMU_SDA] = g_pin[IMU_SCL] = g_imu_wired;
  setup();
}
const long ECHO_HALF_FT = 889;    // ~0.5 ft: inside stop distance
const long ECHO_2FT = 3556;       // ~2.0 ft: clear

// ---------- scenarios ----------
static void t_boot() {
  boot(); step(10);
  CHECK(leftPWM() == 0 && rightPWM() == 0);
  CHECK(outHas("I,SecureBot ESP32 fw 2.1 ready"));
}
static void t_drive_and_clamp() {
  boot();
  send("M,100,100\n"); step(5);
  CHECK(leftPWM() == 100 && rightPWM() == 100);
  send("M,999,-999\n"); step(5);
  CHECK(leftPWM() == MAX_PWM && rightPWM() == -MAX_PWM);
  send("M,-60,40\n"); step(5);
  CHECK(leftPWM() == -60 && rightPWM() == 40);
}
static void t_speed_cap() {
  boot();
  send("speed 2000\n"); send("patrol\n"); step(5);
  CHECK(leftPWM() == MAX_PWM && rightPWM() == MAX_PWM);
  send("speed -5\n"); step(5);
  CHECK(leftPWM() == 0 && rightPWM() == 0);
}
static void t_output_stage_cap() {
  boot();
  applyMotors(999, -999);                       // bypasses the command parser entirely
  CHECK(leftPWM() == MAX_PWM && rightPWM() == -MAX_PWM);
  for (int pin : {LEFT_IN_A, LEFT_IN_B, RIGHT_IN_A, RIGHT_IN_B}) CHECK(g_pwm[pin] <= MAX_PWM);
}
static void t_watchdog() {
  boot();
  send("M,100,100\n"); step(200);
  CHECK(leftPWM() == 100);
  step(100);                                   // 300 ms with no command (> WATCHDOG_MS)
  CHECK(leftPWM() == 0 && rightPWM() == 0);
  CHECK(outHas("E,watchdog"));
}
static void t_heartbeat_keeps_running() {
  boot();
  stepHB(3000, "M,80,80");
  CHECK(leftPWM() == 80 && rightPWM() == 80);
  CHECK(!outHas("E,watchdog"));
}
static void t_partial_line() {
  boot();
  send("M,50,5"); step(2000);                  // no newline: must not act or block
  CHECK(leftPWM() == 0);
  send("0\n"); step(5);
  CHECK(leftPWM() == 50 && rightPWM() == 50);
}
static void t_bad_input() {
  boot();
  send(std::string(100, 'x') + "\n"); step(5);
  CHECK(outHas("E,line too long"));
  CHECK(leftPWM() == 0);
  send("M,abc\n"); send("jump\n"); step(5);
  CHECK(outHas("E,bad command: M,abc") && outHas("E,bad command: jump"));
  CHECK(leftPWM() == 0);
  send("  M,30,30  \r\n"); step(5);            // whitespace and CRLF tolerated
  CHECK(leftPWM() == 30 && rightPWM() == 30);
}
static void t_bad_command_not_heartbeat() {
  boot();
  send("M,100,100\n"); step(300);
  for (int i = 0; i < 6; i++) { send("garbage\n"); step(100); }   // junk must not feed the watchdog
  CHECK(leftPWM() == 0);
}
static void t_drive_obstacle_gate() {
  boot();
  g_echo_us[ULTRA_LEFT_ECHO] = ECHO_HALF_FT; g_echo_us[ULTRA_RIGHT_ECHO] = ECHO_HALF_FT;
  stepHB(300, "M,100,100");
  CHECK(leftPWM() == 0 && rightPWM() == 0);     // net-forward blocked
  stepHB(300, "M,-100,100");
  CHECK(leftPWM() == -100 && rightPWM() == 100);// rotate in place allowed
  stepHB(300, "M,-100,-100");
  CHECK(leftPWM() == -100 && rightPWM() == -100);// reverse allowed
  g_echo_us[ULTRA_LEFT_ECHO] = 0; g_echo_us[ULTRA_RIGHT_ECHO] = 0;
  stepHB(300, "M,100,100");
  CHECK(leftPWM() == 100 && rightPWM() == 100); // clears when obstacle gone
}
static void t_single_glitch_ignored() {
  boot();
  stepHB(100, "M,100,100");
  g_echo_us[ULTRA_LEFT_ECHO] = ECHO_HALF_FT; step(41);   // exactly one close left reading
  g_echo_us[ULTRA_LEFT_ECHO] = 0;
  stepHB(400, "M,100,100");
  CHECK(leftPWM() == 100 && rightPWM() == 100);
}
static void t_patrol_avoidance_cycle() {
  boot();
  stepHB(300, "patrol");
  CHECK(leftPWM() == DEFAULT_SPEED_PWM && rightPWM() == DEFAULT_SPEED_PWM);
  g_echo_us[ULTRA_LEFT_ECHO] = ECHO_HALF_FT; g_echo_us[ULTRA_RIGHT_ECHO] = ECHO_HALF_FT;
  bool sawStop = false, sawReverse = false, sawRotate = false;
  for (int i = 0; i < 2000 && !sawRotate; i++) {
    stepHB(1, "patrol");
    if (avoidState == 1 && leftPWM() == 0) sawStop = true;
    if (leftPWM() == -DEFAULT_SPEED_PWM && rightPWM() == -DEFAULT_SPEED_PWM) sawReverse = true;
    if (leftPWM() == DEFAULT_SPEED_PWM && rightPWM() == -DEFAULT_SPEED_PWM) sawRotate = true;
  }
  CHECK(sawStop && sawReverse && sawRotate);
  g_echo_us[ULTRA_LEFT_ECHO] = ECHO_2FT; g_echo_us[ULTRA_RIGHT_ECHO] = ECHO_2FT;
  bool sawSettle = false, resumed = false;
  for (int i = 0; i < 1500 && !resumed; i++) {
    stepHB(1, "patrol");
    if (avoidState == 3 && leftPWM() == 0) sawSettle = true;
    if (avoidState == 0 && leftPWM() == DEFAULT_SPEED_PWM && rightPWM() == DEFAULT_SPEED_PWM) resumed = true;
  }
  CHECK(sawSettle && resumed);
}
static void t_avoid_timeout() {
  boot();
  g_echo_us[ULTRA_LEFT_ECHO] = ECHO_HALF_FT; g_echo_us[ULTRA_RIGHT_ECHO] = ECHO_HALF_FT;
  int i = 0;
  for (; i < 7000 && !outHas("E,avoid timeout"); i++) stepHB(1, "patrol");   // obstacle never clears
  CHECK(outHas("E,avoid timeout"));
  CHECK((unsigned long)i > AVOID_REVERSE_MS + AVOID_ROTATE_TIMEOUT_MS - 400 && i < 6500);
  CHECK(leftPWM() == 0 && rightPWM() == 0);
  CHECK(mode == MODE_STOP);
  step(100);                                    // no further commands: stays stopped
  CHECK(leftPWM() == 0 && rightPWM() == 0 && mode == MODE_STOP);
}
static void t_telemetry() {
  boot(); g_out.clear();
  step(250);
  CHECK(outHas("U,-1.00,-1.00"));
  g_echo_us[ULTRA_LEFT_ECHO] = ECHO_2FT; g_echo_us[ULTRA_RIGHT_ECHO] = ECHO_2FT; g_out.clear();
  step(300);
  CHECK(outHas("U,2.00,2.00"));
  CHECK(outHas("S,stop,0,0,0"));
}

static void t_pinout() {
  int pins[] = {LEFT_IN_A, LEFT_IN_B, RIGHT_IN_A, RIGHT_IN_B,
                ULTRA_LEFT_TRIG, ULTRA_LEFT_ECHO, ULTRA_RIGHT_TRIG, ULTRA_RIGHT_ECHO,
                LEFT_ENCODER_BLUE, LEFT_ENCODER_WHITE, RIGHT_ENCODER_BLUE, RIGHT_ENCODER_WHITE,
                IMU_SDA, IMU_SCL};
  int n = sizeof pins / sizeof pins[0];
  for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) CHECK(pins[i] != pins[j]);
  CHECK(RIGHT_IN_A == 32 && RIGHT_IN_B == 26 && LEFT_IN_A == 25 && LEFT_IN_B == 33);
  // No input/output may sit on a boot-strapping, flash or USB-serial pin, nor on GPIO 14
  // (it goes high during boot and drove a motor while flashing).
  const int forbidden[] = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15};
  for (int i = 0; i < n; i++) for (int f : forbidden) CHECK(pins[i] != f);
  CHECK(RIGHT_ENCODER_WHITE == 18);
  CHECK(ULTRA_LEFT_TRIG == 23 && ULTRA_LEFT_ECHO == 19 && ULTRA_RIGHT_TRIG == 4 && ULTRA_RIGHT_ECHO == 17);
  boot();
  send("M,120,-70\n"); step(5);                   // left forward 120, right reverse 70
  CHECK(leftPWM() == 120 && rightPWM() == -70);
}

static void t_polarity() {
  // Measured on the robot 2026-11-22 (DRV8262, front flipped): forward pulses IN2 (GPIO 26)
  // on the right H-bridge and IN3 (GPIO 25) on the left one.
  CHECK(LEFT_REVERSED && !RIGHT_REVERSED && FRONT_FLIPPED && !DRIVER_PH_EN);
  boot();
  send("M,100,100\n"); step(5);                  // both wheels forward
  CHECK(g_pwm[RIGHT_IN_A] == 0 && g_pwm[RIGHT_IN_B] == 100);
  CHECK(g_pwm[LEFT_IN_A] == 100 && g_pwm[LEFT_IN_B] == 0);
  send("M,-100,-100\n"); step(5);                // both wheels backward
  CHECK(g_pwm[RIGHT_IN_A] == 100 && g_pwm[RIGHT_IN_B] == 0);
  CHECK(g_pwm[LEFT_IN_A] == 0 && g_pwm[LEFT_IN_B] == 100);
  send("M,100,0\n"); step(5);                    // robot-left forward = motor on IN1/IN2
  CHECK(g_pwm[RIGHT_IN_B] == 100 && braked(LEFT_IN_A, LEFT_IN_B));
  send("stop\n"); step(200);                     // stopped: held braked
  CHECK(braked(LEFT_IN_A, LEFT_IN_B) && braked(RIGHT_IN_A, RIGHT_IN_B));
}

static void t_pwm_setup() {
  // 10-bit duty at 10 kHz, one LEDC channel per IN pin (the DRV8262 has no enable pins).
  boot();
  const int ch[4] = {RIGHT_A_CHANNEL, RIGHT_B_CHANNEL, LEFT_A_CHANNEL, LEFT_B_CHANNEL};
  const int pin[4] = {RIGHT_IN_A, RIGHT_IN_B, LEFT_IN_A, LEFT_IN_B};
  for (int i = 0; i < 4; i++) {
    CHECK(g_chan_pin[ch[i]] == pin[i]);
    CHECK(g_chan_bits[ch[i]] == 10 && g_chan_freq[ch[i]] == 10000);
    for (int j = i + 1; j < 4; j++) CHECK(ch[i] != ch[j]);
  }
  CHECK(MAX_PWM < FULL_DUTY && MAX_PWM == 800 && DEFAULT_SPEED_PWM == 650);
  send("M,660,660\n"); step(5);
  CHECK(leftPWM() == 660 && rightPWM() == 660);
}

static void t_freq_command() {
  boot();
  send("freq 1000\n"); step(5);
  for (int c : {RIGHT_A_CHANNEL, RIGHT_B_CHANNEL, LEFT_A_CHANNEL, LEFT_B_CHANNEL})
    CHECK(g_chan_freq[c] == 1000 && g_chan_bits[c] == 10);
  CHECK(pwmFrequencyHz == 1000 && outHas("I,freq 1000"));
  send("freq 999999\n"); step(5);                 // clamped
  CHECK(pwmFrequencyHz == MAX_PWM_FREQ_HZ);
  send("freq 1\n"); step(5);
  CHECK(pwmFrequencyHz == MIN_PWM_FREQ_HZ);
  // freq is not a heartbeat: a driving robot still stops when motion commands stop
  send("M,300,300\n"); step(100);
  for (int i = 0; i < 4; i++) { send("freq 500\n"); step(100); }
  CHECK(leftPWM() == 0 && rightPWM() == 0 && outHas("E,watchdog"));
}

// ---------- IMU ----------
static float glineYaw() {          // yaw from the last G line, or NAN
  size_t i = g_out.rfind("G,");
  return i == std::string::npos ? NAN : (float)atof(g_out.c_str() + i + 2);
}
static void t_imu_setup() {
  g_gyro_z_raw = 131;              // a gyro that reads 2 deg/s standing still (bias)
  boot();
  CHECK(g_wire_sda == 21 && g_wire_scl == 22);
  CHECK(g_imu_reg[0x6B] == 0x01 && g_imu_reg[0x1B] == 0x08 && g_imu_reg[0x1A] == 0x03);
  CHECK(imuOk && fabs(imuBiasDps - 2.0) < 0.01);
  CHECK(outHas("I,imu ready, gyro bias 2.000"));
  step(200);
  CHECK(outHas("G,") && fabs(glineYaw()) < 0.5);      // bias removed: yaw stays at 0
}
static void t_imu_yaw_integrates() {
  g_gyro_z_raw = 131;
  boot();
  send("M,-300,300\n");                                // turning: bias tracking paused
  g_gyro_z_raw = 131 + (int16_t)(90 * 65.5);           // +90 deg/s on top of the bias
  stepHB(1000, "M,-300,300");
  CHECK(fabs(yawDeg - 90.0) < 3.0);
  g_gyro_z_raw = 131 - (int16_t)(90 * 65.5);           // and back
  stepHB(1000, "M,300,-300");
  CHECK(fabs(yawDeg) < 3.0);
  g_gyro_z_raw = 131 + (int16_t)(120 * 65.5);          // wraps at 180
  stepHB(2000, "M,-300,300");
  CHECK(yawDeg < -110 && yawDeg > -130);               // 240 deg = -120 deg
}
static void t_imu_bias_tracks_when_still() {
  g_gyro_z_raw = 0;
  boot();
  g_gyro_z_raw = 33;                                   // bias drifts to ~0.5 deg/s
  step(15000);                                         // motors off: bias follows
  CHECK(fabs(imuBiasDps - 33 / 65.5) < 0.05);
  float yaw = yawDeg;
  step(5000);
  CHECK(fabs(yawDeg - yaw) < 0.3);                     // no more creep
}
static void t_imu_absent() {
  g_imu_present = false;
  g_imu_wired = false;                                 // wires off: I2C is never started
  boot();
  CHECK(!imuOk && outHas("E,imu not found"));
  CHECK(outHas("E,imu not found (MPU-6050 on SDA 21 / SCL 22); wire check (1 = connected): SDA=0 SCL=0"));
  CHECK(outHas("I,SecureBot ESP32 fw 2.1 ready"));
  step(500);
  CHECK(!outHas("G,"));                                // no G lines, everything else runs
  send("M,100,100\n"); step(5);
  CHECK(leftPWM() == 100 && rightPWM() == 100);
}
static void t_imu_wired_but_silent() {
  g_imu_present = false;                               // wires there, chip does not answer
  boot();
  CHECK(!imuOk && outHas("wire check (1 = connected): SDA=1 SCL=1"));
  CHECK(outHas("E,imu i2c devices"));
  CHECK(outHas("I,SecureBot ESP32 fw 2.1 ready"));
}
static void t_imu_slow_to_wake() {
  // the MPU-6050 NACKs for a while after power-on: setup must keep trying
  g_imu_present = false;
  g_wake_after_writes = 6;
  boot();
  CHECK(imuOk && outHas("I,imu ready"));
}
static void t_imu_lost() {
  boot();
  CHECK(imuOk);
  g_imu_present = false;                               // wire comes off while running
  step(500);
  CHECK(!imuOk && outHas("E,imu stopped answering"));
  send("M,100,100\n"); step(5);
  CHECK(leftPWM() == 100);
}

static void t_brakes_while_stopped() {
  boot();
  CHECK(braked(LEFT_IN_A, LEFT_IN_B) && braked(RIGHT_IN_A, RIGHT_IN_B));   // from boot
  send("M,300,-300\n"); step(5);                      // spinning on the spot
  CHECK(leftPWM() == 300 && rightPWM() == -300);
  send("M,0,0\n"); step(5);
  CHECK(braked(LEFT_IN_A, LEFT_IN_B) && braked(RIGHT_IN_A, RIGHT_IN_B));
  CHECK(leftPWM() == 0 && rightPWM() == 0);
  stepHB(1000, "M,0,0");                               // held, not released to coast
  CHECK(braked(LEFT_IN_A, LEFT_IN_B) && braked(RIGHT_IN_A, RIGHT_IN_B));
  send("M,100,100\n"); step(5);                       // driving again releases the brake
  CHECK(leftPWM() == 100 && rightPWM() == 100);
}

static void t_encoders() {
  boot();
  CHECK(outHas("I,encoders idle levels"));
  CHECK(g_isr[LEFT_ENCODER_BLUE] && g_isr[LEFT_ENCODER_WHITE] && g_isr[RIGHT_ENCODER_BLUE] && g_isr[RIGHT_ENCODER_WHITE]);
  turnEncoder(LEFT_ENCODER_BLUE, LEFT_ENCODER_WHITE, 100);
  turnEncoder(RIGHT_ENCODER_BLUE, RIGHT_ENCODER_WHITE, -40);
  CHECK(leftTicks == 100 && rightTicks == -40);       // direction comes from the channel order
  turnEncoder(LEFT_ENCODER_BLUE, LEFT_ENCODER_WHITE, -30);
  CHECK(leftTicks == 70);
  // W reports the robot's wheels: counters swapped, the right one reversed, then the front
  // flipped (robot left = -(physical right) = -(-(70)), robot right = -(physical left) = 40).
  long l, r; readWheelTicks(l, r);
  CHECK(l == 70 && r == 40);
  g_out.clear(); step(60);
  CHECK(outHas("W,70,40"));
}

// Robot-frame encoder pins: turning these by `steps` moves that robot wheel forward by
// `steps` ticks (inverse of readWheelTicks for this wiring).
static void turnRobotWheel(int m, int steps) {
  // robot left  = -(physical right) ; physical right = -(counter on LEFT_* pins) (swapped, reversed)
  //   -> robot left forward = LEFT_* counter forward
  // robot right = -(physical left)  ; physical left  = +(counter on RIGHT_* pins)
  //   -> robot right forward = RIGHT_* counter backward
  if (m == 0) turnEncoder(LEFT_ENCODER_BLUE, LEFT_ENCODER_WHITE, steps);
  else turnEncoder(RIGHT_ENCODER_BLUE, RIGHT_ENCODER_WHITE, -steps);
}
static void t_robot_wheel_helper() {          // guards turnRobotWheel against wiring changes
  boot();
  turnRobotWheel(0, 25); turnRobotWheel(1, -12);
  long l, r; readWheelTicks(l, r);
  CHECK(l == 25 && r == -12);
}

// V command: a simulated wheel whose speed follows the PWM above a friction threshold.
static void stepWheels(int ms, const std::string &cmd, float tps_per_pwm[2], float thresh = 230) {
  float acc[2] = {0, 0};
  for (int i = 0; i < ms; i++) {
    if (g_ms % 100 == 0) send(cmd + "\n");
    g_ms++; loop();
    int pwm[2] = {leftPWM(), rightPWM()};
    for (int m = 0; m < 2; m++) {
      float mag = fabs((float)pwm[m]) - thresh;
      float tps = mag > 0 ? (pwm[m] > 0 ? 1 : -1) * mag * tps_per_pwm[m] : 0;
      acc[m] += tps / 1000.0f;
      int whole = (int)acc[m];
      if (whole) { turnRobotWheel(m, whole); acc[m] -= whole; }
    }
  }
}
static void t_speed_command_tracks() {
  boot();
  float gain[2] = {1.33f, 0.9f};               // the two motors differ a lot
  stepWheels(3000, "V,300,300", gain);
  CHECK(mode == MODE_SPEED);
  CHECK(fabs(measuredTps[0] - 300) < 30 && fabs(measuredTps[1] - 300) < 30);
  CHECK(leftPWM() != rightPWM());              // each wheel got its own PWM
  CHECK(leftPWM() <= MAX_PWM && rightPWM() <= MAX_PWM);
  stepWheels(3000, "V,-200,200", gain);        // spin in place
  CHECK(fabs(measuredTps[0] + 200) < 30 && fabs(measuredTps[1] - 200) < 30);
  stepWheels(200, "V,0,0", gain);
  CHECK(braked(LEFT_IN_A, LEFT_IN_B) && braked(RIGHT_IN_A, RIGHT_IN_B));
}
static void t_speed_command_kick_and_clamp() {
  boot();
  send("V,50,0\n"); step(SPEED_PERIOD_MS + 2);  // wheel standing: kicked past static friction
  CHECK(leftPWM() >= (int)SPEED_KICK_PWM && rightPWM() == 0);
  send("V,99999,-99999\n"); step(SPEED_PERIOD_MS + 2);
  CHECK(targetTps[0] == MAX_SPEED_TPS && targetTps[1] == -MAX_SPEED_TPS);
  CHECK(abs(leftPWM()) <= MAX_PWM && abs(rightPWM()) <= MAX_PWM);
  send("V,abc\n"); step(5);
  CHECK(outHas("E,bad command: V,abc"));
}
static void t_speed_command_watchdog_and_gate() {
  boot();
  send("V,300,300\n"); step(300);              // no heartbeat
  CHECK(leftPWM() == 0 && rightPWM() == 0 && outHas("E,watchdog"));
  g_echo_us[ULTRA_LEFT_ECHO] = ECHO_HALF_FT; g_echo_us[ULTRA_RIGHT_ECHO] = ECHO_HALF_FT;
  stepHB(300, "V,300,300");
  CHECK(leftPWM() == 0 && rightPWM() == 0);    // net forward into an obstacle: blocked
  stepHB(300, "V,-300,300");
  CHECK(leftPWM() < 0 && rightPWM() > 0);      // turning on the spot still allowed
}

static void t_imu_rejects_bad_calibration() {
  // a gyro reading 41 deg/s at rest (seen once after waking): never used as the bias
  g_gyro_z_raw = (int16_t)(41 * 65.5);
  boot();
  CHECK(outHas("E,imu calibration rejected (try 1)"));
  CHECK(outHas("E,imu calibration failed"));
  CHECK(!imuOk);
  step(200);
  CHECK(!outHas("G,"));
  send("M,100,100\n"); step(5);
  CHECK(leftPWM() == 100);                     // the robot still drives without the gyro
}

static void t_encoder_scope() {
  boot();
  send("M,300,300\n"); step(5);
  for (int i = 0; i < 64; i++) g_mv[i] = 3300;
  g_mv[LEFT_ENCODER_BLUE] = 200;                    // a line that does go low
  send("encscope\n"); step(1);
  CHECK(leftPWM() == 0 && rightPWM() == 0);          // motors stopped for the measurement
  CHECK(outHas("I,encscope GPIO34 min/avg/max mV 200/200/200 low 100%"));
  CHECK(outHas("I,encscope GPIO35 min/avg/max mV 3300/3300/3300 low 0%"));
}

struct T { const char *name; void (*fn)(); };
int main(int argc, char **argv) {
  T tests[] = {
    {"boot", t_boot}, {"pinout", t_pinout}, {"polarity", t_polarity}, {"pwm_setup", t_pwm_setup}, {"freq_command", t_freq_command}, {"drive_and_clamp", t_drive_and_clamp}, {"speed_cap", t_speed_cap}, {"output_stage_cap", t_output_stage_cap},
    {"watchdog", t_watchdog}, {"heartbeat_keeps_running", t_heartbeat_keeps_running},
    {"partial_line", t_partial_line}, {"bad_input", t_bad_input},
    {"bad_command_not_heartbeat", t_bad_command_not_heartbeat},
    {"drive_obstacle_gate", t_drive_obstacle_gate}, {"single_glitch_ignored", t_single_glitch_ignored},
    {"patrol_avoidance_cycle", t_patrol_avoidance_cycle}, {"avoid_timeout", t_avoid_timeout},
    {"telemetry", t_telemetry},
    {"imu_setup", t_imu_setup}, {"imu_yaw_integrates", t_imu_yaw_integrates},
    {"imu_bias_tracks_when_still", t_imu_bias_tracks_when_still}, {"imu_absent", t_imu_absent},
    {"imu_lost", t_imu_lost}, {"imu_wired_but_silent", t_imu_wired_but_silent}, {"imu_slow_to_wake", t_imu_slow_to_wake}, {"brakes_while_stopped", t_brakes_while_stopped}, {"encoders", t_encoders}, {"encoder_scope", t_encoder_scope},
    {"robot_wheel_helper", t_robot_wheel_helper}, {"speed_command_tracks", t_speed_command_tracks},
    {"speed_command_kick_and_clamp", t_speed_command_kick_and_clamp},
    {"speed_command_watchdog_and_gate", t_speed_command_watchdog_and_gate},
    {"imu_rejects_bad_calibration", t_imu_rejects_bad_calibration}};
  if (argc < 2) { for (auto &t : tests) printf("%s\n", t.name); return 0; }
  for (auto &t : tests) if (!strcmp(t.name, argv[1])) { t.fn(); return failures ? 1 : 0; }
  printf("unknown test %s\n", argv[1]); return 2;
}
