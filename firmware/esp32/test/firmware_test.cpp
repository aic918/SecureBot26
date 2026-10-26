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
void delay(unsigned long) {}
void delayMicroseconds(unsigned int) {}
long pulseIn(int pin, int, unsigned long) { return g_echo_us[pin]; }
void pinMode(int, int) {}
void digitalWrite(int pin, int v) { g_pin[pin] = v; }
void analogWrite(int pin, int v) { g_pwm[pin] = v; }
static int g_chan_pin[16];
static int g_chan_bits[16];
static double g_chan_freq[16];
double ledcSetup(int ch, double f, int bits) { g_chan_freq[ch] = f; g_chan_bits[ch] = bits; return f; }
void ledcAttachPin(int pin, int ch) { g_chan_pin[ch] = pin; }
void ledcWrite(int ch, int duty) { g_pwm[g_chan_pin[ch]] = duty; }  // recorded by EN pin
SerialClass Serial;
int SerialClass::available() { return (int)(g_in.size() - g_in_pos); }
int SerialClass::read() { return g_in_pos < g_in.size() ? (unsigned char)g_in[g_in_pos++] : -1; }
void SerialClass::print(const char *s) { g_out += s; }
void SerialClass::print(char c) { g_out += c; }
void SerialClass::print(int v) { g_out += std::to_string(v); }
void SerialClass::print(unsigned long v) { g_out += std::to_string(v); }
void SerialClass::print(double v, int d) { char b[32]; snprintf(b, sizeof b, "%.*f", d, v); g_out += b; }
void SerialClass::println() { g_out += "\n"; }
void SerialClass::println(const char *s) { g_out += s; g_out += "\n"; }
void SerialClass::println(int v) { print(v); g_out += "\n"; }
void SerialClass::println(double v, int d) { print(v, d); g_out += "\n"; }

// ---------- helpers ----------
static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("  FAIL line %d: %s\n", __LINE__, #cond); failures++; } } while (0)

static int leftPWM() {
  int v = 0;
  if (g_pin[LEFT_IN_A] == HIGH && g_pin[LEFT_IN_B] == LOW) v = g_pwm[LEFT_EN];
  if (g_pin[LEFT_IN_A] == LOW && g_pin[LEFT_IN_B] == HIGH) v = -g_pwm[LEFT_EN];
  return LEFT_REVERSED ? -v : v;   // report the logical command
}
static int rightPWM() {
  int v = 0;
  if (g_pin[RIGHT_IN_A] == HIGH && g_pin[RIGHT_IN_B] == LOW) v = g_pwm[RIGHT_EN];
  if (g_pin[RIGHT_IN_A] == LOW && g_pin[RIGHT_IN_B] == HIGH) v = -g_pwm[RIGHT_EN];
  return RIGHT_REVERSED ? -v : v;   // report the logical command
}
static void send(const std::string &s) { g_in += s; }
static void step(int ms) { for (int i = 0; i < ms; i++) { g_ms++; loop(); } }
// step while resending `cmd` every 100 ms like the Jetson's bridge does
static void stepHB(int ms, const std::string &cmd) {
  for (int i = 0; i < ms; i++) { if (g_ms % 100 == 0) send(cmd + "\n"); g_ms++; loop(); }
}
static bool outHas(const char *s) { return g_out.find(s) != std::string::npos; }
static void boot() { memset(g_pin, 0, sizeof g_pin); memset(g_pwm, 0, sizeof g_pwm); setup(); }
const long ECHO_HALF_FT = 889;    // ~0.5 ft: inside stop distance
const long ECHO_2FT = 3556;       // ~2.0 ft: clear

// ---------- scenarios ----------
static void t_boot() {
  boot(); step(10);
  CHECK(leftPWM() == 0 && rightPWM() == 0);
  CHECK(outHas("I,SecureBot ESP32 fw 2.0 ready"));
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
  CHECK(g_pwm[LEFT_EN] <= MAX_PWM && g_pwm[RIGHT_EN] <= MAX_PWM);
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
  int pins[] = {LEFT_EN, LEFT_IN_A, LEFT_IN_B, RIGHT_EN, RIGHT_IN_A, RIGHT_IN_B,
                ULTRA_LEFT_TRIG, ULTRA_LEFT_ECHO, ULTRA_RIGHT_TRIG, ULTRA_RIGHT_ECHO,
                LEFT_ENCODER_BLUE, LEFT_ENCODER_WHITE, RIGHT_ENCODER_BLUE, RIGHT_ENCODER_WHITE};
  int n = sizeof pins / sizeof pins[0];
  for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) CHECK(pins[i] != pins[j]);
  CHECK(LEFT_EN == 27 && LEFT_IN_A == 32 && LEFT_IN_B == 14);
  CHECK(RIGHT_EN == 26 && RIGHT_IN_A == 25 && RIGHT_IN_B == 33);
  // No input/output may sit on a boot-strapping, flash or USB-serial pin.
  const int forbidden[] = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15};
  for (int i = 0; i < n; i++) for (int f : forbidden) CHECK(pins[i] != f);
  CHECK(RIGHT_ENCODER_WHITE == 18);
  CHECK(ULTRA_LEFT_TRIG == 23 && ULTRA_LEFT_ECHO == 19 && ULTRA_RIGHT_TRIG == 4 && ULTRA_RIGHT_ECHO == 17);
  boot();
  send("M,120,-70\n"); step(5);                   // left forward 120, right reverse 70
  CHECK(g_pwm[LEFT_EN] == 120);     // direction pins depend on the polarity flags
  CHECK(g_pin[LEFT_IN_A] == (LEFT_REVERSED ? LOW : HIGH) && g_pin[LEFT_IN_B] == (LEFT_REVERSED ? HIGH : LOW));
  CHECK(g_pwm[RIGHT_EN] == 70);     // logical reverse
  CHECK(g_pin[RIGHT_IN_A] == (RIGHT_REVERSED ? HIGH : LOW) && g_pin[RIGHT_IN_B] == (RIGHT_REVERSED ? LOW : HIGH));
}

static void t_polarity() {
  // Verified with the senior1typetest sketch: forward = right IN1 (RIGHT_IN_A) high,
  // left IN3 (LEFT_IN_A) low.
  CHECK(LEFT_REVERSED && !RIGHT_REVERSED);
  boot();
  send("M,100,100\n"); step(5);                  // both wheels forward
  CHECK(g_pin[LEFT_IN_A] == LOW && g_pin[LEFT_IN_B] == HIGH && g_pwm[LEFT_EN] == 100);
  CHECK(g_pin[RIGHT_IN_A] == HIGH && g_pin[RIGHT_IN_B] == LOW && g_pwm[RIGHT_EN] == 100);
  send("M,-100,-100\n"); step(5);                // both wheels backward
  CHECK(g_pin[LEFT_IN_A] == HIGH && g_pin[LEFT_IN_B] == LOW);
  CHECK(g_pin[RIGHT_IN_A] == LOW && g_pin[RIGHT_IN_B] == HIGH);
  send("stop\n"); step(5);                       // stopped: both inputs low
  CHECK(g_pin[LEFT_IN_A] == LOW && g_pin[LEFT_IN_B] == LOW && g_pwm[LEFT_EN] == 0);
  CHECK(g_pin[RIGHT_IN_A] == LOW && g_pin[RIGHT_IN_B] == LOW && g_pwm[RIGHT_EN] == 0);
}

static void t_pwm_setup() {
  // 10-bit duty at 10 kHz on both enable pins, like the senior1typetest sketch.
  boot();
  CHECK(g_chan_pin[LEFT_PWM_CHANNEL] == LEFT_EN && g_chan_pin[RIGHT_PWM_CHANNEL] == RIGHT_EN);
  CHECK(LEFT_PWM_CHANNEL != RIGHT_PWM_CHANNEL);
  CHECK(g_chan_bits[LEFT_PWM_CHANNEL] == 10 && g_chan_bits[RIGHT_PWM_CHANNEL] == 10);
  CHECK(g_chan_freq[LEFT_PWM_CHANNEL] == 10000 && g_chan_freq[RIGHT_PWM_CHANNEL] == 10000);
  CHECK(MAX_PWM <= 1023 && MAX_PWM == 655);
  send("M,655,655\n"); step(5);
  CHECK(g_pwm[LEFT_EN] == 655 && g_pwm[RIGHT_EN] == 655);
}

struct T { const char *name; void (*fn)(); };
int main(int argc, char **argv) {
  T tests[] = {
    {"boot", t_boot}, {"pinout", t_pinout}, {"polarity", t_polarity}, {"pwm_setup", t_pwm_setup}, {"drive_and_clamp", t_drive_and_clamp}, {"speed_cap", t_speed_cap}, {"output_stage_cap", t_output_stage_cap},
    {"watchdog", t_watchdog}, {"heartbeat_keeps_running", t_heartbeat_keeps_running},
    {"partial_line", t_partial_line}, {"bad_input", t_bad_input},
    {"bad_command_not_heartbeat", t_bad_command_not_heartbeat},
    {"drive_obstacle_gate", t_drive_obstacle_gate}, {"single_glitch_ignored", t_single_glitch_ignored},
    {"patrol_avoidance_cycle", t_patrol_avoidance_cycle}, {"avoid_timeout", t_avoid_timeout},
    {"telemetry", t_telemetry}};
  if (argc < 2) { for (auto &t : tests) printf("%s\n", t.name); return 0; }
  for (auto &t : tests) if (!strcmp(t.name, argv[1])) { t.fn(); return failures ? 1 : 0; }
  printf("unknown test %s\n", argv[1]); return 2;
}
