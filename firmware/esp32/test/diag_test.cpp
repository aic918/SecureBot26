// Host-side tests for diag_ultrasonic.ino using a simulated HC-SR04 on chosen pins.
#include <Arduino.h>
#include <set>
#include <string>
#include "../diag_ultrasonic/diag_ultrasonic.ino"

// ---------- simulated hardware ----------
static unsigned long g_ms = 0;
static std::string g_in, g_out;
static size_t g_in_pos = 0;
static int g_mode[64], g_level[64];
static std::set<int> g_touched;                 // every pin the sketch configured or drove
static struct { void (*fn)(void *); void *arg; } g_isr[64];
static int g_trigMax[64];                       // did a pin ever go HIGH?
// the simulated sensor
static int simTrig = -1, simEcho = -1;
static bool objectPresent = false;
static long simUs = 1200;                       // ~20.6 cm
static bool echoReady = false;

unsigned long millis() { return g_ms; }
void delay(unsigned long ms) { g_ms += ms; }
void delayMicroseconds(unsigned int) {}
void pinMode(int pin, int m) { g_mode[pin] = m; g_touched.insert(pin); }
void analogWrite(int, int) {}
void attachInterruptArg(uint8_t pin, void (*fn)(void *), void *arg, int) { g_isr[pin] = {fn, arg}; }
void digitalWrite(int pin, int v) {
  g_touched.insert(pin);
  int prev = g_level[pin];
  g_level[pin] = v;
  if (v) g_trigMax[pin] = 1;
  if (pin == simTrig && prev == HIGH && v == LOW && objectPresent) {
    echoReady = true;
    if (g_mode[simEcho] != OUTPUT && g_isr[simEcho].fn) {      // echo pin rises then falls
      g_isr[simEcho].fn(g_isr[simEcho].arg);
      g_isr[simEcho].fn(g_isr[simEcho].arg);
    }
  }
}
long pulseIn(int pin, int, unsigned long) {
  if (pin == simEcho && echoReady && g_mode[pin] != OUTPUT) { echoReady = false; return simUs; }
  return 0;
}
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
static bool has(const char *s) { return g_out.find(s) != std::string::npos; }
static void wire(int trig, int echo, bool object) { simTrig = trig; simEcho = echo; objectPresent = object; }
static void cmd(const std::string &c) { g_in += c + "\n"; readCommands(); }

// ---------- scenarios ----------
static void t_normal_wiring() {
  setup(); wire(LEFT_TRIG, LEFT_ECHO, true);
  pairsReport();
  CHECK(has("LEFT  normal  (TRIG 23, ECHO 19): 20/20 echoes, 20.6 to 20.6 cm"));
  CHECK(has("LEFT  swapped (TRIG 19, ECHO 23): 0/20 echoes"));
  CHECK(has("RIGHT normal  (TRIG 4, ECHO 17): 0/20 echoes"));
}
static void t_right_sensor() {
  setup(); wire(RIGHT_TRIG, RIGHT_ECHO, true);
  pairsReport();
  CHECK(has("RIGHT normal  (TRIG 4, ECHO 17): 20/20 echoes"));
  CHECK(has("LEFT  normal  (TRIG 23, ECHO 19): 0/20 echoes"));
}
static void t_swapped_wires() {
  setup(); wire(LEFT_ECHO, LEFT_TRIG, true);   // sensor TRIG wire is on GPIO 19, ECHO wire on 23
  pairsReport();
  CHECK(has("LEFT  normal  (TRIG 23, ECHO 19): 0/20 echoes"));
  CHECK(has("LEFT  swapped (TRIG 19, ECHO 23): 20/20 echoes"));
}
static void t_no_object() {
  setup(); wire(LEFT_TRIG, LEFT_ECHO, false);
  pairsReport(); scanAll();
  CHECK(has("LEFT  normal  (TRIG 23, ECHO 19): 0/20 echoes"));
  CHECK(!has("echo seen"));
}
static void t_scan_finds_unexpected_pins() {
  setup(); wire(21, 22, true);                  // sensor wired to pins the firmware does not expect
  scanAll();
  CHECK(has("<-- echo seen: TRIG GPIO 21 -> ECHO GPIO 22"));
  CHECK(!has("TRIG GPIO 23 ->") && !has("TRIG GPIO 4 ->"));
}
static void t_scan_finds_configured_pins() {
  setup(); wire(RIGHT_TRIG, RIGHT_ECHO, true);
  scanAll();
  CHECK(has("<-- echo seen: TRIG GPIO 4 -> ECHO GPIO 17"));
}
static void t_hold_commands() {
  setup();
  cmd("hold left");
  CHECK(g_trigMax[LEFT_TRIG] == 1 && g_level[LEFT_TRIG] == LOW);   // went HIGH, ended LOW
  CHECK(has("LEFT TRIG (GPIO 23) held HIGH for 5 s"));
  cmd("hold right");
  CHECK(g_trigMax[RIGHT_TRIG] == 1 && g_level[RIGHT_TRIG] == LOW);
  cmd("bogus");
  CHECK(has("E,unknown command: bogus"));
}
static void t_safe_pins_only() {
  setup(); wire(LEFT_TRIG, LEFT_ECHO, true);
  pairsReport(); scanAll(); cmd("hold left"); cmd("hold right");
  // never motors, encoders, strapping, flash or USB-serial pins
  const int forbidden[] = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25, 26, 27, 32, 33, 34, 35};
  for (int pin : g_touched) for (int f : forbidden) CHECK(pin != f);
  CHECK(g_touched.count(18) == 1);              // listened on...
  CHECK(g_trigMax[18] == 0);                    // ...but never driven HIGH
}

struct T { const char *name; void (*fn)(); };
int main(int argc, char **argv) {
  T tests[] = {{"normal_wiring", t_normal_wiring}, {"right_sensor", t_right_sensor},
               {"swapped_wires", t_swapped_wires}, {"no_object", t_no_object},
               {"scan_finds_unexpected_pins", t_scan_finds_unexpected_pins},
               {"scan_finds_configured_pins", t_scan_finds_configured_pins},
               {"hold_commands", t_hold_commands}, {"safe_pins_only", t_safe_pins_only}};
  if (argc < 2) { for (auto &t : tests) printf("%s\n", t.name); return 0; }
  for (auto &t : tests) if (!strcmp(t.name, argv[1])) { t.fn(); if (failures) printf("--- output ---\n%s\n", g_out.c_str()); return failures ? 1 : 0; }
  printf("unknown test %s\n", argv[1]); return 2;
}
