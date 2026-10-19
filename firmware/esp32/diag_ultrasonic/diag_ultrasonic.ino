// Ultrasonic wiring diagnostic for the two HC-SR04 sensors.
//
// THIS IS NOT THE ROBOT FIRMWARE. Flash it, read the output, then flash
// securebot_esp32 again. It does not touch the motor, encoder, boot-strapping
// (0, 2, 5, 12, 15), flash (6-11) or USB-serial (1, 3) pins.
//
// Serial, 115200 baud. Every 2 s it prints a "pairs" report for the configured wiring.
// Commands (type then Enter):
//   pairs       run the report now
//   scan        try every candidate pin as TRIG and show which other pins see echo edges
//   hold left   drive the LEFT TRIG pin HIGH for 5 s (meter should read ~3.3 V at the sensor TRIG pin)
//   hold right  same for the RIGHT TRIG pin
// Hold a flat, hard object (a book) 20-30 cm in front of ONE sensor while it runs.
#include <Arduino.h>

// ---- configured wiring (same as securebot_esp32) ----
const int LEFT_TRIG = 23;
const int LEFT_ECHO = 19;
const int RIGHT_TRIG = 4;
const int RIGHT_ECHO = 17;

// ---- pins used by `scan` (all free of motors, encoders, strapping, flash and UART) ----
// GPIO 18 only listens: the right encoder white wire may be connected there.
const int SCAN_PINS[] = {4, 16, 17, 18, 19, 21, 22, 23};
const bool SCAN_CAN_DRIVE[] = {true, true, true, false, true, true, true, true};
const int NUM_SCAN = sizeof(SCAN_PINS) / sizeof(SCAN_PINS[0]);

const int PINGS = 20;                 // pings per orientation in the report
const int SCAN_PINGS = 15;            // pings per TRIG candidate in the scan
const int PING_GAP_MS = 60;
const unsigned long REPORT_MS = 2000;
const float US_TO_CM = 0.0135 / 2.0 * 2.54;   // same speed-of-sound constant as the firmware

volatile uint32_t edges[NUM_SCAN];
unsigned long lastReportMs = 0;
char lineBuf[32];
size_t lineLen = 0;

void IRAM_ATTR onEdge(void *arg) {
  edges[(int)(intptr_t)arg]++;
}

// ---- prototypes ----
void pairsReport();
void reportOne(const char *name, int trig, int echo);
void scanAll();
void holdTrig(const char *name, int trig);
long ping(int trig, int echo);
void handleLine(const char *cmd);
void readCommands();

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM_SCAN; i++) {
    pinMode(SCAN_PINS[i], INPUT_PULLDOWN);
    attachInterruptArg(SCAN_PINS[i], onEdge, (void *)(intptr_t)i, CHANGE);
  }
  Serial.println("I,ultrasonic diagnostic ready (NOT the robot firmware)");
  Serial.println("I,hold a flat object 20-30 cm in front of ONE sensor; commands: pairs, scan, hold left, hold right");
}

void loop() {
  readCommands();
  if (millis() - lastReportMs >= REPORT_MS) {
    lastReportMs = millis();
    pairsReport();
  }
  delay(1);
}

// One TRIG pulse (same method as the robot firmware); returns echo time in us, 0 = none.
long ping(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  return pulseIn(echo, HIGH, 25000);
}

void reportOne(const char *name, int trig, int echo) {
  pinMode(trig, OUTPUT);
  digitalWrite(trig, LOW);
  pinMode(echo, INPUT_PULLDOWN);

  int hits = 0;
  float minCm = 1e9;
  float maxCm = 0;
  for (int i = 0; i < PINGS; i++) {
    long us = ping(trig, echo);
    if (us > 0) {
      float cm = us * US_TO_CM;
      hits++;
      if (cm < minCm) minCm = cm;
      if (cm > maxCm) maxCm = cm;
    }
    delay(PING_GAP_MS);
  }

  Serial.print(name);
  Serial.print(" (TRIG ");
  Serial.print(trig);
  Serial.print(", ECHO ");
  Serial.print(echo);
  Serial.print("): ");
  Serial.print(hits);
  Serial.print("/");
  Serial.print(PINGS);
  Serial.print(" echoes");
  if (hits > 0) {
    Serial.print(", ");
    Serial.print(minCm, 1);
    Serial.print(" to ");
    Serial.print(maxCm, 1);
    Serial.print(" cm");
  }
  Serial.println();

  pinMode(trig, INPUT_PULLDOWN);   // leave both pins as quiet inputs again
}

void pairsReport() {
  Serial.println("--- pairs report (configured wiring, then TRIG/ECHO swapped) ---");
  reportOne("LEFT  normal ", LEFT_TRIG, LEFT_ECHO);
  reportOne("LEFT  swapped", LEFT_ECHO, LEFT_TRIG);
  reportOne("RIGHT normal ", RIGHT_TRIG, RIGHT_ECHO);
  reportOne("RIGHT swapped", RIGHT_ECHO, RIGHT_TRIG);
}

// Drive each candidate pin as TRIG and count edges on every other candidate pin.
// A working sensor shows up as a row (its TRIG pin) with a big number in one column (its ECHO pin).
void scanAll() {
  Serial.println("--- scan: row = pin driven as TRIG, column = pin where echo edges were counted ---");
  Serial.print("TRIG\\ECHO");
  for (int j = 0; j < NUM_SCAN; j++) {
    Serial.print("  ");
    Serial.print(SCAN_PINS[j]);
  }
  Serial.println();

  for (int i = 0; i < NUM_SCAN; i++) {
    if (!SCAN_CAN_DRIVE[i]) continue;

    for (int j = 0; j < NUM_SCAN; j++) {
      pinMode(SCAN_PINS[j], j == i ? OUTPUT : INPUT_PULLDOWN);
      // re-attach: some ESP32 core versions drop the interrupt when the pin mode changes
      if (j != i) attachInterruptArg(SCAN_PINS[j], onEdge, (void *)(intptr_t)j, CHANGE);
    }
    digitalWrite(SCAN_PINS[i], LOW);
    delay(5);
    for (int j = 0; j < NUM_SCAN; j++) edges[j] = 0;

    for (int k = 0; k < SCAN_PINGS; k++) {
      digitalWrite(SCAN_PINS[i], HIGH);
      delayMicroseconds(10);
      digitalWrite(SCAN_PINS[i], LOW);
      delay(PING_GAP_MS);
    }

    Serial.print("   ");
    Serial.print(SCAN_PINS[i]);
    Serial.print("     ");
    int best = -1;
    int bestEdges = 0;
    for (int j = 0; j < NUM_SCAN; j++) {
      Serial.print("  ");
      if (j == i) {
        Serial.print("-");
      } else {
        int e = (int)edges[j];
        Serial.print(e);
        if (e > bestEdges) { bestEdges = e; best = j; }
      }
    }
    // A sensor gives 2 edges per ping (rise + fall): about 2 x SCAN_PINGS when an object is in front.
    if (best >= 0 && bestEdges >= SCAN_PINGS) {
      Serial.print("   <-- echo seen: TRIG GPIO ");
      Serial.print(SCAN_PINS[i]);
      Serial.print(" -> ECHO GPIO ");
      Serial.print(SCAN_PINS[best]);
    }
    Serial.println();
  }

  for (int j = 0; j < NUM_SCAN; j++) pinMode(SCAN_PINS[j], INPUT_PULLDOWN);
  Serial.println("--- scan done ---");
}

void holdTrig(const char *name, int trig) {
  pinMode(trig, OUTPUT);
  digitalWrite(trig, HIGH);
  Serial.print(name);
  Serial.print(" TRIG (GPIO ");
  Serial.print(trig);
  Serial.println(") held HIGH for 5 s: measure ~3.3 V at the sensor's TRIG pin now");
  delay(5000);
  digitalWrite(trig, LOW);
  pinMode(trig, INPUT_PULLDOWN);
  Serial.println("hold done");
}

void handleLine(const char *cmd) {
  if (strcmp(cmd, "pairs") == 0) {
    pairsReport();
  } else if (strcmp(cmd, "scan") == 0) {
    scanAll();
  } else if (strcmp(cmd, "hold left") == 0) {
    holdTrig("LEFT", LEFT_TRIG);
  } else if (strcmp(cmd, "hold right") == 0) {
    holdTrig("RIGHT", RIGHT_TRIG);
  } else {
    Serial.print("E,unknown command: ");
    Serial.println(cmd);
  }
}

void readCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      if (lineLen > 0) handleLine(lineBuf);
      lineLen = 0;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }
}
