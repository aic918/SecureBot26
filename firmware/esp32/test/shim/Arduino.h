// Minimal Arduino stand-in so the firmware logic can be compiled and tested on a PC.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLDOWN 9
#define INPUT_PULLUP 5
#define OUTPUT_OPEN_DRAIN 0x13
#define CHANGE 3
#define IRAM_ATTR
#define constrain(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define min(a, b) ((a) < (b) ? (a) : (b))

unsigned long millis();
unsigned long micros();
void delay(unsigned long ms);
void delayMicroseconds(unsigned int us);
long pulseIn(int pin, int state, unsigned long timeout);
void pinMode(int pin, int mode);
void digitalWrite(int pin, int value);
int digitalRead(int pin);
int analogReadMilliVolts(int pin);
#define max(a, b) ((a) > (b) ? (a) : (b))
int digitalPinToInterrupt(int pin);
void attachInterrupt(int irq, void (*isr)(), int mode);
void interrupts();
void noInterrupts();
void analogWrite(int pin, int value);
// ESP32 LEDC (Arduino-ESP32 2.x API)
double ledcSetup(int channel, double freq, int resolution_bits);
void ledcAttachPin(int pin, int channel);
void ledcWrite(int channel, int duty);
uint32_t ledcChangeFrequency(int channel, uint32_t freq, int resolution_bits);

struct SerialClass {
  void begin(long) {}
  int available();
  int read();
  void print(const char *s);
  void print(char c);
  void print(int v);
  void print(unsigned long v);
  void print(long v);
  void print(double v, int digits = 2);
  void println();
  void println(const char *s);
  void println(int v);
  void println(long v);
  void println(double v, int digits = 2);
};
extern SerialClass Serial;
