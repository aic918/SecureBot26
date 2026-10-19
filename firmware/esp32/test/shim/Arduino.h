// Minimal Arduino stand-in so the firmware logic can be compiled and tested on a PC.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLDOWN 3
#define CHANGE 2
#define IRAM_ATTR
#define constrain(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define min(a, b) ((a) < (b) ? (a) : (b))

unsigned long millis();
void delay(unsigned long ms);
void delayMicroseconds(unsigned int us);
long pulseIn(int pin, int state, unsigned long timeout);
void pinMode(int pin, int mode);
void digitalWrite(int pin, int value);
void analogWrite(int pin, int value);
void attachInterruptArg(uint8_t pin, void (*fn)(void *), void *arg, int mode);

struct SerialClass {
  void begin(long) {}
  int available();
  int read();
  void print(const char *s);
  void print(char c);
  void print(int v);
  void print(unsigned long v);
  void print(double v, int digits = 2);
  void println();
  void println(const char *s);
  void println(int v);
  void println(double v, int digits = 2);
};
extern SerialClass Serial;
