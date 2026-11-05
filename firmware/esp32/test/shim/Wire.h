// Minimal Arduino Wire (I2C) stand-in; the test file provides the fake device behind it.
#pragma once
#include <stdint.h>

struct TwoWire {
  bool begin(int sda, int scl);
  void setClock(uint32_t hz);
  void setTimeOut(uint16_t ms);
  void beginTransmission(uint8_t addr);
  size_t write(uint8_t b);
  uint8_t endTransmission(bool stop = true);
  uint8_t requestFrom(int addr, int n);
  int available();
  int read();
};
extern TwoWire Wire;
