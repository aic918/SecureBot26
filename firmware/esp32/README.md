# ESP32 firmware

`securebot_esp32/securebot_esp32.ino` is the low-level controller (Arduino
framework): L298N-style motor driver, two HC-SR04 ultrasonics, and the current
"straight, then back up and turn" obstacle avoidance.

Imported unmodified from the working firmware (originally `message.c`). The
`.ino` extension and matching folder name are required for the Arduino IDE.

## Serial protocol (115200 baud, newline-terminated)
Jetson -> ESP32: `forward`, `patrol`, `backward`, `left`, `right`, `stop`, `speed <0-255>`
ESP32 -> Jetson: `Mode: <mode>`, `Speed set to <n>`, and every 500 ms
`Upper sensor: <ft> ft | Lower sensor: <ft> ft` (or `No reading`).

## Pins
Left motor ENA 25, IN1 26, IN2 27 · Right motor ENB 14, IN3 12, IN4 13
Upper ultrasonic TRIG 5 / ECHO 18 · Lower ultrasonic TRIG 17 / ECHO 16
Encoders (defined, unused): R 33/32, L 35/34

## Flashing
Arduino IDE: board "ESP32 Dev Module", open the sketch folder, Upload.
Close anything else holding the serial port first.
