# ESP32 firmware (v2.0)

`securebot_esp32/securebot_esp32.ino` is the low-level controller (Arduino
framework): L298N-style motor driver and two HC-SR04 ultrasonics. The original
v1 firmware is preserved in git history (commit "S0-4").

## What changed from v1
- **PWM cap** (`MAX_PWM = 150`) enforced in one output function; applies to every command.
- **Watchdog:** no valid command for 500 ms -> motors stop. The sender must resend (~every 200 ms).
- **Non-blocking serial** line parser (v1 `readStringUntil` could block up to 1 s). Bad lines are
  rejected with `E,...` and do not feed the watchdog.
- **`M,<left>,<right>`** signed-PWM command, matching `esp32_serial_bridge.py` (`/cmd_vel`).
  Legacy text commands still work so `securebot_live_node.py` keeps running.
- **Machine-readable telemetry:** `U,<upper_ft>,<lower_ft>` at 10 Hz (`-1` = no reading), `S,...` status.
- Ultrasonics sampled one at a time (no cross-talk, shorter blocking), a close reading must repeat
  twice before it counts, and `M` forward motion is blocked near an obstacle (rotate/reverse allowed).
- Avoidance sequence unchanged, but the rotation now times out after 4 s instead of spinning forever.
- No `String` objects (no heap fragmentation); motors are zeroed first thing in `setup()`.

## Serial protocol (115200 baud, newline-terminated)
| Direction | Message | Meaning |
|---|---|---|
| to ESP32 | `M,<l>,<r>` | wheel PWM, -150..150 |
| to ESP32 | `stop` `forward` `patrol` `backward` `left` `right` | legacy modes |
| to ESP32 | `speed <n>` | legacy-mode PWM, clamped 0..150 |
| from ESP32 | `U,<upper_ft>,<lower_ft>` | distances in feet, `-1` = none |
| from ESP32 | `S,<mode>,<l>,<r>,<avoidState>` | status, 2 Hz |
| from ESP32 | `I,<text>` / `E,<text>` | info / safety event |

## Pins
Left motor ENA 25, IN1 26, IN2 27 · Right motor ENB 14, IN3 12, IN4 13
Upper ultrasonic TRIG 5 / ECHO 18 · Lower ultrasonic TRIG 17 / ECHO 16
Encoders (defined, unused): R 33/32, L 35/34.
Note: GPIO 12 (IN3) is a boot-strapping pin; if the board sometimes fails to boot, move it.

## Tests (run on the Jetson/PC, no hardware)
`./test/run_tests.sh` compiles the sketch against an Arduino stand-in (`test/shim`) and runs 14
scenarios: PWM cap, watchdog, heartbeat, partial/oversized/garbage lines, obstacle gate,
glitch rejection, full avoidance cycle, avoidance timeout, telemetry format.
This checks the logic only; it does not replace testing on the robot.

## Flashing
Arduino IDE: board "ESP32 Dev Module", open the sketch folder, Upload. Close anything else holding
the serial port first. **First test with the wheels off the ground**: send `M,80,80` and confirm
the wheels spin, then stop sending and confirm they stop within ~0.5 s.
