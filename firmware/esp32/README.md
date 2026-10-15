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
- **Machine-readable telemetry:** `U,<left_ft>,<right_ft>` at 10 Hz (`-1` = no reading), `S,...` status.
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
| from ESP32 | `U,<left_ft>,<right_ft>` | distances in feet, `-1` = none |
| from ESP32 | `S,<mode>,<l>,<r>,<avoidState>` | status, 2 Hz |
| from ESP32 | `I,<text>` / `E,<text>` | info / safety event |

## Pins (current wiring)
| Function | GPIO |
|---|---|
| Right motor: EN / IN1 / IN2 (driver channel A) | 26 / 25 / 33 |
| Left motor: EN / IN3 / IN4 (driver channel B) | 27 / 32 / 14 |
| Ultrasonic 1 (left): TRIG / ECHO | 23 / 19 |
| Ultrasonic 2 (right): TRIG / ECHO | 4 / 17 |
| Right encoder: blue / white (unused yet) | 13 / 12 |
| Left encoder: blue / white (unused yet) | 34 / 35 |

Motor polarity: on both motors the black wire is on the driver's first output (OUT1 right,
OUT3 left) and red on the second (OUT2 / OUT4), and red is the forward direction. So
`LEFT_REVERSED` and `RIGHT_REVERSED` are both `true`. If a wheel turns the wrong way for a
positive command, flip its flag at the top of the sketch instead of rewiring.

### Hardware cautions
- **GPIO 12 (right encoder white) is a boot-strapping pin.** If the encoder output is HIGH while the
  ESP32 resets, many modules fail to boot ("flash read err"). If you see random boot failures,
  unplug that wire and retry, or move it to another pin (e.g. 36/39 are input-only and safe).
- **GPIO 34 and 35 have no internal pull-up.** Open-collector encoder outputs need an external
  pull-up (about 10 k to 3.3 V) on those two lines.
- **GPIO 14 pulses at boot** on some boards, so the left motor may twitch once at power-up.
- **HC-SR04 ECHO is a 5 V output.** ESP32 pins are 3.3 V only; use a voltage divider (e.g. 1 k / 2 k)
  on both ECHO lines if you have not already.

## Tests (run on the Jetson/PC, no hardware)
`./test/run_tests.sh` compiles the sketch against an Arduino stand-in (`test/shim`) and runs 16
scenarios: pinout, PWM cap, watchdog, heartbeat, partial/oversized/garbage lines, obstacle gate,
glitch rejection, full avoidance cycle, avoidance timeout, telemetry format.
This checks the logic only; it does not replace testing on the robot.

## Flashing
Arduino IDE: board "ESP32 Dev Module", open the sketch folder, Upload. Close anything else holding
the serial port first. **First test with the wheels off the ground**: send `M,80,80` and confirm
the wheels spin, then stop sending and confirm they stop within ~0.5 s.
