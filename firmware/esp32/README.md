# ESP32 firmware (v2.2)

`securebot_esp32/securebot_esp32.ino` is the low-level controller (Arduino
framework): TI DRV8262 dual H-bridge motor driver, two HC-SR04 ultrasonics, an MPU-6050 gyro and
accelerometer, an INA226 battery monitor and the two wheel encoders. The original v1 firmware is preserved in git history (commit "S0-4").

## What changed from v1
- **Motor PWM:** 10-bit duty (0-1023) at 10 kHz, one LEDC channel per IN pin: the DRV8262 has
  no enable pins, so speed and direction both go on IN1..IN4 (PWM mode, MODE2 floating: the
  driving pin is pulsed, the other held low). `freq <Hz>` changes it until reboot (for tuning
  with `tools/motor_tune.py`). A wheel at rest needs ~550 to break free; once rolling, far less.
  **PWM cap** (`MAX_PWM = 800`) enforced in one output function; applies to every command.
- **Wheel speed control (`V`):** each wheel is held at a target speed in encoder ticks/s by its
  own PI loop (every 40 ms) on top of a feed-forward, with a kick to at least 580 while a wheel
  that should turn is still standing. This balances the two motors (they differ by up to 50 %):
  straight drives drift ~3 deg instead of ~12. It is what the ROS bridge sends by default
  (`speed_control:=true`); `M` (raw PWM) still works.
- **Brake while stopped:** a wheel told 0 is held braked (both IN pins high) for as long as it
  is told 0, not released to coast. With a short brake and then coasting, the robot turned on
  another ~60 deg after a spin.
- **Front flipped (`FRONT_FLIPPED`):** the robot's front is the end with the ultrasonics. The
  flag swaps the wheels and reverses them for the motors and the encoders alike, so `M`/`V`
  commands, `W` lines, patrol and the obstacle gate all work in that frame.
- **Watchdog:** no valid command for 250 ms -> motors stop. The sender must resend (the ROS
  bridge does every 100 ms).
- **Non-blocking serial** line parser (v1 `readStringUntil` could block up to 1 s). Bad lines are
  rejected with `E,...` and do not feed the watchdog.
- **`M,<left>,<right>`** signed-PWM command, matching `esp32_serial_bridge.py` (`/cmd_vel`).
  Legacy text commands still work so `securebot_live_node.py` keeps running.
- **Machine-readable telemetry:** `U,<left_ft>,<right_ft>` at 10 Hz (`-1` = no reading), `S,...` status.
- Ultrasonics sampled one at a time (no cross-talk, shorter blocking), a close reading must repeat
  twice before it counts, and `M` forward motion is blocked near an obstacle (rotate/reverse allowed).
- Avoidance sequence unchanged, but the rotation now times out after 4 s instead of spinning forever.
- No `String` objects (no heap fragmentation); motors are zeroed first thing in `setup()`.
- **Gyro (v2.1):** the MPU-6050 is sampled at 200 Hz, its bias is measured at boot (after a 1 s
  warm-up, ~1.2 s, motors stopped; **keep the robot still**) and refined whenever the robot stands
  still; the yaw goes out as `G` lines. A calibration with a bias over 5 deg/s or a spread over
  1 deg/s (robot moving) is rejected and retried up to 5 times, then the gyro is switched off
  (`E,imu calibration failed`) rather than run with a wrong bias, which made the robot spin in place.
  After 20 failed reads in a row the gyro is switched off (`E,imu stopped answering`).
- **Wheel encoders:** quadrature counts (both edges of both channels) go out as `W` lines, per
  robot wheel (`ENCODERS_SWAPPED`, `*_ENCODER_REVERSED` and `FRONT_FLIPPED` applied, + = forward).
  The ROS side uses 1190 ticks per metre (measured against LiDAR scans).

- **Battery monitor (v2.2):** an INA226 on the same I2C bus as the MPU-6050 (address 0x40)
  measures the pack voltage and current every 500 ms (`B` lines; the ROS guard node turns them
  into a percentage and sends the robot home when low). Absent at boot: `E,power monitor not
  found`, no `B` lines, everything else works. `SHUNT_OHMS` must match the module's shunt.
- **Tamper detection (v2.2):** "down" is measured with the accelerometer at boot (any board
  orientation). Armed with `tamper on` (the ROS bridge does it at start), the robot tipped more
  than 35 deg for 0.3 s, or lifted (total acceleration off 1 g by 0.3 g for 0.15 s), stops the
  motors at once and keeps them stopped whatever is commanded (`T,tripped,<tilt|lift>`) until
  `tamper clear`. Off at boot, so the robot can be carried while the software is off.

## Serial protocol (115200 baud, newline-terminated)
| Direction | Message | Meaning |
|---|---|---|
| to ESP32 | `M,<l>,<r>` | wheel PWM (10-bit duty), -800..800 |
| to ESP32 | `V,<l>,<r>` | wheel speeds in encoder ticks/s (+ = forward), -1000..1000; 0 brakes |
| to ESP32 | `stop` `forward` `patrol` `backward` `left` `right` | legacy modes |
| to ESP32 | `speed <n>` | legacy-mode PWM, clamped 0..800 (default 650) |
| to ESP32 | `freq <hz>` | motor PWM frequency, 50..25000 Hz, until reboot (not a heartbeat) |
| to ESP32 | `tamper on` / `tamper off` | arm / disarm tamper detection (not a heartbeat; off at boot) |
| to ESP32 | `tamper clear` | acknowledge a tamper alarm: the motors may run again |
| to ESP32 | `encscope` | encoder line check: 3 s of min/avg/max mV on GPIO 34, 35, 13 (turn a wheel by hand) |
| from ESP32 | `U,<left_ft>,<right_ft>` | distances in feet, `-1` = none |
| from ESP32 | `S,<mode>,<l>,<r>,<avoidState>` | status, 2 Hz |
| from ESP32 | `G,<yaw_deg>,<yaw_rate_dps>` | gyro yaw since boot (CCW +, -180..180), 20 Hz, only if the IMU answered at boot |
| from ESP32 | `W,<left_ticks>,<right_ticks>` | encoder counts since boot (+ = that wheel forward), 20 Hz |
| from ESP32 | `B,<mV>,<mA>` | battery pack voltage and current (+ = discharging), 2 Hz, only if the INA226 answered at boot |
| from ESP32 | `T,<state>[,<reason>]` | tamper: `off`, `armed`, `tripped,tilt`, `tripped,lift` or `unavailable` (no accelerometer); 2 Hz and at once on a change |
| from ESP32 | `I,<text>` / `E,<text>` | info / safety event |

## Pins (current wiring)
| Function | GPIO |
|---|---|
| Right motor: IN1 / IN2 (H-bridge 1) | 32 / 26 |
| Left motor: IN3 / IN4 (H-bridge 2) | 25 / 33 |
| DRV8262 MODE2 (pin 29) | not wired: floating = PWM mode (`DRIVER_PH_EN = false`) |
| Ultrasonic 1 (left): TRIG / ECHO | 23 / 19 |
| Ultrasonic 2 (right): TRIG / ECHO | 4 / 17 |
| Right encoder: blue / white | 13 / 18 |
| Left encoder: blue / white | 34 / 35 |
| Gyro (MPU-6050, address 0x68 or 0x69): SDA / SCL | 21 / 22 |
| Battery monitor (INA226, address 0x40: A0 and A1 to GND): SDA / SCL | 21 / 22 (shared with the gyro), VCC 3.3 V |

Motor polarity (measured on the robot 2026-11-22): `LEFT_REVERSED = true`,
`RIGHT_REVERSED = false`, `FRONT_FLIPPED = true`; forward pulses IN2 (GPIO 26) and IN3 (GPIO 25).
If a wheel turns the wrong way for a positive command, flip its flag at the top of the sketch
instead of rewiring. If the whole robot drives backwards, flip `FRONT_FLIPPED`.

### Hardware cautions
- **Gyro (MPU-6050) bus held after a reset:** a reset in the middle of an I2C transfer can
  leave the MPU-6050 holding SDA low (boot reports `SDA=0`) until it loses power. At boot the
  firmware clocks SCL up to 16 times and sends a STOP first (`imuBusRecover`), which frees it.
- **Never put a wire on a boot-strapping pin (GPIO 0, 2, 5, 12, 15).** The right encoder's white wire
  used to be on GPIO 12; with it connected the ESP32 could not read its flash, uploads failed
  ("Failed to communicate with the flash chip") and the serial port streamed garbage. It is now on
  GPIO 18. A test fails if any pin in this sketch lands on a strapping, flash (6-11) or USB-serial (1, 3) pin.
- **GPIO 34 and 35 have no internal pull-up.** Open-collector encoder outputs need an external
  pull-up (about 10 k to 3.3 V) on those two lines.
- **GPIO 14 goes high during boot and while flashing**, and drove a motor when IN4 was on it.
  IN4 moved off it; a test fails if any pin in the sketch is put back on GPIO 14.
- **INA226 wiring:** high side, battery + to IN+ and IN- to the robot's supply; GND shared
  with the ESP32; VCC from the ESP32's 3.3 V (its SDA / SCL pull-ups then stay at 3.3 V).
  Check the shunt resistor printed on the module: most boards have `R100` (0.1 ohm, only
  ~0.8 A before the reading saturates, too little for the motors). Use an `R010` (0.01 ohm,
  ~8 A) module or shunt and keep `SHUNT_OHMS = 0.01`; with `R100` set `SHUNT_OHMS = 0.1`.
  Do not use an INA219: its 26 V limit is below a full 7S pack (29.4 V).
- **HC-SR04 ECHO is a 5 V output.** ESP32 pins are 3.3 V only; use a voltage divider (e.g. 1 k / 2 k)
  on both ECHO lines if you have not already.

## Tests (run on the Jetson/PC, no hardware)
`./test/run_tests.sh` compiles the sketch against an Arduino stand-in (`test/shim`) and runs 43
scenarios: pinout and polarity, PWM setup and `freq`, PWM cap, watchdog, heartbeat,
partial/oversized/garbage lines, obstacle gate, glitch rejection, full avoidance cycle,
avoidance timeout, telemetry format, gyro (setup, yaw, bias, missing, lost, slow to wake,
bad calibration rejected), braking while stopped, encoders and `encscope`, and the `V` speed
loop (tracking two unequal simulated motors, kick and clamp, watchdog and obstacle gate), tamper
detection (off by default, tilt latches until cleared, short knocks and ramps ignored, lift,
any board orientation, no accelerometer, not a heartbeat) and the INA226 (readings, absent,
without the gyro).
This checks the logic only; it does not replace testing on the robot.

## Flashing
This sketch is the source of truth; the PlatformIO project on the Jetson
(`~/Documents/PlatformIO/Projects/securebot/src/main.ino`) was where it was developed, so copy
any change made there back here. Arduino IDE: board "ESP32 Dev Module", open the sketch folder, Upload. Close anything else holding
the serial port first. **First test with the wheels off the ground**: send `M,600,600` and confirm
the wheels spin forward (towards the ultrasonics; lower values may not start them), then stop sending and confirm they stop within ~0.5 s.
**v2.2:** after flashing, check the boot lines for `I,power monitor ready (INA226)` and
`I,accelerometer ready`; then `tamper on`, tilt the robot by hand and look for
`T,tripped,tilt` (the wheels must not turn until `tamper clear`). Lift it gently and firmly to
see whether `TAMPER_LIFT_G` / `TAMPER_LIFT_MS` need tuning for this robot.
