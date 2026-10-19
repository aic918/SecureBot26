# Ultrasonic wiring diagnostic

A standalone sketch for finding out why an HC-SR04 reports "no echo". It is **not** the robot
firmware: flash it, read the report, then flash `securebot_esp32` again.

It leaves the motor, encoder, boot-strapping (0, 2, 5, 12, 15), flash (6-11) and USB-serial (1, 3)
pins alone, and only listens on GPIO 18 (the right encoder white wire may be connected there).

## Flash (PlatformIO, separate project so the robot firmware project is untouched)
`~/Documents/PlatformIO/Projects/ultrasonic_diag` has `src/main.cpp` linked to this file.
Open that folder in VS Code, click Upload, then open the Serial Monitor (115200).
Close any other program using the serial port first.

## Use
Hold a flat, hard object (a book) 20-30 cm in front of **one** sensor.
Every 2 s it prints, for each sensor, how many of 20 pings got an echo, with the wiring as
configured and with TRIG/ECHO swapped:

    LEFT  normal  (TRIG 23, ECHO 19): 20/20 echoes, 20.4 to 20.8 cm
    LEFT  swapped (TRIG 19, ECHO 23): 0/20 echoes

Typed commands (then Enter):

| Command | What it does |
|---|---|
| `pairs` | print the report now |
| `scan` | drive each candidate pin as TRIG and count echo edges on all the others; a sensor on unexpected pins shows up as `<-- echo seen: TRIG GPIO a -> ECHO GPIO b` |
| `hold left` / `hold right` | hold that sensor's TRIG pin HIGH for 5 s; a meter at the sensor's TRIG pin should read about 3.3 V |

## Reading the result
| You see | Meaning |
|---|---|
| `normal` has echoes | that sensor works as wired |
| only `swapped` has echoes | the sensor's TRIG and ECHO wires are swapped: swap them |
| `scan` shows echo on other pins | the wires are on different pins than the firmware expects: move them (or change the pin constants) |
| no echoes anywhere | check power/ground at the sensor, the ECHO voltage divider, `hold` voltage at TRIG; try another sensor |

Host tests (`../test/run_tests.sh`) simulate a sensor on various pins and check each of these cases.
