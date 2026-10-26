#!/usr/bin/env python3
"""Find a motor PWM frequency and power that start the robot without a push and don't whine.

Talks to the ESP32 directly (no ROS). Stop everything else that uses the ESP32 port first
(securebot_teleop / securebot_live launches, the ultrasonic monitor, a serial monitor).

  python3 tools/motor_tune.py              # ESP32 on its usual USB socket
  python3 tools/motor_tune.py /dev/ttyUSB1

Hold a key to drive; let go and the robot stops (the firmware's watchdog also stops it within
250 ms if this program dies).
  w forward   s backward   a spin left   d spin right   space stop
  frequency down / up:  1 / 2  50 Hz   3 / 4  100 Hz   5 / 6  500 Hz   7 / 8  1000 Hz
                        (50..25000 Hz, starts at the firmware's 10000 Hz)
  power down / up (Shift + the same keys):  ! / @  10   # / $  25   % / ^  50   & / *  100
                        (of 1023, 200..850; + / - also step it by 25)   q quit
Lower frequencies give the L298N more torque and a lower, quieter hum; 10 kHz whines.
Look for the lowest power that starts the robot from standstill on the floor, at a frequency
whose sound you can live with, and tell the result to whoever sets the firmware defaults.
"""
import argparse
import select
import sys
import termios
import time
import tty

ESP32_PORT = '/dev/serial/by-path/platform-3610000.usb-usb-0:2.3:1.0-port0'
# key -> frequency change in Hz (odd keys down, even keys up)
FREQ_STEPS = {'1': -50, '2': 50, '3': -100, '4': 100, '5': -500, '6': 500, '7': -1000, '8': 1000}
FREQ_MIN, FREQ_MAX = 50, 25000      # the firmware's limits
MOVES = {'w': (1, 1), 's': (-1, -1), 'a': (-1, 1), 'd': (1, -1), ' ': (0, 0)}
# key -> power change (of 1023): Shift + 1..8 on a US keyboard, odd keys down, even keys up
POWER_STEPS = {'!': -10, '@': 10, '#': -25, '$': 25, '%': -50, '^': 50, '&': -100, '*': 100,
               '-': -25, '_': -25, '+': 25, '=': 25}
POWER_MIN, POWER_MAX = 200, 850
SEND_PERIOD = 0.1     # firmware watchdog is 250 ms
HOLD_TIME = 0.6       # longer than the terminal's key-repeat delay


class TuneState:
    """Key handling and the command to send; no serial port, so it can be tested."""

    def __init__(self, power=500, freq=10000):
        self.power, self.freq = power, freq
        self.move, self.last_key = (0, 0), None
        self.pending = []                  # one-off commands (frequency changes) to send

    def key(self, ch, now):
        """Returns False for quit."""
        if ch in ('q', '\x03'):
            self.move = (0, 0)
            return False
        if ch in MOVES:
            self.move, self.last_key = MOVES[ch], now
        elif ch in FREQ_STEPS:
            freq = max(FREQ_MIN, min(FREQ_MAX, self.freq + FREQ_STEPS[ch]))
            if freq != self.freq:
                self.freq = freq
                self.pending.append(f'freq {self.freq}')
        elif ch in POWER_STEPS:
            self.power = max(POWER_MIN, min(POWER_MAX, self.power + POWER_STEPS[ch]))
        return True

    def command(self, now):
        """The drive command for this tick: M,l,r while a key is held, else stop."""
        held = self.last_key is not None and now - self.last_key <= HOLD_TIME
        if not held or self.move == (0, 0):
            return 'stop'
        left, right = self.move
        return f'M,{left * self.power},{right * self.power}'


def read_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('port', nargs='?', default=ESP32_PORT)
    parser.add_argument('--power', type=int, default=500, help='starting power (500 of 1023)')
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        sys.exit('motor_tune needs a terminal (it reads single key presses).')

    import serial
    try:
        esp = serial.Serial(args.port, 115200, timeout=0)
    except serial.SerialException as e:
        sys.exit(f'Cannot open {args.port}: {e}\nIs the bridge or another serial program running?')
    time.sleep(2.0)                        # opening the port resets the ESP32
    esp.reset_input_buffer()               # drop what the USB driver queued while closed

    state = TuneState(power=max(POWER_MIN, min(POWER_MAX, args.power)))
    state.pending.append(f'freq {state.freq}')
    print(__doc__.split('\n\n')[2])
    saved = termios.tcgetattr(sys.stdin)
    status, info, buf = '-', '', b''
    try:
        tty.setcbreak(sys.stdin.fileno())
        next_send = 0.0
        while True:
            ch = read_key(0.02)
            if ch is not None and not state.key(ch, time.monotonic()):
                break
            now = time.monotonic()
            while state.pending:
                esp.write((state.pending.pop(0) + '\n').encode())
            if now >= next_send:
                esp.write((state.command(now) + '\n').encode())
                next_send = now + SEND_PERIOD
            buf += esp.read(4096)
            *lines, buf = buf.split(b'\n')
            for line in lines:
                text = line.decode(errors='ignore').strip()
                if text.startswith('S,'):
                    status = text[2:]
                elif text.startswith(('I,freq', 'E,')):
                    info = text
            print(f'\r  freq {state.freq:>5} Hz   power {state.power:>3}/1023 '
                  f'({state.power * 100 // 1023:>2} %)   ESP32: {status:<22} {info:<28}',
                  end='', flush=True)
    finally:
        for _ in range(3):
            esp.write(b'stop\n')
            time.sleep(0.05)
        esp.close()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        print(f'\nstopped. Last setting: freq {state.freq} Hz, power {state.power}/1023.')


if __name__ == '__main__':
    main()
