"""Put the RPLidar A1 back into a known state before rplidar_node connects.

After a session that ended uncleanly (killed launch, crash) the A1 can still be scanning,
and rplidar_node then hangs on connect without exiting, so its respawn never helps. This
sends STOP and a soft RESET, waits for the A1 to reboot (~2 s), and exits. Run by
securebot_lidar.launch.py before the driver; harmless if the LiDAR was already idle.

  ros2 run securebot_ros lidar_reset [port]
"""
import sys
import time

DEFAULT_PORT = '/dev/serial/by-path/platform-3610000.usb-usb-0:2.2:1.0-port0'
STOP, RESET = b'\xa5\x25', b'\xa5\x40'


def reset(port, settle=2.0):
    import serial
    with serial.Serial(port, 115200, timeout=0.2) as s:
        s.dtr = True                      # A1 USB adapter: DTR asserted = motor stopped
        s.write(STOP)
        s.flush()
        time.sleep(0.05)
        s.write(RESET)
        s.flush()
        banner, end = b'', time.monotonic() + settle + 1.0
        while time.monotonic() < end and b'Model' not in banner:   # the A1 reboots, ~2 s
            banner += s.read(256)
        time.sleep(max(0.0, settle - 1.0))
    return b'RP LIDAR' in banner


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith('--')]   # ignore --ros-args etc.
    port = args[0] if args else DEFAULT_PORT
    try:
        ok = reset(port)
    except Exception as e:                # never block the launch: the driver retries anyway
        print(f'lidar_reset: could not reset {port}: {e}')
        return 0
    print(f'lidar_reset: {port} ' + ('reset, LiDAR rebooted' if ok else 'reset sent (no banner seen)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
