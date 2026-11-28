"""Odometry from the wheel encoders (or LiDAR scan matching) plus the gyro heading (no ROS).

rf2o (scan matching) estimates both how far the robot moved and how much it turned, but it
loses track during SecureBot's fast full-power spins, and its heading then drifts or jumps.
The MPU-6050 gyro measures turning directly and does not care how fast the robot spins. So:

  heading  = gyro yaw (relative to where it was when this started)
  position = rf2o's movement, measured in the robot's own frame between two rf2o updates,
             turned into the odom frame with the gyro heading; while the gyro says the robot
             spins faster than `spin_gate` (it only does that turning on the spot) rf2o's
             movement is ignored - it is scan-matching noise, the wheels are not translating

If the gyro goes quiet (older than `imu_timeout`), rf2o's own heading changes are used until
it is back.

Wheel encoders, when they report (`update_wheels`), replace rf2o for the distance: each update
moves the robot by the mean of the two wheels' travel along the gyro heading. In a long, plain
hallway scan matching cannot tell how far along it the robot is (and the LiDAR is blind ahead
of the camera tower); the wheels measure exactly that. The heading stays the gyro's: on the
spot the wheels slip and scrub (a measured 90 deg spin read 115 deg on the encoders). rf2o's
distance is used again while the encoders are silent (older than `wheel_timeout`). A jump of
more than `max_step` m in one update (the ESP32 restarted and its counts began again at 0) only
re-bases the counts.
"""
import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class HeadingOdom:
    def __init__(self, spin_gate=0.6, imu_timeout=0.5, ticks_per_meter=0.0, wheel_timeout=0.5,
                 max_step=0.5):
        self.spin_gate = spin_gate        # rad/s
        self.imu_timeout = imu_timeout    # s
        self.ticks_per_meter = ticks_per_meter    # 0: do not use the encoders
        self.wheel_timeout = wheel_timeout        # s
        self.max_step = max_step                  # m in one encoder update
        self.ticks = None                 # last (left, right) encoder counts
        self.wheel_time = -math.inf
        self.distance = 0.0               # m driven forward (minus backward) by the wheels
        self.x = self.y = self.yaw = 0.0
        self.imu_offset = None            # gyro yaw that maps to self.yaw
        self.imu_yaw = None
        self.imu_rate = 0.0
        self.imu_time = -math.inf
        self.laser = None                 # last rf2o pose (x, y, yaw)

    def imu_ok(self, now):
        return self.imu_yaw is not None and now - self.imu_time <= self.imu_timeout

    def wheels_ok(self, now):
        return self.ticks_per_meter > 0 and now - self.wheel_time <= self.wheel_timeout

    def update_wheels(self, now, left, right):
        """Encoder counts since the ESP32 started (+ = forward). Returns the fused pose."""
        if self.ticks_per_meter <= 0:
            return self.x, self.y, self.yaw
        if self.ticks is not None:
            dl = (left - self.ticks[0]) / self.ticks_per_meter
            dr = (right - self.ticks[1]) / self.ticks_per_meter
            if max(abs(dl), abs(dr)) <= self.max_step:
                d = (dl + dr) / 2.0
                self.x += d * math.cos(self.yaw)
                self.y += d * math.sin(self.yaw)
                self.distance += d
        self.ticks, self.wheel_time = (left, right), now
        return self.x, self.y, self.yaw

    def update_imu(self, now, yaw, rate):
        """Gyro heading (rad, any zero) and yaw rate (rad/s)."""
        if self.imu_offset is None or not self.imu_ok(now):
            self.imu_offset = wrap(yaw - self.yaw)          # continue from the current heading
        self.imu_yaw, self.imu_rate, self.imu_time = yaw, rate, now
        self.yaw = wrap(yaw - self.imu_offset)

    def update_laser(self, now, x, y, yaw):
        """rf2o pose (its own odom frame). Returns the fused pose (x, y, yaw)."""
        if self.laser is not None:
            lx, ly, lyaw = self.laser
            dx, dy = x - lx, y - ly
            # the movement in the robot's frame at the previous rf2o pose
            c, s = math.cos(-lyaw), math.sin(-lyaw)
            fwd, side = c * dx - s * dy, s * dx + c * dy
            spinning = self.imu_ok(now) and abs(self.imu_rate) > self.spin_gate
            if not spinning and not self.wheels_ok(now):
                c, s = math.cos(self.yaw), math.sin(self.yaw)
                self.x += c * fwd - s * side
                self.y += s * fwd + c * side
            if not self.imu_ok(now):
                self.yaw = wrap(self.yaw + wrap(yaw - lyaw))
        self.laser = (x, y, yaw)
        return self.x, self.y, self.yaw
