"""Odometry from LiDAR scan matching plus the gyro heading (no ROS).

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
"""
import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class HeadingOdom:
    def __init__(self, spin_gate=0.6, imu_timeout=0.5):
        self.spin_gate = spin_gate        # rad/s
        self.imu_timeout = imu_timeout    # s
        self.x = self.y = self.yaw = 0.0
        self.imu_offset = None            # gyro yaw that maps to self.yaw
        self.imu_yaw = None
        self.imu_rate = 0.0
        self.imu_time = -math.inf
        self.laser = None                 # last rf2o pose (x, y, yaw)

    def imu_ok(self, now):
        return self.imu_yaw is not None and now - self.imu_time <= self.imu_timeout

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
            if not spinning:
                c, s = math.cos(self.yaw), math.sin(self.yaw)
                self.x += c * fwd - s * side
                self.y += s * fwd + c * side
            if not self.imu_ok(now):
                self.yaw = wrap(self.yaw + wrap(yaw - lyaw))
        self.laser = (x, y, yaw)
        return self.x, self.y, self.yaw
