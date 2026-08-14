"""
search_state_node
------------------
The core "reasoning" loop. Subscribes to /target_class (what to look
for) and /detections (what the camera currently sees), and drives the
robot through a small state machine, publishing /cmd_vel_raw.

IMPORTANT: this publishes /cmd_vel_raw, NOT /cmd_vel directly. Your
existing safety_node should subscribe to /cmd_vel_raw, apply the
ultrasonic hard-stop / wall-avoidance override, and republish the
gated result to /cmd_vel for base_driver. Do not duplicate wall-safety
logic here - reuse what you already have.

States:
  IDLE            - no target set yet, motors stopped
  SEARCHING       - sweeping/exploring, watching for target in /detections
  APPROACH_VERIFY - candidate seen, driving toward it, re-checking as it gets closer
  FOUND           - stopped, target confirmed at close range
  TIMEOUT         - gave up after global time limit

TODO before running:
  - Tune SWEEP_* constants for your actual mecanum robot's speed/turning.
  - Tune CONFIDENCE_THRESHOLD, APPROACH_BBOX_AREA_FRAC (or swap in
    ultrasonic range if that's a more reliable "close enough" signal
    for your setup) against real test runs.
  - Confirm /range topic name/type matches your existing ultrasonic node
    if you wire it in below (optional, currently unused by this file -
    safety_node already handles hard-stop independently).
"""

import json
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)

# --- Tunable constants -------------------------------------------------
CONTROL_HZ = 10.0
CONFIDENCE_THRESHOLD = 0.45
GLOBAL_TIMEOUT_SEC = 150.0          # give up after this long
DETECTION_STALE_SEC = 1.0           # ignore detections older than this
LOST_TARGET_FRAMES_TO_REVERT = 8    # consecutive misses before APPROACH_VERIFY -> SEARCHING

# Sweep behavior (SEARCHING) - placeholder pattern, tune for your robot.
# Example: slow rotate in place. Swap for a mecanum strafe-sweep if you
# want to lean into the holonomic advantage.
SWEEP_ANGULAR_Z = 0.4

# Approach behavior (APPROACH_VERIFY)
APPROACH_FORWARD_SPEED = 0.15
APPROACH_LATERAL_GAIN = 0.5          # proportional gain on bbox horizontal offset -> strafe
APPROACH_BBOX_AREA_FRAC_FOUND = 0.25  # bbox covering this fraction of frame area = "close enough"
FRAME_AREA_ASSUMED = 320 * 320        # matches perception_node's default image_size

# --- Safety (inline, since this robot has no separate safety_node) ---
# Ultrasonic is front-facing only - this guards forward motion, NOT
# strafing sideways. Known limitation, acceptable for a front-approach
# search-and-drive-to-target behavior, but be aware of it during test runs.
ULTRASONIC_STOP_DISTANCE_M = 0.25
ULTRASONIC_STALE_SEC = 0.5  # if range data goes stale, treat as unsafe and stop


class SearchState(Enum):
    IDLE = auto()
    SEARCHING = auto()
    APPROACH_VERIFY = auto()
    FOUND = auto()
    TIMEOUT = auto()


class SearchStateNode(Node):
    def __init__(self):
        super().__init__('search_state_node')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/search_state', LATCHED_QOS)

        self.create_subscription(String, '/target_class', self.on_target, LATCHED_QOS)
        self.create_subscription(String, '/detections', self.on_detections, 10)
        self.create_subscription(Range, '/ultrasonic/range', self.on_range, 10)

        self.state = SearchState.IDLE
        self.target_class = None
        self.latest_detections = []
        self.latest_detections_time = 0.0
        self.search_start_time = None
        self.lost_target_frames = 0
        self.latest_range_m = None
        self.latest_range_time = 0.0

        self.timer = self.create_timer(1.0 / CONTROL_HZ, self.control_loop)

        self._publish_state()
        self.get_logger().info('search_state_node ready. Waiting for /target_class.')

    # --- Subscriptions ---------------------------------------------------

    def on_target(self, msg: String):
        self.target_class = msg.data
        self.get_logger().info(f'New target: {self.target_class}. Starting search.')
        self.search_start_time = time.time()
        self.lost_target_frames = 0
        self._set_state(SearchState.SEARCHING)

    def on_detections(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.latest_detections = payload.get('detections', [])
        self.latest_detections_time = payload.get('stamp', time.time())

    def on_range(self, msg: Range):
        self.latest_range_m = msg.range
        self.latest_range_time = time.time()

    # --- Helpers -----------------------------------------------------------

    def _forward_motion_blocked(self):
        """True if it's unsafe to command forward motion right now."""
        if self.latest_range_m is None:
            return True  # never heard from ultrasonic - don't assume it's safe
        if time.time() - self.latest_range_time > ULTRASONIC_STALE_SEC:
            return True  # stale reading - don't trust it, fail safe
        return self.latest_range_m < ULTRASONIC_STOP_DISTANCE_M

    def _set_state(self, new_state: SearchState):
        if new_state != self.state:
            self.get_logger().info(f'{self.state.name} -> {new_state.name}')
        self.state = new_state
        self._publish_state()

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state.name))

    def _fresh_detections(self):
        if time.time() - self.latest_detections_time > DETECTION_STALE_SEC:
            return []
        return self.latest_detections

    def _best_target_detection(self):
        candidates = [
            d for d in self._fresh_detections()
            if d['class'] == self.target_class and d['confidence'] >= CONFIDENCE_THRESHOLD
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d['confidence'])

    def _stop(self):
        self.cmd_pub.publish(Twist())

    # --- Main control loop ---------------------------------------------

    def control_loop(self):
        if self.state == SearchState.IDLE:
            return  # nothing to do until a target is set

        if self.state in (SearchState.FOUND, SearchState.TIMEOUT):
            return  # terminal states, motors already stopped

        # Global timeout check applies to any active searching state
        if self.search_start_time and (time.time() - self.search_start_time) > GLOBAL_TIMEOUT_SEC:
            self._stop()
            self._set_state(SearchState.TIMEOUT)
            self.get_logger().warn(f'Timed out looking for "{self.target_class}".')
            return

        if self.state == SearchState.SEARCHING:
            self._do_searching()
        elif self.state == SearchState.APPROACH_VERIFY:
            self._do_approach_verify()

    def _do_searching(self):
        detection = self._best_target_detection()
        if detection is not None:
            self.get_logger().info(f'Candidate spotted: {detection}')
            self._set_state(SearchState.APPROACH_VERIFY)
            return

        # TODO: replace with your real sweep pattern. This placeholder
        # just rotates in place - no forward component, so it doesn't
        # need the ultrasonic gate below. If you change this to a
        # strafe-sweep or add any linear.x, gate it with
        # self._forward_motion_blocked() the same way _do_approach_verify does.
        cmd = Twist()
        cmd.angular.z = SWEEP_ANGULAR_Z
        self.cmd_pub.publish(cmd)

    def _do_approach_verify(self):
        detection = self._best_target_detection()

        if detection is None:
            self.lost_target_frames += 1
            if self.lost_target_frames >= LOST_TARGET_FRAMES_TO_REVERT:
                self.get_logger().info('Lost target during approach, reverting to SEARCHING.')
                self._set_state(SearchState.SEARCHING)
                self.lost_target_frames = 0
            # else: hold last command briefly rather than immediately reverting -
            # avoids flapping on a single missed frame
            return

        self.lost_target_frames = 0

        x1, y1, x2, y2 = detection['bbox']
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        bbox_center_x = (x1 + x2) / 2.0
        frame_center_x = FRAME_AREA_ASSUMED ** 0.5 / 2.0  # assumes square-ish frame; adjust if not

        if bbox_area >= APPROACH_BBOX_AREA_FRAC_FOUND * FRAME_AREA_ASSUMED:
            self._stop()
            self._set_state(SearchState.FOUND)
            self.get_logger().info(f'FOUND "{self.target_class}".')
            return

        # Proportional control: strafe to center the target, creep forward.
        # This is what actually uses the mecanum base's holonomic advantage -
        # lateral correction without needing to turn the chassis.
        offset = (bbox_center_x - frame_center_x) / frame_center_x  # normalized [-1, 1]

        cmd = Twist()
        if self._forward_motion_blocked():
            # Too close to safely creep forward - hold lateral centering only.
            cmd.linear.x = 0.0
            self.get_logger().warn(
                f'Forward blocked by ultrasonic ({self.latest_range_m}), holding position.'
            )
        else:
            cmd.linear.x = APPROACH_FORWARD_SPEED
        cmd.linear.y = -APPROACH_LATERAL_GAIN * offset  # sign depends on your frame convention - verify!
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SearchStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
