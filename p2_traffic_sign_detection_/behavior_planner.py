# behavior_planner.py
import time

class BehaviorPlanner:
    """
    Finite State Machine Behavior Planner for simple traffic sign-based decisions.

    States:
    - FOLLOW: normal lane following
    - STOPPED: stopped at a stop sign for a fixed duration
    - TURN_LEFT: execute left turn (setpoint offset)
    - TURN_RIGHT: execute right turn (setpoint offset)

    Usage:
    - call step(signs, image_width) every frame, where signs is the latest list
      returned by detect_traffic_signs (format: [ [class, x, y, w, h], ... ])
    - step returns (throttle_multiplier, desired_center_diff_pixels, state)
    """
    STATE_FOLLOW = "FOLLOW"
    STATE_STOPPED = "STOPPED"
    STATE_TURN_LEFT = "TURN_LEFT"
    STATE_TURN_RIGHT = "TURN_RIGHT"

    def __init__(self,
                 stop_duration=3.0,
                 turn_duration=1.0,
                 turn_offset_pixels_fraction=0.12):
        self.state = self.STATE_FOLLOW
        self.stop_until = 0.0
        self.turn_until = 0.0
        self.forbidden_left_until = 0.0
        self.forbidden_right_until = 0.0

        # Parameters
        self.stop_duration = stop_duration
        self.turn_duration = turn_duration
        self.turn_offset_fraction = turn_offset_pixels_fraction

    def _now(self):
        return time.time()

    def _set_state(self, new_state, duration=0.0):
        now = self._now()
        self.state = new_state
        if new_state == self.STATE_STOPPED:
            self.stop_until = now + duration
        elif new_state in (self.STATE_TURN_LEFT, self.STATE_TURN_RIGHT):
            self.turn_until = now + duration

    def step(self, signs, image_width):
        """
        signs: list like [['stop', x, y, w, h], ...]
        image_width: width in pixels for converting fraction to pixels
        returns: (throttle_multiplier, desired_center_diff_pixels, state)
        desired_center_diff_pixels is the target "center_diff" used elsewhere
        (center_diff = image_center - lane_center).
        """
        now = self._now()

        # Get latest detected sign classes
        classes = [s[0] for s in signs] if signs else []

        # Handle forbids: if 'no_left' seen recently, block left turns (for a short time)
        if 'no_left' in classes:
            self.forbidden_left_until = now + 3.0
        if 'no_right' in classes:
            self.forbidden_right_until = now + 3.0

        # Transition logic
        # STOP sign has highest priority
        if 'stop' in classes:
            if self.state != self.STATE_STOPPED:
                self._set_state(self.STATE_STOPPED, duration=self.stop_duration)
        else:
            # If currently stopped, check if stop duration elapsed
            if self.state == self.STATE_STOPPED:
                if now >= self.stop_until:
                    self._set_state(self.STATE_FOLLOW)
            # Check turn signs
            if self.state != self.STATE_STOPPED:
                if 'left' in classes and now >= self.forbidden_left_until:
                    self._set_state(self.STATE_TURN_LEFT, duration=self.turn_duration)
                elif 'right' in classes and now >= self.forbidden_right_until:
                    self._set_state(self.STATE_TURN_RIGHT, duration=self.turn_duration)
                else:
                    # If currently in a turn state, check timeout
                    if self.state == self.STATE_TURN_LEFT and now >= self.turn_until:
                        self._set_state(self.STATE_FOLLOW)
                    if self.state == self.STATE_TURN_RIGHT and now >= self.turn_until:
                        self._set_state(self.STATE_FOLLOW)

        # Outputs: choose throttle multiplier and desired center offset (pixels)
        img_w = image_width
        turn_offset_pixels = int(img_w * self.turn_offset_fraction)

        if self.state == self.STATE_STOPPED:
            throttle_mul = 0.0
            desired_center_diff = 0  # keep centered
        elif self.state == self.STATE_TURN_LEFT:
            throttle_mul = 0.35  # slow for turning
            desired_center_diff = +turn_offset_pixels
        elif self.state == self.STATE_TURN_RIGHT:
            throttle_mul = 0.35
            desired_center_diff = -turn_offset_pixels
        else:  # FOLLOW
            throttle_mul = 0.5
            desired_center_diff = 0

        return throttle_mul, desired_center_diff, self.state