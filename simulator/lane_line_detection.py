"""Multi-row lane tracking and time-based, bounded steering for the simulator."""
import math
import time
from numbers import Real

import cv2
import numpy as np

# Smaller Y means looking farther ahead. Coordinates are fractions of image size.
SCAN_ROWS = (0.84, 0.76, 0.68)
INITIAL_WIDTHS = (0.68, 0.57, 0.46)
STEERING_KP = 1.7                # Error is normalized by half the image width.
STEERING_KD = 0.025
MAX_STEERING = 0.75
STEERING_RATE = 2.6                # Steering units / second, including sign actions.
CRUISE_THROTTLE = 0.6
TURN_THROTTLE = 0.5
LOST_LANE_STOP_SECONDS = 0.75


def clamp(value, low, high):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value)) if math.isfinite(value) else low


class SteeringLimiter:
    """Bound the final steering change; dt is explicit for reproducible tests."""
    def __init__(self):
        self.value = 0.0

    def update(self, target, dt):
        step = STEERING_RATE * clamp(dt, 0.001, 0.10)
        target = clamp(target, -MAX_STEERING, MAX_STEERING)
        self.value += clamp(target - self.value, -step, step)
        return self.value


def _groups(row, gap):
    xs = np.flatnonzero(row)
    if not len(xs):
        return []
    groups = np.split(xs, np.where(np.diff(xs) > gap)[0] + 1)
    return [float(np.median(g)) for g in groups]


class LaneController:
    def __init__(self):
        self.last_time = None
        self.shape = None
        self.widths = np.array(INITIAL_WIDTHS, dtype=float)
        self.centers = np.full(len(SCAN_ROWS), 0.5)
        self.error = 0.0
        self.derivative = 0.0
        self.last_error = None
        self.steering_filter = 0.0
        self.limiter = SteeringLimiter()
        self.throttle = 0.0
        self.lost_seconds = 0.0
        self.confidence = 0.0
        self.reliable = False
        self.samples = []

    def detect(self, image):
        h, w = image.shape[:2]
        if self.shape != (h, w):
            self.widths = np.array(INITIAL_WIDTHS, dtype=float)
            self.centers = np.full(len(SCAN_ROWS), 0.5)
            self.shape = (h, w)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # White markings first: scenery edges should not pull the steering.
        white = cv2.inRange(hsv, (0, 0, 155), (180, 90, 255))
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 70, 160)
        samples = []
        previous_center = None
        for i, fraction in enumerate(SCAN_ROWS):
            y = min(h - 1, int(h * fraction))
            expected = self.centers[i] * w
            if previous_center is not None:
                expected = 0.55 * expected + 0.45 * previous_center
            lane_width = self.widths[i] * w
            found = None
            for mask, quality in ((white, 1.0), (edges, 0.55)):
                # Require support over a small band, rather than a single noisy row.
                band = mask[max(0, y - 2):min(h, y + 3)]
                candidates = _groups(np.count_nonzero(band, axis=0) >= 2,
                                     max(3, int(w * 0.012)))
                pairs = []
                for j, left in enumerate(candidates):
                    for right in candidates[j + 1:]:
                        width = right - left
                        center = (left + right) / 2
                        if (0.55 * lane_width <= width <= 1.55 * lane_width
                                and abs(center - expected) < 0.24 * w
                                and (previous_center is None
                                     or abs(center - previous_center) < 0.14 * w)):
                            score = (abs(center - expected) / w
                                     + 0.65 * abs(width - lane_width) / w)
                            pairs.append((score, left, right))
                if pairs:
                    _, left, right = min(pairs)
                    found = (left, right, quality, True)
                    break
                # Only extrapolate one edge near a previously expected boundary.
                singles = []
                for x in candidates:
                    for side in (-1, 1):
                        distance = abs(x - (expected + side * lane_width / 2))
                        if distance < 0.12 * w:
                            center = x - side * lane_width / 2
                            if (previous_center is None
                                    or abs(center - previous_center) < 0.14 * w):
                                singles.append((distance, center))
                if singles and found is None:
                    _, center = min(singles)
                    found = (center - lane_width / 2, center + lane_width / 2,
                             quality * 0.4, False)
            if found is None:
                continue
            left, right, quality, paired = found
            center = (left + right) / 2
            if paired:
                self.widths[i] = 0.92 * self.widths[i] + 0.08 * (right - left) / w
            self.centers[i] = center / w
            previous_center = center
            samples.append((fraction, center / w, quality, left, right, y))
        self.samples = samples
        return samples

    def update(self, image, draw=None, dt=None):
        now = time.monotonic()
        if dt is None:
            dt = 1 / 30 if self.last_time is None else now - self.last_time
        self.last_time = now
        elapsed = clamp(dt, 0.001, 1.0)
        dt = clamp(dt, 0.001, 0.10)
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            samples = []
        else:
            samples = self.detect(image)
        self.confidence = sum(s[2] for s in samples) / len(SCAN_ROWS)
        reliable = len(samples) >= 2 and self.confidence >= 0.22
        self.reliable = reliable
        if reliable:
            self.lost_seconds = 0.0
            # Prefer far rows, but keep near rows for lateral position recovery.
            weights = np.array([s[2] * (1.5 if s[0] <= 0.68 else 1.0)
                                for s in samples])
            target = float(np.average([s[1] for s in samples], weights=weights))
            raw_error = (target - 0.5) * 2.0
            # A small dead band suppresses alternating corrections on straights.
            raw_error = math.copysign(max(0.0, abs(raw_error) - 0.012), raw_error)
            self.error += (1 - math.exp(-dt / 0.10)) * (raw_error - self.error)
            derivative = 0 if self.last_error is None else (self.error - self.last_error) / dt
            self.derivative += (1 - math.exp(-dt / 0.16)) * (derivative - self.derivative)
            self.last_error = self.error
            target_steering = STEERING_KP * self.error + STEERING_KD * self.derivative
        else:
            self.lost_seconds += elapsed
            self.last_error = None
            self.derivative = 0.0
            # Brief gaps retain the last bend; prolonged loss gently centers and stops.
            target_steering = self.steering_filter * math.exp(-dt / 0.45)
        target_steering = clamp(target_steering, -MAX_STEERING, MAX_STEERING)
        self.steering_filter += (1 - math.exp(-dt / 0.08)) * (target_steering - self.steering_filter)
        steering = self.limiter.update(self.steering_filter, dt)
        turn = clamp(max(abs(target_steering), abs(steering)) / 0.6, 0, 1)
        target_throttle = CRUISE_THROTTLE - (CRUISE_THROTTLE - TURN_THROTTLE) * turn
        target_throttle *= 0.55 + 0.45 * self.confidence
        if not reliable:
            target_throttle = min(target_throttle, 0.18)
        if self.lost_seconds >= LOST_LANE_STOP_SECONDS:
            self.throttle = 0.0
        else:
            rate = 0.45 if target_throttle > self.throttle else 1.8
            self.throttle += clamp(target_throttle - self.throttle, -rate * dt, rate * dt)
        if draw is not None and image is not None:
            h, w = draw.shape[:2]
            for _, center, quality, left, right, y in samples:
                color = (0, 255, 0) if quality >= 0.5 else (0, 180, 255)
                cv2.line(draw, (int(clamp(left, 0, w-1)), y),
                         (int(clamp(right, 0, w-1)), y), color, 1)
                cv2.circle(draw, (int(center * w), y), 4, (0, 0, 255), -1)
            cv2.putText(draw, f"Lane confidence:{self.confidence:.2f} Lost:{self.lost_seconds:.2f}s",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return float(self.throttle), float(steering)


_controller = LaneController()


def reset_control_state():
    global _controller
    _controller = LaneController()


def get_lane_status():
    """Expose the current frame's lane result without running detection twice."""
    return {'reliable': _controller.reliable,
            'lost_seconds': _controller.lost_seconds}


def find_lane_lines(image):
    """Legacy three-value interface, using the closest available preview row."""
    samples = _controller.detect(image)
    if not samples:
        return -1, -1, image.shape[1] // 2
    sample = min(samples, key=lambda s: abs(s[0] - 0.68))
    return int(sample[3]), int(sample[4]), image.shape[1] // 2


def calculate_control_signal(*args, **kwargs):
    """Accept (image, draw), (speed, image, draw), or image=/draw= keywords.

    Speed telemetry has no documented unit here; throttle does not assume one.
    """
    draw = kwargs.get('draw')
    image = kwargs.get('image')
    if args:
        offset = 1 if isinstance(args[0], Real) else 0
        image = args[offset] if len(args) > offset else image
        if len(args) > offset + 1:
            draw = args[offset + 1]
    return _controller.update(image, draw=draw, dt=kwargs.get('dt'))
