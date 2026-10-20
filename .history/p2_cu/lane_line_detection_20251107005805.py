"""
lane_line_detection.py

Improved lane detection and control based on lane_line_detection_good.py.
Features:
- Robust lane edge detection using Canny + small fallback row
- PID steering controller tuned for responsive & stable driving
- Compatibility wrapper calculate_control_signal that accepts:
    * calculate_control_signal(image, draw=None)
    * calculate_control_signal(current_speed, image)
  so it works with both drive.py and drive_good.py calling styles.
- Throttle control that reduces speed on sharp turns and when lanes are lost.
"""

import time
from typing import Tuple, Optional

import cv2
import numpy as np

# --- state ---
_frame_id = 0


class PID:
    """Simple PID controller (supports calling as function)."""
    def __init__(self, kp=0.02, ki=0.0, kd=0.03, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.time_func = time.monotonic
        self.last_time = self.time_func()
        self.last_state = None
        self.integral = 0.0

    def __call__(self, state: float) -> float:
        error = self.setpoint - state
        now = self.time_func()
        dt = now - self.last_time
        if dt <= 0.0:
            dt = 1e-12
        d_state = 0.0 if self.last_state is None else (state - self.last_state)

        # integrate error
        self.integral += self.ki * error * dt
        proportional = self.kp * error
        derivative = -self.kd * d_state / dt

        out = proportional + self.integral + derivative

        self.last_state = state
        self.last_time = now
        return out


_pid = PID()  # tuned default


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        xx = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xx))


def find_lane_lines(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Detect left and right lane edge points on a single row near the bottom.
    Returns (left_point, right_point, center_x).
    If a side is not found, returns -1 for that side.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 120, 200)  # slightly more permissive lower threshold

    h, w = edges.shape
    interested_line_y = int(h * 0.9)
    line = edges[interested_line_y, :]

    left_point = -1
    right_point = -1
    lane_width = 120  # expected lane width in pixels
    center = w // 2

    # search left from center
    for x in range(center, 0, -1):
        if line[x] > 0:
            left_point = x
            break

    # search right from center
    for x in range(center + 1, w):
        if line[x] > 0:
            right_point = x
            break

    # try a second pass with a slightly different row if not detected
    if left_point == -1 or right_point == -1:
        alt_y = int(h * 0.85)
        line2 = edges[alt_y, :]
        if left_point == -1:
            for x in range(center, 0, -1):
                if line2[x] > 0:
                    left_point = x
                    break
        if right_point == -1:
            for x in range(center + 1, w):
                if line2[x] > 0:sss
                    right_point = x
                    break

    # Predict missing side using expected lane width
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    return left_point, right_point, center


def birdview_transform(img: np.ndarray) -> np.ndarray:
    """
    Optional bird-view transform. Keeps default IMAGE_H/W of current image size.
    """
    h, w = img.shape[:2]
    IMAGE_H = h
    IMAGE_W = w

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H],
                      [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[int(IMAGE_W*0.375), IMAGE_H],
                      [int(IMAGE_W*0.625), IMAGE_H],
                      [-int(IMAGE_W*0.25), 0],
                      [IMAGE_W + int(IMAGE_W*0.25), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped


def _compute_throttle(base_speed: float, steering: float, lanes_found: bool) -> float:
    """
    Compute throttle (0..1) from base_speed, steering magnitude and whether lanes are visible.
    - base_speed is desired top throttle (0..1)
    - steering is steering command (unclamped)
    Decrease speed when steering is large or lanes are lost.
    """
    steer_mag = min(abs(steering), 1.0)
    throttle = base_speed * (1.0 - steer_mag * 0.3)
    if not lanes_found:
        throttle = min(throttle, base_speed * 0.4)
    return clamp(throttle, 0.0, 1.0)


def calculate_control_signal(*args, **kwargs) -> Tuple[float, float]:
    """
    Compatibility wrapper. Supports two call styles:
    1) calculate_control_signal(image, draw=None)
    2) calculate_control_signal(current_speed, image)

    Returns (throttle, steering_angle)
    throttle in [0,1], steering_angle in [-1,1]
    """
    global _frame_id
    _frame_id += 1

    image = None
    draw = kwargs.get("draw", None)
    current_speed = None

    if len(args) == 1:
        image = args[0]
    elif len(args) == 2:
        a0, a1 = args[0], args[1]
        if isinstance(a0, (int, float)):
            current_speed = float(a0)
            image = a1
        else:
            image = a0
            draw = a1
    elif len(args) >= 3:
        if isinstance(args[0], (int, float)):
            current_speed = float(args[0])
            image = args[1]
            draw = args[2]
        else:
            image = args[0]
            draw = args[1]

    if image is None:
        image = kwargs.get("image", None)
    if image is None:
        return 0.0, 0.0

    try:
        left_point, right_point, im_center = find_lane_lines(image)
    except Exception:
        return 0.0, 0.0

    steering = 0.0
    lanes_found = (left_point != -1 and right_point != -1)

    if lanes_found:
        center_point = (left_point + right_point) // 2
        center_diff = center_point - im_center  # positive -> lane center is to the right
        steering = -_pid(center_diff)
        steering = clamp(steering, -1.2, 1.2)
        steering = clamp(steering, -1.0, 1.0)
    else:
        if left_point != -1:
            steering = -_pid((left_point + 60) - im_center)
            steering = clamp(steering, -0.6, 0.6)
        elif right_point != -1:
            steering = -_pid((right_point - 60) - im_center)
            steering = clamp(steering, -0.6, 0.6)
        else:
            steering = 0.0

    base_speed = 0.91
    if current_speed is not None:
        try:
            cs = float(current_speed)
            if cs > 0.0:
                base_speed = clamp(1.2 - cs, 0.3, 1.0)
        except Exception:
            base_speed = 0.9

    throttle = _compute_throttle(base_speed, steering, lanes_found)

    if draw is not None:
        try:
            h, w = draw.shape[:2]
            y_line = int(h * 0.9)
            cv2.line(draw, (0, y_line), (w, y_line), (0, 0, 255), 2)
            if left_point != -1:
                cv2.circle(draw, (left_point, y_line), 6, (255, 255, 0), -1)
            if right_point != -1:
                cv2.circle(draw, (right_point, y_line), 6, (0, 255, 0), -1)
            cv2.circle(draw, (im_center, y_line), 4, (255, 0, 0), -1)
            txt = f"T:{throttle:.3f} S:{steering:.3f}"
            cv2.putText(draw, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception:
            pass

    return float(throttle), float(steering)