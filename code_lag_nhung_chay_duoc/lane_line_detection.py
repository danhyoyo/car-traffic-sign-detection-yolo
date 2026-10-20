"""
lane_line_detection.py

Improved lane detection and control based on lane_line_detection_good.py.
Features:
- Robust lane edge detection using Canny + small fallback rows and histogram fallback
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
    def __init__(self, kp=0.03, ki=0.0, kd=0.05, setpoint=0.0, verbose: bool = False):
        # PID parameters
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)

        # time
        self.time_func = time.monotonic
        self.last_time = self.time_func()

        # state
        self.last_state = None
        self.integral = 0.0

        # debug
        self.verbose = bool(verbose)

    def __call__(self, state):
        """
        Compute PID control signal.
        state: current measured value (e.g., center_diff)
        """
        # error (setpoint - measurement)
        error = self.setpoint - float(state)
        # change in state (used to compute derivative)
        d_state = 0.0 if self.last_state is None else (state - self.last_state)

        # time delta
        now = self.time_func()
        d_time = now - self.last_time
        if d_time <= 0.0:
            d_time = 1e-6  # avoid div by zero

        # integral update (store integral of error)
        self.integral += error * d_time

        proportional = self.kp * error
        derivative = 0.0
        # derivative term based on change in measurement (avoids derivative kick on setpoint change)
        derivative = -self.kd * (d_state / d_time)

        control_signal = proportional + (self.ki * self.integral) + derivative

        # update state
        self.last_state = state
        self.last_time = now

        if self.verbose:
            print(f"[PID] ctrl={control_signal:.4f}, err={error:.4f}, P={proportional:.4f}, I={self.ki*self.integral:.4f}, D={derivative:.4f}")

        return control_signal


pid = PID(kp=0.005, ki=0.00, kd=0.05, setpoint=0.0)


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        xx = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xx))


def find_lane_lines(image):
    """
    Find left and right lane edge x-positions and image center using a robust strategy:
    - Canny on blurred gray image
    - Check several rows near bottom (fallback)
    - If still not found, use histogram (sum of edges) on lower half
    Returns (left_x, right_x, center_x) where -1 indicates not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 120, 200)

    height, width = edges.shape
    center = width // 2

    left_point = -1
    right_point = -1

    # scan multiple rows from 90% height upward to 60% height for robustness
    y_start = int(height * 0.9)
    y_end = int(height * 0.6)
    for y in range(y_start, y_end, -2):
        row = edges[y, :]
        # find first nonzero from center to left/right
        lp = -1
        rp = -1
        for x in range(center, 0, -1):
            if row[x] > 0:
                lp = x
                break
        for x in range(center, width):
            if row[x] > 0:
                rp = x
                break
        # prefer rows that have both points
        if lp != -1 and rp != -1:
            left_point, right_point = lp, rp
            interested_line_y = y
            break
        # keep single-side sightings as fallback
        if left_point == -1 and lp != -1:
            left_point = lp
            interested_line_y = y
        if right_point == -1 and rp != -1:
            right_point = rp
            interested_line_y = y

    # histogram fallback on lower half if still not both found
    if left_point == -1 or right_point == -1:
        lower = edges[int(height*0.7):height, :]
        col_sum = lower.sum(axis=0)
        # split half search for peaks
        mid = width // 2
        left_peak = int(np.argmax(col_sum[:mid])) if col_sum[:mid].sum() > 0 else -1
        right_rel = col_sum[mid:]
        right_peak = (mid + int(np.argmax(right_rel))) if right_rel.sum() > 0 else -1
        if left_point == -1 and left_peak > 0:
            left_point = left_peak
        if right_point == -1 and right_peak > 0:
            right_point = right_peak

    # if we only saw one side, estimate the other using average lane width (relative)
    estimated_lane_width = int(width * 0.30)  # ~30% of image width
    if left_point != -1 and right_point == -1:
        right_point = min(width - 1, left_point + estimated_lane_width)
    elif right_point != -1 and left_point == -1:
        left_point = max(0, right_point - estimated_lane_width)

    # final debug print (minimal)
    # print("lane pos:", left_point, right_point, center)
    return left_point, right_point, center


def birdview_transform(img: np.ndarray) -> np.ndarray:
    """
    Optional bird-view transform. Keeps default IMAGE_H/W of current image size.
    Use reasonable destination coordinates inside the image bounds.
    """
    h, w = img.shape[:2]
    IMAGE_H = h
    IMAGE_W = w

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H],
                      [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    # put dst coordinates within image (no negative)
    dst = np.float32([[int(IMAGE_W * 0.1), IMAGE_H],
                      [int(IMAGE_W * 0.9), IMAGE_H],
                      [int(IMAGE_W * 0.25), 0],
                      [int(IMAGE_W * 0.75), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped


# --- Global throttle smoothing state ---
_last_throttle = 0.0


def _compute_throttle(base_speed: float, steering: float, lanes_found: bool) -> float:
    """
    Compute throttle (0..1) from base_speed, steering magnitude and whether lanes are visible.
    Uses smoothing and minimum throttle.
    """
    global _last_throttle
    steer_factor = 0.6
    steer_mag = min(abs(steering), 1.0)
    raw = float(base_speed) * (1.0 - steer_mag * steer_factor)

    # if lanes are lost, reduce speed but keep minimum reasonable
    if not lanes_found:
        raw = min(raw, float(base_speed) * 0.5)

    min_throttle = 0.35
    raw = max(raw, min_throttle)

    alpha = 0.45
    throttle = alpha * raw + (1.0 - alpha) * _last_throttle

    _last_throttle = throttle
    return clamp(throttle, 0.0, 1.0)


def calculate_control_signal(*args, **kwargs) -> Tuple[float, float]:
    """
    Compatibility wrapper:
    - calculate_control_signal(image, draw=None)
    - calculate_control_signal(current_speed, image)
    Returns (throttle, steering_angle) with steering in [-1,1].
    """
    global _frame_id
    _frame_id += 1

    image = None
    draw = kwargs.get("draw", None)
    current_speed = None

    # parse args
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

    # birdview then lane detection
    try:
        bird_img = birdview_transform(image)
        left_point, right_point, im_center = find_lane_lines(bird_img)
    except Exception:
        return 0.0, 0.0

    steering = 0.0
    # lanes_found: consider true if at least one real detection
    lanes_found = (left_point != -1 and right_point != -1)

    try:
        if lanes_found:
            center_point = (left_point + right_point) // 2
            center_diff = float(center_point - im_center)
            steering = -pid(center_diff)
            steering = clamp(steering, -0.9, 0.9)
        else:
            # if only one side seen we try to steer toward that side
            if left_point != -1:
                steering = -pid((left_point + 60) - im_center)
                steering = clamp(steering, -0.6, 0.6)
            elif right_point != -1:
                steering = -pid((right_point - 60) - im_center)
                steering = clamp(steering, -0.6, 0.6)
            else:
                steering = 0.0
    except Exception:
        steering = 0.0

    # base speed depends on reported current_speed (if provided)
    base_speed = 0.7
    if current_speed is not None:
        try:
            cs = float(current_speed)
            if cs > 0.0:
                base_speed = clamp(1.2 - cs, 0.3, 1.0)
        except Exception:
            base_speed = 0.7

    throttle = _compute_throttle(base_speed, steering, lanes_found)

    # draw overlay if requested
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