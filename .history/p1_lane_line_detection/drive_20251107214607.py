import cv2
import numpy as np
import time
from typing import Tuple, Optional

frame_id = 0

class PID(object):
    def __init__(self, kp=0.05, ki=0.0003, kd=0.002, setpoint=0.0, verbose: bool = False):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self.time_func = time.monotonic
        self.last_time = self.time_func()
        self.last_state = None
        self.integral = 0.0
        self.verbose = verbose

    def __call__(self, state):
        state = float(state)
        error = self.setpoint - state
        d_state = 0.0 if self.last_state is None else (state - self.last_state)
        now = self.time_func()
        d_time = now - self.last_time
        if d_time <= 0.0:
            d_time = 1e-6

        # integrate error (note: ki multiplied here later)
        self.integral += error * d_time

        proportional = self.kp * error
        derivative = -self.kd * (d_state / d_time)
        control_signal = proportional + (self.ki * self.integral) + derivative

        self.last_state = state
        self.last_time = now

        if self.verbose:
            print(f"[PID] ctrl={control_signal:.4f} err={error:.4f} P={proportional:.4f} I={self.ki*self.integral:.4f} D={derivative:.4f}")

        return control_signal

pid = PID(kp=0.05, ki=0.0003, kd=0.002, setpoint=0.0)


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        xx = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xx))


def birdview_transform(img: np.ndarray) -> np.ndarray:
    """
    Bird's-eye transform with destination coordinates kept inside image bounds.
    """
    h, w = img.shape[:2]
    IMAGE_H = h
    IMAGE_W = w

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H],
                      [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    # Keep destination coords within image (no negatives)
    dst = np.float32([
        [int(IMAGE_W * 0.1), IMAGE_H],
        [int(IMAGE_W * 0.9), IMAGE_H],
        [int(IMAGE_W * 0.25), 0],
        [int(IMAGE_W * 0.75), 0]
    ])
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    except Exception:
        # fallback: return original image if transform fails
        warped = img.copy()
    return warped


def find_lane_lines(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Simple robust bottom-row lane edge detection with small fallbacks.
    Returns (left_x, right_x, center_x), -1 means not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 120, 200)

    height, width = edges.shape
    center = width // 2

    left_point = -1
    right_point = -1

    # scan a few rows near bottom for robustness
    for y in range(int(height * 0.9), int(height * 0.6), -2):
        row = edges[y, :]
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
        if lp != -1 and rp != -1:
            left_point, right_point = lp, rp
            break
        if left_point == -1 and lp != -1:
            left_point = lp
        if right_point == -1 and rp != -1:
            right_point = rp

    # histogram fallback if still missing
    if left_point == -1 or right_point == -1:
        lower = edges[int(height * 0.7):height, :]
        col_sum = lower.sum(axis=0)
        mid = width // 2
        if col_sum[:mid].sum() > 0:
            lp = int(np.argmax(col_sum[:mid]))
        else:
            lp = -1
        if col_sum[mid:].sum() > 0:
            rp = mid + int(np.argmax(col_sum[mid:]))
        else:
            rp = -1
        if left_point == -1 and lp > 0:
            left_point = lp
        if right_point == -1 and rp > 0:
            right_point = rp

    # estimate missing side using typical lane width
    estimated_lane_width = int(width * 0.30)
    if left_point != -1 and right_point == -1:
        right_point = min(width - 1, left_point + estimated_lane_width)
    elif right_point != -1 and left_point == -1:
        left_point = max(0, right_point - estimated_lane_width)

    return left_point, right_point, center


def calculate_control_signal(*args, **kwargs) -> Tuple[float, float]:
    """
    Compatibility wrapper:
      - calculate_control_signal(image, draw=None)
      - calculate_control_signal(current_speed, image)

    Returns (throttle, steering) where steering in [-1,1], throttle in [0,1].
    """
    global frame_id
    frame_id += 1

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

    # birdview
    bird_img = birdview_transform(image)

    # lane detection
    left_point, right_point, im_center = find_lane_lines(bird_img)

    steering = 0.0
    lanes_found = (left_point != -1 and right_point != -1)

    try:
        if lanes_found:
            center_point = (left_point + right_point) // 2
            center_diff = float(center_point - im_center)
            steering = -pid(center_diff)
            steering = clamp(steering, -0.9, 0.9)
        else:
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

    # base speed from current_speed if provided
    base_speed = 0.5
    if current_speed is not None:
        try:
            cs = float(current_speed)
            if cs > 0.0:
                base_speed = clamp(1.2 - cs, 0.3, 1.0)
        except Exception:
            base_speed = 0.5

    # simple throttle rule: reduce with steering
    steer_mag = min(abs(steering), 1.0)
    throttle = base_speed * (1.0 - 0.5 * steer_mag)
    throttle = clamp(throttle, 0.0, 1.0)

    # draw overlays if requested
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