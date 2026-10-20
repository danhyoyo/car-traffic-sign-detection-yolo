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
    def __init__(self, kp=0.00, ki=0.0, kd=0.00, setpoint=0.0):
        # --- Tham số PID ---
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        # --- Thời gian ---
        self.time_func = time.monotonic
        self.last_time = self.time_func()

        # --- Trạng thái ---
        self.last_state = None
        self.integral = 0.0

    def __call__(self, state):
        """
        Tính tín hiệu điều khiển PID.
        state: giá trị hiện tại (ví dụ: center_diff)
        """
        # --- Tính sai số ---
        error = self.setpoint - state
        d_state = 0.0 if self.last_state is None else (state - self.last_state)

        # --- Lấy thời gian ---
        now = self.time_func()
        d_time = now - self.last_time
        if d_time <= 0.0:
            d_time = 1e-12  # tránh chia 0

        # --- Thành phần PID ---
        self.integral += self.ki * error * d_time
        proportional = self.kp * error
        derivative = -self.kd * d_state / d_time

        # --- Tín hiệu điều khiển ---
        control_signal = proportional + self.integral + derivative

        # --- Cập nhật ---
        self.last_state = state
        self.last_time = now
        print("pid: ", control_signal, "center_diff: ", state, "error: ", error)
        return control_signal

pid = PID(kp=0.03, ki=0.00, kd=0.05, setpoint=0.0)  # tuned default



def clamp(x: float, lo: float, hi: float) -> float:
    try:
        xx = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xx))


def find_lane_lines(image):
    """
    Tìm hai biên trái/phải của làn đường (đơn giản)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 150, 200)

    height, width = edges.shape
    interested_line_y = int(height * 0.9)
    line_data = edges[interested_line_y, :]

    left_point = -1
    right_point = -1
    lane_width = 100
    center = width // 2

    # tìm trái
    for x in range(center, 0, -1):
        if line_data[x] > 0:
            left_point = x
            break

    # tìm phải
    for x in range(center, width):
        if line_data[x] > 0:
            right_point = x
            break

    # Dự đoán nếu chỉ thấy 1 bên
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
    elif right_point != -1 and left_point == -1:
        left_point = right_point - lane_width
    print("vi tri", left_point, right_point, center)
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


# --- Thêm ở đầu file, global state để làm mượt throttle ---
_last_throttle = 0.0

def _compute_throttle(base_speed: float, steering: float, lanes_found: bool) -> float:
    """
    Compute throttle (0..1) from base_speed, steering magnitude and whether lanes are visible.
    Added:
    - reduced steering impact (steer_factor)
    - minimum throttle (min_throttle)
    - exponential smoothing using _last_throttle and alpha
    """
    global _last_throttle
    # giảm nhạy với steering: hệ số thấp hơn (trước là 0.85)
    steer_factor = 0.6  # <--- giảm từ 0.85 xuống 0.5 (thử điều chỉnh)
    steer_mag = min(abs(steering), 1.0)
    raw = base_speed * (1.0 - steer_mag * steer_factor)

    # nếu lane không thấy, giảm tốc nhưng không quá thấp
    if not lanes_found:
        raw = min(raw, base_speed * 0.5)

    # đặt min throttle để không đi quá chậm
    min_throttle = 0.6  # <-- tăng lên nếu muốn nhanh hơn
    raw = max(raw, min_throttle)

    # smoothing (low-pass)
    alpha = 0.45  # 0..1, nhỏ => mượt hơn. thử 0.3..0.6
    throttle = alpha * raw + (1.0 - alpha) * _last_throttle

    # cập nhật trạng thái
    _last_throttle = throttle

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
    bird_img = birdview_transform(image)
    try:
        left_point, right_point, im_center = find_lane_lines(bird_img)
    except Exception:
        return 0.0, 0.0

    steering = -0.0001
    lanes_found = (left_point != -1 and right_point != -1)

    if lanes_found:
        center_point = (left_point + right_point) // 2
        center_diff = center_point - im_center  # positive -> lane center is to the right 
        steering = -pid(center_diff)
        steering =clamp(steering, -0.9, 0.9)
    else:
        if left_point != -1:
            steering = -pid((left_point + 60) - im_center)
            steering = lamp(steering, -0.6, 0.6)
        elif right_point != -1:
            steering = -pid((right_point - 60) - im_center)
            steering = lamp(steering, -0.6, 0.6)
        else:
            steering = 0.0
    print("steering_angle", steering)
    base_speed = 0.7
    if current_speed is not None:
        try:
            cs = float(current_speed)
            if cs > 0.0:
                base_speed = clamp(1.2 - cs, 0.3, 1.0)
        except Exception:
            base_speed = 0.7

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