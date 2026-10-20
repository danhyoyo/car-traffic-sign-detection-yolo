"""
lane_line_detection.py

Sửa để tránh steering luôn clamp về -1/1:
- chuẩn hóa lỗi pixel -> [-1,1] trước khi vào PID
- PID được tune cho lỗi đã chuẩn hóa
- smoothing (low-pass) cho steering
- giới hạn khi không đủ thông tin lane
- hiển thị thông tin debug lên draw nếu được truyền
"""
import time
from typing import Tuple, Optional

import cv2
import numpy as np

# --- state ---
_frame_id = 0
_last_steering = 0.0


class PID:
    """PID nhận input đã chuẩn hóa (x in approx -1..1)."""
    def __init__(self, kp=0.05, ki=0.00, kd=0.02, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.time_func = time.monotonic
        self.last_time = self.time_func()
        self.last_state = None
        self.integral = 0.0

    def __call__(self, state: float) -> float:
        # state expected normalized (-~1..1)
        error = self.setpoint - state
        now = self.time_func()
        dt = now - self.last_time
        if dt <= 0.0:
            dt = 1e-6
        d_state = 0.0 if self.last_state is None else (state - self.last_state)

        # integral (with small ki)
        self.integral += self.ki * error * dt
        proportional = self.kp * error
        derivative = -self.kd * d_state / dt

        out = proportional + self.integral + derivative

        self.last_state = state
        self.last_time = now
        return out


# PID tuned for normalized error
_pid = PID(kp=0.5, ki=0.0, kd=0.02, setpoint=0.0)


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        xx = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xx))


def find_lane_lines(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Tìm điểm trái/phải trên một hàng gần đáy ảnh như trước.
    Trả về (left_x, right_x, center_x). Nếu không tìm thấy, trả -1.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 120, 200)

    h, w = edges.shape
    interested_line_y = int(h * 0.9)
    line = edges[interested_line_y, :]

    left_point = -1
    right_point = -1
    lane_width = int(w * 0.18)  # dự đoán width theo tỉ lệ ảnh
    center = w // 2

    for x in range(center, 0, -1):
        if line[x] > 0:
            left_point = x
            break

    for x in range(center + 1, w):
        if line[x] > 0:
            right_point = x
            break

    # fallback hàng khác
    if left_point == -1 or right_point == -1:
        alt_y = int(h * 0.85)
        if alt_y >= 0 and alt_y < h:
            line2 = edges[alt_y, :]
            if left_point == -1:
                for x in range(center, 0, -1):
                    if line2[x] > 0:
                        left_point = x
                        break
            if right_point == -1:
                for x in range(center + 1, w):
                    if line2[x] > 0:
                        right_point = x
                        break

    # dự đoán nếu chỉ thấy một bên
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    return left_point, right_point, center


def birdview_transform(img: np.ndarray) -> np.ndarray:
    """
    Bird-view warping (giữ nguyên size input).
    """
    h, w = img.shape[:2]
    src = np.float32([[0, h], [w, h], [0, int(h * 0.4)], [w, int(h * 0.4)]])
    dst = np.float32([[int(w * 0.2), h], [int(w * 0.8), h],
                      [-int(w * 0.25), 0], [w + int(w * 0.25), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped


def _compute_throttle(base_speed: float, steering: float, lanes_found: bool) -> float:
    """
    Throttle nhẹ nhàng: giảm trọng số ảnh hưởng steering.
    """
    steer_mag = min(abs(steering), 1.0)
    steer_factor = 0.35  # giảm tác động của steering lên throttle
    throttle = base_speed * (1.0 - steer_mag * steer_factor)
    if not lanes_found:
        throttle = min(throttle, base_speed * 0.5)
    # đặt min throttle để không quá chậm
    min_throttle = 0.32
    return clamp(max(throttle, min_throttle), 0.0, 1.0)


def calculate_control_signal(*args, **kwargs) -> Tuple[float, float]:
    """
    Hỗ trợ các signature:
    - calculate_control_signal(image, draw=None)
    - calculate_control_signal(current_speed, image)
    """
    global _frame_id, _last_steering
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

    # detect lane points
    try:
        left_point, right_point, im_center = find_lane_lines(image)
    except Exception:
        return 0.0, 0.0

    lanes_found = (left_point != -1 and right_point != -1)

    steering = 0.0
    center_diff_px = 0
    center_diff_norm = 0.0
    im_width = image.shape[1]

    if lanes_found:
        center_point = (left_point + right_point) // 2
        center_diff_px = center_point - im_center  # pixel error
        # chuẩn hóa theo nửa chiều rộng ảnh -> approx in [-1,1]
        denom = float(im_center) if im_center != 0 else 1.0
        center_diff_norm = center_diff_px / (denom)
        # feed normalized error to PID
        steering_unclamped = -_pid(center_diff_norm)
        # clamp safety
        steering_unclamped = clamp(steering_unclamped, -1.0, 1.0)
        # smoothing (low-pass)
        alpha = 0.6
        steering = alpha * steering_unclamped + (1 - alpha) * _last_steering
        _last_steering = steering
    else:
        # if partial info, do gentle steering only
        if left_point != -1:
            approx_center_px = (left_point + (left_point + int(im_width * 0.18))) // 2
            center_diff_px = approx_center_px - im_center
            center_diff_norm = center_diff_px / float(im_center if im_center != 0 else 1.0)
            steering_unclamped = -_pid(center_diff_norm)
            steering_unclamped = clamp(steering_unclamped, -0.6, 0.6)
            alpha = 0.5
            steering = alpha * steering_unclamped + (1 - alpha) * _last_steering
            _last_steering = steering
        elif right_point != -1:
            approx_center_px = ((right_point - int(im_width * 0.18)) + right_point) // 2
            center_diff_px = approx_center_px - im_center
            center_diff_norm = center_diff_px / float(im_center if im_center != 0 else 1.0)
            steering_unclamped = -_pid(center_diff_norm)
            steering_unclamped = clamp(steering_unclamped, -0.6, 0.6)
            alpha = 0.5
            steering = alpha * steering_unclamped + (1 - alpha) * _last_steering
            _last_steering = steering
        else:
            # no lanes: keep previous steering gradually decay to 0
            steering = 0.6 * _last_steering
            _last_steering = steering

    # base speed (đơn giản, không phụ thuộc current_speed)
    base_speed = 0.9
    throttle = _compute_throttle(base_speed, steering, lanes_found)

    # debug overlay lên draw nếu có
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

            info = f"pix_err:{center_diff_px:+d} norm:{center_diff_norm:+.3f} steer:{steering:+.3f}"
            cv2.putText(draw, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            txt = f"T:{throttle:.3f} S:{steering:.3f}"
            cv2.putText(draw, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        except Exception:
            pass

    # final clamp safety
    steering = clamp(steering, -1.0, 1.0)
    throttle = clamp(throttle, 0.0, 1.0)
    return float(throttle), float(steering)