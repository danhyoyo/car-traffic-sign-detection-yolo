import cv2
import numpy as np
import time

frame_id = 0


class PID(object):
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0):
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
        print("pid: ", control_signal, "center_diff: ", state)
        return control_signal


# -----------------------------
# Khai báo bộ điều khiển PID lái
# -----------------------------
pid = PID(kp=1.0, ki=0.0, kd=0.05, setpoint=0.0)


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


def calculate_control_signal(current_speed, image):
    global frame_id
    frame_id += 1

    steering_angle = 0.02
    throttle = 0.7  # tốc độ mặc định

    left_point, right_point, im_center = find_lane_lines(image)

    if left_point != -1 and right_point != -1:
        # --- Tính độ lệch ---
        center_point = (right_point + left_point) // 2
        center_diff = center_point - im_center  # sai số

        # --- Dùng PID ---
        steering_angle = -pid(center_diff)
        print("steering_angle", steering_angle)

        # --- Giảm tốc nếu cua gắt ---
        throttle = 1.0 * (1.0 - min(abs(steering_angle), 1.0) * 0.7)

    return throttle, steering_angle
