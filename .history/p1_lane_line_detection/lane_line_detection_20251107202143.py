import cv2
import numpy as np
import time

frame_id = 0


class PID(object):
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.time_func = time.monotonic
        self.last_time = self.time_func()
        self.last_state = None
        self.integral = 0.0

    def __call__(self, state):
        error = self.setpoint - state
        d_state = 0.0 if self.last_state is None else (state - self.last_state)
        now = self.time_func()
        d_time = now - self.last_time
        if d_time <= 0.0:
            d_time = 1e-12

        self.integral += self.ki * error * d_time
        proportional = self.kp * error
        derivative = -self.kd * d_state / d_time
        control_signal = proportional + self.integral + derivative
        self.last_state = state
        self.last_time = now
        return control_signal


pid = PID(kp=0.05, ki=0.0003, kd=0.002, setpoint=0.0)


def birdview_transform(img):
    """Chuyển ảnh sang bird-view"""
    IMAGE_H, IMAGE_W = img.shape[:2]
    src = np.float32([
        [0, IMAGE_H],
        [IMAGE_W, IMAGE_H],
        [0, IMAGE_H * 0.4],
        [IMAGE_W, IMAGE_H * 0.4]
    ])
    dst = np.float32([
        [240, IMAGE_H],
        [IMAGE_W - 240, IMAGE_H],
        [-160, 0],
        [IMAGE_W + 160, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped


def find_lane_lines(image):
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

    for x in range(center, 0, -1):
        if line_data[x] > 0:
            left_point = x
            break

    for x in range(center, width):
        if line_data[x] > 0:
            right_point = x
            break

    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
    elif right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    return left_point, right_point, center


def calculate_control_signal(current_speed, image):
    global frame_id
    frame_id += 1

    steering_angle = 0.0
    throttle = 0.5

    # --- Bird View ---
    bird_img = birdview_transform(image)

    # --- Tìm vạch ---
    left_point, right_point, im_center = find_lane_lines(bird_img)

    if left_point != -1 and right_point != -1:
        center_point = (right_point + left_point) // 2
        center_diff = center_point - im_center
        steering_angle = -pid(center_diff)
        throttle = 0.5 * (1.0 - min(abs(steering_angle), 1.0) * 0.5)

    return throttle, steering_angle
