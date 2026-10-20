# pid_controller.py
import time

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

        return control_signal