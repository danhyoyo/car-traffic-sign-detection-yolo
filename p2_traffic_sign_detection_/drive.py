"""
drive.py

Updated drive server with smoother control to reduce "cứng" (stiff / jerky) motion.

Key improvements vs previous version:
- Steering rate limiter (limits how fast steering can change per second).
- Pre-filtering (low-pass) of the PID input (state) to reduce derivative noise.
- Deadzone around small center errors to avoid unnecessary twitching.
- Throttle ramp rate limiting and smoother EMA parameters.
- Reset PID on planner state changes to avoid integral windup (kept).
- Tuned default PID gains and smoothing defaults (you can still tune these).
"""

import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue

import cv2
import numpy as np
import websockets
from PIL import Image

from lane_line_detection import find_lane_lines, birdview_transform, find_left_right_points
from traffic_sign_detection import detect_traffic_signs

# -------------------------
# PID class (user provided)
# -------------------------
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
        Note: error = setpoint - state
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
        # Note: the user snippet integrates ki * error * dt into self.integral
        self.integral += self.ki * error * d_time
        proportional = self.kp * error
        derivative = -self.kd * d_state / d_time

        # --- Tín hiệu điều khiển ---
        control_signal = proportional + self.integral + derivative

        # --- Cập nhật ---
        self.last_state = state
        self.last_time = now

        return control_signal

# -------------------------
# Behavior Planner (FSM)
# -------------------------
class BehaviorPlanner:
    STATE_FOLLOW = "FOLLOW"
    STATE_STOPPED = "STOPPED"
    STATE_TURN_LEFT = "TURN_LEFT"
    STATE_TURN_RIGHT = "TURN_RIGHT"

    def __init__(self, stop_duration=3.0, turn_duration=2.0, turn_offset_frac=0.12):
        self.state = self.STATE_FOLLOW
        self.stop_until = 0.0
        self.turn_until = 0.0
        self.forbidden_left_until = 0.0
        self.forbidden_right_until = 0.0

        self.stop_duration = stop_duration
        self.turn_duration = turn_duration
        self.turn_offset_frac = turn_offset_frac

    def _now(self):
        return time.time()

    def step(self, signs, image_width):
        now = self._now()
        classes = [s[0] for s in signs] if signs else []

        if 'no_left' in classes:
            self.forbidden_left_until = now + 3.0
        if 'no_right' in classes:
            self.forbidden_right_until = now + 3.0

        # stop has priority
        if 'stop' in classes:
            if self.state != self.STATE_STOPPED:
                self.state = self.STATE_STOPPED
                self.stop_until = now + self.stop_duration
        else:
            if self.state == self.STATE_STOPPED and now >= self.stop_until:
                self.state = self.STATE_FOLLOW

            if self.state != self.STATE_STOPPED:
                if 'left' in classes and now >= self.forbidden_left_until:
                    self.state = self.STATE_TURN_LEFT
                    self.turn_until = now + self.turn_duration
                elif 'right' in classes and now >= self.forbidden_right_until:
                    self.state = self.STATE_TURN_RIGHT
                    self.turn_until = now + self.turn_duration
                else:
                    if self.state == self.STATE_TURN_LEFT and now >= self.turn_until:
                        self.state = self.STATE_FOLLOW
                    if self.state == self.STATE_TURN_RIGHT and now >= self.turn_until:
                        self.state = self.STATE_FOLLOW

        img_w = image_width
        offset_px = int(img_w * self.turn_offset_frac)

        if self.state == self.STATE_STOPPED:
            throttle_mul = 0.0
            desired_center_diff = 0
        elif self.state == self.STATE_TURN_LEFT:
            throttle_mul = 0.35
            desired_center_diff = +offset_px
        elif self.state == self.STATE_TURN_RIGHT:
            throttle_mul = 0.35
            desired_center_diff = -offset_px
        else:
            throttle_mul = 1.0
            desired_center_diff = 0

        return throttle_mul, desired_center_diff, self.state

# -------------------------
# Multiprocessing queues
# -------------------------
g_image_queue = Queue(maxsize=4)
g_sign_queue = Queue(maxsize=4)

# -------------------------
# Traffic sign detector process
# -------------------------
def process_traffic_sign_loop(g_image_queue, g_sign_queue, model_path="traffic_sign_classifier_lenet_v3.onnx"):
    try:
        model = cv2.dnn.readNetFromONNX(model_path)
    except Exception as e:
        print("Traffic sign model load failed in subprocess:", e)
        model = None

    while True:
        if g_image_queue.empty():
            time.sleep(0.03)
            continue
        img = g_image_queue.get()
        draw = img.copy()
        signs = []
        try:
            if model is not None:
                signs = detect_traffic_signs(img, model, draw=draw)
        except Exception as e:
            print("Sign detection error:", e)
            signs = []

        try:
            if g_sign_queue.full():
                try:
                    _ = g_sign_queue.get_nowait()
                except:
                    pass
            g_sign_queue.put_nowait(signs)
        except:
            pass

        try:
            cv2.imshow("Sign detector (subprocess)", draw)
            cv2.waitKey(1)
        except Exception:
            pass

# -------------------------
# Controller & Planner (tuning)
# -------------------------
# Use user's PID structure but tuned values:
pid = PID(kp=0.03, ki=0.0007, kd=0.004, setpoint=0.0)
planner = BehaviorPlanner(stop_duration=0.1, turn_duration=0.05, turn_offset_frac=0.12)

# smoothing + filtering configuration
_prev_throttle = 0.0
_prev_steering = 0.0
_prev_steering_unfiltered = 0.0
_prev_state_filtered = 0.0
_prev_control_time = time.monotonic()

# EMA alphas (lower alpha -> smoother / slower response)
STEER_EMA_ALPHA = 0.95   # smoothing applied to final steering (smaller -> smoother)
THROTTLE_EMA_ALPHA = 0.7

# steering rate limit (units: steering_value per second)
STEERING_MAX_RATE = 20   # max change in steering per second (tune lower if still jerky)
THROTTLE_MAX_RATE = 1   # max change in throttle per second

# PID input low-pass filter alpha (0..1). Higher -> follow faster, lower -> smoother
STATE_FILTER_ALPHA = 0.0

# small deadzone in pixels to avoid micro corrections
STEERING_DEADZONE_PX = 0

# absolute steering output clamp
STEERING_OUTPUT_LIMIT = 0.4

_prev_planner_state = None

# -------------------------
# Websocket handler
# -------------------------
async def process_image(websocket, path):
    """
    Main per-frame processing coroutine for the simulator websocket.
    """
    global _prev_throttle, _prev_steering, _prev_steering_unfiltered, _prev_state_filtered, _prev_control_time, _prev_planner_state, pid

    async for message in websocket:
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw = image.copy()

        # Lane detection
        lanes = find_lane_lines(image)
        lanes_bird = birdview_transform(lanes)
        draw_bird = birdview_transform(draw)

        left_pt, right_pt = find_left_right_points(lanes_bird, draw=draw_bird)
        im_center = image.shape[1] // 2

        if left_pt != -1 and right_pt != -1:
            lane_center = (left_pt + right_pt) // 2
            center_diff = im_center - lane_center
        else:
            center_diff = 0

        # Send image to sign detection process (non-blocking)
        try:
            if not g_image_queue.full():
                g_image_queue.put_nowait(image.copy())
        except:
            pass

        # Read latest signs (drain queue to get most recent)
        signs = []
        try:
            while not g_sign_queue.empty():
                signs = g_sign_queue.get_nowait()
        except:
            pass

        # Behavior planning
        throttle_mul, desired_center_diff, state = planner.step(signs, image.shape[1])

        # Reset PID integrator / memory when planner state changes to avoid windup
        if _prev_planner_state is None:
            _prev_planner_state = state
        if state != _prev_planner_state:
            pid.last_state = None
            pid.integral = 0.0
            pid.last_time = pid.time_func()
            _prev_planner_state = state

        # Prepare PID input and filtering
        # We want to drive (center_diff - desired_center_diff) -> 0
        raw_state_for_pid = float(center_diff - desired_center_diff)

        # Apply small deadzone
        if abs(raw_state_for_pid) < STEERING_DEADZONE_PX:
            raw_state_for_pid = 0.0

        # Low-pass filter the state passed into PID to reduce derivative noise
        _prev_state_filtered = STATE_FILTER_ALPHA * _prev_state_filtered + (1.0 - STATE_FILTER_ALPHA) * raw_state_for_pid
        state_for_pid_filtered = _prev_state_filtered

        # Compute steering from PID (this uses user's PID class)
        try:
            steering_unfiltered = float(pid(state_for_pid_filtered))
        except Exception:
            steering_unfiltered = 0.0

        # Clamp absolute output
        steering_unfiltered = max(-STEERING_OUTPUT_LIMIT, min(STEERING_OUTPUT_LIMIT, steering_unfiltered))

        # Steering rate limiting (based on wall-clock dt)
        now = time.monotonic()
        dt = now - _prev_control_time
        if dt <= 0.0:
            dt = 1e-6
        _prev_control_time = now

        max_delta = STEERING_MAX_RATE * dt
        steering_delta = steering_unfiltered - _prev_steering_unfiltered
        if steering_delta > max_delta:
            steering_unfiltered_limited = _prev_steering_unfiltered + max_delta
        elif steering_delta < -max_delta:
            steering_unfiltered_limited = _prev_steering_unfiltered - max_delta
        else:
            steering_unfiltered_limited = steering_unfiltered

        _prev_steering_unfiltered = steering_unfiltered_limited

        # Final EMA smoothing for steering (adds smoothness across frames)
        _prev_steering = _prev_steering * (1.0 - STEER_EMA_ALPHA) + steering_unfiltered_limited * STEER_EMA_ALPHA
        smoothed_steering = float(_prev_steering)

        # Compose throttle (base throttle multiplied by planner suggestion)
        base_throttle = 0.61  # slightly lower base to help smoothness
        target_throttle = base_throttle * throttle_mul

        # Throttle ramp limiting
        max_throttle_delta = THROTTLE_MAX_RATE * dt
        throttle_delta = target_throttle - _prev_throttle
        if throttle_delta > max_throttle_delta:
            target_throttle_limited = _prev_throttle + max_throttle_delta
        elif throttle_delta < -max_throttle_delta:
            target_throttle_limited = _prev_throttle - max_throttle_delta
        else:
            target_throttle_limited = target_throttle

        # EMA smoothing for throttle
        _prev_throttle = _prev_throttle * (1.0 - THROTTLE_EMA_ALPHA) + target_throttle_limited * THROTTLE_EMA_ALPHA
        smoothed_throttle = float(_prev_throttle)

        # Draw debug info
        status = f"State:{state} Throttle:{smoothed_throttle:.2f} Steer:{smoothed_steering:.3f}"
        cv2.putText(draw, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # overlay birdview small window
        try:
            small_bird = cv2.resize(draw_bird, (320, 240))
            h, w = small_bird.shape[:2]
            draw[0:h, 0:w] = small_bird
        except Exception:
            pass

        # overlay signs text
        y_txt = 60
        try:
            for s in signs:
                cls, x, y0, w0, h0 = s
                cv2.putText(draw, f"{cls}", (10, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
                y_txt += 24
        except Exception:
            pass

        # show result window
        try:
            cv2.imshow("Result", draw)
            cv2.waitKey(1)
        except Exception:
            pass

        # send control back to simulator
        out_msg = json.dumps({"throttle": smoothed_throttle, "steering": smoothed_steering})
        await websocket.send(out_msg)


        # send response
        send_msg = json.dumps({
            "throttle": float(smoothed_throttle),
            "steering": float(smoothed_steering)
        })
        print("SEND:", send_msg)
        try:
            await websocket.send(send_msg)
        except Exception as e:
            print("Failed to send message:", e)
        # Gửi lại JSON (chỉ gửi các giá trị cần thiết)

# -------------------------
# Main entrypoint
# -------------------------
async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue, g_sign_queue))
    p.daemon = True
    p.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")