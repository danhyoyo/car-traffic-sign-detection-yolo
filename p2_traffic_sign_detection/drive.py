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
from pid_controller import PIDController
from behavior_planner import BehaviorPlanner

# Initalize traffic sign classifier (kept global so subprocess can load it)
traffic_sign_model = cv2.dnn.readNetFromONNX("traffic_sign_classifier_lenet_v3.onnx")

# Global queues for inter-process communication
g_image_queue = Queue(maxsize=3)   # images for sign detection
g_sign_queue = Queue(maxsize=3)    # detected signs from sign detection process

def process_traffic_sign_loop(g_image_queue, g_sign_queue):
    """
    Runs in a separate process: continuously takes images from g_image_queue,
    runs detection, draws a visualization and pushes detected signs to g_sign_queue.
    """
    # Load model inside subprocess as well (some platforms require re-loading)
    model = traffic_sign_model

    while True:
        if g_image_queue.empty():
            time.sleep(0.05)
            continue
        image = g_image_queue.get()

        # Prepare visualization image
        draw = image.copy()
        # Detect traffic signs
        signs = detect_traffic_signs(image, model, draw=draw)

        # Put latest signs into g_sign_queue (keep only most recent)
        try:
            # non-blocking: if queue full, remove oldest
            if g_sign_queue.full():
                try:
                    _ = g_sign_queue.get_nowait()
                except:
                    pass
            g_sign_queue.put_nowait(signs)
        except:
            pass

        # Show the result to a window (for debugging/visual)
        cv2.imshow("Traffic signs (detector)", draw)
        cv2.waitKey(1)


# Controller and planner instances (in main process)
pid = PIDController(kp=0.012, ki=0.0006, kd=0.0018, output_limits=(-1.0, 1.0))
planner = BehaviorPlanner(stop_duration=3.0, turn_duration=2.5, turn_offset_pixels_fraction=0.12)

# Smoothing state
_prev_throttle = 0.0
_prev_steering = 0.0
STEER_EMA_ALPHA = 0.6   # how much new steering replaces previous (0..1)
THROTTLE_EMA_ALPHA = 0.4

async def process_image(websocket, path):
    global _prev_throttle, _prev_steering, pid, planner

    async for message in websocket:
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        draw = image.copy()

        # 1) Lane detection pipeline
        lanes = find_lane_lines(image)
        lanes_bird = birdview_transform(lanes)
        draw_bird = birdview_transform(draw)

        left_pt, right_pt = find_left_right_points(lanes_bird, draw=draw_bird)

        im_center = image.shape[1] // 2
        if left_pt != -1 and right_pt != -1:
            center_point = (left_pt + right_pt) // 2
            center_diff = im_center - center_point
        else:
            # fallback to zero if cannot see both lines
            center_diff = 0

        # 2) Send image to sign detection process (non-blocking)
        if not g_image_queue.full():
            try:
                g_image_queue.put_nowait(image.copy())
            except:
                pass

        # 3) Try to read latest signs from g_sign_queue (take the most recent)
        signs = []
        try:
            # drain queue to get the most recent element
            while not g_sign_queue.empty():
                signs = g_sign_queue.get_nowait()
        except:
            # ignore queue errors
            pass

        # 4) Behavior planning -> desired throttle multiplier and desired center diff
        throttle_mul, desired_center_diff, state = planner.step(signs, image.shape[1])

        # 5) PID controller: compute steering from (center_diff - desired_center_diff)
        error = center_diff - desired_center_diff
        steering = pid.update(error, current_time=time.time())

        # Smooth steering and throttle
        base_throttle = 0.5  # base throttle (when planner returns throttle 1.0, etc.)
        target_throttle = base_throttle * throttle_mul
        _prev_throttle = _prev_throttle * (1.0 - THROTTLE_EMA_ALPHA) + target_throttle * THROTTLE_EMA_ALPHA
        smoothed_throttle = float(_prev_throttle)

        _prev_steering = _prev_steering * (1.0 - STEER_EMA_ALPHA) + steering * STEER_EMA_ALPHA
        smoothed_steering = float(_prev_steering)

        # Draw debug info
        if left_pt != -1:
            cv2.circle(draw_bird, (left_pt, int(draw_bird.shape[0] * 0.9)), 7, (255, 255, 0), -1)
        if right_pt != -1:
            cv2.circle(draw_bird, (right_pt, int(draw_bird.shape[0] * 0.9)), 7, (0, 255, 0), -1)

        # Put text about current state and numbers
        info = f"State:{state} Throttle:{smoothed_throttle:.2f} Steer:{smoothed_steering:.3f}"
        cv2.putText(draw, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw bird view and compose small overlay
        small_bird = cv2.resize(draw_bird, (320, 240))
        h, w = small_bird.shape[:2]
        draw[0:h, 0:w] = small_bird

        # overlay detected signs info (for convenience)
        y = 60
        for s in signs:
            cls, x, y0, w0, h0 = s
            cv2.putText(draw, f"{cls}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
            y += 25

        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Send back throttle and steering angle
        message = json.dumps({"throttle": smoothed_throttle, "steering": smoothed_steering})
        await websocket.send(message)


async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue, g_sign_queue))
    p.daemon = True
    p.start()
    asyncio.run(main())