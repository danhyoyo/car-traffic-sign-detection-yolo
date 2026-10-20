"""
drive.py

WebSocket server that receives images and returns throttle/steering.
This version includes:
- traffic sign confirmation (candidate -> confirmed) logic,
- execute_after: after a sign is confirmed, wait EXECUTE_AFTER_SECONDS before applying its action,
  so the car delays executing the action even after confirmation.
- communication with the sign detection process via two multiprocessing.Queues:
    g_image_queue -> images sent to detector
    g_sign_queue  -> detections received from detector (payloads)

Adjust the parameters near the top to tune confirmation and execution delays.
"""
import asyncio
import base64
import json
from io import BytesIO
from multiprocessing import Process, Queue
import os
import time

import cv2
import numpy as np
import websockets
from PIL import Image

# Try to import calculate_control_signal from lane_line_detection
try:
    from lane_line_detection import calculate_control_signal
except Exception:
    calculate_control_signal = None

# Import sign process helper (expects process_traffic_sign_loop to accept an output queue)
from traffic_sign_detection import process_traffic_sign_loop, TrafficSignDetector

# Configuration (tweak to your needs)
TRAFFIC_SIGN_MODEL = os.getenv("TRAFFIC_SIGN_MODEL", "traffic_sign_classifier_lenet_v3.onnx")
IMAGE_QUEUE_MAXSIZE = 5
SIGN_QUEUE_MAXSIZE = 8

# Confirmation/delay params
CONFIRM_DELAY_SECONDS = 0.1        # require detection to persist >= this to confirm
MIN_CONSISTENT_DETECTIONS = 3       # or require this many consistent detections
MIN_BBOX_AREA = 2800                # min bbox area (px^2) to consider as "near"
EXECUTE_AFTER_SECONDS = 0.3           # after confirmation, wait this long before executing action

# Active durations of actions (once executed)
DURATION_BY_SIGN = {
    'stop': 2.5,
    'left': 1.1,
    'right': 1.1,
    'straight': 1.0,
    'no_left': 1.0,
    'no_right': 1.0
}

# Queues for inter-process communication
g_image_queue = Queue(maxsize=IMAGE_QUEUE_MAXSIZE)
g_sign_queue = Queue(maxsize=SIGN_QUEUE_MAXSIZE)

# Active sign state (after execution start)
_sign_state = {
    'name': None,          # sign name being actively enforced
    'exec_start': 0.0,     # time execution started (execute_at)
    'duration': 0.0,
    'score': 0.0
}

# Candidate sign state (before execution)
_sign_candidate = {
    'name': None,
    'first_seen_ts': 0.0,
    'last_seen_ts': 0.0,
    'count': 0,
    'score': 0.0,
    'bbox': None
}


def _reset_candidate():
    global _sign_candidate
    _sign_candidate = {
        'name': None,
        'first_seen_ts': 0.0,
        'last_seen_ts': 0.0,
        'count': 0,
        'score': 0.0,
        'bbox': None
    }


def _reset_active():
    global _sign_state
    _sign_state = {'name': None, 'exec_start': 0.0, 'duration': 0.0, 'score': 0.0}


def _promote_candidate_to_confirmed(name, score, bbox):
    """
    When candidate is confirmed, schedule execution at now + EXECUTE_AFTER_SECONDS.
    We store exec_start as the time execution will begin.
    """
    global _sign_state
    exec_at = time.time() + EXECUTE_AFTER_SECONDS
    dur = DURATION_BY_SIGN.get(name, 1.6)
    # We'll set exec_start to exec_at; later _active_sign checks now against exec_start.
    _sign_state = {'name': name, 'exec_start': exec_at, 'duration': dur, 'score': score}


def _update_sign_state_from_queue():
    """
    Non-blocking read of g_sign_queue to update _sign_candidate and schedule execution only
    when confirmation criteria are met. Confirmed signs are scheduled to execute after EXECUTE_AFTER_SECONDS.
    """
    global _sign_candidate, _sign_state
    try:
        while not g_sign_queue.empty():
            payload = g_sign_queue.get_nowait()
            if not payload or 'name' not in payload:
                continue
            name = payload.get('name')
            score = float(payload.get('score', 0.0))
            bbox = payload.get('bbox', None)
            ts = float(payload.get('ts', time.time()))

            if bbox is None:
                continue
            x, y, w, h = bbox
            area = int(w) * int(h)
            # ignore detections that are too small (not near)
            if area < MIN_BBOX_AREA:
                continue

            # Candidate aggregation
            if _sign_candidate['name'] == name:
                _sign_candidate['last_seen_ts'] = ts
                _sign_candidate['count'] += 1
                if score > _sign_candidate['score']:
                    _sign_candidate['score'] = score
                _sign_candidate['bbox'] = bbox
            else:
                # New candidate (different sign)
                _sign_candidate['name'] = name
                _sign_candidate['first_seen_ts'] = ts
                _sign_candidate['last_seen_ts'] = ts
                _sign_candidate['count'] = 1
                _sign_candidate['score'] = score
                _sign_candidate['bbox'] = bbox

            # Check confirmation criteria
            elapsed = _sign_candidate['last_seen_ts'] - _sign_candidate['first_seen_ts']
            if (_sign_candidate['count'] >= MIN_CONSISTENT_DETECTIONS) or (elapsed >= CONFIRM_DELAY_SECONDS):
                # Promote: schedule execution after EXECUTE_AFTER_SECONDS
                _promote_candidate_to_confirmed(_sign_candidate['name'], _sign_candidate['score'], _sign_candidate['bbox'])
                # reset candidate to avoid re-promoting
                _reset_candidate()
    except Exception:
        pass


def _active_sign():
    """
    Return active sign name if current time is within execution window:
    exec_start <= now <= exec_start + duration
    If a sign is scheduled (exec_start in the future) but not yet reached, return None.
    """
    now = time.time()
    name = _sign_state.get('name')
    if not name:
        return None
    exec_start = _sign_state.get('exec_start', 0.0)
    duration = _sign_state.get('duration', 0.0)
    # Execution hasn't started yet
    if now < exec_start:
        return None
    # Execution ongoing?
    if now <= exec_start + duration:
        return name
    # Execution expired
    _reset_active()
    return None


def apply_sign_action(base_throttle: float, base_steering: float):
    """
    Modify throttle and steering according to active sign (if any).
    """
    name = _active_sign()
    throttle = float(base_throttle)
    steering = float(base_steering)

    if name is None:
        return throttle, steering

    if name == 'left':
        steering = -0.9
        throttle = min(throttle, 0.85)
    elif name == 'right':
        steering = 0.9
        throttle = min(throttle, 0.85)
    elif name == 'straight':
        steering = 0.0
        throttle = max(throttle, 9.8)
    elif name == 'stop':
        steering = 0.0
        throttle = 0.0
    elif name == 'no_left':
        if steering < -0.2:
            steering = -0.2
    elif name == 'no_right':
        if steering > 0.2:
            steering = 0.2

    throttle = max(0.0, min(1.0, throttle))
    steering = max(-0.9, min(0.9, steering))
    return throttle, steering


# helper clamp
def clamp_val(x, lo, hi):
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


async def handle_message(websocket):
    """
    Handle incoming messages from simulator.
    """
    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print("Invalid JSON received, skip.")
            continue

        b64img = data.get("image")
        if not b64img:
            print("No image field in message, skip.")
            continue

        current_speed = data.get("current_speed", None)

        # decode image
        try:
            img_bytes = base64.b64decode(b64img)
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            image = np.asarray(pil_img)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Failed to decode/convert image:", e)
            continue

        draw = image.copy()

        # update candidate/confirmed/execution scheduling from sign queue
        _update_sign_state_from_queue()

        # compute base control
        throttle, steering = 0.0, 0.0
        if calculate_control_signal is None:
            print("calculate_control_signal is not available.")
        else:
            try:
                throttle, steering = calculate_control_signal(image, draw=draw)
            except TypeError:
                try:
                    sp = current_speed if current_speed is not None else 0.0
                    throttle, steering = calculate_control_signal(sp, image)
                except Exception as e:
                    print("calculate_control_signal failed alt:", e)
                    throttle, steering = 0.0, 0.0
            except Exception as e:
                print("calculate_control_signal runtime error:", e)
                throttle, steering = 0.0, 0.0

        # apply sign action only if execute_at passed and within duration
        throttle, steering = apply_sign_action(throttle, steering)

        # clamp final outputs
        throttle = clamp_val(throttle, 0.0, 1.0)
        steering = clamp_val(steering, -1.0, 1.0)

        # send image to sign detection process (non-blocking)
        try:
            if not g_image_queue.full():
                g_image_queue.put(image.copy())
        except Exception:
            pass

        # display status (show if scheduled and when it will execute)
        try:
            scheduled_name = _sign_state.get('name')
            exec_start = _sign_state.get('exec_start', 0.0)
            exec_in = max(0.0, exec_start - time.time()) if scheduled_name else 0.0
            text = f"T:{throttle:.3f} S:{steering:.3f} Scheduled:{scheduled_name} ExecIn:{exec_in:.2f}"
            cv2.putText(draw, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Result", draw)
            cv2.waitKey(1)
        except Exception:
            pass

        # send response
        send_msg = json.dumps({
            "throttle": float(throttle),
            "steering": float(steering)
        })
        print("SEND:", send_msg)
        try:
            await websocket.send(send_msg)
        except Exception as e:
            print("Failed to send message:", e)
        


async def echo(websocket, path):
    try:
        await handle_message(websocket)
    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print("Connection handler error:", e)


async def main():
    async with websockets.serve(echo, "0.0.0.0", 4567, ping_interval=None):
        print("WebSocket server started on 0.0.0.0:4567")
        await asyncio.Future()


if __name__ == '__main__':
    # start sign detection process (it must accept an output queue as second arg)
    try:
        sign_process = Process(target=process_traffic_sign_loop,
                               args=(g_image_queue, g_sign_queue, TRAFFIC_SIGN_MODEL, None, "Traffic signs"))
        sign_process.daemon = True
        sign_process.start()
        print("Traffic sign process started (pid=%s)" % (sign_process.pid,))
    except Exception as e:
        print("Failed to start sign process:", e)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        try:
            if g_image_queue is not None:
                try:
                    g_image_queue.put(None)
                except Exception:
                    pass
            if 'sign_process' in locals() and sign_process.is_alive():
                sign_process.terminate()
                sign_process.join(timeout=1.0)
        except Exception:
            pass
        cv2.destroyAllWindows()