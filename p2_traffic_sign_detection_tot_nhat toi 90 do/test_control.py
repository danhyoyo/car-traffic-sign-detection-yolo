"""Run: python -m unittest discover -s p2_traffic_sign_detection_vip_v1 -p 'test_*.py'"""
import asyncio
import base64
import json
from pathlib import Path
from queue import Queue
import time
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
import websockets

import drive
from lane_line_detection import (LaneController, SteeringLimiter, SCAN_ROWS,
                                 INITIAL_WIDTHS, STEERING_RATE, MAX_STEERING,
                                 calculate_control_signal, reset_control_state)


def road(shift=0.0, bend=0.0, size=(240, 480), side=None):
    h, w = size
    image = np.full((h, w, 3), 45, np.uint8)
    ys = np.array(SCAN_ROWS[::-1])
    widths = np.array(INITIAL_WIDTHS[::-1])
    for direction, name in ((-1, 'left'), (1, 'right')):
        if side is not None and name != side:
            continue
        points = []
        for y in range(int(h * 0.50), h):
            fraction = y / h
            width = np.interp(fraction, ys, widths)
            center = 0.5 + shift + bend * max(0, (0.84 - fraction) / 0.30)
            points.append((round((center + direction * width / 2) * w), y))
        cv2.polylines(image, [np.array(points, np.int32)], False, (255, 255, 255),
                      max(2, round(w * 0.008)))
    return image


def settle(controller, image, count=90):
    for _ in range(count):
        result = controller.update(image, dt=1/30)
    return result


class LaneTests(unittest.TestCase):
    def test_centered_and_resolution_independent(self):
        values = []
        for size in ((160, 320), (240, 480), (480, 960)):
            throttle, steering = settle(LaneController(), road(shift=0.08, size=size))
            self.assertGreater(throttle, 0.3)
            self.assertGreater(steering, 0.08)
            values.append(steering)
        self.assertLess(max(values) - min(values), 0.025)
        self.assertLess(abs(settle(LaneController(), road())[1]), 0.02)

    def test_preview_sees_curve_before_near_row(self):
        right = settle(LaneController(), road(bend=0.18))[1]
        left = settle(LaneController(), road(bend=-0.18))[1]
        self.assertGreater(right, 0.08)
        self.assertLess(left, -0.08)

    def test_jitter_does_not_saturate_or_oscillate_hard(self):
        c = LaneController()
        settle(c, road())
        values = [c.update(road(shift=0.01 * (-1)**i), dt=1/30)[1]
                  for i in range(120)]
        self.assertLess(max(map(abs, values)), 0.04)

    def test_single_edge_has_lower_confidence(self):
        c = LaneController()
        settle(c, road())
        throttle, steering = settle(c, road(side='left'))
        self.assertLess(c.confidence, 0.6)
        self.assertLess(abs(steering), 0.05)
        self.assertLess(throttle, 0.5)

    def test_lost_lane_slows_then_stops_and_recovers_gradually(self):
        c = LaneController()
        initial, _ = settle(c, road(shift=0.10))
        blank = np.zeros((240, 480, 3), np.uint8)
        throttle, steering = c.update(blank, dt=1/30)
        self.assertLess(throttle, initial)
        self.assertGreater(steering, 0)
        throttle, _ = settle(c, blank, 30)
        self.assertEqual(throttle, 0)
        throttle, _ = c.update(road(), dt=1/30)
        self.assertLessEqual(throttle, 0.45 / 30 + 1e-8)

    def test_steering_rate_with_variable_frame_times(self):
        c = LaneController()
        previous = 0
        for i in range(100):
            dt = (0.008, 0.033, 0.075)[i % 3]
            _, steering = c.update(road(shift=0.18 * (-1)**(i // 8)), dt=dt)
            self.assertLessEqual(abs(steering-previous), STEERING_RATE * dt + 1e-8)
            self.assertLessEqual(abs(steering), MAX_STEERING)
            previous = steering

    def test_long_frame_gap_counts_toward_loss_timeout(self):
        c = LaneController()
        settle(c, road())
        self.assertEqual(c.update(None, dt=1.0)[0], 0)

    def test_call_compatibility(self):
        for call in (lambda: calculate_control_signal(road()),
                     lambda: calculate_control_signal(5.0, road()),
                     lambda: calculate_control_signal(image=road(), draw=road())):
            reset_control_state()
            throttle, steering = call()
            self.assertTrue(np.isfinite([throttle, steering]).all())

    def test_repository_images(self):
        archive = Path(__file__).resolve().parent.parent / 'content/images/lane_line_images.zip'
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                image = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), 1)
                c = LaneController()
                throttle, steering = settle(c, image, 30)
                self.assertGreater(c.confidence, 0.2, name)
                self.assertTrue(0 <= throttle <= 1, name)
                self.assertLessEqual(abs(steering), MAX_STEERING, name)


class SignTests(unittest.TestCase):
    def setUp(self):
        drive._reset_sign_session()
        self.queue_patch = patch.object(drive, 'g_sign_queue', Queue())
        self.queue_patch.start()
        self.addCleanup(self.queue_patch.stop)

    def payload(self, ts):
        return dict(name='right', score=0.95, bbox=(0, 0, 60, 60), ts=ts)

    def test_repeated_sign_does_not_postpone_or_repeat_action(self):
        now = time.time()
        for i in range(3):
            drive.g_sign_queue.put(self.payload(now + i * 0.01))
        with patch.object(drive.time, 'time', return_value=now):
            drive._update_sign_state_from_queue()
        self.assertIsNone(drive._sign_state['name'])
        self.assertEqual(drive._pending_turn['name'], 'right')
        start = None
        for i in range(1, 51):
            ts = now + i * 0.1
            drive.g_sign_queue.put(self.payload(ts))
            with patch.object(drive.time, 'time', return_value=ts):
                drive._update_sign_state_from_queue()
                drive._update_junction_turn({'reliable': False, 'lost_seconds': .2})
                if drive._sign_state['name']:
                    if start is None:
                        start = drive._sign_state['exec_start']
                    self.assertEqual(start, drive._sign_state['exec_start'])
                drive._active_sign()
        self.assertAlmostEqual(start, now + .1)
        self.assertIsNone(drive._sign_state['name'])
        for i in range(3):
            drive.g_sign_queue.put(self.payload(now + 6 + i * .01))
        with patch.object(drive.time, 'time', return_value=now + 6.05):
            drive._update_sign_state_from_queue()
        self.assertEqual(drive._pending_turn['name'], 'right')

    def test_all_turns_start_on_lane_loss_without_extra_delay(self):
        for i, name in enumerate(('left', 'stop', 'right', 'straight', 'left', 'right')):
            now = 1000.0 + i * 10
            with patch.object(drive.time, 'time', return_value=now):
                self.assertTrue(drive._promote_candidate_to_confirmed(name, .95, None))
            if name in ('left', 'right'):
                self.assertIsNone(drive._sign_state['name'])
                with patch.object(drive.time, 'time', return_value=now + .5):
                    drive._update_junction_turn({'reliable': True, 'lost_seconds': 0})
                self.assertIsNone(drive._sign_state['name'])
                with patch.object(drive.time, 'time', return_value=now + 1):
                    drive._update_junction_turn({'reliable': False, 'lost_seconds': .2})
            start = (now + 1 if name in ('left', 'right')
                     else now + drive.EXECUTE_AFTER_SECONDS)
            self.assertAlmostEqual(drive._sign_state['exec_start'], start)
            with patch.object(drive.time, 'time', return_value=start - .01):
                self.assertIsNone(drive._active_sign())
            with patch.object(drive.time, 'time', return_value=start):
                self.assertEqual(drive._active_sign(), name)

    def test_pending_turn_not_overwritten_and_session_resets(self):
        with patch.object(drive.time, 'time', return_value=1000):
            self.assertTrue(drive._promote_candidate_to_confirmed('left', .95, None))
            self.assertFalse(drive._promote_candidate_to_confirmed('left', .95, None))
        drive._reset_sign_session()
        self.assertIsNone(drive._sign_state['name'])
        self.assertIsNone(drive._sign_candidate['name'])
        self.assertIsNone(drive._pending_turn)

    def test_lane_flicker_and_missing_status_do_not_trigger_turn(self):
        with patch.object(drive.time, 'time', return_value=1000):
            drive._promote_candidate_to_confirmed('left', .95, None)
        with patch.object(drive.time, 'time', return_value=1001):
            for status in (None, {'reliable': False, 'lost_seconds': .05},
                           {'reliable': True, 'lost_seconds': 0}):
                drive._update_junction_turn(status)
                self.assertIsNone(drive._active_sign())
                self.assertIsNotNone(drive._pending_turn)
            drive._update_junction_turn({'reliable': False, 'lost_seconds': .15})
            self.assertEqual(drive._active_sign(), 'left')
            self.assertIsNone(drive._pending_turn)
            # A junction maneuver supplies its own bounded throttle even after
            # the lane controller has stopped due to missing lane markings.
            self.assertEqual(drive.apply_sign_action(0, 0), (.3, -.65))

    def test_pending_turn_expires_and_loss_alone_does_not_turn(self):
        with patch.object(drive.time, 'time', return_value=1000):
            drive._promote_candidate_to_confirmed('left', .95, None)
        with patch.object(drive.time, 'time', return_value=1016):
            drive._update_junction_turn({'reliable': False, 'lost_seconds': 1})
            self.assertIsNone(drive._pending_turn)
            self.assertIsNone(drive._active_sign())
            self.assertEqual(drive.apply_sign_action(0, 0), (0, 0))

    def test_stop_has_priority_over_pending_turn(self):
        with patch.object(drive.time, 'time', return_value=1000):
            drive._promote_candidate_to_confirmed('right', .95, None)
            drive._promote_candidate_to_confirmed('stop', .95, None)
        with patch.object(drive.time, 'time', return_value=1000.5):
            drive._update_junction_turn({'reliable': False, 'lost_seconds': .5})
            self.assertEqual(drive._active_sign(), 'stop')
            self.assertEqual(drive.apply_sign_action(.3, 0)[0], 0)
            self.assertIsNotNone(drive._pending_turn)

    def test_stale_sign_is_ignored(self):
        for _ in range(3):
            drive.g_sign_queue.put(self.payload(time.time() - 2))
        drive._update_sign_state_from_queue()
        self.assertIsNone(drive._sign_state['name'])

    def test_sign_limits_and_stop(self):
        with patch.object(drive, '_active_sign', return_value='straight'):
            self.assertLessEqual(drive.apply_sign_action(.4, .3)[0], .4)
        with patch.object(drive, '_active_sign', return_value='stop'):
            self.assertEqual(drive.apply_sign_action(.5, .3)[0], 0)
        limiter = SteeringLimiter()
        previous = 0
        for name in ('right', 'left', 'straight', 'right'):
            with patch.object(drive, '_active_sign', return_value=name):
                _, target = drive.apply_sign_action(.5, 0)
            output = limiter.update(target, 1/30)
            self.assertLessEqual(abs(output - previous), STEERING_RATE / 30 + 1e-8)
            previous = output


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_websocket_frame_roundtrip(self):
        _, encoded = cv2.imencode('.jpg', road(bend=.12))
        message = json.dumps({'image': base64.b64encode(encoded).decode(), 'current_speed': 5})
        with patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey'), \
             patch.object(drive, 'g_sign_queue', Queue()), \
             patch.object(drive, 'g_image_queue', Queue(maxsize=1)):
            async with websockets.serve(drive.echo, '127.0.0.1', 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f'ws://127.0.0.1:{port}') as ws:
                    for _ in range(3):
                        await ws.send(message)
                        response = json.loads(await asyncio.wait_for(ws.recv(), 2))
                        self.assertTrue(0 <= response['throttle'] <= 1)
                        self.assertLessEqual(abs(response['steering']), MAX_STEERING)


class JunctionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_frames_wait_for_lane_loss_then_turn(self):
        frames = []
        for image in (road(), np.zeros((240, 480, 3), np.uint8)):
            _, encoded = cv2.imencode('.jpg', image)
            frames.append(json.dumps({'image': base64.b64encode(encoded).decode()}))

        class Camera:
            def __init__(self):
                self.index = 0
                self.outputs = []

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= 36:
                    raise StopAsyncIteration
                await asyncio.sleep(1/30)
                if self.index < 3:
                    drive.g_sign_queue.put(dict(name='right', score=.95,
                                               bbox=(0, 0, 60, 60), ts=time.time()))
                frame = frames[0 if self.index < 6 else 1]
                self.index += 1
                return frame

            async def send(self, message):
                self.outputs.append((self.index, json.loads(message),
                                     drive._sign_state['name'],
                                     drive._pending_turn is not None))

        camera = Camera()
        with patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey'), \
             patch.object(drive, 'g_sign_queue', Queue()), \
             patch.object(drive, 'g_image_queue', Queue(maxsize=1)):
            await drive.handle_message(camera)
        on_lane = [s for s in camera.outputs if s[0] <= 6]
        self.assertTrue(any(s[3] for s in on_lane))
        self.assertTrue(all(s[2] is None for s in on_lane))
        turns = [s for s in camera.outputs if s[2] == 'right']
        self.assertTrue(turns)
        self.assertGreaterEqual(turns[0][0], 9)
        self.assertTrue(all(not s[3] for s in turns))
        # After >0.75s without lanes, junction control must still move the car.
        self.assertGreater(camera.outputs[-1][1]['throttle'], .25)
        self.assertGreater(camera.outputs[-1][1]['steering'], .4)


if __name__ == '__main__':
    unittest.main()
