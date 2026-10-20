"""
traffic_sign_detection.py

Modular traffic sign detector that:
- finds color candidates (red/blue)
- optionally classifies via ONNX model (cv2.dnn)
- when run in a separate process can send detected sign results back to main process
  using an output multiprocessing.Queue.

Exports:
- load_model(path)
- TrafficSignDetector
- process_traffic_sign_loop(image_queue, output_queue, model_path=None, detector=None, window_name="Traffic signs")
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


def load_model(path: str):
    """Load an ONNX model using OpenCV DNN."""
    return cv2.dnn.readNetFromONNX(path)


def filter_signs_by_color(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red ranges
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask_r = cv2.bitwise_or(mask1, mask2)

    # Blue
    lower_b, upper_b = np.array([100, 150, 0]), np.array([140, 255, 255])
    mask_b = cv2.inRange(hsv, lower_b, upper_b)

    mask = cv2.bitwise_or(mask_r, mask_b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def get_boxes_from_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    bboxes = []
    nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_h, im_w = mask.shape[:2]

    for i in range(1, numLabels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if w < 50 or h < 50:
            continue
        if w > 0.8 * im_w or h > 0.8 * im_h:
            continue
        if w / h > 2.0 or h / w > 2.0:
            continue

        bboxes.append((x, y, w, h))
    return bboxes


class TrafficSignDetector:
    DEFAULT_CLASSES = ['unknown', 'left', 'no_left', 'right',
                       'no_right', 'straight', 'stop']

    def __init__(self, model=None, model_path: Optional[str] = None,
                 classes: Optional[List[str]] = None, conf_threshold: float = 0.5):
        self.model = model
        if self.model is None and model_path is not None:
            try:
                self.model = load_model(model_path)
            except Exception:
                self.model = None
        self.classes = classes if classes is not None else self.DEFAULT_CLASSES
        self.conf_threshold = conf_threshold

    def detect(self, img: np.ndarray, draw: Optional[np.ndarray] = None):
        """
        Returns list of (class_name, x, y, w, h, score)
        If model available, will classify; otherwise returns 'unknown' candidates.
        """
        if img is None:
            return []

        mask = filter_signs_by_color(img)
        bboxes = get_boxes_from_mask(mask)

        img_f = None
        if self.model is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_f = img_rgb.astype(np.float32) / 255.0

        detections = []
        for (x, y, w, h) in bboxes:
            sub = img[y:y+h, x:x+w]
            if sub.shape[0] < 20 or sub.shape[1] < 20:
                continue

            cls_name = 'unknown'
            score = 0.0

            if self.model is not None and img_f is not None:
                try:
                    sub_f = img_f[y:y+h, x:x+w]
                    sub_resized = cv2.resize(sub_f, (32, 32))
                    blob = np.expand_dims(sub_resized, axis=0)
                    self.model.setInput(blob)
                    preds = self.model.forward()
                    preds = preds[0]
                    cls = int(preds.argmax())
                    score = float(preds[cls])
                    cls_name = self.classes[cls] if cls < len(self.classes) else 'unknown'
                except Exception:
                    cls_name = 'unknown'
                    score = 0.0

            if self.model is not None:
                if cls_name == 'unknown':
                    continue
                if score < self.conf_threshold:
                    continue

            detections.append((cls_name, x, y, w, h, score))

            if draw is not None:
                color = (0, 255, 255)
                cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)
                label = f"{cls_name} {score:.2f}"
                cv2.putText(draw, label, (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return detections


def process_traffic_sign_loop(image_queue, output_queue=None, model_path: Optional[str] = None,
                              detector: Optional[TrafficSignDetector] = None,
                              window_name: str = "Traffic signs"):
    """
    Loop to run in separate process:
    - reads images from image_queue (multiprocessing.Queue)
    - runs detection
    - shows visualization window
    - writes detection (best/highest-scored) to output_queue if provided

    Detection payload put to output_queue: dict { 'name': str, 'score': float, 'bbox': (x,y,w,h), 'ts': float }
    """
    if detector is None:
        detector = TrafficSignDetector(model_path=model_path)

    while True:
        try:
            if image_queue.empty():
                time.sleep(0.03)
                continue
            img = image_queue.get()
            if img is None:
                break
            draw = img.copy()
            try:
                dets = detector.detect(img, draw=draw)
                # choose best detection by score (or first)
                best = None
                if dets:
                    best = max(dets, key=lambda x: x[5])  # x = (name,x,y,w,h,score)
                if best is not None and output_queue is not None:
                    payload = {
                        'name': best[0],
                        'score': float(best[5]),
                        'bbox': (int(best[1]), int(best[2]), int(best[3]), int(best[4])),
                        'ts': time.time()
                    }
                    try:
                        # non-blocking put: if full, discard oldest then put
                        if output_queue.full():
                            try:
                                _ = output_queue.get_nowait()
                            except Exception:
                                pass
                        output_queue.put_nowait(payload)
                    except Exception:
                        # ignore output queue errors
                        pass
                cv2.imshow(window_name, draw)
                cv2.waitKey(1)
            except Exception:
                # display raw image if detection fails
                cv2.imshow(window_name, img)
                cv2.waitKey(1)
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(0.05)
            continue