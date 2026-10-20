"""
traffic_sign_detection.py

Fixed and improved version:
- Bug fixes: initialize `detections` and `img_f` (previous version could raise NameError
  so detector returned nothing).
- More robust color thresholds and morphology to reduce noise.
- Configurable "near" filters: min_box_pixels / min_box_fraction / min_box_area with
  more sensible defaults so signs at moderate distance are detected.
- Lowered default confidence threshold slightly to accept more detections (tune later).
- Contour fallback: if connectedComponents gives no boxes, use contours to find candidate boxes.
- process_traffic_sign_loop emits payloads into output_queue as before.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


def load_model(path: str):
    """Load an ONNX model using OpenCV DNN."""
    return cv2.dnn.readNetFromONNX(path)


def filter_signs_by_color(image: np.ndarray) -> np.ndarray:
    """
    Filter candidate traffic sign pixels by HSV color (red and blue).
    Returns a cleaned binary mask.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red ranges (broadened a bit)
    lower1, upper1 = np.array([0, 60, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([160, 60, 50]), np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask_r = cv2.bitwise_or(mask1, mask2)

    # Blue range (broadened)
    lower_b, upper_b = np.array([90, 80, 50]), np.array([140, 255, 255])
    mask_b = cv2.inRange(hsv, lower_b, upper_b)

    mask = cv2.bitwise_or(mask_r, mask_b)

    # Morphology to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional: small blur to smooth boundaries then threshold
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    return mask


def get_boxes_from_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Find bounding boxes from a binary mask using connected components.
    Returns list of (x, y, w, h).
    """
    bboxes = []
    nccomps = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_h, im_w = mask.shape[:2]

    for i in range(1, numLabels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # reasonable filters (these are conservative; the detector class will do stronger filtering)
        if w < 10 or h < 10:
            continue
        if w > 0.95 * im_w or h > 0.95 * im_h:
            continue
        if w / h > 4.0 or h / w > 4.0:
            continue

        bboxes.append((x, y, w, h))
    return bboxes


class TrafficSignDetector:
    """
    Detects traffic sign candidates by color and optionally classifies them
    using a provided OpenCV DNN model.

    Key configurable parameters:
      - conf_threshold: min model confidence to accept a class
      - min_box_pixels: min width/height in px to consider "near"
      - min_box_fraction: min fraction of image width/height (if >0)
      - min_box_area: min area in px^2
    """

    DEFAULT_CLASSES = ['unknown', 'left', 'no_left', 'right',
                       'no_right', 'straight', 'stop']

    def __init__(self, model=None, model_path: Optional[str] = None,
                 classes: Optional[List[str]] = None, conf_threshold: float = 0.45,
                 min_box_pixels: int = 30, min_box_fraction: float = 0.06,
                 min_box_area: int = 1200):
        self.model = model
        if self.model is None and model_path is not None:
            try:
                self.model = load_model(model_path)
            except Exception:
                self.model = None
        self.classes = classes if classes is not None else self.DEFAULT_CLASSES
        self.conf_threshold = float(conf_threshold)

        # "near" filters
        self.min_box_pixels = int(min_box_pixels)
        self.min_box_fraction = float(min_box_fraction)
        self.min_box_area = int(min_box_area)

    def _filter_bboxes_by_size(self, img_shape, bboxes):
        """
        Filter bboxes by configured size thresholds.
        """
        h, w = img_shape[:2]
        min_w_frac = int(w * self.min_box_fraction) if self.min_box_fraction > 0 else 0
        min_h_frac = int(h * self.min_box_fraction) if self.min_box_fraction > 0 else 0

        filtered = []
        for (x, y, bw, bh) in bboxes:
            # require either width or height to exceed pixel/fraction threshold
            if bw < max(self.min_box_pixels, min_w_frac) and bh < max(self.min_box_pixels, min_h_frac):
                continue
            if self.min_box_area > 0 and (bw * bh) < self.min_box_area:
                continue
            filtered.append((x, y, bw, bh))
        return filtered

    def detect(self, img: np.ndarray, draw: Optional[np.ndarray] = None):
        """
        Detect traffic signs in BGR input image.
        Returns list of detections: [ (class_name, x, y, w, h, score), ... ]
        """
        if img is None:
            return []

        mask = filter_signs_by_color(img)
        bboxes = get_boxes_from_mask(mask)

        # If connected components returned nothing, try contour-based boxes as fallback
        if not bboxes:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:  # ignore tiny contours
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append((x, y, w, h))

        # Preprocess for classifier if available
        img_f = None
        if self.model is not None:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_f = img_rgb.astype(np.float32) / 255.0
            except Exception:
                img_f = None

        # Filter boxes by near thresholds
        bboxes = self._filter_bboxes_by_size(img.shape, bboxes)

        detections = []
        for (x, y, w_box, h_box) in bboxes:
            # safe crop bounds
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(img.shape[1], x + w_box), min(img.shape[0], y + h_box)
            sub = img[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            if sub.shape[0] < 12 or sub.shape[1] < 12:
                continue

            cls_name = 'unknown'
            score = 0.0

            # classification if model present
            if self.model is not None and img_f is not None:
                try:
                    sub_f = img_f[y0:y1, x0:x1]
                    sub_resized = cv2.resize(sub_f, (32, 32))
                    blob = np.expand_dims(sub_resized, axis=0)  # shape (1,32,32,3)
                    # setInput accepts NHWC for onnx in many builds; if not, user may need to swap axes
                    self.model.setInput(blob)
                    preds = self.model.forward()
                    preds = preds[0]
                    cls_idx = int(np.argmax(preds))
                    score = float(preds[cls_idx])
                    cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else 'unknown'
                except Exception:
                    cls_name = 'unknown'
                    score = 0.0

            # If model used, enforce class not 'unknown' and confidence threshold
            if self.model is not None:
                if cls_name == 'unknown':
                    continue
                if score < self.conf_threshold:
                    continue

            detections.append((cls_name, x0, y0, x1 - x0, y1 - y0, score))

            if draw is not None:
                try:
                    color = (0, 255, 255)
                    cv2.rectangle(draw, (x0, y0), (x1, y1), color, 2)
                    label = f"{cls_name} {score:.2f}"
                    cv2.putText(draw, label, (x0, max(0, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

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
                best = None
                if dets:
                    # choose highest score; if model not present, score may be 0 so take largest bbox instead
                    if any(d[5] > 0 for d in dets):
                        best = max(dets, key=lambda x: x[5])
                    else:
                        best = max(dets, key=lambda x: x[3] * x[4])  # largest area
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
                        pass
                cv2.imshow(window_name, draw)
                cv2.waitKey(1)
            except Exception:
                # display raw image if detection fails
                try:
                    cv2.imshow(window_name, img)
                    cv2.waitKey(1)
                except Exception:
                    pass
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(0.05)
            continue