"""
traffic_sign_detection.py

Fixed and improved version with optional birdview support:
- Added birdview_transform (fallback if not available from lane modules).
- detect(..., use_birdview=False) supports running detection on birdview image.
- process_traffic_sign_loop(..., use_birdview=False) passes flag to detector and shows birdview window if enabled.
"""
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Try to reuse birdview_transform from lane modules if available
try:
    # prefer lane_line_detection_good (has stable birdview implementation)
    from lane_line_detection_good import birdview_transform  # type: ignore
except Exception:
    try:
        from lane_line_detection import birdview_transform  # type: ignore
    except Exception:
        birdview_transform = None  # will provide fallback below


def _fallback_birdview_transform(img: np.ndarray) -> np.ndarray:
    """
    A fallback birdview transform if lane modules are not importable.
    Based on lane_line_detection_good.py transform but adapts to input size.
    """
    h, w = img.shape[:2]
    IMAGE_H = h
    IMAGE_W = w
    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H],
                      [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[int(IMAGE_W * 0.375), IMAGE_H],
                      [int(IMAGE_W * 0.625), IMAGE_H],
                      [-int(IMAGE_W * 0.25), 0],
                      [IMAGE_W + int(IMAGE_W * 0.25), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped


if birdview_transform is None:
    birdview_transform = _fallback_birdview_transform  # type: ignore


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


def get_boxes_from_mask(mask: np.ndarray):
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

        # reasonable filters
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

    def detect(self, img: np.ndarray, draw: Optional[np.ndarray] = None, use_birdview: bool = False):
        """
        Detect traffic signs in BGR input image.
        If use_birdview=True, the detection is run on the birdview transform of the image
        and drawing is applied on the birdview copy.

        Returns list of detections: [ (class_name, x, y, w, h, score), ... ] in the image coordinates that detection ran on.
        """
        if img is None:
            return []

        proc_img = img
        proc_draw = draw

        if use_birdview:
            try:
                proc_img = birdview_transform(img)
                if draw is not None:
                    proc_draw = birdview_transform(draw.copy())
                else:
                    proc_draw = proc_img.copy()
            except Exception:
                proc_img = img
                proc_draw = draw

        mask = filter_signs_by_color(proc_img)
        bboxes = get_boxes_from_mask(mask)

        # Fallback: contours
        if not bboxes:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append((x, y, w, h))

        # Preprocess for classifier if available
        img_f = None
        if self.model is not None:
            try:
                img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
                img_f = img_rgb.astype('float32') / 255.0
            except Exception:
                img_f = None

        # Filter boxes by size
        bboxes = self._filter_bboxes_by_size(proc_img.shape, bboxes)

        detections = []
        for (x, y, w_box, h_box) in bboxes:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(proc_img.shape[1], x + w_box), min(proc_img.shape[0], y + h_box)
            sub = proc_img[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            if sub.shape[0] < 12 or sub.shape[1] < 12:
                continue

            cls_name = 'unknown'
            score = 0.0

            if self.model is not None and img_f is not None:
                try:
                    sub_f = img_f[y0:y1, x0:x1]
                    sub_resized = cv2.resize(sub_f, (32, 32))
                    blob = np.expand_dims(sub_resized, axis=0)
                    self.model.setInput(blob)
                    preds = self.model.forward()
                    preds = preds[0]
                    cls_idx = int(np.argmax(preds))
                    score = float(preds[cls_idx])
                    cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else 'unknown'
                except Exception:
                    cls_name = 'unknown'
                    score = 0.0

            if self.model is not None:
                if cls_name == 'unknown':
                    continue
                if score < self.conf_threshold:
                    continue

            detections.append((cls_name, x0, y0, x1 - x0, y1 - y0, score))

            if proc_draw is not None:
                try:
                    color = (0, 255, 255)
                    cv2.rectangle(proc_draw, (x0, y0), (x1, y1), color, 2)
                    label = f"{cls_name} {score:.2f}"
                    cv2.putText(proc_draw, label, (x0, max(0, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

        # If proc_draw exists and use_birdview, copy it back to draw variable for caller to display
        if use_birdview and draw is not None and proc_draw is not None:
            try:
                # Copy birdview visualization into caller draw buffer if sizes match; otherwise show separately.
                if proc_draw.shape == draw.shape:
                    draw[:] = proc_draw
                else:
                    # different sizes -> just show birdview separately (caller can ignore)
                    pass
            except Exception:
                pass

        return detections


def process_traffic_sign_loop(image_queue, output_queue=None, model_path: Optional[str] = None,
                              detector: Optional[TrafficSignDetector] = None,
                              window_name: str = "Traffic signs", use_birdview: bool = False):
    """
    Loop to run in separate process.
    If use_birdview=True, detection is run on the birdview transform and the birdview window is shown.
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
                dets = detector.detect(img, draw=draw, use_birdview=use_birdview)
                best = None
                if dets:
                    if any(d[5] > 0 for d in dets):
                        best = max(dets, key=lambda x: x[5])
                    else:
                        best = max(dets, key=lambda x: x[3] * x[4])
                if best is not None and output_queue is not None:
                    payload = {
                        'name': best[0],
                        'score': float(best[5]),
                        'bbox': (int(best[1]), int(best[2]), int(best[3]), int(best[4])),
                        'ts': time.time()
                    }
                    try:
                        if output_queue.full():
                            try:
                                _ = output_queue.get_nowait()
                            except Exception:
                                pass
                        output_queue.put_nowait(payload)
                    except Exception:
                        pass

                # Show birdview or raw visualization
                try:
                    if use_birdview:
                        cv2.imshow(window_name + " (birdview)", draw)
                    else:
                        cv2.imshow(window_name, draw)
                    cv2.waitKey(1)
                except Exception:
                    pass
            except Exception:
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