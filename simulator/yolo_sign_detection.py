"""Adapt full-frame YOLO detections to the existing controller's six names."""
import json
from pathlib import Path
import warnings

import cv2

PROJECT_CLASSES = ('left', 'no_left', 'right', 'no_right', 'straight', 'stop')
# Match meaning exactly. Warning bends, keep-left/right, one-way and parking
# restrictions are deliberately not aliases for the controller's actions.
DEFAULT_MAPPING = {name: name for name in PROJECT_CLASSES}
DEFAULT_MAPPING.update({'cam_re_trai': 'no_left', 'cam_re_phai': 'no_right'})


class YOLOTrafficSignDetector:
    def __init__(self, model_path=None, model=None, conf_threshold=0.45,
                 min_box_area=1200, mapping_path=None):
        if model is None:
            if model_path is None or not Path(model_path).is_file():
                raise FileNotFoundError(f'YOLO weights not found: {model_path}')
            from ultralytics import YOLO
            model = YOLO(str(model_path))
        if getattr(model, 'task', None) != 'detect':
            raise ValueError('Expected YOLO object detection weights, not a classifier.')
        self.model = model
        self.conf_threshold = conf_threshold
        self.min_box_area = min_box_area
        self.mapping = DEFAULT_MAPPING.copy()
        if mapping_path is None and model_path is not None:
            mapping_path = Path(model_path).with_name('sign_mapping.json')
        if mapping_path is not None and Path(mapping_path).is_file():
            config = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
            overrides = config['source_to_project']
            if any(value is not None and value not in PROJECT_CLASSES
                   for value in overrides.values()):
                raise ValueError('Mapping targets must be one of the six project names or null.')
            self.mapping.update(overrides)
        names = model.names.values() if isinstance(model.names, dict) else model.names
        covered = {self.mapping.get(name) for name in names}
        self.missing_classes = sorted(set(PROJECT_CLASSES) - covered)
        if self.missing_classes:
            warnings.warn('YOLO cannot provide these controller classes: '
                          + ', '.join(self.missing_classes), stacklevel=2)

    def detect(self, img, draw=None):
        if img is None:
            return []
        results = self.model.predict(source=img, conf=self.conf_threshold,
                                     imgsz=640, verbose=False)
        detections = []
        height, width = img.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                source_name = result.names[int(box.cls.item())]
                name = self.mapping.get(source_name)
                if name is None:
                    continue
                score = float(box.conf.item())
                if score < self.conf_threshold:
                    continue
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                x0, x1 = max(0, min(width, int(x0))), max(0, min(width, int(x1)))
                y0, y1 = max(0, min(height, int(y0))), max(0, min(height, int(y1)))
                w, h = x1 - x0, y1 - y0
                if w <= 0 or h <= 0 or w * h < self.min_box_area:
                    continue
                detections.append((name, x0, y0, w, h, score))
                if draw is not None:
                    cv2.rectangle(draw, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(draw, f'{name} {score:.2f}', (x0, max(15, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return detections
