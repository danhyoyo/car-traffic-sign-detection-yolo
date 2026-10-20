"""Verify controller mappings without downloading weights or installing torch."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from simulator.yolo_sign_detection import YOLOTrafficSignDetector, PROJECT_CLASSES


def fake_model(names, rows=()):
    boxes = [SimpleNamespace(cls=np.array(class_id), conf=np.array(score),
                             xyxy=np.array([coords]))
             for class_id, score, coords in rows]
    result = SimpleNamespace(names=names, boxes=boxes)
    return SimpleNamespace(task='detect', names=names,
                           predict=lambda **kwargs: [result])


class YOLOMappingTests(unittest.TestCase):
    def test_only_exact_meanings_reach_controller(self):
        names = dict(enumerate(['cam_re_trai', 'cam_re_phai',
                                'cam_dung_xe_va_do_xe', 'duong_mot_chieu',
                                'cho_ngoat_nguy_hiem_ben_trai',
                                'huong_phai_di_vuot_chuong_ngai_vat_phai']))
        model = fake_model(names, [(i, 0.9, (10, 20, 80, 90)) for i in names])
        with self.assertWarns(UserWarning):
            detector = YOLOTrafficSignDetector(model=model)
        detections = detector.detect(np.zeros((120, 160, 3), dtype=np.uint8))
        self.assertEqual([d[0] for d in detections], ['no_left', 'no_right'])
        self.assertEqual(detections[0][1:5], (10, 20, 70, 70))
        self.assertEqual(set(detector.missing_classes), {'left', 'right', 'straight', 'stop'})

    def test_class_ids_can_be_shuffled(self):
        names = dict(enumerate(reversed(PROJECT_CLASSES)))
        model = fake_model(names, [(0, 0.9, (0, 0, 50, 50))])
        detector = YOLOTrafficSignDetector(model=model)
        self.assertEqual(detector.missing_classes, [])
        self.assertEqual(detector.detect(np.zeros((100, 100, 3), np.uint8))[0][0], names[0])

    def test_threshold_area_and_clipping(self):
        model = fake_model({0: 'stop'}, [
            (0, 0.2, (0, 0, 80, 80)),
            (0, 0.9, (0, 0, 10, 10)),
            (0, 0.9, (-20, -10, 200, 200)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            detector = YOLOTrafficSignDetector(model=model)
        detections = detector.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0][1:5], (0, 0, 100, 100))
        self.assertEqual(detector.detect(None), [])

    def test_sidecar_mapping_and_null(self):
        mapping = Path('sign_mapping.json')
        config = json.dumps({'source_to_project': {
            'custom_stop': 'stop', 'cam_re_trai': None}})
        with patch.object(Path, 'is_file', return_value=True), \
                patch.object(Path, 'read_text', return_value=config):
            model = fake_model({0: 'custom_stop', 1: 'cam_re_trai'},
                               [(0, 0.9, (0, 0, 60, 60)), (1, 0.9, (0, 0, 60, 60))])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                detector = YOLOTrafficSignDetector(model=model, mapping_path=mapping)
            self.assertEqual([d[0] for d in detector.detect(np.zeros((100, 100, 3), np.uint8))], ['stop'])
        with patch.object(Path, 'is_file', return_value=True), \
                patch.object(Path, 'read_text', return_value=json.dumps({
                    'source_to_project': {'custom_stop': 'parking'}})):
            with self.assertRaises(ValueError):
                YOLOTrafficSignDetector(model=model, mapping_path=mapping)

    def test_missing_weights_and_wrong_task_fail_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            YOLOTrafficSignDetector(model_path='does_not_exist.pt')
        with self.assertRaises(ValueError):
            YOLOTrafficSignDetector(model=SimpleNamespace(task='classify'))


if __name__ == '__main__':
    unittest.main()
