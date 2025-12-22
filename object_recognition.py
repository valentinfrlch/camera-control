import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import os
os.environ.setdefault("ULTRALYTICS_IGNORE_SAM", "1")

from ultralytics import YOLO

COLOR_PALETTE = np.array(
    [
        (255, 99, 71),
        (127, 255, 0),
        (30, 144, 255),
        (255, 215, 0),
        (138, 43, 226),
        (0, 206, 209),
        (255, 105, 180),
        (60, 179, 113),
    ],
    dtype=np.uint8,
)


class ObjectRecognizer:
    def __init__(
        self,
        model_weights: Optional[str] = None,
        confidence_threshold: float = 0.45,
        img_size: int = 640,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        self.device = self._select_device()
        weights = model_weights or "yolov8n.pt"
        self.model = YOLO(weights)
        self.class_names = self.model.names
        self._inference_lock = threading.Lock()
        self.capture_objects = []

    def annotate(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        with self._inference_lock:
            results = self.model.predict(
                frame,
                verbose=False,
                device=self.device,
                imgsz=self.img_size,
                conf=self.confidence_threshold,
            )[0]

        boxes = results.boxes
        if boxes is None or boxes.shape[0] == 0:
            return frame, []

        annotated = frame.copy()
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: List[Dict[str, object]] = []
        h, w = frame.shape[:2]
        for (start_x, start_y, end_x, end_y), confidence, class_id in zip(
            xyxy, confidences, class_ids
        ):
            if confidence < self.confidence_threshold:
                continue
            start_x = max(0, min(start_x, w - 1))
            end_x = max(0, min(end_x, w - 1))
            start_y = max(0, min(start_y, h - 1))
            end_y = max(0, min(end_y, h - 1))
            label = self.class_names.get(class_id, f"class_{class_id}")
            color = self._color_for(class_id)

            cv2.rectangle(
                annotated, (start_x, start_y), (end_x, end_y), color, thickness=2
            )
            cv2.putText(
                annotated,
                f"{label} {confidence * 100:.1f}%",
                (start_x, max(start_y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "label": label,
                    "confidence": float(confidence),
                    "box": (start_x, start_y, end_x, end_y),
                }
            )

        return annotated, detections

    def _select_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _color_for(self, class_id: int) -> Tuple[int, int, int]:
        color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]
        return int(color[0]), int(color[1]), int(color[2])
