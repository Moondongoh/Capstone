from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2


class Detector:

    def __init__(
        self,
        weights: str = r"C:\GIT\Capstone\Web\flask-template\weights\best.pt",
        conf: float = 0.5,
        device: Optional[str] = None,
    ):
        self.weights = weights
        self.conf = conf
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        # Ultralytics YOLO 예시 (본인 모델에 맞게 교체 가능)
        from ultralytics import YOLO

        self._model = YOLO(self.weights)
        return self

    def predict(self, frame) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")
        results = self._model.predict(
            source=frame,
            verbose=False,
            stream=False,
            conf=self.conf,
            device=self.device,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        names = r.names
        for b, c, cl in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = [int(v) for v in b]
            out.append(
                Detection(
                    label=names[int(cl)], confidence=float(c), bbox=(x1, y1, x2, y2)
                )
            )
        return out


_detector: Optional[Detector] = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector().load()  # 최초 1회 로드
    return _detector
