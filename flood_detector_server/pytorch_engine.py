"""Direct RT-DETR checkpoint inference using Ultralytics and PyTorch."""

import os
from pathlib import Path
import threading
import time

import numpy as np

from detection_result import DEFAULT_CONFIDENCE


class RTDETRInferenceEngine:
    def __init__(self, model_path):
        self.model_path = str(Path(model_path).resolve())
        self.model = None
        self.device = 'cpu'
        self.gpu_name = None
        self.lock = threading.Lock()

    @property
    def is_ready(self):
        return self.model is not None

    def load_model(self):
        import torch

        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f'Không tìm thấy {self.model_path}. Đặt best.pt vào thư mục flood_detector_server.')
        if Path(self.model_path).suffix.lower() != '.pt':
            raise ValueError('Server nhận trực tiếp checkpoint .pt. Hãy cấu hình MODEL_PATH trỏ tới best.pt.')
        os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).parent / '.runtime' / 'ultralytics'))
        from ultralytics import RTDETR

        selected = os.getenv('MODEL_DEVICE', 'auto').strip().lower()
        if selected == 'auto':
            selected = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if selected != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('GPU được yêu cầu nhưng PyTorch chưa nhận CUDA. Dùng MODEL_DEVICE=cpu hoặc kiểm tra PyTorch CUDA.')
        self.device = selected
        if self.device != 'cpu':
            self.gpu_name = torch.cuda.get_device_name(torch.device(self.device))
        self.model = RTDETR(self.model_path)
        try:
            names = {int(k): str(v).strip().lower() for k, v in self.model.names.items()}
            if names != {0: 'flood'}:
                raise ValueError(f'Checkpoint phải có một lớp flood (ID 0), nhận được: {names}')
            # Initialize the predictor and GPU before accepting the first frame.
            self.predict(np.zeros((640, 640, 3), dtype=np.uint8), include_candidates=True)
        except Exception:
            self.model = None
            raise
        print(f'✅ Loaded checkpoint: {self.model_path}')
        print(f'   Runtime: PyTorch | Device: {self.device} | GPU: {self.gpu_name or "none"}')

    def predict(self, img_bgr, include_candidates=False):
        if not self.is_ready:
            raise RuntimeError('Mô hình best.pt chưa được tải.')
        # Ultralytics shares predictor state: serialize requests from all sources.
        with self.lock:
            started = time.perf_counter()
            result = self.model.predict(
                source=img_bgr, imgsz=640, device=self.device,
                conf=0.0 if include_candidates else DEFAULT_CONFIDENCE,
                verbose=False, save=False,
            )[0]
            rows = result.boxes.data.detach().float().cpu().numpy() if result.boxes is not None else []
            height, width = img_bgr.shape[:2]
            detections = []
            for row in rows:
                if not np.isfinite(row).all():
                    continue
                x1, y1, x2, y2, score, class_id = row.tolist()
                if int(class_id) != 0:
                    continue
                x1, x2 = [max(0, min(width, int(round(x)))) for x in (x1, x2)]
                y1, y2 = [max(0, min(height, int(round(y)))) for y in (y1, y2)]
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({
                    'bbox': {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2},
                    'confidence': round(float(score), 6), 'class_id': 0,
                    'label': 'Flood (Dấu hiệu ngập lụt)', 'color': '#EF4444',
                })
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return detections, latency_ms
