from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)

        self.video_path = Path(self.config["paths"]["video"])
        self.output_dir = Path(
            self.config["paths"]["detection_output"]
        )

        self.model_path = self.config["models"]["yolo"]

        self.confidence = self.config["detection"]["confidence"]
        self.image_size = self.config["detection"]["image_size"]

        self.model = YOLO(self.model_path)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def detect(self):
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        capture = cv2.VideoCapture(
            str(self.video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        frame_number = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False
            )

            person_detections = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(
                        box.cls[0].item()
                    )

                    if class_id != 0:
                        continue

                    confidence = float(
                        box.conf[0].item()
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    person_detections.append({
                        "frame": frame_number,
                        "class_id": class_id,
                        "confidence": confidence,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    })

            yield frame_number, frame, person_detections

            frame_number += 1

        capture.release()