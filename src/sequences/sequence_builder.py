
from pathlib import Path
import csv
import shutil
import cv2
import yaml


class SequenceBuilder:
    def __init__(self, config_path=None):
        self.project_root = Path(__file__).resolve().parents[2]

        if config_path is None:
            config_path = (
                self.project_root
                / "config"
                / "config.yaml"
            )
        else:
            config_path = Path(config_path)

            if not config_path.is_absolute():
                config_path = (
                    self.project_root
                    / config_path
                )

        self.config_path = config_path

        self.config = self._load_config(
            self.config_path
        )

        self.video_path = self._resolve_path(
            self.config["paths"]["video"]
        )

        self.tracking_csv = (
            self._resolve_path(
                self.config["paths"]["tracking_output"]
            )
            / "person_tracking.csv"
        )

        self.output_dir = self._resolve_path(
            self.config["paths"]["sequence_output"]
        )

        self.num_frames = int(
            self.config["sequence"]["num_frames"]
        )

        self.image_size = int(
            self.config["sequence"]["image_size"]
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _load_config(self, config_path):
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as file:
            config = yaml.safe_load(file)

        if not config:
            raise ValueError(
                f"Config file is empty: {config_path}"
            )

        return config

    def _resolve_path(self, path_value):
        path = Path(path_value)

        if path.is_absolute():
            return path

        return self.project_root / path

    def _load_tracking_data(self):
        if not self.tracking_csv.exists():
            raise FileNotFoundError(
                f"Tracking file not found: {self.tracking_csv}"
            )

        tracking_data = []

        with open(
            self.tracking_csv,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "Tracking CSV has no header."
                )

            required_columns = {
                "Video_Frame",
                "Person",
                "X1",
                "Y1",
                "X2",
                "Y2",
                "Confidence"
            }

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "Missing tracking columns: "
                    f"{sorted(missing_columns)}"
                )

            for row in reader:

                try:
                    tracking_data.append({
                        "frame": int(
                            float(
                                row["Video_Frame"]
                            )
                        ),

                        "person_id": int(
                            float(
                                row["Person"]
                            )
                        ),

                        "confidence": float(
                            row["Confidence"]
                        ),

                        "x1": int(
                            float(
                                row["X1"]
                            )
                        ),

                        "y1": int(
                            float(
                                row["Y1"]
                            )
                        ),

                        "x2": int(
                            float(
                                row["X2"]
                            )
                        ),

                        "y2": int(
                            float(
                                row["Y2"]
                            )
                        )
                    })

                except (
                    ValueError,
                    TypeError,
                    KeyError
                ):
                    continue

        return tracking_data

    def _group_by_person(self, tracking_data):
        persons = {}

        for item in tracking_data:
            person_id = item["person_id"]

            if person_id not in persons:
                persons[person_id] = []

            persons[person_id].append(item)

        for person_id in persons:
            persons[person_id].sort(
                key=lambda item: item["frame"]
            )

        return persons

    def _create_crop(self, frame, tracking):
        height, width = frame.shape[:2]

        x1 = max(
            0,
            min(tracking["x1"], width - 1)
        )

        y1 = max(
            0,
            min(tracking["y1"], height - 1)
        )

        x2 = max(
            0,
            min(tracking["x2"], width)
        )

        y2 = max(
            0,
            min(tracking["y2"], height)
        )

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            return None

        crop = cv2.resize(
            crop,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_LINEAR
        )

        return crop

    def _read_frame(self, capture, frame_number):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = capture.read()

        if not success:
            return None

        return frame

    def _is_continuous_sequence(self, sequence):
        if len(sequence) != self.num_frames:
            return False

        frame_numbers = [
            item["frame"]
            for item in sequence
        ]

        for index in range(
            len(frame_numbers) - 1
        ):
            if (
                frame_numbers[index] + 1
                != frame_numbers[index + 1]
            ):
                return False

        return True

    def _save_sequence(
        self,
        capture,
        sequence,
        person_id,
        sequence_id
    ):
        sequence_dir = (
            self.output_dir
            / f"person_{person_id}"
            / f"sequence_{sequence_id:04d}"
        )

        if sequence_dir.exists():
            shutil.rmtree(
                sequence_dir
            )

        sequence_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        saved_frames = 0

        for index, tracking in enumerate(sequence):

            frame = self._read_frame(
                capture,
                tracking["frame"]
            )

            if frame is None:
                shutil.rmtree(
                    sequence_dir,
                    ignore_errors=True
                )
                return False

            crop = self._create_crop(
                frame,
                tracking
            )

            if crop is None:
                shutil.rmtree(
                    sequence_dir,
                    ignore_errors=True
                )
                return False

            frame_path = (
                sequence_dir
                / f"{index:03d}.jpg"
            )

            success = cv2.imwrite(
                str(frame_path),
                crop
            )

            if not success:
                shutil.rmtree(
                    sequence_dir,
                    ignore_errors=True
                )
                return False

            saved_frames += 1

        if saved_frames != self.num_frames:
            shutil.rmtree(
                sequence_dir,
                ignore_errors=True
            )
            return False

        return True

    def build(self):
        print(
            f"Project root: {self.project_root}"
        )

        print(
            f"Config: {self.config_path}"
        )

        print(
            f"Video: {self.video_path}"
        )

        print(
            f"Tracking CSV: {self.tracking_csv}"
        )

        print(
            f"Sequence output: {self.output_dir}"
        )

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        tracking_data = self._load_tracking_data()

        if not tracking_data:
            print(
                "No tracking data found."
            )
            return

        persons = self._group_by_person(
            tracking_data
        )

        capture = cv2.VideoCapture(
            str(self.video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        total_sequences = 0

        for person_id in sorted(persons):

            tracks = persons[person_id]

            print(
                f"Person {person_id}: "
                f"{len(tracks)} tracked frames"
            )

            if len(tracks) < self.num_frames:
                print(
                    f"Person {person_id}: "
                    f"not enough frames"
                )
                continue

            sequence_id = 1

            for start in range(
                0,
                len(tracks) - self.num_frames + 1,
                self.num_frames
            ):

                sequence = tracks[
                    start:start + self.num_frames
                ]

                if not self._is_continuous_sequence(
                    sequence
                ):
                    continue

                saved = self._save_sequence(
                    capture,
                    sequence,
                    person_id,
                    sequence_id
                )

                if saved:
                    total_sequences += 1

                    print(
                        f"Created: "
                        f"person_{person_id}/"
                        f"sequence_{sequence_id:04d}"
                    )

                    sequence_id += 1

        capture.release()

        print()
        print(
            f"Created {total_sequences} sequences."
        )

    def clean_output(self):
        if self.output_dir.exists():
            shutil.rmtree(
                self.output_dir
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Sequence output cleaned: "
            f"{self.output_dir}"
        )


if __name__ == "__main__":
    builder = SequenceBuilder()
    builder.build()

