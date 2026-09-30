import csv
from pathlib import Path
import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"

MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"

TRACKING_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tracking"
)

TIMELINE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "timeline.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "objects"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "object_context.csv"
)

OBJECT_CONFIDENCE = 0.20
IMAGE_SIZE = 1280
MAX_DISTANCE = 300

OBJECT_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell_phone",
    73: "book",
    76: "scissors"
}


def load_tracking(tracking_csv):

    if not tracking_csv.exists():
        raise FileNotFoundError(
            f"Tracking file not found: {tracking_csv}"
        )

    tracking = {}

    with open(
        tracking_csv,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "Video_Frame",
            "Person",
            "X1",
            "Y1",
            "X2",
            "Y2"
        }

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Missing tracking columns. "
                f"Expected: {required}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:

            try:
                frame = int(
                    float(row["Video_Frame"])
                )

                person = int(
                    float(row["Person"])
                )

                x1 = float(row["X1"])
                y1 = float(row["Y1"])
                x2 = float(row["X2"])
                y2 = float(row["Y2"])

            except (ValueError, TypeError):
                continue

            if frame not in tracking:
                tracking[frame] = []

            tracking[frame].append({
                "person": person,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    return tracking


def load_timeline(video_name):

    if not TIMELINE_CSV.exists():
        raise FileNotFoundError(
            f"Timeline file not found: {TIMELINE_CSV}"
        )

    timeline = []

    with open(
        TIMELINE_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "Video",
            "Person",
            "Sequence",
            "Start_Frame",
            "End_Frame"
        }

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Missing timeline columns. "
                f"Expected: {required}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:

            if row["Video"] != video_name:
                continue

            try:
                person_text = str(row["Person"]).strip()

                if person_text.startswith("person_"):
                    person = int(
                        person_text.replace(
                            "person_",
                            ""
                        )
                    )
                else:
                    person = int(
                        float(person_text)
                    )

                timeline.append({
                    "person": person,
                    "sequence": row["Sequence"],
                    "start_frame": int(
                        float(row["Start_Frame"])
                    ),
                    "end_frame": int(
                        float(row["End_Frame"])
                    )
                })

            except (ValueError, TypeError):
                continue

    return timeline


def calculate_center(
    x1,
    y1,
    x2,
    y2
):

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def calculate_distance(
    point1,
    point2
):

    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]

    return (
        dx * dx + dy * dy
    ) ** 0.5


def associate_object_to_person(
    object_box,
    people
):

    ox1, oy1, ox2, oy2 = object_box

    object_center = calculate_center(
        ox1,
        oy1,
        ox2,
        oy2
    )

    best_person = None
    best_distance = float("inf")

    for person in people:

        person_center = calculate_center(
            person["x1"],
            person["y1"],
            person["x2"],
            person["y2"]
        )

        distance = calculate_distance(
            object_center,
            person_center
        )

        if distance < best_distance:
            best_distance = distance
            best_person = person

    if best_person is None:
        return None, None

    if best_distance > MAX_DISTANCE:
        return None, best_distance

    return (
        best_person["person"],
        best_distance
    )


def build_timeline_lookup(
    timeline
):

    lookup = {}

    for item in timeline:

        person = item["person"]

        start_frame = item["start_frame"]

        end_frame = item["end_frame"]

        for frame in range(
            start_frame,
            end_frame + 1
        ):

            key = (
                frame,
                person
            )

            lookup[key] = item["sequence"]

    return lookup


def detect_objects(
    model,
    video_path,
    tracking,
    timeline_lookup,
    video_name
):

    results = []

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_number = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        people = tracking.get(
            frame_number,
            []
        )

        if not people:
            continue

        yolo_results = model.predict(
            source=frame,
            conf=OBJECT_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        if not yolo_results:
            continue

        result = yolo_results[0]

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            if class_id not in OBJECT_CLASSES:
                continue

            confidence = float(
                box.conf.item()
            )

            coordinates = (
                box.xyxy[0].tolist()
            )

            x1, y1, x2, y2 = coordinates

            person_id, distance = (
                associate_object_to_person(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    people
                )
            )

            if person_id is None:
                continue

            sequence = timeline_lookup.get(
                (
                    frame_number,
                    person_id
                )
            )

            if sequence is None:
                continue

            results.append({
                "Video": video_name,
                "Video_Frame": frame_number,
                "Person": person_id,
                "Sequence": sequence,
                "Object": OBJECT_CLASSES[class_id],
                "Object_Class_ID": class_id,
                "Confidence": round(
                    confidence,
                    6
                ),
                "Distance": round(
                    distance,
                    2
                ),
                "X1": round(
                    x1,
                    2
                ),
                "Y1": round(
                    y1,
                    2
                ),
                "X2": round(
                    x2,
                    2
                ),
                "Y2": round(
                    y2,
                    2
                )
            })

        if frame_number % 500 == 0:

            print(
                f"{video_name} | "
                f"Processed frames: "
                f"{frame_number}/{total_frames}"
            )

    capture.release()

    return results


def save_results(
    results
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        "Video",
        "Video_Frame",
        "Person",
        "Sequence",
        "Object",
        "Object_Class_ID",
        "Confidence",
        "Distance",
        "X1",
        "Y1",
        "X2",
        "Y2"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


def main():

    print("=" * 60)
    print("Object Context Builder")
    print("=" * 60)

    print(
        f"Videos      : {VIDEOS_DIR}"
    )

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Tracking    : {TRACKING_DIR}"
    )

    print(
        f"Timeline    : {TIMELINE_CSV}"
    )

    print(
        f"Output      : {OUTPUT_CSV}"
    )

    print(
        f"Confidence  : {OBJECT_CONFIDENCE}"
    )

    print(
        f"Max distance: {MAX_DISTANCE}"
    )

    video_files = sorted(
        VIDEOS_DIR.glob("s_*.mp4")
    )

    if not video_files:
        raise FileNotFoundError(
            f"No videos found in: {VIDEOS_DIR}"
        )

    print(
        f"Videos found: {len(video_files)}"
    )

    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "YOLO model loaded."
    )

    all_results = []

    for index, video_path in enumerate(
        video_files,
        start=1
    ):

        video_name = video_path.stem

        print()
        print("=" * 60)
        print(
            f"[{index}/{len(video_files)}] "
            f"Processing: {video_name}"
        )
        print("=" * 60)

        tracking_csv = (
            TRACKING_DIR
            / f"{video_name}_tracking.csv"
        )

        print(
            f"Tracking file: {tracking_csv}"
        )

        tracking = load_tracking(
            tracking_csv
        )

        print(
            f"Tracking frames loaded: "
            f"{len(tracking)}"
        )

        timeline = load_timeline(
            video_name
        )

        print(
            f"Timeline rows loaded: "
            f"{len(timeline)}"
        )

        timeline_lookup = (
            build_timeline_lookup(
                timeline
            )
        )

        print(
            f"Timeline frame mappings: "
            f"{len(timeline_lookup)}"
        )

        results = detect_objects(
            model,
            video_path,
            tracking,
            timeline_lookup,
            video_name
        )

        print(
            f"Object records for "
            f"{video_name}: "
            f"{len(results)}"
        )

        all_results.extend(
            results
        )

    save_results(
        all_results
    )

    print()
    print("=" * 60)
    print("Object Context completed")
    print("=" * 60)

    print(
        f"Videos processed: "
        f"{len(video_files)}"
    )

    print(
        f"Total object records: "
        f"{len(all_results)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()