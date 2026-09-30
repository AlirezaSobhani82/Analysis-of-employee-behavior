import csv
from pathlib import Path
from collections import defaultdict
import cv2

PROJECT_ROOT = Path(r"D:\SlowFast_Project")

VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "s_001.mp4"
LABEL_FILE = PROJECT_ROOT / "data" / "labels" / "s_001.txt"
TRACKING_FILE = PROJECT_ROOT / "outputs" / "tracking" / "person_tracking.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "action"
OUTPUT_FILE = OUTPUT_DIR / "label_person_candidates.csv"


CLASS_NAMES = {
    1: "Typing",
    2: "Writing",
    3: "Drinking",
    4: "Eating",
    5: "Talking_in_phone",
    6: "Talking_with_human",
    7: "Shaking_hands",
    8: "Thinking",
    9: "Sleeping_on_desk",
    10: "Close_Door",
    11: "Open_Door",
    12: "Sitting_down",
    13: "Standing_up",
    14: "Standing",
    15: "Falling_down",
    16: "Surprised",
    17: "Walking",
    18: "Sneezing",
}


def load_labels():
    labels = []

    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 3:
                continue

            class_id = int(parts[0])
            start_frame = int(float(parts[1]))
            end_frame = int(float(parts[2]))

            labels.append({
                "line": line_number,
                "class_id": class_id,
                "action": CLASS_NAMES.get(class_id, f"Class_{class_id}"),
                "start_frame": start_frame,
                "end_frame": end_frame,
            })

    return labels


def load_tracking():
    tracking = defaultdict(list)

    with open(TRACKING_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = int(float(row["Video_Frame"]))
            person = int(row["Person"])

            x1 = float(row["X1"])
            y1 = float(row["Y1"])
            x2 = float(row["X2"])
            y2 = float(row["Y2"])

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            tracking[person].append({
                "frame": frame,
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    return tracking


def calculate_person_stats(person_frames, start_frame, end_frame):
    frames = [
        item
        for item in person_frames
        if start_frame <= item["frame"] <= end_frame
    ]

    if not frames:
        return None

    frames.sort(key=lambda x: x["frame"])

    frame_numbers = [item["frame"] for item in frames]

    first = frames[0]
    last = frames[-1]

    displacement = (
        (last["cx"] - first["cx"]) ** 2
        + (last["cy"] - first["cy"]) ** 2
    ) ** 0.5

    min_x = min(item["cx"] for item in frames)
    max_x = max(item["cx"] for item in frames)
    min_y = min(item["cy"] for item in frames)
    max_y = max(item["cy"] for item in frames)

    movement_range = (
        (max_x - min_x) ** 2
        + (max_y - min_y) ** 2
    ) ** 0.5

    expected_frames = end_frame - start_frame + 1

    coverage = len(frames) / expected_frames

    return {
        "tracked_frames": len(frames),
        "expected_frames": expected_frames,
        "coverage": coverage,
        "first_tracked_frame": min(frame_numbers),
        "last_tracked_frame": max(frame_numbers),
        "displacement": displacement,
        "movement_range": movement_range,
    }


def main():
    print("=" * 80)
    print("LABEL / PERSON CANDIDATE ANALYSIS")
    print("=" * 80)

    print()
    print("Loading labels...")
    labels = load_labels()
    print(f"Labels loaded: {len(labels)}")

    print()
    print("Loading tracking...")
    tracking = load_tracking()

    total_rows = sum(len(v) for v in tracking.values())

    print(f"Tracking rows loaded: {total_rows}")
    print(f"Persons found: {sorted(tracking.keys())}")

    print()
    print("Opening video...")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    print(f"Video frames: {total_video_frames}")
    print(f"FPS: {fps}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print()
    print("=" * 80)
    print("CANDIDATES")
    print("=" * 80)

    for label in labels:
        print()
        print(
            f"Line={label['line']:02d} | "
            f"{label['action']:<20} | "
            f"{label['start_frame']:5d}-{label['end_frame']:5d}"
        )

        candidates = []

        for person_id, person_frames in tracking.items():
            stats = calculate_person_stats(
                person_frames,
                label["start_frame"],
                label["end_frame"]
            )

            if stats is None:
                continue

            candidates.append({
                "line": label["line"],
                "class_id": label["class_id"],
                "action": label["action"],
                "start_frame": label["start_frame"],
                "end_frame": label["end_frame"],
                "person": person_id,
                **stats
            })

        candidates.sort(
            key=lambda x: (
                x["coverage"],
                x["movement_range"]
            ),
            reverse=True
        )

        for rank, candidate in enumerate(candidates, start=1):
            print(
                f"  Candidate {rank}: "
                f"Person {candidate['person']} | "
                f"Coverage={candidate['coverage']:.2%} | "
                f"Tracked={candidate['tracked_frames']} | "
                f"Movement={candidate['movement_range']:.1f}"
            )

            candidate["candidate_rank"] = rank
            rows.append(candidate)

    fieldnames = [
        "line",
        "class_id",
        "action",
        "start_frame",
        "end_frame",
        "person",
        "candidate_rank",
        "tracked_frames",
        "expected_frames",
        "coverage",
        "first_tracked_frame",
        "last_tracked_frame",
        "displacement",
        "movement_range",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()