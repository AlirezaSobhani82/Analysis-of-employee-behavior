import os
import cv2
import shutil
import yaml
import pandas as pd
from collections import defaultdict, deque

PROJECT_ROOT = r"D:\SlowFast_Project"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "videos")
TRACKING_DIR = os.path.join(PROJECT_ROOT, "outputs", "tracking")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "sequences")

SEQUENCE_LENGTH = 10
IMAGE_SIZE = 224
VIDEO_PATTERN = "s_"
VIDEO_EXTENSION = ".mp4"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tracking(csv_path):
    df = pd.read_csv(csv_path)

    required = [
        "Video_Frame",
        "Person",
        "X1",
        "Y1",
        "X2",
        "Y2",
        "Confidence"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing tracking columns: {missing}")

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df["Video_Frame"] = df["Video_Frame"].astype(int)
    df["Person"] = df["Person"].astype(int)

    df = df.sort_values(
        ["Video_Frame", "Person"]
    ).reset_index(drop=True)

    return df


def create_crop(frame, x1, y1, x2, y2):
    height, width = frame.shape[:2]

    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    return crop


def save_sequence(output_root, person_id, sequence_number, crops):
    sequence_dir = os.path.join(
        output_root,
        f"person_{person_id}",
        f"sequence_{sequence_number:04d}"
    )

    os.makedirs(sequence_dir, exist_ok=True)

    for index, crop in enumerate(crops):
        image_path = os.path.join(
            sequence_dir,
            f"{index:03d}.jpg"
        )

        cv2.imwrite(
            image_path,
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )


def process_video(
    video_path,
    tracking_path,
    output_root,
    video_index,
    total_videos
):
    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    print()
    print("=" * 80)
    print(f"[{video_index}/{total_videos}] Processing: {video_name}")
    print(f"Video: {video_path}")
    print(f"Tracking CSV: {tracking_path}")
    print(f"Output: {output_root}")
    print("=" * 80)

    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    os.makedirs(output_root, exist_ok=True)

    tracking = load_tracking(tracking_path)

    if tracking.empty:
        print(f"[SKIP] {video_name}: tracking CSV is empty")
        return 0

    frame_map = defaultdict(list)

    for row in tracking.itertuples(index=False):
        frame_map[int(row.Video_Frame)].append(row)

    person_rows = tracking.groupby("Person").size().to_dict()

    print("Tracking rows:")

    for person_id in sorted(person_rows):
        print(
            f"  Person {person_id}: "
            f"{person_rows[person_id]} rows"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print("Reading video sequentially...")

    buffers = {}
    last_frame = {}
    sequence_counters = defaultdict(int)

    total_sequences = 0
    processed_frames = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_number = processed_frames

        if frame_number in frame_map:

            for row in frame_map[frame_number]:

                person_id = int(row.Person)

                if (
                    person_id not in buffers
                    or person_id not in last_frame
                    or frame_number != last_frame[person_id] + 1
                ):
                    buffers[person_id] = deque(
                        maxlen=SEQUENCE_LENGTH
                    )

                crop = create_crop(
                    frame,
                    row.X1,
                    row.Y1,
                    row.X2,
                    row.Y2
                )

                if crop is None:
                    buffers[person_id].clear()
                    last_frame[person_id] = frame_number
                    continue

                buffers[person_id].append(crop)
                last_frame[person_id] = frame_number

                if len(buffers[person_id]) == SEQUENCE_LENGTH:

                    sequence_counters[person_id] += 1

                    save_sequence(
                        output_root,
                        person_id,
                        sequence_counters[person_id],
                        list(buffers[person_id])
                    )

                    total_sequences += 1

        processed_frames += 1

        if processed_frames % 500 == 0:

            percent = (
                processed_frames / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"  Progress: "
                f"{processed_frames}/{total_frames} "
                f"({percent:.1f}%) | "
                f"Sequences: {total_sequences}",
                flush=True
            )

    cap.release()

    print()

    for person_id in sorted(sequence_counters):

        print(
            f"Person {person_id}: "
            f"{person_rows.get(person_id, 0)} tracking rows -> "
            f"{sequence_counters[person_id]} sequences"
        )

    print(
        f"[DONE] {video_name}: "
        f"{total_sequences} sequences"
    )

    return total_sequences


def main():

    load_config()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    videos = sorted(
        [
            os.path.join(
                VIDEO_DIR,
                name
            )
            for name in os.listdir(VIDEO_DIR)
            if name.startswith(VIDEO_PATTERN)
            and name.endswith(VIDEO_EXTENSION)
        ]
    )

    print("=" * 80)
    print("FAST ALL VIDEO SEQUENCE BUILDER")
    print("=" * 80)
    print(f"Videos found: {len(videos)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    total_sequences = 0
    successful_videos = 0
    failed_videos = 0

    for index, video_path in enumerate(
        videos,
        start=1
    ):

        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        tracking_path = os.path.join(
            TRACKING_DIR,
            f"{video_name}_tracking.csv"
        )

        output_root = os.path.join(
            OUTPUT_DIR,
            video_name
        )

        if not os.path.exists(tracking_path):

            print()
            print(
                f"[SKIP] {video_name}: "
                f"tracking CSV not found"
            )

            failed_videos += 1
            continue

        try:

            count = process_video(
                video_path,
                tracking_path,
                output_root,
                index,
                len(videos)
            )

            total_sequences += count
            successful_videos += 1

        except Exception as e:

            print()
            print(
                f"[ERROR] {video_name}: {e}"
            )

            failed_videos += 1

    print()
    print("=" * 80)
    print("BUILD FINISHED")
    print("=" * 80)
    print(
        f"Successful videos: "
        f"{successful_videos}"
    )
    print(
        f"Failed videos: "
        f"{failed_videos}"
    )
    print(
        f"Total sequences: "
        f"{total_sequences}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()