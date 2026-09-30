import os
import csv
import cv2
import mediapipe as mp

VIDEOS_DIR = r"D:\SlowFast_Project\data\videos"
TRACKING_DIR = r"D:\SlowFast_Project\outputs\tracking"
MODEL_PATH = r"D:\SlowFast_Project\models\pose_landmarker_full.task"
OUTPUT_DIR = r"D:\SlowFast_Project\outputs\pose"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pose_context.csv")

MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_POSE_TRACKING_CONFIDENCE = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tracking_data(tracking_path):
    tracking_data = {}

    with open(
        tracking_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            frame = int(float(row["Video_Frame"]))
            person_id = int(float(row["Person"]))

            tracking_data.setdefault(frame, []).append({
                "person_id": person_id,
                "x1": int(float(row["X1"])),
                "y1": int(float(row["Y1"])),
                "x2": int(float(row["X2"])),
                "y2": int(float(row["Y2"]))
            })

    return tracking_data


def calculate_angle(point_a, point_b):
    import math

    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    return round(angle, 3)


def calculate_pose_features(landmarks):
    if landmarks is None:
        return {
            "pose_detected": 0,
            "shoulder_angle": "",
            "body_angle": "",
            "left_hand_x": "",
            "left_hand_y": "",
            "right_hand_x": "",
            "right_hand_y": ""
        }

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    shoulder_angle = calculate_angle(
        (
            left_shoulder.x,
            left_shoulder.y
        ),
        (
            right_shoulder.x,
            right_shoulder.y
        )
    )

    shoulder_center_x = (
        left_shoulder.x +
        right_shoulder.x
    ) / 2.0

    shoulder_center_y = (
        left_shoulder.y +
        right_shoulder.y
    ) / 2.0

    hip_center_x = (
        left_hip.x +
        right_hip.x
    ) / 2.0

    hip_center_y = (
        left_hip.y +
        right_hip.y
    ) / 2.0

    body_angle = calculate_angle(
        (
            shoulder_center_x,
            shoulder_center_y
        ),
        (
            hip_center_x,
            hip_center_y
        )
    )

    return {
        "pose_detected": 1,
        "shoulder_angle": shoulder_angle,
        "body_angle": body_angle,
        "left_hand_x": round(left_wrist.x, 4),
        "left_hand_y": round(left_wrist.y, 4),
        "right_hand_x": round(right_wrist.x, 4),
        "right_hand_y": round(right_wrist.y, 4)
    }


def process_video(
    video_path,
    tracking_path,
    video_name,
    writer,
    landmarker
):

    print()
    print("=" * 60)
    print(f"Processing: {video_name}")
    print("=" * 60)

    tracking_data = load_tracking_data(
        tracking_path
    )

    print(
        f"Tracking frames loaded: "
        f"{len(tracking_data)}"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(
            f"Could not open video: "
            f"{video_path}"
        )
        return 0, 0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Video frames: {total_frames}"
    )

    processed_frames = 0
    pose_records = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frames += 1

        if processed_frames not in tracking_data:
            continue

        persons = tracking_data[
            processed_frames
        ]

        for person in persons:

            x1 = max(
                0,
                person["x1"]
            )

            y1 = max(
                0,
                person["y1"]
            )

            x2 = min(
                frame.shape[1],
                person["x2"]
            )

            y2 = min(
                frame.shape[0],
                person["y2"]
            )

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[
                y1:y2,
                x1:x2
            ]

            if person_crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(
                person_crop,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_crop
            )

            result = landmarker.detect(
                mp_image
            )

            landmarks = None

            if result.pose_landmarks:
                if len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]

            features = calculate_pose_features(
                landmarks
            )

            writer.writerow({
                "Video": video_name,
                "Video_Frame": processed_frames,
                "Person": person["person_id"],
                "Pose_Detected": features["pose_detected"],
                "Shoulder_Angle": features["shoulder_angle"],
                "Body_Angle": features["body_angle"],
                "Left_Hand_X": features["left_hand_x"],
                "Left_Hand_Y": features["left_hand_y"],
                "Right_Hand_X": features["right_hand_x"],
                "Right_Hand_Y": features["right_hand_y"]
            })

            pose_records += 1

        if processed_frames % 500 == 0:
            print(
                f"Processed frames: "
                f"{processed_frames}/"
                f"{total_frames}"
            )

    cap.release()

    print(
        f"Pose records: {pose_records}"
    )

    return processed_frames, pose_records


def main():

    if not os.path.exists(VIDEOS_DIR):
        raise FileNotFoundError(
            f"Videos directory not found: "
            f"{VIDEOS_DIR}"
        )

    if not os.path.exists(TRACKING_DIR):
        raise FileNotFoundError(
            f"Tracking directory not found: "
            f"{TRACKING_DIR}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found: "
            f"{MODEL_PATH}"
        )

    video_files = [
        file_name
        for file_name in os.listdir(VIDEOS_DIR)
        if file_name.lower().endswith(".mp4")
    ]

    video_files.sort()

    print("=" * 60)
    print("Pose Context Builder")
    print("=" * 60)
    print(
        f"Videos found: {len(video_files)}"
    )
    print(
        f"Model: {MODEL_PATH}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    base_options = BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_POSE_TRACKING_CONFIDENCE
    )

    fieldnames = [
        "Video",
        "Video_Frame",
        "Person",
        "Pose_Detected",
        "Shoulder_Angle",
        "Body_Angle",
        "Left_Hand_X",
        "Left_Hand_Y",
        "Right_Hand_X",
        "Right_Hand_Y"
    ]

    total_processed_frames = 0
    total_pose_records = 0
    processed_videos = 0

    with PoseLandmarker.create_from_options(options) as landmarker:

        with open(
            OUTPUT_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for index, file_name in enumerate(
                video_files,
                start=1
            ):

                video_name = os.path.splitext(
                    file_name
                )[0]

                video_path = os.path.join(
                    VIDEOS_DIR,
                    file_name
                )

                tracking_path = os.path.join(
                    TRACKING_DIR,
                    f"{video_name}_tracking.csv"
                )

                print()
                print(
                    f"[{index}/{len(video_files)}] "
                    f"{video_name}"
                )

                if not os.path.exists(
                    tracking_path
                ):
                    print(
                        f"Tracking file not found: "
                        f"{tracking_path}"
                    )
                    continue

                processed_frames, pose_records = process_video(
                    video_path,
                    tracking_path,
                    video_name,
                    writer,
                    landmarker
                )

                total_processed_frames += processed_frames
                total_pose_records += pose_records
                processed_videos += 1

    print()
    print("=" * 60)
    print("Pose Context completed")
    print("=" * 60)
    print(
        f"Videos processed: "
        f"{processed_videos}"
    )
    print(
        f"Total processed frames: "
        f"{total_processed_frames}"
    )
    print(
        f"Total pose records: "
        f"{total_pose_records}"
    )
    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()