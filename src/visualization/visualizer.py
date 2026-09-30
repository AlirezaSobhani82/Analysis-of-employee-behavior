from pathlib import Path
import csv
import cv2
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "s_018.mp4"
TRACKING_PATH = PROJECT_ROOT / "outputs" / "tracking" / "s_018_tracking.csv"
TIMELINE_PATH = PROJECT_ROOT / "outputs" / "decision" / "final_timeline.csv"
PRODUCTIVITY_PATH = PROJECT_ROOT / "outputs" / "productivity" / "productivity_report.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualization"
OUTPUT_VIDEO = OUTPUT_DIR / "s_018_visualization.mp4"

FONT = cv2.FONT_HERSHEY_SIMPLEX

PERSON_COLORS = {
    1: (0, 255, 0),
    2: (255, 165, 0),
    3: (255, 0, 255),
    4: (0, 255, 255),
    5: (255, 0, 0)
}


def load_tracking():

    tracking = defaultdict(list)

    with open(
        TRACKING_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            frame = int(float(row["Video_Frame"]))
            person = int(float(row["Person"]))
            x1 = int(float(row["X1"]))
            y1 = int(float(row["Y1"]))
            x2 = int(float(row["X2"]))
            y2 = int(float(row["Y2"]))

            tracking[frame].append({
                "Person": person,
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2
            })

    return tracking


def load_timeline():

    timeline = defaultdict(list)

    with open(
        TIMELINE_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["Video"] != "s_018":
                continue

            start_frame = int(float(row["Start_Frame"]))
            end_frame = int(float(row["End_Frame"]))
            person = int(float(row["Person"]))

            sequence = row["Sequence"]
            action = row["Action"]
            label = row["Final_Decision"]

            for frame in range(
                start_frame,
                end_frame + 1
            ):

                timeline[frame].append({
                    "Person": person,
                    "Sequence": sequence,
                    "Action": action,
                    "Label": label
                })

    return timeline


def load_productivity():

    productivity = {}

    if not PRODUCTIVITY_PATH.exists():
        return productivity

    with open(
        PRODUCTIVITY_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            person = row["Person"]

            productivity[person] = {
                "Productivity": float(
                    row["Productivity_Percent"]
                ),
                "Working": float(
                    row["Working_Ratio_Percent"]
                )
            }

    return productivity


def get_color(person):

    if person in PERSON_COLORS:
        return PERSON_COLORS[person]

    colors = [
        (0, 255, 0),
        (255, 165, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 0, 0),
        (0, 128, 255),
        (128, 0, 255)
    ]

    return colors[
        (person - 1) % len(colors)
    ]


def get_status_color(label):

    if label == "WORKING":
        return (0, 255, 0)

    if label == "NOT_WORKING":
        return (0, 0, 255)

    return (0, 255, 255)


def draw_text(
    frame,
    text,
    position,
    color,
    scale=0.55,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_person(
    frame,
    person,
    box,
    timeline_data
):

    x1 = box["X1"]
    y1 = box["Y1"]
    x2 = box["X2"]
    y2 = box["Y2"]

    person_color = get_color(person)

    action = "Unknown"
    label = "NEUTRAL"

    for item in timeline_data:

        if item["Person"] == person:

            action = item["Action"]
            label = item["Label"]

            break

    status_color = get_status_color(label)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        person_color,
        2
    )

    text_x = x1
    text_y = max(
        25,
        y1 - 10
    )

    draw_text(
        frame,
        f"Person {person}",
        (text_x, text_y),
        person_color,
        0.65,
        2
    )

    draw_text(
        frame,
        f"Action: {action}",
        (
            text_x,
            text_y + 25
        ),
        (255, 255, 255),
        0.50,
        2
    )

    draw_text(
        frame,
        f"Status: {label}",
        (
            text_x,
            text_y + 48
        ),
        status_color,
        0.55,
        2
    )


def draw_productivity_panel(
    frame,
    productivity
):

    height, width = frame.shape[:2]

    panel_height = 120

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            0,
            height - panel_height
        ),
        (
            width,
            height
        ),
        (20, 20, 20),
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0
    )

    draw_text(
        frame,
        "PRODUCTIVITY",
        (20, height - 90),
        (255, 255, 255),
        0.65,
        2
    )

    x = 20
    y = height - 55

    persons = []

    for person in productivity:

        if person == "TOTAL":
            persons.append((999999, person))
        else:
            persons.append((int(person), person))

    persons.sort()

    for _, person in persons:

        data = productivity[person]

        if person == "TOTAL":

            text = (
                f"TOTAL: "
                f"{data['Productivity']:.2f}%"
            )

        else:

            text = (
                f"P{person}: "
                f"{data['Productivity']:.2f}%"
            )

        draw_text(
            frame,
            text,
            (x, y),
            (255, 255, 255),
            0.55,
            2
        )

        x += 180


def main():

    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    print(
        f"Video       : {VIDEO_PATH}"
    )

    print(
        f"Tracking    : {TRACKING_PATH}"
    )

    print(
        f"Timeline    : {TIMELINE_PATH}"
    )

    print(
        f"Productivity: {PRODUCTIVITY_PATH}"
    )

    print(
        f"Output      : {OUTPUT_VIDEO}"
    )

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    if not TRACKING_PATH.exists():

        raise FileNotFoundError(
            f"Tracking file not found: {TRACKING_PATH}"
        )

    if not TIMELINE_PATH.exists():

        raise FileNotFoundError(
            f"Timeline file not found: {TIMELINE_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Loading tracking...")

    tracking = load_tracking()

    print(
        f"Tracking frames: {len(tracking)}"
    )

    print("Loading timeline...")

    timeline = load_timeline()

    print(
        f"Timeline frames: {len(timeline)}"
    )

    print("Loading productivity...")

    productivity = load_productivity()

    print(
        f"Productivity records: "
        f"{len(productivity)}"
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"FPS          : {fps}"
    )

    print(
        f"Resolution   : "
        f"{width}x{height}"
    )

    print(
        f"Total frames : "
        f"{total_frames}"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        frame_tracking = tracking.get(
            frame_number,
            []
        )

        frame_timeline = timeline.get(
            frame_number,
            []
        )

        for person_data in frame_tracking:

            person = person_data["Person"]

            draw_person(
                frame,
                person,
                person_data,
                frame_timeline
            )

        draw_productivity_panel(
            frame,
            productivity
        )

        draw_text(
            frame,
            f"Frame: {frame_number}",
            (
                20,
                30
            ),
            (255, 255, 255),
            0.55,
            2
        )

        writer.write(
            frame
        )

        if (
            frame_number % 500 == 0
            or frame_number == total_frames
        ):

            percent = (
                frame_number
                / total_frames
                * 100
            )

            print(
                f"Processed: "
                f"{frame_number}/"
                f"{total_frames} "
                f"({percent:.1f}%)"
            )

    cap.release()
    writer.release()

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)

    print(
        f"Processed frames: "
        f"{frame_number}"
    )

    print(
        f"Output: {OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()
