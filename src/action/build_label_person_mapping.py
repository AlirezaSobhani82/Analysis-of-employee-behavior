import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(r"D:\SlowFast_Project")

LABEL_FILE = PROJECT_ROOT / "data" / "labels" / "s_001.txt"
TRACKING_FILE = PROJECT_ROOT / "outputs" / "tracking" / "person_tracking.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "action" / "label_person_mapping.csv"

AMBIGUOUS_LINES = list(range(5, 14))


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
                "start_frame": start_frame,
                "end_frame": end_frame
            })

    return labels


def load_tracking():
    tracking = defaultdict(set)

    with open(TRACKING_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = int(float(row["Video_Frame"]))
            person = int(row["Person"])

            tracking[frame].add(person)

    return tracking


def calculate_person_coverage(start_frame, end_frame, tracking):
    total_frames = end_frame - start_frame + 1

    person_counts = defaultdict(int)

    for frame in range(start_frame, end_frame + 1):
        persons = tracking.get(frame, set())

        for person in persons:
            person_counts[person] += 1

    results = []

    for person, count in sorted(person_counts.items()):
        coverage = count / total_frames

        results.append({
            "person": person,
            "tracked_frames": count,
            "coverage": coverage
        })

    return results


def get_action_name(class_id):
    actions = {
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
        18: "Sneezing"
    }

    return actions.get(class_id, "Unknown")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    tracking = load_tracking()

    selected_labels = [
        label for label in labels
        if label["line"] in AMBIGUOUS_LINES
    ]

    output_rows = []

    print()
    print("=" * 80)
    print("LABEL -> PERSON MAPPING")
    print("=" * 80)

    for label in selected_labels:
        line_number = label["line"]
        action = get_action_name(label["class_id"])
        start_frame = label["start_frame"]
        end_frame = label["end_frame"]

        person_results = calculate_person_coverage(
            start_frame,
            end_frame,
            tracking
        )

        print()
        print(
            f"Line {line_number:02d} | "
            f"{action} | "
            f"{start_frame}-{end_frame}"
        )

        for result in person_results:
            person = result["person"]
            tracked_frames = result["tracked_frames"]
            coverage = result["coverage"]

            print(
                f"  Person {person}: "
                f"{tracked_frames} frames | "
                f"coverage={coverage:.2%}"
            )

            output_rows.append({
                "Line": line_number,
                "Action": action,
                "Start_Frame": start_frame,
                "End_Frame": end_frame,
                "Person": person,
                "Tracked_Frames": tracked_frames,
                "Coverage": f"{coverage:.4f}"
            })

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "Line",
            "Action",
            "Start_Frame",
            "End_Frame",
            "Person",
            "Tracked_Frames",
            "Coverage"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print("=" * 80)
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()