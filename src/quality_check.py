from pathlib import Path
import csv
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACKING_PATH = PROJECT_ROOT / "outputs" / "tracking" / "person_tracking.csv"
ACTION_PATH = PROJECT_ROOT / "outputs" / "action" / "slowfast_predictions.csv"
TIMELINE_PATH = PROJECT_ROOT / "outputs" / "fusion" / "timeline.csv"
OBJECT_PATH = PROJECT_ROOT / "outputs" / "objects" / "object_context.csv"
POSE_PATH = PROJECT_ROOT / "outputs" / "pose" / "pose_context.csv"
RULE_PATH = PROJECT_ROOT / "outputs" / "decision" / "rule_engine.csv"
FINAL_PATH = PROJECT_ROOT / "outputs" / "decision" / "final_timeline.csv"
PRODUCTIVITY_PATH = PROJECT_ROOT / "outputs" / "productivity" / "productivity_report.csv"
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "s_001.mp4"
VISUALIZATION_PATH = PROJECT_ROOT / "outputs" / "visualization" / "final_visualization.mp4"


def read_csv(path):

    if not path.exists():
        return []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def check_file(path, name):

    exists = path.exists()

    if exists:
        size = path.stat().st_size

        print(
            f"[OK] {name}"
        )

        print(
            f"     Path: {path}"
        )

        print(
            f"     Size: {size:,} bytes"
        )

    else:

        print(
            f"[ERROR] {name}"
        )

        print(
            f"        Missing: {path}"
        )

    return exists


def unique_people(rows):

    people = set()

    for row in rows:

        if "Person" in row:

            try:
                people.add(
                    int(float(row["Person"]))
                )
            except:
                pass

    return sorted(people)


def check_tracking():

    print()
    print("=" * 70)
    print("1. TRACKING CHECK")
    print("=" * 70)

    rows = read_csv(
        TRACKING_PATH
    )

    print(
        f"Tracking rows: {len(rows):,}"
    )

    people = unique_people(
        rows
    )

    print(
        f"Persons: {people}"
    )

    frames = set()

    for row in rows:

        try:
            frames.add(
                int(float(row["Video_Frame"]))
            )
        except:
            pass

    print(
        f"Tracked frames: {len(frames):,}"
    )

    for person in people:

        person_rows = [
            row
            for row in rows
            if int(float(row["Person"])) == person
        ]

        person_frames = [
            int(float(row["Video_Frame"]))
            for row in person_rows
        ]

        if person_frames:

            print(
                f"Person {person}: "
                f"{len(person_rows):,} rows | "
                f"frames "
                f"{min(person_frames)}-"
                f"{max(person_frames)}"
            )


def check_sequences():

    print()
    print("=" * 70)
    print("2. SEQUENCE CHECK")
    print("=" * 70)

    timeline_rows = read_csv(
        TIMELINE_PATH
    )

    sequences = set()

    people = Counter()

    for row in timeline_rows:

        sequences.add(
            row["Sequence"]
        )

        people[
            int(float(row["Person"]))
        ] += 1

    print(
        f"Timeline sequences: "
        f"{len(sequences):,}"
    )

    for person in sorted(people):

        print(
            f"Person {person}: "
            f"{people[person]:,} sequences"
        )


def check_action():

    print()
    print("=" * 70)
    print("3. SLOWFAST ACTION CHECK")
    print("=" * 70)

    rows = read_csv(
        ACTION_PATH
    )

    print(
        f"Action rows: {len(rows):,}"
    )

    if not rows:
        return

    action_counter = Counter()

    for row in rows:

        action = row.get(
            "Action",
            row.get(
                "Class_Name",
                "UNKNOWN"
            )
        )

        action_counter[action] += 1

    for action, count in action_counter.most_common():

        print(
            f"{action:25} "
            f"{count:4}"
        )


def check_timeline():

    print()
    print("=" * 70)
    print("4. TIMELINE CHECK")
    print("=" * 70)

    rows = read_csv(
        TIMELINE_PATH
    )

    print(
        f"Timeline rows: {len(rows):,}"
    )

    actions = Counter()

    for row in rows:

        actions[
            row["Action"]
        ] += 1

    print()
    print("Actions:")

    for action, count in actions.most_common():

        print(
            f"{action:25} "
            f"{count:4}"
        )


def check_objects():

    print()
    print("=" * 70)
    print("5. OBJECT CHECK")
    print("=" * 70)

    rows = read_csv(
        OBJECT_PATH
    )

    print(
        f"Object records: {len(rows):,}"
    )

    if not rows:
        return

    objects = Counter()

    people = Counter()

    for row in rows:

        objects[
            row["Object"]
        ] += 1

        try:
            people[
                int(float(row["Person"]))
            ] += 1
        except:
            pass

    print()
    print("Objects:")

    for obj, count in objects.most_common():

        print(
            f"{obj:20} "
            f"{count:6}"
        )

    print()
    print("Object records by person:")

    for person, count in sorted(
        people.items()
    ):

        print(
            f"Person {person}: "
            f"{count:,}"
        )


def check_pose():

    print()
    print("=" * 70)
    print("6. POSE CHECK")
    print("=" * 70)

    rows = read_csv(
        POSE_PATH
    )

    print(
        f"Pose records: {len(rows):,}"
    )

    if not rows:
        return

    detected = 0
    not_detected = 0

    for row in rows:

        value = row.get(
            "Pose_Detected",
            ""
        ).strip().lower()

        if value in (
            "1",
            "true",
            "yes"
        ):

            detected += 1

        else:

            not_detected += 1

    total = (
        detected
        + not_detected
    )

    print(
        f"Pose detected: "
        f"{detected:,}"
    )

    print(
        f"Pose not detected: "
        f"{not_detected:,}"
    )

    if total > 0:

        ratio = (
            detected
            / total
            * 100
        )

        print(
            f"Detection ratio: "
            f"{ratio:.2f}%"
        )


def check_decision():

    print()
    print("=" * 70)
    print("7. RULE ENGINE CHECK")
    print("=" * 70)

    rows = read_csv(
        RULE_PATH
    )

    print(
        f"Rule Engine rows: "
        f"{len(rows):,}"
    )

    labels = Counter()

    for row in rows:

        labels[
            row["Label"]
        ] += 1

    for label, count in labels.items():

        print(
            f"{label:15} "
            f"{count:4}"
        )


def check_final():

    print()
    print("=" * 70)
    print("8. FINAL TIMELINE CHECK")
    print("=" * 70)

    rows = read_csv(
        FINAL_PATH
    )

    print(
        f"Final Timeline rows: "
        f"{len(rows):,}"
    )

    labels = Counter()

    for row in rows:

        labels[
            row["Label"]
        ] += 1

    print()

    for label in (
        "WORKING",
        "NOT_WORKING",
        "NEUTRAL"
    ):

        print(
            f"{label:15} "
            f"{labels[label]:4}"
        )


def check_productivity():

    print()
    print("=" * 70)
    print("9. PRODUCTIVITY CHECK")
    print("=" * 70)

    rows = read_csv(
        PRODUCTIVITY_PATH
    )

    print(
        f"Productivity rows: "
        f"{len(rows):,}"
    )

    for row in rows:

        print(
            f"Person {row['Person']} | "
            f"Total: "
            f"{row['Total_Time_Seconds']} sec | "
            f"Working: "
            f"{row['Working_Time_Seconds']} sec | "
            f"Productivity: "
            f"{row['Productivity_Percent']}%"
        )


def check_visualization():

    print()
    print("=" * 70)
    print("10. VISUALIZATION CHECK")
    print("=" * 70)

    if VISUALIZATION_PATH.exists():

        size = (
            VISUALIZATION_PATH.stat().st_size
        )

        print(
            "[OK] Final visualization exists"
        )

        print(
            f"Size: {size:,} bytes"
        )

    else:

        print(
            "[ERROR] Visualization not found"
        )


def check_files():

    print()
    print("=" * 70)
    print("FILE CHECK")
    print("=" * 70)

    files = [
        (VIDEO_PATH, "Input Video"),
        (TRACKING_PATH, "Tracking CSV"),
        (ACTION_PATH, "Action CSV"),
        (TIMELINE_PATH, "Timeline CSV"),
        (OBJECT_PATH, "Object CSV"),
        (POSE_PATH, "Pose CSV"),
        (RULE_PATH, "Rule Engine CSV"),
        (FINAL_PATH, "Final Timeline CSV"),
        (PRODUCTIVITY_PATH, "Productivity CSV"),
        (
            VISUALIZATION_PATH,
            "Visualization Video"
        )
    ]

    for path, name in files:

        check_file(
            path,
            name
        )


def main():

    print("=" * 70)
    print("EMPLOYEE ACTIVITY INTELLIGENCE")
    print("QUALITY CHECK")
    print("=" * 70)

    check_files()

    check_tracking()

    check_sequences()

    check_action()

    check_timeline()

    check_objects()

    check_pose()

    check_decision()

    check_final()

    check_productivity()

    check_visualization()

    print()
    print("=" * 70)
    print("QUALITY CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()