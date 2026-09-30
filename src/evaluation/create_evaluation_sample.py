import csv
import os
import random
from collections import defaultdict


BASE_DIR = r"D:\SlowFast_Project"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "decision",
    "final_timeline.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "evaluation",
    "evaluation_sample.csv"
)

SAMPLES_PER_PERSON = 40

RANDOM_SEED = 42


def safe_int(value):

    try:
        return int(float(value))
    except:
        return 0


def load_data():

    rows = []

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def main():

    print("=" * 60)
    print("Evaluation Sample Builder")
    print("=" * 60)

    print("Loading final timeline...")

    rows = load_data()

    print(
        f"Total rows: {len(rows)}"
    )

    groups = defaultdict(list)

    for row in rows:

        key = (
            row["Video"],
            row["Person"]
        )

        groups[key].append(
            row
        )

    print(
        f"Video/Person groups: {len(groups)}"
    )

    random.seed(
        RANDOM_SEED
    )

    selected = []

    for key, group in groups.items():

        if len(group) <= SAMPLES_PER_PERSON:

            samples = group

        else:

            samples = random.sample(
                group,
                SAMPLES_PER_PERSON
            )

        selected.extend(
            samples
        )

    selected.sort(
        key=lambda x: (
            x["Video"],
            safe_int(x["Person"]),
            safe_int(x["Sequence"])
        )
    )

    output_rows = []

    for row in selected:

        output_rows.append({
            "Video": row["Video"],
            "Person": row["Person"],
            "Sequence": row["Sequence"],
            "Start_Frame": row["Start_Frame"],
            "End_Frame": row["End_Frame"],
            "Predicted_Decision": row[
                "Final_Decision"
            ],
            "Ground_Truth": ""
        })

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    fieldnames = [
        "Video",
        "Person",
        "Sequence",
        "Start_Frame",
        "End_Frame",
        "Predicted_Decision",
        "Ground_Truth"
    ]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    print()
    print("=" * 60)
    print("Evaluation Sample")
    print("=" * 60)

    print(
        f"Selected samples: {len(output_rows)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print(
        "Ground_Truth is empty and must be labeled manually."
    )

    print("=" * 60)
    print("Evaluation Sample Builder completed")
    print("=" * 60)


if __name__ == "__main__":
    main()