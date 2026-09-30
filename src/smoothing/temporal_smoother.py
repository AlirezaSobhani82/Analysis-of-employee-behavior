from pathlib import Path
import csv
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "decision"
    / "rule_engine.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "decision"
    / "final_timeline.csv"
)


def load_rows():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "Input CSV has no header."
            )

        return list(reader), reader.fieldnames


def save_rows(rows, fieldnames):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def count_labels(rows):

    counter = Counter()

    for row in rows:

        label = row.get(
            "Label",
            ""
        ).strip().upper()

        counter[label] += 1

    return counter


def count_changes(
    original_rows,
    final_rows
):

    changed = 0

    total = min(
        len(original_rows),
        len(final_rows)
    )

    for index in range(total):

        original_label = (
            original_rows[index]
            .get("Label", "")
            .strip()
            .upper()
        )

        final_label = (
            final_rows[index]
            .get("Label", "")
            .strip()
            .upper()
        )

        if original_label != final_label:
            changed += 1

    return changed


def main():

    print("=" * 70)
    print("TEMPORAL SMOOTHER")
    print("=" * 70)

    print(
        f"Input : {INPUT_PATH}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    rows, fieldnames = load_rows()

    print()
    print(
        f"Input rows: {len(rows)}"
    )

    before = count_labels(
        rows
    )

    print()
    print("Before smoothing:")

    print(
        f"WORKING      : "
        f"{before['WORKING']}"
    )

    print(
        f"NOT_WORKING  : "
        f"{before['NOT_WORKING']}"
    )

    print(
        f"NEUTRAL      : "
        f"{before['NEUTRAL']}"
    )

    final_rows = [
        row.copy()
        for row in rows
    ]

    after = count_labels(
        final_rows
    )

    changed_rows = count_changes(
        rows,
        final_rows
    )

    save_rows(
        final_rows,
        fieldnames
    )

    print()
    print("After smoothing:")

    print(
        f"WORKING      : "
        f"{after['WORKING']}"
    )

    print(
        f"NOT_WORKING  : "
        f"{after['NOT_WORKING']}"
    )

    print(
        f"NEUTRAL      : "
        f"{after['NEUTRAL']}"
    )

    print()
    print(
        f"Changed rows : "
        f"{changed_rows}"
    )

    print(
        f"Output rows  : "
        f"{len(final_rows)}"
    )

    print()
    print("=" * 70)
    print("TEMPORAL SMOOTHING COMPLETED")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()