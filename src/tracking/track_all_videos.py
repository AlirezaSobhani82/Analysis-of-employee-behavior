import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\SlowFast_Project")
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import person_tracker


def main():
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        VIDEOS_DIR.glob("s_*.mp4"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    if not videos:
        raise RuntimeError(f"No videos found in: {VIDEOS_DIR}")

    print(f"Videos found: {len(videos)}")
    print("=" * 80)

    for index, video_path in enumerate(videos, start=1):
        video_name = video_path.stem

        person_tracker.VIDEO_PATH = str(video_path)
        person_tracker.OUTPUT_PATH = str(
            TRACKING_DIR / f"{video_name}_tracking.csv"
        )
        person_tracker.DEBUG_VIDEO_PATH = str(
            TRACKING_DIR / f"{video_name}_debug.mp4"
        )
        person_tracker.SAVE_DEBUG_VIDEO = False

        print()
        print("=" * 80)
        print(f"[{index}/{len(videos)}] Processing: {video_name}")
        print(f"Video: {video_path}")
        print(f"CSV  : {person_tracker.OUTPUT_PATH}")
        print("=" * 80)

        try:
            person_tracker.main()
            print(f"[DONE] {video_name}")
        except KeyboardInterrupt:
            print()
            print("Stopped by user.")
            break
        except Exception as exc:
            print(f"[ERROR] {video_name}: {exc}")
            print("Continuing with the next video...")

    print()
    print("=" * 80)
    print("ALL VIDEO TRACKING FINISHED")
    print(f"Output directory: {TRACKING_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()