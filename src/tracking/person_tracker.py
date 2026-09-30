import csv
import math
import os
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = r"D:\SlowFast_Project"
VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "videos",
    "s_001.mp4"
)
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "yolo11n.pt"
)

TRACKER_PATH = os.path.join(
    PROJECT_ROOT,
    "botsort.yaml"
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "tracking",
    "person_tracking.csv"
)
DEBUG_VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "tracking",
    "debug_tracking.mp4"
)
SAVE_DEBUG_VIDEO = True
DETECT_CONF = 0.10
NMS_IOU = 0.50
IMAGE_SIZE = 1280
PERSON_CLASS = 0
MIN_BOX_AREA = 2500
CONFIRM_FRAMES = 8
TENTATIVE_MAX_MISSING = 5
MAX_LOST_FRAMES = 450
DUP_IOU = 0.70
DUP_CONTAIN = 0.90
USE_TRACKER_REID = True
USE_COLOR_HIST = True
GALLERY_SIZE = 30
GALLERY_EVERY = 3
GALLERY_MIN_CONF = 0.50
GALLERY_MAX_OVERLAP_IOU = 0.25
BASE_POSITION_DISTANCE = 1.0
POSITION_GROWTH = 0.04
MAX_POSITION_DISTANCE = 5.0
VELOCITY_HORIZON = 15
MAX_APPEARANCE_DIST = 0.45
W_APP = 0.45
W_POS = 0.30
W_SIZE = 0.15
W_IOU = 0.10
RELINK_COST_THRESHOLD = 0.58
STRONG_GEOMETRY_THRESHOLD = 0.32
RELAXED_APPEARANCE_THRESHOLD = 0.72
RELAXED_RELINK_COST = 0.72
RAW_ID_KEEP_FRAMES = 30
VELOCITY_ALPHA = 0.65
MAX_VELOCITY = 120.0
VERBOSE_EVENTS = True
PRINT_INTERVAL = 500
BIG = 1e6
PALETTE = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
    (255, 160, 40),
    (160, 80, 255),
]


def box_area(box):
    return (
        max(1.0, box[2] - box[0])
        * max(1.0, box[3] - box[1])
    )


def box_center(box):
    return (
        (box[0] + box[2]) * 0.5,
        (box[1] + box[3]) * 0.5
    )


def box_iou(a, b):
    iw = max(
        0.0,
        min(a[2], b[2])
        - max(a[0], b[0])
    )

    ih = max(
        0.0,
        min(a[3], b[3])
        - max(a[1], b[1])
    )

    inter = iw * ih

    if inter <= 0:
        return 0.0

    union = (
        box_area(a)
        + box_area(b)
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def box_containment(a, b):
    iw = max(
        0.0,
        min(a[2], b[2])
        - max(a[0], b[0])
    )

    ih = max(
        0.0,
        min(a[3], b[3])
        - max(a[1], b[1])
    )

    inter = iw * ih

    if inter <= 0:
        return 0.0

    smaller_area = min(
        box_area(a),
        box_area(b)
    )

    if smaller_area <= 0:
        return 0.0

    return inter / smaller_area


def sanitize_box(
    box,
    width,
    height
):
    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(
        0.0,
        min(x1, width - 1)
    )

    y1 = max(
        0.0,
        min(y1, height - 1)
    )

    x2 = max(
        0.0,
        min(x2, width - 1)
    )

    y2 = max(
        0.0,
        min(y2, height - 1)
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        x1,
        y1,
        x2,
        y2
    ]


def color_feature(
    frame,
    box
):
    import cv2

    x1, y1, x2, y2 = [
        int(v)
        for v in box
    ]

    w = x2 - x1
    h = y2 - y1

    if w < 8 or h < 16:
        return None

    cx1 = x1 + int(0.15 * w)
    cx2 = x2 - int(0.15 * w)

    cy1 = y1 + int(0.15 * h)
    cy2 = y2 - int(0.05 * h)

    crop = frame[
        max(0, cy1):cy2,
        max(0, cx1):cx2
    ]

    if crop.size == 0:
        return None

    if crop.shape[0] < 4:
        return None

    if crop.shape[1] < 2:
        return None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    mid = hsv.shape[0] // 2

    parts = []

    for part in (
        hsv[:mid],
        hsv[mid:]
    ):
        hist = cv2.calcHist(
            [part],
            [0, 1, 2],
            None,
            [12, 4, 4],
            [0, 180, 0, 256, 0, 256]
        ).flatten().astype(
            np.float32
        )

        total = float(
            hist.sum()
        )

        if total <= 0:
            return None

        hist = np.sqrt(
            hist / total
        )

        parts.append(hist)

    feat = np.concatenate(
        parts
    )

    norm = float(
        np.linalg.norm(feat)
    )

    if norm <= 1e-8:
        return None

    return feat / norm


def feature_distance(
    a,
    b
):
    if a is None or b is None:
        return None

    try:
        a = np.asarray(
            a,
            dtype=np.float32
        ).reshape(-1)

        b = np.asarray(
            b,
            dtype=np.float32
        ).reshape(-1)

        if a.shape != b.shape:
            return None

        na = float(
            np.linalg.norm(a)
        )

        nb = float(
            np.linalg.norm(b)
        )

        if na <= 1e-8 or nb <= 1e-8:
            return None

        a = a / na
        b = b / nb

        return float(
            np.clip(
                1.0 - np.dot(a, b),
                0.0,
                2.0
            )
        )

    except Exception:
        return None


def gallery_distance(
    gallery,
    feature,
    top=3
):
    if feature is None:
        return None

    if not gallery:
        return None

    distances = []

    for item in gallery:
        d = feature_distance(
            item,
            feature
        )

        if d is not None:
            distances.append(d)

    if not distances:
        return None

    distances.sort()

    return float(
        np.mean(
            distances[:min(
                top,
                len(distances)
            )]
        )
    )


def get_tracker_object(model):
    try:
        trackers = getattr(
            model.predictor,
            "trackers",
            None
        )

        if trackers:
            return trackers[0]

    except Exception:
        pass

    return None


def extract_tracker_features(
    tracker
):
    features = {}

    if tracker is None:
        return features

    tracks = getattr(
        tracker,
        "tracked_stracks",
        []
    )

    for track in tracks:
        track_id = getattr(
            track,
            "track_id",
            None
        )

        if track_id is None:
            continue

        feat = getattr(
            track,
            "smooth_feat",
            None
        )

        if feat is None:
            feat = getattr(
                track,
                "curr_feat",
                None
            )

        if feat is None:
            continue

        try:
            feat = np.asarray(
                feat,
                dtype=np.float32
            ).reshape(-1)

            norm = float(
                np.linalg.norm(feat)
            )

            if norm > 1e-8:
                features[
                    int(track_id)
                ] = feat / norm

        except Exception:
            continue

    return features


def prediction_to_detections(
    result,
    features,
    width,
    height
):
    detections = []

    if result.boxes is None:
        return detections

    if len(result.boxes) == 0:
        return detections

    if result.boxes.id is None:
        return detections

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    confs = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    raw_ids = (
        result.boxes.id
        .cpu()
        .numpy()
        .astype(int)
    )

    for i in range(
        len(boxes)
    ):
        box = sanitize_box(
            boxes[i],
            width,
            height
        )

        if box is None:
            continue

        if box_area(box) < MIN_BOX_AREA:
            continue

        raw_id = int(
            raw_ids[i]
        )

        detections.append({
            "raw_id": raw_id,
            "box": box,
            "confidence": float(
                confs[i]
            ),
            "reid": (
                features.get(
                    raw_id
                )
                if USE_TRACKER_REID
                else None
            )
        })

    return detections


class Identity:

    def __init__(
        self,
        uid,
        det,
        frame_index
    ):
        self.uid = uid

        self.person_id = None

        self.raw_id = det[
            "raw_id"
        ]

        self.box = list(
            det["box"]
        )

        self.last_center = (
            box_center(
                det["box"]
            )
        )

        self.velocity = (
            0.0,
            0.0
        )

        self.confidence = (
            det["confidence"]
        )

        self.first_seen = (
            frame_index
        )

        self.last_seen = (
            frame_index
        )

        self.hits = 0

        self.confirmed = False

        self.visible = False

        self.reid_gallery = deque(
            maxlen=GALLERY_SIZE
        )

        self.color_gallery = deque(
            maxlen=GALLERY_SIZE
        )

        self.pending = []
    @property
    def name(self):
        if self.person_id is not None:
            return f"P{self.person_id}"

        return f"tmp{self.uid}"

class IdentityManager:

    def __init__(
        self,
        width,
        height
    ):
        self.width = width
        self.height = height

        self.identities = {}

        self.raw_to_uid = {}

        self.raw_last_seen = {}

        self.next_uid = 1

        self.next_person_id = 1

        self.rows = []

        self.stats = {
            "confirmed": 0,
            "relinked": 0,
            "merged": 0,
            "discarded": 0,
            "duplicates": 0,
        }

    def update(
        self,
        frame_index,
        frame,
        detections
    ):
        for ident in self.identities.values():
            ident.visible = False

        detections = self._remove_duplicates(
            detections
        )

        current_raw_ids = {
            det["raw_id"]
            for det in detections
        }

        for raw_id in list(
            self.raw_to_uid.keys()
        ):
            last_seen = self.raw_last_seen.get(
                raw_id,
                -1
            )

            if raw_id not in current_raw_ids:
                if (
                    frame_index
                    - last_seen
                    > RAW_ID_KEEP_FRAMES
                ):
                    del self.raw_to_uid[
                        raw_id
                    ]

                    self.raw_last_seen.pop(
                        raw_id,
                        None
                    )

        for det in detections:
            self.raw_last_seen[
                det["raw_id"]
            ] = frame_index

        known = {}
        unknown = []

        for det in detections:
            uid = self.raw_to_uid.get(
                det["raw_id"]
            )

            if (
                uid is not None
                and uid in self.identities
            ):
                if uid in known:
                    self.stats[
                        "duplicates"
                    ] += 1

                    if (
                        det["confidence"]
                        > known[uid][
                            "confidence"
                        ]
                    ):
                        known[uid] = det
                else:
                    known[uid] = det
            else:
                unknown.append(det)

        validated_known = {}
        rejected_known = []

        for uid, det in known.items():
            ident = self.identities[
                uid
            ]

            if self._known_assignment_is_valid(
                ident,
                det,
                frame_index
            ):
                validated_known[
                    uid
                ] = det
            else:
                rejected_known.append(
                    det
                )

                self.raw_to_uid.pop(
                    det["raw_id"],
                    None
                )

                self.raw_last_seen.pop(
                    det["raw_id"],
                    None
                )

        unknown.extend(
            rejected_known
        )

        for uid, det in validated_known.items():
            self._update_identity(
                self.identities[uid],
                det,
                frame_index,
                frame,
                detections
            )

        self._link_unknown(
            unknown,
            frame_index,
            frame,
            detections
        )

        self._confirm_ready(
            frame_index
        )

        self._emit_rows(
            frame_index
        )

        self._cleanup(
            frame_index
        )

        return [
            i
            for i in self.identities.values()
            if i.visible
        ]

    def finalize(self):
        for ident in self.identities.values():
            if not ident.confirmed:
                self.stats[
                    "discarded"
                ] += 1

    def write_csv(
        self,
        path
    ):
        fieldnames = [
            "Video_Frame",
            "Person",
            "X1",
            "Y1",
            "X2",
            "Y2",
            "Confidence",
            "Missing"
        ]

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for (
                frame,
                pid,
                x1,
                y1,
                x2,
                y2,
                conf
            ) in sorted(
                self.rows,
                key=lambda r: (
                    r[0],
                    r[1]
                )
            ):
                writer.writerow({
                    "Video_Frame": frame,
                    "Person": pid,
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
                    ),
                    "Confidence": round(
                        conf,
                        4
                    ),
                    "Missing": 0,
                })

    def person_summary(self):
        info = {}

        for frame, pid, *_ in self.rows:
            if pid not in info:
                info[pid] = [
                    frame,
                    frame,
                    0
                ]

            info[pid][0] = min(
                info[pid][0],
                frame
            )

            info[pid][1] = max(
                info[pid][1],
                frame
            )

            info[pid][2] += 1

        return info

    def _remove_duplicates(
        self,
        detections
    ):
        def priority(det):
            uid = self.raw_to_uid.get(
                det["raw_id"]
            )

            ident = self.identities.get(
                uid
            )

            return (
                1 if (
                    ident is not None
                    and ident.confirmed
                )
                else 0,
                det["confidence"]
            )

        kept = []

        for det in sorted(
            detections,
            key=priority,
            reverse=True
        ):
            is_dup = False

            for other in kept:
                iou = box_iou(
                    det["box"],
                    other["box"]
                )

                containment = box_containment(
                    det["box"],
                    other["box"]
                )

                if (
                    iou > DUP_IOU
                    or containment > DUP_CONTAIN
                ):
                    is_dup = True
                    break

            if is_dup:
                self.stats[
                    "duplicates"
                ] += 1
            else:
                kept.append(det)

        return kept

    def _known_assignment_is_valid(
        self,
        ident,
        det,
        frame_index
    ):
        gap = max(
            1,
            frame_index
            - ident.last_seen
        )

        if gap <= 1:
            return True

        last_center = box_center(
            ident.box
        )

        det_center = box_center(
            det["box"]
        )

        horizon = min(
            gap,
            VELOCITY_HORIZON
        )

        predicted = (
            last_center[0]
            + ident.velocity[0]
            * horizon,

            last_center[1]
            + ident.velocity[1]
            * horizon
        )

        distance_last = (
            math.dist(
                det_center,
                last_center
            )
        )

        distance_pred = (
            math.dist(
                det_center,
                predicted
            )
        )

        distance = min(
            distance_last,
            distance_pred
        )

        height = max(
            ident.box[3]
            - ident.box[1],
            1.0
        )

        allowed = (
            height
            * min(
                MAX_POSITION_DISTANCE,
                BASE_POSITION_DISTANCE
                + POSITION_GROWTH * gap
            )
        )

        if distance <= allowed:
            return True

        if ident.confirmed:
            return gap <= 3

        return False

    def _link_unknown(
        self,
        unknown,
        frame_index,
        frame,
        detections
    ):
        if not unknown:
            return

        for det in unknown:
            if (
                USE_COLOR_HIST
                and frame is not None
                and "color" not in det
            ):
                det["color"] = color_feature(
                    frame,
                    det["box"]
                )

        candidates = [
            i
            for i in self.identities.values()
            if (
                not i.visible
                and self._can_be_relinked(
                    i,
                    frame_index
                )
            )
        ]

        assigned = {}

        if candidates:
            cost = np.full(
                (
                    len(unknown),
                    len(candidates)
                ),
                BIG,
                dtype=np.float64
            )

            for r, det in enumerate(
                unknown
            ):
                for c, ident in enumerate(
                    candidates
                ):
                    cost[r, c] = (
                        self._match_cost(
                            ident,
                            det["box"],
                            det.get(
                                "reid"
                            ),
                            det.get(
                                "color"
                            ),
                            frame_index
                        )
                    )

            rows, cols = (
                linear_sum_assignment(
                    cost
                )
            )

            for r, c in zip(
                rows,
                cols
            ):
                value = float(
                    cost[r, c]
                )

                threshold = (
                    RELAXED_RELINK_COST
                    if self._geometry_is_strong(
                        candidates[c],
                        unknown[r]["box"],
                        frame_index
                    )
                    else RELINK_COST_THRESHOLD
                )

                if value <= threshold:
                    assigned[r] = (
                        candidates[c],
                        value
                    )

        for r, det in enumerate(
            unknown
        ):
            if r in assigned:
                ident, cost_value = (
                    assigned[r]
                )

                gap = (
                    frame_index
                    - ident.last_seen
                )

                self.raw_to_uid[
                    det["raw_id"]
                ] = ident.uid

                self.raw_last_seen[
                    det["raw_id"]
                ] = frame_index

                self._update_identity(
                    ident,
                    det,
                    frame_index,
                    frame,
                    detections
                )

                if ident.confirmed:
                    self.stats[
                        "relinked"
                    ] += 1

                    self._log(
                        f"frame {frame_index}: "
                        f"raw {det['raw_id']} "
                        f"re-linked to "
                        f"{ident.name} "
                        f"(lost {gap} frames, "
                        f"cost {cost_value:.2f})"
                    )

            else:
                ident = Identity(
                    self.next_uid,
                    det,
                    frame_index
                )

                self.next_uid += 1

                self.identities[
                    ident.uid
                ] = ident

                self.raw_to_uid[
                    det["raw_id"]
                ] = ident.uid

                self.raw_last_seen[
                    det["raw_id"]
                ] = frame_index

                self._update_identity(
                    ident,
                    det,
                    frame_index,
                    frame,
                    detections
                )

    @staticmethod
    def _can_be_relinked(
        ident,
        frame_index
    ):
        gap = (
            frame_index
            - ident.last_seen
        )

        limit = (
            MAX_LOST_FRAMES
            if ident.confirmed
            else TENTATIVE_MAX_MISSING
        )

        return (
            1 <= gap <= limit
        )

    def _geometry_metrics(
        self,
        ident,
        box,
        frame_index
    ):
        gap = max(
            1,
            frame_index
            - ident.last_seen
        )

        ident_height = max(
            ident.box[3]
            - ident.box[1],
            1.0
        )

        det_height = max(
            box[3]
            - box[1],
            1.0
        )

        reference_height = max(
            ident_height,
            det_height
        )

        last_center = box_center(
            ident.box
        )

        det_center = box_center(
            box
        )

        horizon = min(
            gap,
            VELOCITY_HORIZON
        )

        predicted_center = (
            last_center[0]
            + ident.velocity[0]
            * horizon,

            last_center[1]
            + ident.velocity[1]
            * horizon
        )

        distance_last = math.dist(
            det_center,
            last_center
        )

        distance_pred = math.dist(
            det_center,
            predicted_center
        )

        distance = min(
            distance_last,
            distance_pred
        )

        normalized_distance = (
            distance
            / reference_height
        )

        allowed_distance = min(
            MAX_POSITION_DISTANCE,
            BASE_POSITION_DISTANCE
            + POSITION_GROWTH * gap
        )

        position_cost = min(
            1.0,
            normalized_distance
            / max(
                allowed_distance,
                1e-6
            )
        )

        size_cost = min(
            1.0,
            abs(
                math.log(
                    det_height
                    / ident_height
                )
            )
            / math.log(2.0)
        )

        predicted_box = [
            box[0]
            + (
                predicted_center[0]
                - last_center[0]
            ),
            box[1]
            + (
                predicted_center[1]
                - last_center[1]
            ),
            box[2]
            + (
                predicted_center[0]
                - last_center[0]
            ),
            box[3]
            + (
                predicted_center[1]
                - last_center[1]
            )
        ]

        predicted_iou = box_iou(
            predicted_box,
            box
        )

        return (
            normalized_distance,
            position_cost,
            size_cost,
            predicted_iou,
            allowed_distance
        )

    def _geometry_is_strong(
        self,
        ident,
        box,
        frame_index
    ):
        (
            distance,
            position_cost,
            size_cost,
            predicted_iou,
            allowed
        ) = self._geometry_metrics(
            ident,
            box,
            frame_index
        )

        geometry_score = (
            0.55 * position_cost
            + 0.25 * size_cost
            + 0.20 * (
                1.0 - predicted_iou
            )
        )

        return (
            geometry_score
            <= STRONG_GEOMETRY_THRESHOLD
        )

    def _appearance_cost(
        self,
        ident,
        reid,
        color
    ):
        values = []

        reid_distance = gallery_distance(
            ident.reid_gallery,
            reid,
            top=5
        )

        color_distance = gallery_distance(
            ident.color_gallery,
            color,
            top=5
        )

        if reid_distance is not None:
            values.append(
                reid_distance
            )

        if color_distance is not None:
            values.append(
                color_distance
            )

        if not values:
            return 0.5

        if (
            reid_distance is not None
            and color_distance is not None
        ):
            return (
                0.70 * reid_distance
                + 0.30 * color_distance
            )

        return float(
            values[0]
        )

    def _match_cost(
        self,
        ident,
        box,
        reid,
        color,
        frame_index
    ):
        (
            normalized_distance,
            position_cost,
            size_cost,
            predicted_iou,
            allowed_distance
        ) = self._geometry_metrics(
            ident,
            box,
            frame_index
        )

        if (
            normalized_distance
            > allowed_distance
        ):
            return BIG

        appearance = self._appearance_cost(
            ident,
            reid,
            color
        )

        strong_geometry = (
            self._geometry_is_strong(
                ident,
                box,
                frame_index
            )
        )

        if (
            appearance
            > MAX_APPEARANCE_DIST
        ):
            if not strong_geometry:
                return BIG

            appearance_for_cost = min(
                1.0,
                appearance
                / RELAXED_APPEARANCE_THRESHOLD
            )
        else:
            appearance_for_cost = (
                appearance
                / MAX_APPEARANCE_DIST
            )

        iou_cost = (
            1.0
            - predicted_iou
        )

        if strong_geometry:
            app_weight = 0.32
            pos_weight = 0.38
            size_weight = 0.18
            iou_weight = 0.12
        else:
            app_weight = W_APP
            pos_weight = W_POS
            size_weight = W_SIZE
            iou_weight = W_IOU

        total = (
            app_weight
            * appearance_for_cost
            + pos_weight
            * position_cost
            + size_weight
            * size_cost
            + iou_weight
            * iou_cost
        )

        if strong_geometry:
            return min(
                total,
                RELAXED_RELINK_COST
            )

        return total

    def _tentative_cost(
        self,
        other,
        tentative,
        frame_index
    ):
        best = BIG

        reids = list(
            tentative.reid_gallery
        )

        colors = list(
            tentative.color_gallery
        )

        for reid in reids:
            best = min(
                best,
                self._match_cost(
                    other,
                    tentative.box,
                    reid,
                    None,
                    frame_index
                )
            )

        for color in colors:
            best = min(
                best,
                self._match_cost(
                    other,
                    tentative.box,
                    None,
                    color,
                    frame_index
                )
            )

        current_cost = self._match_cost(
            other,
            tentative.box,
            None,
            None,
            frame_index
        )

        best = min(
            best,
            current_cost
        )

        return best

    def _update_identity(
        self,
        ident,
        det,
        frame_index,
        frame,
        all_dets
    ):
        gap = max(
            1,
            frame_index
            - ident.last_seen
        )

        old_center = box_center(
            ident.box
        )

        new_center = box_center(
            det["box"]
        )

        dx = (
            new_center[0]
            - old_center[0]
        ) / gap

        dy = (
            new_center[1]
            - old_center[1]
        ) / gap

        dx = max(
            -MAX_VELOCITY,
            min(
                dx,
                MAX_VELOCITY
            )
        )

        dy = max(
            -MAX_VELOCITY,
            min(
                dy,
                MAX_VELOCITY
            )
        )

        alpha = VELOCITY_ALPHA

        ident.velocity = (
            alpha * ident.velocity[0]
            + (1.0 - alpha) * dx,

            alpha * ident.velocity[1]
            + (1.0 - alpha) * dy
        )

        ident.box = list(
            det["box"]
        )

        ident.last_center = (
            new_center
        )

        ident.raw_id = (
            det["raw_id"]
        )

        ident.confidence = (
            det["confidence"]
        )

        ident.last_seen = (
            frame_index
        )

        ident.hits += 1

        ident.visible = True

        self._maybe_store_features(
            ident,
            det,
            frame,
            all_dets
        )

    def _maybe_store_features(
        self,
        ident,
        det,
        frame,
        all_dets
    ):
        x1, y1, x2, y2 = det[
            "box"
        ]

        margin = 3

        if (
            x1 <= margin
            or y1 <= margin
            or x2 >= self.width - margin
            or y2 >= self.height - margin
        ):
            return

        if (
            det["confidence"]
            < GALLERY_MIN_CONF
        ):
            return

        for other in all_dets:
            if other is det:
                continue

            if (
                box_iou(
                    det["box"],
                    other["box"]
                )
                > GALLERY_MAX_OVERLAP_IOU
            ):
                return

        if (
            ident.hits
            % GALLERY_EVERY != 0
            and len(
                ident.color_gallery
            ) >= 3
        ):
            return

        if det.get(
            "reid"
        ) is not None:
            ident.reid_gallery.append(
                det["reid"]
            )

        if (
            USE_COLOR_HIST
            and frame is not None
        ):
            if "color" not in det:
                det["color"] = (
                    color_feature(
                        frame,
                        det["box"]
                    )
                )

        if det.get(
            "color"
        ) is not None:
            ident.color_gallery.append(
                det["color"]
            )

    def _confirm_ready(
        self,
        frame_index
    ):
        ready = [
            i
            for i in self.identities.values()
            if (
                i.visible
                and not i.confirmed
                and i.hits >= CONFIRM_FRAMES
            )
        ]

        for ident in ready:
            best = None

            best_cost = (
                RELINK_COST_THRESHOLD
            )

            for other in self.identities.values():
                if other is ident:
                    continue

                if other.visible:
                    continue

                if not other.confirmed:
                    continue

                if (
                    other.last_seen
                    >= ident.first_seen
                ):
                    continue

                if (
                    frame_index
                    - other.last_seen
                    > MAX_LOST_FRAMES
                ):
                    continue

                cost = self._tentative_cost(
                    other,
                    ident,
                    frame_index
                )

                threshold = (
                    RELAXED_RELINK_COST
                    if self._geometry_is_strong(
                        other,
                        ident.box,
                        frame_index
                    )
                    else RELINK_COST_THRESHOLD
                )

                if (
                    cost <= threshold
                    and cost < best_cost
                ):
                    best = other
                    best_cost = cost

            if best is not None:
                self._log(
                    f"frame {frame_index}: "
                    f"{ident.name} merged into "
                    f"{best.name} "
                    f"(cost {best_cost:.2f})"
                )

                self._merge(
                    ident,
                    best,
                    frame_index
                )

                self.stats[
                    "merged"
                ] += 1

            else:
                ident.confirmed = True

                ident.person_id = (
                    self.next_person_id
                )

                self.next_person_id += 1

                self.stats[
                    "confirmed"
                ] += 1

                for (
                    f,
                    box,
                    conf
                ) in ident.pending:
                    self.rows.append(
                        (
                            f,
                            ident.person_id,
                            *box,
                            conf
                        )
                    )

                ident.pending.clear()

                self._log(
                    f"frame {frame_index}: "
                    f"new person "
                    f"P{ident.person_id} "
                    f"(first seen at "
                    f"frame "
                    f"{ident.first_seen})"
                )

    def _merge(
        self,
        tentative,
        target,
        frame_index
    ):
        for (
            f,
            box,
            conf
        ) in tentative.pending:
            self.rows.append(
                (
                    f,
                    target.person_id,
                    *box,
                    conf
                )
            )

        target.raw_id = (
            tentative.raw_id
        )

        target.box = list(
            tentative.box
        )

        target.last_center = (
            box_center(
                tentative.box
            )
        )

        target.velocity = (
            tentative.velocity
        )

        target.confidence = (
            tentative.confidence
        )

        target.last_seen = (
            frame_index
        )

        target.hits += (
            tentative.hits
        )

        target.visible = True

        target.reid_gallery.extend(
            tentative.reid_gallery
        )

        target.color_gallery.extend(
            tentative.color_gallery
        )

        for raw, uid in list(
            self.raw_to_uid.items()
        ):
            if uid == tentative.uid:
                self.raw_to_uid[
                    raw
                ] = target.uid

        for raw in list(
            self.raw_last_seen.keys()
        ):
            if (
                self.raw_to_uid.get(raw)
                == target.uid
            ):
                self.raw_last_seen[
                    raw
                ] = frame_index

        del self.identities[
            tentative.uid
        ]

    def _emit_rows(
        self,
        frame_index
    ):
        for ident in self.identities.values():
            if not ident.visible:
                continue

            if ident.confirmed:
                self.rows.append(
                    (
                        frame_index,
                        ident.person_id,
                        *ident.box,
                        ident.confidence
                    )
                )

            else:
                ident.pending.append(
                    (
                        frame_index,
                        tuple(
                            ident.box
                        ),
                        ident.confidence
                    )
                )

    def _cleanup(
        self,
        frame_index
    ):
        for uid, ident in list(
            self.identities.items()
        ):
            gap = (
                frame_index
                - ident.last_seen
            )

            limit = (
                MAX_LOST_FRAMES
                if ident.confirmed
                else TENTATIVE_MAX_MISSING
            )

            if gap > limit:
                if not ident.confirmed:
                    self.stats[
                        "discarded"
                    ] += 1

                for raw, mapped in list(
                    self.raw_to_uid.items()
                ):
                    if mapped == uid:
                        del self.raw_to_uid[
                            raw
                        ]

                        self.raw_last_seen.pop(
                            raw,
                            None
                        )

                del self.identities[
                    uid
                ]

    @staticmethod
    def _log(message):
        if VERBOSE_EVENTS:
            print(message)


def draw_identities(
    frame,
    identities,
    frame_index
):
    import cv2

    for ident in identities:
        x1, y1, x2, y2 = [
            int(v)
            for v in ident.box
        ]

        if ident.confirmed:
            color = PALETTE[
                (
                    ident.person_id - 1
                )
                % len(PALETTE)
            ]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                f"ID {ident.person_id}",
                (
                    x1,
                    max(
                        24,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        else:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (150, 150, 150),
                1
            )

    cv2.putText(
        frame,
        f"frame {frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


def main():
    import cv2
    from ultralytics import YOLO

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    print(
        "Starting stable identity tracking..."
    )

    print(
        f"Video   : {VIDEO_PATH}"
    )

    print(
        f"Model   : {MODEL_PATH}"
    )

    print(
        f"Tracker : {TRACKER_PATH}"
    )

    model = YOLO(
        MODEL_PATH
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: "
            f"{VIDEO_PATH}"
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

    print(
        f"FPS: {fps} | "
        f"Resolution: "
        f"{width}x{height} | "
        f"Frames: {total_frames}"
    )

    writer = None

    if SAVE_DEBUG_VIDEO:
        writer = cv2.VideoWriter(
            DEBUG_VIDEO_PATH,
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps
            if fps and fps > 0
            else 30.0,
            (
                width,
                height
            )
        )

    manager = IdentityManager(
        width,
        height
    )

    frame_index = 0

    det_total = 0

    reid_total = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_PATH,
                classes=[
                    PERSON_CLASS
                ],
                conf=DETECT_CONF,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

            features = (
                extract_tracker_features(
                    get_tracker_object(
                        model
                    )
                )
            )

            detections = []

            if results:
                detections = (
                    prediction_to_detections(
                        results[0],
                        features,
                        width,
                        height
                    )
                )

            det_total += len(
                detections
            )

            reid_total += sum(
                d["reid"] is not None
                for d in detections
            )

            visible = manager.update(
                frame_index,
                frame,
                detections
            )

            if writer is not None:
                draw_identities(
                    frame,
                    visible,
                    frame_index
                )

                writer.write(
                    frame
                )

            if (
                frame_index
                % PRINT_INTERVAL
                == 0
            ):
                ids = sorted(
                    i.person_id
                    for i in visible
                    if i.confirmed
                )

                print(
                    f"\nFrame "
                    f"{frame_index}/"
                    f"{total_frames} | "
                    f"visible IDs: "
                    f"{ids} | "
                    f"stats: "
                    f"{manager.stats}"
                )

    except KeyboardInterrupt:
        print(
            "\nInterrupted - "
            "saving what has been "
            "tracked so far..."
        )

    finally:
        cap.release()

        if writer is not None:
            writer.release()

    manager.finalize()

    manager.write_csv(
        OUTPUT_PATH
    )

    print(
        "\nTracking completed."
    )

    print(
        f"Processed frames : "
        f"{frame_index}"
    )

    print(
        f"Persons (IDs)    : "
        f"{manager.next_person_id - 1}"
    )

    print(
        f"Stats            : "
        f"{manager.stats}"
    )

    if det_total:
        print(
            f"ReID features    : "
            f"{100.0 * reid_total / det_total:.0f}% "
            f"of detections"
        )

        if (
            reid_total == 0
            and USE_TRACKER_REID
        ):
            print(
                "  -> BoT-SORT gave "
                "no ReID features. "
                "Set with_reid: True "
                "in botsort.yaml."
            )

    print(
        "\nPerson | first frame | "
        "last frame | rows"
    )

    for pid, (
        first,
        last,
        count
    ) in sorted(
        manager.person_summary().items()
    ):
        print(
            f"{pid:>6} | "
            f"{first:>11} | "
            f"{last:>10} | "
            f"{count}"
        )

    print(
        f"\nCSV   : "
        f"{OUTPUT_PATH}"
    )

    if SAVE_DEBUG_VIDEO:
        print(
            f"Video : "
            f"{DEBUG_VIDEO_PATH}"
        )


if __name__ == "__main__":
    main()