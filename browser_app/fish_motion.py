"""Compare joint fish association with and without Kalman motion prediction."""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def center(box):
    return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])


def overlap(a, b):
    x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = x * y
    return intersection / max(1, a[2] * a[3] + b[2] * b[3] - intersection)


class FishDetector:
    """Local dark-contrast components inside a calibrated, fixed-camera tank."""

    def __init__(self, shape, config):
        self.config = config
        self.region = np.zeros(shape, np.uint8)
        cv2.ellipse(
            self.region,
            tuple(config["tank_center"]),
            tuple(config["tank_axes"]),
            0,
            0,
            360,
            (255,),
            -1,
        )
        blocked = np.zeros(shape, np.uint8)
        for polygon in config["excluded_polygons"]:
            cv2.fillPoly(blocked, [np.array(polygon, np.int32)], (255,))
        blocked = cv2.dilate(blocked, np.ones((7, 7), np.uint8))
        self.region[blocked > 0] = 0

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (0, 0), self.config["background_sigma"])
        mask = (
            (gray.astype(float) < smooth.astype(float) - self.config["contrast"])
            & (gray < self.config["maximum_gray"])
            & (self.region > 0)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask[self.region == 0] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            if not (
                self.config["minimum_area"] <= area <= self.config["maximum_area"]
                and 8 <= width <= 140
                and 8 <= height <= 140
                and max(width, height) / min(width, height) <= 6
            ):
                continue
            component = np.zeros((height, width), np.uint8)
            cv2.drawContours(component, [contour - [x, y]], -1, (255,), -1)
            hsv = cv2.cvtColor(bgr[y : y + height, x : x + width], cv2.COLOR_BGR2HSV)
            appearance = (
                np.array(cv2.mean(hsv, mask=component)[:3]) / [179, 255, 255]
            ).tolist()
            detections.append(
                {
                    "box": [x, y, width, height],
                    "area": float(area),
                    "appearance": appearance,
                }
            )
        return sorted(detections, key=lambda row: (row["box"][0], row["box"][1]))


@dataclass
class Track:
    identity: int
    box: np.ndarray
    state: np.ndarray
    covariance: np.ndarray
    area: float
    appearance: np.ndarray
    missed: int = 0
    hits: int = 1
    measurement: list | None = None

    @classmethod
    def create(cls, identity, detection, box=None):
        box = np.array(detection["box"] if box is None else box, dtype=float)
        return cls(
            identity,
            box,
            np.r_[center(box), 0.0, 0.0],
            np.diag([9.0, 9.0, 25.0, 25.0]),
            detection["area"],
            np.array(detection["appearance"]),
            measurement=box.tolist(),
        )

    def predicted_box(self):
        return [
            float(self.state[0] - self.box[2] / 2),
            float(self.state[1] - self.box[3] / 2),
            float(self.box[2]),
            float(self.box[3]),
        ]


class JointTracker:
    def __init__(self, use_motion=True, max_missed=50):
        self.use_motion = use_motion
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 1
        self.transition = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float
        )
        acceleration = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], float)
        self.process_noise = acceleration @ acceleration.T * 0.5**2
        self.measurement_noise = np.eye(2) * 3.0**2
        self.gate = 9.21034

    def initialize(self, detections, selected_box):
        if not detections:
            raise ValueError(
                "No fish candidates were detected on the initialization frame."
            )
        selected = max(
            range(len(detections)),
            key=lambda i: overlap(detections[i]["box"], selected_box),
        )
        if overlap(detections[selected]["box"], selected_box) < 0.1:
            raise ValueError("The selected fish was not detected.")
        self.tracks = [Track.create(1, detections[selected], selected_box)]
        self.next_id = 2
        for i, detection in enumerate(detections):
            if i != selected:
                self.tracks.append(Track.create(self.next_id, detection))
                self.next_id += 1

    def association_cost(self, track, detection):
        if track.missed > self.max_missed:
            return 1e6
        delta = center(detection["box"]) - track.state[:2]
        if self.use_motion:
            innovation_covariance = track.covariance[:2, :2] + self.measurement_noise
            distance = float(delta @ np.linalg.solve(innovation_covariance, delta))
            if distance > self.gate:
                return 1e6
            motion_cost = distance / self.gate
        else:
            distance = float(np.linalg.norm(delta))
            if distance > 25:
                return 1e6
            motion_cost = (distance / 25) ** 2
        if np.linalg.norm(delta) > 25 + 5 * track.missed:
            return 1e6
        ratio = detection["area"] / max(1, track.area)
        ratio_limit = 2 if track.missed > 1 else 4
        if not 1 / ratio_limit <= ratio <= ratio_limit:
            return 1e6
        appearance_cost = float(
            np.linalg.norm(np.array(detection["appearance"]) - track.appearance)
        )
        return motion_cost + 0.15 * abs(math.log(ratio)) + 0.3 * appearance_cost

    def step(self, detections):
        for track in self.tracks:
            if self.use_motion:
                track.state = self.transition @ track.state
                track.covariance = (
                    self.transition @ track.covariance @ self.transition.T
                    + self.process_noise
                )
            track.missed += 1
            track.measurement = None
        costs = np.array(
            [
                [self.association_cost(track, detection) for detection in detections]
                for track in self.tracks
            ]
        )
        matched_detections = set()
        if self.tracks and detections:
            # Give recently observed tracks first access to candidates. Broad
            # uncertainty on a lost track must not let it steal a visible fish.
            for missed in sorted({track.missed for track in self.tracks}):
                track_indices = [
                    i for i, track in enumerate(self.tracks) if track.missed == missed
                ]
                available = [
                    j for j in range(len(detections)) if j not in matched_detections
                ]
                if not available:
                    break
                rows, columns = linear_sum_assignment(
                    costs[np.ix_(track_indices, available)]
                )
                for local_i, local_j in zip(rows, columns):
                    i, j = track_indices[local_i], available[local_j]
                    if costs[i, j] >= 1e6:
                        continue
                    self.update_track(self.tracks[i], detections[j])
                    matched_detections.add(j)
        self.tracks = [
            track for track in self.tracks if track.missed <= self.max_missed
        ]
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                self.tracks.append(Track.create(self.next_id, detection))
                self.next_id += 1

    def update_track(self, track, detection):
        observation = center(detection["box"])
        if self.use_motion:
            innovation_covariance = track.covariance[:2, :2] + self.measurement_noise
            gain = np.linalg.solve(innovation_covariance, track.covariance[:2, :]).T
            track.state += gain @ (observation - track.state[:2])
            correction = np.eye(4)
            correction[:, :2] -= gain
            track.covariance = (
                correction @ track.covariance @ correction.T
                + gain @ self.measurement_noise @ gain.T
            )
        else:
            track.state[:2] = observation
        track.box = np.array(detection["box"], float)
        track.area = 0.8 * track.area + 0.2 * detection["area"]
        track.appearance = 0.8 * track.appearance + 0.2 * np.array(
            detection["appearance"]
        )
        track.missed = 0
        track.hits += 1
        track.measurement = detection["box"]

    def snapshot(self):
        return [
            {
                "id": track.identity,
                "box": track.measurement,
                "predicted_box": track.predicted_box(),
                "state": "observed" if track.missed == 0 else "predicted",
                "missed": track.missed,
                "confirmed": track.hits >= 3 or track.identity == 1,
                "position_covariance": track.covariance[:2, :2].tolist(),
            }
            for track in self.tracks
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    baseline = json.loads(args.baseline.read_text())
    config = json.loads(args.detector_config.read_text())
    start = baseline["start_frame"]
    frames = sorted(args.frames.glob("*.jpg"))[start : start + args.count]
    if len(frames) != args.count or args.count < 1:
        parser.error("The requested frame interval is unavailable.")
    args.output.mkdir(parents=True, exist_ok=False)
    first = cv2.imread(str(frames[0]))
    detector = FishDetector(first.shape[:2], config)
    trackers = {
        "kalman": JointTracker(),
        "nearest_position": JointTracker(use_motion=False),
    }
    result = {
        "sequence": baseline["sequence"],
        "fps": baseline["fps"],
        "initial_box": baseline["initial_box"],
        "frame_index_base": 0,
        "target_id": 1,
        "detector": "Fixed-camera local dark-contrast components",
        "detector_config": config,
        "motion": {
            "state": "x,y,vx,vy in pixels and frames",
            "acceleration_sigma": 0.5,
            "measurement_sigma": 3,
            "gate_chi_squared": 9.21034,
            "max_missed": 50,
            "association": "Recency cascade with joint Hungarian assignment",
            "lost_area_ratio_limit": 2,
        },
        "status": "running",
        "rows": [],
    }
    begun = time.perf_counter()
    for offset, path in enumerate(frames):
        detections = detector.detect(cv2.imread(str(path)))
        snapshots = {}
        for name, tracker in trackers.items():
            if offset == 0:
                tracker.initialize(detections, baseline["initial_box"])
            else:
                tracker.step(detections)
            snapshots[name] = tracker.snapshot()
        result["rows"].append(
            {"frame": start + offset, "detections": detections, **snapshots}
        )
        if offset % 25 == 0:
            target = next(
                (track for track in snapshots["kalman"] if track["id"] == 1), None
            )
            print(
                f"Frame {start + offset + 1}. {len(detections)} candidates. Target {target}.",
                flush=True,
            )
    result["elapsed"] = time.perf_counter() - begun
    result["status"] = "complete"
    (args.output / "predictions.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Processed {len(frames)} frames in {result['elapsed']:.2f} seconds.")


if __name__ == "__main__":
    main()
