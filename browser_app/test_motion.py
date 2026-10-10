"""Check identity preservation, rejection, and expiry in joint association."""

import unittest

import numpy as np

try:
    from fish_motion import JointTracker
except ImportError:
    JointTracker = None


def detection(x, y=0):
    return {"box": [x, y, 10, 20], "area": 150.0, "appearance": [0.4, 0.3, 0.2]}


class MotionAssociationTests(unittest.TestCase):
    def setUp(self):
        if JointTracker is None:
            self.skipTest("Install the motion optional dependencies.")
        self.tracker = JointTracker()
        self.tracker.initialize([detection(0)], [0, 0, 10, 20])

    def establish_motion(self):
        for frame in range(1, 13):
            self.tracker.step([detection(2 * frame)])

    def target(self):
        return next(track for track in self.tracker.snapshot() if track["id"] == 1)

    def test_short_dropout_is_prediction_then_same_identity_recovers(self):
        self.establish_motion()
        observed = self.target()
        self.tracker.step([])
        self.tracker.step([])
        predicted = self.target()
        self.assertIsNone(predicted["box"])
        self.assertEqual(predicted["state"], "predicted")
        self.assertGreater(predicted["predicted_box"][0], observed["box"][0] + 2)
        self.assertGreater(
            predicted["position_covariance"][0][0],
            observed["position_covariance"][0][0],
        )
        self.tracker.step([detection(30)])
        self.assertEqual(self.target()["box"], [30, 0, 10, 20])
        self.assertEqual(self.target()["state"], "observed")

    def test_large_jump_creates_another_track_instead_of_changing_target(self):
        self.establish_motion()
        self.tracker.step([detection(200, -200)])
        self.assertIsNone(self.target()["box"])
        other = [track for track in self.tracker.snapshot() if track["id"] != 1]
        self.assertEqual(len(other), 1)
        self.assertEqual(other[0]["box"], [200, -200, 10, 20])

    def test_uncertain_lost_track_cannot_steal_visible_fish(self):
        self.tracker.initialize([detection(0), detection(20)], [0, 0, 10, 20])
        lost = self.tracker.tracks[0]
        lost.missed = 8
        lost.state = np.array([29.0, 10.0, 0.0, 0.0])
        lost.covariance = np.eye(4) * 1000
        self.tracker.step([detection(24)])
        snapshot = self.tracker.snapshot()
        self.assertIsNone(self.target()["box"])
        visible = next(track for track in snapshot if track["id"] == 2)
        self.assertEqual(visible["box"], [24, 0, 10, 20])
        self.assertEqual(sum(track["box"] is not None for track in snapshot), 1)

    def test_expired_identity_is_not_reused(self):
        self.tracker.max_missed = 2
        self.tracker.step([])
        self.tracker.step([])
        self.tracker.step([detection(0)])
        snapshot = self.tracker.snapshot()
        self.assertNotIn(1, [track["id"] for track in snapshot])
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]["id"], 2)


if __name__ == "__main__":
    unittest.main()
