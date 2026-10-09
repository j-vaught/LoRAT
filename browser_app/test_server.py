"""Boundary tests for exact-frame initialization and original-pixel boxes."""

import unittest

from server import ApiError, validate_request


class AnnotationValidationTests(unittest.TestCase):
    def test_arbitrary_start_frame_and_xywh_are_preserved(self):
        self.assertEqual(
            validate_request(
                {"start_frame": 123, "end_frame": 200, "box": [301.5, 200, 12.5, 19]},
                275,
                1904,
                1072,
            ),
            (123, 200, [301.5, 200.0, 12.5, 19.0]),
        )

    def test_last_frame_can_be_initialized(self):
        self.assertEqual(
            validate_request(
                {"start_frame": 274, "box": [1900, 1068, 4, 4]}, 275, 1904, 1072
            )[0:2],
            (274, 274),
        )

    def test_invalid_indices_and_boxes_are_rejected(self):
        cases = [
            {"start_frame": -1, "box": [0, 0, 20, 20]},
            {"start_frame": 275, "box": [0, 0, 20, 20]},
            {"start_frame": True, "box": [0, 0, 20, 20]},
            {"start_frame": 10, "end_frame": 9, "box": [0, 0, 20, 20]},
            {"start_frame": 10, "box": [1899, 0, 20, 20]},
            {"start_frame": 10, "box": [0, 0, 1, 20]},
            {"start_frame": 10, "box": [0, float("nan"), 20, 20]},
            {"start_frame": 10, "box": [0, 0, 20]},
        ]
        for body in cases:
            with self.subTest(body=body), self.assertRaises(ApiError):
                validate_request(body, 275, 1904, 1072)


if __name__ == "__main__":
    unittest.main()
