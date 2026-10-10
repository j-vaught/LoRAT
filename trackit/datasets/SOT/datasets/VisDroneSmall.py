import os

import numpy as np

from trackit.datasets.common.seed import BaseSeed


class VisDroneSmall_Seed(BaseSeed):
    """One VisDrone MOT validation identity exposed as an SOT sequence."""

    def __init__(self, root_path: str | None = None):
        if root_path is None:
            root_path = self.get_path_from_config("VisDroneSmall_PATH")
        super().__init__("VisDroneSmall", root_path)

    def construct(self, constructor):
        sequence_name = "uav0000339_00001_v"
        sequence_root = os.path.join(self.root_path, "sequences", sequence_name)
        annotation_path = os.path.join(
            self.root_path, "annotations", f"{sequence_name}_track_42.txt"
        )
        boxes = np.loadtxt(annotation_path, delimiter=",", dtype=np.float32, ndmin=2)

        constructor.set_total_number_of_sequences(1)
        constructor.set_bounding_box_format("XYWH")

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(f"{sequence_name}_id42")
            sequence_constructor.set_fps(30)
            for frame_index, box in enumerate(boxes, start=1):
                frame_path = os.path.join(sequence_root, f"{frame_index:07d}.jpg")
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(frame_path)
                    frame_constructor.set_bounding_box(box.tolist(), validity=True)
