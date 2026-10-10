"""A single-session adapter using the upstream LoRAT inference operators."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.runtime.utils.custom_yaml_loader import load_yaml
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping import (
    apply_siamfc_cropping,
    apply_siamfc_cropping_to_boxes,
    get_siamfc_cropping_params,
    reverse_siamfc_cropping_params,
)
from trackit.models import ModelManager
from trackit.models.methods.builder import create_model_build_context
from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.simple import (
    SiamFCCroppingParameterSimpleProvider,
)
from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.builder import (
    build_post_process,
)
from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.utils.bbox_mask_gen import (
    get_foreground_bounding_box,
)


class LoRATTracker:
    def __init__(self, root: Path, weights: Path, device: str):
        torch.set_num_threads(4)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        constants = load_yaml(str(root / "consts.yaml"))
        config = load_yaml(str(root / "config/LoRAT/g-378/config.yaml"), constants)
        self.common = config["common"]
        pipeline = config["run"]["runner"]["test"]["evaluator"]["pipeline"]
        manager = ModelManager(create_model_build_context(config), 0)
        manager.load_state_dict_from_file(str(weights), print_missing=False)
        self.model = manager.create_unmanaged(
            self.device, optimize_for_inference=True, load_pretrained=True
        ).eval()
        self.normalize = get_dataset_norm_stats_transform(
            self.common["normalization"], inplace=True
        )
        self.post = build_post_process(
            pipeline["post_process"], self.common, self.device
        )
        self.post.start()
        self.crop_config = pipeline["search_region_cropping"]

    def _image(self, path: Path):
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), copy=True)
        return (
            torch.from_numpy(pixels)
            .permute(2, 0, 1)
            .to(self.device, dtype=torch.float32)
        )

    @torch.inference_mode()
    def initialize(self, path: Path, xywh: list[float]):
        box = np.array(
            [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]],
            dtype=np.float64,
        )
        size = np.array(self.common["template_size"])
        params = get_siamfc_cropping_params(box, 2.0, size)
        template, self.mean, params = apply_siamfc_cropping(
            self._image(path),
            size,
            params,
            self.common["interpolation_mode"],
            self.common["interpolation_align_corners"],
        )
        template.div_(255.0)
        self.normalize(template)
        self.template = template.unsqueeze(0)
        feat_size = np.array(self.common["template_feat_size"])
        stride = size / feat_size
        foreground = get_foreground_bounding_box(box, params, tuple(stride))
        bbox_clip_to_image_boundary_(foreground, feat_size)
        self.mask = torch.zeros(
            (1, int(feat_size[1]), int(feat_size[0])),
            dtype=torch.long,
            device=self.device,
        )
        x0, y0, x1, y1 = foreground
        self.mask[:, y0:y1, x0:x1] = 1
        self.provider = SiamFCCroppingParameterSimpleProvider(
            self.crop_config["area_factor"], self.crop_config["min_object_size"]
        )
        self.provider.initialize(box)

    @torch.inference_mode()
    def track(self, path: Path) -> tuple[list[float], float]:
        image = self._image(path)
        height, width = image.shape[-2:]
        size = np.array(self.common["search_region_size"])
        search, _, params = apply_siamfc_cropping(
            image,
            size,
            self.provider.get(size),
            self.common["interpolation_mode"],
            self.common["interpolation_align_corners"],
            self.mean,
        )
        search.div_(255.0)
        self.normalize(search)
        with torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            outputs = self.model(
                z=self.template, x=search.unsqueeze(0), z_feat_mask=self.mask
            )
        outputs = self.post(outputs)
        score = float(outputs["confidence"][0].cpu())
        box = outputs["box"][0].cpu().to(torch.float64).numpy()
        box = apply_siamfc_cropping_to_boxes(
            box, reverse_siamfc_cropping_params(params)
        )
        image_size = np.array((width, height))
        bbox_clip_to_image_boundary_(box, image_size)
        if not np.isfinite(box).all() or not np.isfinite(score):
            raise RuntimeError("The tracker returned non-finite predictions.")
        self.provider.update(score, box, image_size)
        return [
            float(box[0]),
            float(box[1]),
            float(box[2] - box[0]),
            float(box[3] - box[1]),
        ], score
