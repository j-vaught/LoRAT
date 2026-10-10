"""Test a SAM 2.1 video predictor using a saved browser run's exact seed box."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


def save_json(path: Path, value: dict):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-repo", type=Path, required=True)
    parser.add_argument("--sam-deps", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("The frame count must be positive.")
    if args.sam_deps:
        sys.path.insert(0, str(args.sam_deps.resolve()))
    sys.path.insert(0, str(args.sam_repo.resolve()))
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    torch.set_num_threads(4)
    source = json.loads(args.baseline.read_text())
    start = source["start_frame"]
    frames = sorted(args.frames.glob("*.jpg"))
    selected = frames[start : min(start + args.count, source["end_frame"] + 1)]
    if len(selected) != args.count:
        parser.error("The requested interval exceeds the baseline or source frames.")
    x, y, width, height = source["initial_box"]
    prompt_box = np.array([x, y, x + width, y + height], dtype=np.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    masks_dir = args.output / "masks"
    masks_dir.mkdir()
    result_path = args.output / "predictions.json"
    revision = subprocess.check_output(
        ["git", "-C", str(args.sam_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "model": "SAM 2.1 Hiera Large",
        "sequence": source["sequence"],
        "fps": source["fps"],
        "frame_index_base": 0,
        "box_format": "xywh",
        "start_frame": start,
        "end_frame": start + len(selected) - 1,
        "initial_box": source["initial_box"],
        "baseline_id": source["id"],
        "sam_revision": revision,
        "checkpoint": args.checkpoint.name,
        "config": args.config,
        "torch_version": torch.__version__,
        "device": args.device,
        "precision": "float32",
        "correction_prompts": 0,
        "fill_hole_area": 0,
        "status": "running",
        "rows": [],
    }
    save_json(result_path, result)
    begun = time.perf_counter()
    try:
        with (
            torch.inference_mode(),
            tempfile.TemporaryDirectory(
                prefix="sam-input-", dir=args.output
            ) as frame_directory,
        ):
            for local_index, path in enumerate(selected):
                (Path(frame_directory) / f"{local_index:07d}.jpg").symlink_to(
                    path.resolve()
                )
            predictor = build_sam2_video_predictor(
                args.config, str(args.checkpoint.resolve()), device=args.device
            )
            # The optional connected-component CUDA extension is not installed.
            predictor.fill_hole_area = 0
            state = predictor.init_state(
                frame_directory,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1, box=prompt_box
            )
            tracking_started = time.perf_counter()
            for local_index, object_ids, logits in predictor.propagate_in_video(state):
                if list(object_ids) != [1]:
                    raise RuntimeError("Unexpected target identifiers.")
                mask = (logits[0, 0] > 0).cpu().numpy()
                ys, xs = np.nonzero(mask)
                box = None
                if xs.size:
                    box = [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ]
                frame = start + int(local_index)
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    masks_dir / f"{frame:07d}.png"
                )
                result["rows"].append(
                    {
                        "frame": frame,
                        "box": box,
                        "mask_pixels": int(xs.size),
                        "seed": local_index == 0,
                    }
                )
                result["elapsed"] = time.perf_counter() - begun
                result["tracking_elapsed"] = time.perf_counter() - tracking_started
                save_json(result_path, result)
                if local_index % 10 == 0:
                    print(
                        f"Processed {local_index + 1}/{len(selected)}. Box {box}.",
                        flush=True,
                    )
            if len(result["rows"]) != len(selected):
                raise RuntimeError("The predictor did not return every selected frame.")
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        result["elapsed"] = time.perf_counter() - begun
        save_json(result_path, result)
    print(
        f"Finished {len(result['rows'])} frames in {result['elapsed']:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
