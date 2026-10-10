"""Render synchronized tracking annotations from two recorded runs."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sam-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop", action="store_true")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    sam = json.loads((args.sam_results / "predictions.json").read_text())
    if sam["status"] != "complete":
        parser.error("SAM inference must finish before rendering.")
    if sam["initial_box"] != baseline["initial_box"] or sam["fps"] != baseline["fps"]:
        parser.error("Both runs must use the same initialization and frame rate.")
    baseline_rows = {row["frame"]: row for row in baseline["rows"]}
    frames = sorted(args.frames.glob("*.jpg"))
    indices = [row["frame"] for row in sam["rows"]]
    if indices != list(range(sam["start_frame"], sam["end_frame"] + 1)):
        parser.error("Predictions must cover consecutive source frames.")
    if any(frame not in baseline_rows for frame in indices):
        parser.error("A baseline prediction is missing.")
    with Image.open(frames[indices[0]]) as first:
        width, height = first.size
    crop = (0, 0, width, height)
    if args.crop:
        boxes = [row["box"] for row in sam["rows"] if row["box"] is not None]
        boxes += [baseline_rows[i]["box"] for i in indices]
        crop = (
            max(0, math.floor(min(box[0] for box in boxes)) - 100),
            max(0, math.floor(min(box[1] for box in boxes)) - 100),
            min(width, math.ceil(max(box[0] + box[2] for box in boxes)) + 100),
            min(height, math.ceil(max(box[1] + box[3] for box in boxes)) + 100),
        )
    scale = min(2.0, 1920 / (crop[2] - crop[0])) if args.crop else 1.0
    output_size = (
        math.ceil((crop[2] - crop[0]) * scale / 2) * 2,
        math.ceil((crop[3] - crop[1]) * scale / 2) * 2,
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{output_size[0]}x{output_size[1]}",
        "-framerate",
        str(sam["fps"]),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for candidate in ("DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc"):
        try:
            font = ImageFont.truetype(candidate, 20)
            break
        except OSError:
            continue
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("The video encoder input is unavailable.")
        for row in sam["rows"]:
            index = row["frame"]
            with Image.open(frames[index]) as original:
                image = original.convert("RGB")
            with Image.open(
                args.sam_results / "masks" / f"{index:07d}.png"
            ) as original:
                if original.size != image.size:
                    raise ValueError("Mask dimensions differ from source dimensions.")
                mask = np.asarray(original) > 0
            pixels = np.array(image)
            pixels[mask] = (
                0.65 * pixels[mask] + 0.35 * np.array([39, 229, 255])
            ).astype(np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            for box, color in (
                (baseline_rows[index]["box"], "#ffca58"),
                (row["box"], "#27e5ff"),
            ):
                if box is None:
                    continue
                x, y, box_width, box_height = box
                draw.rectangle(
                    (x - 1, y - 1, x + box_width + 1, y + box_height + 1),
                    outline="black",
                    width=4,
                )
                draw.rectangle(
                    (x, y, x + box_width, y + box_height), outline=color, width=2
                )
            image = image.crop(crop).resize(output_size, Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, output_size[0], 62), fill="black")
            draw.text((12, 8), "SAM 2.1 mask and box", font=font, fill="#27e5ff")
            draw.text((12, 34), "LoRAT box", font=font, fill="#ffca58")
            if row["box"] is None:
                draw.text((320, 8), "SAM target not visible", font=font, fill="white")
            label = f"Frame {index + 1}   {(index - indices[0]) / sam['fps']:.2f}s"
            draw.text((output_size[0] - 350, 22), label, font=font, fill="white")
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait(timeout=120):
            raise RuntimeError("The video encoder failed.")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()
    args.output.with_suffix(".render.json").write_text(
        json.dumps(
            {"crop_xyxy": crop, "output_size": output_size, "frames": len(indices)},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
