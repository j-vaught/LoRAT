"""Render measured, predicted, and lost states from joint fish tracking."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def draw_box(draw, box, color, dashed=False, label=None, font=None):
    if box is None:
        return
    x, y, width, height = box
    if not dashed:
        draw.rectangle(
            (x - 1, y - 1, x + width + 1, y + height + 1), outline="black", width=4
        )
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
    else:
        for left in range(round(x), round(x + width), 10):
            for top in (y, y + height):
                draw.line(
                    (left, top, min(left + 5, x + width), top), fill=color, width=2
                )
        for top in range(round(y), round(y + height), 10):
            for left in (x, x + width):
                draw.line(
                    (left, top, left, min(top + 5, y + height)), fill=color, width=2
                )
    if label:
        draw.text(
            (x, y - 18),
            label,
            font=font,
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--motion-results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sam-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-tracks", action="store_true")
    parser.add_argument("--position-control", action="store_true")
    parser.add_argument("--crop-box", nargs=4, type=int)
    args = parser.parse_args()
    motion = json.loads(args.motion_results.read_text())
    baseline = json.loads(args.baseline.read_text())
    sam = json.loads(args.sam_results.read_text())
    if motion["status"] != "complete":
        parser.error("Tracking must be complete before rendering.")
    if not (
        motion["fps"] == sam["fps"] == baseline["fps"]
        and motion["initial_box"] == sam["initial_box"] == baseline["initial_box"]
    ):
        parser.error(
            "The compared runs must use the same frame rate and initialization."
        )
    frames = sorted(args.frames.glob("*.jpg"))
    sam_rows = {row["frame"]: row for row in sam["rows"]}
    baseline_rows = {row["frame"]: row for row in baseline["rows"]}
    with Image.open(frames[motion["rows"][0]["frame"]]) as first:
        width, height = first.size
    crop = tuple(args.crop_box) if args.crop_box else (0, 0, width, height)
    if not (0 <= crop[0] < crop[2] <= width and 0 <= crop[1] < crop[3] <= height):
        parser.error("The crop is outside the original frame.")
    scale = min(2, 1920 / (crop[2] - crop[0])) if args.crop_box else 1
    size = (
        math.ceil((crop[2] - crop[0]) * scale / 2) * 2,
        math.ceil((crop[3] - crop[1]) * scale / 2) * 2,
    )
    font = ImageFont.load_default()
    for candidate in ("DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc"):
        try:
            font = ImageFont.truetype(candidate, 18)
            break
        except OSError:
            continue
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
        f"{size[0]}x{size[1]}",
        "-framerate",
        str(motion["fps"]),
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
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("Video encoder input is unavailable.")
        for row in motion["rows"]:
            index = row["frame"]
            with Image.open(frames[index]) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            if args.all_tracks:
                for track in row["kalman"]:
                    if (
                        track["id"] != 1
                        and track["confirmed"]
                        and track["box"] is not None
                    ):
                        draw_box(
                            draw,
                            track["box"],
                            "#dddddd",
                            label=str(track["id"]),
                            font=font,
                        )
            elif args.position_control:
                control = next(
                    (track for track in row["nearest_position"] if track["id"] == 1),
                    None,
                )
                if control:
                    draw_box(
                        draw,
                        control["box"] or control["predicted_box"],
                        "#f6a1ff",
                        dashed=control["box"] is None,
                    )
            else:
                draw_box(draw, baseline_rows[index]["box"], "#ffca58")
                draw_box(draw, sam_rows[index]["box"], "#27e5ff")
            target = next((track for track in row["kalman"] if track["id"] == 1), None)
            status = "LOST"
            if target:
                if target["box"] is not None:
                    draw_box(draw, target["box"], "#60f4ad")
                    status = "OBSERVED"
                else:
                    draw_box(draw, target["predicted_box"], "#ff9055", dashed=True)
                    status = f"PREDICTION ONLY   {target['missed'] / motion['fps']:.2f}s since measurement"
            image = image.crop(crop).resize(size, Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, size[0], 76), fill="black")
            draw.text((12, 6), "Kalman + joint assignment", font=font, fill="#60f4ad")
            draw.text(
                (12, 29), "Orange dashed box is a prediction", font=font, fill="#ff9055"
            )
            if args.all_tracks:
                draw.text(
                    (12, 52),
                    "Gray boxes show other observed candidate tracks",
                    font=font,
                    fill="#dddddd",
                )
            elif args.position_control:
                draw.text(
                    (12, 52),
                    "Position-only joint matching in magenta",
                    font=font,
                    fill="#f6a1ff",
                )
            else:
                draw.text(
                    (12, 52), "SAM 2.1 cyan   LoRAT yellow", font=font, fill="#27e5ff"
                )
            draw.text((size[0] - 660, 6), status, font=font, fill="white")
            draw.text(
                (size[0] - 330, 52),
                f"Frame {index + 1}   {index / motion['fps']:.2f}s",
                font=font,
                fill="white",
            )
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait(timeout=120):
            raise RuntimeError("Video encoding failed.")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()


if __name__ == "__main__":
    main()
