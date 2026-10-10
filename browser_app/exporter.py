"""Encode original frames with their recorded tracking annotations."""

import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def annotate(image: Image.Image, row: dict) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, width, height = row["box"]
    color = "#ffca58" if row["seed"] else "#60f4ad"
    label = "INITIAL TARGET" if row["seed"] else f"LoRAT {row['score']:.3f}"
    box = (x, y, x + width, y + height)
    draw.rectangle(box, outline="black", width=4)
    draw.rectangle(box, outline=color, width=2)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    text_bounds = draw.textbbox((0, 0), label, font=font)
    label_width = text_bounds[2] - text_bounds[0] + 10
    label_height = text_bounds[3] - text_bounds[1] + 10
    label_x = max(0, min(x, image.width - label_width))
    label_y = max(0, y - label_height)
    draw.rectangle(
        (label_x, label_y, label_x + label_width, label_y + label_height), fill="black"
    )
    draw.text((label_x + 5, label_y + 5 - text_bounds[1]), label, font=font, fill=color)
    return image


def render_video(
    frames: Sequence[Path],
    rows: list[dict],
    fps: float,
    destination: Path,
    progress: Callable[[int], None],
):
    if not rows:
        raise ValueError("No tracked frames are available to export.")
    indices = [row["frame"] for row in rows]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Tracked frames must be consecutive and ordered.")
    with Image.open(frames[indices[0]]) as first:
        width, height = first.size
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.mp4")
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
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log
        )
        try:
            if process.stdin is None:
                raise RuntimeError("The video encoder input is unavailable.")
            for count, row in enumerate(rows, start=1):
                with Image.open(frames[row["frame"]]) as original:
                    if original.size != (width, height):
                        raise ValueError("Original frame dimensions are inconsistent.")
                    image = annotate(original, row)
                process.stdin.write(image.tobytes())
                progress(count)
            process.stdin.close()
            code = process.wait(timeout=120)
            if code:
                error_log.seek(0)
                message = error_log.read().decode(errors="replace")[-2000:]
                raise RuntimeError(f"Video encoding failed. {message}")
            partial.replace(destination)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            partial.unlink(missing_ok=True)
