"""Loopback-only browser application with one serialized GPU worker."""

import argparse
import csv
import io
import json
import math
import mimetypes
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(ROOT))
ACTIVE = {"queued", "loading", "running", "stopping"}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_request(body: dict, count: int, width: int, height: int) -> tuple:
    start = body.get("start_frame")
    end = body.get("end_frame", count - 1)
    if type(start) is not int or type(end) is not int:
        raise ApiError("Frame indices must be integers.")
    if not 0 <= start <= end < count:
        raise ApiError("Select a start and end frame inside the sequence.")
    box = body.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise ApiError("Draw one box around the target object.")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ApiError("Box coordinates must be finite numbers.")
    x, y, w, h = (float(v) for v in box)
    if w < 2 or h < 2 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ApiError("Draw a box at least two pixels wide and high inside the image.")
    return start, end, [x, y, w, h]


class Application:
    def __init__(self, args):
        self.args = args
        self.frames = sorted(args.frames.glob("*.jpg"))
        if not self.frames:
            raise RuntimeError(f"No JPEG frames found in {args.frames}")
        with Image.open(self.frames[0]) as image:
            self.width, self.height = image.size
        self.asset_version = str(self.frames[0].stat().st_mtime_ns)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.cancel: dict[str, threading.Event] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.model_state = "Not loaded"
        self.exports: dict[str, dict] = {}
        self.export_queue: queue.Queue[str] = queue.Queue()
        args.results.mkdir(parents=True, exist_ok=True)
        for path in args.results.glob("*.json"):
            try:
                job = json.loads(path.read_text())
                if job["sequence"] != args.frames.name:
                    continue
                if job["status"] in ACTIVE:
                    job["status"] = "failed"
                    job["error"] = "The server restarted before this run finished."
                self.jobs[job["id"]] = job
            except (ValueError, KeyError):
                continue
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._export_worker, daemon=True).start()

    def config(self):
        return {
            "sequence": self.args.frames.name,
            "count": len(self.frames),
            "width": self.width,
            "height": self.height,
            "fps": self.args.fps,
            "model": "LoRAT g-378",
            "model_state": self.model_state,
            "asset_version": self.asset_version,
            "video": (
                f"/media/sequence.mp4?v={self.asset_version}"
                if self.args.video.is_file()
                else None
            ),
        }

    def _save(self, job):
        path = self.args.results / f"{job['id']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, allow_nan=False))
        temporary.replace(path)

    def submit(self, body):
        start, end, box = validate_request(
            body, len(self.frames), self.width, self.height
        )
        with self.lock:
            if any(job["status"] in ACTIVE for job in self.jobs.values()):
                raise ApiError(
                    "A run is active. Stop it or wait for it to finish.", 409
                )
            run_id = uuid.uuid4().hex[:16]
            job = {
                "id": run_id,
                "sequence": self.args.frames.name,
                "model": "LoRAT g-378",
                "box_format": "xywh",
                "frame_index_base": 0,
                "fps": self.args.fps,
                "start_frame": start,
                "end_frame": end,
                "initial_box": box,
                "created": time.time(),
                "status": "queued",
                "rows": [],
                "error": None,
                "elapsed": 0.0,
            }
            self.jobs[run_id] = job
            self.cancel[run_id] = threading.Event()
            self._save(job)
            self.pending.put(run_id)
            return self.snapshot(run_id)

    def snapshot(self, run_id, since=0):
        with self.lock:
            if run_id not in self.jobs:
                raise ApiError("Run not found.", 404)
            job = self.jobs[run_id]
            data = {key: value for key, value in job.items() if key != "rows"}
            data["rows"] = list(job["rows"][since:])
            data["completed_frames"] = len(job["rows"])
            data["total_frames"] = job["end_frame"] - job["start_frame"] + 1
            return data

    def stop(self, run_id):
        with self.lock:
            if run_id not in self.jobs:
                raise ApiError("Run not found.", 404)
            if self.jobs[run_id]["status"] in ACTIVE:
                self.cancel[run_id].set()
                self.jobs[run_id]["status"] = "stopping"
            return self.snapshot(run_id)

    def _worker(self):
        tracker = None
        while True:
            run_id = self.pending.get()
            job = self.jobs[run_id]
            canceled = self.cancel[run_id]
            started = time.monotonic()
            try:
                if not canceled.is_set():
                    if tracker is None:
                        with self.lock:
                            job["status"] = "loading"
                            self.model_state = "Loading model"
                        from tracker import LoRATTracker

                        tracker = LoRATTracker(
                            ROOT, self.args.weights, self.args.device
                        )
                        self.model_state = "Ready"
                    if not canceled.is_set():
                        tracker.initialize(
                            self.frames[job["start_frame"]], job["initial_box"]
                        )
                        with self.lock:
                            job["status"] = "running"
                        for index in range(job["start_frame"], job["end_frame"] + 1):
                            if canceled.is_set():
                                break
                            seed = index == job["start_frame"]
                            box, score = (
                                (job["initial_box"], None)
                                if seed
                                else tracker.track(self.frames[index])
                            )
                            with self.lock:
                                job["rows"].append(
                                    {
                                        "frame": index,
                                        "box": box,
                                        "score": score,
                                        "seed": seed,
                                    }
                                )
                                job["elapsed"] = time.monotonic() - started
                                self._save(job)
                with self.lock:
                    job["status"] = "canceled" if canceled.is_set() else "complete"
            except Exception as error:  # noqa: BLE001 - keep the GPU worker alive after failed runs.
                traceback.print_exc()
                with self.lock:
                    job["error"] = str(error)
                    job["status"] = "failed"
                    self.model_state = "Ready" if tracker is not None else "Load failed"
            finally:
                with self.lock:
                    job["elapsed"] = time.monotonic() - started
                    self._save(job)
                self.pending.task_done()

    def export_csv(self, run_id):
        job = self.snapshot(run_id)
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame_index",
                "frame_number",
                "time_seconds",
                "x",
                "y",
                "width",
                "height",
                "score",
                "seed",
            ]
        )
        for row in job["rows"]:
            writer.writerow(
                [
                    row["frame"],
                    row["frame"] + 1,
                    row["frame"] / job["fps"],
                    *row["box"],
                    row["score"],
                    row["seed"],
                ]
            )
        return stream.getvalue().encode()

    def video_path(self, run_id):
        self.snapshot(run_id, since=1000000)
        return self.args.results.parent / "videos" / f"lorat-{run_id}.mp4"

    def video_status(self, run_id):
        with self.lock:
            path = self.video_path(run_id)
            if path.is_file():
                return {"status": "ready", "url": f"/api/jobs/{run_id}/mp4"}
            return dict(self.exports.get(run_id, {"status": "not_started"}))

    def export_video(self, run_id):
        with self.lock:
            job = self.snapshot(run_id)
            if job["status"] in ACTIVE or not job["rows"]:
                raise ApiError(
                    "Wait for tracking to finish before exporting a video.", 409
                )
            status = self.video_status(run_id)
            if status["status"] in {"ready", "rendering", "queued"}:
                return status
            self.exports[run_id] = {
                "status": "queued",
                "processed": 0,
                "total": len(job["rows"]),
            }
            self.export_queue.put(run_id)
            return dict(self.exports[run_id])

    def _export_worker(self):
        while True:
            run_id = self.export_queue.get()
            try:
                from exporter import render_video

                job = self.snapshot(run_id)
                with self.lock:
                    self.exports[run_id]["status"] = "rendering"

                def update(count, run_id=run_id):
                    with self.lock:
                        self.exports[run_id]["processed"] = count

                render_video(
                    self.frames,
                    job["rows"],
                    job["fps"],
                    self.video_path(run_id),
                    update,
                )
                with self.lock:
                    self.exports[run_id]["status"] = "ready"
            except Exception as error:  # noqa: BLE001 - preserve the export worker after encoder failures.
                traceback.print_exc()
                with self.lock:
                    self.exports[run_id] = {"status": "failed", "error": str(error)}
            finally:
                self.export_queue.task_done()


class Handler(BaseHTTPRequestHandler):
    app: Application
    protocol_version = "HTTP/1.1"

    def send_bytes(self, data: bytes, content_type: str, status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, data, status=200):
        self.send_bytes(
            json.dumps(data, allow_nan=False).encode(), "application/json", status
        )

    def send_file(self, path: Path, filename=None):
        if not path.is_file():
            raise ApiError("File not found.", 404)
        size = path.stat().st_size
        start, end = 0, size - 1
        partial = False
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if match is None or not any(match.groups()):
                raise ApiError("Invalid byte range.", 416)
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), end) if last else end
            else:
                start = max(0, size - int(last))
            if start > end or start >= size:
                raise ApiError("Byte range outside the file.", 416)
            partial = True
        self.send_response(206 if partial else 200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.send_header(
            "Cache-Control",
            "public, max-age=3600" if path.suffix in {".jpg", ".mp4"} else "no-cache",
        )
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD":
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = stream.read(min(remaining, 256 * 1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self._get()
        except ApiError as error:
            self.send_json({"error": str(error)}, error.status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _get(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/config":
            return self.send_json(self.app.config())
        if path == "/api/jobs":
            with self.app.lock:
                jobs = sorted(
                    self.app.jobs.values(), key=lambda job: job["created"], reverse=True
                )
                return self.send_json(
                    [self.app.snapshot(job["id"], len(job["rows"])) for job in jobs]
                )
        match = re.fullmatch(
            r"/api/jobs/([a-f0-9]{16})(?:/(csv|json|video|mp4))?", path
        )
        if match:
            run_id, kind = match.groups()
            if kind == "video":
                return self.send_json(self.app.video_status(run_id))
            if kind == "mp4":
                return self.send_file(
                    self.app.video_path(run_id), filename=f"lorat-{run_id}.mp4"
                )
            if kind == "csv":
                return self.send_bytes(
                    self.app.export_csv(run_id),
                    "text/csv",
                    filename=f"lorat-{run_id}.csv",
                )
            if kind == "json":
                data = json.dumps(self.app.snapshot(run_id), allow_nan=False).encode()
                return self.send_bytes(
                    data, "application/json", filename=f"lorat-{run_id}.json"
                )
            query = parse_qs(parsed.query)
            try:
                since = max(0, int(query.get("since", ["0"])[0]))
            except ValueError:
                raise ApiError("Invalid result offset.") from None
            return self.send_json(self.app.snapshot(run_id, since))
        match = re.fullmatch(r"/api/frames/(\d+)\.jpg", path)
        if match:
            index = int(match.group(1))
            if index >= len(self.app.frames):
                raise ApiError("Frame not found.", 404)
            return self.send_file(self.app.frames[index])
        if path == "/media/sequence.mp4":
            return self.send_file(self.app.args.video)
        static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
        if path in static:
            return self.send_file(STATIC / static[path])
        raise ApiError("Not found.", 404)

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise ApiError("Cross-origin requests are not accepted.", 403)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ApiError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ApiError("Request must be a JSON object.")
            if self.path == "/api/jobs":
                return self.send_json(self.app.submit(body), 202)
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/video", self.path)
            if match:
                return self.send_json(self.app.export_video(match.group(1)), 202)
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/cancel", self.path)
            if match:
                return self.send_json(self.app.stop(match.group(1)))
            raise ApiError("Not found.", 404)
        except (ValueError, json.JSONDecodeError):
            self.send_json({"error": "Invalid JSON request."}, 400)
        except ApiError as error:
            self.send_json({"error": str(error)}, error.status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        if "/api/frames/" not in args[0] and "/api/jobs" not in args[0]:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=Path,
        default=ROOT / "dataset/visdrone_small/sequences/uav0000339_00001_v",
    )
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "pretrained_models/giant-378.bin"
    )
    parser.add_argument("--results", type=Path, default=ROOT / "work/browser/runs")
    parser.add_argument(
        "--video", type=Path, default=ROOT / "work/browser/sequence.mp4"
    )
    parser.add_argument("--device", default=os.environ.get("LORAT_DEVICE", "cuda:1"))
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--source-config", type=Path, help="Saved video source settings."
    )
    args = parser.parse_args()
    if args.source_config is not None:
        source = json.loads(args.source_config.read_text())
        args.frames = Path(source["frames"]).expanduser()
        args.video = Path(source["video"]).expanduser()
        args.results = Path(source["results"]).expanduser()
        args.fps = float(source["fps"])
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be a positive number")
    Handler.app = Application(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"LoRAT browser ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
