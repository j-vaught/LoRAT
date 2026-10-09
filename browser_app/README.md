# LoRAT browser tracker

Authored by J.C. Vaught.

The browser displays the original VisDrone sequence and initializes LoRAT g-378 from one bounding box on any selected frame. The server runs inference on comech-1176 and sends predictions to the browser as each frame finishes. Frame numbers in the interface begin at one. Frame indices in the API and exported JSON begin at zero. Boxes use original image pixels in the order x, y, width, height.

Choose a frame with the timeline, frame field, or arrow keys. Pause playback and drag a tight box around the target. Zoom enlarges the frame without changing annotation coordinates. Choose the last frame to process and press Track from selected frame. Use Play to inspect the predicted box or enable Follow the latest tracked frame to see each new result. Stop run preserves completed predictions. CSV and JSON downloads include the user-supplied initialization box and subsequent predictions. Previous runs remain available after restarting the server.

The interface uses a clean, browser-compatible video for playback and individual original JPEG frames for annotation. The selected annotation frame is decoded from its exact frame file. The playback timeline uses the preview rate of 15 frames per second. That rate matches the prior preview and determines displayed time offsets.

The model stays loaded between runs. One run uses the GPU at a time. The default device is cuda:1, which is the second GTX 1080 Ti on this server. Set LORAT_DEVICE when starting the server to choose another device. Tracking scores are the raw model outputs.

After tracking finishes, press Prepare tracking video. The server encodes an MP4 with the recorded boxes and scores. Download video appears when encoding finishes. Videos contain the processed interval and use the same 15-frame-per-second preview rate. The export adds one padding row when the original height is odd, preserving the original pixel coordinates. The imageio-ffmpeg package supplies the H.264 encoder independently of the server's system FFmpeg build. Install that package in the parent LoRAT environment before starting the server.

## Start the server

Run these commands in the LoRAT checkout. The parent Python environment must already contain the upstream LoRAT dependencies, including PyTorch and the cached DINOv2 backbone. The giant-378.bin weights belong in pretrained_models. The default frame directory is dataset/visdrone_small/sequences/uav0000339_00001_v.

```sh
bash browser_app/start.sh
```

The application binds to 127.0.0.1 on port 8766. Open a local SSH tunnel from your computer and visit the resulting address.

```sh
ssh -N -L 127.0.0.1:8766:127.0.0.1:8766 comech-1176
```

Open http://127.0.0.1:8766 in your browser. Keep the SSH tunnel running while using the application. The start script records the server process identifier and log under work/browser. Stop the server with its recorded process identifier when it is no longer needed.

## Prepare clean playback

```sh
mkdir -p work/browser
ffmpeg -framerate 15 -start_number 1 -i dataset/visdrone_small/sequences/uav0000339_00001_v/%07d.jpg -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -movflags +faststart work/browser/sequence.mp4
```

## Validate the browser adapter

The browser application uses uv for its small development environment. The GPU server continues using the parent LoRAT environment.

```sh
cd browser_app
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run python -m unittest -v
```

The adapter uses the upstream model builder, DINOv2 backbone weights, LoRA checkpoint merging, template mask, SiamFC cropping operators, search-region provider, and box postprocessor. Each new run replaces the template and resets the search-region state. It tracks every frame from the selected start through the selected end.
