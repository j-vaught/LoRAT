# LoRAT browser tracker

Authored by J.C. Vaught.

The browser displays a selected video sequence and initializes LoRAT g-378 from one bounding box on any selected frame. The server runs inference on comech-1176 and sends predictions to the browser as each frame finishes. Frame numbers in the interface begin at one. Frame indices in the API and exported JSON begin at zero. Boxes use original image pixels in the order x, y, width, height.

Choose a frame with the timeline, frame field, or arrow keys. Pause playback and drag a tight box around the target. Zoom enlarges the frame without changing annotation coordinates. Choose the last frame to process and press Track from selected frame. Use Play to inspect the predicted box or enable Follow the latest tracked frame to see each new result. Stop run preserves completed predictions. CSV and JSON downloads include the user-supplied initialization box and subsequent predictions. Previous runs remain available after restarting the server.

The interface uses a clean, browser-compatible video for playback and individual JPEG frames for annotation. The selected annotation frame is decoded from its exact frame file. Playback, time offsets, and exports use the configured frame rate. The VisDrone preview uses 15 frames per second. MSK-002 uses its source rate of 25 frames per second.

The model stays loaded between runs. One run uses the GPU at a time. The default device is cuda:1, which is the second GTX 1080 Ti on this server. Set LORAT_DEVICE when starting the server to choose another device. Tracking scores are the raw model outputs.

After tracking finishes, press Prepare tracking video. The server encodes an MP4 with the recorded boxes and scores. Download video appears when encoding finishes. Videos contain the processed interval and use the configured frame rate. The export adds one padding row when the original height is odd, preserving the original pixel coordinates. The imageio-ffmpeg package supplies the H.264 encoder independently of the server's system FFmpeg build. Install that package in the parent LoRAT environment before starting the server.

## Select a video source

The start script reads work/browser/source.json when that file exists. Save the frames directory, playback video path, results directory, and frame rate under the keys frames, video, results, and fps. Paths may be absolute or relative to the checkout. The frames directory name identifies the sequence. Each video should have its own results directory so saved runs and exports remain associated with that source. Restart the server after changing these settings. Without saved settings, the server uses its VisDrone defaults. The server also accepts --source-config to read a settings file directly.

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

## Compare SAM 2.1 on a saved run

The comparison script uses the saved initialization box and original JPEG frames. It propagates one mask without additional prompts, preserves empty masks as missing targets, and records mask-derived boxes in original pixels. The standard browser tracker continues using LoRAT. The comparison runs separately.

Prepare the official SAM 2 repository and its lightweight dependencies in the work directory. The October 9 comparison used revision 2b90b9f5ceec907a1c18123530e92e794ad901a4. It used the existing PyTorch 2.2.2 CUDA environment, float32 inference on cuda:2, and the Hiera Large checkpoint. SAM's recommended environment uses PyTorch 2.5.1 or newer. The optional connected-component extension and hole filling are disabled for this comparison. Video frames and prediction memory stay on the CPU between inference steps.

```sh
git clone https://github.com/facebookresearch/sam2.git work/browser/sam2-vendor
git -C work/browser/sam2-vendor checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
uv pip install --python .venv/bin/python --target work/browser/sam2-deps hydra-core iopath 'tqdm>=4.66.1'
curl -fL -o work/browser/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
.venv/bin/python browser_app/compare_sam2.py --sam-repo work/browser/sam2-vendor --sam-deps work/browser/sam2-deps --checkpoint work/browser/sam2.1_hiera_large.pt --frames work/browser/media/MSK-002 --baseline work/browser/MSK-002/runs/72b8e7c71e654499.json --output work/browser/MSK-002/sam2-first250 --count 250 --device cuda:2
.venv/bin/python browser_app/render_comparison.py --frames work/browser/media/MSK-002 --baseline work/browser/MSK-002/runs/72b8e7c71e654499.json --sam-results work/browser/MSK-002/sam2-first250 --output work/browser/MSK-002/sam2-comparison.mp4 --crop
```

Use a new output directory for each inference run. The cropped comparison video contains synchronized LoRAT boxes in yellow and SAM masks and boxes in cyan. Its render JSON records the crop, output dimensions, and frame count. Review identity continuity against the source images. Raw tracker scores and segmentation masks do not establish identity accuracy.

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
