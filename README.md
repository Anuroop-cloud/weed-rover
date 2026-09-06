# Weed Rover Vision & Targeting Pipeline

Agricultural weed detection and targeting system featuring a dual-mode presentation architecture for machine learning demonstration and robotic hardware execution.

---

## Architecture Overview

The system provides two completely decoupled entrypoints:

```text
weed-rover/
├── vision/
│   ├── run_yolo_demo.py     # MODE A: ML Showcase (YOLO Inference & Visualization)
│   ├── main.py              # MODE B: Robot Execution (Dot Detector -> Geometry -> State Machine)
│   ├── camera.py            # Modular OpenCV USB Camera capture abstraction
│   ├── detector.py          # Modular YOLODetector wrapper & Detection dataclass
│   ├── black_dot_detector.py# Black/Color Dot detector interface
│   ├── dual_dot_detector.py # Dual-channel HSV dot detection & temporal tracker
│   ├── robot_geometry.py    # Pixel-to-ground 2D coordinate transformer (X_cm, Y_cm)
│   └── target_controller.py # Lock-and-Execute state machine & simulated LED firing
├── models/
│   └── best.pt              # Fine-tuned YOLO weed detector weights (post-training)
├── ml/
│   ├── prepare_dataset.py   # COCO-to-YOLO conversion, 90/10 train/val split & verification
│   └── visualize_dataset.py # Visualizer for YOLO bounding-box annotations
├── weed_dataset_yolo/       # Clean YOLO dataset (images/ and labels/ for train, val, test)
│   └── data.yaml            # Dataset configuration for Ultralytics YOLO training
└── config/
    └── vision_config.py     # Centralized vision, geometric, and state machine tuning
```

---

## Operating Modes

### Mode A — ML Showcase (Visualization Only)

Dedicated machine learning demonstration that streams live webcam video, executes custom YOLO weed detection, and overlays bounding boxes, confidence badges, and FPS.

> [!NOTE]
> Mode A is visualization only. It does not import `TargetController`, does not execute robot movement, and does not fire actuators.

**Launch Command (Default - requires trained model):**
```bash
python vision/run_yolo_demo.py --model models/best.pt
```

*If `models/best.pt` has not been trained yet, the script displays a clear prompt explaining how to train the model.*

**Test with Pretrained Baseline Model:**
```bash
python vision/run_yolo_demo.py --model yolov8n.pt
```

**Available CLI Options for Mode A:**
- `--model <path>`: Path to YOLO weights (default: `models/best.pt`)
- `--camera <int>`: Camera device index (default: `1`)
- `--conf <float>`: Confidence detection threshold (default: `0.35`)
- `--width <int>`: Capture width in pixels (default: `640`)
- `--height <int>`: Capture height in pixels (default: `480`)
- `--fps <float>`: Frame rate pacing limit (default: `30.0`)
- `--list-cameras`: Scan available system video capture indices and exit
- `--debug`: Print detection coordinates to terminal

---

### Mode B — Robot Execution (Prototype Loop)

The full prototype robot execution loop for the mock farm test field:
$$\text{Webcam} \longrightarrow \text{Dot Detector} \longrightarrow \text{Temporal Tracking} \longrightarrow \text{Robot Geometry} \longrightarrow \text{Target Controller}$$

**Features:**
- Real-time dark/color dot detection with HSV thresholding & ROI filtering
- Frame-to-frame temporal persistence tracking
- Camera-to-ground geometric coordinate conversion ($X_{\text{cm}}, Y_{\text{cm}}$)
- Lock-and-Execute State Machine (`SEARCHING` $\rightarrow$ `LOCKED_ON_TARGET` $\rightarrow$ `APPROACHING` $\rightarrow$ `BLIND_EXECUTION` $\rightarrow$ `FIRING_ACTION` $\rightarrow$ `TARGET_COMPLETED`)
- Prototype LED firing signal indicator

**Launch Command:**
```bash
python vision/main.py
```

**Available CLI Options for Mode B:**
- `--camera <int>`: Camera device index (default: `1`)
- `--width <int>`: Camera capture width (default: `640`)
- `--height <int>`: Camera capture height (default: `480`)
- `--fps <float>`: Frame rate limit (default: `30.0`)
- `--min-area <float>`: Minimum dot area filter in pixels (default: `25.0`)
- `--max-area <float>`: Maximum dot area filter in pixels (default: `4000.0`)
- `--v-max <int>`: Maximum brightness for black threshold (default: `75`)
- `--no-mask`: Disable the threshold debug mask windows
- `--debug`: Print state machine transitions and target coordinates
- `--list-cameras`: List detected camera indices and exit

---

## Machine Learning Pipeline

### 1. Dataset Preparation
To convert the original COCO dataset (`weed_dataset/`) into YOLO normalized format with a 90% train / 10% validation split:
```bash
python ml/prepare_dataset.py --source-dir weed_dataset --output-dir weed_dataset_yolo --val-ratio 0.10
```

### 2. Dataset Visualization
To inspect sampled images with bounding boxes overlaid:
```bash
python ml/visualize_dataset.py --split val --num-samples 5
```

### 3. Training Custom YOLO Model
Train a YOLO model using Ultralytics with the generated `data.yaml`:
```bash
yolo detect train data=weed_dataset_yolo/data.yaml model=yolov8n.pt epochs=50 imgsz=640 project=runs/train name=weed_exp
```
After training completes, copy `runs/train/weed_exp/weights/best.pt` to `models/best.pt`.

---

## Running Automated Tests

Run the full pytest suite:
```bash
pytest
```
All unit and integration tests execute headlessly with synthetic test images.
