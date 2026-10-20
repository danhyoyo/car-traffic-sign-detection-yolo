# yolo-traffic-sign-detection-via

Train YOLO11s to detect six traffic signs and integrate the detector with lane following and vehicle control in the [VIA Simulator](https://via-sim.makerviet.org/).

Developed by **[danhyoyo](https://github.com/danhyoyo)**, based on the [MakerViet VIA learning resources](https://github.com/makerviet/via-docs).

## Demo

[![Autonomous driving in the VIA Simulator](docs/images/via_demo.gif)](docs/media/car_yolo.mp4)

Animated preview · [Watch the full video (MP4)](docs/media/car_yolo.mp4)

## Project Structure

```text
yolo-traffic-sign-detection-via/
├── simulator/              # Vehicle control and traffic sign detection
│   ├── drive.py
│   ├── lane_line_detection.py
│   ├── traffic_sign_detection.py
│   ├── yolo_sign_detection.py
│   └── pid_controller.py
├── tests/                  # Controller and YOLO class mapping tests
├── notebooks/              # Tutorials 01–07, from basics to YOLO training
├── models/
│   ├── traffic_sign_classifier_lenet_v3.onnx  # Legacy classifier
│   ├── pretrained/         # Downloaded pretrained weights
│   └── yolo11/             # Exported PT, ONNX, and class mapping files
├── data/
│   ├── samples/            # Included sample image archives
│   └── datasets/           # Downloaded Roboflow datasets
├── outputs/                # Training runs, evaluations, predictions, and ZIPs
├── maps/                   # VIA maps
├── docs/
│   ├── images/             # Illustrations and demo GIF
│   ├── media/              # Full demo video
│   └── controller_tuning.md
├── requirements.txt        # Simulator dependencies
├── requirements-train.txt  # Training and ONNX export dependencies
├── .env                    # Local API key; excluded from Git
└── .env.example            # Shareable configuration template
```

The notebook creates `data/datasets/`, `outputs/`, `models/pretrained/`, and `models/yolo11/` as needed. These generated directories are excluded from Git.

## Setup

Run from the project root using Python 3.10 or later. On Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-train.txt
```

Select `.venv/Scripts/python.exe` as the notebook kernel in Jupyter or VS Code. Training dependencies pin NumPy to 2.2.6; mixing MKL and OpenMP runtimes in an existing Conda environment can cause kernel crashes. Use a project environment when setting up on another machine.

For simulator use only, install `requirements.txt`. GPU training requires a CUDA-enabled PyTorch installation; a Colab GPU runtime is another option.

## Training and ONNX Export

1. Open `notebooks/07_train_yolo11.ipynb` in Jupyter or Colab.
2. Copy your **Private API Key** from [Roboflow API Settings](https://app.roboflow.com/settings/api) into `.env`:

```dotenv
ROBOFLOW_API_KEY=your_private_key
```

3. Run the notebook cells in order. On Colab, upload `.env` to `/content`.

The notebook uses [trafficsign by Sa Tr](https://universe.roboflow.com/sa-tr/trafficsign-jryf1), workspace `sa-tr`, project `trafficsign-jryf1`, version `1`, under CC BY 4.0. Its six classes match the controller actions:

| Class | Meaning |
| --- | --- |
| `left` | Turn left |
| `no_left` | No left turn |
| `right` | Turn right |
| `no_right` | No right turn |
| `straight` | Go straight |
| `stop` | Stop |

The notebook checks class coverage and training/validation annotations. If validation data is missing, it creates an 80/10/10 train/validation/test split with seed 42, grouping augmented images by source filename without moving the originals. Different filenames may still represent nearby video frames, so review the split before reporting results.

Training settings are grouped in step 2: YOLO11s, 100 epochs, image size 640, and GPU batch size 8. Reduce the batch size to 4 or 2 if GPU memory runs out. Existing downloaded data is reused.

Step 12 exports and checks ONNX; step 13 saves the following in `models/yolo11/`:

- `bien_bao_best.pt`
- `bien_bao_best.onnx`
- `sign_mapping.json`, dataset configuration, and dependency versions

Detailed results and the ZIP archive are saved in `outputs/`. Download the archive before ending a Colab session. Keep API keys out of notebooks and shared files.

## Running the Simulator

```powershell
python -m simulator.drive
```

By default, the controller uses the included LeNet classifier in `models/`. To use your trained YOLO detector:

```powershell
$env:TRAFFIC_SIGN_MODEL = (Resolve-Path models/yolo11/bien_bao_best.pt).Path
python -m simulator.drive
```

Keep `sign_mapping.json` beside the weights. The YOLO adapter maps predictions by class name and ignores unsupported classes. The exported YOLO ONNX model detects signs in full images and cannot directly replace the legacy 32×32 LeNet classifier; the driving integration currently uses `.pt` weights for YOLO.

Open VIA, load a map from `maps/`, and connect to `ws://localhost:4567`. Stop the controller with **Ctrl+C**. Confirmed turn signs are remembered until sustained lane loss triggers a turn at a junction. Adjust thresholds in `simulator/drive.py` and `simulator/lane_line_detection.py`; see the [controller tuning notes](docs/controller_tuning.md) (Vietnamese).

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover synthetic roads, sample images, sign handling, YOLO class mapping, and WebSocket communication. The latest local run passed 22 of 24 tests; two existing controller tests disagree with the current turn settings.

Evaluate the trained detector on VIA camera frames before using it for control. Matching all six class names does not establish detection accuracy in the simulator.
