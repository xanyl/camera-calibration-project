# Camera Calibration + Object Measurement (OpenCV)

## Requirements
- Python 3
- Install:
  ```bash
  pip install opencv-python numpy
  ```

## Folder Structure
```
camera-calibration-project/
├─ calibrate_camera.py
├─ measure_object.py
├─ data/
│  ├─ calib/       # chessboard photos
│  └─ test/*.jpg    # object photo with A4 paper
└─ outputs/
```

## Step 1: Camera Calibration
1) Put 15–25 chessboard images in `data/calib/`
2) Run:

**Windows PowerShell**
```powershell
python .\calibrate_camera.py --images "data/calib/*.jpg" --cols 9 --rows 6 --square_size 0.025 --out outputs/calibration.npz --show
```

**Linux / macOS**
```bash
python calibrate_camera.py --images "data/calib/*.jpg" --cols 9 --rows 6 --square_size 0.025 --out outputs/calibration.npz --show
```

Notes:
- `square_size` is in meters (25 mm = 0.025).
- Output file: `outputs/calibration.npz`

## Step 2: Measure Object Size (using A4 paper)
1) Put an A4 sheet flat + object on the same plane.
2) Take a photo from > 2 meters and save it in `data/` (example: `test3.jpg`)
3) Run:

**Windows PowerShell**
```powershell
python .\measure_object.py --image "data/test3.jpg" --calib "outputs/calibration.npz" --out_dir "outputs"
```

**Linux / macOS**
```bash
python measure_object.py --image "data/test3.jpg" --calib "outputs/calibration.npz" --out_dir "outputs"
```

## Click Order in the Image Window
1) Click A4 corners: **TL → TR → BR → BL**
2) Click object corners: **TL → TR → BR → BL**

## Output
- Prints object width/height (meters)
- Saves: `outputs/measurement_points.jpg`
