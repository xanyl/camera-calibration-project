# Camera Calibration + 2D Object Measurement (OpenCV)

This repo covers the assignment:

- **Step 1:** Camera calibration (OpenCV, smartphone camera)
- **Step 2:** Measure real-world **2D** object dimensions using perspective projection (planar homography)
- **Step 3:** Validate with an experiment using **distance > 2 meters** measured accurately (tape/laser)

---

## Requirements

- Python 3.x
- Install:

```bash
pip install opencv-python numpy
```

---

## Folder Structure (example)

```
camera-calibration-project/
├─ calibrate_camera.py
├─ measure_object.py
├─ data/
│  ├─ calib/              # chessboard calibration images
│  └─ test/               # measurement images
│     └─ test7.jpeg
└─ outputs/               # generated files
```

---

## Step 1 — Camera Calibration

1. Take **15–25** chessboard photos with your smartphone and put them in:
   `data/calib/`

2. Run calibration:

### Windows PowerShell

```powershell
python .\calibrate_camera.py --images "data/calib/*.jpg" --cols 8 --rows 4 --square_size 0.037 --out outputs/calibration.npz
```

### Linux / macOS

```bash
python calibrate_camera.py --images "data/calib/*.jpg" --cols 8 --rows 4 --square_size 0.037 --out outputs/calibration.npz
```

Notes:

- `--cols` and `--rows` are **INNER corners** of the chessboard.
- `--square_size` is in **meters** (25 mm = 0.037).
- Output: `outputs/calibration.npz`

---

## Step 2 + Step 3 — Measure Object + Validation (>2m)

Setup:

- Place an **A4 paper** and the **object** on the **same flat plane** (table/floor).
- Take one photo from **> 2 meters** away.
- Measure the distance with **tape/laser** (this is the “accurate” Step 3 distance).

### Run (Windows PowerShell)

```powershell
python .\measure_object.py --image "data/test/test7.jpeg" --calib "outputs/calibration.npz" --ref_w 0.297 --ref_h 0.210 --measured_distance_m 2.50 --distance_method tape --true_w 0.11 --true_h 0.08
```

### Run (Linux / macOS)

```bash
python measure_object.py --image "data/test/test7.jpeg" --calib "outputs/calibration.npz" --ref_w 0.297 --ref_h 0.210 --measured_distance_m 2.50 --distance_method tape --true_w 0.11 --true_h 0.08
```

Arguments:

- `--ref_w`, `--ref_h`: reference size in meters. For **A4 landscape** use `0.297 x 0.210`.
  - If A4 is portrait, use `--ref_w 0.210 --ref_h 0.297`
- `--measured_distance_m`: tape/laser measured camera-to-plane distance (**must be > 2**)
- `--true_w`, `--true_h`: object ground truth (meters) to print % error (optional)

---

## Clicking Order (IMPORTANT)

You must click 4 corners in this exact order for BOTH reference and object:

1. **Top-Left (TL)**
2. **Top-Right (TR)**
3. **Bottom-Right (BR)**
4. **Bottom-Left (BL)**

Keys while clicking:

- `u` = undo last point
- `r` = reset points
- `ESC` = cancel

---

## Outputs

After running measurement:

- `outputs/measurement_result.jpg` (shows clicked points)
- `outputs/measurement_report.json` (includes Step 3 distance evidence and accuracy)

---

## Validation Checklist (Step 3)

For submission/report:

1. Show **tape/laser distance** (e.g., 2.50 m) and confirm it is **> 2 m**
2. Show **estimated width/height** from the script
3. Measure object with a ruler and compare (optional but recommended)
