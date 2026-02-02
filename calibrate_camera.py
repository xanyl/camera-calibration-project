"""
Step 1: Camera Calibration (OpenCV)
- Detect chessboard corners from smartphone images
- Calibrate intrinsics K and distortion dist
- Save outputs to NPZ + JSON report (reprojection errors, settings)

Example (PowerShell):
  python .\calibrate_camera_fixed.py --images "data/calib/*.jpg" --cols 9 --rows 6 --square_size 0.025 --out outputs/calibration.npz --report outputs/calibration_report.json --show
"""

import cv2
import numpy as np
import glob
import argparse
import os
import json


def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Per-view mean reprojection error (pixels) and overall mean."""
    per_view = []
    total_err = 0.0
    total_pts = 0

    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        obs = imgpoints[i].reshape(-1, 2)

        err = np.linalg.norm(obs - proj, axis=1)  # per-point error
        per_view.append(float(np.mean(err)))

        total_err += float(np.sum(err))
        total_pts += int(err.size)

    overall_mean = float(total_err / max(total_pts, 1))
    return per_view, overall_mean


def main():
    parser = argparse.ArgumentParser(description="Step 1: Camera Calibration (OpenCV)")
    parser.add_argument("--images", default="data/calib/*.jpg",
                        help="Glob pattern to chessboard images, e.g. data/calib/*.jpg")
    parser.add_argument("--rows", type=int, default=4, help="Number of INNER corners (rows)")
    parser.add_argument("--cols", type=int, default=8, help="Number of INNER corners (cols)")
    parser.add_argument("--square_size", type=float, default=0.037, help="Square size in meters (e.g. 0.025=25mm)")
    parser.add_argument("--out", default="outputs/calibration.npz", help="Output NPZ file")
    parser.add_argument("--report", default="outputs/calibration_report.json", help="Output JSON report")
    parser.add_argument("--show", action="store_true", help="Show detected corners while processing")
    parser.add_argument("--min_images", type=int, default=10, help="Minimum successful images required")
    args = parser.parse_args()

    images = sorted(glob.glob(args.images))
    if not images:
        raise FileNotFoundError(f"No images found: {args.images}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    pattern_size = (args.cols, args.rows)  # (cols, rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # Prepare object points (Z=0 plane), scaled by square size
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= float(args.square_size)

    objpoints = []  # 3D points in world
    imgpoints = []  # 2D points in image
    used_files = []
    img_size = None

    print(f"[INFO] Found {len(images)} images. Detecting corners...")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARN] Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])  # (w, h)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f" - Failed: {fname}")
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        used_files.append(fname)
        print(f" - Detected: {fname}")

        if args.show:
            vis = cv2.drawChessboardCorners(img.copy(), pattern_size, corners2, ret)
            cv2.imshow("Corners", vis)
            cv2.waitKey(150)

    if args.show:
        cv2.destroyAllWindows()

    if img_size is None or len(objpoints) == 0:
        raise RuntimeError("No valid detections. Check chessboard size (rows/cols) and image quality.")

    if len(objpoints) < args.min_images:
        print(f"[WARN] Only {len(objpoints)} valid images. Calibration may be poor (recommended 15–25).")

    print("\n[INFO] Calibrating camera...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    per_view_err, mean_err = compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    # Recommended camera matrix for undistortion (alpha=0 crops black borders)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, alpha=0.0, centerPrincipalPoint=True)

    print("\n=== Calibration Results ===")
    print(f"OpenCV RMS reprojection error: {rms:.4f} px")
    print(f"Mean reprojection error:      {mean_err:.4f} px")
    print("Camera Matrix (K):\n", K)
    print("Distortion (dist):\n", dist.ravel())

    np.savez(
        args.out,
        K=K,
        dist=dist,
        rms=float(rms),
        mean_reproj_err=float(mean_err),
        per_view_reproj_err=np.array(per_view_err, dtype=np.float32),
        img_size=np.array(img_size, dtype=np.int32),
        pattern_size=np.array([args.cols, args.rows], dtype=np.int32),
        square_size=float(args.square_size),
        newK=newK,
        roi=np.array(roi, dtype=np.int32),
        used_files=np.array(used_files, dtype=object),
    )
    print(f"\n[SAVED] {args.out}")

    report = {
        "images_glob": args.images,
        "num_images_total": int(len(images)),
        "num_images_used": int(len(objpoints)),
        "pattern_inner_corners": {"cols": int(args.cols), "rows": int(args.rows)},
        "square_size_m": float(args.square_size),
        "image_size_px": {"width": int(img_size[0]), "height": int(img_size[1])},
        "opencv_rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": float(mean_err),
        "per_view_mean_reprojection_error_px": [float(x) for x in per_view_err],
        "camera_matrix_K": K.tolist(),
        "distortion_coeffs": dist.ravel().tolist(),
        "new_camera_matrix_newK": newK.tolist(),
        "roi": [int(x) for x in roi],
        "used_files": used_files,
        "notes": "Take 15–25 smartphone images with varied angles/positions for best results.",
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] {args.report}")


if __name__ == "__main__":
    main()
