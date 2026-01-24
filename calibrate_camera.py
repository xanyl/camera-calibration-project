import cv2
import numpy as np
import glob
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Glob path to calibration images, e.g. data/calib/*.jpg")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners per row (columns)")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners per column (rows)")
    parser.add_argument("--square_size", type=float, default=0.025, help="Square size in meters (e.g. 0.025 = 25mm)")
    parser.add_argument("--out", default="outputs/calibration.npz", help="Output calibration file")
    parser.add_argument("--show", action="store_true", help="Show detection window")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pattern_size = (args.cols, args.rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # Prepare object points like (0,0,0), (1,0,0), ... scaled by square_size
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image

    images = sorted(glob.glob(args.images))
    if len(images) == 0:
        raise FileNotFoundError(f"No images matched: {args.images}")

    img_shape = None
    good = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            good += 1

            if args.show:
                vis = cv2.drawChessboardCorners(img.copy(), pattern_size, corners2, ret)
                cv2.imshow("Corners", vis)
                cv2.waitKey(200)
        else:
            print(f"[WARN] Chessboard not found in: {fname}")

    if args.show:
        cv2.destroyAllWindows()

    if good < 10:
        raise RuntimeError(f"Not enough valid images. Found corners in {good} images. Try 15–25 images total.")

    print(f"[INFO] Using {good}/{len(images)} images for calibration")

    # Calibrate
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print("\n=== Calibration Results ===")
    print(f"RMS reprojection error: {rms:.4f} pixels")
    print("\nCamera matrix K:\n", K)
    print("\nDistortion coeffs:\n", dist.ravel())

    np.savez(args.out, K=K, dist=dist, rms=rms)
    print(f"\n[SAVED] {args.out}")

if __name__ == "__main__":
    main()
