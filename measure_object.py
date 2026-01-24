import cv2
import numpy as np
import argparse
import os

CLICK_POINTS = []

def mouse_callback(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((x, y))
        print(f"Clicked: {(x, y)}")

def get_points(window_name, image, n_points, help_text):
    global CLICK_POINTS
    CLICK_POINTS = []
    img_vis = image.copy()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + help_text)
    print(f"Please click {n_points} points...")

    while True:
        disp = img_vis.copy()
        for i, pt in enumerate(CLICK_POINTS):
            cv2.circle(disp, pt, 6, (0, 255, 0), -1)
            cv2.putText(disp, str(i+1), (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if len(CLICK_POINTS) == n_points:
            break
        if key == 27:  # ESC
            raise RuntimeError("User cancelled.")

    cv2.destroyWindow(window_name)
    return np.array(CLICK_POINTS, dtype=np.float32)

def dist2(a, b):
    return float(np.linalg.norm(a - b))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--calib", default="outputs/calibration.npz", help="Calibration NPZ file")
    parser.add_argument("--out_dir", default="outputs", help="Output folder")
    parser.add_argument("--a4_w", type=float, default=0.210, help="A4 width in meters")
    parser.add_argument("--a4_h", type=float, default=0.297, help="A4 height in meters")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.calib)
    K = data["K"]
    dist = data["dist"]

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    h, w = img.shape[:2]
    undist = cv2.undistort(img, K, dist)

    # --- Click A4 corners in order: TL, TR, BR, BL ---
    a4_img_pts = get_points(
        "Undistorted Image",
        undist,
        4,
        "Click the 4 CORNERS of the A4 sheet in this order:\n"
        "1) Top-Left, 2) Top-Right, 3) Bottom-Right, 4) Bottom-Left"
    )

    # A4 world points (meters), Z=0 plane
    a4_obj_pts = np.array([
        [0, 0, 0],
        [args.a4_w, 0, 0],
        [args.a4_w, args.a4_h, 0],
        [0, args.a4_h, 0]
    ], dtype=np.float32)

    # Pose estimation (distance check)
    ok, rvec, tvec = cv2.solvePnP(a4_obj_pts, a4_img_pts, K, np.zeros((5,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        dist_cam = float(np.linalg.norm(tvec))
        print(f"\n[Pose] Estimated camera->A4 distance ≈ {dist_cam:.3f} meters")
    else:
        print("\n[Pose] solvePnP failed (still ok, measurement can continue).")

    # Homography for plane mapping (world 2D <-> image)
    a4_world_2d = np.array([
        [0, 0],
        [args.a4_w, 0],
        [args.a4_w, args.a4_h],
        [0, args.a4_h]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(a4_img_pts, a4_world_2d)  # image -> world
    if H is None:
        raise RuntimeError("Homography failed. Re-click points carefully.")

    # --- Click object corners in order: TL, TR, BR, BL ---
    obj_img_pts = get_points(
        "Undistorted Image",
        undist,
        4,
        "Now click the 4 CORNERS of the OBJECT in this order:\n"
        "1) Top-Left, 2) Top-Right, 3) Bottom-Right, 4) Bottom-Left"
    )

    # Convert object points to world plane coordinates
    obj_img_h = np.hstack([obj_img_pts, np.ones((4, 1), dtype=np.float32)])
    world_h = (H @ obj_img_h.T).T
    obj_world_pts = world_h[:, :2] / world_h[:, 2:3]

    TL, TR, BR, BL = obj_world_pts

    width_top = dist2(TL, TR)
    width_bottom = dist2(BL, BR)
    height_left = dist2(TL, BL)
    height_right = dist2(TR, BR)

    width = (width_top + width_bottom) / 2.0
    height = (height_left + height_right) / 2.0

    print("\n=== Measured Object Size ===")
    print(f"Width  ≈ {width:.4f} meters")
    print(f"Height ≈ {height:.4f} meters")

    # Save visualization
    vis = undist.copy()
    for p in a4_img_pts.astype(int):
        cv2.circle(vis, tuple(p), 7, (255, 0, 0), -1)
    for p in obj_img_pts.astype(int):
        cv2.circle(vis, tuple(p), 7, (0, 255, 0), -1)

    out_path = os.path.join(args.out_dir, "measurement_points.jpg")
    cv2.imwrite(out_path, vis)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    main()
