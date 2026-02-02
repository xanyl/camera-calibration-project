import cv2
import numpy as np
import argparse
import os

class SmartPointSelector:
    def __init__(self, image, win_name="Measurement Tool"):
        self.original_image = image
        self.display_image = image.copy()
        self.points = []
        self.win_name = win_name
        self.mouse_pos = (0, 0)
        self.done = False
        
        # Settings
        self.zoom_factor = 4
        self.zoom_size = 200 # Size of the overlay box in pixels

    def mouse_callback(self, event, x, y, flags, param):
        # 1. Track mouse for Zoom/Crosshair
        self.mouse_pos = (x, y)

        # 2. Left Click: Add Point
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y]) # Use list for mutability (nudging)
                print(f" - Point {len(self.points)} added at {x},{y}")

        # 3. Right Click: Remove Last Point
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f" - Point removed: {removed}")

    def draw_hud(self, img):
        """Draws Crosshairs and Magnifier Overlay"""
        h, w = img.shape[:2]
        mx, my = self.mouse_pos
        
        # --- A. Full Screen Crosshairs (Pointing Aid) ---
        # Draw faint green lines across the whole screen intersecting at mouse
        cv2.line(img, (mx, 0), (mx, h), (0, 255, 0), 1)
        cv2.line(img, (0, my), (w, my), (0, 255, 0), 1)

        # --- B. Magnifier Overlay (Zoom Aid) ---
        # 1. Define the region of interest (ROI) around mouse
        patch_radius = self.zoom_size // (2 * self.zoom_factor)
        
        x1 = max(0, mx - patch_radius)
        y1 = max(0, my - patch_radius)
        x2 = min(w, mx + patch_radius)
        y2 = min(h, my + patch_radius)
        
        patch = self.original_image[y1:y2, x1:x2]
        
        if patch.size > 0:
            # 2. Zoom in (Resize)
            zoomed = cv2.resize(patch, (self.zoom_size, self.zoom_size), interpolation=cv2.INTER_NEAREST)
            
            # 3. Draw a cross on the zoomed view (Target Reticle)
            center = self.zoom_size // 2
            cv2.line(zoomed, (center, 0), (center, self.zoom_size), (0, 0, 255), 1)
            cv2.line(zoomed, (0, center), (self.zoom_size, center), (0, 0, 255), 1)
            
            # 4. Add a border
            cv2.rectangle(zoomed, (0,0), (self.zoom_size-1, self.zoom_size-1), (255,255,255), 2)
            
            # 5. Overlay on Main Image (Picture-in-Picture)
            # Default placement: Top-Right
            overlay_x = w - self.zoom_size - 20
            overlay_y = 20
            
            # If mouse is in Top-Right, move overlay to Top-Left
            if mx > w - self.zoom_size - 50 and my < self.zoom_size + 50:
                overlay_x = 20
            
            # Place the overlay
            img[overlay_y:overlay_y+self.zoom_size, overlay_x:overlay_x+self.zoom_size] = zoomed
            
            # Label
            cv2.putText(img, f"ZOOM x{self.zoom_factor}", (overlay_x, overlay_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def select_points(self, instructions):
        self.points = []
        self.done = False
        
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        print(f"\n=== {instructions} ===")
        print("  [Mouse] Move to aim")
        print("  [L-Click] Add Point")
        print("  [R-Click] Undo")
        print("  [Arrows] Nudge last point (Pixel Precision)")
        print("  [Enter] Confirm")

        while not self.done:
            display = self.original_image.copy()
            
            # 1. Draw Instructions
            cv2.putText(display, instructions, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Points: {len(self.points)}/4", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 2. Draw Selected Points and Lines
            for i, pt in enumerate(self.points):
                cv2.circle(display, tuple(pt), 4, (0, 0, 255), -1) # Red dots
                cv2.putText(display, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw box lines
                if i > 0:
                    cv2.line(display, tuple(self.points[i-1]), tuple(pt), (255, 0, 0), 1)
            if len(self.points) == 4:
                cv2.line(display, tuple(self.points[3]), tuple(self.points[0]), (255, 0, 0), 1)

            # 3. Draw HUD (Crosshair + Zoom)
            self.draw_hud(display)

            cv2.imshow(self.win_name, display)
            
            # 4. Keyboard Controls
            key = cv2.waitKey(10) & 0xFF
            
            if key == 13: # Enter
                if len(self.points) == 4:
                    self.done = True
                else:
                    print(" >> Please select exactly 4 points.")
            elif key == 27: # ESC
                print("Exiting.")
                exit()
            
            # Arrow Key Nudging (Fine Tuning)
            if self.points:
                last_pt = self.points[-1]
                if key == 82: # Up
                    last_pt[1] -= 1
                elif key == 84: # Down
                    last_pt[1] += 1
                elif key == 81: # Left
                    last_pt[0] -= 1
                elif key == 83: # Right
                    last_pt[0] += 1

        cv2.destroyWindow(self.win_name)
        return np.array(self.points, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Step 2 & 3: High-Precision Measurement")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--calib", default="outputs/calibration.npz", help="Path to calibration file")
    # A4 Paper Default (0.297m x 0.210m)
    parser.add_argument("--ref_w", type=float, default=0.297, help="Reference width (m)")
    parser.add_argument("--ref_h", type=float, default=0.210, help="Reference height (m)")
    parser.add_argument("--true_w", type=float, default=0, help="Ground truth width (m)")
    parser.add_argument("--true_h", type=float, default=0, help="Ground truth height (m)")
    args = parser.parse_args()

    # --- 1. Load Calibration ---
    if not os.path.exists(args.calib):
        print("Error: Calibration file not found. Run Step 1 (calibrate_camera.py) first.")
        return
        
    data = np.load(args.calib)
    K, dist = data['K'], data['dist']

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return

    # --- 2. Undistort Image ---
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(img, K, dist, None, newcameramtx)

    print("\nStarting Measurement Tool...")
    selector = SmartPointSelector(undistorted_img)

    # --- 3. Step 2: Perspective Projection (Homography) ---
    # A. Select Reference (A4)
    ref_pts = selector.select_points("Select A4 Paper Corners (TL->TR->BR->BL)")

    # Define real-world coordinates (Z=0 plane)
    real_pts = np.array([
        [0, 0, 0],
        [args.ref_w, 0, 0],
        [args.ref_w, args.ref_h, 0],
        [0, args.ref_h, 0]
    ], dtype=np.float32)

    # --- 4. Step 3: Validation Experiment ---
    ret, rvec, tvec = cv2.solvePnP(real_pts, ref_pts, newcameramtx, None)
    distance = np.linalg.norm(tvec)
    
    print(f"\n--- Validation Check ---")
    print(f"Distance to Object: {distance:.3f} meters")
    if distance > 2.0:
        print("✅ PASS: Distance > 2m")
    else:
        print("⚠️ WARNING: Distance < 2m. Accuracy may be lower.")

    # Compute Homography
    H, status = cv2.findHomography(ref_pts, real_pts[:, :2])

    # B. Select Target Object
    obj_pts = selector.select_points("Select TARGET Object (TL->TR->BR->BL)")

    # --- 5. Calculation ---
    obj_pts_reshaped = obj_pts.reshape(-1, 1, 2)
    world_points = cv2.perspectiveTransform(obj_pts_reshaped, H).reshape(-1, 2)

    w_top = np.linalg.norm(world_points[0] - world_points[1])
    w_bot = np.linalg.norm(world_points[2] - world_points[3])
    h_left = np.linalg.norm(world_points[0] - world_points[3])
    h_right = np.linalg.norm(world_points[1] - world_points[2])

    avg_width = (w_top + w_bot) / 2.0
    avg_height = (h_left + h_right) / 2.0

    print(f"\n=== FINAL RESULTS ===")
    print(f"Estimated Width : {avg_width:.4f} m ({avg_width*100:.1f} cm)")
    print(f"Estimated Height: {avg_height:.4f} m ({avg_height*100:.1f} cm)")

    if args.true_w > 0:
        err_w = abs(avg_width - args.true_w) / args.true_w * 100
        err_h = abs(avg_height - args.true_h) / args.true_h * 100
        print(f"\n--- Accuracy ---")
        print(f"Width Error : {err_w:.2f}%")
        print(f"Height Error: {err_h:.2f}%")

    out_path = "outputs/measurement_result.jpg"
    for i in range(4):
        cv2.line(undistorted_img, tuple(obj_pts[i].astype(int)), 
                 tuple(obj_pts[(i+1)%4].astype(int)), (0, 255, 0), 2)
        
    cv2.imwrite(out_path, undistorted_img)
    print(f"Visual result saved to: {out_path}")

if __name__ == "__main__":
    main()