import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Utility: show debug image resized
# --------------------------------------------------
def debug_show(name, img):
    if img is None:
        return
    resized = cv.resize(img, None, fx=0.5, fy=0.5)
    cv.imshow(name, resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


# --------------------------------------------------
# 1. Load images
# --------------------------------------------------
def load_images(pitch_path, logo_path):
    pitch = cv.imread(pitch_path)
    logo = cv.imread(logo_path, cv.IMREAD_UNCHANGED)

    if pitch is None or logo is None:
        raise IOError("Error loading images")

    debug_show("Input Pitch", pitch)
    debug_show("Logo", logo)

    return pitch, logo


# --------------------------------------------------
# 2. Extract white pitch lines
# --------------------------------------------------
def extract_white_lines(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])

    mask = cv.inRange(hsv, lower_white, upper_white)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    debug_show("White Lines Mask", mask)

    return mask


# --------------------------------------------------
# 3. Detect pitch boundary lines
# --------------------------------------------------
def detect_pitch_lines(mask):
    edges = cv.Canny(mask, 50, 150, apertureSize=3)
    debug_show("Canny Edges", edges)

    lines = cv.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=200, maxLineGap=20
    )

    line_vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv.line(line_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    debug_show("Detected Lines", line_vis)

    return lines


# --------------------------------------------------
# 4. Extract pitch corners (feature points)
# --------------------------------------------------
def extract_pitch_corners(lines, img_shape):
    h, w = img_shape[:2]

    horizontals = []
    verticals = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:
            horizontals.append((x1, y1, x2, y2))
        elif abs(x1 - x2) < 10:
            verticals.append((x1, y1, x2, y2))

    top = min(horizontals, key=lambda l: l[1])
    bottom = max(horizontals, key=lambda l: l[1])
    left = min(verticals, key=lambda l: l[0])
    right = max(verticals, key=lambda l: l[0])

    src_pts = np.array(
        [
            [left[0], top[1]],
            [right[0], top[1]],
            [right[0], bottom[1]],
            [left[0], bottom[1]],
        ],
        dtype=np.float32,
    )

    corner_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for p in src_pts.astype(int):
        cv.circle(corner_vis, tuple(p), 10, (0, 255, 0), -1)

    debug_show("Detected Pitch Corners", corner_vis)

    return src_pts


# --------------------------------------------------
# 5. Synthetic frontal plane
# --------------------------------------------------
def synthetic_pitch(size=(800, 500)):
    w, h = size
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    return dst_pts, size


# --------------------------------------------------
# 6. Homography
# --------------------------------------------------
def compute_homography(src_pts, dst_pts):
    H, _ = cv.findHomography(src_pts, dst_pts)
    return H


# --------------------------------------------------
# 7. Warp to frontal view
# --------------------------------------------------
def warp_to_frontal(img, H, size):
    frontal = cv.warpPerspective(img, H, size)
    debug_show("Frontal View", frontal)
    return frontal


# --------------------------------------------------
# 8. Detect center circle in frontal view
# --------------------------------------------------
def detect_center_circle(frontal):
    gray = cv.cvtColor(frontal, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 1.5)

    debug_show("Frontal Gray", gray)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=150,
    )

    if circles is None:
        raise RuntimeError("Center circle not detected")

    circle = np.uint16(np.around(circles))[0][0]

    circle_vis = frontal.copy()
    cv.circle(circle_vis, (circle[0], circle[1]), circle[2], (0, 0, 255), 3)
    cv.circle(circle_vis, (circle[0], circle[1]), 5, (255, 0, 0), -1)

    debug_show("Detected Center Circle", circle_vis)

    return circle


# --------------------------------------------------
# 9. Blend logo on frontal plane
# --------------------------------------------------
def blend_logo(frontal, logo, center):
    cx, cy, r = center
    logo = cv.resize(logo, (2 * r, 2 * r))
    x, y = cx - r, cy - r

    if logo.shape[2] == 4:
        alpha = logo[:, :, 3] / 255.0
        for c in range(3):
            frontal[y : y + 2 * r, x : x + 2 * r, c] = (
                alpha * logo[:, :, c]
                + (1 - alpha) * frontal[y : y + 2 * r, x : x + 2 * r, c]
            )
    else:
        frontal[y : y + 2 * r, x : x + 2 * r] = logo

    debug_show("Frontal With Logo", frontal)

    return frontal


# --------------------------------------------------
# 10. Warp back to original view
# --------------------------------------------------
def warp_back(img, H, shape):
    H_inv = np.linalg.inv(H)
    final = cv.warpPerspective(img, H_inv, (shape[1], shape[0]))
    debug_show("Final Inpainted Image", final)
    return final


# --------------------------------------------------
# 11. Main
# --------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python inpaint.py pitch.jpg logo.jpg")
        return

    pitch, logo = load_images(sys.argv[1], sys.argv[2])

    mask = extract_white_lines(pitch)
    lines = detect_pitch_lines(mask)
    src_pts = extract_pitch_corners(lines, pitch.shape)

    dst_pts, size = synthetic_pitch()
    H = compute_homography(src_pts, dst_pts)

    frontal = warp_to_frontal(pitch, H, size)
    center_circle = detect_center_circle(frontal)

    frontal = blend_logo(frontal, logo, center_circle)
    _ = warp_back(frontal, H, pitch.shape)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
