import cv2 as cv
import numpy as np
import sys

LOWE_RATION = 0.75
THRESHOLD_GOOD_MATCHES = 50


def detect_and_match(img1, img2):
    """
    Returns the number of good matches and the homography matrix.
    We match img2 -> img1.
    """
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    detector = cv.BRISK_create()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0, None, None

    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < LOWE_RATION * n.distance:
            good.append(m)

    if len(good) < 4:
        return 0, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Calculate Homography
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)

    # Count inliers (how many points fit the homography)
    matches_count = np.sum(mask) if mask is not None else 0
    return matches_count, H, pts2


def stitch_images(base_img, next_img, H, side="right"):
    """
    Stitches next_img onto base_img using Homography H.
    Handles canvas expansion for both Left and Right sides.
    """
    h1, w1 = base_img.shape[:2]
    h2, w2 = next_img.shape[:2]

    # Get the corners of the next image (to see where it lands)
    corners_next = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(corners_next, H)

    # Combine corners to find new canvas size
    # (0,0) is top-left of base_img. Warped corners might be negative (left side).
    all_pts = np.concatenate(
        ([[0, 0], [0, h1], [w1, h1], [w1, 0]], warped_corners.reshape(-1, 2)), axis=0
    )

    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift everything to positive coordinates
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # Warping
    # Note: We warp next_img using (Ht dot H) because we need to apply the shift AND the homography
    warped_next = cv.warpPerspective(next_img, Ht.dot(H), (xmax - xmin, ymax - ymin))

    # Create the final canvas
    # Place base_img at its new offset position
    output_img = warped_next.copy()
    output_img[t[1] : h1 + t[1], t[0] : w1 + t[0]] = base_img

    return output_img


def main():
    # 1. Load Images
    image_paths = sys.argv[1:]
    if not image_paths:
        print("Usage: python stitch.py img1.jpg img2.jpg ...")
        image_paths = ["./assets/stitch/img1.JPG", "./assets/stitch/img0.JPG"]

    images = []
    for path in image_paths:
        img = cv.imread(path)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        print("Need at least 2 images.")
        return

    print(f"Loaded {len(images)} images. Analyzing order...")

    # 2. Initialization
    # We start with the first image as the 'center' and try to attach others to it.
    panorama = images.pop(0)

    # 3. Iterative Stitching
    # We keep looping through the remaining list until it's empty or we can't find matches
    while images:
        best_match_idx = -1
        best_match_H = None
        best_match_score = 0
        best_side = "none"  # 'left' or 'right'

        # Check every remaining image against the current panorama
        for i, candidate in enumerate(images):

            # --- Check RIGHT side ---
            # Try matching candidate (img2) -> panorama (img1)
            # If H maps candidate to fit inside panorama, it overlaps.
            # But we want to know relative position.
            # Usually, we check translation in H.

            count, H, _ = detect_and_match(panorama, candidate)

            if count > THRESHOLD_GOOD_MATCHES:  # Threshold for a "good" match
                # Check translation component of H (H[0, 2])
                # If H[0,2] > 0, candidate is to the left (shifted right to match center)
                # If H[0,2] < 0, candidate is to the right (shifted left to match center)
                # Wait... homography is tricky.
                # Let's verify by checking where the center of candidate lands.
                h_c, w_c = candidate.shape[:2]
                center_pt = np.array([[[w_c / 2, h_c / 2]]], dtype=np.float32)
                warped_center = cv.perspectiveTransform(center_pt, H)

                # Center of panorama
                h_p, w_p = panorama.shape[:2]

                # If warped center x < 0, it belongs on the LEFT.
                # If warped center x > width, it belongs on the RIGHT.
                center_x = warped_center[0][0][0]

                side = "right" if center_x > w_p else "left"  # Simplified heuristic

                if count > best_match_score:
                    best_match_score = count
                    best_match_idx = i
                    best_match_H = H
                    best_side = side

        # 4. Apply Stitch if match found
        if best_match_idx != -1:
            print(f"Stitching image {best_match_idx} to the {best_side}...")
            next_img = images.pop(best_match_idx)
            panorama = stitch_images(panorama, next_img, best_match_H, best_side)
        else:
            print("Warning: Could not match remaining images.")
            break

    # Save the result
    cv.imwrite("./assets/stitch/panorama_result.jpg", panorama)
    print("Result saved as panorama_result.jpg")

    # Show the result
    cv.imshow("Result", cv.resize(panorama, None, fx=0.20, fy=0.20))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
