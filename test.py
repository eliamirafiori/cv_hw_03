import cv2 as cv
import numpy as np
import sys
import os


def stitch_two_images(img_left, img_right):
    """
    Stitches two images together by warping img_right to match img_left's perspective.
    """
    # 1. Convert to grayscale for feature detection
    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    # 2. Feature Detection (SIFT)
    # SIFT is robust to scale and rotation changes
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(
        gray_left, None
    )  # Keypoints & Descriptors for Left
    kp2, des2 = sift.detectAndCompute(
        gray_right, None
    )  # Keypoints & Descriptors for Right

    # 3. Feature Matching
    # We use FLANN or BFMatcher. KNN (k=2) allows us to use the Ratio Test.
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Filter Matches (Lowe's Ratio Test)
    # Only keep matches where the best match is significantly better than the second best.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # We need at least 4 matches to calculate a Homography (usually we want many more)
    if len(good_matches) < 4:
        print("Error: Not enough matches found between images.")
        return img_left

    # 5. Extract coordinates of the matching points
    # pts1 = points in left image, pts2 = points in right image
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 6. Compute Homography (RANSAC)
    # RANSAC is critical here: it finds the matrix that fits the majority of points
    # and ignores the "outliers" (wrong matches).
    # We map pts2 (right) -> pts1 (left)
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)

    # 7. Warping
    # We need a canvas large enough to hold both images.
    # For simplicity, we assume horizontal stitching (Right image added to Left).
    height_l, width_l = img_left.shape[:2]
    height_r, width_r = img_right.shape[:2]

    # The new width is roughly the sum of both (minus overlap), but we'll use sum to be safe.
    canvas_width = width_l + width_r
    canvas_height = max(height_l, height_r)

    # Warp the right image onto the new canvas using the Homography
    warped_right = cv.warpPerspective(img_right, H, (canvas_width, canvas_height))

    # 8. Blending (Linear/Overlay)
    # To avoid a sharp seam, we can blend.
    # A simple approach for this exercise:
    # Create a mask of where the warped image is, and overwrite with the left image.

    # Place the left image on the canvas
    result = warped_right.copy()

    # Simple Overlay: Just overwrite the left part.
    # Note: For "seamless" linear blending, you would calculate alpha masks here.
    # For this exercise, simple overlay + RANSAC usually satisfies the "alignment" requirement.
    result[0:height_l, 0:width_l] = img_left

    # Optional: Trimming the black border on the right
    # (Find the last non-black column to crop the result)
    gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_result, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv.boundingRect(contours[0])  # Get bounding box of valid pixels
        result = result[0:h, 0 : x + w]  # Crop

    return result


def main():
    # Load command line arguments
    # Usage: python stitch.py img1.jpg img2.jpg ...
    image_paths = sys.argv[1:]

    # Handle optional default values as per prompt instructions
    if len(image_paths) == 0:
        print("No images provided. Trying default 'img0.jpg' and 'img1.jpg'...")
        image_paths = ["./assets/stitching/img0.JPG", "./assets/stitching/img1.JPG"]

    # Load all images
    images = []
    for path in image_paths:
        img = cv.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load {path}")

    if len(images) < 2:
        print("Need at least 2 images to stitch.")
        return

    print(f"Loaded {len(images)} images. Starting stitching...")

    # Iterative Stitching
    # We take the first image as the base panorama, and stitch the next ones to it.
    panorama = images[0]

    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)}...")
        panorama = stitch_two_images(panorama, images[i])

    # Visualization
    cv.imshow("Final Panorama", panorama)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save the result
    cv.imwrite("./assets/stitching/panorama_result.jpg", panorama)
    print("Result saved as panorama_result.jpg")


if __name__ == "__main__":
    main()
