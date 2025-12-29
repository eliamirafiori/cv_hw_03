import cv2
import numpy as np
import sys
import os


def order_points(pts):
    """
    Helper function to order coordinates:
    top-left, top-right, bottom-right, bottom-left.
    This is crucial for mapping the corners correctly.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def main():
    # --- Step 0: Input Handling (as per exercise description) ---
    # "Load the image myimage.jpg... optional: set a default value"
    image_path = "sudoku.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Could not find image '{image_path}'")
        return

    # Load image and keep a copy for display
    image = cv2.imread(image_path)
    original = image.copy()

    # --- Step A: Feature Detection ---
    # "Extract the corners in the image"
    # We use Canny edge detection + Contours to find the grid.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # Sort by largest area

    puzzle_contour = None

    # Loop over the contours to find the 4-sided polygon (the sudoku grid)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        # approximate the contour to a polygon
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, we can assume we found the screen
        if len(approx) == 4:
            puzzle_contour = approx
            break

    if puzzle_contour is None:
        print("Could not find the Sudoku grid corners.")
        return

    # Draw the found corners on the original image for visualization
    cv2.drawContours(image, [puzzle_contour], -1, (0, 255, 0), 2)

    # Prepare the Source Points (from the image)
    # We reshape to (4, 2) and order them consistently
    src_pts = order_points(puzzle_contour.reshape(4, 2))

    # --- Step B: Define Destination (Synthetic) Plane ---
    # "coordinates of the corners in a synthetic image plane (i.e. frontal undistorted view)"

    # Let's define a target width and height for our new square image
    # (You can calculate max width/height dynamically, but hardcoding a square is fine for Sudoku)
    width = 400
    height = 400

    # These are the destination points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    dst_pts = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )

    # --- Step C: Homography Computation ---
    # "Estimate the homography matrix"
    # This maps the source points (tilted) to the destination points (flat)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # --- Step D: Warping ---
    # "Visualize the warped image using the homography matrix"
    warped = cv2.warpPerspective(original, H, (width, height))

    # Show results
    cv2.imshow("Original with Corners", image)
    cv2.imshow("Rectified (Warped)", warped)

    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
