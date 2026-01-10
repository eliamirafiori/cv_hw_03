import cv2 as cv
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
    # Input Handling (as per exercise description)
    image_path = "./assets/rectify/sudoku.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Could not find image '{image_path}'")
        return

    # Load image and keep a copy for display
    image = cv.imread(image_path)
    original = image.copy()

    ### Feature Detection ###

    # We use Canny edge detection + Contours to find the grid

    # Algorithms like Canny Edge Detection operate on intensity changes (light to dark),
    # so color information is treated as unnecessary noise.
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray Scale", cv.resize(gray, None, fx=0.30, fy=0.30))
    cv.waitKey(0)
    cv.destroyAllWindows()

    # This smooths the image to remove "high-frequency noise".
    # If you don't blur, a single speck of dust or digital grain could be mistaken for an edge.
    #
    # This is a Convolution operation.
    # A small matrix (kernel) slides over the image.
    # The kernel values follow a 2D Gaussian distribution (a bell curve).
    blur = cv.GaussianBlur(
        gray, (5, 5), 0
    )  # The size must be an odd number to find the center
    cv.imshow("Gaussian Blur", cv.resize(blur, None, fx=0.30, fy=0.30))
    cv.waitKey(0)
    cv.destroyAllWindows()

    # The Canny algorithm is a multi-stage edge detector.
    # It finds pixels where the intensity changes most drastically (gradients).
    edged = cv.Canny(blur, 75, 200)
    cv.imshow("Edged (Canny)", cv.resize(edged, None, fx=0.30, fy=0.30))
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Find contours
    # This function analyzes the binary image (black and white edges) to find connected curves.
    # It walks along the boundary of white pixels to separate "objects" from the black background
    #
    # The algorithm (based on Suzuki & Abe, 1985) scans the image rows.
    # When it transitions from black (0) to white (1), it marks a "border".
    # It then follows this border until it returns to the start point.
    # - cv.RETR_EXTERNAL: This flag tells the algorithm to only retrieve the outermost contours.
    # It ignores contours inside other contours (e.g., the numbers inside the Sudoku grid).
    # Mathematically, it only keeps the "parents" in the hierarchy tree of nested shapes.
    #
    # - cv.CHAIN_APPROX_SIMPLE: This compresses the contour data.
    # A vertical line of 100 pixels normally requires 100 coordinate pairs (x, y).
    # This flag reduces it to just the endpoints (2 coordinates), discarding the redundant points in between.
    # This saves memory and speeds up later calculations.
    cnts, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # This sorts the list of detected shapes from largest to smallest and keeps only the top 5
    # It calculates the Green's Theorem area for a polygon
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]  # Sort by largest area

    puzzle_contour = None

    # Loop over the contours to find the 4-sided polygon (the sudoku grid)
    for c in cnts:
        # This calculates the total length of the contour boundary
        #
        # It sums the Euclidean distances between consecutive points in the contour.
        # If the contour is closed (the True flag), it includes the distance from the last point back to the first
        peri = cv.arcLength(c, True)

        # Approximate the contour to a polygon
        # This simplifies a jagged, noisy contour into a cleaner geometric shape with fewer vertices.
        # It asks: "Can I represent this complex shape with a simpler polygon that doesn't deviate more than ϵ from the original?"
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        # If our approximated contour has four points, we can assume we found the screen
        if len(approx) == 4:
            puzzle_contour = approx
            break

    if puzzle_contour is None:
        print("Could not find the Sudoku grid corners.")
        return

    # Draw the found corners on the original image for visualization
    cv.drawContours(image, [puzzle_contour], -1, (0, 255, 0), 2)

    # Prepare the Source Points (from the image)
    # We reshape to (4, 2) and order them consistently
    src_pts = order_points(puzzle_contour.reshape(4, 2))

    ### Define Destination (Synthetic) Plane ###

    # Let's define a target width and height for our new square image
    # (You can calculate max width/height dynamically, but hardcoding a square is fine for Sudoku)
    width = image.shape[1]
    height = image.shape[0]

    # These are the destination points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    dst_pts = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )

    ### Homography Computation ###

    # This maps the source points (tilted) to the destination points (flat)
    # This function calculates the 3×3 transformation matrix
    # needed to map the 4 corners of the detected polygon to the 4 corners of the flattened square.
    #
    # A homography matrix has 9 elements,
    # but because it is scale-invariant (multiplying the whole matrix by 5 doesn't change the transformation),
    # we fix the last element (h33​) to 1. This leaves 8 unknowns (degrees of freedom).
    # To solve for 8 unknowns, we need 8 equations.
    # Each pair of matching points (x,y)→ -> (x′,y′) provides exactly 2 equations.
    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    print(f"Homography Matrix:\n\t{H}")

    ### Warping ###

    # This function takes the original image and the matrix H and renders the new image
    #
    # (Backward Mapping & Interpolation):
    # You might think the computer takes a pixel from the Source and moves it to the Destination.
    # It actually does the opposite. This is called "Inverse Mapping".
    warped = cv.warpPerspective(original, H, (width, height))

    ### Show results ###

    cv.imshow("Original with Corners", cv.resize(image, None, fx=0.30, fy=0.30))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow("Rectified (Warped)", cv.resize(warped, None, fx=0.30, fy=0.30))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
