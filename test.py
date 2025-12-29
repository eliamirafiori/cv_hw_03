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
    # Input Handling (as per exercise description)
    image_path = "sudoku.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Could not find image '{image_path}'")
        return

    # Load image and keep a copy for display
    image = cv2.imread(image_path)
    original = image.copy()

    ### Feature Detection ###
    # We use Canny edge detection + Contours to find the grid

    # Algorithms like Canny Edge Detection operate on intensity changes (light to dark),
    # so color information is treated as unnecessary noise.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Technically: This smooths the image to remove "high-frequency noise".
    # If you don't blur, a single speck of dust or digital grain could be mistaken for an edge.
    #
    # Mathematically: This is a Convolution operation.
    # A small matrix (kernel) slides over the image.
    # The kernel values follow a 2D Gaussian distribution (a bell curve).
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Technically: The Canny algorithm is a multi-stage edge detector.
    # It finds pixels where the intensity changes most drastically (gradients).
    edged = cv2.Canny(blur, 75, 200)

    # Find contours
    # Technically: This function analyzes the binary image (black and white edges) to find connected curves.
    # It walks along the boundary of white pixels to separate "objects" from the black background
    #
    # Mathematically (Topological Analysis): The algorithm (based on Suzuki & Abe, 1985) scans the image rows.
    # When it transitions from black (0) to white (1), it marks a "border".
    # It then follows this border until it returns to the start point.
    # - cv2.RETR_EXTERNAL: This flag tells the algorithm to only retrieve the outermost contours.
    # It ignores contours inside other contours (e.g., the numbers inside the Sudoku grid).
    # Mathematically, it only keeps the "parents" in the hierarchy tree of nested shapes.
    #
    # - cv2.CHAIN_APPROX_SIMPLE: This compresses the contour data.
    # A vertical line of 100 pixels normally requires 100 coordinate pairs (x, y).
    # This flag reduces it to just the endpoints (2 coordinates), discarding the redundant points in between.
    # This saves memory and speeds up later calculations.
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Technically: This sorts the list of detected shapes from largest to smallest and keeps only the top 5
    # Mathematically: It calculates the Green's Theorem area for a polygon
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # Sort by largest area

    puzzle_contour = None

    # Loop over the contours to find the 4-sided polygon (the sudoku grid)
    for c in cnts:
        # Technically: This calculates the total length of the contour boundary
        #
        # Mathematically: It sums the Euclidean distances between consecutive points in the contour.
        # If the contour is closed (the True flag), it includes the distance from the last point back to the first
        peri = cv2.arcLength(c, True)

        # Approximate the contour to a polygon
        # Technically: This simplifies a jagged, noisy contour into a cleaner geometric shape with fewer vertices.
        # It asks: "Can I represent this complex shape with a simpler polygon that doesn't deviate more than ϵ from the original?"
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If our approximated contour has four points, we can assume we found the screen
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

    ### Define Destination (Synthetic) Plane ###

    # Let's define a target width and height for our new square image
    # (You can calculate max width/height dynamically, but hardcoding a square is fine for Sudoku)
    width = 400
    height = 400

    # These are the destination points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    dst_pts = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )

    ### Homography Computation ###

    # This maps the source points (tilted) to the destination points (flat)
    # Technically: This function calculates the 3×3 transformation matrix
    # needed to map the 4 corners of the detected polygon to the 4 corners of the flattened square.
    #
    # Mathematically (Solving a Linear System): A homography matrix has 9 elements,
    # but because it is scale-invariant (multiplying the whole matrix by 5 doesn't change the transformation),
    # we fix the last element (h33​) to 1. This leaves 8 unknowns (degrees of freedom).
    # To solve for 8 unknowns, we need 8 equations.
    # Each pair of matching points (x,y)→(x′,y′) provides exactly 2 equations.
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    ### Warping ###

    # Technically: This function takes the original image and the matrix H and renders the new image
    # 
    # Mathematically (Backward Mapping & Interpolation):
    # You might think the computer takes a pixel from the Source and moves it to the Destination.
    # It actually does the opposite. This is called "Inverse Mapping".
    warped = cv2.warpPerspective(original, H, (width, height))

    # Show results
    cv2.imshow("Original with Corners", image)
    cv2.imshow("Rectified (Warped)", warped)

    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
