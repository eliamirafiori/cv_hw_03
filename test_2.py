import cv2 as cv
import numpy as np
import sys
import os


def debug_show(name, img):
    if img is None:
        return
    resized = cv.resize(img, None, fx=0.5, fy=0.5)
    cv.imshow(name, resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


def insert_logo_alpha_roi(original_img, logo_img, box_pts):
    # 1. Ensure logo has an alpha channel (BGRA)
    if logo_img.shape[2] != 4:
        print("Error: Logo does not have an alpha channel.")
        return original_img

    # 2. Get the axis-aligned bounding box for the ROI
    x, y, w, h = cv.boundingRect(box_pts.astype(np.int32))
    img_h, img_w = original_img.shape[:2]
    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(img_w, x + w), min(img_h, y + h)

    # Crop the pitch ROI (BGR)
    roi_pitch = original_img[y_start:y_end, x_start:x_end].copy().astype(float)
    actual_h, actual_w = roi_pitch.shape[:2]

    # 3. Adjust target points to the ROI coordinate system
    dst_pts_roi = box_pts - [x_start, y_start]

    # 4. Warp the logo (including alpha channel)
    lh, lw = logo_img.shape[:2]
    src_pts = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])
    H = cv.getPerspectiveTransform(src_pts, dst_pts_roi.astype(np.float32))

    # Warping a 4-channel image keeps the alpha channel intact
    warped_logo_roi = cv.warpPerspective(logo_img, H, (actual_w, actual_h)).astype(
        float
    )

    # 5. Extract RGB and Alpha from the warped result
    logo_rgb = warped_logo_roi[:, :, :3]
    # Normalize alpha to 0.0 - 1.0.
    # Optional: Multiply by 0.7 if you want the logo itself to be semi-transparent
    logo_alpha = (warped_logo_roi[:, :, 3] / 255.0) * 0.8

    # 6. Manual Alpha Blending
    # Formula: Out = (Foreground * Alpha) + (Background * (1 - Alpha))
    # We use [:, :, None] to allow the 2D alpha map to multiply the 3D RGB image
    blended_roi = logo_rgb * logo_alpha[:, :, None] + roi_pitch * (
        1.0 - logo_alpha[:, :, None]
    )

    # 7. Recompose
    result = original_img.copy()
    result[y_start:y_end, x_start:x_end] = blended_roi.astype(np.uint8)

    return result


def insert_logo_roi(original_img, logo_img, box_pts):
    if logo_img.shape[2] == 4:
        return insert_logo_alpha_roi(original_img, logo_img, box_pts)

    # 1. Get the axis-aligned bounding box for the ROI
    # This lets us crop a small square area from the pitch
    x, y, w, h = cv.boundingRect(box_pts.astype(np.int32))

    # Safety check: ensure coordinates are within image boundaries
    x, y = max(0, x), max(0, y)
    roi_pitch = original_img[y : y + h, x : x + w].copy()

    debug_show("ROI PITCH", roi_pitch)

    # 2. Adjust target points to the ROI coordinate system
    # Since we cropped the image, we must subtract (x, y) from the points
    dst_pts_roi = box_pts - [x, y]

    # 3. Define Logo Source Points
    lh, lw = logo_img.shape[:2]
    src_pts = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])

    # 4. Warp the logo to the ROI size
    H = cv.getPerspectiveTransform(src_pts, dst_pts_roi.astype(np.float32))
    warped_logo_roi = cv.warpPerspective(logo_img, H, (w, h))

    debug_show("WARPED LOGO ROI", warped_logo_roi)

    # 5. Create a mask of the logo within the ROI
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.fillConvexPoly(mask, dst_pts_roi.astype(np.int32), 255)

    # 6. Blend only the ROI
    # We use a 50/50 blend here, but you can change 0.5 to your liking
    blended_roi = cv.addWeighted(roi_pitch, 0.5, warped_logo_roi, 0.5, 0)

    # 7. Use the mask to place the blend ONLY inside the parallelogram
    # Outside the parallelogram, we keep the original roi_pitch pixels
    mask_bool = mask == 255
    roi_pitch[mask_bool] = blended_roi[mask_bool]

    # 8. Put the processed ROI back into the full original image
    result = original_img.copy()
    result[y : y + h, x : x + w] = roi_pitch

    return result


# Blend logo on frontal plane
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

    return frontal


# Warp back to original view
def warp_back(img, H, shape):
    H_inv = np.linalg.inv(H)
    final = cv.warpPerspective(img, H_inv, (shape[1], shape[0]))
    return final


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
    image_path = "./pitch.jpg"
    logo_path = "./logo_transparent.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Could not find image '{image_path}'")
        return

    # Load image and keep a copy for display
    image = cv.imread(image_path)
    original = image.copy()
    logo = cv.imread(logo_path, cv.IMREAD_UNCHANGED)

    ### Feature Detection ###
    # We use Canny edge detection + Contours to find the grid

    # Algorithms like Canny Edge Detection operate on intensity changes (light to dark),
    # so color information is treated as unnecessary noise.
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow("Gray Scale", cv.resize(gray, None, fx=0.30, fy=0.30))
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    gray = image

    # Technically: This smooths the image to remove "high-frequency noise".
    # If you don't blur, a single speck of dust or digital grain could be mistaken for an edge.
    #
    # Mathematically: This is a Convolution operation.
    # A small matrix (kernel) slides over the image.
    # The kernel values follow a 2D Gaussian distribution (a bell curve).
    blur = cv.GaussianBlur(
        gray, (5, 5), 0
    )  # The size must be an odd number to find the center
    debug_show("Gaussian Blur", blur)

    ##################################

    # 1. Convert to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # 2. Define Green Range
    # These values usually cover most grass under standard lighting
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # 3. Create the mask
    mask = cv.inRange(hsv, lower_green, upper_green)

    # 4. Clean up the mask (Optional but recommended)
    # Removes small holes (players) and noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # Algorithms like Canny Edge Detection operate on intensity changes (light to dark),
    # so color information is treated as unnecessary noise.
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    debug_show("Gray Scale", gray)

    blur = gray

    ##################################

    # Technically: The Canny algorithm is a multi-stage edge detector.
    # It finds pixels where the intensity changes most drastically (gradients).
    edged = cv.Canny(blur, 75, 200)
    edged = cv.bitwise_and(edged, edged, mask=mask)
    debug_show("Edged (Canny)", edged)

    # Find contours
    # Technically: This function analyzes the binary image (black and white edges) to find connected curves.
    # It walks along the boundary of white pixels to separate "objects" from the black background
    #
    # Mathematically (Topological Analysis): The algorithm (based on Suzuki & Abe, 1985) scans the image rows.
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
    cnts, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Technically: This sorts the list of detected shapes from largest to smallest and keeps only the top 5
    # Mathematically: It calculates the Green's Theorem area for a polygon
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)  # Sort by largest area

    # Create a copy to draw ellipses on
    ellipse_image = original.copy()

    biggest_ellipse = None
    max_area = 0
    ellipse_contour = None

    # Loop over the contours to find the 4-sided polygon (the sudoku grid)
    for c in cnts:
        if len(c) < 5:
            print("Could not find enough contours.")
            continue

        ellipse = cv.fitEllipse(c)

        # 3. Optional: Filter by area or aspect ratio to ignore noise
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        if minor_axis > 0 and 3 <= (major_axis / minor_axis) <= 3.5:
            area = np.pi * (major_axis / 2) * (minor_axis / 2)
            if area > max_area:
                max_area = area
                biggest_ellipse = ellipse
                box = cv.boxPoints(biggest_ellipse)
                ellipse_contour = np.intp(box)  # Convert to integers
                continue

                # 1. Get the convex hull of the contour (the 'tightest' wrapping)
                # This removes internal noise and keeps only the outer boundary
                hull = cv.convexHull(c)

                # 2. Approximate the hull to a 4-sided polygon
                # This will force the 'bounding shape' to try to be a parallelogram/quadrilateral
                peri = cv.arcLength(hull, True)
                approx = cv.approxPolyDP(hull, 0.02 * peri, True)

                # 3. Draw the resulting 4-sided tilted box
                if len(approx) > 4:
                    # If it found more points, we take the 'Minimum Area Rect'
                    # as the best mathematical parallelogram approximation
                    rect = cv.minAreaRect(c)
                    box = cv.boxPoints(rect)
                    ellipse_contour = np.intp(box)  # Convert to integers

    cv.ellipse(ellipse_image, biggest_ellipse, (0, 0, 255), 2)
    debug_show("Ellipse Detected", ellipse_image)

    if biggest_ellipse is None or ellipse_contour is None:
        print("Could not find the ellipse.")
        return

    # Draw the found corners on the original image for visualization
    cv.drawContours(image, [ellipse_contour], -1, (0, 255, 0), 2)
    debug_show("Contours Detected", image)

    # Prepare the Source Points (from the image)
    # We reshape to (4, 2) and order them consistently
    src_pts = order_points(ellipse_contour.reshape(4, 2))

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
    # Technically: This function calculates the 3×3 transformation matrix
    # needed to map the 4 corners of the detected polygon to the 4 corners of the flattened square.
    #
    # Mathematically (Solving a Linear System): A homography matrix has 9 elements,
    # but because it is scale-invariant (multiplying the whole matrix by 5 doesn't change the transformation),
    # we fix the last element (h33​) to 1. This leaves 8 unknowns (degrees of freedom).
    # To solve for 8 unknowns, we need 8 equations.
    # Each pair of matching points (x,y)→ -> (x′,y′) provides exactly 2 equations.
    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    print(f"Homography Matrix:\n\t{H}")

    ### Warping ###

    # Technically: This function takes the original image and the matrix H and renders the new image
    #
    # Mathematically (Backward Mapping & Interpolation):
    # You might think the computer takes a pixel from the Source and moves it to the Destination.
    # It actually does the opposite. This is called "Inverse Mapping".
    warped = cv.warpPerspective(original, H, (width, height))

    #################################
    # 2. Define the Target Size (The dimensions of your logo file)
    logo_h, logo_w = logo.shape[:2]

    # 3. Define Source Points (The square logo corners)
    src_pts = np.float32(
        [[0, 0], [logo_w - 1, 0], [logo_w - 1, logo_h - 1], [0, logo_h - 1]]
    )

    # 4. Define Destination Points (The tilted box on the pitch)
    # We must ensure 'box' points are in the same order as src_pts
    # using your existing order_points function
    dst_pts = order_points(box)

    # 5. Compute Homography
    H = cv.getPerspectiveTransform(src_pts, dst_pts)

    # 6. Warp the Logo to the pitch perspective
    h_bg, w_bg = original.shape[:2]
    warped_logo = cv.warpPerspective(logo, H, (w_bg, h_bg))
    debug_show("Warped Logo", warped_logo)

    # 7. Create a Mask of the box area
    mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    cv.fillConvexPoly(mask, np.int32(dst_pts), 255)

    # 8. Blending and Recomposition
    # Create an inverse mask to 'cut a hole' in the original image
    mask_inv = cv.bitwise_not(mask)

    # Keep everything EXCEPT the box from the original image
    bg_with_hole = cv.bitwise_and(original, original, mask=mask_inv)

    # Keep ONLY the warped logo inside the box
    logo_only = cv.bitwise_and(warped_logo, warped_logo, mask=mask)

    # Combine them (Recomposition)
    result = insert_logo_roi(original, logo, order_points(ellipse_contour))
    debug_show("Frontal With Logo", result)

    #################################

    frontal = blend_logo(warped, logo, biggest_ellipse[0])
    debug_show("Frontal With Logo", frontal)

    final = warp_back(frontal, H, image.shape)
    debug_show("Final Inpainted Image", final)


if __name__ == "__main__":
    main()
