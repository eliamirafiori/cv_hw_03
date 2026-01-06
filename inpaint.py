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

    # We use Canny edge detection + Contours to find the edges

    # Technically: This smooths the image to remove "high-frequency noise".
    # If you don't blur, a single speck of dust or digital grain could be mistaken for an edge.
    #
    # Mathematically: This is a Convolution operation.
    # A small matrix (kernel) slides over the image.
    # The kernel values follow a 2D Gaussian distribution (a bell curve).
    blur = cv.GaussianBlur(
        image, (5, 5), 0
    )  # The size must be an odd number to find the center
    debug_show("Gaussian Blur", blur)

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

    # Technically: The Canny algorithm is a multi-stage edge detector.
    # It finds pixels where the intensity changes most drastically (gradients).
    edged = cv.Canny(gray, 75, 200)
    edged = cv.bitwise_and(edged, edged, mask=mask)
    debug_show("Edged (Canny)", edged)

    # Find contours
    # Technically: This function analyzes the binary image (black and white edges) to find connected curves.
    # It walks along the boundary of white pixels to separate "objects" from the black background
    #
    # Mathematically (Topological Analysis): The algorithm (based on Suzuki & Abe, 1985) scans the image rows.
    # When it transitions from black (0) to white (1), it marks a "border".
    # It then follows this border until it returns to the start point.
    # - cv.RETR_LIST: It finds every single contour in the image,
    # whether it is an external boundary (like the soccer pitch)
    # or an internal hole (like the logo, a player, or the center circle).
    #
    # - cv.CHAIN_APPROX_NONE: This compresses the contour data.
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

    cv.ellipse(ellipse_image, biggest_ellipse, (0, 0, 255), 2)
    debug_show("Ellipse Detected", ellipse_image)

    if biggest_ellipse is None or ellipse_contour is None:
        print("Could not find the ellipse.")
        return

    # Draw the found corners on the original image for visualization
    cv.drawContours(image, [ellipse_contour], -1, (0, 255, 0), 2)
    debug_show("Contours Detected", image)

    # Combine them (Recomposition)
    result = insert_logo_roi(original, logo, order_points(ellipse_contour))
    debug_show("Final Inpainted Image", result)


if __name__ == "__main__":
    main()
