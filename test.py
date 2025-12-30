import cv2 as cv
import numpy as np
import sys


# --- Helper: Order Points ---
def order_points(pts):
    """
    Sorts 4 points in standard order: TL, TR, BR, BL.
    Crucial for correct Homography mapping.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # TL: min(sum), BR: max(sum)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # TR: min(diff), BL: max(diff)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# --- Step 1: Load Images ---
def load_images(img_path, logo_path):
    """Loads the pitch and logo (preserving alpha if present)."""
    img = cv.imread(img_path)
    # IMREAD_UNCHANGED is vital for loading PNG transparency (Alpha channel)
    logo = cv.imread(logo_path, cv.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error: Could not load pitch image from {img_path}")
        sys.exit(1)
    if logo is None:
        print(f"Error: Could not load logo image from {logo_path}")
        sys.exit(1)

    return img, logo


# --- Step 2: Extract Features (Center Circle) ---
def extract_circle_features(image):
    """
    Robustly detects the center circle by filtering for size and
    centrality (distance from image center).
    """
    h, w = image.shape[:2]
    image_center = np.array([w // 2, h // 2])

    # 1. Pre-processing (White Line Detection)
    # Convert to HLS to find high Lightness (White)
    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    L = hls[:, :, 1]

    # Thresholding
    # We use a high threshold to pick up ONLY the bright white lines
    _, mask = cv.threshold(L, 140, 255, cv.THRESH_BINARY)

    # Close gaps (halfway line often splits the circle)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # 2. Find Contours
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best_box = None
    min_dist_to_center = float("inf")

    # We define min/max area relative to image size to avoid the outer stadium
    img_area = h * w
    min_area = img_area * 0.001  # Circle must be at least 0.5% of image
    max_area = (
        img_area * 0.50
    )  # Circle cannot be bigger than 30% (avoids stadium boundary)

    debug_img = image.copy()  # For visualization

    for c in cnts:
        area = cv.contourArea(c)

        # Filter 1: Size Check
        if area < min_area or area > max_area:
            continue

        if len(c) < 5:
            continue  # Need points for ellipse

        try:
            # Fit Ellipse
            ellipse = cv.fitEllipse(c)
            (center, axes, angle) = ellipse

            # Filter 2: Aspect Ratio Check
            # A circle in perspective looks like a squashed ellipse,
            # but usually not a super thin line.
            major, minor = max(axes), min(axes)
            if minor / major < 0.15:  # If it's too thin, it's a line, not a circle
                continue

            # Filter 3: Centrality Check
            # We assume the "Center Circle" is roughly near the middle of the view
            dist_to_center = np.linalg.norm(np.array(center) - image_center)

            # We want the candidate that is closest to the center
            # AND passes our size checks.
            if dist_to_center < min_dist_to_center:
                min_dist_to_center = dist_to_center
                best_box = cv.boxPoints(ellipse)

                # Draw accepted candidates in Green
                cv.drawContours(debug_img, [np.int32(best_box)], -1, (0, 255, 0), 2)
            else:
                # Draw rejected candidates in Red
                rejected_box = cv.boxPoints(ellipse)
                cv.drawContours(debug_img, [np.int32(rejected_box)], -1, (0, 0, 255), 1)

        except Exception as e:
            continue

    # Show the debug image so you know what happened
    cv.imshow("Debug: Green=Selected, Red=Rejected", debug_img)
    # cv.waitKey(0) # Uncomment if you want to pause here

    if best_box is not None:
        return order_points(best_box)
    else:
        print("No valid center circle found. Try adjusting min/max_area.")
        return None


# --- Step 3: Estimate Homography ---
def estimate_homography(src_pts, side_length=400):
    """
    Calculates H matrix mapping source points to a 'side_length' square.
    """
    # Destination points: A perfect square
    dst_pts = np.float32(
        [
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1],
        ]
    )

    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    return H, (side_length, side_length)


# --- Step 4: Warp to Frontal Plane ---
def warp_to_frontal(image, H, size):
    """Warps the image to the top-down view."""
    return cv.warpPerspective(image, H, size)


# --- Step 5: Blend Logo (Linear Blending) ---
def blend_logo_on_circle(frontal_image, logo):
    """
    Resizes and blends the logo into the center of the frontal image.
    Handles both PNG (Alpha) and JPG (No Alpha).
    """
    bg_h, bg_w = frontal_image.shape[:2]

    # Resize logo to fit inside (e.g., 60% of the circle width)
    scale_factor = 0.6
    target_w = int(bg_w * scale_factor)
    aspect_ratio = logo.shape[1] / logo.shape[0]
    target_h = int(target_w / aspect_ratio)

    logo_resized = cv.resize(logo, (target_w, target_h))

    # Calculate center offset
    x_off = (bg_w - target_w) // 2
    y_off = (bg_h - target_h) // 2

    # Region of Interest (ROI)
    roi = frontal_image[y_off : y_off + target_h, x_off : x_off + target_w]

    # -- Blending Logic --
    if logo.shape[2] == 4:
        # PNG with Transparency: Use Alpha Channel
        alpha = logo_resized[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha

        # Expand dims for broadcasting: (H, W) -> (H, W, 3)
        alpha = np.dstack([alpha] * 3)
        alpha_inv = np.dstack([alpha_inv] * 3)

        logo_rgb = logo_resized[:, :, :3]

        # Formula: Final = (Logo * Alpha) + (Background * (1-Alpha))
        blended = (logo_rgb * alpha) + (roi * alpha_inv)
        frontal_image[y_off : y_off + target_h, x_off : x_off + target_w] = (
            blended.astype(np.uint8)
        )

    else:
        # JPG/No Transparency: Linear Weighted Add
        # Formula: Final = (Logo * 0.7) + (Background * 0.3)
        blended = cv.addWeighted(roi, 0.3, logo_resized, 0.7, 0)
        frontal_image[y_off : y_off + target_h, x_off : x_off + target_w] = blended

    return frontal_image


# --- Step 6: Inverse Homography & Composition ---
def warp_back_and_composite(original_image, frontal_image, H):
    """
    Warps the modified frontal image back and merges it seamlessly
    onto the original background.
    """
    h_orig, w_orig = original_image.shape[:2]

    # 1. Invert Homography
    H_inv = np.linalg.inv(H)

    # 2. Warp Back
    # This creates a black image with ONLY the modified circle area tilted correctly
    warped_back = cv.warpPerspective(frontal_image, H_inv, (w_orig, h_orig))

    # 3. Create Mask
    # We need to know which pixels in 'warped_back' are valid data vs black background.
    gray = cv.cvtColor(warped_back, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    # 4. Composite
    # Area A: The Original Image where the mask is BLACK (Background)
    bg = cv.bitwise_and(original_image, original_image, mask=cv.bitwise_not(mask))

    # Area B: The Warped Image where the mask is WHITE (Foreground/Logo)
    fg = cv.bitwise_and(warped_back, warped_back, mask=mask)

    # Final = Area A + Area B
    result = cv.add(bg, fg)

    return result


# --- Step 7: Main Execution ---
def main():
    # 1. Load
    img_path = "./pitch.jpg"
    logo_path = "./logo_transparent.png"
    # logo_path = "./logo2.jpg"

    # Handle CLI args
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    if len(sys.argv) > 2:
        logo_path = sys.argv[2]

    print(f"Loading {img_path} and {logo_path}...")
    original_img, logo_img = load_images(img_path, logo_path)

    # 2. Extract Features
    print("Detecting circle features...")
    src_pts = extract_circle_features(original_img)

    if src_pts is None:
        print("Failed to detect center circle. Please check image lighting/contrast.")
        return

    # Visual debug of detection (Optional)
    debug = original_img.copy()
    cv.drawContours(debug, [np.int32(src_pts)], -1, (0, 0, 255), 2)
    cv.imshow("Debug: Detected Corners", cv.resize(debug, None, fx=0.50, fy=0.50))

    # 3. Estimate Homography
    print("Computing Homography...")
    H, frontal_size = estimate_homography(src_pts, side_length=400)

    # 4. Warp Forward
    print("Warping to frontal plane...")
    frontal_img = warp_to_frontal(original_img, H, frontal_size)

    # 5. Blend Logo
    print("Blending logo...")
    frontal_blended = blend_logo_on_circle(frontal_img, logo_img)

    # 6. Inverse Warp & Composite
    print("Warping back to original view...")
    final_result = warp_back_and_composite(original_img, frontal_blended, H)

    # 7. Visualize
    cv.imshow("1. Frontal View (Edited)", frontal_blended)
    cv.imshow("2. Final Result", cv.resize(final_result, None, fx=0.50, fy=0.50))

    print("Press any key to exit.")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
