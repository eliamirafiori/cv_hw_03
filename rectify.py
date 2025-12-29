import numpy as np
import cv2 as cv
import glob
import os


def load_calibration(path: str = "assets/calibration/calibration.npz"):
    if not os.path.exists(path):
        print("Calibration file not found.")
        return None, None

    with np.load(path) as data:
        K = data["K"]
        dist = data["dist_coeffs"]
        return K, dist


def calibration(
    calibration_assets_path: str = "assets/calibration/",
    columns: int = 8,
    rows: int = 8,
    square_size: float = 1,
    debug: bool = False,
):
    """
    Calibration function
    """

    # Termination criteria for corner refinement (sub-pixel accuracy)
    # Stops when either:
    #  - max iterations are reached
    #  - or the desired accuracy is achieved
    criteria = (
        cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,
        30,  # max number of iterations
        0.001,  # minimum required accuracy (epsilon)
    )

    # Chessboard configuration
    inner_corners = (columns - 1, rows - 1)  # number of INNER corners (columns, rows)

    # Prepare 3D object points in real-world coordinates
    # The chessboard lies on the Z = 0 plane
    objp = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)

    # Generate grid and scale it by the square size (meter)
    # ':'	All rows, ':2'	First two columns only (index 0 and 1)
    objp[:, :2] = (
        np.mgrid[0 : inner_corners[0], 0 : inner_corners[1]].T.reshape(-1, 2)
        * square_size
    )

    # Containers for calibration points
    objpoints = []  # 3D points in real-world space (meter)
    imgpoints = []  # 2D points in image plane (pixels)

    # Load all calibration images from disk
    # Each image should show the same chessboard pattern
    images = glob.glob(f"{calibration_assets_path}/*.jpg")

    # Loop over each calibration image
    for img_path in images:
        if debug:
            print(f"Path:\n\t{img_path}")

        # Read image from disk (OpenCV loads images in BGR format)
        img_bgr = cv.imread(img_path)

        # Convert image to grayscale
        # Chessboard detection works on single-channel images
        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        # Detect chessboard inner corners
        #
        # corners_found:
        #   - True if all expected corners are detected
        # corners:
        #   - Detected corner locations (pixel coordinates)
        #
        # chessboard_size = (columns, rows)
        # Must match the object points definition exactly
        corners_found, corners = cv.findChessboardCorners(img_gray, inner_corners, None)

        if debug:
            print(f"Corners found:\n\t{corners_found}")

        # If the chessboard was successfully detected
        if corners_found:

            # Store the known 3D object points (real-world coordinates)
            # Same for every image, since the chessboard geometry is fixed
            objpoints.append(objp)

            # Refine corner positions to sub-pixel accuracy
            #
            # This improves calibration precision significantly
            #
            # (11, 11)  -> search window size
            # (-1, -1)  -> use default dead zone
            # criteria  -> termination criteria defined earlier
            corners_refined = cv.cornerSubPix(
                img_gray, corners, (11, 11), (-1, -1), criteria
            )

            # Store the refined 2D image points (pixel coordinates)
            imgpoints.append(corners_refined)

            if debug:
                # Visual feedback: draw detected corners on the image
                cv.drawChessboardCorners(
                    img_bgr, inner_corners, corners_refined, corners_found
                )

                # Display the image briefly
                cv.imshow(
                    "Calibration Image",
                    cv.resize(
                        img_bgr,
                        (img_bgr.shape[1] // 4, img_bgr.shape[0] // 4),
                    ),
                )
                cv.waitKey(500)  # display for 500 ms

    if debug:
        cv.destroyAllWindows()

    # Use any image size from your dataset
    image_shape = cv.imread(images[0]).shape[:2][::-1]  # width, height

    # Camera calibration
    #
    # Inputs:
    #  - objpoints : list of 3D real-world points (meter)
    #  - imgpoints : list of corresponding 2D image points (pixels)
    #  - image size: (width, height)
    #
    # Outputs:
    #  - rms_error  : RMS re-projection error
    #  - K          : camera intrinsic matrix (3x3)
    #  - dist_coeffs: distortion coefficients (5x1)
    #  - rot_vecs   : rotation vectors (3x1) (one per image)
    #  - trans_vecs : translation vectors (3x1) (one per image)
    #
    # OpenCV uses Rodrigues vectors to represent rotation
    # - 3 numbers → axis-angle representation
    # - Converts to a 3×3 rotation matrix using:
    #   - R, _ = cv.Rodrigues(rot_vecs[i])
    rms_error, K, dist_coeffs, rot_vecs, trans_vecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    # Compute the mean re-projection error (in pixels) over the calibration images
    errors = []

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rot_vecs, trans_vecs):
        projected, _ = cv.projectPoints(objp, rvec, tvec, K, dist_coeffs)
        projected = projected.reshape(-1, 2)
        imgp = imgp.reshape(-1, 2)

        # Euclidean pixel error per point
        err = np.linalg.norm(imgp - projected, axis=1)
        errors.append(err)

    errors = np.concatenate(errors)

    mean_error = np.mean(errors)
    std_error = np.std(errors)

    if debug:
        print(f"\nCamera Matrix K:\n{K}")
        print(f"Re-projection Error (in pixels):\n\t{rms_error}")
        # The error is good when it's under 0.08
        print(f"Mean reprojection error: {mean_error:.3f} px")
        print(f"Std dev reprojection error: {std_error:.3f} px")

        # Iterate over all images to show their individual poses
        for i, (r_vec, t_vec) in enumerate(zip(rot_vecs, trans_vecs)):
            R, _ = cv.Rodrigues(r_vec)
            print(f"\nImage {i} Pose")
            print(f"Rotation Matrix R:\n{R}")
            print(f"Translation t:\n{t_vec}")

    # Save calibration parameters
    param_path = os.path.join(calibration_assets_path, "calibration.npz")

    # Save several arrays into a single file in uncompressed .npz format
    np.savez(
        param_path,
        rms_error=rms_error,
        K=K,
        dist_coeffs=dist_coeffs,
        rot_vecs=rot_vecs,
        trans_vecs=trans_vecs,
    )

    return K, dist_coeffs


def remove_distortion(
    img_path: str,
    K=None,
    dist=None,
):
    # Health check
    assert os.path.exists(img_path), f"Image not found: {img_path}"

    if K is None and dist is None:
        raise RuntimeError(
            "Intrisic paramenters are and Distortion Coefficients are None. Try different images or a different feature extractor."
        )

    # Loads in GRAYSCALE because feature detection relies on intensity changes (gradients).
    # Color information is usually unnecessary for this and adds computational cost (3 channels vs 1).
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Query image (left)

    # Refine camera matrix to avoid losing pixels at the edges
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    # CV -> Undistort
    dst = cv.undistort(img, K, dist, None, new_camera_matrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    cv.imwrite("calibresult_undistort.png", dst)

    # CV -> Remapping
    mapx, mapy = cv.initUndistortRectifyMap(K, dist, None, new_camera_matrix, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    cv.imwrite("calibresult_remapping.png", dst)

    # Undistort
    img = cv.undistort(img, K, dist, None, K)

    return dst


def feature_detection(
    img_path: str,
    detector=None,
    K=None,
    dist=None,
    debug: bool = False,
):
    """
    Detects features in two images using ORB and routes them to a matcher.
    """

    # Health check
    assert os.path.exists(img_path), f"Left image not found: {img_path}"

    # Loads in GRAYSCALE because feature detection relies on intensity changes (gradients).
    # Color information is usually unnecessary for this and adds computational cost (3 channels vs 1).
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Query image (left)

    if K is not None and dist is not None:
        if debug:
            print("Undistorting images before detection...")

        # Refine camera matrix to avoid losing pixels at the edges
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
            K, dist, (w, h), 1, (w, h)
        )

        # CV -> Undistort
        dst1 = cv.undistort(img, K, dist, None, new_camera_matrix)

        # crop the image
        x, y, w, h = roi
        dst1 = dst1[y : y + h, x : x + w]
        cv.imwrite("calibresult.png", dst1)

        # Undistort
        img = cv.undistort(img, K, dist, None, K)

    # Initialize Detector
    # 'nfeatures=5000' is the maximum number of keypoints to retain.
    # The default is often 500, but 5000 is better for high-res images or detailed scenes
    # to ensure enough matches are found later.
    if detector is None:
        detector = cv.ORB_create(nfeatures=10000)

    # Detection & Description
    # detectAndCompute performs two steps:
    #   1. Detect: Finds 'Keypoints' (points of interest like corners/edges)
    #   2. Compute: Calculates 'Descriptors' (binary vectors that describe the area around the keypoint)
    # The 'mask=None' argument means we look for features in the entire image.
    kp, des = detector.detectAndCompute(img, None)

    # akaze = cv.AKAZE_create(threshold=0.0005)
    #
    # kp1, des1 = akaze.detectAndCompute(img1, None)
    # kp2, des2 = akaze.detectAndCompute(img2, None)

    if debug:
        print(f"Detected {len(kp)} keypoints in image.")

    # Safety Check
    # If an image has no texture (e.g., a blank wall), descriptors might be None.
    # Proceeding without this check would cause the matchers to crash.
    if des is None:
        raise RuntimeError(
            "Descriptors are None. Try different images or a different feature extractor."
        )

    return img, kp, des


if __name__ == "__main__":
    K, dist_coeffs = calibration(debug=True)
    remove_distortion(
        img_path="./assets/sudoku/sudoku_photo.jpg", K=K, dist=dist_coeffs
    )
    img, kp, des = feature_detection(
        img_path="./assets/sudoku/sudoku_photo.jpg", K=K, dist=dist_coeffs, debug=True
    )
