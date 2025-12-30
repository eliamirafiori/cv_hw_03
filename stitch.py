import cv2 as cv
import numpy as np

### Step 1: Load Images ###

# Load the images you want to stitch.
# Ensure the images have some overlapping regions.
img1 = cv.imread("./assets/stitch/img1.JPG")
img2 = cv.imread("./assets/stitch/img0.JPG")

### Step 2: Detect Features and Find Matches ###

# We'll use the SIFT (Scale-Invariant Feature Transform)
# algorithm to detect and describe features.

# Initialize SIFT detector
sift = cv.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Use BFMatcher to find matches
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)
matches

### Step 3: Draw Matches (Optional) ###

# Visualize the matches to understand how well the
# features are detected and matched.

# img_matches = cv.drawMatches(
#     img1,
#     kp1,
#     img2,
#     kp2,
#     matches[:50],
#     None,
#     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
# )
# cv.imshow(img_matches)
# cv.waitKey(0)
# cv.destroyAllWindows()

### Step 4: Homography Estimation ###

# Calculate the homography matrix using the matched keypoints.

# Extract location of good matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute homography
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

### Step 5: Warp Images ###

# Warp one image to align with the other using the homography matrix.

# Get the dimensions of the images
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Get the canvas dimesions
pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
dst = cv.perspectiveTransform(pts, H)
img2_warped = cv.warpPerspective(img2, H, (w1 + w2, h1))

# Place the first image on the canvas
img2_warped[0:h1, 0:w1] = img1

### Step 6: Blend Images ###

# Blend the images to create a seamless panorama.

# Simple blending technique
result = img2_warped

cv.imshow("Result", cv.resize(result, None, fx=0.20, fy=0.20))
cv.waitKey(0)
cv.destroyAllWindows()
