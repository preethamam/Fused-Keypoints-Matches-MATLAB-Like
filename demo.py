import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from show_matched_features import show_matched_features

img1_name = r"images/peppers_color.png"
img2_name = r"images/peppers_color_rotsc.png"
MIN_MATCH_COUNT = 10

img1 = cv.imread(img1_name, cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread(img2_name, cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask_putative = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask_putative[i] = [1, 0]
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

# cv.drawMatchesKnn expects list of lists as matches.
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask_putative,
    flags=cv.DrawMatchesFlags_DEFAULT,
)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
plt.figure("Putative matches")
plt.imshow(img3)
plt.axis('off')
plt.show()

imfused = show_matched_features(img1, src_pts, img2, dst_pts, matchesMask)
plt.figure("Inlier matches")
plt.imshow(imfused)
plt.axis('off')
plt.show()
