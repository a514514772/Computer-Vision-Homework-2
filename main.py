from lib.hw2 import *
from lib.util import *
import cv2 as cv

import argparse

parser = argparse.ArgumentParser(description="CV HW2 image stitching")
parser.add_argument('-img1', type=str, default='./data/1.jpg')
parser.add_argument('-img2', type=str, default='./data/2.jpg')

args = parser.parse_args()

# Reading Data
img1 = cv.imread(args.img1)
img2 = cv.imread(args.img2)

# Converting Images
gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


# Make SIFT Object
sift = cv.xfeatures2d.SIFT_create()

# Detect Local Descriptors and Keypoints
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# Brute Force Local Descripter Matcher
bf = cv.BFMatcher(cv.NORM_L2)


# For each local descriptor des1, find 2 best correspondence from des 2
matches = bf.knnMatch(des1, des2, k=2)  # Think of it just finding a cluster of ssd.

# Show current correspondences
output_1 =  cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)

# Filter those correspondences using ratio test
matches, kp1, kp2 = filter_ratio_matches(matches, kp1, kp2, 0.75)
# Show current correspondences
output_2 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)


# Real Homography
src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in matches ]).reshape(-1,1,2)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)
mask = mask.tolist()

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = mask,
                   flags = 0)

output_1 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

"""
    FILTERING CORRESPONDENCES
    
"""

#
# Do Ransac to get homography and true correspondences
max_homography, ransac_matches, max_kp1, max_kp2 = ransac(matches, kp1, kp2, threshold=4, returnMatches = True)
# max_homography = ransac(matches, kp1, kp2, threshold=22.5) # to get only homography value

# Show current correspondences
output_3 = cv.drawMatchesKnn(img1,max_kp1,img2,max_kp2,ransac_matches,None, flags=2)

print(max_homography)
print(M) # compare with the target Homography

print ("Warping ...")
stiched_img = warp(img1, img2, max_homography)

show_comparison_image(output_1, output_3, stiched_img)
