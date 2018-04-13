import random
import numpy as np
import cv2 as cv

def filter_ratio_matches(matches, kp1, kp2, ratio=0.7):
    """ Returns Filtered Image Matches based on ratio matches

        :param matches: List of Correspondence
        :param kp1: List points in Image 1
        :param kp2: List points in Image 2
        :param ratio: ratio between best/2nd best [0,1]. Default = 0.7 - Based on LOWA papers

        :type matches: List [cv2.DMatch]
        :type kp1: List cv2.KeyPoint
        :type kp2: List cv2.KeyPoint
        :type ratio: float

        :returns filtered matches, filtered keypoints in image 1, filtered keypoints in image 2
        :rtype List [cv2.DMatch],  List cv2.KeyPoint,  List cv2.KeyPoint
    """
    new_kp1, new_kp2, new_matches = [], [], []
    ctr = 0
    for i, (m, n) in enumerate(matches):  #
        if m.distance < ratio * n.distance:
            new_kp1.append(kp1[m.queryIdx])
            new_kp2.append(kp2[m.trainIdx])
            new_matches.append([cv.DMatch(ctr, ctr, m.distance)])
            ctr += 1
    return new_matches, new_kp1, new_kp2


# http://www.robots.ox.ac.uk/~vgg/presentations/bmvc97/criminispaper/node3.html
# other ref : https://github.com/alex011235/CVmosaicPi/blob/master/ransac.py#L43
def homomat(points_in_img1, points_in_img2):
    """
        Calculate Homography matrix between two set of points.
        Each row represent an item. 1st Column = x. 2nd Column = y

        :param points_in_img1: points in image 1.
        :param points_in_img2: points in image 2
        :type points_in_img1: numpy.array
        :type points_in_img2: numpy.array
        :return 3x3 homography matrix
        :rtype numpy.array

    """
    s = points_in_img1.shape[0]
    A = np.zeros((s * 2, 9))  # To
    for index in range(0, s):
        x, y = points_in_img1[index][0], points_in_img1[index][1]
        tx, ty = points_in_img2[index][0], points_in_img2[index][1]
        A[2 * index] = [x, y, 1, 0, 0, 0, -1 * tx * x, -1 * tx * y, -1 * tx]
        A[2 * index + 1] = [0, 0, 0, x, y, 1, -1 * ty * x, -1 * ty * y, -1 * ty]

    u, s, v = np.linalg.svd(A)
    H = v[-1].reshape(3, 3)  # eigenvector with the least eigenvalue
    return H / H[2, 2]


def ransac(matches, kp1, kp2, s=4, threshold=3, maxIterations=2000, returnMatches=False, inlierRatio=0.05, ransacRatio=0.6):
    """
        Return a ransac-optimized Homography matrix (np.array) with the size 3x3

        :param matches: List of Correspondence
        :param kp1: List points in Image 1
        :param kp2: List points in Image 2
        :param s: sample number
        :param threshold: outlier threshold. error > threshold = outlier. Between [1-10]
        :param maxIterations: maximal iteration of the ransac algorithm
        :param returnMatches: if True, return final c
        :param inlierRatio: probability minimal inlierRatio to check the model
        :param ransacRatio: the ratio between inliers / #sample to be considered a good model.

        :type matches: List [cv2.DMatch]
        :type kp1: List cv2.KeyPoint
        :type kp2: List cv2.KeyPoint
        :type s: int
        :type threshold:float
        :type maxIterations: int
        :type returnMatches: bool
        :type ransacRatio: float
        :type inlierRatio: float

        :return: 3x3 Homography Matrix
        :rtype: numpy.array
    """

    sizes_kp1 = [kp1[dt[0].queryIdx].size for dt in matches]
    sizes_kp2 = [kp1[dt[0].trainIdx].size for dt in matches]
    tup_matches_kp1 = [kp1[dt[0].queryIdx].pt for dt in matches]
    tup_matches_kp2 = [kp2[dt[0].trainIdx].pt for dt in matches]
    matches_kp1 = np.array([[h for h in kp] + [1] for kp in tup_matches_kp1])
    matches_kp2 = np.array([[h for h in kp] + [1] for kp in tup_matches_kp2])

    cnt_matches = len(matches)

    max_matches = []
    max_p1, max_p2 = [], []
    max_p1_sizes, max_p2_sizes = [], []
    max_total = 0

    for iter in range(maxIterations):
        # Find Homography based on random sample
        data = random.sample(matches, s)
        data_p1 = np.array([matches_kp1[dt[0].queryIdx] for dt in data])
        data_p2 = np.array([matches_kp2[dt[0].trainIdx] for dt in data])
        homography = homomat(data_p1[:, :2], data_p2[:, :2])

        # Find P1 projection from the homography matrix
        projected_p2 = np.dot(homography, matches_kp1.transpose())
        projected_p2 = projected_p2[0:3] / projected_p2[2]  # make sure w' is 1
        projected_p2 = projected_p2.transpose()

        # Initialize Current Matches
        current_matches = []
        current_p1, current_p2 = [], []
        current_p1_sizes, current_p2_sizes = [], []
        current_total = 0

        # Check for inliers and outliers for each matches
        for i, (match) in enumerate(matches):
            # normalize the error
            error = np.linalg.norm(matches_kp2[i] - projected_p2[i])

            # Check for inliers
            if error < threshold:
                current_matches.append([cv.DMatch(current_total, current_total, match[0].distance)])
                current_p1.append(matches_kp1[i][0:2])
                current_p2.append(matches_kp2[i][0:2])
                current_p1_sizes.append(sizes_kp1[i])
                current_p2_sizes.append(sizes_kp2[i])
                current_total += 1

        # If
        if current_total > max_total and current_total >= np.round(inlierRatio*cnt_matches):
            max_matches = current_matches
            max_p1 = current_p1
            max_p2 = current_p2
            max_p1_sizes = current_p1_sizes
            max_p2_sizes = current_p2_sizes
            max_total = current_total

            # # we are done in case we have enough inliers
            if current_total > cnt_matches * ransacRatio:
                break


    # Re-evaluate the Homography based on the best inliers
    max_homography = homomat(np.array(max_p1), np.array(max_p2))

    if returnMatches:
        max_kp1 = [cv.KeyPoint(d[0], d[1], max_p1_sizes[i]) for i, d in enumerate(max_p1)]
        max_kp2 = [cv.KeyPoint(d[0], d[1], max_p2_sizes[i]) for i, d in enumerate(max_p2)]
        return max_homography, max_matches, max_kp1, max_kp2

    return max_homography

def myPerspectiveTransform(pts, H):
    """
        Return points after the perspective transformation

        :param pts: List of points to be perspective transformed
        :param H: A 3x3 perspective transform matrix

        :type pts: Ndarray or list with size (-1, 2)
        :type H: 3x3 Ndarray

        :return: List of points after perspective transformation
        :rtype: numpy.array
    """

    # Clone and reshape the list of points
    new_pts = np.reshape(pts, (-1, 2))
    # Allocate a vector filled with one with size (-1, 1)
    one_vector = np.zeros((pts.shape[0], 1)) + 1
    # Concatenate the one vector to the list of points to form the homogenious coordiniate system
    new_pts = np.concatenate((new_pts, one_vector), axis=len(new_pts.shape)-1)

    # Perform transformation and transform results into the pixel coord. system
    # i.e., x' = x/w, and y' = y/w
    for i, pt in enumerate(new_pts):
        new_pts[i] = H.dot(pt.T)
        new_pts[i] /= new_pts[i, -1]

    # Return results with the same shape as the input has
    return new_pts[:, :-1].reshape(pts.shape)

def myWarpPerspective(img, H, output_shapes):
    """
        Return a transformed image according to the input homography matrix

        :param img: An image to be perspective transformed
        :param H: A 3x3 homography matrix
        :param output_shapes: The shape of output canvas

        :type img: Ndarray with the size (height, width, channels)
        :type H: 3x3 Ndarray
        :type output_shapes: Tuple (width, height)

        :return: An image after perspective transformation
        :rtype: numpy.array
    """
    c, r = output_shapes
    
    # Create an output canvas according to the parameter "output_shapes"
    if len(img.shape) == 3:
        output = np.zeros((r, c, 3))
    else:
        output = np.zeros((r, c, 1))

    # List of pixel coordinates in canvas
    inverse_map = [[i, j] for i in range(c) for j in range(r)]

    # Covert the coordinates in the system of img2 back to the system of img1 
    # to find out the reference points
    inverse_map = np.asarray(inverse_map)
    inverse_map = myPerspectiveTransform(inverse_map, np.linalg.inv(H))
    
    
    for i in range(c):
        for j in range(r):
            index = i*r + j
            ix, iy = inverse_map[index]
            
            # Because the converted coords. are float, 
            # we need to find out four ref. points to do bilinear interpolation
            tix, bix = np.ceil(ix), np.floor(ix)
            tiy, biy = np.ceil(iy), np.floor(iy)

            x_ratio = ix - bix
            y_ratio = iy - biy

            # Indexing does not allow float indices
            tix, bix, tiy, biy = np.int32(tix), np.int32(bix), np.int32(tiy), np.int32(biy)
            
            # Boundary checking: each ref point should locate within the input image
            if bix < 0 or biy < 0 or tix >= img.shape[1] or tiy >= img.shape[0]:
                continue
            else:
            # Bilinear interpolation
                output[j, i] = x_ratio*y_ratio*img[tiy, tix] \
                    + x_ratio*(1-y_ratio)*img[biy, tix] \
                    + (1-x_ratio)*y_ratio*img[tiy, bix] \
                    + (1-x_ratio)*(1-y_ratio)*img[biy, bix]
                output[j, i] = np.round(output[j, i])

    # Cast back to uint8 because of displaying and return results
    return np.uint8(output)

def warp(img1, img2, M):
    """
        Return a stiched image given img1 and img2 according to the input homography matrix

        :param img1: An image to be perspective transformed
        :param img2: An image to be stitched to
        :param H: A 3x3 homography matrix

        :type img1: Ndarray with the size (height, width, channels)
        :type img2: Ndarray with the size (height, width, channels)
        :type H: 3x3 Ndarray

        :return: A stiched image 
        :rtype: numpy.array
    """

    # Get width and height of input images 
    w1,h1 = img1.shape[:2]
    w2,h2 = img2.shape[:2]

    # Get the canvas dimesions
    img2_dims = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)
    img1_dims_temp = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)

    # Find out the boundary of img1 after projected onto the coord. system of img2
    img1_dims = myPerspectiveTransform(img1_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)
    
    # Getting images together
    # Calculate dimensions of match points
    x_min, y_min = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation 
    transform_dist = [-x_min,-y_min]
    transform_array = np.array([[1, 0, transform_dist[0]], 
                                [0, 1, transform_dist[1]], 
                                [0,0,1]]) 
    
    # Warp images to get the resulting image
    result_img = myWarpPerspective(img1, transform_array.dot(M),
                                    (x_max-x_min, y_max-y_min))
    
    result_img[transform_dist[1]:w1+transform_dist[1], 
                transform_dist[0]:h1+transform_dist[0]] = img2

    return result_img