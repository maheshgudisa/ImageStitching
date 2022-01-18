"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np

np.random.seed(27)

def solution(left_img, right_img):
    '''
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    '''

    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp1 = np.float32([i.pt for i in kp1])

    gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)
    kp2 = np.float32([i.pt for i in kp2])

    y = np.asarray(des2)
    res = []
    for index, ds in enumerate(des1):
        x = np.array(ds)
        dist = np.linalg.norm(x - y, axis=1)
        indices = sorted(range(len(dist)), key=lambda sub: dist[sub])[:2]
        top_2 = [dist[i] for i in indices]
        res.append((index, x, top_2, indices))

    y = np.asarray(des1)
    res2 = []
    for index, ds in enumerate(des2):
        x = np.array(ds)
        dist = np.linalg.norm(x - y, axis=1)
        indices = sorted(range(len(dist)), key=lambda sub: dist[sub])[:2]
        top_2 = [dist[i] for i in indices]
        res2.append((index, x, top_2, indices))

    goodmatches = []
    for index, src, dest, indices in res:
        dist1 = np.linalg.norm(src - dest[0])
        dist2 = np.linalg.norm(src - dest[1])
        if dist1 < 0.75 * dist2:
            goodmatches.append([index, src, dest, indices])

    goodmatches2 = []
    for index, src, des, c_indices in goodmatches:
        found = False
        if index == res2[c_indices[0]][3][0]:
            goodmatches2.append((kp1[index], kp2[c_indices[0]]))

    max_inliers = 0
    max_inliers_list = []
    probs = [1 / len(goodmatches2) for i in range(len(goodmatches2))]
    indices = np.asarray([i for i in range(len(goodmatches2))])
    for i in range(5000):
        n_samples = 4
        inds = np.random.choice(len(goodmatches2), n_samples, p=probs)
        rand_points = [goodmatches2[ind] for ind in inds]
        transformation_matrix = np.zeros((len(rand_points) * 2, 9))
        count = 0
        for index, (src, dest) in enumerate(rand_points):
            for j in range(2):
                if j == 0:
                    transformation_matrix[count][0:3] = [dest[0], dest[1], 1]
                    transformation_matrix[count][3:6] = [0, 0, 0]
                else:
                    transformation_matrix[count][0:3] = [0, 0, 0]
                    transformation_matrix[count][3:6] = [dest[0], dest[1], 1]
                transformation_matrix[count][6:] = [-1 * dest[0] * src[j],
                                                    -1 * dest[1] * src[j],
                                                    -1 * src[j]]
                count += 1
        u, sigma, vt = np.linalg.svd(transformation_matrix)
        H = vt[-1].reshape(3, 3)
        if H[2, 2] != 0:
            H = H / H[2, 2]
        remaining_points = np.asarray([goodmatches2[pt][0] for pt in np.delete(indices, inds)])
        left_pts = np.hstack((remaining_points, np.ones((remaining_points.shape[0], 1), dtype=remaining_points.dtype)))
        left_pts = left_pts / left_pts[:, -1].reshape(-1, 1)
        remaining_points = np.asarray([goodmatches2[pt][1] for pt in np.delete(indices, inds)])
        right_pts = np.hstack((remaining_points, np.ones((remaining_points.shape[0], 1), dtype=remaining_points.dtype)))
        right_pts = right_pts / right_pts[:, -1].reshape(-1, 1)
        cal_left_pts = np.dot(right_pts, H)
        diff = np.linalg.norm(left_pts - cal_left_pts, axis=1)
        inliers_idx = list(np.where(diff <= 500))
        count = (diff <= 500).sum()
        if count > max_inliers:
            max_inliers = count
            max_inliers_list = inliers_idx.copy()
    all_points = [goodmatches2[i] for i in max_inliers_list[0]]
    transformation_matrix = np.zeros((len(all_points) * 2, 9))
    count = 0
    for index, (src, dest) in enumerate(all_points):
        for j in range(2):
            if j == 0:
                transformation_matrix[count][0:3] = [dest[0], dest[1], 1]
                transformation_matrix[count][3:6] = [0, 0, 0]
            else:
                transformation_matrix[count][0:3] = [0, 0, 0]
                transformation_matrix[count][3:6] = [dest[0], dest[1], 1]
            transformation_matrix[count][6:] = [-1 * dest[0] * src[j],
                                                -1 * dest[1] * src[j],
                                                -1 * src[j]]
            count += 1
    u, sigma, vt = np.linalg.svd(transformation_matrix)
    H = vt[-1].reshape(3, 3)
    if H[2, 2] != 0:
        H = H / H[2, 2]
    left_points = np.float32(
        [[0, 0], [0, left_img.shape[0]], [left_img.shape[1], left_img.shape[0]], [left_img.shape[1], 0]]).reshape(-1, 1,
                                                                                                                  2)
    temp = np.float32(
        [[0, 0], [0, right_img.shape[0]], [right_img.shape[1], right_img.shape[0]], [right_img.shape[1], 0]]).reshape(
        -1, 1, 2)
    right_points = cv2.perspectiveTransform(temp, H)
    list_of_points = np.vstack((left_points, right_points))

    [min_x, min_y] = np.int32(list_of_points.min(axis=0).flatten())
    [max_x, max_y] = np.int32(list_of_points.max(axis=0).flatten())

    temp = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    new_H = np.dot(temp, H)

    result = cv2.warpPerspective(right_img, new_H, (max_x - min_x, max_y - min_y))
    result[-min_y:left_img.shape[0] + (-min_y), -min_x:left_img.shape[1] + (-min_x)] = left_img
    return result


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)