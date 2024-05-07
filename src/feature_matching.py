'''
Step2 | 

RanSac算法进行特征匹配

Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-11 16:59:11
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-07 21:33:07
FilePath: /code/feature_matching.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
from .feature_extraction import extract_features
import time
import os
import numpy as np
from typing import List
from .utils import print_colored

def match_features(img1, img2, features1, features2, img_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
                   thres= 0.4, tree = 5, checks = 50, flan_k = 2, save_dir = "output", alg = None, alg_params = None):
    """
    Match the features from two sets of keypoints and descriptors, and visualize the matches.
    
    Args:
        img1 (np.array): The first image.
        img2 (np.array): The second image.
        features1 (tuple): Tuple containing keypoints and descriptors of the first image.
        features2 (tuple): Tuple containing keypoints and descriptors of the second image.

    Returns:
        img_matches (np.array): An image showing the matches between the two input images.
        points1 (tuple): Keypoints from the first image that correspond to the good matches. <keypoints>
        points2 (tuple): Keypoints from the second image that correspond to the good matches. <keypoints>
    """
    if type(img1) == str:
        img1 = cv2.imread(img1)
    if type(img2) == str:
        img2 = cv2.imread(img2)
    

    if alg == "ransac":
        alg_ = cv2.RANSAC
    elif alg == "magsac":
        alg_ = cv2.USAC_MAGSAC
    elif alg == "None":
        alg_ = None
        alg = None
    else:
        raise ValueError("Invalid algorithm")
    
    keypoints1, descriptors1, _ = features1
    keypoints2, descriptors2, _ = features2
    
    # Create FLANN matcher object
    FLANN_INDEX_KDTREE = 1 
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=tree)
    search_params = dict(checks=checks)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors.
    matches = flann.knnMatch(descriptors1, descriptors2, k=flan_k)
    
    match_to_good_map = {}
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < thres * n.distance:
            match_to_good_map[i] = len(good_matches)
            good_matches.append((m,n))
    
    # good_matches = [(m,n) for m, n in matches if m.distance < thres * n.distance]
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m,_ in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m,_ in good_matches]).reshape(-1, 1, 2)
    
    if alg is not None:
        M, mask = cv2.findHomography(src_pts, dst_pts, alg_, **alg_params)

    # keep only the keypoints that correspond to the good matches
    points1 = []
    points2 = []
    for i, (m,n) in enumerate(good_matches): # NOTE here the point1 and points2 have been ordered in pairs, thus we don't need matches anymore
        if alg is not None:
            if mask[i] == 1: # if the match is good
                points1.append(keypoints1[m.queryIdx])
                points2.append(keypoints2[m.trainIdx])
        else:
            points1.append(keypoints1[m.queryIdx])
            points2.append(keypoints2[m.trainIdx])
            
    
        
    # # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if alg is not None:
            if m.distance < 0.25 * n.distance and mask[match_to_good_map[i]] == 1: # https://blog.csdn.net/qq_45832961/article/details/122776322 , here we set thres = 0.25 to better visualize the match
                matchesMask[i] = [1, 0]
        else:
            if m.distance < 0.25 * n.distance: # https://blog.csdn.net/qq_45832961/article/details/122776322 , here we set thres = 0.25 to better visualize the match
                matchesMask[i] = [1, 0]
    
    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=(255,0,0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    
    
    good_matches = tuple([(m,n) for m, n in matches if m.distance < thres * n.distance])
        
    good_matches = tuple([(m,n) for m, n in matches if m.distance < thres * n.distance])

    
    # turn list into tuple
    points1 = tuple(points1)
    points2 = tuple(points2)
    

    # Draw matches.
    img_matches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **draw_params) # use matches to draw, corresponding to the mask
    
                
    # Show the result
    save_path = os.path.join(save_dir,"sift_matches")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{img_name}_sift_matches.png')
    
    print_colored(f"Number of good matches: {len(good_matches)}", "green")
    
    if alg is not None:
        print_colored(f"Number of matches filtered by {alg}: {len(points1)}", "green")
    
    cv2.imwrite(save_path, img_matches)

    return img_matches, good_matches, points1, points2





