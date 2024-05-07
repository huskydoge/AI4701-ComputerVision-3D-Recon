'''
Step 1

Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-11 16:58:26
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-03 21:50:48
FilePath: /Final/code/feature_extraction.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import os
import numpy as np
from .utils import print_colored

def extract_features(image, contrast_thresh=0.04, edge_thresh=10, sigma=1.6, image_name = "0000", save_dir = "output"):
    """
    Extract feature vectors from an image using the SIFT algorithm with customizable threshold settings.

    Args:
        image (str): Path to the input image or CV2 image object.
        contrast_thresh (float): Threshold used to filter out weak features in low-contrast regions.
        edge_thresh (float): Threshold used to filter out edge-like features.

    Returns:
        keypoints (list): List of keypoints detected in the image.
        descriptors (numpy.ndarray): Array of feature descriptors.
        image_with_keypoints (numpy.ndarray): Image with keypoints drawn.
    """
    # Read the image from the file
    if type(image) == str:
        image = cv2.imread(image)
    elif type(image) == np.ndarray:
        pass
    else:
        raise ValueError("Invalid input image type")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector with specified thresholds
    sift = cv2.SIFT_create(contrastThreshold=contrast_thresh, edgeThreshold=edge_thresh, sigma = sigma) 

    # Detect keypoints and descriptors using SIFT
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Draw keypoints on the image for visualization
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Create a directory for saving output images if it doesn't exist
    dir = os.path.join(save_dir,"sift_keypoints")
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    save_path = os.path.join(dir, f'{image_name}_sift_keypoints.png')
    cv2.imwrite(save_path, image_with_keypoints)
    print_colored(f"\nkeypoints number for {image_name}: {len(keypoints)}", "green")

    return keypoints, descriptors, image_with_keypoints

# Example usage
if __name__ == "__main__":
    image_path = 'code/images/0000.png'
    keypoints, descriptors, image_with_keypoints = extract_features(image_path)
    print("Number of keypoints Detected:", len(keypoints))
    # Descriptors are the feature vectors
    print("Feature vectors shape:", descriptors.shape)

    