'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-11 17:00:40
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-07 21:33:34
FilePath: /code/initial_recon.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points

import numpy as np
import cv2
from .feature_extraction import extract_features
from .feature_matching import match_features
import os
from tqdm import tqdm

class CameraPoseEstimator:
    def __init__(self, intrinsic_path, alg = "ransac", save_dir = "output", **kargs):
        self.K = np.loadtxt(intrinsic_path)  # Load camera intrinsic parameters
        self.match_thres = kargs["match"].thres
        self.save_dir = save_dir
        self.params = None
        # https://opencv.org/blog/evaluating-opencvs-new-ransacs/
        if alg == "ransac":
            self.alg = cv2.FM_RANSAC
            self.params = kargs["ransac_params"]
        elif alg == "magsac":
            self.alg = cv2.USAC_MAGSAC
        
        self.contrast_thresh = kargs["extract"].contrast_thresh
        self.edge_thresh = kargs["extract"].edge_thresh
        self.sigma = kargs["extract"].sigma

        self.tree = kargs["match"].tree
        self.checks = kargs["match"].checks
        self.flan_k = kargs["match"].flan_k
        
        self.match_alg = kargs["match"].alg
        self.match_alg_params = kargs["match"].ransac_params
    
    def find_fundamental_essential_matrices(self, points1, points2):
        if self.params is not None:
            F, mask = cv2.findFundamentalMat(points1, points2, self.alg, **self.params)
        else:
            F, mask = cv2.findFundamentalMat(points1, points2, self.alg)
        points1 = points1[mask.ravel() == 1]
        points2 = points2[mask.ravel() == 1]

        E = self.K.T @ F @ self.K
        return F, E

    def find_camera_pose(self, E, points1, points2, K):

        _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

        return R, t
    
    def triangulate_points(self, R, t, R_ref = np.eye(3), t_ref = np.zeros((3,1)), K = None, points1 = None, points2 = None, verbose = False):
        """
        Perform triangulation of point pairs with the projection matrices.
        """
        P1 = K @ np.hstack((R_ref, t_ref))
        P2 = K @ np.hstack((R, t.reshape(3, 1)))

        # Ensure points are 2D and transpose them for cv2.triangulatePoints
        
        if type(points1) == tuple:
            points1 = np.float32([point.pt for point in points1])
            points2 = np.float32([point.pt for point in points2])
        
        if points1.ndim > 1 and points2.ndim > 1:
            points1 = points1.T
            points2 = points2.T

        points_4d_hom = cv2.triangulatePoints(P1, P2, points1, points2) # NOTE: the 3D point are referred to the camera 1 coordinate system
        points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Normalize homogeneous coordinates
        
        if verbose:
            # calculate reprojection error
            cam1_points = P1 @ np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
            cam1_points = cam1_points / cam1_points[2, :]
            cam2_points = P2 @ np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
            cam2_points = cam2_points / cam2_points[2, :]
            
            cam1_error = np.linalg.norm(points1 - cam1_points[:2, :], axis=0)
            cam2_error = np.linalg.norm(points2 - cam2_points[:2, :], axis=0)
            
            print(f"Mean reprojection error for camera 1: {np.mean(cam1_error)}")
            print(f"Mean reprojection error for camera 2: {np.mean(cam2_error)}")
        
        return points_3d.T

    def get_pose_and_3d_points(self, world_cord_path, camera_path_list):
        
        img1 = cv2.imread(world_cord_path)
        pose_save_path = os.path.join(self.save_dir, "camera_pose")
        color_save_path = os.path.join(self.save_dir, "pcolor")
        if not os.path.exists(pose_save_path):
            os.makedirs(pose_save_path, exist_ok=True)
        if not os.path.exists(color_save_path):
            os.makedirs(color_save_path, exist_ok=True)
        poses = []
        world_name = world_cord_path.split("/")[-1].split(".")[0]
        
        for camera_path in tqdm(camera_path_list, total=len(camera_path_list), desc="Processing images {}/{}".format(0, len(camera_path_list))):
            save_name = camera_path.split("/")[-1].split(".")[0]
            if camera_path == world_cord_path:
                pose = (np.eye(4))
                poses.append(pose)
                np.savetxt(os.path.join(pose_save_path, f"{save_name}.txt"), pose, fmt="%f")
                continue
            

            img2 = cv2.imread(camera_path)
            world_id = world_cord_path.split("/")[-1].split(".")[0]
            camera_id = camera_path.split("/")[-1].split(".")[0]

            if img1 is None or img2 is None:
                raise ValueError("One of the images didn't load correctly. Check the paths.")

            features1 = extract_features(img1, image_name=world_id, save_dir=self.save_dir, contrast_thresh=self.contrast_thresh, edge_thresh=self.edge_thresh, sigma = self.sigma)
            features2 = extract_features(img2, image_name=camera_id, save_dir=self.save_dir, contrast_thresh=self.contrast_thresh, edge_thresh=self.edge_thresh, sigma = self.sigma)

            _, _, points1, points2 = match_features(img1, img2, features1, features2, img_name=f"{world_name}-{save_name}", save_dir=self.save_dir, thres = self.match_thres, tree=self.tree, checks=self.checks, flan_k=self.flan_k, alg = self.match_alg, alg_params=self.match_alg_params)
            
  
            points1_ = np.float32([p.pt for p in points1])
            points2_ = np.float32([p.pt for p in points2])

            _, E = self.find_fundamental_essential_matrices(points1_, points2_)
            best_R, best_t = self.find_camera_pose(E, points1_, points2_, self.K)  # relevant R,t
            

            matrix = np.hstack((best_R, best_t.reshape(3, 1)))
            matrix = np.vstack((matrix, np.array([0, 0, 0, 1])))
            # points_3d = self.triangulate_points(best_R, best_t, R_ref=world_R, t_ref=world_t.reshape(3,1), K = self.K, points1=points1, points2=points2)
            points_3d = self.triangulate_points(best_R, best_t, K = self.K, points1=points1, points2=points2)
            
            pair_2D_3D_one = [(points1[i], points_3d[i]) for i in range(len(points_3d))] # 3D-2D pair for image 1
            pair_2D_3D_two = [(points2[i], points_3d[i]) for i in range(len(points_3d))] # 3D-2D pair for image 2
            
            pose = matrix
            poses.append(pose)
            np.savetxt(os.path.join(pose_save_path,f"{save_name}.txt"), pose, fmt='%f')

        return poses, points_3d, pair_2D_3D_one, pair_2D_3D_two
    

def init_recon(world_path = "images/0000.png", camera_paths = ["images/0000.png", "images/0001.png"], save_dir = "output", **kargs):
    intrinsic_path = "camera_intrinsic.txt"
    estimator = CameraPoseEstimator(intrinsic_path, save_dir=save_dir, **kargs)
    
    camera_paths = sorted(camera_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    
    _, points_3d, _, _ = estimator.get_pose_and_3d_points(world_path, camera_paths)
    
    # save point-3d to local
    np.savetxt(os.path.join(save_dir, "init_points_3d.txt"), points_3d, fmt='%f')
    


