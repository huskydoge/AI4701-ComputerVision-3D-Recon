'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-11 17:02:10
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-07 14:14:53
FilePath: /code/pnp_recon.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE 
'''
# perform 3D reconstruction using PnP

# now we have the 3D points and camera intrinsic parameters, with additional 9 images

from .initial_recon import *
import numpy as np
from typing import Optional, Tuple
from .feature_extraction import extract_features
from .feature_matching import match_features
from .utils import triangulate_points, save_r_t, visualize_point_cloud, print_colored, find_color_by_pair, get_pos, get_format_result
import cv2
from tqdm import tqdm



class PnP:
    def __init__(self, img1, img2, pairs_2d_3d_for_img1: Optional[Tuple], points1, points2, K = None, save_dir = "output") -> None:
        if type(img1) == "str":
            self.img1 = cv2.imread(img1)
        else:
            self.img1 = img1
        if type(img2) == "str":
            self.img2 = cv2.imread(img2)
        else:
            self.img2 = img2
        
        self.pairs_2d_3d = pairs_2d_3d_for_img1
        self.points1 = points1
        self.points2 = points2
        self.K = K
    
    def _find_pairs_for_img2(self):
        """
        if point in points1 is in pairs_2d_3d, then find the corresponding point in points2
        """
        pairs_2d_3d_for_img2 = []
        for i in range(len(self.points1)):
            for pair in self.pairs_2d_3d:
                if get_pos(self.points1[i]) == get_pos(pair[0]): # pair: (<KeyPoint 0x291c6a640>, array([-0.5110827 ,  0.09887001,  0.9231529 ], dtype=float32))
                    pairs_2d_3d_for_img2.append((get_pos(self.points2[i]), pair[1]))
        return pairs_2d_3d_for_img2
    
    def _get_pose_and_3d_points(self, pairs_2d_3d):
        """
        main implementation for PnP
        using opencv
        Estimates the camera pose using the 2D-3D point correspondences and the camera intrinsic parameters.

        Args:
        pairs_2d_3d (list of tuples): A list where each tuple contains a 2D image point and its corresponding 3D world point.

        Returns:
        success (bool): Whether the pose estimation was successful.
        rvec (np.array): The rotation vector (Rodrigues' rotation formula).
        tvec (np.array): The translation vector.
        """
        
        camera_matrix = self.K
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion 

        # Preparing data for solvePnP
        image_points = np.array([pair[0] for pair in pairs_2d_3d], dtype=np.float32).reshape(-1, 1, 2)
        object_points = np.array([pair[1] for pair in pairs_2d_3d], dtype=np.float32).reshape(-1, 1, 3)

        # Use solvePnP to find the camera pose
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        R, _ = cv2.Rodrigues(rvec)  
        
        
        return success, R, tvec
    
    def get_pose_and_3d_points(self):
        pair = self._find_pairs_for_img2()
        s, r, t = self._get_pose_and_3d_points(pair)
        return s, r, t, pair
    
def use_pnp_recon(intrinsic_path = "camera_intrinsic.txt", save_dir = "output", alg = "ransac", **kargs):
    """_summary_

    Args:
        match_thres (float, optional): _description_. Defaults to 0.28.
        intrinsic_path (str, optional): _description_. Defaults to "camera_intrinsic.txt".
        save_dir (str, optional): _description_. Defaults to "output".
        alg (str, optional): the algorithm used to calculate Fundamental/Essential matrix. Defaults to "ransac".

    Returns:
        _type_: _description_
    """    
    
    K = np.loadtxt(intrinsic_path)
    
    match_thres = kargs["match"].thres
    tree = kargs["match"].tree
    checks = kargs["match"].checks
    flan_k = kargs["match"].flan_k
    match_alg = kargs["match"].alg
    match_alg_params = kargs["match"].ransac_params
    
    contrast_thresh = kargs["extract"].contrast_thresh
    edge_thresh = kargs["extract"].edge_thresh
    sigma = kargs["extract"].sigma

    
    color_save_path = os.path.join(save_dir, "pcolor")
    
    path1 = "images/0001.png"
    path2 = "images/0002.png"
        
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    world_path = "images/0000.png"
    estimator = CameraPoseEstimator(intrinsic_path, thres = match_thres, alg = alg, save_dir=save_dir, **kargs)
    
    # this is used for init_recon, then we should not use this method, but use pnp
    camera_paths = ["images/0001.png"] # init recon with 2 images, normally we should use 0000 and 0001
    _, points_3d, pair_2D_3D_one, pair_2D_3D_two = estimator.get_pose_and_3d_points(world_path, camera_paths)
    
    result_dict = {}
    # result_dict["0000"] = {"points_3d": points_3d, "pair_2D_3D": pair_2D_3D_one, "r": np.ones((3,3)), "t": np.array([0,0,0])}
    pose_save_path = os.path.join(save_dir, "camera_pose")
    img1_r_t = np.loadtxt(os.path.join(pose_save_path,"0001.txt"))

    r_1 = img1_r_t[:3, :3]
    t_1 = img1_r_t[:3, 3].reshape(3, 1)
    
    pcolor = find_color_by_pair(img1, pair_2D_3D_two, os.path.join(color_save_path, f"0001_color.npy"))

    result_dict["0001"] = {"points_3d": points_3d, "pair_2D_3D": pair_2D_3D_two, "r": r_1, "t": t_1, "pcolor": pcolor}
    

    # features1 = extract_features(img1, image_name="0001", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh, sigma=sigma)
    # features2 = extract_features(img2, image_name="0002", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh, sigma=sigma)

    # _, _, points1, points2 = match_features(img1, img2, features1, features2, img_name="0001-0002", thres=match_thres, save_dir=save_dir, tree=tree, checks=checks, flan_k=flan_k)
    
    # # camera_paths = sorted(camera_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    # pnp = PnP(img1, img2, pair_2D_3D_two, points1, points2, K = K)
    
    # _, r_2, t_2, pair_2 = pnp.get_pose_and_3d_points() # the r and t of 0002 camera
    
    # save_r_t(r_2, t_2, os.path.join(pose_save_path, "0002.txt"))
    
    # result_dict["0001"]["r"] = r_1
    # result_dict["0001"]["t"] = t_1

    
    # points_3d_2, pair_2D_3D_one, pair_2D_3D_two = triangulate_points(K, K, points1, points2, r_1, t_1, r_2, t_2)
    
    # # merge the pair from PnP and traiangulate locating

    # # pair_2D_3D_two = pair_2D_3D_two + pair_2
    
    # pcolor = find_color_by_pair(img2, pair_2D_3D_two, os.path.join(color_save_path, f"0002_color.npy"))
    
    # result_dict["0002"] = {"points_3d": points_3d_2, "pair_2D_3D": pair_2D_3D_two, "r": r_2, "t": t_2, "pcolor": pcolor}
    
    
    for i in tqdm(range(1, 10), total = 8, desc="processing img pairs"): # 2,..9, (i, i + 1), for each i, we have its 3D-2D pair, then we match the image i and i + 1, find the 3D-2D pair for i + 1, as well as the r and t for i + 1
        img1 = f"images/{i:04}.png"
        img2 = f"images/{i+1:04}.png"
        features1 = extract_features(img1, image_name=f"{i:04}", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh)
        features2 = extract_features(img2, image_name=f"{i+1:04}", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh)
        pair_2D_3D_img1 = result_dict[f"{i:04}"]["pair_2D_3D"]
        
        _, _, points1, points2 = match_features(img1, img2, features1, features2, img_name=f"{i:04}-{i+1:04}", thres=match_thres, save_dir=save_dir, tree=tree, checks=checks, flan_k=flan_k, alg = match_alg, alg_params=match_alg_params)
        pnp = PnP(img1, img2, pair_2D_3D_img1, points1, points2, K = K)
        
        _, r, t, pair_2 = pnp.get_pose_and_3d_points() # the r and t for i + 1 camera
        
        save_r_t(r, t, os.path.join(pose_save_path, f"{i+1:04}.txt"))
    
        points_3d_2, pair_2D_3D_one, pair_2D_3D_two = triangulate_points(K, K, points1, points2, result_dict[f"{i:04}"]["r"], result_dict[f"{i:04}"]["t"], r, t)
        

        pcolor = find_color_by_pair(img2, pair_2D_3D_two, os.path.join(color_save_path, f"{i+1:04}_color.npy"))
        
        result_dict[f"{i+1:04}"] = {"points_3d": points_3d_2, "pair_2D_3D": pair_2D_3D_two, "r": r, "t": t, "pcolor": pcolor}
    
    for i in tqdm(range(1,9)):
        break
        img1 = f"images/{i:04}.png"
        img2 = f"images/{i+2:04}.png"
        features1 = extract_features(img1, image_name=f"{i:04}", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh)
        features2 = extract_features(img2, image_name=f"{i+2:04}", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh)
        pair_2D_3D_two = result_dict[f"{i:04}"]["pair_2D_3D"]
        
        _, _, points1, points2 = match_features(img1, img2, features1, features2, img_name=f"{i:04}-{i+2:04}", thres=match_thres, save_dir=save_dir, tree=tree, checks=checks, flan_k=flan_k)
        if len(points1) < 10: # too few
            continue
        pnp = PnP(img1, img2, pair_2D_3D_two, points1, points2, K = K)
        
        _, r, t, pair_2 = pnp.get_pose_and_3d_points() # the r and t for i + 2 camera
        
        # save_r_t(r, t, os.path.join(pose_save_path, f"{i+2:04}.txt"))
    
        points_3d_2, pair_2D_3D_one, pair_2D_3D_two = triangulate_points(K, K, points1, points2, result_dict[f"{i:04}"]["r"], result_dict[f"{i:04}"]["t"], r, t)
        
        pcolor = find_color_by_pair(img2, pair_2D_3D_two, os.path.join(color_save_path, f"{i+2:04}_color.npy"))
        
        print("before", len(result_dict[f"{i+2:04}"]["points_3d"]))
        result_dict[f"{i+2:04}"]["points_3d"] = np.concatenate((result_dict[f"{i+2:04}"]["points_3d"], points_3d_2), axis=0)
        result_dict[f"{i+2:04}"]["pcolor"] = np.concatenate((result_dict[f"{i+2:04}"]["pcolor"], pcolor), axis=0)
        print("after", len(result_dict[f"{i+2:04}"]["points_3d"]))
        
    for i in tqdm(range(1,8)):
        break
        img1 = f"images/{i:04}.png"
        img2 = f"images/{i+3:04}.png"
        features1 = extract_features(img1, image_name=f"{i:04}", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh)
        features2 = extract_features(img2, image_name=f"{i+3:04}", save_dir=save_dir, contrast_thresh=contrast_thresh, edge_thresh=edge_thresh)
        pair_2D_3D_two = result_dict[f"{i:04}"]["pair_2D_3D"]
        
        _, _, points1, points2 = match_features(img1, img2, features1, features2, img_name=f"{i:04}-{i+3:04}", thres=match_thres, save_dir=save_dir, tree=tree, checks=checks, flan_k=flan_k)
        if len(points1) < 10: # too few
            continue
        pnp = PnP(img1, img2, pair_2D_3D_two, points1, points2, K = K)
        
        _, r, t, pair_2 = pnp.get_pose_and_3d_points() # the r and t for i + 2 camera
        
        # save_r_t(r, t, os.path.join(pose_save_path, f"{i+3:04}.txt"))

        points_3d_2, pair_2D_3D_one, pair_2D_3D_two = triangulate_points(K, K, points1, points2, result_dict[f"{i:04}"]["r"], result_dict[f"{i:04}"]["t"], r, t)
        
                
        pcolor = find_color_by_pair(img2, pair_2D_3D_two, os.path.join(color_save_path, f"{i+3:04}_color.npy"))
        
        print("before", len(result_dict[f"{i+3:04}"]["points_3d"]))
        result_dict[f"{i+3:04}"]["points_3d"] = np.concatenate((result_dict[f"{i+3:04}"]["points_3d"], points_3d_2), axis=0)
        result_dict[f"{i+3:04}"]["pcolor"] = np.concatenate((result_dict[f"{i+3:04}"]["pcolor"], pcolor), axis=0)
        print("after", len(result_dict[f"{i+3:04}"]["points_3d"]))

        
    # collect 3d points
    all_points = []
    pcolors = []
    for key in result_dict.keys():
        all_points.extend((result_dict[key]["points_3d"]))
        pcolors.extend(result_dict[key]["pcolor"])
        

        
    for key in result_dict.keys():
        item = result_dict[key]
        item["pair_2D_3D"] = [(get_pos(t), k) for (t,k) in item["pair_2D_3D"]]
    
    # save 
    np.save(os.path.join(save_dir,"results_dict"), result_dict)
    
    p3d_set = set()
    all_points_filtered = []
    pcolors_filtered = []
    for i, p in enumerate(all_points):
        if tuple(p) not in p3d_set:
            p3d_set.add(tuple(p))
            all_points_filtered.append(p)
            pcolors_filtered.append(pcolors[i]) # filter
        
    all_points = np.array(all_points_filtered)
    pcolors = np.array(pcolors_filtered)
    

            
    
    r_list = []
    t_list = []
    for pose in os.listdir(pose_save_path):
        r_t = np.loadtxt(os.path.join(pose_save_path, pose))
        r = r_t[:3, :3]
        t = r_t[:3, 3].reshape(3, 1)
        r_list.append(r)
        t_list.append(t)
    
    np.savetxt(os.path.join(save_dir,"all_points.txt"), all_points)
    np.savetxt(os.path.join(save_dir, "pcolors.txt"), pcolors)
    np.save(os.path.join(save_dir,"r_list"), r_list)
    np.save(os.path.join(save_dir,"t_list"), t_list)
    
    get_format_result(os.path.join(save_dir,"all_points.txt"), os.path.join(save_dir,"pcolors.txt"), os.path.join(save_dir,"camera_pose"), save_dir)
    
    print_colored("==================================", "green")
    print_colored(f"collect totall points: {len(all_points)}", "green")
    print_colored("==================================", "green")
    
    return r_list, t_list, all_points, result_dict



if __name__ == "__main__":
    intrinsic_path = "camera_intrinsic.txt"
    
    K = np.loadtxt(intrinsic_path)
    
    path1 = "images/0001.png"
    path2 = "images/0002.png"
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    features1 = extract_features(img1, image_name="0001")
    features2 = extract_features(img2, image_name="0002")
    
    keypoints1, _, _ = features1
    keypoints2, _, _ = features2
    
    _, matches, points1, points2 = match_features(img1, img2, features1, features2)
    
    matches = [m for m,_ in matches]
    
    estimator = CameraPoseEstimator(intrinsic_path)
    
    world_path = "images/0000.png"
    
    camera_paths = ["images/0001.png"] # init recon with 2 images
    # camera_paths = sorted(camera_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    
    poses, points_3d, pair_2D_3D_one, pair_2D_3D_two = estimator.get_pose_and_3d_points(world_path, camera_paths)

    pnp = PnP(img1, img2, pair_2D_3D_two, points1, points2, K = K)
    

    s, r, t = pnp.get_pose_and_3d_points()
    