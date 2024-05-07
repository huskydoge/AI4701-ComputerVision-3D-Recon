'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-30 22:21:33
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-06 20:08:54
FilePath: /code/src/epipolar_recon.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import cv2
import open3d as o3d
from .feature_extraction import extract_features
from .feature_matching import match_features
import os
from tqdm import tqdm
from .utils import triangulate_points, save_r_t, visualize_point_cloud, print_colored, find_color_by_pair, get_format_result
from .initial_recon import CameraPoseEstimator

def epipolar_recon(intrinsic_path = "camera_intrinsic.txt", save_dir = "output", **kargs):
      
    match_thres = kargs["match"].thres
    tree = kargs["match"].tree
    checks = kargs["match"].checks
    flan_k = kargs["match"].flan_k
    
    K = np.loadtxt(intrinsic_path)
    
    contrast_thresh = kargs["extract"].contrast_thresh
    edge_thresh = kargs["extract"].edge_thresh
    sigma = kargs["extract"].sigma
    
    normalize = kargs["normalize_epi"]
    
    estimator = CameraPoseEstimator(intrinsic_path, thres = match_thres, save_dir=save_dir, **kargs)


    pose_save_path = os.path.join(save_dir, "camera_pose")
    color_save_path = os.path.join(save_dir, "pcolor")
    
    if not os.path.exists(pose_save_path):
        os.makedirs(pose_save_path, exist_ok=True)
    if not os.path.exists(color_save_path):
        os.makedirs(color_save_path, exist_ok=True)
    poses = []
    
    scale_t = 1
    
    result_dict = {}
    for i in tqdm(range(0, 10), desc="processing img pairs"): # 2,..9, (i, i + 1), for each i, we have its 3D-2D pair, then we match the image i and i + 1, find the 3D-2D pair for i + 1, as well as the r and t for i + 1
        img = f"images/{i:04}.png"
        if i == 0:
            img_list = [f"images/{i:04}.png", f"images/{i+1:04}.png"] # get the original pose
        else:
            img_list = [f"images/{i+1:04}.png"] # get the original pose
        
        _, points_3d, pair_2D_3D_one, pair_2D_3D_two = estimator.get_pose_and_3d_points(img, img_list)
        
        img_r_t_prev = np.loadtxt(os.path.join(pose_save_path,f"{i:04}.txt"))
        
        img_r_t = np.loadtxt(os.path.join(pose_save_path,f"{i+1:04}.txt"))
        
                
        if i == 1:
            scale_t = np.linalg.norm(img_r_t_prev[:3, 3].reshape(3, 1))
            print("\nscale: ", scale_t)
        
        if normalize:
            print("\nnormalize:", np.linalg.norm(img_r_t[:3, 3]))
            img_r_t[:3, 3] = img_r_t[:3, 3] * scale_t / np.linalg.norm(img_r_t[:3, 3])

        r_prev = img_r_t_prev[:3, :3]
        t_prev = img_r_t_prev[:3, 3].reshape(3,1)
        
        r = img_r_t[:3, :3] @ r_prev # Correct?
        
        t = img_r_t[:3, 3].reshape(3, 1) + img_r_t[:3, :3] @ t_prev # Correct?

            
        save_r_t(r, t, os.path.join(pose_save_path,f"{i+1:04}.txt"))

        tran_points_3d = [(np.linalg.inv(r_prev) @ (p.reshape(3,1) - t_prev)).flatten() for p in points_3d]  # Correct?

        points_3d = tran_points_3d
        
        tran_pair_2D_3D_two = []
        for pair in pair_2D_3D_two:
            tran_pair_2D_3D_two.append((pair[0], (np.linalg.inv(r_prev) @ (pair[1].reshape(3,1) - t_prev)).flatten()))

        pair_2D_3D_two = tran_pair_2D_3D_two
        
        pcolor = find_color_by_pair(img, pair_2D_3D_two, os.path.join(color_save_path, f"{i+1:04}_color.npy"))
        
        result_dict[f"{i+1:04}"] = {"points_3d": points_3d, "pair_2D_3D": pair_2D_3D_two, "r": r, "t": t, "pcolor": pcolor}

        save_r_t(r, t, os.path.join(pose_save_path, f"{i+1:04}.txt"))
    
       
    # collect 3d points
    all_points = []
    pcolors = []
    for key in result_dict.keys():
        all_points.extend((result_dict[key]["points_3d"]))
        pcolors.extend(result_dict[key]["pcolor"])
        
    for key in result_dict.keys():
        item = result_dict[key]
        item["pair_2D_3D"] = [(t.pt, k) for (t,k) in item["pair_2D_3D"]]
    
    # save 
    np.save(os.path.join(save_dir,"results_dict"), result_dict)
        
    all_points = np.array(all_points)
    pcolors = np.array(pcolors)
    
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
    
    get_format_result(os.path.join(save_dir,"all_points.txt"),os.path.join(save_dir,"pcolors.txt"), os.path.join(save_dir,"camera_pose"), save_dir)
    
    print_colored("==================================", "green")
    print_colored(f"collect totall points: {len(all_points)}", "green")
    print_colored("==================================", "green")
        
    return r_list, t_list, all_points, result_dict
    
