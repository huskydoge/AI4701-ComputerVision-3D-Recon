'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-29 20:37:30
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-06 20:09:14
FilePath: /code/bundle_adjustment_scipy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE


cameras_params: (n_cameras, 6), 0:3 is rotation vector, 3:6 is translation vector
points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
'''



import numpy as np
from scipy.sparse import lil_matrix
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from scipy.optimize import least_squares
from .utils import draw_pic, print_colored, save_r_t, get_format_result


def project(points, camera_params, fx, fy, k1, k2):
    """Convert 3-D points to 2-D by projecting onto images using distinct focal lengths for each axis."""
 
    t_list = camera_params[:, 3:6].reshape(-1, 3, 1)
    r_list = np.array([cv2.Rodrigues(r)[0] for r in camera_params[:, :3]])
    r_t_list = np.array([np.hstack((r.reshape(3,3), t.reshape(3,1))) for r, t in zip(r_list, t_list)]) # （3，4）

    K = np.array([[fx, 0, k1], [0, fy, k2], [0, 0, 1]])

    P_list = np.array([K @ r_t for r_t in r_t_list])
    
    
    tmp = np.array([np.hstack((point, np.ones(1))) for point in points])
    
    points_proj = np.array([P @ point for P, point in zip(P_list, tmp)])

    projected_points = points_proj[:, :2] / points_proj[:, 2, None] 
    
    # get reprojection error
    
    return projected_points


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, fx, fy, k1, k2):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], fx, fy, k1, k2)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size) # What's the meaning of below
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A


def build_dataset(data_path):
    results_dict = np.load(os.path.join(data_path,"results_dict.npy"), allow_pickle=True).item()
    points_3d = [] # (n_points, 3)
    pcolors = []
    camera_indices = [] # (n_observations, )
    point_indices = [] # (n_observations, )
    points_2d = [] # (n_observations, 2)
    camera_params = []
    
    points_3d_set = set()
    points_3d_index_map = {}
    
    for img in results_dict.keys():
        pair = results_dict[img]['pair_2D_3D']
        pcolor_list = results_dict[img]['pcolor']
        R = results_dict[img]["r"]
        t = results_dict[img]["t"].ravel()
        rvec = cv2.Rodrigues(R)[0].ravel()
        camera_params.append(np.hstack((rvec, t)))
        
        # filter out the same 3D points
        for i, p in tqdm(enumerate(pair), desc=f"Processing {img}"):
            if tuple(p[1]) not in points_3d_set:
                points_3d_set.add(tuple(p[1]))
                points_3d.append(p[1])
                pcolors.append(pcolor_list[i])
                index = len(points_3d) - 1
                points_3d_index_map[tuple(p[1])] = index
                points_2d.append(p[0]) # we dont care the repeat of 2d points, since it's not included in params
                camera_indices.append(int(img) - 1)
                point_indices.append(index)
            else:
                points_2d.append(p[0])
                camera_indices.append(int(img) - 1)
                point_indices.append(points_3d_index_map[tuple(p[1])])
    
    print_colored("==================================", "green")
    print_colored(f"distinct points after filtered: {len(points_3d_set)}", color="green")
    print_colored("==================================", "green")
    
    return np.array(camera_params), np.array(points_3d), np.array(camera_indices), np.array(point_indices), np.array(points_2d), pcolors


def get_intrinsic(datapath):
    intrinsic_path = "camera_intrinsic.txt"
    K = np.loadtxt(intrinsic_path)
    # get k1, k2, f
    
    k1 = K[0, 2]
    k2 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    
    return fx, fy, k1, k2


def bundle_adjustment(datapath, least_square_params = {"method": "trf", "ftol": 1e-7}):
    
    print_colored("\n bundle adjustment... \n", "red")

    fx, fy, k1, k2 = get_intrinsic(datapath)
        
    camera_params, points_3d, camera_indices, point_indices, points_2d, pcolors = build_dataset(data_path=datapath)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    init_points_3d = points_3d  
    init_r_list = [camera_params[i, :3] for i in range(n_cameras)]
    init_R_list = [cv2.Rodrigues(r)[0] for r in init_r_list]
    init_t_list = [camera_params[i, 3:6].reshape(3, 1) for i in range(n_cameras)]
    
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, fx, fy, k1, k2)
    save_path = os.path.join(datapath, "bundle_adjustment")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    draw_pic(f0, "initial error", os.path.join(save_path, "initial_error.png"))
    
   
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    ftol = least_square_params["ftol"]
    method = least_square_params["method"]
    res = least_squares(
        fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=ftol, method=method,
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, fx, fy, k1, k2),
    )
    t1 = time.time()
    
    print_colored("==================================", "green")
    print_colored("Optimization took {0:.0f} seconds".format(t1 - t0), "green")
    print_colored(f"Loss: {res.cost}", "green")
    print_colored("==================================", "green")
    
    draw_pic(res.fun, "final error", os.path.join(save_path, "final_error.png"))
    
    # get the optimized camera parameters and 3D points
    camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
    
    # get r, t
    r_list = []
    t_list = []
    for i in range(n_cameras):
        r = cv2.Rodrigues(camera_params[i, :3])[0]
        t = camera_params[i, 3:6].reshape(3, 1)
        r_list.append(r)
        t_list.append(t)
    
    init_R_list = np.array(init_R_list)
    init_t_list = np.array(init_t_list)
    r_list = np.array(r_list)
    t_list = np.array(t_list)
    
    # get the distance between the optimized and initial 3D points
    print_colored("==================================", "green")
    print_colored("distances between optimized and initial 3D points", "blue")
    print_colored("point 3d diff: {}".format(np.linalg.norm(init_points_3d - points_3d)), "blue")
    print_colored("r_list diff: {}".format(np.linalg.norm(init_R_list - r_list)), "blue")
    print_colored("t_list diff: {}".format(np.linalg.norm(init_t_list - t_list)), "blue")
    print_colored("==================================", "green")
    
    
        
    # add the cam0 to R_hat and t_hat, since we don't optimize the params of cam0 in BA
    R_0 = np.expand_dims(np.eye(3,3), axis=0)
    t_0 = np.expand_dims(np.zeros((3,1)), axis = 0)
    
    
        
    r_list = np.append(R_0, r_list, axis=0)
    t_list = np.append(t_0, t_list, axis=0)
    
    if not os.path.exists(os.path.join(save_path, "camera_pose")):
        os.makedirs(os.path.join(save_path, "camera_pose"), exist_ok = True)
    for i in range(len(r_list)):
        save_r_t(r_list[i], t_list[i], path = os.path.join(save_path, "camera_pose", f"{i:04}.txt"))
        

    np.save(os.path.join(save_path, "R_hat"), r_list)
    np.save(os.path.join(save_path,"t_hat"), t_list)
    np.savetxt(os.path.join(save_path,"p3d_hat.txt"), points_3d)
    np.savetxt(os.path.join(save_path, "pcolors.txt"), pcolors)
    
    get_format_result(os.path.join(save_path, "p3d_hat.txt"), os.path.join(save_path,"pcolors.txt"), os.path.join(save_path, "camera_pose"), save_path)

if __name__ == "__main__":
    bundle_adjustment("output")

    