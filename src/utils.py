'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-19 19:18:47
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-07 22:48:18
FilePath: /code/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from termcolor import colored
import os
import sys

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def print_colored(message, color='white'):
    print(colored(message, color))


def find_color_by_pair(img, pair_2D_3D, save_path):
    # Check if the results already exist

    if isinstance(img, str):
        img = cv2.imread(img)
    res = []
    for pair in pair_2D_3D:
        p2d = get_pos(pair[0])
        p3d = tuple(pair[1])  # Corresponding 3D point
    
        # Check if the coordinate is within the image bounds
        if p2d[1] < img.shape[0] and p2d[0] < img.shape[1]:
            # OpenCV stores color images in BGR format
            bgr_color = img[round(p2d[1]), round(p2d[0])]
            # You can store the color in BGR format, or convert it to RGB if needed
            rgb_color = bgr_color[::-1]

            # Store the 3D point and its corresponding color
            res.append(rgb_color)
        else:
            print(f"Warning: 2D point {p2d} is out of image bounds.")

    return res

def get_pos(point):
    if isinstance(point, cv2.KeyPoint):
        return point.pt
    else:
        return point


def triangulate_points(K1, K2, points1, points2, R1,t1,R2,t2, verbose = False):
    """
    Perform triangulation of point pairs with the projection matrices.
    """
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    
    # Ensure points are 2D and transpose them for cv2.triangulatePoints
    
    if type(points1) == tuple:
        points1_ = np.float32([point.pt for point in points1])
        points2_ = np.float32([point.pt for point in points2])
        
    if points1_.ndim > 1 and points2_.ndim > 1:
        points1_ = points1_.T
        points2_ = points2_.T
        
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1_, points2_)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Normalize homogeneous coordinates
    
    points_3d = points_3d.T
    
    pair_2D_3D_one = [(points1[i].pt, points_3d[i]) for i in range(len(points_3d))] # 3D-2D pair for image 1
    pair_2D_3D_two = [(points2[i].pt, points_3d[i]) for i in range(len(points_3d))] # 3D-2D pair for image 2
    
    
    if verbose:
        # calculate reprojection error

        cam1_points = P1 @ np.vstack((points_3d.T, np.ones((1, points_3d.T.shape[1]))))

        cam1_points = cam1_points / cam1_points[2, :]
        cam2_points = P2 @ np.vstack((points_3d.T, np.ones((1, points_3d.T.shape[1]))))
        cam2_points = cam2_points / cam2_points[2, :]
        
        cam1_error = np.linalg.norm(points1_ - cam1_points[:2, :], axis=0)
        cam2_error = np.linalg.norm(points2_ - cam2_points[:2, :], axis=0)
        
        print(f"Mean reprojection error for camera 1: {np.mean(cam1_error)}")
        print(f"Mean reprojection error for camera 2: {np.mean(cam2_error)}")
    
    return points_3d, pair_2D_3D_one, pair_2D_3D_two

def save_r_t(r, t, path):
    """_summary_

    Args:
        r (_type_): 3x3 rotation matrix.
        t (_type_): 3x1 translation vector.
        return 4x4 transformation matrix.
    """
    
    bottom = np.array([0, 0, 0, 1])
    
    matrix = np.hstack([r, t])
    matrix = np.vstack([matrix, bottom])
    
    np.savetxt(path, matrix, fmt="%f")
    
def create_camera_mesh(color = None):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    if color is not None:
        mesh.paint_uniform_color(color)
    return mesh
    

def visualize_point_cloud(points_3d, r_list, t_list, pcolors=None, view_params = None):
    print_colored("========================================", "green")
    print_colored("Visualizing the point cloud and camera poses...", "green")
    print_colored("Total number of points: {}".format(len(points_3d)), "green")
    print_colored("Number of camera poses: {}".format(len(r_list)), "green")
    print_colored("========================================", "green")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Recon", height=1000, width = 2000)    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)  
    
    if pcolors is not None:
        pcolors = np.array(pcolors)
        pcolors = pcolors / 255
        pcd.colors = o3d.utility.Vector3dVector(pcolors)
    else:
        z_coords = points_3d[:, 2]
        colors = plt.get_cmap('viridis')((z_coords - z_coords.min()) / (z_coords.max() - z_coords.min()))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    opt = vis.get_render_option()

    opt.point_size = 3
    opt.show_coordinate_frame = True
    opt.point_show_normal = True

    opt.background_color = np.array([0, 0, 0])  # black



    for R, t in zip(r_list, t_list):
        cam = create_camera_mesh()
        R_t = np.vstack((np.hstack((R, t.reshape(3,1))), np.array([0,0,0,1])))

        # R_ = R_t_inv[:3,:3]
        R_ = R_t[:3,:3].T
        # t_ = R_t_inv[:3, 3]
        t_ = - R_ @ R_t[:3, 3]
        cam.rotate(R_, center=[0, 0, 0])
        cam.translate(t_)
        vis.add_geometry(cam)
    
    def create_grid(xyz, size, n):
        # Creates a set of lines that form a grid in the xz plane centered at 'xyz'
        lines = []
        colors = []
        for i in range(-n, n + 1):
            lines.append([xyz + np.array([i * size, 0, -n * size]), xyz + np.array([i * size, 0, n * size])])
            lines.append([xyz + np.array([-n * size, 0, i * size]), xyz + np.array([n * size, 0, i * size])])
            colors.append([0.5, 0.5, 0.5])
            colors.append([0.5, 0.5, 0.5])
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array([line for pair in lines for line in pair])),
            lines=o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(0, len(lines) * 2, 2)])),
        )
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
        return line_set

    grid = create_grid(np.array([0, -0.1, 0]), size=100, n=1000)  # 'size' is the gap between grid，'n' is the number of grid
    vis.add_geometry(grid)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

def wrapped_up_visualize(data_path):
        
    all_points = np.loadtxt(f"{data_path}/all_points.txt")
    r_list = np.load(f"{data_path}/r_list.npy")
    t_list = np.load(f"{data_path}/t_list.npy")
    pcolors = np.loadtxt(f"{data_path}/pcolors.txt")
    visualize_point_cloud(all_points, r_list, t_list, pcolors)
    

    ba_path = f"{data_path}/bundle_adjustment"
    R_hat = np.load(ba_path + "/R_hat.npy")
    t_hat = np.load(ba_path + "/t_hat.npy")
    
    # add the cam0 to R_hat and t_hat, since we don't optimize the params of cam0 in BA
    R_0 = np.expand_dims(np.eye(3,3), axis=0)
    t_0 = np.expand_dims(np.zeros((3,1)), axis = 0)
    
    R_hat = np.append(R_hat, R_0, axis=0)
    t_hat = np.append(t_hat, t_0, axis=0)
    
    p3d_hat = np.loadtxt(ba_path + "/p3d_hat.txt")
    pcolors = np.loadtxt(ba_path + "/pcolors.txt")
    visualize_point_cloud(p3d_hat, R_hat, t_hat, pcolors)

def get_cloud_density(points_3d):
    """
    Get the density of the point cloud by calculating the average distance between each point and its nearest neighbors.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    distances = []
    for i in range(len(points_3d)):
        [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        distances.append(k[1])
    
    return np.mean(distances)

def draw_pic(data, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    
def get_format_result(all_points_path, pcolors_path, camera_pose_path, save_path):
    all_points = np.loadtxt(all_points_path)
    pcolors = np.loadtxt(pcolors_path)
    
    # convert to .ply format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    if pcolors is not None:
        pcolors = np.array(pcolors)
        pcolors = pcolors / 255
        pcd.colors = o3d.utility.Vector3dVector(pcolors)
    
    o3d.io.write_point_cloud(os.path.join(save_path, "all_points.ply"), pcd)
    
    camera_list = []
    cam_txts = sorted(os.listdir(camera_pose_path),key = lambda x: int(x.split(".")[0]))
    for cam in cam_txts:
        if cam.split(".")[-1] == "txt":
            cam_params = np.loadtxt(os.path.join(camera_pose_path, cam)).flatten()
            camera_list.append(cam_params)
    
    camera_list = np.array(camera_list)
    
    np.savetxt(os.path.join(save_path, "flatted_cam.txt"), camera_list)
    
    return 
    

def get_diff_between_cameras(path1, path2):
    """
    get the l2 norm of two set of cameras (each has 0000 to 0010)
    """

    camera_list_1 = []
    camera_list_2 = []
    p1 = os.listdir(path1)
    p2 = os.listdir(path2)
    p1 = sorted(p1, key = lambda x: int(x.split(".")[0]))
    p2 = sorted(p2, key = lambda x: int(x.split(".")[0]))
    
    for cam in p1:
        if cam.split(".")[-1] == "txt":
            tmp = np.loadtxt(os.path.join(path1, cam))
            camera_list_1.append(tmp)
            
    for cam in p2:
        if cam.split(".")[-1] == "txt":
            tmp = np.loadtxt(os.path.join(path2, cam))
            camera_list_2.append(tmp)
    camera_list_1 = np.array(camera_list_1)
    camera_list_2 = np.array(camera_list_2)
    diff = np.linalg.norm(camera_list_1 - camera_list_2)
    print("3 * 4 norm diff: ", diff)

    rotation_diffs = []
    angular_diffs = []
    t_diffs = []
    for i in range(len(camera_list_1)):
        cam_1 = camera_list_1[i]
        cam_2 = camera_list_2[i]

        rotation_diff = cam_1[:3, :3] @ cam_2[:3, :3].T

        rotation_diffs.append(np.linalg.norm((rotation_diff - np.eye(3,3)) / np.linalg.norm(np.eye(3,3))))
        
        trace = np.clip(np.trace(rotation_diff), - np.inf, 3)
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        t_diff = (np.linalg.norm(cam_1[:3, 3] - cam_2[:3, 3], ord=1))
        
        angular_diffs.append(angular_distance)
        t_diffs.append(t_diff)
        
    # Create a figure
    fig = plt.figure(figsize=(12, 6))

    # Add subplots manually
    ax1 = fig.add_subplot(2, 2, 1)  # Top left
    ax2 = fig.add_subplot(2, 2, 2)  # Top right
    ax3 = fig.add_subplot(2, 1, 2)  # Bottom, spanning the full width

    # Plot rotation difference on the top left
    ax1.plot(rotation_diffs)
    ax1.set_title("Rotation Difference / Each Camera")
    ax1.set_xlabel("Camera Index")
    ax1.set_ylabel("Rotation Norm Difference (Frobenius Norm)")

    # Plot translation difference on the top right
    ax2.plot(t_diffs)
    ax2.set_title("Translation Difference / Each Camera")
    ax2.set_xlabel("Camera Index")
    ax2.set_ylabel("Translation Norm Difference (units)")

    # Plot angular difference on the bottom
    ax3.plot(angular_diffs)
    ax3.set_title("Angular Difference / Each Camera")
    ax3.set_xlabel("Camera Index")
    ax3.set_ylabel("Angular Difference (degrees)")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def compare_cams(r_list_1, t_list_1, r_list_2, t_list_2):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Recon", height=1000, width = 2000)    
    
    opt = vis.get_render_option()

    opt.show_coordinate_frame = True
    opt.point_show_normal = True

    opt.background_color = np.array([0, 0, 0])  # black

    for R, t in zip(r_list_1, t_list_1):
        cam = create_camera_mesh(color=[0,0,1]) # Blue for cam1
        R_t = np.vstack((np.hstack((R, t.reshape(3,1))), np.array([0,0,0,1])))

        # R_ = R_t_inv[:3,:3]
        R_ = R_t[:3,:3].T
        # t_ = R_t_inv[:3, 3]
        t_ = - R_ @ R_t[:3, 3]
        cam.rotate(R_, center=[0, 0, 0])
        cam.translate(t_)
        vis.add_geometry(cam)
    
    for R, t in zip(r_list_2, t_list_2):
        cam = create_camera_mesh(color=[0,1,0]) # Green for cam2
        R_t = np.vstack((np.hstack((R, t.reshape(3,1))), np.array([0,0,0,1])))

        # R_ = R_t_inv[:3,:3]
        R_ = R_t[:3,:3].T
        # t_ = R_t_inv[:3, 3]
        t_ = - R_ @ R_t[:3, 3]
        cam.rotate(R_, center=[0, 0, 0])
        cam.translate(t_)
        vis.add_geometry(cam)
    
    def create_grid(xyz, size, n):
        # Creates a set of lines that form a grid in the xz plane centered at 'xyz'
        lines = []
        colors = []
        for i in range(-n, n + 1):
            lines.append([xyz + np.array([i * size, 0, -n * size]), xyz + np.array([i * size, 0, n * size])])
            lines.append([xyz + np.array([-n * size, 0, i * size]), xyz + np.array([n * size, 0, i * size])])
            colors.append([0.5, 0.5, 0.5])
            colors.append([0.5, 0.5, 0.5])
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array([line for pair in lines for line in pair])),
            lines=o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(0, len(lines) * 2, 2)])),
        )
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
        return line_set



    grid = create_grid(np.array([0, -0.1, 0]), size=100, n=1000) 
    vis.add_geometry(grid)

    vis.run()
    vis.destroy_window()
        
    
if __name__ == "__main__":

    
    intrinsics = np.loadtxt("camera_intrinsic.txt")

    ba_path = "outputs/pnp_alg=magsac_match_thres=0.5_extract_contrast_thresh=0.001/bundle_adjustment"
    pnp_path = "outputs/pnp_alg=magsac_match_thres=0.5_extract_contrast_thresh=0.001"
    epi_ba_path = "outputs/epipolar_alg=ransac_match_thres=0.5_extract_contrast_thresh=0.001/bundle_adjustment"
    epi_path = "outputs/epipolar_alg=ransac_match_thres=0.5_extract_contrast_thresh=0.001"
    
    
    r_list_1 = np.load(os.path.join(ba_path, "R_hat.npy"))
    t_list_1 = np.load(os.path.join(ba_path, "t_hat.npy"))
    
    r_list_2 = np.load(os.path.join(epi_path, "r_list.npy"))
    t_list_2 = np.load(os.path.join(epi_path, "t_list.npy"))  
    
    compare_cams(r_list_1, t_list_1, r_list_2, t_list_2)
    
    get_diff_between_cameras(path1 = f"{ba_path}/camera_pose", path2 = f"{epi_path}/camera_pose")
    # get_diff_between_cameras(path1 = f"{epi_ba_path}/camera_pose", path2 = f"{epi_path}/camera_pose")
    # get_diff_between_cameras(path1 = f"{ba_path}/camera_pose", path2 = f"{epi_ba_path}/camera_pose")
    