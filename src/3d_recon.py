'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-11 17:12:01
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-04-30 17:19:50
FilePath: /code/src/3d_recon.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# perform complete 3D reconstruction from 11 images

from initial_recon import *
from .pnp_recon import *
from .feature_extraction import extract_features
from src.utils import triangulate_points, save_r_t
from tqdm import tqdm
from src.utils import visualize_point_cloud
from .initial_recon import init_recon


if __name__ == "__main__":
    init_recon()
    use_pnp_recon()