'''
Author: huskydoge hbh001098hbh@sjtu.edu.cn
Date: 2024-04-30 16:49:24
LastEditors: huskydoge hbh001098hbh@sjtu.edu.cn
LastEditTime: 2024-05-06 22:07:41
FilePath: /code/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import hydra
from omegaconf import DictConfig
from src.initial_recon import init_recon
from src.pnp_recon import use_pnp_recon
from src.bundle_adjustment import bundle_adjustment
from src.utils import wrapped_up_visualize, Logger
from src.epipolar_recon import epipolar_recon
import os
import sys



@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current Output Directory: {output_dir}")
    
    sys.stdout = Logger(os.path.join(output_dir, "log.txt"))

    visualize = cfg.RECON.visualize

    if cfg.RECON.method == "pnp":
        init_recon(world_path=cfg.GLOB.world_cam_path, save_dir=output_dir, **(cfg.RECON_INIT), **(cfg.ESTIMATOR))
        use_pnp_recon(intrinsic_path=cfg.GLOB.intrinsic_path, save_dir=output_dir, **(cfg.RECON), **(cfg.ESTIMATOR))
    elif cfg.RECON.method == "epipolar":
        epipolar_recon(intrinsic_path=cfg.GLOB.intrinsic_path, save_dir=output_dir, **(cfg.ESTIMATOR), **(cfg.RECON))
    else:
        raise ValueError("Invalid reconstruction method")
    
    bundle_adjustment(output_dir, least_square_params = (cfg.BA.least_square_params))
    
    if visualize:
        wrapped_up_visualize(output_dir)
        
    # new_output_dir = f"{output_dir}_{cfg.RECON.method}_match_thres={cfg.RECON.match.thres}_extract_contrast_thresh={cfg.RECON.extract.contrast_thresh}_edge_thresh={cfg.RECON.extract.edge_thresh}_sigma={cfg.RECON.extract.sigma}"
    new_output_dir = f"{output_dir}_{cfg.RECON.method}_alg={cfg.ESTIMATOR.alg}_match_thres={cfg.ESTIMATOR.match.thres}_extract_contrast_thresh={cfg.ESTIMATOR.extract.contrast_thresh}"
    os.rename(output_dir, new_output_dir)

if __name__ == "__main__":
    
    main()
    
    # data_path = "outputs/2024-04-30/21-16-09"
    # data_path = "outputs/2024-04-30/21-03-51"
    
    # wrapped_up_visualize("outputs/2024-04-30/22-14-31")

