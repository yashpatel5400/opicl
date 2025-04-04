from dataset import sample_random_operator, get_spatial_coordinates
import numpy as np

import torch
from FANO import FANO

exp_dict = {
    # data settings
    'split_frac': [{'train': 0.9, 'val': 0.05, 'test': 0.05}],
    'random_state': [0],
    'domain_dim': [2], # 1 for timeseries, 2 for 2D spatial
    'train_sample_rate': [1],
    'test_sample_rates': [[0.5,0.75,1,1.5,2]],
    'test_im_sizes': [[832,624,416,312,208]],
    'test_patch_sizes': [[64,48,32,24,16]],
    'batch_size': [2],
    'dyn_sys_name': ['NavierStokes'], #'darcy_high_res','darcy_discontinuous',
    # optimizer settings
    'learning_rate': [1e-3,1e-4],
    'dropout': [1e-4],
    'lr_scheduler_params': [
                            {'patience': 2, 'factor': 0.5},
                             ],
    'max_epochs': [80],
    'monitor_metric': ['loss/val/mse'],
    
    # model settings (modest model size for debugging)
    'patch': [True],
    'patch_size': [16],
    'modes': [[8,8]],
    'im_size': [64],
    'd_model': [32,64,96],
    'nhead': [8],
    'num_layers': [6],
    'dim_feedforward': [128],
    'activation': ['gelu'],
    'gradient_clip_val':[None] #or 10.0 whenever want to use
}

model_hyperparams = {
    'input_dim': [0],
    'output_dim': [-1],
    'domain_dim': exp_dict["domain_dim"],
    'patch': exp_dict["patch"],
    'patch_size': exp_dict["patch_size"],
    'modes': exp_dict["modes"],
    'im_size': exp_dict["im_size"],
    'd_model': exp_dict["d_model"],
    'nhead': exp_dict["nhead"],
    'num_layers': exp_dict["num_layers"],
    'learning_rate': exp_dict["learning_rate"],
    'dropout': exp_dict["dropout"],
    'dim_feedforward': exp_dict["dim_feedforward"],
    'activation': exp_dict["activation"],
}
model_hyperparams = {key : value[0] for key, value in model_hyperparams.items()}

samples = sample_random_operator(N=model_hyperparams["im_size"])
fs, Ofs = samples

coords_f_x, coords_f_y = get_spatial_coordinates(fs.shape[1])
coords_f = np.array([coords_f_x, coords_f_y])

device = "cuda:0"

T = 5
f_pt = torch.from_numpy(fs[0:T]).unsqueeze(0).to(device).to(torch.float32)
coords_f_pt = torch.from_numpy(coords_f).to(device).to(torch.float32)

fano = FANO(**model_hyperparams).to(device)
with torch.no_grad():
    Of_pred = fano(f_pt, coords_f_pt) # B x H x W x 1
    print(Of_pred.shape)