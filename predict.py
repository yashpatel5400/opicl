from dataset import sample_random_operator, get_spatial_coordinates
import numpy as np

import torch
from opformer import TransformerOperator

num_layers = 6
im_size    = [128, 64]

samples = sample_random_operator(N=im_size[-1]) # size is 2 w x w (since Z is stack of x y)
fs, Ofs = samples

coords_f_x, coords_f_y = get_spatial_coordinates(fs.shape[1])
coords_f = np.array([coords_f_x, coords_f_y])

Z      = np.concatenate([fs, Ofs], axis=1)
coords = np.concatenate([coords_f, coords_f], axis=1)

device = "cpu"
T = 5
Z_pt = torch.from_numpy(Z[0:T]).unsqueeze(0).to(device).to(torch.float32)
coords_pt = torch.from_numpy(coords).to(device).to(torch.float32)

opformer = TransformerOperator(num_layers=num_layers, im_size=im_size).to(device)
with torch.no_grad():
    Of_pred = opformer(Z_pt, coords_pt) # B x H x W x 1
    print(Of_pred.shape)