import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt

from dataset import sample_random_operator, get_spatial_coordinates
from opformer import TransformerOperator


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()
num_layers = 6
im_size    = [128, 64]

samples = sample_random_operator(N=im_size[-1], sigma_gauss=5.0) # size is 2 w x w (since Z is stack of x y)
fs, Ofs = samples

coords_f_x, coords_f_y = get_spatial_coordinates(fs.shape[1])
coords_f = np.array([coords_f_x, coords_f_y])

Z      = np.concatenate([fs, Ofs], axis=1)
coords = np.concatenate([coords_f, coords_f], axis=1)

device = "cuda"
T = 5
Z_pt = torch.from_numpy(Z[0:T]).unsqueeze(0).to(device).to(torch.float32)
coords_pt = torch.from_numpy(coords).to(device).to(torch.float32)

y_test = Z_pt[:,-1,im_size[0]//2:].clone().cpu().numpy()
Z_pt[:,-1,im_size[0]//2:] = 0

layer_trials       = list(range(25, 50, 5))
errors_matching    = []
errors_nonmatching = []

for num_layers in layer_trials:
    opformer_match    = TransformerOperator(num_layers=num_layers, im_size=im_size, icl_init=True, sigma=5.0).to(device)
    opformer_nonmatch = TransformerOperator(num_layers=num_layers, im_size=im_size, icl_init=True, sigma=1.0).to(device)

    y_hat_icl_match = opformer_match(Z_pt, coords_pt)
    error_icl_match = (y_hat_icl_match.detach().cpu().numpy() - y_test)
    errors_matching.append(np.linalg.norm(error_icl_match))
    
    y_hat_icl_nonmatch = opformer_nonmatch(Z_pt, coords_pt)
    error_icl_nonmatch = (y_hat_icl_nonmatch.detach().cpu().numpy() - y_test)
    errors_nonmatching.append(np.linalg.norm(error_icl_nonmatch))

plt.plot(layer_trials, errors_matching, label="Matching")
plt.plot(layer_trials, errors_nonmatching, label="Non-Matching")
plt.legend()
plt.savefig("results.png")