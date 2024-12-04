import argparse
import einops
import matplotlib.pyplot as plt
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from train import get_dataset_matlab, get_h5_dataset
from opformer import OpFormer

sns.set_theme()

mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
mpl.rcParams['figure.figsize'] = (12,8)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

device = "cuda:0"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Train PDE Solver",
        description="Training script to fit OpFormer in the PDEBench dataset",
        epilog="",
    )
    arg_parser.add_argument(
        "--pde_name", type=str, help="Name of the PDE dataset to fit to"
    )
    args = arg_parser.parse_args()

    if args.pde_name == "ns":
        fn = "ns_data.mat"
        dataset, N, hw = get_dataset_matlab(fn)
    elif args.pde_name == "swe":
        fn = "./data/2D_rdb_NA_NA.h5"
        dataset, N, hw = get_h5_dataset(fn)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = OpFormer(in_chans=1, patch_size=(1,1,1), embed_dim=4, window_size=(8,4,4), num_heads=[1,2,2,2])
    model.to(device)

    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint["model_state_dict"])

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        break

    fig, axs = plt.subplots(3, 2, figsize=(12,18))
    axs[0,0].set_title("Predicted")
    axs[0,1].set_title("Ground Truth")
    for i in range(3):
        axs[i,0].imshow(output[i,0].detach().cpu().numpy())
        axs[i,1].imshow(target[i,0].detach().cpu().numpy())
        
        for j in range(2):
            axs[i,j].grid(visible=False)
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("results.png")
    plt.clf()