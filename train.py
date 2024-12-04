import os
import einops
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset

import argparse
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np

from opformer import OpFormer
from utils import MatlabFileReader

device = "cuda:0"

class PDEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def get_dataset_from_timeseries(raw_data):
    chunk_size = 20
    chunks = torch.stack(torch.split(raw_data, chunk_size, dim=-1), dim=1) # t n1 c h w n2, t = 20, n_2 = 10
    XY_chunks = einops.rearrange(chunks, "n1 n2 c h w t -> n1 n2 c t h w")

    X = XY_chunks[:,:-1,...]
    Y = XY_chunks[:,1:,:,0,:,:]

    X = einops.rearrange(X, "n1 n2 c t h w -> (n1 n2) c t h w")
    Y = einops.rearrange(Y, "n1 n2 c h w -> (n1 n2) c h w")
    return PDEDataset(X, Y), len(X), (Y.shape[-2], Y.shape[-1])


def get_h5_dataset(fn):
    # Read the h5 file and store the data
    h5_file = h5py.File(fn, "r")
    N = len(h5_file.keys())
    raw_data = np.stack([np.array(h5_file[f"{key}/data"], dtype="f") for key in h5_file.keys()]) # N x T x H x W x C
    raw_data = einops.rearrange(raw_data, "n t h w c -> n c h w t")
    raw_data = raw_data[...,:100] # just consider 100 time steps for now
    raw_data = torch.from_numpy(raw_data)
    return get_dataset_from_timeseries(raw_data)


def get_dataset_matlab(fn):
    raw_data_obj = MatlabFileReader(fn)
    raw_data = torch.from_numpy(raw_data_obj.read_file('u')).unsqueeze(1) # N x C x H x W x T
    return get_dataset_from_timeseries(raw_data)


def train(pde_name, model, train_dataloader, val_dataloader):
    criterion = nn.MSELoss()  # Or any appropriate loss function
    optimizer = torch.optim.Adam(model.parameters()) 

    num_epochs = 10
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)}]\tLoss: {loss.item():.6f}')

        # Evaluation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()  # Sum up batch loss
                
        val_loss /= len(val_dataloader)

        print(f'\nValidation set: Average loss: {val_loss:.4f}\n')
        
        results_dir = "results"
        os.makedirs(results_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(results_dir, f'{pde_name}.pth'))

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

    prop_train = 0.8
    N_train = int(prop_train * N)
    train_set, val_set = torch.utils.data.random_split(dataset, [N_train, N - N_train])

    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=4, shuffle=True)

    model = OpFormer(in_chans=1, patch_size=(1,1,1), embed_dim=4, window_size=(8,4,4), num_heads=[1,2,2,2], hw=hw)
    model.to(device)
    train(args.pde_name, model, train_dataloader, val_dataloader)