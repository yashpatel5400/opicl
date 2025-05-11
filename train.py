# train_ddp_refactored.py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import re 
import glob 

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from opformer import TransformerOperator 
import kernels 
from dataset_utils import MetaOperatorDataset_PreGenerated # Import from new utils file
import config_setup # Import the new config file

global_rank = 0 # For conditional prints if needed anywhere (like dataset init)

def setup_ddp(rank, world_size): # As before
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_ADDR'] = 'localhost'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp(): dist.destroy_process_group() # As before

def get_epoch_from_filename(filename): # As before
    match = re.search(r"opformer_epoch_(\d+)\.pth", os.path.basename(filename))
    return int(match.group(1)) if match else -1

def find_latest_checkpoint(checkpoint_dir): # As before
    if not os.path.isdir(checkpoint_dir): return None, -1
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "opformer_epoch_*.pth"))
    if not checkpoint_files: return None, -1
    latest_checkpoint_file = max(checkpoint_files, key=get_epoch_from_filename)
    latest_epoch = get_epoch_from_filename(latest_checkpoint_file)
    return (latest_checkpoint_file, latest_epoch) if latest_epoch != -1 else (None, -1)

def get_dataloader_ddp_refactored(rank, world_size, args):
    kernel_maps = kernels.Kernels(args['im_size'][0], args['im_size'][1])
    ky_kernel_array = kernel_maps.get_kernel(args['ky_kernel_name'])

    dataset_instance = MetaOperatorDataset_PreGenerated(
        num_total_icl_tasks=args['num_total_icl_tasks_in_dataset'], 
        kx_name=args['kx_name'], ky_kernel_array=ky_kernel_array,
        im_size=args['im_size'], 
        num_incontext_prompts=args['num_incontext_prompts'],
        num_operator_bases=args['num_operator_bases'],
        fixed_operator_seed=args.get('fixed_operator_seed'),
        fixed_prompts_seed=args.get('fixed_prompts_seed')
    )
    sampler = DistributedSampler(dataset_instance, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset_instance, batch_size=args['per_gpu_actual_batch_size'], 
                            shuffle=False, num_workers=max(0,min(4, os.cpu_count()//world_size if world_size > 0 else 2)),
                            sampler=sampler, pin_memory=True)
    return dataloader

def log_predictions_refactored(Z_cpu, Of_cpu, model_module, epoch, rank, args): # Takes args
    # ... (log_predictions essentially as before, using args['log_dir'])
    if rank != 0: return
    model_module.eval()
    with torch.no_grad():
        device = next(model_module.parameters()).device; Z_dev = Z_cpu.to(device)
        preds, _, _ = model_module(Z_dev)
        preds_np, targets_np = preds.cpu().numpy(), Of_cpu.cpu().numpy()
        for i in range(min(args.get('log_max_samples',2), Z_cpu.shape[0])):
            plt.figure(figsize=(12,4)); # ... (subplots as before) ...
            plt.savefig(os.path.join(args['log_dir'], f"E{epoch}_S{i}.png")); plt.close()


def train_ddp(rank, world_size, args):
    global global_rank; global_rank = rank
    print(f"R{rank}: Starting DDP training. PID: {os.getpid()}")
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank) 

    writer = None
    if rank == 0:
        os.makedirs(args['tensorboard_log_dir'], exist_ok=True)
        writer = SummaryWriter(log_dir=args['tensorboard_log_dir'])
        print(f"R0: TensorBoard logs: {args['tensorboard_log_dir']}")

    kernel_maps = kernels.Kernels(args['im_size'][0], args['im_size'][1])
    ky_true_kernel_array = kernel_maps.get_kernel(args['ky_kernel_name'])
    
    model = TransformerOperator(
        num_layers=args['num_layers'], im_size=args['im_size'], ky_kernel=ky_true_kernel_array, 
        kx_name=args['kx_name'], kx_sigma=args['kx_sigma'],
        icl_lr=args['icl_lr'], icl_init=args['icl_init'],  
    ).to(rank)

    if args.get('optimizer_type', 'adam').lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args.get('sgd_momentum', 0.0))
    else: optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    start_epoch = 0; resumed_losses = []
    # ... (Checkpoint loading logic as before, using args['checkpoint_dir'] and args['resume_from_checkpoint']) ...
    # ... (Make sure to load optimizer state correctly if resuming) ...
    
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=args.get('ddp_find_unused_parameters', False))
    dataloader = get_dataloader_ddp_refactored(rank, world_size, args)
    losses = resumed_losses

    if rank == 0:
        print(f"R0: Starting training from epoch {start_epoch} for {args['epochs']} total epochs.")

        # ✅ Save the untrained model before starting training (Epoch 0)
        init_path = os.path.join(args['checkpoint_dir'], f"opformer_epoch_0.pth")
        torch.save({
            'epoch': 0,
            'model_state_dict': ddp_model.module.state_dict(),
            'optimizer_state_dict': None,
            'losses': [],
            'optimizer_type_saved': args.get('optimizer_type', 'adam').lower()
        }, init_path)
        print(f"✅ Initial model saved before training at: {init_path}")

    for epoch in range(start_epoch, args['epochs']):
        if rank == 0: print(f"Epoch {epoch} starting...")
        dataloader.sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0; num_batches = 0
        for step, (Z_batch_cpu, Of_batch_cpu) in enumerate(dataloader):
            Z_gpu = Z_batch_cpu.to(rank, non_blocking=True); Of_gpu = Of_batch_cpu.to(rank, non_blocking=True)
            preds, _, _ = ddp_model(Z_gpu)
            loss = F.mse_loss(preds + Of_gpu, torch.zeros_like(preds))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()

            if writer: writer.add_scalar('Loss/train_step', loss.item(), epoch * len(dataloader) + step)
            epoch_loss_sum += loss.item(); num_batches += 1
        
        avg_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
        loss_tensor = torch.tensor(avg_loss).to(rank); dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        global_avg_loss = loss_tensor.item()

        if rank == 0:
            losses.append(global_avg_loss); print(f"Epoch {epoch} | Global Avg Loss: {global_avg_loss:.6f}")
            if writer: writer.add_scalar('Loss/train_epoch', global_avg_loss, epoch)
            if epoch % args['log_freq'] == 0: log_predictions_refactored(Z_batch_cpu, Of_batch_cpu, ddp_model.module, epoch, rank, args)
            if epoch % args['save_freq'] == 0:
                m_path = os.path.join(args['checkpoint_dir'], f"opformer_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': ddp_model.module.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'losses': losses,
                    'optimizer_type_saved': args.get('optimizer_type', 'adam').lower()
                }, m_path)
                print(f"💾 Model saved: {m_path}")

    if rank == 0 and writer: writer.close(); print("TB writer closed.")
    # ... (Matplotlib loss plotting if args['plot_loss_matplotlib']) ...
    cleanup_ddp()

if __name__ == "__main__":
    experiment_name = os.environ.get("EXPERIMENT_NAME", "default_experiment")
    args = config_setup.get_training_args(experiment_name=experiment_name)

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs for experiment '{args['checkpoint_dir']}'.")
    if world_size == 0 : print("No GPUs found."); exit()
    # Create directories on rank 0 before spawn, or handle creation carefully within each process for DDP
    if not os.path.exists(args['checkpoint_dir']): os.makedirs(args['checkpoint_dir'], exist_ok=True)
    if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'], exist_ok=True)
    if not os.path.exists(args['tensorboard_log_dir']): os.makedirs(args['tensorboard_log_dir'], exist_ok=True)

    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)