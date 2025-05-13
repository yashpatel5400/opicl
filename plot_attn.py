# simplified_avg_qk_diff.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import pandas as pd

from opformer import TransformerOperator
import kernels 
from dataset_utils import MetaOperatorDataset_PreGenerated
import config_setup

def compute_avg_crosslayer_diffs(model, normalize=False, k_max=8):
    W_q_layers, W_k_layers = [], []
    min_val_mags, imag_val_mags = [], []
    avg_k_frob_norms = []

    for l in range(len(model.layers)):
        k = model.layers[l].self_attn.key_operator
        w1, w2 = k.weights1.detach().cpu(), k.weights2.detach().cpu()

        W_k_layers.append((w1, w2))
        avg_k_frob_norms.append(torch.norm(w1).item() + torch.norm(w2).item())

        w1_low = w1[:, :, :k_max, :k_max, :].reshape(-1)
        w2_low = w2[:, :, :k_max, :k_max, :].reshape(-1)
        all_vals = torch.cat([w1_low, w2_low])
        min_mag = all_vals.abs().min().item()
        avg_imag = all_vals.imag.abs().mean().item()
        min_val_mags.append(min_mag)
        imag_val_mags.append(avg_imag)

    W_q_layers = [(l.self_attn.query_operator.weights1.detach().cpu(),
                   l.self_attn.query_operator.weights2.detach().cpu()) for l in model.layers]

    def normalize_layer_pair(w1, w2):
        norm1 = torch.norm(w1)
        norm2 = torch.norm(w2)
        return w1 / (norm1 + 1e-8), w2 / (norm2 + 1e-8)

    def avg_pairwise_diff(layers):
        total_diff, count = 0.0, 0
        for i in range(len(layers)):
            for j in range(len(layers)):
                if i == j: continue
                w1_i, w2_i = layers[i]
                w1_j, w2_j = layers[j]
                if normalize:
                    w1_i, w1_j = normalize_layer_pair(w1_i, w1_j)
                    w2_i, w2_j = normalize_layer_pair(w2_i, w2_j)
                total_diff += torch.norm(w1_i - w1_j).item() + torch.norm(w2_i - w2_j).item()
                count += 1
        return total_diff / count if count > 0 else np.nan

    avg_qq_diff = avg_pairwise_diff(W_q_layers)
    avg_kk_diff = avg_pairwise_diff(W_k_layers)

    total_diff, count = 0.0, 0
    for (w1_q, w2_q) in W_q_layers:
        for (w1_k, w2_k) in W_k_layers:
            if normalize:
                w1_q, w1_k = normalize_layer_pair(w1_q, w1_k)
                w2_q, w2_k = normalize_layer_pair(w2_q, w2_k)
            total_diff += torch.norm(w1_q - w1_k).item() + torch.norm(w2_q - w2_k).item()
            count += 1
    avg_qk_diff = total_diff / count if count > 0 else np.nan

    all_diffs = [avg_qk_diff, avg_qq_diff, avg_kk_diff]
    avg_overall_diff = np.mean([d for d in all_diffs if not np.isnan(d)])

    avg_frob_k = np.mean(avg_k_frob_norms)

    return avg_qk_diff, avg_qq_diff, avg_kk_diff, avg_overall_diff, np.mean(min_val_mags), np.mean(imag_val_mags), avg_frob_k

def get_epoch_from_filename(filename):
    match = re.search(r"opformer_epoch_(\d+)\.pth", os.path.basename(filename))
    return int(match.group(1)) if match else -1

def compute_and_save_for_directory(exp_name, normalized=False):
    os.environ["EXPERIMENT_NAME"] = exp_name
    args = config_setup.get_analysis_args(experiment_name=exp_name)
    results_dir = args['results_output_directory']
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(args['checkpoint_directory']):
        print(f"Error: Ckpt dir '{args['checkpoint_directory']}' not found."); return

    kernel_maps = kernels.Kernels(args['im_size'][0], args['im_size'][1])
    ky_kernel_arr = kernel_maps.get_kernel(args['ky_kernel_name'])
    model_params = {
        'num_layers': args['num_layers_in_model'],
        'im_size': args['im_size'],
        'ky_kernel': ky_kernel_arr,
        'kx_name': args['kx_name'],
        'kx_sigma': args['kx_sigma']
    }
    model = TransformerOperator(**model_params, icl_lr=args['fixed_value_op_scalar'], icl_init=False).to(args['analysis_device_str'])

    ckpt_paths = sorted(glob.glob(os.path.join(args['checkpoint_directory'], "opformer_epoch_*.pth")), key=get_epoch_from_filename)
    if not ckpt_paths:
        print(f"Error: No valid ckpts in '{args['checkpoint_directory']}'."); return

    records = []
    for ckpt_path in ckpt_paths:
        epoch = get_epoch_from_filename(ckpt_path)
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict)
            model.eval()
            qk_diff, qq_diff, kk_diff, overall, min_mag, avg_imag, avg_frob_k = compute_avg_crosslayer_diffs(model, normalize=normalized)
            records.append({
                "epoch": epoch,
                "avg_qk_crosslayer_diff": qk_diff,
                "avg_qq_crosslayer_diff": qq_diff,
                "avg_kk_crosslayer_diff": kk_diff,
                "avg_overall_diff": overall,
                "avg_min_real_magnitude": min_mag,
                "avg_imag_component_magnitude": avg_imag,
                "avg_k_frobenius_norm": avg_frob_k
            })
        except Exception as e:
            print(f"Error at epoch {epoch} in {exp_name}: {e}")

    df = pd.DataFrame(records).sort_values("epoch")
    if not df.empty:
        suffix = "normalized" if normalized else "unnormalized"
        csv_path = os.path.join(results_dir, f"avg_Wqk_crosslayer_convergence_{suffix}.csv")
        df.to_csv(csv_path, index=False)

        # Save Frobenius diff plots
        for col, color in zip(["avg_qk_crosslayer_diff", "avg_qq_crosslayer_diff", "avg_kk_crosslayer_diff", "avg_overall_diff"], ["purple", "blue", "green", "black"]):
            plt.plot(df["epoch"], df[col], marker='o', label=col, color=color)
        plt.xlabel("Epoch")
        plt.ylabel("Average Frobenius Norm")
        plt.title(f"{suffix.title()} Operator Diffs: {exp_name}")
        plt.legend()
        plt.grid(True)
        path = os.path.join(results_dir, f"avg_Wqk_crosslayer_convergence_{suffix}.png")
        plt.savefig(path)
        plt.close()

        # Save min real magnitude
        plt.figure(figsize=(10, 5))
        plt.plot(df["epoch"], df["avg_min_real_magnitude"], color="teal", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Min Magnitude of Real Components (low-freq modes)")
        plt.title(f"$|R_{{k,\ell}}|$ Minimum Magnitude: {exp_name}")
        plt.grid(True)
        path = os.path.join(results_dir, f"avg_min_real_magnitude_{suffix}.png")
        plt.savefig(path)
        plt.close()

        # Save imaginary component magnitude
        plt.figure(figsize=(10, 5))
        plt.plot(df["epoch"], df["avg_imag_component_magnitude"], color="orange", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Avg Imaginary Component Magnitude (low-freq modes)")
        plt.title(f"$\Im(R_{{k,\ell}})$ Average: {exp_name}")
        plt.grid(True)
        path = os.path.join(results_dir, f"avg_imag_component_magnitude_{suffix}.png")
        plt.savefig(path)
        plt.close()

        # Save key Frobenius norm
        plt.figure(figsize=(10, 5))
        plt.plot(df["epoch"], df["avg_k_frobenius_norm"], color="gray", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Avg $\\|W_k^{(\\ell)}\\|_F$")
        plt.title(f"Avg Frobenius Norm of Key Matrices: {exp_name}")
        plt.grid(True)
        path = os.path.join(results_dir, f"avg_k_frobenius_norm_{suffix}.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved all plots to {results_dir}")

def aggregate_across_seeds(prefix, results_root="results", output_dir="results/aggregated", normalized=False):
    suffix = "normalized" if normalized else "unnormalized"
    pattern = os.path.join(results_root, f"{prefix}*_kx*_analysis", f"avg_Wqk_crosslayer_convergence_{suffix}.csv")
    csv_paths = glob.glob(pattern)
    if not csv_paths:
        print("No seed result CSVs found for prefix", prefix)
        return

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["seed"] = os.path.basename(os.path.dirname(path))
        dfs.append(df)

    df_all = pd.concat(dfs)
    grouped = df_all.groupby("epoch")

    os.makedirs(output_dir, exist_ok=True)

    for col, ylabel, color in [
        ("avg_overall_diff", "Avg. Frobenius Norm", "black"),
        ("avg_min_real_magnitude", "Min $|R_{k,\ell}|$ (low-freq)", "teal"),
        ("avg_imag_component_magnitude", "Avg $|\\Im(R_{k,\ell})|$ (low-freq)", "orange"),
        ("avg_k_frobenius_norm", "Avg $\\|W_k^{(\\ell)}\\|_F$", "gray")
    ]:
        mean = grouped[col].mean()
        std = grouped[col].std()
        plt.figure(figsize=(10, 5))
        plt.plot(mean.index, mean.values, color=color)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.3, color=color)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Across Seeds [{suffix}]")
        plt.grid(True)
        path = os.path.join(output_dir, f"aggregated_{col}_{suffix}_{prefix}.png")
        plt.savefig(path)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, help="Prefix for experiment subdirectories (e.g. seed_sweep_20250511_120707_seed)")
    parser.add_argument("--aggregate_only", action="store_true", help="Skip per-run computation and just aggregate")
    parser.add_argument("--normalized", action="store_true", help="Compute and plot normalized operator differences")
    args = parser.parse_args()

    if not args.aggregate_only:
        subdirs = [d for d in os.listdir("checkpoints") if d.startswith(args.prefix)]
        for exp_name in subdirs:
            compute_and_save_for_directory(exp_name, normalized=args.normalized)

    aggregate_across_seeds(args.prefix, normalized=args.normalized)
