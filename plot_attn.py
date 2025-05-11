# simplified_avg_qk_diff.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import pandas as pd
import seaborn as sns

from opformer import TransformerOperator
import kernels 
from dataset_utils import MetaOperatorDataset_PreGenerated
import config_setup

def compute_avg_crosslayer_diffs(model):
    W_q_layers, W_k_layers = [], []

    for l in range(len(model.layers)):
        q = model.layers[l].self_attn.query_operator
        k = model.layers[l].self_attn.key_operator

        W_q_layers.append((q.weights1.detach().cpu(), q.weights2.detach().cpu()))
        W_k_layers.append((k.weights1.detach().cpu(), k.weights2.detach().cpu()))

    def avg_pairwise_diff(layers):
        total_diff, count = 0.0, 0
        for i in range(len(layers)):
            for j in range(len(layers)):
                if i == j: continue
                w1_i, w2_i = layers[i]
                w1_j, w2_j = layers[j]
                total_diff += torch.norm(w1_i - w1_j).item() + torch.norm(w2_i - w2_j).item()
                count += 1
        return total_diff / count if count > 0 else np.nan

    avg_qq_diff = avg_pairwise_diff(W_q_layers)
    avg_kk_diff = avg_pairwise_diff(W_k_layers)

    total_diff, count = 0.0, 0
    for (w1_q, w2_q) in W_q_layers:
        for (w1_k, w2_k) in W_k_layers:
            total_diff += torch.norm(w1_q - w1_k).item() + torch.norm(w2_q - w2_k).item()
            count += 1
    avg_qk_diff = total_diff / count if count > 0 else np.nan

    all_diffs = [avg_qk_diff, avg_qq_diff, avg_kk_diff]
    avg_overall_diff = np.mean([d for d in all_diffs if not np.isnan(d)])

    return avg_qk_diff, avg_qq_diff, avg_kk_diff, avg_overall_diff

def get_epoch_from_filename(filename):
    match = re.search(r"opformer_epoch_(\d+)\.pth", os.path.basename(filename))
    return int(match.group(1)) if match else -1

def compute_and_save_for_directory(exp_name):
    os.environ["EXPERIMENT_NAME"] = exp_name
    args = config_setup.get_analysis_args(experiment_name=exp_name)
    results_dir = args['results_output_directory']
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(args['checkpoint_directory']):
        print(f"Error: Ckpt dir '{args['checkpoint_directory']}' not found.")
        return

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
        print(f"Error: No valid ckpts in '{args['checkpoint_directory']}'.")
        return

    records = []
    for ckpt_path in ckpt_paths:
        epoch = get_epoch_from_filename(ckpt_path)
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict)
            model.eval()
            qk_diff, qq_diff, kk_diff, overall = compute_avg_crosslayer_diffs(model)
            records.append({"epoch": epoch, "avg_qk_crosslayer_diff": qk_diff, "avg_qq_crosslayer_diff": qq_diff, "avg_kk_crosslayer_diff": kk_diff, "avg_overall_diff": overall})
        except Exception as e:
            print(f"Error at epoch {epoch} in {exp_name}: {e}")

    df = pd.DataFrame(records).sort_values("epoch")
    if not df.empty:
        csv_path = os.path.join(results_dir, "avg_Wqk_crosslayer_convergence_simple.csv")
        df.to_csv(csv_path, index=False)
        for col, color in zip(["avg_qk_crosslayer_diff", "avg_qq_crosslayer_diff", "avg_kk_crosslayer_diff", "avg_overall_diff"], ["purple", "blue", "green", "black"]):
            plt.plot(df["epoch"], df[col], marker='o', label=col, color=color)
        plt.xlabel("Epoch")
        plt.ylabel("Average Frobenius Norm")
        plt.title(f"Cross-Layer Operator Diffs: {exp_name}")
        plt.legend()
        plt.grid(True)
        path = os.path.join(results_dir, "avg_Wqk_crosslayer_convergence_simple.png")
        plt.savefig(path)
        print(f"Saved plot to {path}")
        plt.close()


def aggregate_across_seeds(prefix, results_root="results", output_dir="results/aggregated"):
    pattern = os.path.join(results_root, f"{prefix}*_kx*_analysis", "avg_Wqk_crosslayer_convergence_simple.csv")
    csv_paths = glob.glob(pattern)
    if not csv_paths:
        print("No seed result CSVs found for prefix", prefix)
        return

    kx_name_to_dfs = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        exp_dir = os.path.basename(os.path.dirname(path))
        match = re.search(r"_kx(\w+)", exp_dir)
        if match:
            kx_name = match.group(1).replace("_kx_analysis", "")
            kx_name_to_dfs.setdefault(kx_name, []).append(df)

    os.makedirs(output_dir, exist_ok=True)

    # Apply Seaborn style like your reference figure
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.4)

    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("colorblind", n_colors=len(kx_name_to_dfs))

    for idx, (kx_name, dfs) in enumerate(kx_name_to_dfs.items()):
        df_all = pd.concat(dfs)
        grouped = df_all.groupby("epoch")["avg_overall_diff"]
        mean = grouped.mean()
        std = grouped.std()

        x = mean.index
        y = mean.values
        lower = y - std
        upper = y + std

        plt.plot(x, y, label=kx_name, color=colors[idx], linewidth=2.5)
        plt.fill_between(x, lower, upper, color=colors[idx], alpha=0.3)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"$\overline{D}(t)$", fontsize=16)
    plt.title("Mean Operator Differences Over Training", fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Kernel", fontsize=13, title_fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"avg_Wqk_crosslayer_convergence_by_kx_{prefix}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved styled plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, help="Prefix for experiment subdirectories (e.g. seed_sweep_20250511_120707_seed)")
    parser.add_argument("--aggregate_only", action="store_true", help="Skip per-run computation and just aggregate")
    args = parser.parse_args()

    if not args.aggregate_only:
        subdirs = [d for d in os.listdir("checkpoints") if d.startswith(args.prefix)]
        for exp_name in subdirs:
            compute_and_save_for_directory(exp_name)

    aggregate_across_seeds(args.prefix)
