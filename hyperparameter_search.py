import os
import subprocess
from datetime import datetime
from multiprocessing import Process

# Timestamp for this sweep session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Seed values to try (limit to 4 jobs)
seed_list = [0, 1, 2, 3]

# kx kernel types to sweep over
kx_name_list = ['linear', 'laplacian', 'gradient_rbf', 'energy']

# Fixed config params
sweep_args_fixed = {
    "experiment_prefix": "seed_sweep",
}

# Available GPU IDs
available_gpus = [2, 3, 4, 5]

def run_experiment(experiment_name, fixed_operator_seed, kx_name, cuda_device):
    env = os.environ.copy()
    env["EXPERIMENT_NAME"] = experiment_name
    env["FIXED_OPERATOR_SEED"] = str(fixed_operator_seed)
    env["KX_NAME"] = kx_name
    env["MASTER_PORT"] = str(12355 + fixed_operator_seed)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    print(f"\n🚀 Launching experiment: {experiment_name} on CUDA_VISIBLE_DEVICES={cuda_device}")
    subprocess.run(["python", "train.py"], env=env, check=True)

if __name__ == "__main__":
    experiment_prefix = sweep_args_fixed['experiment_prefix']
    all_experiment_names = []

    for j, kx_name in enumerate(kx_name_list):
        processes = []
        for i, seed_i in enumerate(seed_list):
            seed = 10 * j + seed_i

            experiment_name = f"{experiment_prefix}_{timestamp}_seed{seed}_kx{kx_name}"
            all_experiment_names.append(experiment_name)
            cuda_device = available_gpus[i % len(available_gpus)]
            p = Process(target=run_experiment, args=(experiment_name, seed, kx_name, cuda_device))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()