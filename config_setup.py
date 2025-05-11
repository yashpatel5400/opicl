import torch
import os

def parse_env_var(varname, default, cast_fn):
    val = os.getenv(varname.upper())
    return cast_fn(val) if val is not None else default

def get_model_base_config(num_layers=5, H=64, W=64):
    return {
        'num_layers': num_layers,
        'im_size': (H, W), 
        'kx_name': parse_env_var("kx_name", "linear", str),
        'kx_sigma': 1.0,
        'icl_lr': -0.01,
    }

def get_dataset_base_config(H=64, W=64):
    return {
        'ky_kernel_name': "gaussian",
        'num_incontext_prompts': 25,
        'num_operator_bases': 30,
        'fixed_operator_seed': 0,
        'fixed_prompts_seed': 101,
        'im_size_tuple': (H, W)
    }

def get_training_args(experiment_name=None):
    experiment_name = experiment_name or os.getenv("EXPERIMENT_NAME", "default_experiment")
    model_cfg = get_model_base_config()
    dataset_cfg = get_dataset_base_config(H=model_cfg['im_size'][0], W=model_cfg['im_size'][1])

    return {
        'im_size': model_cfg['im_size'],
        'num_layers': model_cfg['num_layers'],
        'kx_name': model_cfg['kx_name'],
        'kx_sigma': model_cfg['kx_sigma'],
        'icl_lr': model_cfg['icl_lr'],

        'icl_init': False,
        'fix_value_operator_scalar': True,

        'ky_kernel_name': dataset_cfg['ky_kernel_name'],
        'num_total_icl_tasks_in_dataset': 128,
        'per_gpu_actual_batch_size': 32,
        'num_operator_bases': dataset_cfg['num_operator_bases'],
        'num_incontext_prompts': dataset_cfg['num_incontext_prompts'],
        'fixed_operator_seed': dataset_cfg['fixed_operator_seed'],
        'fixed_prompts_seed': dataset_cfg['fixed_prompts_seed'],

        'optimizer_type': "adam",
        'lr': 1e-2,
        'sgd_momentum': 0.0,

        'epochs': 100,
        'log_freq': 10,
        'save_freq': 1,

        'log_dir': f"logs/{experiment_name}",
        'tensorboard_log_dir': f"runs/{experiment_name}_tb",
        'checkpoint_dir': f"checkpoints/{experiment_name}",
        'ddp_find_unused_parameters': False,
        'resume_from_checkpoint': True,
    }

def get_analysis_args(experiment_name="default_experiment"):
    training_cfg = get_training_args(experiment_name)
    model_cfg = get_model_base_config(num_layers=training_cfg['num_layers'], 
                                      H=training_cfg['im_size'][0], W=training_cfg['im_size'][1])
    dataset_cfg = get_dataset_base_config(H=training_cfg['im_size'][0], W=training_cfg['im_size'][1])

    return {
        'checkpoint_directory': training_cfg['checkpoint_dir'], 
        'results_output_directory': f"results/{experiment_name}_kx_analysis",
        'num_layers_in_model': model_cfg['num_layers'],
        'analysis_device_str': "cuda" if torch.cuda.is_available() else "cpu",

        'im_size': model_cfg['im_size'],
        'ky_kernel_name': dataset_cfg['ky_kernel_name'],
        'kx_name': model_cfg['kx_name'],
        'kx_sigma': model_cfg['kx_sigma'],
        'fixed_value_op_scalar': model_cfg['icl_lr'],

        'fixed_operator_seed_analysis': dataset_cfg['fixed_operator_seed'], 
        'fixed_prompts_seed_analysis': dataset_cfg['fixed_prompts_seed'] + 1,
        'num_incontext_samples_analysis': dataset_cfg['num_incontext_prompts'],
        'num_operator_bases_analysis': dataset_cfg['num_operator_bases'],

        'plot_kx_difference_evolution': True,
        'plot_individual_kx_matrices': False, 
        'epochs_to_plot_kx_for': [0], 
        'layers_to_plot_kx_for': [0, model_cfg['num_layers'] // 2, model_cfg['num_layers'] - 1 if model_cfg['num_layers'] > 0 else 0],
    }
