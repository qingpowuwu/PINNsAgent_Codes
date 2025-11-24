import argparse
import yaml
from itertools import product
import random
import os
from typing import Dict, List, Tuple
from datetime import date
import time

# -------------------- PDE list definition --------------------
pde_list_1d = [
    "Burgers1D",
    "Wave1D",
    # "KuramotoSivashinskyEquation",
]

pde_list_2d = [
    "Burgers2D",
    "Wave2D_Heterogeneous",
    "Heat2D_ComplexGeometry",
    "NS2D_LidDriven",
    "GrayScottEquation",
    "Heat2D_Multiscale",
    "Heat2D_VaryingCoef",
    "Poisson2D_ManyArea",
    # Above are presented in Table
    "Poisson2D_Classic",
    "PoissonBoltzmann2D",
    "NS2D_BackStep",
    # Very high MSE, we discard them
    # "Wave2D_LongTime",
    # "NS2D_LongTime",
    # "Heat2D_LongTime" # High MSE, we discard it for all configurations
]

pde_list_3d = [
    "Poisson3D_ComplexGeometry",
]

pde_list_nd = [
    "PoissonND",
    "HeatND",
]

# -------------------- Fixed parameters --------------------
fixed_params = {
    "seed": 44,
    "log_every": 100,
    "plot_every": 2000,
    "repeat": 1
}

# -------------------- Random point number generation --------------------
def generate_point_configs():
    num_domain_points = random.choice(["Default"] + list(range(100, 10001, 500)))
    num_boundary_points = random.choice(["Default"] + list(range(100, 10001, 500)))
    num_initial_points = random.choice(["Default"] + list(range(100, 10001, 500)))
    return num_domain_points, num_boundary_points, num_initial_points

# -------------------- yaml output generation --------------------
def generate_config_output(fixed_params, experiment_config, output_dir):
    output = f"""name: {fixed_params['name']}
device: '{fixed_params['device']}'
output_dir: './output/{output_dir}'

seed: {fixed_params['seed']}
log_every: {fixed_params['log_every']}
plot_every: {fixed_params['plot_every']}
repeat: {fixed_params['repeat']}
iter: {fixed_params['iter']}

# (0) pde list
pde_list: {fixed_params['pde_list']}

# (1) general method
general_method: "{experiment_config['general_method']}"

# (2) activation
activation: "{experiment_config['activation']}"

# (3) net
net: "{experiment_config['net']}"

# (4) optimizer
optimizer: "{experiment_config['optimizer']}"

# (5) sampler
sampler: "{experiment_config['sampler']}"

# (7) layers
width: {experiment_config['width']}
depth: {experiment_config['depth']}

# (6) loss weight
loss_weight: "{experiment_config['loss_weight']}"

# (8) lr
lr: {experiment_config['lr']}

# (9) sampler points
num_domain_points: {f'"{experiment_config["num_domain_points"]}"' if experiment_config['num_domain_points'] == "Default" else experiment_config['num_domain_points']}
num_boundary_points: {f'"{experiment_config["num_boundary_points"]}"' if experiment_config['num_boundary_points'] == "Default" else experiment_config['num_boundary_points']}
num_initial_points: {f'"{experiment_config["num_initial_points"]}"' if experiment_config['num_initial_points'] == "Default" else experiment_config['num_initial_points']}
num_test_points: {f'"{experiment_config["num_test_points"]}"' if experiment_config['num_test_points'] == "Default" else experiment_config['num_test_points']}

# (10) initializer
initializer: "{experiment_config['initializer']}"
"""
    return output

# -------------------- Main configuration generation function --------------------
def generate_configs(search_space, num_experiments, fixed_params, out_dir, pde_name, yaml_output_dir=None):
    seed = int(time.time())
    random.seed(seed)

    configurations = list(product(*search_space.values()))
    selected_configs_values = random.sample(configurations, num_experiments)
    
    # New directory structure: dim/pde-num/
    experiment_dir = os.path.join(
        out_dir, fixed_params['dim'], f"{pde_name}-{num_experiments}"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    for i, selected_config_value_i in enumerate(selected_configs_values, start=1):
        experiment_config = dict(zip(search_space.keys(), selected_config_value_i))
        experiment_config.update(fixed_params)
        num_domain_points, num_boundary_points, num_initial_points = generate_point_configs()
        experiment_config["num_domain_points"] = num_domain_points
        experiment_config["num_boundary_points"] = num_boundary_points
        experiment_config["num_initial_points"] = num_initial_points
        config_file = os.path.join(experiment_dir, f"train_{i}.yaml")
        with open(config_file, "w") as file:
            # 使用 yaml_output_dir 如果提供，否则使用 out_dir
            final_output_dir = yaml_output_dir if yaml_output_dir else out_dir
            output = generate_config_output(fixed_params, experiment_config, final_output_dir)
            file.write(output)
        print(f"Experiment {i} configuration saved to {config_file}")

# -------------------- argparse entry point --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINNs Random Search Config Generator")
    parser.add_argument('--pde_type', type=str, choices=['1d', '2d', '3d', 'nd'], default='1d')
    parser.add_argument('--device', type=str, default='0', help="Device field in YAML")
    parser.add_argument('--nexp', type=int, default=100, help="Number of configurations to generate per PDE")
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--out_dir', type=str, default='ICML_2025/RandomSearch', help="Output directory for config files")
    parser.add_argument('--yaml_output_dir', type=str, default=None, help="Output directory path to write in YAML files")
    parser.add_argument('--name', type=str, default='default', help='Batch name for this run')
    args = parser.parse_args()

    # Select PDE list
    if args.pde_type == '1d':
        pde_list = pde_list_1d
        dim = "1d"
    elif args.pde_type == '2d':
        pde_list = pde_list_2d
        dim = "2d"
    elif args.pde_type == '3d':
        pde_list = pde_list_3d
        dim = "3d"
    else:
        pde_list = pde_list_nd
        dim = "nd"

    # search_space
    search_space = {
        "general_method": ["none"], 
        "activation": ["elu", "relu", "selu", "sigmoid", "silu", "sin", "swish", "tanh"],  # Limited activation functions
        "net": ["fnn", "laaf", "gaaf"],
        "optimizer": ["adam", "multiadam", "lbfgs"],  # Limited optimizer range
        "sampler": ["none"],
        "loss_weight": ["none"],
        "width": list(range(100, 251, 4)),  # Limited width range to 100-250
        "depth": list(range(3, 8)),  # Limited depth range to 3-7
        "lr": [1e-5, 1e-4, 1e-3],  # Limited learning rate range
        "num_test_points": ["Default"],
        "initializer": ["Glorot normal", "Glorot uniform", "He normal", "He uniform", "zeros"]
    }

    fixed_params_local = fixed_params.copy()
    fixed_params_local['device'] = args.device
    fixed_params_local['iter'] = args.iter
    fixed_params_local['dim'] = dim

    # Iterate through each PDE and generate yaml files separately
    for pde in pde_list:
        fixed_params_local['pde_list'] = [pde]
        fixed_params_local['name'] = f"{pde.lower()}"
        
        # USE yaml_output_dir if provided, else use out_dir
        yaml_out_path = args.yaml_output_dir if args.yaml_output_dir else args.out_dir
        
        generate_configs(search_space, args.nexp, fixed_params_local, args.out_dir, pde, yaml_out_path)

# python 1_random_search-yaml-argsparser.py --pde_type 1d --device 0 --nexp 10 --out_dir ./ICML_2025/RandomSearch
# python 1_random_search-yaml-argsparser.py --pde_type 2d --device 0 --nexp 10 --out_dir ./ICML_2025/RandomSearch
# python 1_random_search-yaml-argsparser.py --pde_type 3d --device 0 --nexp 10 --out_dir ./ICML_2025/RandomSearch