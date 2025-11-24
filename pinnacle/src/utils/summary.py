import numpy as np
import pandas as pd
import wandb
import string
from utils.util_colors import RED, GRAY, BLUE, YELLOW, GREEN, RESET
from utils.util_prints import print_dict, print_namespace
from src.utils.args import parse_hidden_layers, parse_loss_weight, parse_width_depth
import os
import csv

# 这个函数的目的是对多个文件进行批量处理,并返回处理结果的 mean 和 std。
def _process(func, path, repeat):
    """
        func: 一个函数, 用于处理数据, 例如提取时间, 提取损失等。
        path: 文件路径模版, 例如 '{}/{}-0/log.txt'.format(path, i)
        repeat: 重复次数, 例如 5
    """
    # 创建一个空列表 data 用于存储处理结果。
    data = []
    # 使用 np.loadtxt() 从 path.fomrat(i) 中加载数据,并将 func 函数的返回值添加到 data 列表中。
    # 适用于 np.array 数据
    try:
        for i in range(repeat):
            data.append(func(np.loadtxt(path.format(i))))
        # 将列表转换为 NumPy 数组,并排除任何 NaN 值
        data = np.array(data)[~np.isnan(data)]  # exclude nan
        return np.mean(data), np.std(data)
    except ValueError as e:  # should use method below
        if len(data) != 0:
            print(e)
            return np.nan, np.nan

    # 使用 readlines () 来从指定路径读取文本文件
    # 适用于文本数据
    try:
        for i in range(repeat):
            data.append(func(open(path.format(i)).readlines()))
        # 将列表转换为 NumPy 数组,并排除任何 NaN 值
        data = np.array(data)[~np.isnan(data)]
        return np.mean(data), np.std(data)
    except Exception as e:
        print(e)
        return np.nan, np.nan

def extract_time(lines):
    "用来提取训练时间, 通过检查 log.txt 里面是否有 'train' took 253.845810 s"
    # example: 'train' took 253.845810 s
    for line in lines:
        line = line.strip()
        if line.startswith("'train'"):
            return float(line.split(' ')[2])
    print("\033[33mWarning:Could not find training time.\033[0m")  # yellow
    return np.nan

def extract_name(path):
    "用来提取 PDE Class Name, 通过检查 log.txt 里面是否有 PDE Class Name: PoissonND"
    # example: PDE Class Name: PoissonND
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("PDE Class Name:"):
                return line.split(' ')[3]
    print("\033[33mWarning:Could not find PDE Class Name.\033[0m")  # yellow
    return ""

def extract_success(lines):
    "用来检查是否 train 成功, 通过检查 log.txt 里面是否有 Epoch 20000: saving model to runs/08.10-05.59.14-LBFGS_MainExp/0-0/20000.pt ...'train' took 253.845810 s"
    # example: Epoch 20000: saving model to runs/08.10-05.59.14-LBFGS_MainExp/0-0/20000.pt ...
    flags = [False, False]
    for line in lines:
        line = line.strip()
        if line.startswith("Epoch 20000:"):
            flags[0]=True
        elif line.startswith("'train'"):
            flags[1]=True
    return 1 if flags[0] and flags[1] else 0

def summary(runs_dir, tasknum, repeat, iters, csv_name, exp_time):
    print(f"\n{GRAY} summary 函数: 开始 (把 arg 导入到 wandb) ====================================================================  {RESET}\n")
    columns = ['pde', 'iter', 'success_rate', 'run_time', 'run_time_std', 'train_loss', 'train_loss_std', \
               'mse', 'mse_std', 'mxe', 'mxe_std', 'l2rel', 'l2rel_std', 'crmse', 'crmse_std', \
               'frmse_low', 'frmse_low_std', 'frmse_mid', 'frmse_mid_std', 'frmse_high', 'frmse_high_std']
    result = []

    for i in range(tasknum): # 遍历每一个任务
        name_pde = extract_name('{}/{}-0/log.txt'.format(runs_dir, i)) # Poisson2D_Classic'
        # 读取 command_args.csv 文件
        command_args_df = pd.read_csv('{}/{}-0/command_args.csv'.format(runs_dir, i), header=0)
        command_args_dict = command_args_df.iloc[0].to_dict() # 将 command_args_df 转换为字典, 选取第一行
        # (1) 提取是否成功, 训练时间, 训练损失
        try:
            success_mean, success_std = _process(extract_success, '{}/{}-{{}}/log.txt'.format(runs_dir, i), repeat)
            run_time_mean, run_time_std = _process(extract_time, '{}/{}-{{}}/log.txt'.format(runs_dir, i), repeat)
            train_loss_mean, train_loss_std = _process(lambda data: data[-1, 1], '{}/{}-{{}}/loss.txt'.format(runs_dir, i), repeat)
        except (FileNotFoundError, IOError):
            success_mean = np.nan
            run_time_mean = run_time_std = np.nan
            train_loss_mean = train_loss_std = np.nan
        try:
            # -1: 表示最后一行
            # 2，3，... 表示第几列
            # 从第2列读取数据,计算均值和标准差
            mse_mean, mse_std = _process(lambda data: data[-1, 2], '{}/{}-{{}}/errors.txt'.format(runs_dir, i), repeat)
            mxe_mean, mxe_std = _process(lambda data: data[-1, 3], '{}/{}-{{}}/errors.txt'.format(runs_dir, i), repeat)
            l2rel_mean, l2rel_std = _process(lambda data: data[-1, 5], '{}/{}-{{}}/errors.txt'.format(runs_dir, i), repeat)
            crmse_mean, crmse_std = _process(lambda data: data[-1, 6], '{}/{}-{{}}/errors.txt'.format(runs_dir, i), repeat)
            flow_mean, flow_std = _process(lambda data: data[-1, 7], '{}/{}-{{}}/errors.txt'.format(runs_dir, i), repeat)
            fmid_mean, fmid_std = _process(lambda data: data[-1, 8], '{}/{}-{{}}/errors.txt'.format(runs_dir, i), repeat)
            fhigh_mean, fhigh_std = _process(lambda data: data[-1, 9], '{}/{}-{{}}/errors.txt'.format(runs_dir, i), repeat)
        except Exception:
            mse_mean = mse_std = mxe_mean = mxe_std = np.nan
            l2rel_mean = l2rel_std = crmse_mean = crmse_std = np.nan
            flow_mean = flow_std = fmid_mean = fmid_std = np.nan
            fhigh_mean = fhigh_std = np.nan
        result.append([name_pde, iters[i], success_mean, run_time_mean, run_time_std, train_loss_mean, train_loss_std, mse_mean, mse_std, \
                       mxe_mean, mxe_std, l2rel_mean, l2rel_std, crmse_mean, crmse_std, \
                       flow_mean, flow_std, fmid_mean, fmid_std, fhigh_mean, fhigh_std])

        # (1) 从 command_args_df 中提取 hyperparameters
        wandb.init(project="PINNsAgent-RandomSearch",  # 截断以确保不超过128个字符
                name=f'{csv_name}-{command_args_dict["name"]}',
                entity='qingpowuwu-study',
                save_code=True,  # 保存代码
                dir=f'{runs_dir}/{i}-0', # 保存到指定的文件夹
                mode="offline",
                config={
                    "name":   command_args_dict["name"],
                    "device": command_args_dict["device"],
                    "seed":   command_args_dict["seed"],
                    "log_every":  command_args_dict["log_every"],
                    "plot_every": command_args_dict["plot_every"],
                    "repeat": command_args_dict["repeat"],
                    "iter":   command_args_dict["iter"],
                    "1_general_method": command_args_dict["general_method"], # (1)
                    "2_activation":     command_args_dict["activation"],     # (2)
                    "3_net":       command_args_dict["net"],                 # (3)
                    "4_optimizer": command_args_dict["optimizer"],           # (4)
                    "4_1_switch_epoch": command_args_dict["switch_epoch"],   # (4)
                    "5_sampler":   command_args_dict["sampler"],             # (5)
                    "6_loss_weight": command_args_dict["loss_weight"],       # (6)
                    "7_1_width": command_args_dict["width"],                 # (7)
                    "7_2_depth": command_args_dict["depth"],                 # (7)
                    "8_lr": command_args_dict["lr"],                         # (8)
                    "9_1_domain_points": command_args_dict["num_domain_points"], # (9)
                    "9_2_boundary_points": command_args_dict["num_boundary_points"], # (9)
                    "9_3_initial_points": command_args_dict["num_initial_points"], # (9)
                    "10_initializer": command_args_dict["initializer"],

                    "pde_list": command_args_dict["pde_list"],
                    "pde": command_args_dict["pde"],
                    "pde_loss_config": command_args_dict["pde_loss_config"],
                    "pde_ref_data_shape": command_args_dict["pde_ref_data_shape"],
                    "pde_input_dim": command_args_dict["pde_input_dim"],
                    "pde_output_dim": command_args_dict["pde_output_dim"],
                    "pde_num_pde": command_args_dict["pde_num_pde"],
                    "pde_num_boundary": command_args_dict["pde_num_boundary"],
                    "pde_num_loss": command_args_dict["pde_num_loss"],
                    "pde_loss_weights": command_args_dict["pde_loss_weights"],
                    "pde_num_domain_points": command_args_dict["pde_num_domain_points"],
                    "pde_num_boundary_points": command_args_dict["pde_num_boundary_points"],
                    "pde_num_test_points": command_args_dict["pde_num_test_points"],
                    "pde_num_initial_points": command_args_dict["pde_num_initial_points"]
                    }
                    )

        # (2) 记录 metrics
        metrics_dict = {
            "task": name_pde,
            "exp_time": exp_time,
            "iter": int(iters[i]),
            "success_rate": float(success_mean),
            "0_run_time": float(run_time_mean),
            "1_train_loss": float(train_loss_mean),
            "2_mse": float(mse_mean),
            "3_mxe": float(mxe_mean),
            "4_l2rel": float(l2rel_mean),
            "5_crmse": float(crmse_mean),
            "6_frmse_low": float(flow_mean),
            "7_frmse_mid": float(fmid_mean),
            "8_frmse_high": float(fhigh_mean),
            "0_run_time_std": float(run_time_std),
            "1_train_loss_std": float(train_loss_std),
            "2_mse_std": float(mse_std),
            "3_mxe_std": float(mxe_std),
            "4_l2rel_std": float(l2rel_std),
            "5_crmse_std": float(crmse_std),
            "6_frmse_low_std": float(flow_std),
            "7_frmse_mid_std": float(fmid_std),
            "8_frmse_high_std": float(fhigh_std)
        }

        # 使用 wandb.log() 记录指标数据
        wandb.log(metrics_dict)

        # 将 wandb.config 和 metrics_dict 保存到 final_results.csv 文件
        run_config_file = f'{runs_dir}/{i}-0/final_results.csv'
        with open(run_config_file, 'w', newline='') as csvfile:
            fieldnames = list(wandb.config.keys()) + list(metrics_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader() # 写入表头
            row = {key: value for key, value in wandb.config.items()} # 写入数据
            row.update({key: value for key, value in metrics_dict.items()})
            writer.writerow(row)
            print(f' - {YELLOW}已经把当次训练 PINNs 的 wandb.config 和 metrics_dict 保存到 PINNAcle 目录下的 {run_config_file} 文件中 {RESET}\n')

        # 完成 wandb run
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error during wandb.finish(): {e}")

    print(f"\n{GRAY} summary 函数: 结束 (把 arg 导入到 wandb) ====================================================================  {RESET}\n")

