import dill
import json
import multiprocessing
import os
import random
import shutil
import sys
import time

dill.settings['recurse'] = True
from utils.util_colors import RED, GRAY, BLUE, YELLOW, GREEN, RESET
import csv
import shutil

class HookedStdout:
    original_stdout = sys.stdout

    def __init__(self, filename, stdout=None) -> None:
        if stdout is None:
            self.stdout = self.original_stdout
        else:
            self.stdout = stdout
        self.file = open(filename, 'w')

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def isatty(self):
        # wandb
        return getattr(self.stdout, "isatty", lambda: False)()


def train_process(data, save_path, device, seed):
    hooked = HookedStdout(f"{save_path}/log.txt")
    sys.stdout = hooked
    sys.stderr = HookedStdout(f"{save_path}/logerr.txt", sys.stderr)

    import torch
    import deepxde as dde
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dde.config.set_default_float('float32')
    dde.config.set_random_seed(seed)

    get_model, train_args = dill.loads(data)
    model, command_args = get_model()
    model.train(**train_args, model_save_path=save_path)


class Trainer:

    def __init__(self, exp_name, device) -> None:
        self.exp_name = exp_name
        self.device = device.split(",")
        self.repeat = 1
        self.tasks = []

    def set_repeat(self, repeat):
        self.repeat = repeat

    def add_task(self, get_model, train_args):
        data = dill.dumps((get_model, train_args))
        self.tasks.append((data, train_args))

    def setup(self, pyfile_name, yaml_path, seed):
        """
            pythonfile_name: 脚本文件的路径 => 用来备份
            yaml_path:       配置文件的路径 => 用来复制
        """
        os.makedirs(f"{self.exp_name}", exist_ok=True)
        shutil.copy(pyfile_name, f"{self.exp_name}/script_back.py")
        yaml_name = os.path.basename(yaml_path)
        shutil.copy(yaml_path, f"{self.exp_name}/{yaml_name}")
        print(f"{GRAY} (1) 创建实验的目录: {self.exp_name} {RESET}")
        print(f"{GRAY} (2) 备份脚本文件: {pyfile_name} => {self.exp_name}/script_back.py {RESET}")
        print(f"{GRAY} (3) 复制配置文件: {yaml_name} => {self.exp_name}/{yaml_name} {RESET}")
        print(f"{GRAY} (4) 保存配置信息: {self.exp_name}/config.json {RESET}")
        json.dump({"seed": seed, "task": self.tasks}, open(f"{self.exp_name}/config.json", 'w'), indent=4, default=lambda _: "...")

    def train_all(self):
        if len(self.device) > 1:
            return self.train_all_parallel()

        # no multi-processing when only one device is available
        import torch
        import deepxde as dde

        if self.device[0] != 'cpu':
            device = "cuda:" + self.device[0]
            torch.cuda.set_device(device)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        dde.config.set_default_float('float32')

        for j in range(self.repeat):
            for i, (data, _) in enumerate(self.tasks):
                print(f"{GRAY}\n******************************************* Begin #{i}-{j} *******************************************{RESET}\n")
                save_path = f"{self.exp_name}/{i}-{j}"
                os.makedirs(save_path, exist_ok=True)

                hooked = HookedStdout(f"{save_path}/log.txt")
                sys.stdout = hooked
                sys.stderr = HookedStdout(f"{save_path}/logerr.txt", sys.stderr)
                dde.config.set_random_seed(42)

                get_model, train_args = dill.loads(data)
                model, command_args = get_model()

                # 将 command_args 保存为 CSV 文件
                command_args_file = os.path.join(save_path, "command_args.csv")
                with open(command_args_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    # 将 Namespace 对象转换为字典
                    command_args_dict = vars(command_args)
                    # 获取所有键
                    keys = list(command_args_dict.keys())
                    # 写入键作为第一行
                    writer.writerow(keys)
                    # 写入值作为第二行
                    writer.writerow(command_args_dict.values())

                print(f'\n已经将 command_args (i.e, hypaerparameters) 保存到了 {command_args_file} 文件中\n')
                
                model.train(**train_args, model_save_path=save_path)
      
    def train_all_parallel(self):
        # maintain a pool of processes
        # do not start all processes at the same time
        # keep the number of processes equal to the number of devices
        # if a process is done, start a new one on the same device

        multiprocessing.set_start_method('spawn')
        processes = [None] * len(self.device)
        for j in range(self.repeat):
            for i, (data, _) in enumerate(self.tasks):
                # find a free device
                for k, p in enumerate(processes):
                    if p is None:
                        device = "cuda:" + self.device[k]
                        save_path = f"{self.exp_name}/{i}-{j}"
                        os.makedirs(save_path)

                        print(f"***** Start #{i}-{j} *****")
                        p = multiprocessing.Process(target=train_process, args=(data, save_path, device, 42), daemon=True)
                        p.start()
                        processes[k] = p
                        break
                else:
                    raise RuntimeError("No free device")

                # wait for a process to finish
                while True:
                    for k, p in enumerate(processes):
                        if p is None or not p.is_alive():
                            # free device
                            processes[k] = None
                            break
                    else:
                        time.sleep(5)
                        continue
                    break

        for p in processes:
            if p is not None:
                p.join()

    def summary(self, exp_time):
        from src.utils import summary
        summary.summary(runs_dir = f"{self.exp_name}", 
                        tasknum = len(self.tasks), # 20
                        repeat = self.repeat,  # 1
                        iters = list(map(lambda t:t[1]['iterations'], self.tasks)),
                        csv_name = self.exp_name,
                        exp_time = exp_time
                        )