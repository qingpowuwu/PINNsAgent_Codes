import argparse
import time
import os
import yaml
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde
from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime, HeatND
from src.pde.ns import NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime
from src.pde.poisson import Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea, PoissonND
from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime
from src.pde.inverse import PoissonInv, HeatInv
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.utils.rar import rar_wrapper
from utils.util_colors import RED, GRAY, BLUE, YELLOW, GREEN, RESET
from utils.util_prints import print_dict, print_namespace
from src.utils.args import parse_hidden_layers, parse_loss_weight, parse_width_depth
import sys


pde_classes = {
    # 1D
    'Burgers1D': Burgers1D,
    'Wave1D': Wave1D,
    'KuramotoSivashinskyEquation': KuramotoSivashinskyEquation,
    # 2D
    'Burgers2D': Burgers2D,
    'Poisson2D_Classic': Poisson2D_Classic,
    'PoissonBoltzmann2D': PoissonBoltzmann2D,
    'Poisson2D_ManyArea': Poisson2D_ManyArea,
    'Heat2D_VaryingCoef': Heat2D_VaryingCoef,
    'Heat2D_Multiscale': Heat2D_Multiscale,
    'Heat2D_ComplexGeometry': Heat2D_ComplexGeometry,
    'Heat2D_LongTime': Heat2D_LongTime,
    'NS2D_LidDriven': NS2D_LidDriven,
    'NS2D_BackStep': NS2D_BackStep,
    'NS2D_LongTime': NS2D_LongTime,
    'Wave2D_Heterogeneous': Wave2D_Heterogeneous,
    'Wave2D_LongTime': Wave2D_LongTime,
    'GrayScottEquation': GrayScottEquation,
    # 3D
    'Poisson3D_ComplexGeometry': Poisson3D_ComplexGeometry,
    # ND
    'PoissonND': PoissonND,
    'HeatND': HeatND
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer')
    parser.add_argument('--name', type=str, default="benchmark")
    parser.add_argument('--device', type=str, default="0")  # set to "cpu" enables cpu training 
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--pde_list', type=str, nargs='+', default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    
    # general_method can be none or gepinn
    # (1) gepinn: general PINN method
    parser.add_argument('--general_method', type=str, default="none", 
                        choices=['none', 'gepinn'], 
                        help="general_method can be chosen from 'none', 'gepinn'")
    # (2) activation function
    parser.add_argument('--activation', type=str, default="tanh",
                        choices=['elu', 'relu', 'selu', 'sigmoid', 'silu', 'sin', 'swish', 'tanh'],
                        help="activation can be chosen from 'elu', 'relu', 'selu', 'sigmoid', 'silu', 'sin', 'swish', 'tanh'")
    # (3) network structure
    parser.add_argument('--net', type=str, default="fnn", 
                        choices=['fnn', 'laaf', 'gaaf'], 
                        help="net can be chosen from 'fnn', 'laaf', 'gaaf'")
    # (4) optimizer
    parser.add_argument('--optimizer', type=str, default="adam", 
                        choices=['adam', 'sgd', 'multiadam', 'lra', 'ntk', 'lbfgs'], 
                        help="optimizer can be chosen from 'adam', 'sgd', 'multiadam', 'lra', 'ntk', 'lbfgs'")
    parser.add_argument('--switch_epoch', type=int, default=5000,
                        help="Epoch to switch from Adam to L-BFGS")
    # (5) sampler
    parser.add_argument('--sampler', type=str, default="none", 
                        choices=['none', 'rar'], 
                        help="sampler can be chosen from 'rar'")
    # (6) loss weight
    parser.add_argument('--loss_weight', type=str, default='none',
                        help="Loss weight for different loss terms, e.g., '1.0,2.0,3.0'")
    # (7) layers
    parser.add_argument('--width', type=int, default=100,
                        help="Number of neurons in each hidden layer")
                        
    parser.add_argument('--depth', type=int, default=5,
                        help="Number of hidden layers")

    # (8) lr
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for optimizer")

    # (9) sampler points
    def custom_type(value):
        if value == "Default":
            return value
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}' for --num_domain_points. Expected 'Default' or an integer.")

    parser.add_argument('--num_domain_points', type=custom_type, default="Default",
                        help='Number of domain points for training. If not provided, use the default value.')
    parser.add_argument('--num_boundary_points', type=custom_type, default="Default",
                        help='Number of boundary points for training. If not provided, use the default value.')
    parser.add_argument('--num_initial_points', type=custom_type, default="Default",
                        help='Number of initial points for training (for time-dependent problems). If not provided, use the default value.')
    parser.add_argument('--num_test_points', type=custom_type, default="Default",
                        help='Number of test points for evaluation. If not provided, use the default value.')

    # (10) initializers
    parser.add_argument('--initializer', type=str, default="Glorot normal",
                        help="Initializer for weights")

    parser.add_argument('--yaml_path', type=str, default=None,
                        help="Path to the YAML configuration file, e.g., config/train_5.yaml")
    
    command_args = parser.parse_args()

    if command_args.yaml_path is not None:
        with open(command_args.yaml_path, "r") as f:
            config = yaml.safe_load(f)

        command_args.pde_list = command_args.pde_list = config['pde_list']

        # Update the command-line arguments with the values from the YAML file
        for key, value in config.items():
            if key == 'lr':
                value = float(value)
            setattr(command_args, key, value)

    pde_list = [pde_classes[pde_name] for pde_name in command_args.pde_list]

    suffix = f"PDEs_{len(pde_list)}个--{command_args.activation}-{command_args.net}-{command_args.optimizer}-{command_args.sampler}-{command_args.lr}-{command_args.width}-{command_args.depth}"
    time_str = time.strftime('%H-%M', time.localtime())

    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)

    if command_args.output_dir is not None:
        exp_dir = os.path.join(command_args.output_dir, command_args.name, f"{suffix}---{time_str}")
        trainer = Trainer(exp_dir, command_args.device)
    else:
        exp_dir = f"{command_args.name}/{suffix}---{time_str}"
        trainer = Trainer(exp_dir, command_args.device)

    print(f"\n{YELLOW}{'== (2). 开始遍历 pde_list 来 创建 src.pde.burgers.Burgers1D, src.pde.burgers.Burgers2D, ... 等 PDE 的模型 ======================= '}{RESET}\n")
    for pde_config in pde_list:

        def get_model_dde(): # 获得一个 model, 这个变量是 deepxde 里面定义的类型
            if isinstance(pde_config, tuple):
                pde = pde_config[0](**pde_config[1])
            else:
                pde = pde_config()
            
            # (2).2 定义 PDE 的训练点
            # print(f"\n{BLUE}{'== (2).2 定义 PDE 的训练点 ======================================= '}{RESET}\n")
            pde.training_points(
                domain=command_args.num_domain_points,
                boundary=command_args.num_boundary_points,
                initial=command_args.num_initial_points,
                test=command_args.num_test_points,
                mul=1
            )
            
            if command_args.general_method == "gepinn": # Gradient-enhanced pinns
                pde.use_gepinn()


            # (2).3 默认的网络结构 (根据 command_args.net 来选择)
            # print(f"\n{BLUE}{'== (2).3 根据 command_args.net 来选择 网络结构 ======================================= '}{RESET}\n")
            if command_args.net == "fnn":            
                net = dde.nn.FNN(
                                layer_sizes=[pde.input_dim] + parse_width_depth(command_args.width, command_args.depth) + [pde.output_dim],
                                activation=command_args.activation,
                                kernel_initializer=command_args.initializer
                )
            elif command_args.net == "laaf":
                net = DNN_LAAF(
                                n_layers=command_args.depth,
                                n_hidden=command_args.width,
                                x_dim=pde.input_dim,
                                u_dim=pde.output_dim,
                                activation=command_args.activation,
                                kernel_initializer=command_args.initializer
                )
            elif command_args.net == "gaaf":
                net = DNN_GAAF(
                                n_layers=command_args.depth,
                                n_hidden=command_args.width,
                                x_dim=pde.input_dim,
                                u_dim=pde.output_dim,
                                activation=command_args.activation,
                                kernel_initializer=command_args.initializer
                )
            net = net.float() # 将网络转换为 float 类型
            print(' - type(net):', type(net))

            # (2).4 定义损失函数的权重
            # print(f"\n{BLUE}{'== (2).4 定义损失函数的权重 ======================================= '}{RESET}\n")

            loss_weights = parse_loss_weight(command_args.loss_weight) # 解析损失权重, e.g, '1,2,3' -> [1.0, 2.0, 3.0]
            if loss_weights is None:
                loss_weights = np.ones(pde.num_loss)
            else:
                loss_weights = np.array(loss_weights)


            # (2).5 定义优化器 (根据 command_args.optimizer 来选择)
            # print(f"\n{BLUE}{'== (2).5 根据 command_args.optimizer 来选择 优化器 ======================================= '}{RESET}\n")

            # print(f'{GRAY}')
            opt = torch.optim.Adam(net.parameters(), command_args.lr) # 创建默认的 Adam 优化器
            if command_args.optimizer == "adam":    # Adam
                opt = torch.optim.Adam(net.parameters(), command_args.lr)
                # print('使用 Adam 优化器')
            elif command_args.optimizer == "sgd":   # SGD
                opt = torch.optim.SGD(net.parameters(), lr=command_args.lr, momentum=0.0)
            elif command_args.optimizer == "multiadam": # Multiscale Adam
                # opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
                opt = MultiAdam(net.parameters(), lr=command_args.lr, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
                # print('使用 MultiAdam 优化器')
            elif command_args.optimizer == "lra":     # reweighting using gradient norm
                opt = LR_Adaptor(optimizer = opt, loss_weight = loss_weights, num_pde = pde.num_pde, alpha = 0.1, mode = "max")
                # print('使用 LR_Adaptor 优化器')
            elif command_args.optimizer == "ntk":     # neural tangent kernel
                opt = LR_Adaptor_NTK(optimizer = opt, loss_weight = loss_weights, pde = pde)
                # print('使用 LR_Adaptor_NTK 优化器')
            elif command_args.optimizer == "lbfgs":   # Limited-memory BFGS: Broyden–Fletcher–Goldfarb–Shanno algorithm
                opt = Adam_LBFGS(net.parameters(), switch_epoch=command_args.switch_epoch, adam_param={'lr':command_args.lr})
                # print('使用 Adam_LBFGS 优化器')

            # (2).6 把 PDE 和 net 传入到模型中 => 创建 deepxde 期望的模型 类型
            # print(f"\n{BLUE}{'== (2).6 调用 pde.createmodel(net): 把 dde.data.TimePDE or dde.data.PDE 和 net 传入到模型中 => 创建 deepxde 期望的模型 类型 ======================================= '}{RESET}\n")
            model = pde.create_model(net)
            model.compile(opt, loss_weights=loss_weights) # 编译模型

            # (2).7 定义 sampler (根据 command_args.sampler 来选择)
            # print(f"\n{BLUE}{'== (2).7 根据 command_args.sampler 来选择 sampler ======================================= '}{RESET}\n")
            # print(f'{GRAY}')
            if command_args.sampler == "rar":  # 如果使用 RAR 方法 (i.e, residual-based sampling) 
                model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1}) 
            # the trainer calls model.train(**train_args)

            # (2).8 打印出所有的 pde 参数，并且赋值给 command_args
            # print(f"\n{BLUE}{'== 2).8 打印出所有的 pde 参数，并且赋值给 command_args ======================================= '}{RESET}\n")
            # print(f'{GRAY}')
            command_args.pde = pde
            command_args.pde_loss_config    = pde.loss_config
            if pde.ref_data is not None:
                if pde.ref_data.shape is not None:
                    command_args.pde_ref_data_shape = pde.ref_data.shape
            else:
                command_args.pde_ref_data_shape = None
            command_args.pde_input_dim      = pde.input_dim
            command_args.pde_output_dim     = pde.output_dim
            command_args.pde_num_pde        = pde.num_pde
            command_args.pde_num_boundary   = pde.num_boundary
            command_args.pde_num_loss       = pde.num_loss
            command_args.pde_loss_weights   = loss_weights
            command_args.pde_num_domain_points   = pde.num_domain_points
            command_args.pde_num_boundary_points = pde.num_boundary_points
            command_args.pde_num_test_points     = pde.num_test_points
            command_args.pde_num_initial_points  = pde.num_initial_points

            print("="*30)
            print("Command Arguments:")
            print("-"*30)
            for arg_name, arg_value in vars(command_args).items():
                print(f"{arg_name:<25}: {arg_value}")
            print("="*30)

            print(f'{RESET}')

            # print(f"\n{BLUE} get_model_dde() 函数: 结束 ==================================================================== {RESET}\n")
            return model, command_args

        # 7. 把每1个 pde_config 所对应的 model, train_args 字典 加入到 trainer 对象中的 self.tasks list 中
        # # print(f"\n{BLUE}{f'== (2).0 把 pde_config 所对应的 model {YELLOW}{pde_config}{RESET}, {BLUE}train_args 字典 加入到 trainer 对象中的 self.tasks list 中 ======================================= '}{RESET}\n")
        trainer.add_task(
            get_model = get_model_dde,
            train_args = {  
                "iterations":    command_args.iter,      # 20000
                "display_every": command_args.log_every, # 100
                "callbacks": [
                    TesterCallback(log_every=command_args.log_every),
                    PlotCallback(log_every=command_args.plot_every, fast=True),
                    LossCallback(verbose=True)
                ]
            }
        )

    # trainer.setup(__file__, seed)
    # (3). 设置实验的名字, 设备, 重复次数等
    # print(f"\n{YELLOW}{'== (3). 设置实验的名字, 设备, 重复次数等 ======================================= '}{RESET}\n")
    # print(f'{YELLOW} (3).1 调用 trainer.setup() 函数 => 创建实验目录 & 备份各种文件 {RESET}')
    trainer.setup(
                pyfile_name=__file__,             # 脚本文件的路径 => 用来备份
                yaml_path=command_args.yaml_path, # 配置文件的路径 => 用来复制
                seed=seed
    )
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary(exp_time=time_str)
    
    try:
        ok_dir = exp_dir + "-OK"
        if not os.path.exists(ok_dir):
            os.rename(exp_dir, ok_dir)
            print(f"\n{GREEN}实验目录重命名为: {ok_dir}{RESET}\n")
        else:
            print(f"\n{YELLOW}警告: 目标目录 {ok_dir} 已存在，未自动重命名！{RESET}\n")
    except Exception as e:
        print(f"\n{RED}重命名实验目录失败: {e}{RESET}\n")
