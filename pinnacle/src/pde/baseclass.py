import deepxde as dde
import numpy as np

DEFAULT_NUM_DOMAIN_POINTS = 8192
DEFAULT_NUM_BOUNDARY_POINTS = 2048
DEFAULT_NUM_TEST_POINTS = 8192
DEFAULT_NUM_INITIAL_POINTS = 2048

# ============ 修改1: 添加自己的包 ========================================
from utils.util_colors import RED, GRAY, BLUE, YELLOW, GREEN, RESET
# ============ 修改1: 添加自己的包 ========================================

class BasePDE():

    def __init__(self):
        self.pde = None
        self.bcs = None
        self.geom = None
        self.bbox = None
        self.loss_config = []
        self.output_config = None

        self.num_domain_points = DEFAULT_NUM_DOMAIN_POINTS
        self.num_boundary_points = DEFAULT_NUM_BOUNDARY_POINTS
        self.num_test_points = DEFAULT_NUM_TEST_POINTS
        self.num_initial_points = DEFAULT_NUM_INITIAL_POINTS

        self.ref_sol = None
        self.ref_data = None

        self.recommend_net = None

    @property
    def input_dim(self):
        return self.geom.dim

    @property
    def output_dim(self):
        if self.output_config is None:
            raise ValueError("output_config not set")
        return len(self.output_config)

    @output_dim.setter
    def output_dim(self, value):
        if self.output_config is None:
            self.output_config = [{'name': f'y_{i+1}'} for i in range(value)]
        else:
            assert self.output_dim == value, "output_config and output_dim not matched"

    @property
    def num_pde(self):
        count = 0
        for config in self.loss_config:
            if config['type'] == 'pde':
                count += 1
        return count

    @property
    def num_gepinn(self):
        count = 0
        for config in self.loss_config:
            if config['type'] == 'gepinn':
                count += 1
        return count

    @property
    def num_boundary(self):
        count = 0
        for config in self.loss_config:
            if config['type'] == 'boundary':
                count += 1
        return count

    @property
    def num_loss(self):
        return len(self.loss_config)

    def trans_time_data_to_dataset(self, datapath):
        data = self.ref_data
        slice = (data.shape[1] - self.input_dim + 1) // self.output_dim
        assert slice * self.output_dim == data.shape[1] - self.input_dim + 1, "Data shape is not multiple of pde.output_dim"
        
        with open(datapath, "r") as f:
            def extract_time(string):
                index = string.find("t=")
                if index == -1:
                    return None
                return float(string[index+2:].split(' ')[0])
            t = None
            for line in f.readlines():
                if line.startswith('%') and line.count('@') == slice * self.output_dim:
                    t = line.split('@')[1:]
                    t = list(map(extract_time, t))
            if t is None or None in t: 
                raise ValueError("Reference Data not in Comsol format or does not contain time info")
            t = np.array(t[::self.output_dim])

        t, x0 = np.meshgrid(t, data[:, 0])
        list_x = [x0.reshape(-1)]
        for i in range(1, self.input_dim - 1):
            list_x.append(np.stack([data[:, i] for _ in range(slice)]).T.reshape(-1))
        list_x.append(t.reshape(-1))
        for i in range(self.output_dim):
            list_x.append(data[:, self.input_dim - 1 + i::self.output_dim].reshape(-1))
        self.ref_data = np.stack(list_x).T
        
        # print('self.ref_data.shape = ', self.ref_data.shape)
        # print(f'{RESET}')
        # print(f"\n{GRAY} class BasePDE 的 trans_time_data_to_dataset(self, datapath) 方法: 结束 ====================================================================  {RESET}\n")

    def load_ref_data(self, datapath, transform_fn=None, t_transpose=False):
        # print(f"\n{GRAY} class BasePDE 的 load_ref_data(self, datapath, transform_fn=None, t_transpose=False) 方法: 开始 ====================================================================  {RESET}\n")
        self.ref_data = np.loadtxt(datapath, comments="%").astype(np.float32)
        if t_transpose:  # originally used only in BaseTimePDE, but needed from some TimePDE using BasePDE as baseclass.
            self.trans_time_data_to_dataset(datapath)
        if transform_fn is not None:
            self.ref_data = transform_fn(self.ref_data)
        # print(f'{GRAY}')
        # print('self.ref_data.shape = ', self.ref_data.shape) # (1111, 3)
        # print(f'{RESET}')
        # print(f"\n{GRAY} class BasePDE 的 load_ref_data(self, datapath, transform_fn=None, t_transpose=False) 方法: 结束 ====================================================================  {RESET}\n")

    # def set_pdeloss(self, names=None, num=1):
    #     if names is not None:
    #         self.loss_config += [{"name": name, "type": 'pde'} for name in names]
    #     else:
    #         self.loss_config += [{"name": f"pde_{i}", "type": 'pde'} for i in range(num)]

    def set_pdeloss(self, names=None, num=1):
        # 如果 names 不为 None
        if names is not None:
            # 遍历 names 列表
            for name in names:
                # 为每个名称创建一个字典
                config = {"name": name, "type": 'pde'}
                
                # 将字典添加到 self.loss_config 列表
                self.loss_config.append(config)
        else:
            # 如果 names 为 None
            # 遍历从 0 到 num - 1 的范围
            for i in range(num):
                # 为每个 PDE 损失创建一个字典
                config = {"name": f"pde_{i}", "type": 'pde'}
                
                # 将字典添加到 self.loss_config 列表
                self.loss_config.append(config)

    def add_bcs(self, config, geom=None):
        geom = geom if geom is not None else self.geom

        if self.bcs is None:
            self.bcs = []
        for bc in config:
            if bc.get('name') is None:
                bc['name'] = bc['type'] + ('' if bc['type'] == 'ic' else 'bc') + f"_{len(self.bcs) + 1}"
            if bc['type'] == 'dirichlet':
                self.bcs.append(dde.DirichletBC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'robin':
                self.bcs.append(dde.RobinBC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'ic':
                self.bcs.append(dde.IC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'operator':
                self.bcs.append(dde.OperatorBC(geom, bc['function'], bc['bc']))
            elif bc['type'] == 'neumann':
                self.bcs.append(dde.NeumannBC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'periodic':
                self.bcs.append(dde.PeriodicBC(geom, bc['component_x'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'pointset':
                self.bcs.append(dde.PointSetBC(bc['points'], bc['values'], component=bc['component']))
            else:
                raise ValueError(f"Unknown bc type: {bc['type']}")
            self.loss_config.append({'name': bc['name'], 'type': 'boundary'})

    def training_points(self, domain=DEFAULT_NUM_DOMAIN_POINTS, boundary=DEFAULT_NUM_BOUNDARY_POINTS, initial=DEFAULT_NUM_INITIAL_POINTS, test=DEFAULT_NUM_TEST_POINTS, mul=1):
        # print(f"\n{GRAY} class BasePDE 的 training_points(self, domain, boundary, test, mul) 方法: 开始 ====================================================================  {RESET}\n")
        if domain == "Default":
            domain = DEFAULT_NUM_DOMAIN_POINTS
        if boundary == "Default":
            boundary = DEFAULT_NUM_BOUNDARY_POINTS
        if test == "Default":
            test = DEFAULT_NUM_TEST_POINTS

        self.num_domain_points = domain * mul
        self.num_boundary_points = boundary * mul
        self.num_test_points = test * mul
        # print(f"{GRAY}")
        # print(f" domain = {domain}")
        # print(f" boundary = {boundary}")
        # print(f" test = {test}")
        # print(f" self.num_domain_points  = {self.num_domain_points}")
        # print(f" self.num_boundary_points = {self.num_boundary_points}")
        # print(f" self.num_test_points     = {self.num_test_points}")
        # print(f"{RESET}")
        # print(f"\n{GRAY} class BasePDE 的 training_points(self, domain, boundary, test, mul) 方法: 结束 ====================================================================  {RESET}\n")


    def check(self):
        if self.pde is None:
            raise ValueError("PDE could not be None")
        if self.geom is None:
            raise ValueError("geometry could not be None")
        if self.output_config is None:
            raise ValueError("output config could not be None, please set output dim or output config")
        if self.bbox is None:
            raise ValueError("bbox could not be None")
        if self.num_pde == 0:
            raise ValueError("No pde loss specified")

        for i in range(self.num_pde):
            if self.loss_config[i]['type'] != 'pde':
                raise ValueError("All PDE loss should be set before Boundary loss to avoid potential issues with methods like NTK")

    def use_gepinn(self):
        pde_original = self.pde # 将原始的 PDE 函数 self.pde 赋值给变量 pde_original

        # 定义一个新的函数 pde_wrapper,它接收输入坐标 x 和解 u,并在计算原始 PDE 函数的残差 res 之后,还计算 res 对输入坐标的梯度 res_g。
        def pde_wrapper(x, u): # add regularize terms to pde function
            res = pde_original(x, u) # 计算原始 PDE 残差

            # 如果 res 不是列表或元组, 则将 PDE 的残差 res 转换为列表
            if not isinstance(res, (list, tuple)):  # convert single pde loss to list
                res = [res]
            # 如果 res 是元组, 则将 PDE 的残差 res 转换为列表
            elif isinstance(res, tuple):  # convert tuple_like pde loss to list
                res = list(res)

            # 对于每个残差项 r,如果 r 的 shape 为  (batch_size,), 则将其 在最后一个维度上增加一个单位维度,使其形状变为 (batch_size, 1)
            for r in res:  # unsqueeze pde loss of shape (batch_size,) to shape (batch_size, 1)
                if r.dim() == 1:
                    r = r.unsqueeze(dim=-1)
                else:
                    assert r.dim() == 2 and r.shape[1] == 1, "improper pde residue shape"

            res_g = [] # 初始化一个空列表,用于存储梯度正则化项
            for j in range(self.input_dim): # 对于每个输入维度 j
                for i in range(len(res)):   # 对于每个残差项 res[i]
                    res_g.append(dde.grad.jacobian(res[i], x, i=0, j=j)) # 计算 res[i] 对输入坐标在第 j 个维度上的梯度,并添加到 res_g 列表中

            return res + res_g # 返回原始残差和梯度正则化项的连接列表

        self.pde = pde_wrapper # 将 self.pde 更新为新定义的 pde_wrapper 函数

        config = [] # 初始化一个空列表,用于存储新添加的梯度正则化项的配置
        for j in range(self.input_dim):
            for i in range(self.num_pde):
                config.append({"name": self.loss_config[i]['name'] + f"_grad{j}", "type": "gepinn"})  # 添加一个新的梯度正则化项的配置,类型为 'gepinn',名称为原始项名称加上 _grad{j}
        self.loss_config = self.loss_config[:self.num_pde] + config + self.loss_config[self.num_pde:] # 将新添加的梯度正则化项的配置插入到 self.loss_config 列表中,位置在原始 PDE 损失项之后

    def create_model(self, net):
        """
            这个函数用于创建一个 Model 对象,并将其返回
        """
        # print(f"\n{GRAY} class BasePDE 的 create_model(self, net) 方法: 开始 ====================================================================  {RESET}\n")
        self.check()
        self.net = net
        self.data = dde.data.PDE(
            self.geom,
            self.pde,
            self.bcs,
            num_domain=self.num_domain_points,
            num_boundary=self.num_boundary_points,
            num_test=self.num_test_points,
        )
        self.model = dde.Model(self.data, net)
        self.model.pde = self

        # print(f"{GRAY}")
        # print(f" type(self.net)       = {type(self.net)}")
        # print(f" type(self.data)      = {type(self.data)}")
        # print(f" type(self.model)     = {type(self.model)}")
        # print(f" type(self.model.pde) = {type(self.model.pde)}")
        # print(f" self.net       = {self.net}")
        # print(f" self.data      = {self.data}")
        # print(f" self.model     = {self.model}")
        # print(f" self.model.pde = {self.model.pde}")
        # print(f"{RESET}")

        # print(f"\n{GRAY} class BasePDE 的 create_model(self, net) 方法: 结束 ====================================================================  {RESET}\n")
        return self.model


class BaseTimePDE(BasePDE):

    def __init__(self):
        # print(f"\n{GRAY} class BaseTimePDE 的 __init__(self) 方法: 开始 ====================================================================  {RESET}\n")
        super().__init__()
        self.geomtime = None
        self.num_initial_points = DEFAULT_NUM_INITIAL_POINTS
        # print(f"{GRAY}")
        # # print(f" type(self.geomtime)           = {type(self.geomtime)}")
        # # print(f" type(self.num_domain_points)   = {type(self.num_domain_points)}")
        # # print(f" type(self.num_boundary_points) = {type(self.num_boundary_points)}")
        # # print(f" type(self.num_test_points)     = {type(self.num_test_points)}")
        # # print(f" type(self.num_initial_points) = {type(self.num_initial_points)}")
        # print(f"{RESET}")
        # print(f"\n{GRAY} class BaseTimePDE 的 __init__(self) 方法: 结束 ====================================================================  {RESET}\n")

    @property
    def input_dim(self):
        return self.geomtime.dim

    def add_bcs(self, config):
        super().add_bcs(config, self.geomtime)

    def load_ref_data(self, datapath, transform_fn=None, t_transpose=True):
        super(BaseTimePDE, self).load_ref_data(datapath, transform_fn, t_transpose)

    def training_points(
        self,
        domain=DEFAULT_NUM_DOMAIN_POINTS,
        boundary=DEFAULT_NUM_BOUNDARY_POINTS,
        initial=DEFAULT_NUM_INITIAL_POINTS,
        test=DEFAULT_NUM_TEST_POINTS,
        mul=1,
    ):
        # print(f"\n{GRAY} class BaseTimePDE 的 training_points(self, domain, boundary, initial, test, mul) 方法: 开始 ====================================================================  {RESET}\n")
        # 这个函数没有返回值，只是用来更新 self.num_domain_points, self.num_boundary_points, self.num_initial_points, self.num_test_points 这四个属性
        # 修改1: 如果输入参数为 None, 则使用默认值 ==== 开始 ====
        if domain == "Default":
            domain = DEFAULT_NUM_DOMAIN_POINTS
        if boundary == "Default":
            boundary = DEFAULT_NUM_BOUNDARY_POINTS
        if initial == "Default":
            initial = DEFAULT_NUM_INITIAL_POINTS
        if test == "Default":
            test = DEFAULT_NUM_TEST_POINTS
        # 修改1：如果输入参数为 None, 则使用默认值 ==== 结束 ====

        self.num_domain_points = domain * mul
        self.num_boundary_points = boundary * mul
        self.num_initial_points = initial * mul
        self.num_test_points = test * mul

        # # print(f"{GRAY}")
        # # print(f" type(self.num_domain_points)   = {type(self.num_domain_points)}")
        # # print(f" type(self.num_boundary_points) = {type(self.num_boundary_points)}")
        # # print(f" type(self.num_initial_points)  = {type(self.num_initial_points)}")
        # # print(f" type(self.num_test_points)     = {type(self.num_test_points)}")

        # # print(f" self.num_domain_points  = {self.num_domain_points}")
        # # print(f" self.num_boundary_points = {self.num_boundary_points}")
        # # print(f" self.num_initial_points  = {self.num_initial_points}")
        # # print(f" self.num_test_points     = {self.num_test_points}")
        # # print(f"{RESET}")
        # print(f"\n{GRAY} class BaseTimePDE 的 training_points(self, domain, boundary, initial, test, mul) 方法: 结束 ====================================================================  {RESET}\n")

    def create_model(self, net):
        """
            这个函数用于创建一个 Model 对象,并将其返回
        """
        # print(f"\n{GRAY} class BasePDE 的 create_model(self, net) 方法: 开始 ====================================================================  {RESET}\n")
        self.check()
        self.net = net
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.pde,
            self.bcs,
            num_domain=self.num_domain_points,
            num_boundary=self.num_boundary_points,
            num_initial=self.num_initial_points,
            num_test=self.num_test_points
        )
        self.model = dde.Model(self.data, net)
        self.model.pde = self

        # print(f"{GRAY}")
        # print(f" type(self.net)       = {type(self.net)}")
        # print(f" type(self.data)      = {type(self.data)}")
        # print(f" type(self.model)     = {type(self.model)}")
        # print(f" type(self.model.pde) = {type(self.model.pde)}")
        # print(f" self.net       = {self.net}")
        # print(f" self.data      = {self.data}")
        # print(f" self.model     = {self.model}")
        # print(f" self.model.pde = {self.model.pde}")
        # print(f"{RESET}")

        # print(f"\n{GRAY} class BasePDE 的 create_model(self, net) 方法: 结束 ====================================================================  {RESET}\n")
        return self.model
