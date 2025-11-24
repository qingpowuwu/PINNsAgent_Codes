from collections import OrderedDict
import torch
import sys
sys.path.append('/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagenet_progress/pinnacle') # Add this line
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde.nn.pytorch.nn import NN
from utils.util_colors import RED, GRAY, BLUE, YELLOW, GREEN, RESET
import inspect


class LAAFlayer(torch.nn.Module):

    def __init__(self, n, a, dim_in, dim_out, activation):
        super(LAAFlayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n = n
        self.a = a
        self.activation = activation

        self.fc = torch.nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.n * torch.mul(self.a, x1)
        out = self.activation(x2)
        return out

#%%

# n layers deep neural network with LAAF
class DNN_LAAF(NN): # LAAF (Locally Adaptive Activation Functions)

    def __init__(self, n_layers, n_hidden, x_dim=1, u_dim=1, activation=None, kernel_initializer=None):
        """
            In LAAF, each neuron has its own adaptive parameter vector self.a of size (self.n_layers, self.n_hidden)
            Both LAAF and GAAF use the same adaptive activation function formulation, which is self.activation(self.n * self.a * x), 
                where self.n is a fixed scalar (set to 10 in the code), 
                      self.a is the adaptive parameter (vector for LAAF, scalar for GAAF),      
                            ex1 对于 LAAF: self.a.shape = (n_layers, n_hidden) = (5, 256), self.a[0, :].shape = (256,)
                            ex2 对于 GAAF: self.a.shape = (n_layers, 1)        = (5, 1),   self.a[0].shape = (1,)
                      x is the input to the neuron, 
                            ex: x.shape = (batch_size, x_dim) = (10, 2)
                      self.activation is the base activation function (e.g., tanh, relu, sin, etc.).
        """
        super(DNN_LAAF, self).__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        if activation is None:
            self.activation = activations.get("tanh")
        else:
            self.activation = activations.get(activation)
        self.regularizer = None

        self.a = torch.nn.Parameter(torch.empty(size=(self.n_layers, self.n_hidden))) # self.a.shape = (n_layers, n_hidden) = (5, 256), self.a[0, :].shape = (256,)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer(self.a.data)
                
        layer_list = list()
        layer_list.append(('layer0', LAAFlayer(n=10, a=self.a[0, :], dim_in=x_dim, dim_out=n_hidden, activation=self.activation)))
        for i in range(self.n_layers - 1):
            layer_list.append(('layer%d' % (i + 1), LAAFlayer(n=10, a=self.a[i + 1, :], dim_in=n_hidden, dim_out=n_hidden, activation=self.activation)))
        layer_list.append(('layer%d' % n_layers, torch.nn.Linear(self.n_hidden, self.u_dim)))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class DNN_GAAF(NN): # GAAF (Globally Adaptive Activation Functions).

    def __init__(self, n_layers, n_hidden, x_dim=1, u_dim=1, activation=None, kernel_initializer=None):
        """
            In GAAF, there is a single adaptive scalar parameter self.a of size (self.n_layers, 1), shared across all neurons in each layer.
            Both LAAF and GAAF use the same adaptive activation function formulation, which is self.activation(self.n * self.a * x), 
                where self.n is a fixed scalar (set to 10 in the code), 
                      self.a is the adaptive parameter (vector for LAAF, scalar for GAAF),      
                            ex1 对于 LAAF: self.a.shape = (n_layers, n_hidden) = (5, 256), self.a[0, :].shape = (256,)
                            ex2 对于 GAAF: self.a.shape = (n_layers, 1)        = (5, 1),   self.a[0].shape = (1,)
                      x is the input to the neuron, 
                            ex: x.shape = (batch_size, x_dim) = (10, 2)
                      self.activation is the base activation function (e.g., tanh, relu, sin, etc.).
        """
        super(DNN_GAAF, self).__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        if activation is None:
            self.activation = activations.get("tanh")
        else:
            self.activation = activations.get(activation)

        self.regularizer = None

        self.a = torch.nn.Parameter(torch.empty(size=(self.n_layers, 1))) # 初始化自适应参数标量, self.a.shape = (n_layers, 1) = (5, 1), self.a[0].shape = (1,)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer(self.a.data)

        layer_list = list()
        layer_list.append(('layer0', LAAFlayer(n=10, a=self.a[0], dim_in=x_dim, dim_out=n_hidden, activation=self.activation)))
        for i in range(self.n_layers - 1):
            layer_list.append(('layer%d' % (i + 1), LAAFlayer(10, self.a[i + 1], n_hidden, n_hidden, self.activation)))
        layer_list.append(('layer%d' % n_layers, torch.nn.Linear(self.n_hidden, self.u_dim)))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
