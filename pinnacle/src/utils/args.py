import re

def parse_hidden_layers(hidden_layers):
    layers = []
    for s in re.split(r"[,_-]", hidden_layers):
        if '*' in s:
            siz, num = s.split('*')
            layers += [int(siz)] * int(num)
        else:
            layers += [int(s)]
    return layers


def parse_width_depth(width, depth):
    """
    根据给定的 width 和 depth 创建一个表示 hidden layer 的 list。

        ex: parse_width_depth(100, 5) -> [100, 100, 100, 100, 100]

    参数:
    width (int): 每个 hidden layer 的 neuron 数量。
    depth (int): number of hidden layers。

    返回:
    layers (list): 一个包含每个 hidden layers 中的 neuron 的数量的 list, 
                   (len(layers)) 会等于 depth。
    """
    layers = [width] * depth
    return layers

def parse_loss_weight(loss_weight): # ['1', '2', '3']
    """
        这个函数用来把输入的字符串或列表转换为一个列表
    """
    if loss_weight == 'none':
        return None
    elif isinstance(loss_weight, str):
        weights = []
        for s in re.split(r"[,_-]", loss_weight):
            weights.append(float(s))
        return weights
    elif isinstance(loss_weight, list):
        return [float(w) for w in loss_weight]
    else:
        raise ValueError(f"Invalid input type for loss_weight: {type(loss_weight)}")
