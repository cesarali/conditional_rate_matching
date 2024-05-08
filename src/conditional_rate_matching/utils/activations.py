from torch import nn

def get_activation_function(name: str = 'ReLU'):
    if name is not None:
        activation_functions = {"ReLU": nn.ReLU(),
                                "LeakyReLU": nn.LeakyReLU(),
                                "ELU": nn.ELU(),
                                "SELU": nn.SELU(),
                                "GLU": nn.GLU(),
                                "GELU": nn.GELU(),
                                "CELU": nn.CELU(),
                                "PReLU": nn.PReLU(),
                                "Sigmoid": nn.Sigmoid(),
                                "Tanh": nn.Tanh(),
                                "Hardswish": nn.Hardswish(),
                                "Hardtanh": nn.Hardtanh(),
                                "LogSigmoid": nn.LogSigmoid(),
                                "Softplus": nn.Softplus(),
                                "Softsign": nn.Softsign(),
                                "Softshrink": nn.Softshrink(),
                                "Softmin": nn.Softmin(),
                                "Softmax": nn.Softmax()}
        return activation_functions[name]
    else:
        return None