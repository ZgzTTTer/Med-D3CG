import torch
import torch.nn.functional as F


class NonlinearFunction:
    """非线性函数基类"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def forward(self, x):
        """前向变换"""
        raise NotImplementedError
    
    def inverse(self, y):
        """逆变换"""
        raise NotImplementedError


class LinearFunction(NonlinearFunction):
    """线性函数（恒等变换）"""
    
    def forward(self, x):
        return x
    
    def inverse(self, y):
        return y


class TanhFunction(NonlinearFunction):
    """Tanh非线性函数"""
    
    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)
        self.alpha = alpha
    
    def forward(self, x):
        return torch.tanh(self.alpha * x)
    
    def inverse(self, y):
        # 限制y的范围以避免atanh的数值问题
        y_clamped = torch.clamp(y, -0.99, 0.99)
        return torch.atanh(y_clamped) / self.alpha


class SigmoidFunction(NonlinearFunction):
    """Sigmoid非线性函数"""
    
    def __init__(self, beta=1.0):
        super().__init__(beta=beta)
        self.beta = beta
    
    def forward(self, x):
        return 2 * torch.sigmoid(self.beta * x) - 1
    
    def inverse(self, y):
        # 将y从[-1,1]映射到(0,1)
        sigmoid_input = (y + 1) / 2
        # 限制范围以避免logit的数值问题
        sigmoid_input = torch.clamp(sigmoid_input, 1e-6, 1-1e-6)
        return torch.logit(sigmoid_input) / self.beta


class LeakyReLUFunction(NonlinearFunction):
    """LeakyReLU非线性函数"""
    
    def __init__(self, negative_slope=0.1, scale=1.0):
        super().__init__(negative_slope=negative_slope, scale=scale)
        self.negative_slope = negative_slope
        self.scale = scale
    
    def forward(self, x):
        positive_part = F.leaky_relu(x, negative_slope=self.negative_slope)
        negative_part = F.leaky_relu(-x, negative_slope=self.negative_slope)
        return self.scale * (positive_part - negative_part)
    
    def inverse(self, y):
        y_scaled = y / self.scale
        
        # 正值部分和负值部分的逆变换
        positive_mask = y_scaled >= 0
        negative_mask = y_scaled < 0
        
        result = torch.zeros_like(y_scaled)
        result[positive_mask] = y_scaled[positive_mask]
        result[negative_mask] = -y_scaled[negative_mask] / self.negative_slope
        
        return result


def get_nonlinear_function(func_type="linear", **kwargs):
    """工厂函数：根据类型创建非线性函数"""
    
    if func_type == "linear":
        return LinearFunction()
    elif func_type == "tanh":
        return TanhFunction(**kwargs)
    elif func_type == "sigmoid":
        return SigmoidFunction(**kwargs)
    elif func_type == "leaky_relu":
        return LeakyReLUFunction(**kwargs)
    else:
        raise ValueError(f"Unknown nonlinear function type: {func_type}")