import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import extract
from .wavelet_transforms import get_wavelet_transform
from .nonlinear_functions import get_nonlinear_function


class D3CGUnifiedTrainer_cond(nn.Module):
    """统一的D3CG训练器，支持不同小波类型、激活函数和变换层数"""
    
    def __init__(self, model, beta_1, beta_T, T, 
                 wave_type="haar", 
                 nonlinear_type="linear", 
                 transform_levels=1,
                 backend="pytorch",
                 **nonlinear_kwargs):
        super().__init__()
        self.model = model
        self.T = T
        self.wave_type = wave_type
        self.nonlinear_type = nonlinear_type
        self.transform_levels = transform_levels
        self.backend = backend
        
        # 小波变换对象（延迟初始化）
        self._wavelet_transform = None
        
        # 非线性函数
        self.nonlinear_func = get_nonlinear_function(nonlinear_type, **nonlinear_kwargs)
        
        # 扩散调度参数
        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
    
    def get_wavelet_transform(self, device):
        """获取小波变换对象（延迟初始化）"""
        if self._wavelet_transform is None:
            self._wavelet_transform = get_wavelet_transform(
                wave_type=self.wave_type,
                transform_levels=self.transform_levels,
                device=device,
                backend=self.backend
            )
        return self._wavelet_transform
    
    def forward(self, x0):
        B = x0.size(0)
        device = x0.device
        t = torch.randint(self.T, (B,), device=device)
        
        # 获取小波变换对象
        wt = self.get_wavelet_transform(device)
        
        # 分离CT和CBCT
        C_out = getattr(self.model, "out_channels", 1)
        ct, cbct = x0[:, :C_out], x0[:, C_out:]
        
        # 小波域变换
        ct_coeffs = wt.forward_transform(ct)
        cbct_coeffs = wt.forward_transform(cbct)
        
        # 确保小波系数尺寸一致
        if ct_coeffs.shape != cbct_coeffs.shape:
            target_shape = ct_coeffs.shape[2:]
            cbct_coeffs = F.interpolate(cbct_coeffs, size=target_shape, mode='bilinear', align_corners=False)
        
        # 计算差异域（应用非线性变换）
        linear_diff = ct_coeffs - cbct_coeffs
        d0 = self.nonlinear_func.forward(linear_diff)
        
        # 扩散过程
        noise = torch.randn_like(d0)
        d_t = extract(self.sqrt_alphas_bar, t, d0.shape) * d0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, d0.shape) * noise
        
        # 模型输入
        model_input = torch.cat([d_t, cbct_coeffs], dim=1)
        eps_theta = self.model(model_input, t)
        
        return F.mse_loss(eps_theta, noise, reduction="sum")


class D3CGUnifiedSampler_cond(nn.Module):
    """统一的D3CG采样器"""
    
    def __init__(self, model, beta_1, beta_T, T,
                 wave_type="haar",
                 nonlinear_type="linear",
                 transform_levels=1,
                 backend="pytorch",
                 **nonlinear_kwargs):
        super().__init__()
        self.model = model
        self.T = T
        self.wave_type = wave_type
        self.nonlinear_type = nonlinear_type
        self.transform_levels = transform_levels
        self.backend = backend
        
        # 小波变换对象（延迟初始化）
        self._wavelet_transform = None
        
        # 非线性函数
        self.nonlinear_func = get_nonlinear_function(nonlinear_type, **nonlinear_kwargs)
        
        # 扩散调度参数
        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        
        # 后验方差
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
    
    def get_wavelet_transform(self, device):
        """获取小波变换对象（延迟初始化）"""
        if self._wavelet_transform is None:
            self._wavelet_transform = get_wavelet_transform(
                wave_type=self.wave_type,
                transform_levels=self.transform_levels,
                device=device,
                backend=self.backend
            )
        return self._wavelet_transform
    
    def p_mean_variance(self, d_t, cond, t):
        """计算去噪步骤的均值和方差"""
        eps = self.model(torch.cat([d_t, cond], dim=1), t)
        
        sqrt_alphas_cumprod_t = extract(torch.sqrt(self.alphas_cumprod), t, d_t.shape)
        sqrt_one_minus = extract(torch.sqrt(1 - self.alphas_cumprod), t, d_t.shape)
        
        # 预测的d0
        d0_pred = (d_t - sqrt_one_minus * eps) / sqrt_alphas_cumprod_t
        
        # 计算均值
        model_mean = (1.0 / torch.sqrt(extract(self.alphas, t, d_t.shape))) * (
            d_t - extract(self.betas, t, d_t.shape) * eps / sqrt_one_minus)
        
        # 方差
        model_var = extract(self.posterior_variance, t, d_t.shape)
        
        return model_mean, model_var, d0_pred
    
    def forward(self, cbct):
        with torch.no_grad():
            B = cbct.size(0)
            device = cbct.device
            
            # 获取小波变换对象
            wt = self.get_wavelet_transform(device)
            
            # CBCT的小波变换
            C_out = getattr(self.model, "out_channels", 1)
            cbct = cbct[:, :C_out]
            cbct_coeffs = wt.forward_transform(cbct)
            
            # 初始化噪声
            d_t = torch.randn_like(cbct_coeffs)
            
            # 逆扩散过程
            for time in reversed(range(self.T)):
                t = torch.full((B,), time, dtype=torch.long, device=device)
                mean, var, _ = self.p_mean_variance(d_t, cbct_coeffs, t)
                
                if time > 0:
                    d_t = mean + torch.sqrt(var) * torch.randn_like(d_t)
                else:
                    d_t = mean
            
            # 从差异域恢复CT系数
            if self.nonlinear_type == "linear":
                ct_coeffs = d_t + cbct_coeffs
            else:
                # 应用非线性函数的逆变换
                linear_diff = self.nonlinear_func.inverse(d_t)
                ct_coeffs = cbct_coeffs + linear_diff
            
            # 小波逆变换
            ct_reconstructed = wt.inverse_transform(ct_coeffs, cbct.shape[2:])
            
            # 确保重建图像尺寸正确
            if ct_reconstructed.shape[2:] != cbct.shape[2:]:
                ct_reconstructed = F.interpolate(ct_reconstructed, size=cbct.shape[2:], 
                                               mode='bilinear', align_corners=False)
            
            # 拼接结果
            result = torch.cat([ct_reconstructed, cbct], dim=1)
            return torch.clamp(result, -1.0, 1.0)


# 便利函数：创建特定配置的训练器和采样器
def create_d3cg_trainer(model, beta_1, beta_T, T, config):
    """根据配置创建D3CG训练器"""
    return D3CGUnifiedTrainer_cond(
        model=model,
        beta_1=beta_1,
        beta_T=beta_T,
        T=T,
        wave_type=config.get("wave_type", "haar"),
        nonlinear_type=config.get("nonlinear_type", "linear"),
        transform_levels=config.get("transform_levels", 1),
        backend=config.get("backend", "pytorch"),
        **config.get("nonlinear_params", {})
    )


def create_d3cg_sampler(model, beta_1, beta_T, T, config):
    """根据配置创建D3CG采样器"""
    return D3CGUnifiedSampler_cond(
        model=model,
        beta_1=beta_1,
        beta_T=beta_T,
        T=T,
        wave_type=config.get("wave_type", "haar"),
        nonlinear_type=config.get("nonlinear_type", "linear"),
        transform_levels=config.get("transform_levels", 1),
        backend=config.get("backend", "pytorch"),
        **config.get("nonlinear_params", {})
    )