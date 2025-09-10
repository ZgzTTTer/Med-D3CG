"""
模型工厂：统一创建不同类型的扩散模型
"""

from .gaussian import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond
from .d3cg_unified import create_d3cg_trainer, create_d3cg_sampler
from .ablation.DIFFDDPM import DiffDDPMTrainer_cond, DiffDDPMSampler_cond
from .ablation.wavelet import WTDDPMTrainer_cond, WTDDPMSampler_cond
from .attn.midattnDDPM import MidAttnDDPMTrainer_cond, MidAttnDDPMSampler_cond
from .attn.upattnDDPM import UpAttnDDPMTrainer_cond, UpAttnDDPMSampler_cond


# D3CG模型配置
D3CG_CONFIGS = {
    # 基础D3CG模型（不同小波）
    "D3CG_haar": {
        "wave_type": "haar",
        "nonlinear_type": "linear",
        "transform_levels": 1,
        "backend": "pytorch"
    },
    "D3CG_db4": {
        "wave_type": "db4", 
        "nonlinear_type": "linear",
        "transform_levels": 1,
        "backend": "pytorch"
    },
    "D3CG_coif3": {
        "wave_type": "coif3",
        "nonlinear_type": "linear", 
        "transform_levels": 1,
        "backend": "pytorch"
    },
    
    # 非线性D3CG模型
    "D3CG_tanh": {
        "wave_type": "haar",
        "nonlinear_type": "tanh",
        "transform_levels": 1,
        "backend": "pytorch",
        "nonlinear_params": {"alpha": 1.0}
    },
    "D3CG_sigmoid": {
        "wave_type": "haar", 
        "nonlinear_type": "sigmoid",
        "transform_levels": 1,
        "backend": "pytorch",
        "nonlinear_params": {"beta": 1.0}
    },
    "D3CG_leaky_relu": {
        "wave_type": "haar",
        "nonlinear_type": "leaky_relu", 
        "transform_levels": 1,
        "backend": "pytorch",
        "nonlinear_params": {"negative_slope": 0.1, "scale": 1.0}
    },
    
    # 两层小波变换模型
    "D3CG_twice_haar": {
        "wave_type": "haar",
        "nonlinear_type": "linear",
        "transform_levels": 2,
        "backend": "pytorch"
    },
    "D3CG_twice_db4": {
        "wave_type": "db4",
        "nonlinear_type": "linear", 
        "transform_levels": 2,
        "backend": "pytorch"
    },
    "D3CG_twice_coif3": {
        "wave_type": "coif3",
        "nonlinear_type": "linear",
        "transform_levels": 2, 
        "backend": "pytorch"
    },
}


def get_trainer_sampler(model_name, net_model, beta_1, beta_T, T, device, **kwargs):
    """
    统一的训练器和采样器工厂函数
    
    Args:
        model_name: 模型名称
        net_model: 网络模型
        beta_1, beta_T, T: 扩散参数
        device: 设备
        **kwargs: 额外参数（如非线性函数参数）
    
    Returns:
        (trainer, sampler): 训练器和采样器元组
    """
    
    # 基础DDPM模型
    if model_name == "DDPM":
        trainer = GaussianDiffusionTrainer_cond(net_model, beta_1, beta_T, T).to(device)
        sampler = GaussianDiffusionSampler_cond(net_model, beta_1, beta_T, T).to(device)
    
    elif model_name == "DIFFDDPM":
        trainer = DiffDDPMTrainer_cond(net_model, beta_1, beta_T, T).to(device)
        sampler = DiffDDPMSampler_cond(net_model, beta_1, beta_T, T).to(device)
    
    # 注意力模型
    elif model_name == "midattnDDPM":
        trainer = MidAttnDDPMTrainer_cond(net_model, beta_1, beta_T, T).to(device)
        sampler = MidAttnDDPMSampler_cond(net_model, beta_1, beta_T, T, 
                                        kwargs.get("psi", 1.0), kwargs.get("s", 0.1)).to(device)
    
    elif model_name == "UpAttnDDPM":
        trainer = UpAttnDDPMTrainer_cond(net_model, beta_1, beta_T, T).to(device)
        sampler = UpAttnDDPMSampler_cond(net_model, beta_1, beta_T, T,
                                       kwargs.get("psi", 1.0), kwargs.get("s", 0.1)).to(device)
    
    # 小波模型
    elif model_name == "WTDDPM":
        trainer = WTDDPMTrainer_cond(net_model, beta_1, beta_T, T).to(device)
        sampler = WTDDPMSampler_cond(net_model, beta_1, beta_T, T).to(device)
    
    # D3CG模型系列
    elif model_name in D3CG_CONFIGS:
        config = D3CG_CONFIGS[model_name].copy()
        
        # 更新配置中的非线性参数
        if "nonlinear_params" in config:
            nonlinear_params = config["nonlinear_params"].copy()
            # 根据非线性函数类型，只更新对应的参数
            nonlinear_type = config.get("nonlinear_type", "linear")
            if nonlinear_type == "tanh" and "alpha" in kwargs:
                nonlinear_params["alpha"] = kwargs["alpha"]
            elif nonlinear_type == "sigmoid" and "beta" in kwargs:
                nonlinear_params["beta"] = kwargs["beta"]
            elif nonlinear_type == "leaky_relu":
                if "negative_slope" in kwargs:
                    nonlinear_params["negative_slope"] = kwargs["negative_slope"]
                if "scale" in kwargs:
                    nonlinear_params["scale"] = kwargs["scale"]
            config["nonlinear_params"] = nonlinear_params
        
        trainer = create_d3cg_trainer(net_model, beta_1, beta_T, T, config).to(device)
        sampler = create_d3cg_sampler(net_model, beta_1, beta_T, T, config).to(device)
    
    # 自定义D3CG配置
    elif model_name == "D3CG_custom":
        nonlinear_type = kwargs.get("nonlinear_type", "linear")
        
        # 根据非线性函数类型，只传递对应的参数
        nonlinear_params = {}
        if nonlinear_type == "tanh" and "alpha" in kwargs:
            nonlinear_params["alpha"] = kwargs["alpha"]
        elif nonlinear_type == "sigmoid" and "beta" in kwargs:
            nonlinear_params["beta"] = kwargs["beta"]
        elif nonlinear_type == "leaky_relu":
            if "negative_slope" in kwargs:
                nonlinear_params["negative_slope"] = kwargs["negative_slope"]
            if "scale" in kwargs:
                nonlinear_params["scale"] = kwargs["scale"]
        
        config = {
            "wave_type": kwargs.get("wave_type", "haar"),
            "nonlinear_type": nonlinear_type,
            "transform_levels": kwargs.get("transform_levels", 1),
            "backend": kwargs.get("backend", "pytorch"),
            "nonlinear_params": nonlinear_params
        }
        trainer = create_d3cg_trainer(net_model, beta_1, beta_T, T, config).to(device)
        sampler = create_d3cg_sampler(net_model, beta_1, beta_T, T, config).to(device)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return trainer, sampler


def get_available_models():
    """获取所有可用的模型名称"""
    base_models = ["DDPM", "DIFFDDPM", "midattnDDPM", "UpAttnDDPM", "WTDDPM"]
    d3cg_models = list(D3CG_CONFIGS.keys())
    custom_models = ["D3CG_custom"]
    
    return {
        "base_models": base_models,
        "d3cg_models": d3cg_models, 
        "custom_models": custom_models,
        "all_models": base_models + d3cg_models + custom_models
    }


def print_model_info():
    """打印所有可用模型的信息"""
    models = get_available_models()
    
    print("=== Available Models ===")
    print("\n1. Base Models:")
    for model in models["base_models"]:
        print(f"  - {model}")
    
    print("\n2. D3CG Models:")
    for model in models["d3cg_models"]:
        config = D3CG_CONFIGS[model]
        print(f"  - {model}: {config['wave_type']} wavelet, {config['nonlinear_type']} nonlinear, {config['transform_levels']} levels")
    
    print("\n3. Custom Models:")
    for model in models["custom_models"]:
        print(f"  - {model}: Configurable D3CG model")
    
    print(f"\nTotal: {len(models['all_models'])} models available")