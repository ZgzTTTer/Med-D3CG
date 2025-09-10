"""
使用示例：展示如何使用重构后的D3CG模型
"""

import torch
from models.wavelet import WTUNet
from diffusion.model_factory import get_trainer_sampler, print_model_info

def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 显示所有可用模型
    print_model_info()
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_model = WTUNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                      num_res_blocks=2, dropout=0.3,
                      in_channels=2, out_channels=1).to(device)
    
    # 使用预定义的D3CG配置
    trainer, sampler = get_trainer_sampler(
        model_name="D3CG_db4",
        net_model=net_model,
        beta_1=1e-4,
        beta_T=0.02,
        T=1000,
        device=device
    )
    
    print(f"创建了 D3CG_db4 模型")
    print(f"训练器类型: {type(trainer).__name__}")
    print(f"采样器类型: {type(sampler).__name__}")


def example_custom_d3cg():
    """自定义D3CG模型示例"""
    print("\n=== 自定义D3CG模型示例 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_model = WTUNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                      num_res_blocks=2, dropout=0.3,
                      in_channels=2, out_channels=1).to(device)
    
    # 自定义配置：使用coif3小波 + tanh非线性 + 2层变换
    trainer, sampler = get_trainer_sampler(
        model_name="D3CG_custom",
        net_model=net_model,
        beta_1=1e-4,
        beta_T=0.02,
        T=1000,
        device=device,
        # 自定义参数
        wave_type="coif3",
        nonlinear_type="tanh",
        transform_levels=2,
        alpha=2.0  # tanh的陡峭程度参数
    )
    
    print(f"创建了自定义D3CG模型:")
    print(f"  小波类型: coif3")
    print(f"  非线性函数: tanh (alpha=2.0)")
    print(f"  变换层数: 2")


def example_different_configurations():
    """不同配置示例"""
    print("\n=== 不同配置示例 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_model = WTUNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                      num_res_blocks=2, dropout=0.3,
                      in_channels=2, out_channels=1).to(device)
    
    # 配置1: Haar小波 + Sigmoid非线性
    print("配置1: Haar + Sigmoid")
    trainer1, sampler1 = get_trainer_sampler(
        model_name="D3CG_custom",
        net_model=net_model,
        beta_1=1e-4, beta_T=0.02, T=1000, device=device,
        wave_type="haar",
        nonlinear_type="sigmoid",
        beta=1.5
    )
    
    # 配置2: DB4小波 + LeakyReLU非线性
    print("配置2: DB4 + LeakyReLU")
    trainer2, sampler2 = get_trainer_sampler(
        model_name="D3CG_custom",
        net_model=net_model,
        beta_1=1e-4, beta_T=0.02, T=1000, device=device,
        wave_type="db4",
        nonlinear_type="leaky_relu",
        negative_slope=0.2,
        scale=1.5
    )
    
    # 配置3: 两层小波变换
    print("配置3: Coif3 + 两层变换")
    trainer3, sampler3 = get_trainer_sampler(
        model_name="D3CG_custom",
        net_model=net_model,
        beta_1=1e-4, beta_T=0.02, T=1000, device=device,
        wave_type="coif3",
        nonlinear_type="linear",
        transform_levels=2
    )


def example_training_loop():
    """训练循环示例"""
    print("\n=== 训练循环示例 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_model = WTUNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                      num_res_blocks=2, dropout=0.3,
                      in_channels=2, out_channels=1).to(device)
    
    # 创建训练器
    trainer, sampler = get_trainer_sampler(
        model_name="D3CG_db4",
        net_model=net_model,
        beta_1=1e-4, beta_T=0.02, T=1000, device=device
    )
    
    # 模拟训练数据
    batch_size = 2
    image_size = 256
    condition = torch.randn(batch_size, 1, image_size, image_size).to(device)
    target = torch.randn(batch_size, 1, image_size, image_size).to(device)
    x_0 = torch.cat((target, condition), 1)
    
    # 训练步骤
    print("执行训练步骤...")
    loss = trainer(x_0)
    print(f"训练损失: {loss.item():.4f}")
    
    # 采样步骤
    print("执行采样步骤...")
    with torch.no_grad():
        generated = sampler(condition)
    print(f"生成图像形状: {generated.shape}")


def example_command_line_usage():
    """命令行使用示例"""
    print("\n=== 命令行使用示例 ===")
    
    examples = [
        # 基础D3CG模型
        "python train.py --model_name D3CG_haar --dataset_type ctmri",
        "python train.py --model_name D3CG_db4 --dataset_type ctmri",
        "python train.py --model_name D3CG_coif3 --dataset_type ctmri",
        
        # 非线性D3CG模型
        "python train.py --model_name D3CG_tanh --alpha 2.0 --dataset_type ctmri",
        "python train.py --model_name D3CG_sigmoid --beta 1.5 --dataset_type ctmri",
        "python train.py --model_name D3CG_leaky_relu --negative_slope 0.2 --scale 1.5 --dataset_type ctmri",
        
        # 两层小波变换
        "python train.py --model_name D3CG_twice_haar --dataset_type ctmri",
        "python train.py --model_name D3CG_twice_db4 --dataset_type ctmri",
        
        # 自定义配置
        "python train.py --model_name D3CG_custom --wave_type coif3 --nonlinear_type tanh --transform_levels 2 --alpha 1.5 --dataset_type ctmri",
        
        # 显示所有可用模型
        "python train.py --list_models"
    ]
    
    print("命令行使用示例:")
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example}")


if __name__ == "__main__":
    example_basic_usage()
    example_custom_d3cg()
    example_different_configurations()
    example_training_loop()
    example_command_line_usage()