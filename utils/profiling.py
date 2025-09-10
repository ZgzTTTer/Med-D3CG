import torch
import time
import logging
from thop import profile
import psutil
import os


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_flops(model, input_shape, device):
    """计算模型FLOPS"""
    try:
        # 创建随机输入
        dummy_input = torch.randn(input_shape).to(device)
        
        # 计算FLOPS
        flops, params = profile(model, inputs=(dummy_input, torch.tensor([0]).to(device)))
        return flops, params
    except Exception as e:
        logging.warning(f"FLOPS calculation failed: {e}")
        return 0, 0


def get_gpu_memory_usage():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        # 获取当前GPU显存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        free_memory = total_memory - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total_memory,
            'free': free_memory
        }
    else:
        return None


def measure_inference_time(model, input_shape, device, num_runs=10):
    """测量推理时间"""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    dummy_t = torch.tensor([0]).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input, dummy_t)
    
    # 测量时间
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input, dummy_t)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    return avg_time


def profile_model(model, input_shape, device, model_name):
    """完整的模型性能分析"""
    print("=" * 60)
    print(f"Model Profiling for {model_name}")
    print("=" * 60)
    
    # 1. 参数量统计
    total_params, trainable_params = count_parameters(model)
    print(f"📊 Model Parameters:")
    print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print()
    
    # 2. GPU显存使用情况（模型加载前）
    memory_before = get_gpu_memory_usage()
    
    # 将模型移到GPU（如果还没有的话）
    model = model.to(device)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # GPU显存使用情况（模型加载后）
    memory_after = get_gpu_memory_usage()
    
    if memory_before and memory_after:
        model_memory = memory_after['allocated'] - memory_before['allocated']
        print(f"💾 GPU Memory Usage:")
        print(f"  Before Model: {memory_before['allocated']:.1f}MB allocated, {memory_before['reserved']:.1f}MB reserved")
        print(f"  After Model:  {memory_after['allocated']:.1f}MB allocated, {memory_after['reserved']:.1f}MB reserved")
        print(f"  Model Memory: {model_memory:.1f}MB")
        print(f"  Total GPU Memory: {memory_after['total']:.1f}MB")
        print(f"  Free Memory: {memory_after['free']:.1f}MB")
        print()
    
    # 3. FLOPS计算
    flops, _ = calculate_flops(model, input_shape, device)
    if flops > 0:
        print(f"⚡ Performance Metrics:")
        print(f"  FLOPs: {flops/1e9:.3f}G ({flops:,})")
    
    # 4. 推理时间测量
    try:
        inference_time = measure_inference_time(model, input_shape, device)
        print(f"  Inference Time: {inference_time:.2f}ms")
    except Exception as e:
        print(f"  Inference Time: Failed to measure ({e})")
    
    print()
    
    # 5. 输入输出形状信息
    try:
        dummy_input = torch.randn(input_shape).to(device)
        dummy_t = torch.tensor([0]).to(device)
        with torch.no_grad():
            output = model(dummy_input, dummy_t)
            if isinstance(output, tuple):
                output_shape = output[0].shape
            else:
                output_shape = output.shape
        print(f"📐 Input/Output Shapes:")
        print(f"  Input Shape: {list(input_shape)}")
        print(f"  Output Shape: {list(output_shape)}")
    except Exception as e:
        print(f"📐 Shape Info: Failed to get output shape ({e})")
    
    print("=" * 60)
    print()
    
    # 返回统计信息用于日志记录
    results = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params/1e6,
        'trainable_params_M': trainable_params/1e6,
        'flops': flops,
        'flops_G': flops/1e9 if flops > 0 else 0,
        'model_memory_MB': model_memory if memory_before and memory_after else 0,
        'inference_time_ms': inference_time if 'inference_time' in locals() else 0
    }
    
    return results


def log_profiling_results(results, model_name, log_file_path):
    """将性能分析结果写入日志文件"""
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Model Profiling Results for {model_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total Parameters: {results['total_params']:,} ({results['total_params_M']:.2f}M)\n")
            f.write(f"Trainable Parameters: {results['trainable_params']:,} ({results['trainable_params_M']:.2f}M)\n")
            f.write(f"FLOPs: {results['flops_G']:.3f}G ({results['flops']:,})\n")
            f.write(f"Model Memory: {results['model_memory_MB']:.1f}MB\n")
            f.write(f"Inference Time: {results['inference_time_ms']:.2f}ms\n")
            f.write(f"{'='*60}\n\n")
    except Exception as e:
        logging.warning(f"Failed to write profiling results to log: {e}")


def get_model_input_shape(model_name, dataset_type, image_size, batch_size=1):
    """根据模型名称和数据集类型确定输入形状"""
    # 确定通道数
    if dataset_type in ["mripet", "mrispect"]:
        in_channels = 6  # RGB + RGB
    else:
        in_channels = 2  # Two grayscale images
    
    # 对于小波模型，输入形状会有所不同
    if model_name in ["WTDDPM"] or model_name.startswith("D3CG"):
        # 小波模型的输入形状
        if dataset_type in ["mripet", "mrispect"]:
            # RGB图像的小波系数：每个通道4个系数(LL, LH, HL, HH)，两个图像共6个通道
            wavelet_channels = 24  # 6 channels * 4 coefficients
        else:
            # 灰度图像的小波系数：每个通道4个系数，两个图像共2个通道
            wavelet_channels = 8   # 2 channels * 4 coefficients
        
        # 小波变换后的尺寸是原来的一半
        return (batch_size, wavelet_channels, image_size//2, image_size//2)
    else:
        # 普通模型的输入形状
        return (batch_size, in_channels, image_size, image_size)