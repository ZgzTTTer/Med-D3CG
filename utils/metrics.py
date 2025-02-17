from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error
import numpy as np
import lpips
from pytorch_fid import fid_score
import torch
import os
from PIL import Image

def calculate_metrics(generated_image, target_image):
    """Calculate SSIM, PSNR, MSE and MAE metrics between generated and target images"""
    generated_image_norm = (generated_image + 1) / 2
    target_image_norm = (target_image + 1) / 2
    
    generated_image_norm = np.clip(generated_image_norm, 0, 1)
    target_image_norm = np.clip(target_image_norm, 0, 1)
    
    ssim_value = ssim(generated_image_norm, target_image_norm, data_range=1)
    psnr_value = psnr(target_image_norm, generated_image_norm, data_range=1)
    mse_value = mean_squared_error(target_image_norm, generated_image_norm)
    mae_value = np.mean(np.abs(target_image_norm - generated_image_norm))
    
    return {
        'ssim': ssim_value,
        'psnr': psnr_value,
        'mse': mse_value,
        'mae': mae_value
    }

def calculate_lpips(generated_tensor, target_tensor, device):
    """Calculate LPIPS metric between generated and target tensors"""
    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    return lpips_metric(generated_tensor, target_tensor).item()

def calculate_fid(real_images_dir, generated_images_dir, batch_size, device):
    """Calculate FID score between real and generated images"""
    return fid_score.calculate_fid_given_paths(
        [real_images_dir, generated_images_dir],
        batch_size,
        device,
        dims=2048
    )

def save_images(images, output_dir, prefix):
    """Save images to specified directory with given prefix"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, image in enumerate(images):
        image_uint8 = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_uint8, mode='L')
        image_path = os.path.join(output_dir, f'{prefix}_{idx+1}.png')
        image_pil.save(image_path)