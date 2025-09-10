import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from models.base import BaseUNet
from models.attention import MidAttnUNet, UpAttnUNet
from models.wavelet import WTUNet
from diffusion.model_factory import get_trainer_sampler, get_available_models
from data.dataset import FootDataset2, MRIPET, MRISPECT, CTMRI
from torch.utils.data import DataLoader
from PIL import Image
import lpips
from pytorch_fid import fid_score
import tempfile
import warnings
import time
import argparse

warnings.filterwarnings("ignore")

# 添加性能分析相关导入
from utils.profiling import profile_model, log_profiling_results, get_model_input_shape


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model configuration
    available_models = get_available_models()["all_models"]
    parser.add_argument("--model_name", type=str, default="D3CG_db4",
                        choices=available_models + ["D3CG_custom"])
    
    # D3CG自定义配置参数
    parser.add_argument("--wave_type", type=str, default="haar",
                        choices=["haar", "db4", "coif3", "bior2.2", "dmey"],
                        help="Wavelet type for D3CG models")
    parser.add_argument("--nonlinear_type", type=str, default="linear",
                        choices=["linear", "tanh", "sigmoid", "leaky_relu"],
                        help="Nonlinear function type for D3CG models")
    parser.add_argument("--transform_levels", type=int, default=1,
                        help="Number of wavelet transform levels")
    
    # Dataset configuration
    parser.add_argument("--dataset_type", type=str, default="ctmri",
                        choices=["foot", "mripet", "mrispect", "ctmri"])
    parser.add_argument("--dataset_name", type=str, default="../datasets/HavardMedicalImage/PET-MRI-gray/test/")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--sample_num", type=int, default=12)
    
    # Model weights
    parser.add_argument("--save_weight_dir", type=str, default="./results/MRI-PET/D3CG_db4")
    parser.add_argument("--model_weight_path", type=str, default="./results/MRI-PET/D3CG_db4/model_epoch_4000.pt")
    parser.add_argument("--output_dir", type=str, default="./results/MRI-PET/D3CG_db4/4000_3")
    
    # Nonlinear parameters
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--negative_slope", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=1.0)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Determine channels based on dataset type
    if args.dataset_type in ["mripet", "mrispect"]:
        in_channels = 6  # RGB (target) + RGB (condition)
        out_channels = 3
    else:
        in_channels = 2
        out_channels = 1

    # Initialize model based on model type
    if args.model_name in ["DDPM", "DIFFDDPM"]:
        net_model = BaseUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                           num_res_blocks=2, dropout=0.3,
                           in_channels=in_channels, out_channels=out_channels).to(device)
    elif args.model_name == "midattnDDPM":
        net_model = MidAttnUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                              num_res_blocks=2, dropout=0.3,
                              in_channels=in_channels, out_channels=out_channels).to(device)
    elif args.model_name == "UpAttnDDPM":
        net_model = UpAttnUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                             num_res_blocks=2, dropout=0.3,
                             in_channels=in_channels, out_channels=out_channels).to(device)
    else:  # All wavelet-based models
        net_model = WTUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                         num_res_blocks=2, dropout=0.3,
                         in_channels=in_channels, out_channels=out_channels).to(device)
    
    # Load model weights
    raw_state = torch.load(args.model_weight_path, map_location=device, weights_only=True)
    clean_state = {k: v for k, v in raw_state.items() if ('total_ops' not in k and 'total_params' not in k)}
    missing, unexpected = net_model.load_state_dict(clean_state, strict=False)

    real_missing = [k for k in missing if 'total_ops' not in k and 'total_params' not in k]
    real_unexp   = [k for k in unexpected if 'total_ops' not in k and 'total_params' not in k]
    assert len(real_missing) == 0 and len(real_unexp) == 0, \
           f"Missing: {real_missing}, Unexpected: {real_unexp}"
    net_model.eval()

    # Model profiling
    print("Starting model profiling...")
    input_shape = get_model_input_shape(args.model_name, args.dataset_type, args.image_size, args.batch_size)
    profiling_results = profile_model(net_model, input_shape, device, args.model_name)
    
    # 将性能分析结果写入日志文件
    log_file_path = os.path.join(args.output_dir, f"profiling_{args.model_name}.log")
    log_profiling_results(profiling_results, args.model_name, log_file_path)
    
    print(f"Model profiling completed and saved to: {log_file_path}")
    print(f"Key metrics - Params: {profiling_results['total_params_M']:.2f}M, "
          f"FLOPs: {profiling_results['flops_G']:.3f}G, "
          f"Memory: {profiling_results['model_memory_MB']:.1f}MB")

    # Get appropriate sampler using the factory
    _, sampler = get_trainer_sampler(
        model_name=args.model_name,
        net_model=net_model,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        device=device,
        # D3CG自定义参数
        wave_type=args.wave_type,
        nonlinear_type=args.nonlinear_type,
        transform_levels=args.transform_levels,
        # 非线性函数参数
        alpha=args.alpha,
        beta=args.beta,
        negative_slope=args.negative_slope,
        scale=args.scale
    )

    # Load dataset
    dataset_classes = {
        "foot": FootDataset2,
        "ctmri": CTMRI,
        "mripet": MRIPET,
        "mrispect": MRISPECT
    }
    
    dataset = dataset_classes[args.dataset_type](args.dataset_name, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ssim_values = []
    mse_values = []
    psnr_values = []
    lpips_values = []
    mae_values = []

    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    sample_count = 0

    with tempfile.TemporaryDirectory() as fid_temp_dir:
        real_images_dir = os.path.join(fid_temp_dir, 'real')
        generated_images_dir = os.path.join(fid_temp_dir, 'generated')
        os.makedirs(real_images_dir)
        os.makedirs(generated_images_dir)

        with torch.no_grad():
            start_time = time.time()
            for batch in dataloader:
                target = batch['target'].to(device)
                condition = batch['condition'].to(device)

                # Different sampling strategies for different models
                if args.model_name.startswith("D3CG") or args.model_name == "WTDDPM":
                    generated_images = sampler(condition)
                else:
                    random_noise = torch.randn_like(target)
                    x_T = torch.cat((random_noise, condition), dim=1)
                    generated_images = sampler(x_T)

                if out_channels == 1:
                    generated_image = generated_images[:, 0:1, :, :]
                    target_image = target
                    condition_image = condition
                else:
                    generated_image = generated_images[:, :out_channels, :, :]
                    target_image = target
                    condition_image = condition

                gen_img_np = generated_image[0].detach().cpu().numpy()
                tgt_img_np = target_image[0].detach().cpu().numpy()
                cond_img_np = condition_image[0].detach().cpu().numpy()

                if out_channels > 1:
                    gen_img_np = np.transpose(gen_img_np, (1, 2, 0))
                    tgt_img_np = np.transpose(tgt_img_np, (1, 2, 0))
                    cond_img_np = np.transpose(cond_img_np, (1, 2, 0))
                else:
                    gen_img_np = gen_img_np[0]
                    tgt_img_np = tgt_img_np[0]
                    cond_img_np = cond_img_np[0]

                gen_img_norm = np.clip((gen_img_np + 1) / 2.0, 0, 1)
                tgt_img_norm = np.clip((tgt_img_np + 1) / 2.0, 0, 1)
                cond_img_norm = np.clip((cond_img_np + 1) / 2.0, 0, 1)

                print(f"Sample {sample_count + 1} - Model: {args.model_name}")
                print(f"Generated image min: {gen_img_norm.min()}, max: {gen_img_norm.max()}")
                print(f"Target image min: {tgt_img_norm.min()}, max: {tgt_img_norm.max()}")

                ssim_val = ssim(tgt_img_norm, gen_img_norm, data_range=1, channel_axis=-1 if out_channels > 1 else None)
                ssim_values.append(ssim_val)
                mse_val = mean_squared_error(tgt_img_norm.flatten(), gen_img_norm.flatten())
                mse_values.append(mse_val)
                psnr_val = psnr(tgt_img_norm, gen_img_norm, data_range=1)
                psnr_values.append(psnr_val)
                mae_val = np.mean(np.abs(tgt_img_norm - gen_img_norm))
                mae_values.append(mae_val)

                if out_channels == 1:
                    gen_lpips = generated_image.repeat(1, 3, 1, 1)
                    tgt_lpips = target_image.repeat(1, 3, 1, 1)
                else:
                    gen_lpips = generated_image
                    tgt_lpips = target_image
                lpips_val = lpips_metric(gen_lpips, tgt_lpips).item()
                lpips_values.append(lpips_val)

                # Save images
                if out_channels == 1:
                    mode = 'L'
                    gen_img_to_save = (gen_img_norm * 255).astype(np.uint8)
                    tgt_img_to_save = (tgt_img_norm * 255).astype(np.uint8)
                    cond_img_to_save = (cond_img_norm * 255).astype(np.uint8)
                else:
                    mode = 'RGB'
                    gen_img_to_save = (gen_img_norm * 255).astype(np.uint8)
                    tgt_img_to_save = (tgt_img_norm * 255).astype(np.uint8)
                    cond_img_to_save = (cond_img_norm * 255).astype(np.uint8)

                Image.fromarray(gen_img_to_save, mode=mode).save(
                    os.path.join(generated_images_dir, f'generated_{sample_count + 1}.png'))
                Image.fromarray(tgt_img_to_save, mode=mode).save(
                    os.path.join(real_images_dir, f'real_{sample_count + 1}.png'))

                sample_output_dir = os.path.join(args.output_dir, f'sample_{sample_count + 1}')
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)

                Image.fromarray(cond_img_to_save, mode=mode).save(os.path.join(sample_output_dir, 'condition.png'))
                Image.fromarray(tgt_img_to_save, mode=mode).save(os.path.join(sample_output_dir, 'target.png'))
                Image.fromarray(gen_img_to_save, mode=mode).save(os.path.join(sample_output_dir, 'generated.png'))

                sample_count += 1
                if sample_count >= args.sample_num:
                    break

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time taken for image generation: {total_time:.2f} seconds")

            fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir],
                                                            args.batch_size, device, dims=2048)

    average_ssim = np.mean(ssim_values)
    average_mse = np.mean(mse_values)
    average_psnr = np.mean(psnr_values)
    average_lpips = np.mean(lpips_values)
    average_mae = np.mean(mae_values)

    print(f"\n=== Evaluation Results for {args.model_name} ===")
    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Average MSE: {average_mse:.4f}")
    print(f"Average PSNR: {average_psnr:.4f}")
    print(f"Average LPIPS: {average_lpips:.4f}")
    print(f"Average MAE: {average_mae:.4f}")
    print(f"FID: {fid_value:.4f}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, f'results_{args.model_name}.txt')
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {args.model_name}\n")
        f.write(f"Average SSIM: {average_ssim:.4f}\n")
        f.write(f"Average MSE: {average_mse:.4f}\n")
        f.write(f"Average PSNR: {average_psnr:.4f}\n")
        f.write(f"Average LPIPS: {average_lpips:.4f}\n")
        f.write(f"Average MAE: {average_mae:.4f}\n")
        f.write(f"FID: {fid_value:.4f}\n")