import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from models.base import BaseUNet
from diffusion.gaussian import GaussianDiffusionSampler_cond
from data.dataset import FootDataset2
from torch.utils.data import DataLoader
from PIL import Image
import lpips
from pytorch_fid import fid_score
import tempfile
import warnings
import time


warnings.filterwarnings("ignore")

if __name__ == '__main__':

    image_size = 512
    batch_size = 1  
    T = 1000
    beta_1 = 1e-4
    beta_T = 0.02
    sample_num = 16 

    dataset_name = "./test"
    save_weight_dir = "./results/check/MedD3CG"
    model_weight_path = os.path.join(save_weight_dir, 'model_epoch_98.pt')

    output_dir = './results/check/MedD3CG/diffusion98_ch64'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net_model = BaseUNet(T, ch=64, ch_mult=[1, 2, 3, 4], attn=[2], num_res_blocks=2, dropout=0.3).to(device)
    net_model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    net_model.eval()

    sampler = GaussianDiffusionSampler_cond(model=net_model, beta_1=beta_1, beta_T=beta_T, T=T).to(device)

    dataset = FootDataset2(dataset_name, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 随机抽取样本

    ssim_values = []
    mse_values = []
    psnr_values = []
    lpips_values = []
    mae_values = []

    lpips_metric = lpips.LPIPS(net='vgg').to(device)  # LPIPS metric

    sample_count = 0 


    with tempfile.TemporaryDirectory() as fid_temp_dir:
        real_images_dir = os.path.join(fid_temp_dir, 'real')
        generated_images_dir = os.path.join(fid_temp_dir, 'generated')
        os.makedirs(real_images_dir)
        os.makedirs(generated_images_dir)


        with torch.no_grad():

            start_time = time.time()

            # SSIM、MSE、PSNR、LPIPS、MAE
            for batch in dataloader:
                random_noise = torch.randn_like(batch['target'].to(device))

                x_T = torch.cat((random_noise, batch['condition'].to(device)), dim=1)

                # x_T = batch['condition'].to(device)

                generated_images = sampler(x_T)

                generated_image = generated_images[:, 0, :, :]

                generated_image_np = generated_image[0].detach().cpu().numpy()
                target_image_np = batch['target'][0, 0].cpu().numpy()
                condition_image_np = batch['condition'][0, 0].cpu().numpy()

                print(f"Sample {sample_count + 1}")
                print(f"Generated image min: {generated_image_np.min()}, max: {generated_image_np.max()}")
                print(f"Target image min: {target_image_np.min()}, max: {target_image_np.max()}")
                print(f"Condition image min: {condition_image_np.min()}, max: {condition_image_np.max()}")

                generated_image_norm = (generated_image_np + 1) / 2
                target_image_norm = (target_image_np + 1) / 2
                condition_image_norm = (condition_image_np + 1) / 2

                generated_image_norm = np.clip(generated_image_norm, 0, 1)
                target_image_norm = np.clip(target_image_norm, 0, 1)
                condition_image_norm = np.clip(condition_image_norm, 0, 1)

                print(f"After mapping to [0,1]:")
                print(f"Generated image min: {generated_image_norm.min()}, max: {generated_image_norm.max()}")
                print(f"Target image min: {target_image_norm.min()}, max: {target_image_norm.max()}")
                print(f"Condition image min: {condition_image_norm.min()}, max: {condition_image_norm.max()}")

                # SSIM
                ssim_value = ssim(generated_image_norm, target_image_norm, data_range=1)
                ssim_values.append(ssim_value)

                # MSE
                mse_value = mean_squared_error(target_image_norm, generated_image_norm)
                mse_values.append(mse_value)

                # PSNR
                psnr_value = psnr(target_image_norm, generated_image_norm, data_range=1)
                psnr_values.append(psnr_value)

                # MAE
                mae_value = np.mean(np.abs(target_image_norm - generated_image_norm))
                mae_values.append(mae_value)

                # LPIPS
                generated_image_tensor = generated_image.unsqueeze(0).to(device)
                target_image_tensor = batch['target'].to(device)
                lpips_value = lpips_metric(generated_image_tensor, target_image_tensor).item()
                lpips_values.append(lpips_value)

                generated_image_uint8 = (generated_image_norm * 255).astype(np.uint8)
                target_image_uint8 = (target_image_norm * 255).astype(np.uint8)
                Image.fromarray(generated_image_uint8).save(os.path.join(generated_images_dir, f'generated_{sample_count + 1}.png'))
                Image.fromarray(target_image_uint8).save(os.path.join(real_images_dir, f'real_{sample_count + 1}.png'))


                sample_output_dir = os.path.join(output_dir, f'sample_{sample_count + 1}')
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)

                condition_image_uint8 = (condition_image_norm * 255).astype(np.uint8)
                condition_image_pil = Image.fromarray(condition_image_uint8, mode='L')
                condition_image_path = os.path.join(sample_output_dir, 'condition.png')
                condition_image_pil.save(condition_image_path)

                target_image_pil = Image.fromarray(target_image_uint8, mode='L')
                target_image_path = os.path.join(sample_output_dir, 'target.png')
                target_image_pil.save(target_image_path)

                generated_image_pil = Image.fromarray(generated_image_uint8, mode='L')
                generated_image_path = os.path.join(sample_output_dir, 'generated.png')
                generated_image_pil.save(generated_image_path)

                sample_count += 1

                if sample_count >= sample_num:
                    break

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time taken for image generation: {total_time:.2f} seconds")

            # FID
            fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir], batch_size, device, dims=2048)

    average_ssim = np.mean(ssim_values)
    average_mse = np.mean(mse_values)
    average_psnr = np.mean(psnr_values)
    average_lpips = np.mean(lpips_values)
    average_mae = np.mean(mae_values)

    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Average MSE: {average_mse:.4f}")
    print(f"Average PSNR: {average_psnr:.4f}")
    print(f"Average LPIPS: {average_lpips:.4f}")
    print(f"Average MAE: {average_mae:.4f}")
    print(f"FID: {fid_value:.4f}")
