import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from models.base import BaseUNet
from models.attention import MidAttnUNet, UpAttnUNet
from models.wavelet import WTUNet
from diffusion.gaussian import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond
from diffusion.attn.upattnDDPM import UpAttnDDPMTrainer_cond, UpAttnDDPMSampler_cond
from diffusion.attn.midattnDDPM import MidAttnDDPMTrainer_cond, MidAttnDDPMSampler_cond
from diffusion.ablation.DIFFDDPM import DiffDDPMTrainer_cond, DiffDDPMSampler_cond
from diffusion.ablation.wavelet import WTDDPMTrainer_cond, WTDDPMSampler_cond
from diffusion.D3CG import D3CGTrainer_cond, D3CGSampler_cond
from data.dataset import FootDataset2, MRIPET, MRISPECT, CTMRI
from torch.utils.data import DataLoader
from PIL import Image
import lpips
from pytorch_fid import fid_score
import tempfile
import warnings
import time

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    image_size = 256
    batch_size = 1
    T = 1000
    beta_1 = 1e-4
    beta_T = 0.02
    sample_num = 20

    dataset_name = "."
    save_weight_dir = "."
    model_weight_path = os.path.join(save_weight_dir, 'model_epoch_xxxx.pt')

    output_dir = '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_type = "ctmri"  #
    if dataset_type in ["mripet", "mrispect"]:
        in_channels = 6  # RGB (target) + RGB (condition)
        out_channels = 3
    else:
        in_channels = 2
        out_channels = 1

    net_model = WTUNet(T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                       num_res_blocks=2, dropout=0.3,
                       in_channels=in_channels, out_channels=out_channels).to(device)
    net_model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    net_model.eval()

    sampler = D3CGSampler_cond(model=net_model, beta_1=beta_1, beta_T=beta_T, T=T).to(device)

    dataset = CTMRI(dataset_name, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  

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
            for batch in dataloader:
                target = batch['target'].to(device)
                condition = batch['condition'].to(device)

                random_noise = torch.randn_like(target)

                x_T = torch.cat((random_noise, condition), dim=1)

                generated_images = sampler(condition)  #if sampler=D3CGsample or WTsample :sampler(condition)  else :sampler(x_T)

                if out_channels == 1:
                    generated_image = generated_images[:, 0:1, :, :]  # [B, 1, H, W]
                    target_image = target  # [B, 1, H, W]
                    condition_image = condition
                else:
                    generated_image = generated_images[:, :out_channels, :, :]  # [B, 3, H, W]
                    target_image = target
                    condition_image = condition

                gen_img_np = generated_image[0].detach().cpu().numpy()
                tgt_img_np = target_image[0].detach().cpu().numpy()
                cond_img_np = condition_image[0].detach().cpu().numpy()

                if out_channels > 1:
                    gen_img_np = np.transpose(gen_img_np, (1, 2, 0))  # [H, W, C]
                    tgt_img_np = np.transpose(tgt_img_np, (1, 2, 0))
                    cond_img_np = np.transpose(cond_img_np, (1, 2, 0))
                else:
                    gen_img_np = gen_img_np[0]  # [H, W]
                    tgt_img_np = tgt_img_np[0]
                    cond_img_np = cond_img_np[0]

                gen_img_norm = np.clip((gen_img_np + 1) / 2.0, 0, 1)
                tgt_img_norm = np.clip((tgt_img_np + 1) / 2.0, 0, 1)
                cond_img_norm = np.clip((cond_img_np + 1) / 2.0, 0, 1)

                print(f"Sample {sample_count + 1}")
                print(f"Generated image min: {gen_img_norm.min()}, max: {gen_img_norm.max()}")
                print(f"Target image min: {tgt_img_norm.min()}, max: {tgt_img_norm.max()}")
                print(f"Condition image min: {cond_img_norm.min()}, max: {cond_img_norm.max()}")

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

                if out_channels == 1:
                    mode = 'L'
                    gen_img_to_save = (gen_img_norm * 255).astype(np.uint8)
                    tgt_img_to_save = (tgt_img_norm * 255).astype(np.uint8)
                else:
                    mode = 'RGB'
                    gen_img_to_save = (gen_img_norm * 255).astype(np.uint8)
                    tgt_img_to_save = (tgt_img_norm * 255).astype(np.uint8)
                Image.fromarray(gen_img_to_save, mode=mode).save(
                    os.path.join(generated_images_dir, f'generated_{sample_count + 1}.png'))
                Image.fromarray(tgt_img_to_save, mode=mode).save(
                    os.path.join(real_images_dir, f'real_{sample_count + 1}.png'))

                sample_output_dir = os.path.join(output_dir, f'sample_{sample_count + 1}')
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)
                if out_channels == 1:
                    cond_mode = 'L'
                    cond_img_to_save = (cond_img_norm * 255).astype(np.uint8)
                else:
                    cond_mode = 'RGB'
                    cond_img_to_save = (cond_img_norm * 255).astype(np.uint8)
                Image.fromarray(cond_img_to_save, mode=cond_mode).save(os.path.join(sample_output_dir, 'condition.png'))
                Image.fromarray(tgt_img_to_save, mode=mode).save(os.path.join(sample_output_dir, 'target.png'))
                Image.fromarray(gen_img_to_save, mode=mode).save(os.path.join(sample_output_dir, 'generated.png'))

                sample_count += 1
                if sample_count >= sample_num:
                    break

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time taken for image generation: {total_time:.2f} seconds")

            fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir],
                                                            batch_size, device, dims=2048)

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
