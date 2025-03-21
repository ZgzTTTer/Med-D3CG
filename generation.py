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
    T = 1000
    beta_1 = 1e-4
    beta_T = 0.02

    condition_file = ".png"

    output_dir = "./results/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        

    save_weight_dir = "./"
    model_weight_path = os.path.join(save_weight_dir, 'best_model_epoch_xxxx.pt')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_model = WTUNet(T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2], num_res_blocks=2, dropout=0.3).to(device)
    net_model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    net_model.eval()

    sampler = D3CGSampler_cond(model=net_model, beta_1=beta_1, beta_T=beta_T, T=T).to(device)

    condition_image = Image.open(condition_file).convert('L').resize((image_size, image_size))

    condition_np = np.array(condition_image).astype(np.float32) / 255.0
    condition_np = condition_np * 2 - 1

    condition_tensor = torch.from_numpy(condition_np).unsqueeze(0).unsqueeze(0).to(device)

    random_noise = torch.randn_like(condition_tensor)

    x_T = torch.cat((random_noise, condition_tensor), dim=1)

    with torch.no_grad():
        generated_images = sampler(condition_tensor)

    generated_image = generated_images[:, 0, :, :]
    generated_image_np = generated_image[0].detach().cpu().numpy()

    generated_image_norm = (generated_image_np + 1) / 2
    generated_image_norm = np.clip(generated_image_norm, 0, 1)

    generated_image_uint8 = (generated_image_norm * 255).astype(np.uint8)

    output_path = os.path.join(output_dir, f"generated_{os.path.splitext(os.path.basename(condition_file))[0]}.png")
    Image.fromarray(generated_image_uint8, mode='L').save(output_path)
    print(f"Saved generated image: {output_path}")
