import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def wavelet_transform(x):
    B, C, H, W = x.shape
    coeffs_list = []
    for i in range(B):
        coeffs_channels = []
        for c in range(C):
            img = x[i, c, :, :].cpu().numpy()
            coeffs2 = pywt.dwt2(img, 'haar')
            cA, (cH, cV, cD) = coeffs2

            coeffs_channels.append(torch.from_numpy(cA))
            coeffs_channels.append(torch.from_numpy(cH))
            coeffs_channels.append(torch.from_numpy(cV))
            coeffs_channels.append(torch.from_numpy(cD))

        coeffs_channels = torch.stack(coeffs_channels, dim=0)
        coeffs_list.append(coeffs_channels)
    return coeffs_list

def coeffs_to_tensor(coeffs):

    return torch.stack(coeffs, dim=0)

def tensor_to_coeffs(tensor):

    B, ch, H, W = tensor.shape
    C = ch // 4
    coeffs_list = []
    for i in range(B):
        coeffs_per_img = []
        for c in range(C):
            cA = tensor[i, c*4+0, :, :].cpu().numpy()
            cH = tensor[i, c*4+1, :, :].cpu().numpy()
            cV = tensor[i, c*4+2, :, :].cpu().numpy()
            cD = tensor[i, c*4+3, :, :].cpu().numpy()
            coeffs_per_img.append((cA, (cH, cV, cD)))

        coeffs_list.append(coeffs_per_img if C > 1 else coeffs_per_img[0])
    return coeffs_list

def inverse_wavelet_transform(coeffs, original_size):

    if isinstance(coeffs[0], tuple):

        reconstructed = []
        for coeff in coeffs:
            rec = pywt.idwt2(coeff, 'haar')
            rec = torch.tensor(rec).unsqueeze(0)  # [1, H, W]
            rec = F.interpolate(rec.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
            reconstructed.append(rec)
        reconstructed = torch.stack(reconstructed, dim=0)
        return reconstructed  # [B, 1, H, W]
    else:

        reconstructed = []
        for coeffs_per_img in coeffs:
            rec_channels = []
            for coeff in coeffs_per_img:
                rec = pywt.idwt2(coeff, 'haar')
                rec = torch.tensor(rec)  # [H, W]
                rec = F.interpolate(rec.unsqueeze(0).unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                rec_channels.append(rec)

            rec_channels = torch.stack(rec_channels, dim=0)
            reconstructed.append(rec_channels)
        return torch.stack(reconstructed, dim=0)  # [B, C, H, W]

def wavelet_post_denoise(tensor_image, wavelet='db4', mode='reflect', level=1, threshold=0.05):

    B, C, H, W = tensor_image.shape
    device = tensor_image.device
    out_batch = []

    def soft_threshold(arr, thr):
        return np.sign(arr) * np.maximum(np.abs(arr) - thr, 0.)

    for b in range(B):
        channels_denoised = []
        for c in range(C):

            img_np = tensor_image[b, c, :, :].cpu().numpy()

            cA = coeffs[0]
            detail_levels = coeffs[1:]

            for (cH, cV, cD) in detail_levels:
                cH_d = soft_threshold(cH, threshold)
                cV_d = soft_threshold(cV, threshold)
                cD_d = soft_threshold(cD, threshold)
                new_detail_levels.append((cH_d, cV_d, cD_d))

            new_coeffs = [cA] + new_detail_levels

            rec_img = pywt.waverec2(new_coeffs, wavelet=wavelet, mode=mode)
            rec_img_t = torch.from_numpy(rec_img).float().unsqueeze(0)  # [1, H, W]
            channels_denoised.append(rec_img_t)

        channels_denoised = torch.cat(channels_denoised, dim=0)  # [C, H, W]
        out_batch.append(channels_denoised.unsqueeze(0))         # [1, C, H, W]

    out = torch.cat(out_batch, dim=0).to(device)
    return out


class D3CGTrainer_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        out_channels = self.model.out_channels if hasattr(self.model, "out_channels") else 1
        ct = x_0[:, :out_channels, :, :]
        cbct = x_0[:, out_channels:, :, :]
        original_size = ct.shape[2:]

        ct_coeffs = wavelet_transform(ct)
        cbct_coeffs = wavelet_transform(cbct)

        ct_coeffs_tensor = coeffs_to_tensor(ct_coeffs).to(ct.device)
        cbct_coeffs_tensor = coeffs_to_tensor(cbct_coeffs).to(cbct.device)

        d_0 = ct_coeffs_tensor - cbct_coeffs_tensor  # [B, 4*out_channels, H', W']
        noise = torch.randn_like(d_0)
        sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, d_0.shape)
        sqrt_one_minus_alphas_bar_t = extract(self.sqrt_one_minus_alphas_bar, t, d_0.shape)
        d_t = sqrt_alphas_bar_t * d_0 + sqrt_one_minus_alphas_bar_t * noise  # [B, 4*out_channels, H', W']

        model_input = torch.cat((d_t, cbct_coeffs_tensor), dim=1)  # [B, 8*out_channels, H', W']
        eps_theta = self.model(model_input, t)  # [B, 4*out_channels, H', W']
        loss = F.mse_loss(eps_theta, noise, reduction='sum')
        return loss

class D3CGSampler_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def p_mean_variance(self, d_t, cbct_coeffs_tensor, t):
        model_input = torch.cat((d_t, cbct_coeffs_tensor), dim=1)  # [B, 8*out_channels, H', W']
        eps_theta = self.model(model_input, t)  # [B, 4*out_channels, H', W']
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, d_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, d_t.shape)
        d_0 = (d_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / sqrt_alphas_cumprod_t

        alphas_t = extract(self.alphas, t, d_t.shape)
        betas_t = extract(self.betas, t, d_t.shape)
        model_mean = (1 / torch.sqrt(alphas_t)) * (d_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * eps_theta)
        model_var = extract(self.posterior_variance, t, d_t.shape)
        return model_mean, model_var

    def forward(self, cbct):
        with torch.no_grad():

            out_channels = self.model.out_channels if hasattr(self.model, "out_channels") else 1
            cbct = cbct[:, :out_channels, :, :]
            original_size = cbct.shape[2:]
            device = cbct.device

            cbct_coeffs = wavelet_transform(cbct)
            cbct_coeffs_tensor = coeffs_to_tensor(cbct_coeffs).to(device)  # [B, 4*out_channels, H', W']

            batch_size = cbct.shape[0]
            d_t = torch.randn_like(cbct_coeffs_tensor)

            for time_step in reversed(range(self.T)):
                t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
                model_mean, model_var = self.p_mean_variance(d_t, cbct_coeffs_tensor, t)
                if time_step > 0:
                    noise = torch.randn_like(d_t)
                    d_t = model_mean + torch.sqrt(model_var) * noise
                else:
                    d_t = model_mean

            ct_coeffs_tensor = d_t + cbct_coeffs_tensor  # [B, 4*out_channels, H', W']

            ct_coeffs = tensor_to_coeffs(ct_coeffs_tensor)
            ct_reconstructed = inverse_wavelet_transform(ct_coeffs, original_size)
            ct_reconstructed = ct_reconstructed.to(device)

            x_p = torch.cat((ct_reconstructed, cbct), dim=1)
            x_p = torch.clamp(x_p, -1., 1.)
            
            return x_p