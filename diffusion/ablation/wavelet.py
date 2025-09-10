import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from ..base import extract

def wavelet_transform(x):
    B, C, H, W = x.shape
    coeffs_list = []
    for i in range(B):
        coeffs_channels = []
        for c in range(C):
            img = x[i, c, :, :].cpu().numpy()
            cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
            coeffs_channels.append(torch.from_numpy(cA))
            coeffs_channels.append(torch.from_numpy(cH))
            coeffs_channels.append(torch.from_numpy(cV))
            coeffs_channels.append(torch.from_numpy(cD))

        coeffs_list.append(torch.stack(coeffs_channels, dim=0))
    return coeffs_list

def coeffs_to_tensor(coeffs, device):
    tensor = torch.stack(coeffs, dim=0).to(device)
    return tensor

def tensor_to_coeffs(tensor):
    B, ch, H, W = tensor.shape
    C = ch // 4
    coeffs = []
    for i in range(B):
        coeffs_per_img = []
        for c in range(C):
            cA = tensor[i, c*4+0].cpu().numpy()
            cH = tensor[i, c*4+1].cpu().numpy()
            cV = tensor[i, c*4+2].cpu().numpy()
            cD = tensor[i, c*4+3].cpu().numpy()
            coeffs_per_img.append((cA, (cH, cV, cD)))

        coeffs.append(coeffs_per_img[0] if C == 1 else coeffs_per_img)
    return coeffs

def inverse_wavelet_transform(coeffs, original_size, device):
    if isinstance(coeffs[0], tuple):
        recs = []
        for cf in coeffs:
            rec = pywt.idwt2(cf, 'haar')
            rec = torch.tensor(rec, device=device).unsqueeze(0)
            rec = F.interpolate(rec.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
            recs.append(rec)
        return torch.stack(recs, dim=0)
    else:
        recs = []
        for coeffs_per_img in coeffs:
            channels = []
            for cf in coeffs_per_img:
                rec = pywt.idwt2(cf, 'haar')
                rec = torch.tensor(rec, device=device).unsqueeze(0)
                rec = F.interpolate(rec.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
                channels.append(rec)
            recs.append(torch.cat(channels, dim=0))
        return torch.stack(recs, dim=0)

class WTDDPMTrainer_cond(nn.Module):
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
        t = torch.randint(self.T, (x_0.shape[0],), device=x_0.device)

        out_channels = self.model.out_channels if hasattr(self.model, "out_channels") else 1
        ct = x_0[:, :out_channels, :, :]
        cbct = x_0[:, out_channels:2*out_channels, :, :]

        ct_coeffs = wavelet_transform(ct)
        cbct_coeffs = wavelet_transform(cbct)
        ct_coeffs_tensor = coeffs_to_tensor(ct_coeffs, ct.device)
        cbct_coeffs_tensor = coeffs_to_tensor(cbct_coeffs, cbct.device)

        noise = torch.randn_like(ct_coeffs_tensor)
        sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, ct_coeffs_tensor.shape)
        sqrt_one_minus_alphas_bar_t = extract(self.sqrt_one_minus_alphas_bar, t, ct_coeffs_tensor.shape)
        d_t = sqrt_alphas_bar_t * ct_coeffs_tensor + sqrt_one_minus_alphas_bar_t * noise

        model_input = torch.cat((d_t, cbct_coeffs_tensor), dim=1)
        eps_theta = self.model(model_input, t)
        loss = F.mse_loss(eps_theta, noise, reduction='sum')
        return loss

class WTDDPMSampler_cond(nn.Module):
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
        model_input = torch.cat((d_t, cbct_coeffs_tensor), dim=1)
        eps_theta = self.model(model_input, t)

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
            device = cbct.device
            out_channels = self.model.out_channels if hasattr(self.model, "out_channels") else 1
            cbct = cbct[:, :out_channels, :, :]
            original_size = cbct.shape[2:]

            cbct_coeffs = wavelet_transform(cbct)
            cbct_coeffs_tensor = coeffs_to_tensor(cbct_coeffs, device)

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

            ct_coeffs_tensor = d_t + cbct_coeffs_tensor
            ct_coeffs = tensor_to_coeffs(ct_coeffs_tensor)
            ct_reconstructed = inverse_wavelet_transform(ct_coeffs, original_size, device)

            x_p = torch.cat((ct_reconstructed, cbct), dim=1)
            return torch.clamp(x_p, -1, 1)