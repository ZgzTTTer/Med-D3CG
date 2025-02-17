import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def wavelet_transform(x):
    # x: [B, C, H, W], C=1
    B, C, H, W = x.shape
    coeffs = []
    for i in range(B):
        img = x[i, 0, :, :].cpu().numpy()
        coeffs2 = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs2
        coeffs.append((cA, cH, cV, cD))
    return coeffs  

def coeffs_to_tensor(coeffs):
    # 
    # return [B, 4, H', W']
    tensors = []
    for coeff in coeffs:
        cA, cH, cV, cD = coeff
        cA = torch.from_numpy(cA)
        cH = torch.from_numpy(cH)
        cV = torch.from_numpy(cV)
        cD = torch.from_numpy(cD)
        tensor = torch.stack([cA, cH, cV, cD], dim=0)  # [4, H', W']
        tensors.append(tensor)
    tensors = torch.stack(tensors, dim=0)  # [B, 4, H', W']
    return tensors

def tensor_to_coeffs(tensor):
    # tensor: [B, 4, H', W']
    B = tensor.shape[0]
    coeffs = []
    for i in range(B):
        cA = tensor[i, 0, :, :].cpu().numpy()
        cH = tensor[i, 1, :, :].cpu().numpy()
        cV = tensor[i, 2, :, :].cpu().numpy()
        cD = tensor[i, 3, :, :].cpu().numpy()
        coeffs.append((cA, (cH, cV, cD)))
    return coeffs

def inverse_wavelet_transform(coeffs, original_size):
    reconstructed = []
    for coeff in coeffs:
        rec = pywt.idwt2(coeff, 'haar')
        rec = torch.tensor(rec).unsqueeze(0)  # [1, H, W]
        rec = F.interpolate(rec.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
        reconstructed.append(rec)  # [1, H, W]
    reconstructed = torch.stack(reconstructed, dim=0)  # [B, H, W]
    return reconstructed.unsqueeze(1)  # [B, 1, H, W]

class D3CGTrainer_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        ct = x_0[:,0:1,:,:]   # CT [B, 1, H, W]
        cbct = x_0[:,1:2,:,:] # CBCT [B, 1, H, W]

        original_size = ct.shape[2:]

        ct_coeffs = wavelet_transform(ct)
        cbct_coeffs = wavelet_transform(cbct)

        #  [B, 4, H', W']
        ct_coeffs_tensor = coeffs_to_tensor(ct_coeffs).to(ct.device)
        cbct_coeffs_tensor = coeffs_to_tensor(cbct_coeffs).to(cbct.device)

        d_0 = ct_coeffs_tensor - cbct_coeffs_tensor  #  [B, 4, H', W']

        noise = torch.randn_like(d_0)
        sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, d_0.shape)
        sqrt_one_minus_alphas_bar_t = extract(self.sqrt_one_minus_alphas_bar, t, d_0.shape)
        d_t = sqrt_alphas_bar_t * d_0 + sqrt_one_minus_alphas_bar_t * noise  #  [B, 4, H', W']

        # [B, 8, H', W']
        model_input = torch.cat((d_t, cbct_coeffs_tensor), dim=1)
        eps_theta = self.model(model_input, t)  #  [B, 4, H', W']

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
        model_input = torch.cat((d_t, cbct_coeffs_tensor), dim=1)  # [B, 8, H', W']
        eps_theta = self.model(model_input, t)  # [B, 4, H', W']

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
            cbct = cbct[:,0:1,:,:]   # CBCT [B, 1, H, W]

            original_size = cbct.shape[2:]

            cbct_coeffs = wavelet_transform(cbct)
            cbct_coeffs_tensor = coeffs_to_tensor(cbct_coeffs).to(cbct.device)  # [B, 4, H', W']

            batch_size = cbct.shape[0]
            device = cbct.device

            d_t = torch.randn_like(cbct_coeffs_tensor)

            for time_step in reversed(range(self.T)):
                t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)

                model_mean, model_var = self.p_mean_variance(d_t, cbct_coeffs_tensor, t)

                if time_step > 0:
                    noise = torch.randn_like(d_t)
                    d_t = model_mean + torch.sqrt(model_var) * noise
                else:
                    d_t = model_mean

            d_0 = d_t  # [B, 4, H', W']

            ct_coeffs_tensor = d_0 + cbct_coeffs_tensor  # [B, 4, H', W']

            ct_coeffs = tensor_to_coeffs(ct_coeffs_tensor)

            #  [B, 1, H, W]
            ct_reconstructed = inverse_wavelet_transform(ct_coeffs, original_size)
            ct_reconstructed = ct_reconstructed.squeeze(2)  # 
 
            ct_reconstructed = ct_reconstructed.to(cbct.device)
            x_p = torch.cat((ct_reconstructed, cbct), 1)
            x_p = torch.clamp(x_p, -1., 1.)

            return x_p  # [B, 2, H, W]
