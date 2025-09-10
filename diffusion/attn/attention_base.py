import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from ..base import BaseDiffusionTrainer_cond, BaseDiffusionSampler_cond
from ..base import extract


class AttnDDPMTrainer_cond(BaseDiffusionTrainer_cond):
    def get_initial_signal(self, ct, cbct):
        return ct


class AttnDDPMSampler_cond(BaseDiffusionSampler_cond):
    def __init__(self, model, beta_1, beta_T, T, psi=0.5, s=0.1):
        super().__init__(model, beta_1, beta_T, T)
        self.psi = psi  
        self.s = s  

    def gaussian_blur(self, x):
        kernel_size = 5
        sigma = 1.0
        return TF.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)

    def process_attention(self, attention_map, target_size):
        B, N, _ = attention_map.shape
        H = W = int(N ** 0.5)
        attention_map = attention_map.sum(dim=1).view(B, 1, H, W)
        M_t = (attention_map > self.psi).float()
        return F.interpolate(M_t, size=target_size, mode='nearest')

    def forward(self, x_T):
        with torch.no_grad():
            x_t = x_T
            out_channels = self.model.out_channels
            ct = x_t[:, :out_channels, :, :]
            cbct = x_t[:, out_channels:, :, :]

            for time_step in reversed(range(self.T)):
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step

                model_output, attention_map = self.model(x_t, t)
                eps = model_output

                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var, t, ct.shape)

                M_t = self.process_attention(attention_map, ct.shape[2:])

                sqrt_alpha_bar_t = extract(self.sqrt_alphas_bar, t, ct.shape)
                sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_bar, t, ct.shape)
                x_hat_0 = (ct - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

                x_tilde_0 = self.gaussian_blur(x_hat_0)
                x_tilde_t = sqrt_alpha_bar_t * x_tilde_0 + sqrt_one_minus_alpha_bar_t * eps
                x_bar_t = (1 - M_t) * ct + M_t * x_tilde_t

                x_bar_t_full = torch.cat((x_bar_t, cbct), dim=1)
                eps_bar, _ = self.model(x_bar_t_full, t)
                eps_tilde = eps_bar + (1 + self.s) * (eps - eps_bar)

                alpha_t = extract(self.alphas, t, ct.shape)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                one_minus_alpha_t = 1 - alpha_t
                mean = (1 / sqrt_alpha_t) * (ct - (one_minus_alpha_t / sqrt_one_minus_alpha_bar_t) * eps_tilde)

                noise = torch.randn_like(ct) if time_step > 0 else 0
                ct = mean + torch.sqrt(var) * noise
                x_t = torch.cat((ct, cbct), dim=1)

            return torch.clamp(x_t, -1, 1)