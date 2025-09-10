import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseDiffusionTrainer_cond, BaseDiffusionSampler_cond
from ..base import extract


class DiffDDPMTrainer_cond(BaseDiffusionTrainer_cond):
    def get_initial_signal(self, ct, cbct):
        return ct - cbct


class DiffDDPMSampler_cond(BaseDiffusionSampler_cond):
    def forward(self, x_T):
        with torch.no_grad():
            batch_size = x_T.shape[0]
            device = x_T.device

            out_channels = self.model.out_channels
            ct = x_T[:, :out_channels, :, :]
            cbct = x_T[:, out_channels:, :, :]

            d_t = torch.randn_like(ct)

            for time_step in reversed(range(self.T)):
                t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)

                x_t = torch.cat((d_t, cbct), dim=1)

                model_output, _ = self.model(x_t, t)
                eps = model_output

                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var, t, d_t.shape)

                sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, d_t.shape)
                sqrt_one_minus_alphas_bar_t = extract(self.sqrt_one_minus_alphas_bar, t, d_t.shape)

                alpha_t = extract(self.alphas, t, d_t.shape)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                one_minus_alpha_t = 1 - alpha_t

                mean = (1 / sqrt_alpha_t) * (d_t - (one_minus_alpha_t / sqrt_one_minus_alphas_bar_t) * eps)

                if time_step > 0:
                    noise = torch.randn_like(d_t)
                    d_t = mean + torch.sqrt(var) * noise
                else:
                    d_t = mean

            ct = cbct + d_t
            x_0 = torch.cat((ct, cbct), dim=1)

            return torch.clamp(x_0, -1, 1)