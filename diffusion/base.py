import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class BaseDiffusionTrainer_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def get_initial_signal(self, ct, cbct):
        raise NotImplementedError

    def forward(self, x_0):
        t = torch.randint(self.T, (x_0.shape[0],), device=x_0.device)
        out_channels = self.model.out_channels
        ct = x_0[:, :out_channels, :, :]
        cbct = x_0[:, out_channels:, :, :]

        initial_signal = self.get_initial_signal(ct, cbct)
        noise = torch.randn_like(initial_signal)
        x_t = extract(self.sqrt_alphas_bar, t, initial_signal.shape) * initial_signal + \
              extract(self.sqrt_one_minus_alphas_bar, t, initial_signal.shape) * noise

        x_t = torch.cat((x_t, cbct), 1)
        model_output, _ = self.model(x_t, t)
        loss = F.mse_loss(model_output, noise, reduction='sum')
        return loss


class BaseDiffusionSampler_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1.)[:T]
        posterior_var = self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        self.register_buffer('posterior_var', posterior_var)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_T):
        with torch.no_grad():
            x_t = x_T
            out_channels = self.model.out_channels
            ct = x_t[:, :out_channels, :, :]
            cbct = x_t[:, out_channels:, :, :]

            for time_step in reversed(range(self.T)):
                t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step

                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var, t, ct.shape)

                model_output, _ = self.model(x_t, t)
                eps = model_output

                sqrt_alphas_bar_t = torch.sqrt(extract(self.alphas_bar, t, ct.shape))
                sqrt_one_minus_alphas_bar_t = torch.sqrt(1 - extract(self.alphas_bar, t, ct.shape))

                alpha_t = extract(self.alphas, t, ct.shape)
                one_minus_alpha_t = 1 - alpha_t
                mean = (1 / torch.sqrt(alpha_t)) * (ct - (one_minus_alpha_t / sqrt_one_minus_alphas_bar_t) * eps)

                if time_step > 0:
                    noise = torch.randn_like(ct)
                else:
                    noise = 0

                ct = mean + torch.sqrt(var) * noise
                x_t = torch.cat((ct, cbct), 1)

            return torch.clamp(x_t, -1, 1)