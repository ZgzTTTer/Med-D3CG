from .base import BaseDiffusionTrainer_cond, BaseDiffusionSampler_cond
import torch
from .base import extract

class GaussianDiffusionTrainer_cond(BaseDiffusionTrainer_cond):
    def get_initial_signal(self,ct,cbct):
        return ct

class GaussianDiffusionSampler_cond(BaseDiffusionSampler_cond):
    def forward(self,x_T):
        with torch.no_grad():
            x_t = x_T
            out_channels = self.model.out_channels
            ct = x_t[:, :out_channels, :, :]
            cbct = x_t[:, out_channels:, :, :]
            
            for time_step in reversed(range(self.T)):
                t = x_t.new_ones([x_T.shape[0],],dtype=torch.long)*time_step
                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var,t,ct.shape)
                model_output,_=self.model(x_t,t)
                eps=model_output

                sqrt_alphas_bar_t=torch.sqrt(extract(self.alphas_bar,t,ct.shape))
                sqrt_one_minus_alphas_bar_t=torch.sqrt(1-extract(self.alphas_bar,t,ct.shape))

                alpha_t=extract(self.alphas,t,ct.shape)
                one_minus_alpha_t=1-alpha_t
                mean=(1/torch.sqrt(alpha_t))*(ct-(one_minus_alpha_t/sqrt_one_minus_alphas_bar_t)*eps)
                if time_step>0:
                    noise=torch.randn_like(ct)
                else:
                    noise=0
                ct=mean+torch.sqrt(var)*noise
                x_t=torch.cat((ct,cbct),1)
            return torch.clamp(x_t,-1,1)