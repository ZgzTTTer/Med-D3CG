import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import numpy as np
import math


class WaveletTransformBase:
    """小波变换基类"""
    
    def __init__(self, wave="haar", device="cuda"):
        self.wave = wave
        self.device = torch.device(device)
    
    def forward_transform(self, x):
        raise NotImplementedError
    
    def inverse_transform(self, coeffs, original_size):
        raise NotImplementedError


class PyWTTransform(WaveletTransformBase):
    """使用PyWavelets的小波变换（原始实现）"""
    
    def forward_transform(self, x):
        B, C, H, W = x.shape
        coeffs_list = []
        for i in range(B):
            coeffs_channels = []
            for c in range(C):
                img = x[i, c, :, :].cpu().numpy()
                coeffs2 = pywt.dwt2(img, self.wave)
                cA, (cH, cV, cD) = coeffs2
                coeffs_channels.append(torch.from_numpy(cA))
                coeffs_channels.append(torch.from_numpy(cH))
                coeffs_channels.append(torch.from_numpy(cV))
                coeffs_channels.append(torch.from_numpy(cD))
            coeffs_list.append(torch.stack(coeffs_channels, dim=0))
        return torch.stack(coeffs_list, dim=0).to(self.device)
    
    def inverse_transform(self, coeffs_tensor, original_size):
        B, ch, H, W = coeffs_tensor.shape
        C = ch // 4
        reconstructed = []
        
        for i in range(B):
            if C == 1:
                cA = coeffs_tensor[i, 0].cpu().numpy()
                cH = coeffs_tensor[i, 1].cpu().numpy()
                cV = coeffs_tensor[i, 2].cpu().numpy()
                cD = coeffs_tensor[i, 3].cpu().numpy()
                rec = pywt.idwt2((cA, (cH, cV, cD)), self.wave)
                rec = torch.tensor(rec).unsqueeze(0)
                rec = F.interpolate(rec.unsqueeze(0), size=original_size, 
                                  mode='bilinear', align_corners=False).squeeze(0)
                reconstructed.append(rec)
            else:
                channels = []
                for c in range(C):
                    cA = coeffs_tensor[i, c*4+0].cpu().numpy()
                    cH = coeffs_tensor[i, c*4+1].cpu().numpy()
                    cV = coeffs_tensor[i, c*4+2].cpu().numpy()
                    cD = coeffs_tensor[i, c*4+3].cpu().numpy()
                    rec = pywt.idwt2((cA, (cH, cV, cD)), self.wave)
                    rec = torch.tensor(rec)
                    rec = F.interpolate(rec.unsqueeze(0).unsqueeze(0), size=original_size,
                                      mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    channels.append(rec)
                reconstructed.append(torch.stack(channels, dim=0))
        
        return torch.stack(reconstructed, dim=0).to(self.device)


class PyTorchWaveletTransform(WaveletTransformBase):
    """使用PyTorch-Wavelets的小波变换（统一实现）"""
    
    def __init__(self, wave="haar", J=1, device="cuda"):
        super().__init__(wave, device)
        self.J = J
        self.mode = "periodization"
        self.dwt = DWTForward(J=J, wave=wave, mode=self.mode).to(self.device)
        self.idwt = DWTInverse(wave=wave, mode=self.mode).to(self.device)
        self._input_shape = None
        
        # 记录不同小波基的特性
        self._is_orthogonal = wave in ['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']
        self._needs_size_adjustment = wave not in ['haar']  # haar以外的小波可能需要尺寸调整
    
    def forward_transform(self, x):
        x = x.to(self.device)
        self._input_shape = x.shape[2:]
        
        Yl, Yh_list = self.dwt(x)
        
        # 确保小波系数的尺寸是输入的一半
        target_size = (self._input_shape[0] // 2, self._input_shape[1] // 2)
        
        # 调整低频系数尺寸
        if Yl.shape[2:] != target_size:
            Yl = F.interpolate(Yl, size=target_size, mode='bilinear', align_corners=False)
        
        # 将小波系数转换为张量格式
        if self.J == 1:
            # 单层小波变换
            Yh = Yh_list[0]  # [B, C, 3, H', W']
            B, C, _, H, W = Yh.shape
            Yh = Yh.view(B, C * 3, H, W)  # [B, C*3, H', W']
            
            # 调整高频系数尺寸
            if Yh.shape[2:] != target_size:
                Yh = F.interpolate(Yh, size=target_size, mode='bilinear', align_corners=False)
                
            return torch.cat([Yl, Yh], dim=1)
        else:
            # 多层小波变换，这里以两层为例
            Yh1 = Yh_list[0]  # 第一层高频 [B, C, 3, H1, W1]
            Yh2 = Yh_list[1]  # 第二层高频 [B, C, 3, H2, W2]
            
            # 只使用第二层（最细）高频系数，忽略第一层
            B, C, _, H2, W2 = Yh2.shape
            Yh2 = Yh2.view(B, C * 3, H2, W2)
            # 将第二层高频系数调整到目标尺寸
            if Yh2.shape[2:] != target_size:
                Yh2 = F.interpolate(Yh2, size=target_size, mode='bilinear', align_corners=False)
            
            # 只返回低频系数和第二层高频系数
            return torch.cat([Yl, Yh2], dim=1)
    
    def inverse_transform(self, coeffs_tensor, original_size=None):
        if original_size is None:
            original_size = self._input_shape
            
        if self.J == 1:
            # 单层小波变换的逆变换
            C_out = coeffs_tensor.shape[1] // 4
            Yl = coeffs_tensor[:, :C_out]
            Yh_flat = coeffs_tensor[:, C_out:]  # [B, C*3, H', W']
            
            # 重新整形为正确的格式
            B, C3, H, W = Yh_flat.shape
            C = C3 // 3
            Yh = Yh_flat.view(B, C, 3, H, W)  # [B, C, 3, H', W']
            
            Yh_list = [Yh]
        else:
            # 多层小波变换的逆变换
            # 两层小波变换（只使用第二层）：低频(1) + 第二层高频(3) = 4个系数组
            total_channels = coeffs_tensor.shape[1]
            C_out = total_channels // 4  # 修改为4个系数组
            Yl = coeffs_tensor[:, :C_out]
            
            # 第二层高频系数（从第C_out个通道开始）
            Yh2_flat = coeffs_tensor[:, C_out:]
            B, C3, H2, W2 = Yh2_flat.shape
            C = C3 // 3
            Yh2 = Yh2_flat.view(B, C, 3, H2, W2)
            
            # 第一层高频系数使用条件图像的系数（或者设为零）
            # 这里我们需要从某个地方获取第一层系数，或者设为零
            # 为了简化，我们使用第二层系数的平均值作为第一层的近似
            Yh1 = torch.zeros(B, C, 3, H2*2, W2*2, device=coeffs_tensor.device)
            
            # 调整第二层系数尺寸（如果需要）
            if self.J == 2:
                # 第二层系数应该是第一层的一半尺寸  
                target_size_level2 = (H2, W2)  # 保持原尺寸
                if Yh2.shape[3:] != target_size_level2:
                    Yh2 = F.interpolate(Yh2.view(B, C*3, H2, W2), 
                                      size=target_size_level2, 
                                      mode='bilinear', align_corners=False)
                    Yh2 = Yh2.view(B, C, 3, target_size_level2[0], target_size_level2[1])
            
            Yh_list = [Yh1, Yh2]
        
        x_rec = self.idwt((Yl, Yh_list))
        
        # 确保输出尺寸正确
        if x_rec.shape[2:] != original_size:
            x_rec = F.interpolate(x_rec, original_size, mode="bilinear", align_corners=False)
        
        return x_rec


class PyWTTransformFixed(WaveletTransformBase):
    """修复尺寸问题的PyWavelets小波变换"""
    
    def __init__(self, wave="haar", device="cuda"):
        super().__init__(wave, device)
        self._input_shape = None
    
    def _get_target_size(self, input_size):
        """计算目标尺寸，确保是输入的一半"""
        return (input_size[0] // 2, input_size[1] // 2)
    
    def forward_transform(self, x):
        B, C, H, W = x.shape
        self._input_shape = (H, W)
        target_size = self._get_target_size(self._input_shape)
        
        coeffs_list = []
        for i in range(B):
            coeffs_channels = []
            for c in range(C):
                img = x[i, c, :, :].cpu().numpy()
                coeffs2 = pywt.dwt2(img, self.wave)
                cA, (cH, cV, cD) = coeffs2
                
                # 转换为张量并调整尺寸
                cA_tensor = torch.from_numpy(cA).to(self.device)
                cH_tensor = torch.from_numpy(cH).to(self.device)
                cV_tensor = torch.from_numpy(cV).to(self.device)
                cD_tensor = torch.from_numpy(cD).to(self.device)
                
                # 确保所有系数都是目标尺寸
                if cA_tensor.shape != target_size:
                    cA_tensor = F.interpolate(cA_tensor.unsqueeze(0).unsqueeze(0), 
                                            size=target_size, mode='bilinear', align_corners=False).squeeze()
                if cH_tensor.shape != target_size:
                    cH_tensor = F.interpolate(cH_tensor.unsqueeze(0).unsqueeze(0), 
                                            size=target_size, mode='bilinear', align_corners=False).squeeze()
                if cV_tensor.shape != target_size:
                    cV_tensor = F.interpolate(cV_tensor.unsqueeze(0).unsqueeze(0), 
                                            size=target_size, mode='bilinear', align_corners=False).squeeze()
                if cD_tensor.shape != target_size:
                    cD_tensor = F.interpolate(cD_tensor.unsqueeze(0).unsqueeze(0), 
                                            size=target_size, mode='bilinear', align_corners=False).squeeze()
                
                coeffs_channels.append(cA_tensor)
                coeffs_channels.append(cH_tensor)
                coeffs_channels.append(cV_tensor)
                coeffs_channels.append(cD_tensor)
            
            coeffs_list.append(torch.stack(coeffs_channels, dim=0))
        
        return torch.stack(coeffs_list, dim=0)
    
    def inverse_transform(self, coeffs_tensor, original_size=None):
        if original_size is None:
            original_size = self._input_shape
            
        B, ch, H, W = coeffs_tensor.shape
        C = ch // 4
        reconstructed = []
        
        for i in range(B):
            if C == 1:
                cA = coeffs_tensor[i, 0].cpu().numpy()
                cH = coeffs_tensor[i, 1].cpu().numpy()
                cV = coeffs_tensor[i, 2].cpu().numpy()
                cD = coeffs_tensor[i, 3].cpu().numpy()
                
                rec = pywt.idwt2((cA, (cH, cV, cD)), self.wave)
                rec = torch.tensor(rec, device=self.device).unsqueeze(0)
                
                # 确保输出尺寸正确
                if rec.shape[1:] != original_size:
                    rec = F.interpolate(rec.unsqueeze(0), size=original_size, 
                                      mode='bilinear', align_corners=False).squeeze(0)
                reconstructed.append(rec)
            else:
                channels = []
                for c in range(C):
                    cA = coeffs_tensor[i, c*4+0].cpu().numpy()
                    cH = coeffs_tensor[i, c*4+1].cpu().numpy()
                    cV = coeffs_tensor[i, c*4+2].cpu().numpy()
                    cD = coeffs_tensor[i, c*4+3].cpu().numpy()
                    
                    rec = pywt.idwt2((cA, (cH, cV, cD)), self.wave)
                    rec = torch.tensor(rec, device=self.device)
                    
                    # 确保输出尺寸正确
                    if rec.shape != original_size:
                        rec = F.interpolate(rec.unsqueeze(0).unsqueeze(0), size=original_size,
                                          mode='bilinear', align_corners=False).squeeze()
                    channels.append(rec)
                reconstructed.append(torch.stack(channels, dim=0))
        
        return torch.stack(reconstructed, dim=0)
def get_wavelet_transform(wave_type="haar", transform_levels=1, device="cuda", backend="pytorch"):
    """工厂函数：根据参数创建相应的小波变换对象"""
    
    if backend == "pywt":
        return PyWTTransformFixed(wave=wave_type, device=device)
    elif backend == "pytorch":
        return PyTorchWaveletTransform(wave=wave_type, J=transform_levels, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")