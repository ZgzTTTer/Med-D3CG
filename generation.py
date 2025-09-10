import torch
import numpy as np
import os
from models.base import BaseUNet
from models.attention import MidAttnUNet, UpAttnUNet
from models.wavelet import WTUNet
from diffusion.model_factory import get_trainer_sampler, get_available_models
from PIL import Image
import warnings
import argparse

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model configuration
    available_models = get_available_models()["all_models"]
    parser.add_argument("--model_name", type=str, default="D3CG_db4",
                        choices=available_models + ["D3CG_custom"])
    
    # D3CG自定义配置参数
    parser.add_argument("--wave_type", type=str, default="haar",
                        choices=["haar", "db4", "coif3", "bior2.2", "dmey"],
                        help="Wavelet type for D3CG models")
    parser.add_argument("--nonlinear_type", type=str, default="linear",
                        choices=["linear", "tanh", "sigmoid", "leaky_relu"],
                        help="Nonlinear function type for D3CG models")
    parser.add_argument("--transform_levels", type=int, default=1,
                        help="Number of wavelet transform levels")
    
    # Input/Output
    parser.add_argument("--condition_file", type=str, default="./input.png")
    parser.add_argument("--output_dir", type=str, default="./results/")
    parser.add_argument("--model_weight_path", type=str, default="./results/best_model_epoch_xxxx.pt")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    
    # Nonlinear parameters
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--negative_slope", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=1.0)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model based on model type
    if args.model_name in ["DDPM", "DIFFDDPM"]:
        net_model = BaseUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                           num_res_blocks=2, dropout=0.3).to(device)
    elif args.model_name == "midattnDDPM":
        net_model = MidAttnUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                              num_res_blocks=2, dropout=0.3).to(device)
    elif args.model_name == "UpAttnDDPM":
        net_model = UpAttnUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                             num_res_blocks=2, dropout=0.3).to(device)
    else:  # All wavelet-based models
        net_model = WTUNet(args.T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                         num_res_blocks=2, dropout=0.3).to(device)

    net_model.load_state_dict(torch.load(args.model_weight_path, map_location=device, weights_only=True))
    net_model.eval()

    # Get appropriate sampler using the factory
    _, sampler = get_trainer_sampler(
        model_name=args.model_name,
        net_model=net_model,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        device=device,
        # D3CG自定义参数
        wave_type=args.wave_type,
        nonlinear_type=args.nonlinear_type,
        transform_levels=args.transform_levels,
        # 非线性函数参数
        alpha=args.alpha,
        beta=args.beta,
        negative_slope=args.negative_slope,
        scale=args.scale
    )

    # Load and preprocess condition image
    condition_image = Image.open(args.condition_file).convert('L').resize((args.image_size, args.image_size))
    condition_np = np.array(condition_image).astype(np.float32) / 255.0
    condition_np = condition_np * 2 - 1
    condition_tensor = torch.from_numpy(condition_np).unsqueeze(0).unsqueeze(0).to(device)

    print(f"Generating image using {args.model_name}...")
    
    with torch.no_grad():
        # Different sampling strategies for different models
        if args.model_name.startswith("D3CG") or args.model_name == "WTDDPM":
            generated_images = sampler(condition_tensor)
        else:
            random_noise = torch.randn_like(condition_tensor)
            x_T = torch.cat((random_noise, condition_tensor), dim=1)
            generated_images = sampler(x_T)

    # Extract generated image
    generated_image = generated_images[:, 0, :, :]
    generated_image_np = generated_image[0].detach().cpu().numpy()

    # Normalize to [0, 1]
    generated_image_norm = (generated_image_np + 1) / 2
    generated_image_norm = np.clip(generated_image_norm, 0, 1)

    # Convert to uint8
    generated_image_uint8 = (generated_image_norm * 255).astype(np.uint8)

    # Save generated image
    output_filename = f"generated_{args.model_name}_{os.path.splitext(os.path.basename(args.condition_file))[0]}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    Image.fromarray(generated_image_uint8, mode='L').save(output_path)
    
    print(f"Generated image saved: {output_path}")
    print(f"Model used: {args.model_name}")