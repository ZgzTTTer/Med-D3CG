import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
import logging
import numpy as np
import argparse

from models.base import BaseUNet
from models.attention import MidAttnUNet, UpAttnUNet
from models.wavelet import WTUNet
from utils.profiling import profile_model, log_profiling_results, get_model_input_shape
from diffusion.model_factory import get_trainer_sampler, get_available_models, print_model_info
from data.dataset import FootDataset2, MRIPET, MRISPECT, CTMRI
from utils.metrics import calculate_metrics, calculate_metrics_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model selection
    available_models = get_available_models()["all_models"]
    parser.add_argument("--model_name", type=str, default="D3CG_custom",
                        choices=available_models + ["D3CG_custom"])
    
    # D3CG自定义配置参数
    parser.add_argument("--wave_type", type=str, default="db4",
                        choices=["haar", "db4", "coif3", "bior2.2", "dmey"],
                        help="Wavelet type for D3CG models")
    parser.add_argument("--nonlinear_type", type=str, default="sigmoid",
                        choices=["linear", "tanh", "sigmoid", "leaky_relu"],
                        help="Nonlinear function type for D3CG models")
    parser.add_argument("--transform_levels", type=int, default=1,
                        help="Number of wavelet transform levels")
    
    # Dataset configuration
    parser.add_argument("--dataset_type", type=str, default="ctmri",
                        choices=["foot", "mripet", "mrispect", "ctmri"])
    parser.add_argument("--dataset_train_dir", type=str, default="../datasets/HavardMedicalImage/CT-MRI/train/")
    parser.add_argument("--dataset_val_dir", type=str, default="../datasets/HavardMedicalImage/CT-MRI/val/")
    
    # Training configuration
    parser.add_argument("--out_name", type=str, default="D3CG_custom")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument("--attn", nargs='+', type=int, default=[2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=4000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--image_size", type=int, default=256)
    
    # Attention parameters
    parser.add_argument("--psi", type=float, default=1.0)
    parser.add_argument("--s", type=float, default=0.1)
    
    # Nonlinear transformation parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Tanh steepness parameter")
    parser.add_argument("--beta", type=float, default=1.0, help="Sigmoid steepness parameter")
    parser.add_argument("--negative_slope", type=float, default=0.1, help="LeakyReLU negative slope")
    parser.add_argument("--scale", type=float, default=1.0, help="LeakyReLU scale parameter")
    
    # Training control
    parser.add_argument("--save_weight_dir", type=str, default="./results/CT-MRI")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--val_start_epoch", type=int, default=3951)
    parser.add_argument("--val_num", type=int, default=12)
    
    # 显示可用模型信息
    parser.add_argument("--list_models", action="store_true", help="List all available models")
    
    return parser.parse_args()


def should_save_model(current_metrics, best_metrics):
    """
    Determine if the model should be saved based on metric thresholds
    """
    ssim_threshold = 0.01  # Save if within 0.02 of best SSIM
    psnr_threshold = 0.1  # Save if within 0.2 of best PSNR

    ssim_condition = current_metrics['ssim'] >= best_metrics['ssim'] - ssim_threshold
    psnr_condition = current_metrics['psnr'] >= best_metrics['psnr'] - psnr_threshold
    mae_condition = current_metrics['mae'] < best_metrics['mae']

    # Update best metrics if current ones are better
    if current_metrics['ssim'] > best_metrics['ssim']:
        best_metrics['ssim'] = current_metrics['ssim']
    if current_metrics['psnr'] > best_metrics['psnr']:
        best_metrics['psnr'] = current_metrics['psnr']
    if current_metrics['mae'] < best_metrics['mae']:
        best_metrics['mae'] = current_metrics['mae']

    return ssim_condition or psnr_condition or mae_condition


def main():
    args = parse_args()
    
    # 显示可用模型信息
    if args.list_models:
        print_model_info()
        return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup logging
    save_weight_dir = os.path.join(args.save_weight_dir, args.out_name)
    os.makedirs(save_weight_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_weight_dir, "training_log.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Args: {args}")
    logging.info(f"Model: {args.model_name}")

    # Setup data
    dataset_classes = {
        "foot": FootDataset2,
        "ctmri": CTMRI,
        "mripet": MRIPET,
        "mrispect": MRISPECT
    }

    dataset_class = dataset_classes[args.dataset_type]
    train_dataset = dataset_class(args.dataset_train_dir, image_size=args.image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_dataset = dataset_class(args.dataset_val_dir, image_size=args.image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Determine input and output channels based on dataset type
    if args.dataset_type in ["mrispect"]:
        in_channels = 6  # RGB + RGB
        out_channels = 3  # RGB output
    else:
        in_channels = 2  # Two grayscale images
        out_channels = 1  # One grayscale output

    # Initialize model based on model type
    if args.model_name in ["DDPM", "DIFFDDPM"]:
        net_model = BaseUNet(
            args.T, args.ch, args.ch_mult, args.attn,
            args.num_res_blocks, args.dropout,
            in_channels=in_channels,
            out_channels=out_channels
        ).to(device)
    elif args.model_name == "midattnDDPM":
        net_model = MidAttnUNet(
            args.T, args.ch, args.ch_mult, args.attn,
            args.num_res_blocks, args.dropout,
            in_channels=in_channels,
            out_channels=out_channels
        ).to(device)
    elif args.model_name == "UpAttnDDPM":
        net_model = UpAttnUNet(
            args.T, args.ch, args.ch_mult, args.attn,
            args.num_res_blocks, args.dropout,
            in_channels=in_channels,
            out_channels=out_channels
        ).to(device)
    else:  # All wavelet-based models use WTUNet
        net_model = WTUNet(
            args.T, args.ch, args.ch_mult, args.attn,
            args.num_res_blocks, args.dropout,
            in_channels=in_channels,
            out_channels=out_channels
        ).to(device)

    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    # Initialize trainer and sampler using the factory
    trainer, sampler = get_trainer_sampler(
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
        scale=args.scale,
        # 注意力参数
        psi=args.psi,
        s=args.s
    )

    # Model profiling
    logging.info("Starting model profiling...")
    input_shape = get_model_input_shape(args.model_name, args.dataset_type, args.image_size, args.batch_size)
    profiling_results = profile_model(net_model, input_shape, device, args.model_name)
    
    # 将性能分析结果写入日志
    log_file_path = os.path.join(save_weight_dir, "training_log.log")
    log_profiling_results(profiling_results, args.model_name, log_file_path)
    
    logging.info(f"Model profiling completed:")
    logging.info(f"  Parameters: {profiling_results['total_params_M']:.2f}M")
    logging.info(f"  FLOPs: {profiling_results['flops_G']:.3f}G")
    logging.info(f"  Memory: {profiling_results['model_memory_MB']:.1f}MB")
    logging.info(f"  Inference Time: {profiling_results['inference_time_ms']:.2f}ms")

    # Resume from checkpoint if specified
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        net_model.load_state_dict(torch.load(args.resume_ckpt, map_location=device, weights_only=True))
        logging.info(f"Loaded checkpoint from {args.resume_ckpt}")

    # Training loop
    prev_time = time.time()
    best_metrics = {'psnr': -float('inf'), 'ssim': -float('inf'), 'mae': float('inf')}

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        net_model.train()
        losses = []

        # Training step
        for batch in train_dataloader:
            optimizer.zero_grad()
            condition = batch['condition'].to(device)
            target = batch['target'].to(device)
            x_0 = torch.cat((target, condition), 1)

            loss = trainer(x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        # Logging
        avg_loss = np.mean(losses)
        time_elapsed = datetime.timedelta(seconds=(time.time() - prev_time))
        time_left = datetime.timedelta(seconds=(args.n_epochs - epoch) * (time.time() - prev_time))
        prev_time = time.time()

        logging.info(
            f"[Epoch {epoch}/{args.n_epochs}] "
            f"[ETA: {time_left}] "
            f"[Duration: {time_elapsed}] "
            f"[Loss: {avg_loss:.4f}] "
            f"[Model: {args.model_name}]"
        )

        # Validation step
        if epoch >= args.val_start_epoch:
            net_model.eval()
            metrics_list = []

            with torch.no_grad():
                for i, eval_batch in enumerate(val_dataloader):
                    if i >= args.val_num:
                        break

                    condition = eval_batch['condition'].to(device)
                    target = eval_batch['target'].to(device)

                    # Different sampling strategies for different models
                    if args.model_name.startswith("D3CG") or args.model_name == "WTDDPM":
                        x_T = condition
                    else:
                        x_T = torch.cat((torch.randn_like(target), condition), 1)

                    generated_images = sampler(x_T)

                    # Extract only the target channels from the generated images
                    if args.dataset_type in ["mrispect"]:
                        generated_image = generated_images[0, :3].cpu().numpy()
                        target_image = target[0].cpu().numpy()

                        # Convert from CHW to HWC format
                        generated_image = generated_image.transpose(1, 2, 0)
                        target_image = target_image.transpose(1, 2, 0)

                        metrics = calculate_metrics_rgb(generated_image, target_image)
                    else:
                        generated_image = generated_images[0, 0].cpu().numpy()
                        target_image = target[0, 0].cpu().numpy()
                        metrics = calculate_metrics(generated_image, target_image)

                    metrics_list.append(metrics)

            # Calculate average metrics
            avg_metrics = {
                key: np.mean([m[key] for m in metrics_list])
                for key in metrics_list[0].keys()
            }

            logging.info(
                f"[Epoch {epoch}] "
                f"[SSIM: {avg_metrics['ssim']:.4f}] "
                f"[PSNR: {avg_metrics['psnr']:.4f}] "
                f"[MAE: {avg_metrics['mae']:.4f}] "
                f"[Model: {args.model_name}]"
            )

            # Save model if metrics meet the threshold criteria
            if should_save_model(avg_metrics, best_metrics):
                save_path = os.path.join(save_weight_dir, f'model_epoch_{epoch}.pt')
                torch.save(net_model.state_dict(), save_path)
                logging.info(
                    f"Model saved at epoch {epoch} with "
                    f"SSIM: {avg_metrics['ssim']:.4f} (best: {best_metrics['ssim']:.4f}), "
                    f"PSNR: {avg_metrics['psnr']:.4f} (best: {best_metrics['psnr']:.4f}), "
                    f"MAE: {avg_metrics['mae']:.4f} (best: {best_metrics['mae']:.4f}) "
                    f"[Model: {args.model_name}]"
                )

            # Save checkpoint periodically
            if epoch % 10 == 0:
                torch.save(
                    net_model.state_dict(),
                    os.path.join(save_weight_dir, f'ckpt_{epoch}.pt')
                )

        if epoch % 1000 == 0:
            torch.save(
                net_model.state_dict(),
                os.path.join(save_weight_dir, f'ckpt_{epoch}.pt')
            )


if __name__ == "__main__":
    main()