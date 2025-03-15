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
from diffusion.gaussian import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond
from diffusion.attn.upattnDDPM import UpAttnDDPMTrainer_cond, UpAttnDDPMSampler_cond
from diffusion.attn.midattnDDPM import MidAttnDDPMTrainer_cond, MidAttnDDPMSampler_cond
from diffusion.ablation.DIFFDDPM import DiffDDPMTrainer_cond, DiffDDPMSampler_cond
from diffusion.ablation.wavelet import WTDDPMTrainer_cond, WTDDPMSampler_cond
from diffusion.D3CG import D3CGTrainer_cond, D3CGSampler_cond
from data.dataset import FootDataset2, MRIPET, MRISPECT, CTMRI
from utils.metrics import calculate_metrics, calculate_metrics_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="UpAttnDDPM",
                        choices=["DDPM", "DIFFDDPM", "midattnDDPM", "UpAttnDDPM", "WTDDPM", "D3CG"])
    parser.add_argument("--dataset_type", type=str, default="ctmri",
                        choices=["foot", "mripet", "mrispect", "ctmri"])
    parser.add_argument("--dataset_train_dir", type=str, default="/home/midi/datasets/SynthRAD2023pelvis/train/")
    parser.add_argument("--dataset_val_dir", type=str, default="/home/midi/datasets/SynthRAD2023pelvis/val/")
    parser.add_argument("--out_name", type=str, default="UpAttnDDPM")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument("--attn", nargs='+', type=int, default=[2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--psi", type=float, default=1.0)
    parser.add_argument("--s", type=float, default=0.1)
    parser.add_argument("--save_weight_dir", type=str, default="./results/CTMRIpelvis")
    parser.add_argument("--resume_ckpt", type=str, default="/home/midi/project/MEDD3CG/results/CTMRIpelvis/DDPM/ckpt_1970.pt")
    parser.add_argument("--start_epoch", type=int, default=1971)
    parser.add_argument("--val_start_epoch", type=int, default=1971)
    parser.add_argument("--val_num", type=int, default=20)
    return parser.parse_args()


def should_save_model(current_metrics, best_metrics):
    """
    Determine if the model should be saved based on metric thresholds
    """
    ssim_threshold = 0.02  # Save if within 0.02 of best SSIM
    psnr_threshold = 0.2  # Save if within 0.2 of best PSNR

    ssim_condition = current_metrics['ssim'] >= best_metrics['ssim'] - ssim_threshold
    psnr_condition = current_metrics['psnr'] >= best_metrics['psnr'] - psnr_threshold
    mae_condition = current_metrics['mae'] < best_metrics['mae']  # Keep original condition for MAE

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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
    if args.dataset_type in ["mripet", "mrispect"]:
        in_channels = 6  # RGB + RGB
        out_channels = 3  # RGB output
    else:
        in_channels = 2  # Two grayscale images
        out_channels = 1  # One grayscale output

    # Initialize model
    model_classes = {
        "DDPM": BaseUNet,
        "DIFFDDPM": BaseUNet,
        "midattnDDPM": MidAttnUNet,
        "UpAttnDDPM": UpAttnUNet,
        "WTDDPM": WTUNet,
        "D3CG": WTUNet,
    }

    net_model = model_classes[args.model_name](
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

    # Initialize trainer and sampler
    if args.model_name == "WTDDPM":
        trainer = WTDDPMTrainer_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
        sampler = WTDDPMSampler_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
    elif args.model_name == "D3CG":
        trainer = D3CGTrainer_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
        sampler = D3CGSampler_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
    elif args.model_name == "midattnDDPM":
        trainer = MidAttnDDPMTrainer_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
        sampler = MidAttnDDPMSampler_cond(net_model, args.beta_1, args.beta_T, args.T, args.psi, args.s).to(device)
    elif args.model_name == "UpAttnDDPM":
        trainer = UpAttnDDPMTrainer_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
        sampler = UpAttnDDPMSampler_cond(net_model, args.beta_1, args.beta_T, args.T, args.psi, args.s).to(device)
    elif args.model_name == "DIFFDDPM":
        trainer = DiffDDPMTrainer_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
        sampler = DiffDDPMSampler_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
    else:
        trainer = GaussianDiffusionTrainer_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)
        sampler = GaussianDiffusionSampler_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)

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
            f"[Loss: {avg_loss:.4f}]"
        )

        # Validation step
        if epoch >= args.val_start_epoch:
            net_model.eval()
            metrics_list = []

            with torch.no_grad():
                for i, eval_batch in enumerate(val_dataloader):
                    if i >= args.val_num:  # Validate on 16 samples
                        break

                    condition = eval_batch['condition'].to(device)
                    target = eval_batch['target'].to(device)

                    if args.model_name == "WTDDPM" or args.model_name == "D3CG":
                        x_T = condition
                    else:
                        x_T = torch.cat((torch.randn_like(target), condition), 1)

                    generated_images = sampler(x_T)

                    # Extract only the target channels from the generated images
                    if args.dataset_type in ["mripet", "mrispect"]:
                        generated_image = generated_images[0, :3].cpu().numpy()  # Take first 3 channels for RGB
                        target_image = target[0].cpu().numpy()

                        # Convert from CHW to HWC format
                        generated_image = generated_image.transpose(1, 2, 0)
                        target_image = target_image.transpose(1, 2, 0)

                        metrics = calculate_metrics_rgb(generated_image, target_image)
                    else:
                        generated_image = generated_images[0, 0].cpu().numpy()  # Take first channel for grayscale
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
                f"[MAE: {avg_metrics['mae']:.4f}]"
            )

            # Save model if metrics meet the threshold criteria
            if should_save_model(avg_metrics, best_metrics):
                save_path = os.path.join(save_weight_dir, f'model_epoch_{epoch}.pt')
                torch.save(net_model.state_dict(), save_path)
                logging.info(
                    f"Model saved at epoch {epoch} with "
                    f"SSIM: {avg_metrics['ssim']:.4f} (best: {best_metrics['ssim']:.4f}), "
                    f"PSNR: {avg_metrics['psnr']:.4f} (best: {best_metrics['psnr']:.4f}), "
                    f"MAE: {avg_metrics['mae']:.4f} (best: {best_metrics['mae']:.4f})"
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