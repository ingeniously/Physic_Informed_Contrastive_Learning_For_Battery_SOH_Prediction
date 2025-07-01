from dataloader.dataloader import SimpleLoader
from Model.Model import PINN
from contrastive_Learning.contrastive import ContrastivePINNWrapper
import argparse
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

def load_data(args):
    data = SimpleLoader(
        csv_path=args.csv_file,
        batch_size=args.batch_size,
        normalization=True,
        normalization_method=args.normalization_method
    )
    loaders = data.load()
    return loaders

def load_augmented_data(args):
    data = SimpleLoader(
        csv_path=args.csv_file_augmented,
        batch_size=args.batch_size,
        normalization=True,
        normalization_method=args.normalization_method
    )
    loaders = data.load()
    return loaders

def train_contrastive_pinn(wrapper, orig_loader, aug_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr, weight_decay=1e-5)
    train_loss_list, contrastive_loss_list, pinn_loss_list = [], [], []
    pde_loss_list, physics_loss_list = [], []
     
    for epoch in range(args.epochs):
        wrapper.train()
        total_loss_epoch, pinn_loss_epoch, contrastive_loss_epoch = [], [], []
        pde_loss_epoch, physics_loss_epoch = [], []
        orig_iter = iter(orig_loader['train'])
        aug_iter = iter(aug_loader['train'])
        for _ in range(len(orig_loader['train'])):
            try:
                x1, _, y1, _ = next(orig_iter)
                x2, _, y2, _ = next(aug_iter)
            except StopIteration:
                break
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            optimizer.zero_grad()
            total_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss = wrapper(x1, x2, y1, y2)
            total_loss.backward()
            optimizer.step()
            total_loss_epoch.append(total_loss.item())
            pinn_loss_epoch.append(pinn_loss.item())
            contrastive_loss_epoch.append(contrastive_loss.item())
            pde_loss_epoch.append(pde_loss.item())
            physics_loss_epoch.append(physics_loss.item())
        avg_total = sum(total_loss_epoch) / len(total_loss_epoch)
        avg_pinn = sum(pinn_loss_epoch) / len(pinn_loss_epoch)
        avg_contrastive = sum(contrastive_loss_epoch) / len(contrastive_loss_epoch)
        avg_pde = sum(pde_loss_epoch) / len(pde_loss_epoch)
        avg_phys = sum(physics_loss_epoch) / len(physics_loss_epoch)
        train_loss_list.append(avg_total)
        pinn_loss_list.append(avg_pinn)
        contrastive_loss_list.append(avg_contrastive)
        pde_loss_list.append(avg_pde)
        physics_loss_list.append(avg_phys)
        print(f"[Epoch {epoch+1}] Total loss: {avg_total:.4f} | PINN loss: {avg_pinn:.4f} | Contrastive loss: {avg_contrastive:.4f} | PDE loss: {avg_pde:.4f} | Physics loss: {avg_phys:.4f}")

    # Plot learning curves
    plt.figure(figsize=(10,7))
    plt.plot(train_loss_list, label="Total Loss")
    plt.plot(pinn_loss_list, label="PINN Loss")
    plt.plot(contrastive_loss_list, label="Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, "contrastive_loss_curve.png"))
    plt.show()

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--csv_file_augmented', type=str, required=True, help='Path to augmented CSV file')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
    parser.add_argument('--epochs', type=int, default=300, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='base lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='learning rate for F network')
    parser.add_argument('--save_folder', type=str, default='results', help='save folder')
    parser.add_argument('--alpha', type=float, default=1.0, help='PDE loss weight')
    parser.add_argument('--beta', type=float, default=0.5, help='physics constraint weight')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='contrastive loss weight')
    parser.add_argument('--pinn_weight', type=float, default=1.0, help='PINN loss weight')
    parser.add_argument('--temperature', type=float, default=0.07, help='contrastive temperature')
    parser.add_argument('--log_dir', type=str, default='training_log.txt', help='log dir')
    parser.add_argument('--F_layers_num', type=int, default=4, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=128, help='the hidden dim of F')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--final_lr', type=float, default=1e-5, help='final lr')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    return parser.parse_args()

def main():
    args = get_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    setattr(args, "log_dir", args.log_dir)
    setattr(args, "save_folder", args.save_folder)

    # Load original and augmented data
    orig_loader = load_data(args)
    aug_loader = load_augmented_data(args)

    # Initialize PINN and wrapper
    pinn = PINN(args)
    wrapper = ContrastivePINNWrapper(
        pinn,
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight,
        pinn_weight=args.pinn_weight
    )

    # Train with contrastive + PINN loss
    train_contrastive_pinn(wrapper, orig_loader, aug_loader, args)

if __name__ == '__main__':
    main()