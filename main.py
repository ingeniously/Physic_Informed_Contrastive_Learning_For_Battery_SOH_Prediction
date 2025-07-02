from dataloader.dataloader import SimpleLoader
from Model.Model import PINN
from contrastive_Learning.contrastive import ContrastivePINNWrapper
import argparse
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import r2_score

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
    
    # Use a more effective optimizer setup with lower learning rate
    optimizer = torch.optim.AdamW(
        wrapper.parameters(), 
        lr=args.lr * 0.5,  # Lower learning rate for more stable training
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Better learning rate schedule with warm-up and cosine decay
    total_steps = args.epochs * min(len(orig_loader['train']), len(aug_loader['train']))
    warmup_steps = int(total_steps * 0.1)  # 10% of steps for warmup
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.final_lr)
    
    # Loss tracking
    train_loss_list, contrastive_loss_list, pinn_loss_list = [], [], []
    pde_loss_list, physics_loss_list = [], []
    
    # Training logs
    log_file = os.path.join(args.save_folder, "contrastive_training_log.txt")
    with open(log_file, 'w') as f:
        f.write("Epoch\tTotal Loss\tPINN Loss\tContrastive Loss\tPDE Loss\tPhysics Loss\n")
    
    # Adaptive contrastive weight - start lower
    contrastive_weight = args.contrastive_weight * 0.5
    wrapper.contrastive_weight = contrastive_weight
    
    # Best validation metrics tracking
    best_val_loss = float('inf')
    best_epoch = 0
     
    for epoch in range(args.epochs):
        wrapper.train()
        total_loss_epoch, pinn_loss_epoch, contrastive_loss_epoch = [], [], []
        pde_loss_epoch, physics_loss_epoch = [], []
        
        # Gradually decrease contrastive weight over time
        if epoch > args.epochs // 3:
            # Gradually reduce contrastive weight to focus more on PINN losses
            contrastive_weight = max(0.05, args.contrastive_weight * 0.5 * (1.0 - epoch / args.epochs))
            wrapper.contrastive_weight = contrastive_weight
            print(f"Contrastive weight adjusted to {contrastive_weight:.4f}")
        
        # Reset iterators for each epoch
        orig_iter = iter(orig_loader['train'])
        aug_iter = iter(aug_loader['train'])
        
        # Determine the minimum number of batches and use all original data
        n_batches = len(orig_loader['train'])
        
        for step in range(n_batches):
            # Dynamic learning rate during warmup
            if step + epoch * n_batches < warmup_steps:
                lr_scale = min(1., float(step + epoch * n_batches) / float(warmup_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scale * args.lr * 0.5  # Lower base lr
            
            # Get original data
            x1, _, y1, _ = next(orig_iter)
            
            # Handle the case where we've gone through all augmented data
            try:
                x2, _, y2, _ = next(aug_iter)
            except StopIteration:
                # Reset augmented data iterator if we run out
                aug_iter = iter(aug_loader['train'])
                x2, _, y2, _ = next(aug_iter)
                
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            
            optimizer.zero_grad()
            total_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss = wrapper(x1, x2, y1, y2)
            
            # Gradient clipping for stability
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Log losses
            total_loss_epoch.append(total_loss.item())
            pinn_loss_epoch.append(pinn_loss.item())
            contrastive_loss_epoch.append(contrastive_loss.item())
            pde_loss_epoch.append(pde_loss.item())
            physics_loss_epoch.append(physics_loss.item())
            
            # Print progress more frequently
            if (step + 1) % 1887 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}][Step {step+1}/{n_batches}] "
                      f"Loss: {total_loss.item():.4f} | PINN: {pinn_loss.item():.4f} | "
                      f"Contrastive: {contrastive_loss.item():.4f} | PDE: {pde_loss.item():.4f}")
        
        # Step the scheduler at the end of each epoch
        scheduler.step()
        
        # Compute average losses for the epoch
        avg_total = sum(total_loss_epoch) / len(total_loss_epoch)
        avg_pinn = sum(pinn_loss_epoch) / len(pinn_loss_epoch)
        avg_contrastive = sum(contrastive_loss_epoch) / len(contrastive_loss_epoch)
        avg_pde = sum(pde_loss_epoch) / len(pde_loss_epoch)
        avg_phys = sum(physics_loss_epoch) / len(physics_loss_epoch)
        
        # Store for plotting
        train_loss_list.append(avg_total)
        pinn_loss_list.append(avg_pinn)
        contrastive_loss_list.append(avg_contrastive)
        pde_loss_list.append(avg_pde)
        physics_loss_list.append(avg_phys)
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}\t{avg_total:.6f}\t{avg_pinn:.6f}\t{avg_contrastive:.6f}\t{avg_pde:.6f}\t{avg_phys:.6f}\n")
        
        print(f"[Epoch {epoch+1}] Total loss: {avg_total:.4f} | PINN loss: {avg_pinn:.4f} | "
              f"Contrastive loss: {avg_contrastive:.4f} | PDE loss: {avg_pde:.4f} | Physics loss: {avg_phys:.4f}")
        
        # Evaluate on validation set every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            wrapper.pinn.eval()
            with torch.no_grad():
                val_loss = wrapper.pinn.Valid(orig_loader['valid'])
                print(f"[Validation] Epoch {epoch+1}: MSE = {val_loss:.6f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'pinn_state_dict': wrapper.pinn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'contrastive_weight': contrastive_weight
            }
            
            # Save model if it's the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(checkpoint, os.path.join(args.save_folder, 'best_model.pth'))
                print(f"New best model saved at epoch {epoch+1} with validation MSE: {val_loss:.6f}")
                
                # Test on test set with best model
                wrapper.pinn.eval()
                true_label, pred_label = wrapper.pinn.Test(orig_loader['test'])
                from utils.util import eval_metrix
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                r2 = r2_score(true_label, pred_label)
                print(f"[Test] MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}, R²: {r2:.4f}")
                
                # Save predictions for analysis
                np.save(os.path.join(args.save_folder, 'true_label.npy'), true_label)
                np.save(os.path.join(args.save_folder, 'pred_label.npy'), pred_label)
                
                # Generate scatter plot
                create_prediction_scatter(true_label, pred_label, os.path.join(args.save_folder, f'prediction_scatter_epoch_{epoch+1}.png'))
    
    # Save the final model
    torch.save({
        'epoch': args.epochs,
        'pinn_state_dict': wrapper.pinn.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(args.save_folder, 'final_model.pth'))
    
    # Plot learning curves
    plot_learning_curves(train_loss_list, pinn_loss_list, contrastive_loss_list, 
                         pde_loss_list, physics_loss_list, args.save_folder)
    
    # Load best model and generate final visualization
    try:
        checkpoint = torch.load(os.path.join(args.save_folder, 'best_model.pth'))
        wrapper.pinn.load_state_dict(checkpoint['pinn_state_dict'])
        wrapper.pinn.eval()
        true_label, pred_label = wrapper.pinn.Test(orig_loader['test'])
        create_prediction_scatter(true_label, pred_label, os.path.join(args.save_folder, 'final_prediction_scatter.png'))
    except:
        print("Could not load best model for final visualization")
    
    print(f"Training completed. Best model at epoch {best_epoch} with validation MSE: {best_val_loss:.6f}")
    return best_val_loss

def create_prediction_scatter(true_label, pred_label, save_path):
    """
    Create scatter plot of true vs predicted values with additional metrics.
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate metrics for annotation
    mse = np.mean((true_label - pred_label) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_label - pred_label))
    r2 = r2_score(true_label, pred_label)
    
    # Create scatter plot
    plt.scatter(true_label, pred_label, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(true_label.min(), pred_label.min())
    max_val = max(true_label.max(), pred_label.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add metrics text
    plt.annotate(f'MSE = {mse:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.xlabel('True SoH')
    plt.ylabel('Predicted SoH')
    plt.title('Battery SoH: True vs Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction scatter plot saved to {save_path}")

def plot_learning_curves(train_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss, save_folder):
    """
    Plot and save learning curves for all loss components.
    """
    epochs = range(1, len(train_loss) + 1)
    
    # Main losses plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, 'b-', label='Total Loss')
    plt.plot(epochs, pinn_loss, 'r-', label='PINN Loss')
    plt.plot(epochs, contrastive_loss, 'g-', label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Physics components plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, pde_loss, 'm-', label='PDE Loss')
    plt.plot(epochs, physics_loss, 'c-', label='Physics Constraint Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Physics Component Losses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "learning_curves.png"))
    plt.close()
    
    # Log-scale plot for better visualization of small losses
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.semilogy(epochs, train_loss, 'b-', label='Total Loss')
    plt.semilogy(epochs, pinn_loss, 'r-', label='PINN Loss')
    plt.semilogy(epochs, contrastive_loss, 'g-', label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Losses (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(epochs, pde_loss, 'm-', label='PDE Loss')
    plt.semilogy(epochs, physics_loss, 'c-', label='Physics Constraint Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Physics Component Losses (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "learning_curves_log.png"))
    plt.close()

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--csv_file_augmented', type=str, required=True, help='Path to augmented CSV file')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
    parser.add_argument('--epochs', type=int, default=300, help='epoch')
    parser.add_argument('--early_stop', type=int, default=30, help='early stop patience')
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
    parser.add_argument('--momentum', type=float, default=0.999, help='momentum for encoder')
    parser.add_argument('--projection_dim', type=int, default=128, help='projection dimension for contrastive learning')
    parser.add_argument('--queue_size', type=int, default=4096, help='queue size for momentum contrast')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    args = get_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Save configuration
    with open(os.path.join(args.save_folder, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    # Log setup
    setattr(args, "log_dir", args.log_dir)
    setattr(args, "save_folder", args.save_folder)

    # Load original and augmented data
    print("Loading original dataset...")
    orig_loader = load_data(args)
    print(f"Original dataset loaded: {len(orig_loader['train'])} training batches")
    
    print("Loading augmented dataset...")
    aug_loader = load_augmented_data(args)
    print(f"Augmented dataset loaded: {len(aug_loader['train'])} training batches")

    # Initialize PINN
    print("Initializing PINN model...")
    pinn = PINN(args)
    
    # Initialize contrastive wrapper with projection head
    print("Setting up contrastive learning wrapper...")
    wrapper = ContrastivePINNWrapper(
        pinn,
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight * 0.5,  # Start with lower contrastive weight
        pinn_weight=args.pinn_weight,
        projection_dim=args.projection_dim,
        momentum=args.momentum,
        queue_size=args.queue_size
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    
    # Count parameters
    pinn_params = sum(p.numel() for p in pinn.parameters() if p.requires_grad)
    wrapper_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    print(f"PINN model has {pinn_params} trainable parameters")
    print(f"Full contrastive wrapper has {wrapper_params} trainable parameters")

    # Train with contrastive + PINN loss
    print("Starting training...")
    train_contrastive_pinn(wrapper, orig_loader, aug_loader, args)

if __name__ == '__main__':
    main()