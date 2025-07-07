from dataloader.dataloader import SimpleLoader
from Model.Model import PINN
from contrastive_Learning.contrastive import ContrastivePINNWrapper
import argparse
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import wandb  # Import wandb for visualization
import seaborn as sns  # For better plots
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

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

def visualize_training_progress(train_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss, epoch, save_folder):
    """
    Create detailed training progress visualization
    """
    # Create a 2x2 grid of plots for different metrics
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Plot 1: All losses combined
    ax1 = plt.subplot(gs[0, 0])
    epochs = range(1, epoch + 1)
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, pinn_loss, 'r-', linewidth=2, label='PINN Loss')
    ax1.plot(epochs, contrastive_loss, 'g-', linewidth=2, label='Contrastive Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Physics components
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(epochs, pde_loss, 'm-', linewidth=2, label='PDE Loss')
    ax2.plot(epochs, physics_loss, 'c-', linewidth=2, label='Physics Constraint Loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.set_title('Physics Component Losses', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Loss ratios
    ax3 = plt.subplot(gs[1, 0])
    contrastive_ratio = [c/t if t > 0 else 0 for c, t in zip(contrastive_loss, train_loss)]
    pinn_ratio = [p/t if t > 0 else 0 for p, t in zip(pinn_loss, train_loss)]
    physics_ratio = [ph/t if t > 0 else 0 for ph, t in zip(physics_loss, train_loss)]
    pde_ratio = [pd/t if t > 0 else 0 for pd, t in zip(pde_loss, train_loss)]
    
    ax3.stackplot(epochs, pinn_ratio, contrastive_ratio, pde_ratio, physics_ratio, 
                 labels=['PINN', 'Contrastive', 'PDE', 'Physics'],
                 alpha=0.7, colors=['#ff9999', '#99ff99', '#9999ff', '#ffcc99'])
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss Proportion', fontsize=12)
    ax3.set_title('Loss Component Proportions', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Log scale losses
    ax4 = plt.subplot(gs[1, 1])
    ax4.semilogy(epochs, train_loss, 'b-', linewidth=2, label='Total Loss')
    ax4.semilogy(epochs, pinn_loss, 'r-', linewidth=2, label='PINN Loss')
    ax4.semilogy(epochs, contrastive_loss, 'g-', linewidth=2, label='Contrastive Loss')
    ax4.semilogy(epochs, pde_loss, 'm-', linewidth=2, label='PDE Loss')
    ax4.semilogy(epochs, physics_loss, 'c-', linewidth=2, label='Physics Loss')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Value (log scale)', fontsize=12)
    ax4.set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"detailed_training_progress_epoch_{epoch}.png"), dpi=300)
    plt.close()
    
    # Upload to wandb
    wandb.log({"Detailed Training Progress": wandb.Image(
        os.path.join(save_folder, f"detailed_training_progress_epoch_{epoch}.png"))})

def train_contrastive_pinn(wrapper, orig_loader, aug_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use AdamW with better parameters
    optimizer = torch.optim.AdamW(
        wrapper.parameters(), 
        lr=args.lr * 0.3,  # Lower learning rate for more stable training
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Better learning rate schedule with warm-up and cosine decay with restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=15,      # Restart every 15 epochs
        T_mult=2,    # Double period after each restart
        eta_min=args.final_lr * 0.1  # Lower minimum learning rate
    )
    
    # Loss tracking
    train_loss_list, contrastive_loss_list, pinn_loss_list = [], [], []
    pde_loss_list, physics_loss_list = [], []
    
    # Performance metrics tracking
    val_mse_list, test_mse_list, test_r2_list = [], [], []
    
    # Training logs
    log_file = os.path.join(args.save_folder, "contrastive_training_log.txt")
    with open(log_file, 'w') as f:
        f.write("Epoch\tTotal Loss\tPINN Loss\tContrastive Loss\tPDE Loss\tPhysics Loss\n")
    
    # Adaptive contrastive weight - start lower
    contrastive_weight = args.contrastive_weight * 0.3
    wrapper.contrastive_weight = contrastive_weight
    
    # Best validation metrics tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    # Dynamic temperature adjustment
    base_temperature = args.temperature    
    for epoch in range(args.epochs):
        wrapper.train()
        total_loss_epoch, pinn_loss_epoch, contrastive_loss_epoch = [], [], []
        pde_loss_epoch, physics_loss_epoch = [], []
        wrapper.momentum_encoder.update_momentum(epoch, args.epochs)
        
        
        # Dynamic temperature schedule - start high, gradually decrease
        base_temperature = args.temperature * 1.5
        current_temperature = base_temperature * (1.0 + 0.5 * (1.0 - min(1.0, epoch / (0.7 * args.epochs))))
        wrapper.temperature = current_temperature
        
        # Log the dynamic hyperparameters
        wandb.log({
            "hyperparams/momentum": wrapper.momentum_encoder.momentum,
            "hyperparams/temperature": current_temperature,
            "hyperparams/contrastive_weight": wrapper.contrastive_weight,
            "epoch": epoch
        })
                
        
        # Reset iterators for each epoch
        orig_iter = iter(orig_loader['train'])
        aug_iter = iter(aug_loader['train'])
        
        # Determine the minimum number of batches and use all original data
        n_batches = len(orig_loader['train'])
        
        for step in range(n_batches):
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
                print(f"[Epoch {epoch+1}/{args.epochs}][Step {step+1}/{n_batches}] ")
                       #f"Loss: {total_loss.item():.4f} | PINN: {pinn_loss.item():.4f} | "
                       #f"Contrastive: {contrastive_loss.item():.4f} | PDE: {pde_loss.item():.4f}")
                
                # Log batch-level metrics to wandb
                wandb.log({
                    "batch/total_loss": total_loss.item(),
                    "batch/pinn_loss": pinn_loss.item(),
                    "batch/contrastive_loss": contrastive_loss.item(),
                    "batch/pde_loss": pde_loss.item(),
                    "batch/physics_loss": physics_loss.item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr']
                })
        
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
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch+1,
            "train/total_loss": avg_total,
            "train/pinn_loss": avg_pinn,
            "train/contrastive_loss": avg_contrastive,
            "train/pde_loss": avg_pde, 
            "train/physics_loss": avg_phys,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Create and log detailed visualization every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            visualize_training_progress(
                train_loss_list, pinn_loss_list, contrastive_loss_list, 
                pde_loss_list, physics_loss_list, epoch+1, args.save_folder
            )

        # Evaluate on validation set every 10 epochs instead of 5
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            wrapper.pinn.eval()
            with torch.no_grad():
                val_loss = wrapper.pinn.Valid(orig_loader['valid'])
                print(f"[Validation] Epoch {epoch+1}: MSE = {val_loss:.6f}")
            
                
                # Track validation MSE
                val_mse_list.append(val_loss)
            
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
                patience_counter = 0
                torch.save(checkpoint, os.path.join(args.save_folder, 'best_model.pth'))
                print(f"New best model saved at epoch {epoch+1} with validation MSE: {val_loss:.6f}")
                
                # Test on test set with best model
                wrapper.pinn.eval()
                true_label, pred_label = wrapper.pinn.Test(orig_loader['test'])
                from utils.util import eval_metrix
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                r2 = r2_score(true_label, pred_label)
                print(f"[Test] MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}, R²: {r2:.4f}")
                
                # Log test metrics
                wandb.log({
                    "test/mse": MSE,
                    "test/mae": MAE,
                    "test/mape": MAPE,
                    "test/rmse": RMSE,
                    "test/r2": r2,
                    "epoch": epoch+1
                })
                
                # Track test metrics
                test_mse_list.append(MSE)
                test_r2_list.append(r2)
                
                # Save predictions for analysis
                np.save(os.path.join(args.save_folder, 'true_label.npy'), true_label)
                np.save(os.path.join(args.save_folder, 'pred_label.npy'), pred_label)
                
                # Generate scatter plot
                scatter_path = os.path.join(args.save_folder, f'prediction_scatter_epoch_{epoch+1}.png')
                create_prediction_scatter(true_label, pred_label, scatter_path)
                
                # Log prediction scatter plot to wandb
                wandb.log({
                    "predictions/scatter_plot": wandb.Image(scatter_path),
                    "epoch": epoch+1
                })
                
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop // 3:  # Adjusted for more frequent validation
                    print(f"Early stopping triggered after {patience_counter} validations without improvement")
                    break
    
    # Save the final model
    torch.save({
        'epoch': args.epochs,
        'pinn_state_dict': wrapper.pinn.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(args.save_folder, 'final_model.pth'))
    
    # Plot validation and test metrics
    plot_validation_metrics(val_mse_list, test_mse_list, test_r2_list, args.save_folder)
    
    # Feature correlation visualization (once at end of training)
    try:
        # Create feature correlation matrix
        feature_names = ['Voltage', 'Current', 'Temperature', 'Current_load', 
                        'Voltage_load', 'SoC', 'Resistance', 'Capacity', 'Time']
        
        # Get a batch of data for correlation analysis
        x_batch = next(iter(orig_loader['test']))[0].cpu().numpy()
        
        # Calculate actual correlation matrix from data
        corr_matrix = np.corrcoef(x_batch.T)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=feature_names, yticklabels=feature_names)
        plt.title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        corr_path = os.path.join(args.save_folder, 'feature_correlation.png')
        plt.savefig(corr_path, dpi=300)
        plt.close()
        
        # Log to wandb
        wandb.log({"feature_analysis/correlation_matrix": wandb.Image(corr_path)})
    except Exception as e:
        print(f"Could not generate feature correlation matrix: {e}")
    
    # Load best model and generate final visualization
    try:
        checkpoint = torch.load(os.path.join(args.save_folder, 'best_model.pth'))
        wrapper.pinn.load_state_dict(checkpoint['pinn_state_dict'])
        wrapper.pinn.eval()
        true_label, pred_label = wrapper.pinn.Test(orig_loader['test'])
        
        # Create final prediction scatter
        final_scatter_path = os.path.join(args.save_folder, 'final_prediction_scatter.png')
        create_prediction_scatter(true_label, pred_label, final_scatter_path)
        
        # Log final metrics and visualization to wandb
        wandb.log({
            "final/prediction_scatter": wandb.Image(final_scatter_path),
            "final/best_epoch": best_epoch,
            "final/best_val_mse": best_val_loss
        })
        
        # Create comprehensive final report
        create_final_report(true_label, pred_label, train_loss_list, pinn_loss_list, 
                           contrastive_loss_list, best_epoch, best_val_loss, args.save_folder)
    except Exception as e:
        print(f"Could not load best model for final visualization: {e}")
    
    print(f"Training completed. Best model at epoch {best_epoch} with validation MSE: {best_val_loss:.6f}")
    return best_val_loss


def create_final_report(true_label, pred_label, train_loss, pinn_loss, contrastive_loss, 
                       best_epoch, best_val_mse, save_folder):
    """
    Create a comprehensive final report with all metrics and visualizations
    """
    plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
    
    # 1. Training Loss Overview
    ax1 = plt.subplot(gs[0, 0])
    epochs = range(1, len(train_loss) + 1)
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, pinn_loss, 'r-', linewidth=2, label='PINN Loss')
    ax1.plot(epochs, contrastive_loss, 'g-', linewidth=2, label='Contrastive Loss')
    ax1.axvline(x=best_epoch, color='k', linestyle='--', alpha=0.7, 
               label=f'Best Model (Epoch {best_epoch})')
    ax1.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Prediction Scatter Plot
    ax2 = plt.subplot(gs[0, 1])
    mse = mean_squared_error(true_label, pred_label)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_label, pred_label)
    r2 = r2_score(true_label, pred_label)
    
    # Create enhanced scatter plot
    cmap = plt.cm.viridis
    sc = ax2.scatter(true_label, pred_label, c=np.abs(true_label-pred_label), 
                    cmap=cmap, alpha=0.7, s=30)
    ax2.plot([true_label.min(), true_label.max()], 
             [true_label.min(), true_label.max()], 'r--', linewidth=2)
    
    # Add colorbar for error magnitude
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Absolute Error', fontsize=10)
    
    # Add metrics as text
    metrics_text = f"MSE: {mse:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}"
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title('Prediction Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('True SoH', fontsize=12)
    ax2.set_ylabel('Predicted SoH', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Error Distribution
    ax3 = plt.subplot(gs[1, 0])
    residuals = pred_label - true_label
    sns.histplot(residuals, kde=True, ax=ax3, bins=30, 
                color='darkblue', alpha=0.6)
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Prediction Error', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Add distribution statistics
    dist_text = (f"Mean: {np.mean(residuals):.6f}\n"
                f"Std Dev: {np.std(residuals):.6f}\n"
                f"Min: {np.min(residuals):.6f}\n"
                f"Max: {np.max(residuals):.6f}\n"
                f"Skewness: {stats.skew(residuals.flatten()):.6f}")
    ax3.text(0.95, 0.95, dist_text, transform=ax3.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Cumulative Error Analysis
    ax4 = plt.subplot(gs[1, 1])
    sorted_errors = np.sort(np.abs(residuals).flatten())
    y_values = np.arange(len(sorted_errors)) / float(len(sorted_errors))
    ax4.plot(sorted_errors, y_values, 'b-', linewidth=2)
    ax4.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Absolute Error', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add key percentiles
    percentiles = [50, 80, 90, 95, 99]
    percentile_values = np.percentile(sorted_errors, percentiles)
    for p, v in zip(percentiles, percentile_values):
        ax4.axvline(x=v, color='r', linestyle='--', alpha=0.3)
        ax4.text(v, 0.1, f'{p}%: {v:.4f}', rotation=90, alpha=0.8)
    
    # 5. Time-Based Analysis (If available)
    ax5 = plt.subplot(gs[2, 0])
    # This assumes your data has some time-based order
    # If not, this will just show errors by index
    ax5.plot(range(len(residuals)), residuals, 'o-', alpha=0.4, markersize=3)
    ax5.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax5.set_title('Error by Sample Index', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Sample Index', fontsize=12)
    ax5.set_ylabel('Prediction Error', fontsize=12)
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # Add moving average
    window_size = max(1, len(residuals) // 20)
    if len(residuals) > window_size:
        moving_avg = np.convolve(residuals.flatten(), 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        ax5.plot(range(window_size-1, len(residuals)), 
                moving_avg, 'r-', linewidth=2, 
                label=f'Moving Avg (window={window_size})')
        ax5.legend()
    
    # 6. Precision-Recall Analysis
    ax6 = plt.subplot(gs[2, 1])
    # Create binary classification scenario based on threshold
    # e.g., "is SoH below 80%?"
    threshold = np.mean(true_label)  # Customize this threshold
    
    true_binary = true_label < threshold
    pred_binary = pred_label < threshold
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_binary, pred_binary)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax6)
    ax6.set_title(f'Confusion Matrix (Threshold={threshold:.1f})', 
                 fontsize=14, fontweight='bold')
    ax6.set_xlabel('Predicted', fontsize=12)
    ax6.set_ylabel('True', fontsize=12)
    ax6.set_xticklabels(['≥ Threshold', '< Threshold'])
    ax6.set_yticklabels(['≥ Threshold', '< Threshold'])
    
    # Add classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)
    
    metrics_text = (f"Accuracy: {accuracy:.4f}\n"
                   f"Precision: {precision:.4f}\n"
                   f"Recall: {recall:.4f}\n"
                   f"F1 Score: {f1:.4f}")
    ax6.text(0.95, 0.05, metrics_text, transform=ax6.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 7. Error by Value Range
    ax7 = plt.subplot(gs[3, 0])
    # Create bins based on true values
    bins = 10
    bin_edges = np.linspace(np.min(true_label), np.max(true_label), bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Compute average error and std dev in each bin
    bin_errors = []
    bin_stds = []
    bin_counts = []
    
    for i in range(bins):
        mask = (true_label >= bin_edges[i]) & (true_label < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_errors.append(np.mean(np.abs(residuals[mask])))
            bin_stds.append(np.std(residuals[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_errors.append(0)
            bin_stds.append(0)
            bin_counts.append(0)
    
    # Bar chart with error bars
    ax7.bar(bin_centers, bin_errors, width=bin_widths*0.8, 
           yerr=bin_stds, alpha=0.6, capsize=5)
    
    # Add count labels
    for i, (x, y, count) in enumerate(zip(bin_centers, bin_errors, bin_counts)):
        ax7.text(x, y + bin_stds[i] + 0.01, str(count), 
                ha='center', va='bottom', fontsize=8)
    
    ax7.set_title('Error by Value Range', fontsize=14, fontweight='bold')
    ax7.set_xlabel('True SoH Range', fontsize=12)
    ax7.set_ylabel('Mean Absolute Error', fontsize=12)
    ax7.grid(True, linestyle='--', alpha=0.7)
    
    # 8. Summary Table
    ax8 = plt.subplot(gs[3, 1])
    ax8.axis('off')
    
    # Create summary text
    summary = (
        "MODEL PERFORMANCE SUMMARY\n"
        "=======================\n\n"
        f"Best Epoch: {best_epoch}\n"
        f"Validation MSE: {best_val_mse:.6f}\n\n"
        "TEST METRICS:\n"
        f"MSE: {mse:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"MAE: {mae:.6f}\n"
        f"R²: {r2:.6f}\n\n"
        "ERROR DISTRIBUTION:\n"
        f"Mean Error: {np.mean(residuals):.6f}\n"
        f"Std Dev: {np.std(residuals):.6f}\n"
        f"Min Error: {np.min(residuals):.6f}\n"
        f"Max Error: {np.max(residuals):.6f}\n\n"
        "PERCENTILES OF ABSOLUTE ERROR:\n"
        f"50th: {np.percentile(np.abs(residuals), 50):.6f}\n"
        f"75th: {np.percentile(np.abs(residuals), 75):.6f}\n"
        f"90th: {np.percentile(np.abs(residuals), 90):.6f}\n"
        f"95th: {np.percentile(np.abs(residuals), 95):.6f}\n"
        f"99th: {np.percentile(np.abs(residuals), 99):.6f}\n"
    )
    
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    report_path = os.path.join(save_folder, 'final_model_report.png')
    plt.savefig(report_path, dpi=300)
    plt.close()
    
    # Log to wandb
    wandb.log({
        "final_report": wandb.Image(report_path)
    })
    
    # Save as PDF for paper
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = os.path.join(save_folder, 'model_performance_report.pdf')
        with PdfPages(pdf_path) as pdf:
            plt.figure(figsize=(18, 24))
            plt.figtext(0.5, 0.98, "Battery SoH Prediction with Physics-Informed Contrastive Learning", 
                      ha='center', fontsize=18, fontweight='bold')
            plt.figtext(0.5, 0.96, f"Model Performance Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}", 
                      ha='center', fontsize=14)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # Recreate the report figure for PDF
            plt.figure(figsize=(18, 24))
            gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
            # (repeat the plotting code from above)
            # ...
            pdf.savefig()
            plt.close()
    except Exception as e:
        print(f"Could not save PDF report: {e}")

def plot_validation_metrics(val_mse, test_mse, test_r2, save_folder):
    """
    Plot validation and test metrics
    """
    plt.figure(figsize=(12, 8))
    
    # Create indices list for x-axis (assuming validation every 3 epochs)
    val_epochs = [i*3 + 1 for i in range(len(val_mse))]
    
    # Create two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot validation and test MSE on first y-axis
    ax1.plot(val_epochs, val_mse, 'b-', marker='o', label='Validation MSE')
    if test_mse:
        test_epochs = val_epochs[:len(test_mse)]
        ax1.plot(test_epochs, test_mse, 'g-', marker='s', label='Test MSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error (MSE)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot test R² on second y-axis
    if test_r2:
        ax2.plot(test_epochs, test_r2, 'r-', marker='d', label='Test R²')
        ax2.set_ylabel('R² Score', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
    # Add a single legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Validation and Test Metrics')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, 'validation_metrics.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Log to wandb
    wandb.log({
        "metrics/validation_plot": wandb.Image(save_path)
    })

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
    
    # Create custom colormap for error visualization
    colors = plt.cm.viridis(np.linspace(0, 1, 256))
    custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors)
    
    # Create scatter plot with enhanced styling and color by error magnitude
    sc = plt.scatter(true_label, pred_label, 
                    c=np.abs(true_label-pred_label),  # Color by absolute error
                    cmap=custom_cmap,
                    alpha=0.7, s=50, 
                    edgecolors='w', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute Error', fontsize=10)
    
    # Add perfect prediction line
    min_val = min(true_label.min(), pred_label.min())
    max_val = max(true_label.max(), pred_label.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Add metrics text
    plt.annotate(f'MSE = {mse:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.xlabel('True SoH', fontsize=12)
    plt.ylabel('Predicted SoH', fontsize=12)
    plt.title('Battery SoH: True vs Predicted Values', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
     #print(f"Enhanced prediction scatter plot saved to {save_path}")

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--csv_file_augmented', type=str, required=True, help='Path to augmented CSV file')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--early_stop', type=int, default=30, help='early stop patience')
    parser.add_argument('--lr', type=float, default=0.0005, help='base lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='learning rate for F network')
    parser.add_argument('--save_folder', type=str, default='results', help='save folder')
    parser.add_argument('--alpha', type=float, default=1.0, help='PDE loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='physics constraint weight')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='contrastive loss weight')
    parser.add_argument('--pinn_weight', type=float, default=1.0, help='PINN loss weight')
    parser.add_argument('--temperature', type=float, default=0.1, help='contrastive temperature')
    parser.add_argument('--log_dir', type=str, default='training_log.txt', help='log dir')
    parser.add_argument('--F_layers_num', type=int, default=8, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=128, help='the hidden dim of F')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--final_lr', type=float, default=1e-5, help='final lr')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.999, help='momentum for encoder')
    parser.add_argument('--projection_dim', type=int, default=128, help='projection dimension for contrastive learning')
    parser.add_argument('--queue_size', type=int, default=4096, help='queue size for momentum contrast')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--wandb_project', type=str, default='battery-picle-unit-dataset', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (username or team name)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
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
    
    # Initialize wandb
    run_name = args.wandb_run_name if args.wandb_run_name else f"PICLE_B0005_dataset-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    
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
     #print("Loading original dataset...")
    orig_loader = load_data(args)
    print(f"Original dataset loaded: {len(orig_loader['train'])} training batches")
    
    # print("Loading augmented dataset...")
    aug_loader = load_augmented_data(args)
    print(f"Augmented dataset loaded: {len(aug_loader['train'])} training batches")

    # Initialize PINN
    print("Initializing PINN model...")
    pinn = PINN(args)
    
    # Initialize contrastive wrapper with projection head
    print("Setting up contrastive learning ...")
    wrapper = ContrastivePINNWrapper(
        pinn,
        temperature=args.temperature * 1.5,  # Higher temperature for more stable gradients
        contrastive_weight=args.contrastive_weight * 0.3,  # Start with much lower contrastive weight
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
    # print(f"PINN model has {pinn_params} trainable parameters")
    # print(f"Full contrastive wrapper has {wrapper_params} trainable parameters")
    
    # Log model architecture to wandb
    wandb.config.update({
        "pinn_params": pinn_params,
        "wrapper_params": wrapper_params,
        "device": str(device)
    })

    # Train with contrastive + PINN loss
    print("Starting training...")
    train_contrastive_pinn(wrapper, orig_loader, aug_loader, args)
    
    # Close wandb run
    wandb.finish()

if __name__ == '__main__':
    main()