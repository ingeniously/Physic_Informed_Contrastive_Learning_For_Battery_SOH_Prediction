import torch
import torch.nn as nn
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util import AverageMeter, get_logger, eval_metrix
from Model.other_models import MLP, CNN
from Model.Model import LR_Scheduler
from dataloader.dataloader import SimpleLoader  # Changed to use the same data loader as main.py
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap


class Trainer():
    def __init__(self, model, train_loader, valid_loader, test_loader, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = args.model
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.save_dir = os.path.join(args.save_folder, f"{args.model}_results")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(self.save_dir, args.log_dir))

        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.warmup_lr,
            weight_decay=1e-4,
            eps=1e-8
        )
        self.scheduler = LR_Scheduler(
            optimizer=self.optimizer,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr
        )
        
        # History tracking for visualization
        self.train_loss_history = []
        self.val_loss_history = []
        self.epochs_history = []
        self.lr_history = []

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def train_one_epoch(self, epoch):
        self.model.train()
        self.loss_meter.reset()
        for (x1, _, y1, _) in self.train_loader:
            x1 = x1.to(self.device)
            y1 = y1.to(self.device)

            y_pred = self.model(x1)
            loss = self.loss_func(y_pred, y1)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            self.loss_meter.update(loss.item())
            
        # Log to wandb
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({
                f"baseline/{self.model_name}/train_loss": self.loss_meter.avg,
                f"baseline/{self.model_name}/lr": self.optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
            
        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch, self.loss_meter.avg)
        self.logger.info(info)
        
        # Store history
        self.train_loss_history.append(self.loss_meter.avg)
        self.epochs_history.append(epoch)
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        
        return self.loss_meter.avg

    def valid(self, epoch):
        self.model.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1, _, y1, _) in self.valid_loader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)

                y_pred = self.model(x1)
                loss = self.loss_func(y_pred, y1)
                self.loss_meter.update(loss.item())
                
        # Log to wandb
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({
                f"baseline/{self.model_name}/val_loss": self.loss_meter.avg,
                "epoch": epoch
            })
            
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch, self.loss_meter.avg)
        self.logger.info(info)
        
        # Store history
        self.val_loss_history.append(self.loss_meter.avg)
        
        return self.loss_meter.avg

    def test(self):
        self.model.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1, _, y1, _) in self.test_loader:
                x1 = x1.to(self.device)
                y_pred = self.model(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
                
        true_label = np.concatenate(true_label, axis=0)
        pred_label = np.concatenate(pred_label, axis=0)
        
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir, 'true_label.npy'), true_label)
            np.save(os.path.join(self.save_dir, 'pred_label.npy'), pred_label)
            
            # Create visualization
            save_path = os.path.join(self.save_dir, f'{self.model_name}_predictions.png')
            mse, rmse, mae, r2 = self.visualize_predictions(true_label, pred_label, save_path)
            
            # Log test metrics to wandb
            if hasattr(wandb, 'run') and wandb.run is not None:
                wandb.log({
                    f"baseline/{self.model_name}/test_mse": mse,
                    f"baseline/{self.model_name}/test_rmse": rmse,
                    f"baseline/{self.model_name}/test_mae": mae,
                    f"baseline/{self.model_name}/test_r2": r2
                })
                
        return true_label, pred_label
        
    def visualize_predictions(self, true_label, pred_label, save_path):
        """
        Create scatter plot of true vs predicted values with additional metrics.
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics for annotation
        mse = mean_squared_error(true_label, pred_label)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_label, pred_label)
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
        plt.title(f'{self.model_name} Model: True vs Predicted Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Log to wandb
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({f"baseline/{self.model_name}/prediction_scatter": wandb.Image(save_path)})
        
        return mse, rmse, mae, r2
        
    def visualize_training_history(self):
        """Create visualization of training history"""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs_history, self.train_loss_history, 'b-', label='Train Loss')
        if self.val_loss_history:
            # Create epochs list for validation (assuming validation every epoch)
            val_epochs = self.epochs_history[:len(self.val_loss_history)]
            plt.plot(val_epochs, self.val_loss_history, 'r-', label='Validation Loss')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} Training History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs_history, self.lr_history, 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{self.model_name}_training_history.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Log to wandb
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({f"baseline/{self.model_name}/training_history": wandb.Image(save_path)})

    def train(self):
        min_loss = float('inf')
        early_stop = 0
        patience = getattr(self.args, 'early_stop', 10)
        
        for epoch in range(1, self.epochs+1):
            early_stop += 1
            train_loss = self.train_one_epoch(epoch)
            current_lr = self.scheduler.step()
            valid_loss = self.valid(epoch)
            
            if valid_loss < min_loss:
                min_loss = valid_loss
                
                if self.test_loader is not None:
                    true_label, pred_label = self.test()
                    
                early_stop = 0
                
                # Save model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': min_loss,
                }, os.path.join(self.save_dir, f'{self.model_name}_best_model.pth'))
                
            if early_stop > patience:
                self.logger.info(f"Early stopping triggered after {early_stop} epochs without improvement")
                break
                
        # Create final visualizations
        self.visualize_training_history()
        
        self.clear_logger()


def load_model(args):
    if args.model == 'MLP':
        model = MLP()
    elif args.model == 'CNN':
        model = CNN()
    return model


def load_data(args):
    data = SimpleLoader(
        csv_path=args.csv_file,
        batch_size=args.batch_size,
        normalization=True,
        normalization_method=args.normalization_method
    )
    loaders = data.load()
    return loaders


def get_args():
    parser = argparse.ArgumentParser('Baseline Models Parameters')
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'CNN'])
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=30, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-5, help='final lr')

    parser.add_argument('--save_folder', type=str, default='./baseline_results')
    parser.add_argument('--log_dir', type=str, default='logging.txt')
    
    parser.add_argument('--wandb_project', type=str, default='battery-picle-small-dataset')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    return args


def main():
    args = get_args()
    
    # Initialize wandb
    import pandas as pd
    run_name = args.wandb_run_name if args.wandb_run_name else f"Baseline-{args.model}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    
    # Train models
    model = load_model(args)
    data_loader = load_data(args)
    
    # Count parameters for logging
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model} model has {param_count} trainable parameters")
    
    # Log model architecture to wandb
    wandb.config.update({
        f"{args.model}_params": param_count,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    })
    
    trainer = Trainer(model, data_loader['train'], data_loader['valid'], data_loader['test'], args)
    trainer.train()
    
    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    main()