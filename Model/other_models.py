import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import seaborn as sns
from Model.Model import MLP as Encoder
from Model.Model import Predictor


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Modified to match 9 input features
        self.encoder = Encoder(input_dim=9, output_dim=32, layers_num=3, hidden_dim=60, droupout=0.2)
        self.predictor = Predictor(input_dim=64)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x
        
    def predict(self, x):
        """Added for compatibility with PINN interface"""
        return self.forward(x)
        
    def Test(self, testloader):
        """Added for compatibility with PINN interface"""
        device = next(self.parameters()).device
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                u1 = self.forward(x1)
                true_label.append(y1.cpu().numpy())
                pred_label.append(u1.cpu().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label
        
    def Valid(self, validloader):
        """Added for compatibility with PINN interface"""
        device = next(self.parameters()).device
        self.eval()
        loss_func = nn.MSELoss()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for x1, _, y1, _ in validloader:
                x1 = x1.to(device)
                u1 = self.forward(x1)
                true_label.append(y1.cpu().numpy())
                pred_label.append(u1.cpu().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()
        
    def visualize_predictions(self, true_label, pred_label, save_path):
        """Added visualization capability similar to PINN model"""
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics for annotation
        mse = mean_squared_error(true_label, pred_label)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_label, pred_label)
        r2 = r2_score(true_label, pred_label)
        
        # Create scatter plot
        plt.scatter(true_label, pred_label, alpha=0.6, s=30, color='darkblue')
        
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
        plt.title('MLP Model: True vs Predicted Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({"baseline/mlp_prediction": wandb.Image(save_path)})
        
        return mse, rmse, mae, r2


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Modified to work with 9 input features
        self.layer1 = ResBlock(input_channel=1, output_channel=8, stride=1)  # N,8,9
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2)  # N,16,5
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=1)  # N,24,5
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1)  # N,16,5
        self.layer5 = ResBlock(input_channel=16, output_channel=8, stride=1)  # N,8,5
        self.layer6 = nn.Linear(8*5, 1)

    def forward(self, x):
        N, L = x.shape[0], x.shape[1]
        x = x.view(N, 1, L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out.view(N, -1))
        return out.view(N, 1)
        
    def predict(self, x):
        """Added for compatibility with PINN interface"""
        return self.forward(x)
        
    def Test(self, testloader):
        """Added for compatibility with PINN interface"""
        device = next(self.parameters()).device
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                u1 = self.forward(x1)
                true_label.append(y1.cpu().numpy())
                pred_label.append(u1.cpu().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label
        
    def Valid(self, validloader):
        """Added for compatibility with PINN interface"""
        device = next(self.parameters()).device
        self.eval()
        loss_func = nn.MSELoss()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for x1, _, y1, _ in validloader:
                x1 = x1.to(device)
                u1 = self.forward(x1)
                true_label.append(y1.cpu().numpy())
                pred_label.append(u1.cpu().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()
        
    def visualize_predictions(self, true_label, pred_label, save_path):
        """Added visualization capability similar to PINN model"""
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics for annotation
        mse = mean_squared_error(true_label, pred_label)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_label, pred_label)
        r2 = r2_score(true_label, pred_label)
        
        # Create scatter plot
        plt.scatter(true_label, pred_label, alpha=0.6, s=30, color='darkred')
        
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
        plt.title('CNN Model: True vs Predicted Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({"baseline/cnn_prediction": wandb.Image(save_path)})
        
        return mse, rmse, mae, r2


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


if __name__ == '__main__':
    x = torch.randn(10, 9)  # Modified to match 9 input features
    y1 = MLP()(x)
    y2 = CNN()(x)
    count_parameters(MLP())
    count_parameters(CNN())