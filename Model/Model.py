import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import grad
from dataloader import dataloader
from utils.util import AverageMeter, get_logger, eval_metrix
import os
import wandb  # Import wandb for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Improved activation: you can try nn.SiLU() or nn.Tanh() for PINNs, but let's keep Sin for now.
class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

# Increased hidden_dim and layers_num for better expressiveness
class MLP(nn.Module):
    def __init__(self, input_dim=9, output_dim=1, layers_num=4, hidden_dim=128, droupout=0.2):
        super(MLP, self).__init__()
        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim

        # If this is the dynamical_F network (input_dim=20)
        if input_dim == 20:
            # Enhanced network for F with residual connections
            self.net = nn.ModuleList()
            for i in range(layers_num):
                if i == 0:
                    self.net.append(nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU()
                    ))
                elif i == layers_num - 1:
                    self.net.append(nn.Linear(hidden_dim, 1))
                else:
                    self.net.append(nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(p=droupout)
                    ))
        else:
            # Enhanced separate networks for time and features
            self.time_net = nn.Sequential(
                nn.Linear(1, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

            # Deeper feature network with residual connections
            self.feature_embed = nn.Linear(input_dim - 1, hidden_dim)
            self.feature_layers = nn.ModuleList()
            for _ in range(2):  # Two residual blocks
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(p=droupout),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
            self.feature_out = nn.Linear(hidden_dim, output_dim)
        
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.input_dim == 20:
            # Apply network with residual connections
            h = x
            for i, layer in enumerate(self.net):
                if i == len(self.net) - 1:  # Last layer (no residual)
                    h = layer(h)
                else:
                    h = h + layer(h) if h.shape == layer(h).shape else layer(h)
            return h
        
        # Process time and features separately with enhanced networks
        time = x[:, -1].unsqueeze(-1)
        features = x[:, :-1]
        
        time_embedding = self.time_net(time)
        
        # Process features with residual connections
        h = self.feature_embed(features)
        for layer in self.feature_layers:
            h = h + layer(h)  # Residual connection
        feature_embedding = self.feature_out(h)
        
        return torch.cat([feature_embedding, time_embedding], dim=1)


class Predictor(nn.Module):
    def __init__(self, input_dim=65):
        super(Predictor, self).__init__()
        hidden_dim = 64
        
        # Enhanced predictor with residual connections
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
        ])
        
        # Output projection with multiple branches
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Initial projection
        h = self.input_norm(x)
        h = self.input_proj(h)
        
        # Apply residual blocks
        for block in self.res_blocks:
            h = h + block(h)  # Residual connection
        
        # Output projection
        return self.output(h)

class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        # Keep the encoder the same
        self.encoder = MLP(input_dim=9, output_dim=32, layers_num=4, hidden_dim=64, droupout=0.2)
        
        # Replace the predictor with an improved version
        hidden_dim = 64
        self.predictor = nn.Sequential(
            nn.LayerNorm(64),  # 64 = encoder output (32) + time embedding (32)
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_()





    def forward(self, x, time=None):
        if time is None:
          time = x[:, -1].unsqueeze(-1)
        # Make sure time is the *same tensor* as used for grad
        # Get encoded features (includes time processing)
        encoded = self.encoder(x)
        
        # Pass through predictor
        out = self.predictor(encoded)
        
        # Add explicit time dependence with stronger correlation
        out = out + 0.2 * torch.sin(time * np.pi) + 0.1 * torch.cos(time * 2 * np.pi)
        return out

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr

class BatteryPDE:
    """
    Battery physics differential equations.
    """
    @staticmethod
    def sei_growth_rate(soh, time):
        """SEI growth rate equation"""
        # SEI growth follows sqrt(t) law
        # d(SoH)/dt = -k/2/sqrt(t)
        k = 0.01  # SEI growth constant
        t_small = time + 1e-6  # Prevent div by zero
        return -k / (2 * torch.sqrt(t_small))
    
    @staticmethod
    def calendar_aging_rate(soh, time, temp):
        """Calendar aging rate equation"""
        # Calendar aging with Arrhenius temperature dependence
        # d(SoH)/dt = -A * exp(-Ea/RT) * sqrt(t)
        A = 0.005  # Pre-exponential factor
        Ea = 0.2   # Activation energy (normalized)
        # Assume reference temperature = 0 for normalized temp
        return -A * torch.exp(Ea * temp) / torch.sqrt(time + 1e-6)
    
    @staticmethod
    def cycling_degradation_rate(soh, soc, current):
        """Cycling degradation rate equation"""
        # Higher degradation at extreme SoC and higher currents
        # d(SoH)/dt = -B * f(SoC) * |I|
        B = 0.01  # Cycling factor
        
        # SoC stress function - U-shaped (higher at extremes)
        soc_norm = soc / 100.0  # Convert to [0,1]
        soc_stress = 4 * (soc_norm - 0.5)**2 + 0.2
        
        # Current impact (absolute value)
        current_impact = torch.abs(current)
        
        return -B * soc_stress * current_impact

class PINN(nn.Module):
    def __init__(self, args):
        super(PINN, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        # Updated: Solution_u and dynamical_F input_dim to match 9-feature input
        self.solution_u = Solution_u().to(device)
        # For dynamical_F: input is [x_t, u, u_x, u_t], with:
        # x_t (9 features), u (1), u_x (9), u_t (1) => total 20 features
        # Increased hidden_dim and layers_num for better expressiveness
        self.dynamical_F = MLP(input_dim=20, output_dim=1,
                               layers_num=max(4, args.F_layers_num),
                               hidden_dim=max(128, args.F_hidden_dim),
                               droupout=0.2).to(device)

        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr, weight_decay=1e-5)  # L2 regularization
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F, weight_decay=1e-5)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        self.best_model = None
        # Stronger physics-informed loss weight
        self.alpha = max(10, self.args.alpha)  # Strongly enforce PDE loss
        self.beta = self.args.beta
        
        # Physical constraints weights
        self.gamma = 5.0  # Weight for physical boundary constraints
        
        # Battery PDE models
        self.battery_pde = BatteryPDE()
        
        # For visualization
        self.train_history = {
            'data_loss': [],
            'pde_loss': [],
            'physics_loss': [],
            'boundary_loss': [],
            'total_loss': [],
            'epochs': [],
            'lr': []
        }
        
        # Create model architecture visualization for wandb
        if hasattr(wandb, 'run') and wandb.run is not None:
            try:
                from torchviz import make_dot
                x_sample = torch.randn(1, 9).to(device)
                y_pred, f_pred = self(x_sample)
                model_graph = make_dot(y_pred, params=dict(self.named_parameters()))
                model_graph.render("model_architecture", format="png")
                wandb.log({"model_architecture": wandb.Image("model_architecture.png")})
            except:
                print("Could not create model visualization for wandb")

    def _save_args(self):
        if self.args.log_dir is not None:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self, xt):
        t = xt[:, -1].reshape(-1, 1)
        return self.solution_u(xt, t)

    def visualize_prediction_vs_true(self, true, pred, epoch, save_folder=None):
        """
        Create and save scatter plot of predictions vs true values
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        
        # Create scatter plot with colormap indicating density
        counts, xbins, ybins = np.histogram2d(true.flatten(), pred.flatten(), bins=50)
        sns.scatterplot(x=true.flatten(), y=pred.flatten(), alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Add metrics to plot
        plt.annotate(f"MSE: {mse:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                    verticalalignment='top')
        
        plt.xlabel("True SoH")
        plt.ylabel("Predicted SoH")
        plt.title(f"Prediction vs Ground Truth (Epoch {epoch})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_folder:
            save_path = os.path.join(save_folder, f"prediction_scatter_epoch_{epoch}.png")
            plt.savefig(save_path, dpi=300)
            
            # Log to wandb if initialized
            if hasattr(wandb, 'run') and wandb.run is not None:
                wandb.log({f"prediction/scatter_epoch_{epoch}": wandb.Image(save_path)})
                
        plt.close()
        return mse, rmse, mae, r2

    def visualize_feature_importance(self, x_data, save_folder=None):
        """
        Visualize feature importance by perturbing inputs
        """
        if not save_folder:
            return
            
        feature_names = ['Voltage', 'Current', 'Temperature', 'Current_load', 
                         'Voltage_load', 'SoC', 'Resistance', 'Capacity', 'Time']
        
        # Original predictions
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
            original_pred = self.predict(x_tensor).cpu().numpy()
            
        # Calculate importance by perturbation
        importance = []
        for i in range(x_data.shape[1]):
            # Create perturbed data - increase by 10%
            perturbed_data = x_data.copy()
            perturbed_data[:, i] *= 1.1  # 10% increase
            
            # Get predictions for perturbed data
            with torch.no_grad():
                x_perturbed = torch.tensor(perturbed_data, dtype=torch.float32).to(device)
                perturbed_pred = self.predict(x_perturbed).cpu().numpy()
                
            # Calculate change in prediction
            diff = np.mean(np.abs(perturbed_pred - original_pred))
            importance.append(diff)
                
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        importance = np.array(importance)
        # Normalize importance
        importance = importance / np.sum(importance)
        
        bars = plt.bar(feature_names, importance, color='skyblue')
        plt.xticks(rotation=45)
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importance for SoH Prediction')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.gca().text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom')
                         
       # plt.tight_layout()
       # save_path = os.path.join(save_folder, 'feature_importance.png')
       # plt.savefig(save_path, dpi=300)
        
        # Log to wandb if initialized
        #if hasattr(wandb, 'run') and wandb.run is not None:
        #    wandb.log({"analysis/feature_importance": wandb.Image(save_path)})
            
       # plt.close()

    def visualize_training_history(self, save_folder=None):
        """
        Create visualization of training history
        """
        if not save_folder or len(self.train_history['epochs']) == 0:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(self.train_history['epochs'], self.train_history['data_loss'], 'b-', label='Data Loss')
        plt.plot(self.train_history['epochs'], self.train_history['pde_loss'], 'r-', label='PDE Loss')
        plt.plot(self.train_history['epochs'], self.train_history['physics_loss'], 'g-', label='Physics Loss')
        plt.plot(self.train_history['epochs'], self.train_history['boundary_loss'], 'y-', label='Boundary Loss')
        plt.plot(self.train_history['epochs'], self.train_history['total_loss'], 'k-', label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.train_history['epochs'], self.train_history['lr'], 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(save_folder, 'training_history.png')
        plt.savefig(save_path, dpi=300)
        
        # Log to wandb if initialized
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({"training/history": wandb.Image(save_path)})
            
        plt.close()
        
        # Also create log-scale version
        plt.figure(figsize=(12, 8))
        plt.semilogy(self.train_history['epochs'], self.train_history['data_loss'], 'b-', label='Data Loss')
        plt.semilogy(self.train_history['epochs'], self.train_history['pde_loss'], 'r-', label='PDE Loss')
        plt.semilogy(self.train_history['epochs'], self.train_history['physics_loss'], 'g-', label='Physics Loss')
        plt.semilogy(self.train_history['epochs'], self.train_history['boundary_loss'], 'y-', label='Boundary Loss')
        plt.semilogy(self.train_history['epochs'], self.train_history['total_loss'], 'k-', label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss History (Log Scale)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        log_save_path = os.path.join(save_folder, 'training_history_log.png')
        plt.savefig(log_save_path, dpi=300)
        
        # Log to wandb if initialized
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({"training/history_log": wandb.Image(log_save_path)})
            
        plt.close()

    def Test(self, testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1.cpu().numpy())
                pred_label.append(u1.cpu().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        
        # Create visualization of test results
        try:
            if hasattr(wandb, 'run') and wandb.run is not None:
                mse, rmse, mae, r2 = self.visualize_prediction_vs_true(
                    true_label, pred_label, "final", 
                    save_folder=self.args.save_folder if hasattr(self.args, "save_folder") else None
                )
                wandb.log({
                    "test/mse": mse,
                    "test/rmse": rmse,
                    "test/mae": mae,
                    "test/r2": r2
                })

        except Exception as e:
            print(f"Error creating test visualizations: {e}")
                
        return true_label, pred_label

    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for x1, _, y1, _ in validloader:
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1.cpu().numpy())
                pred_label.append(u1.cpu().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        
        # Log validation metrics to wandb
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({
                "validation/mse": mse.item(),
                "validation/rmse": np.sqrt(mse.item()),
                "validation/r2": r2_score(true_label, pred_label)
            })
            
        return mse.item()
    
    def physical_boundary_loss(self, u, xt):
        """
        Enforces battery physics boundary conditions:
        1. SoH must be between 0 and 100%
        2. SoH should be monotonically decreasing over time
        3. SoH degradation rate should follow physical models
        """
        # 1. SoH range constraint: 0 <= SoH <= 100
        range_loss = torch.mean(self.relu(-u) + self.relu(u - 100))
        
        # 2. Monotonicity constraint
        t = xt[:, -1].reshape(-1, 1)
        u_t = grad(u.sum(), t, create_graph=True)[0]
        monotonicity_loss = torch.mean(self.relu(u_t))  # SoH should decrease over time (negative derivative)
        
        # 3. Physics-based degradation models
        # Extract necessary features
        temperature = xt[:, 2].reshape(-1, 1)  # Temperature
        current = xt[:, 1].reshape(-1, 1)      # Current
        soc = xt[:, 5].reshape(-1, 1)          # SoC
        
        # Combined degradation rate from multiple mechanisms
        sei_rate = self.battery_pde.sei_growth_rate(u, t)
        calendar_rate = self.battery_pde.calendar_aging_rate(u, t, temperature)
        cycling_rate = self.battery_pde.cycling_degradation_rate(u, soc, current)
        
        # Total physics-based degradation rate
        physics_rate = sei_rate + calendar_rate + cycling_rate
        
        # Loss between predicted degradation rate and physics-based rate
        physics_rate_loss = torch.mean((u_t - physics_rate)**2)
        
        # Combined physical boundary loss
        return range_loss + monotonicity_loss + physics_rate_loss

    def forward(self, xt):
        """
        xt: [batch, 9], last column is time
        """
        # Create a copy of input tensor that requires grad
        xt = xt.clone().requires_grad_(True)
        
        # Extract time feature
        t = xt[:, -1].reshape(-1, 1)
        
        # Get prediction
        u = self.solution_u(xt, t)
        
        # Now time gradients should exist due to explicit time dependence
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), xt, create_graph=True)[0]
        
        # Debug prints
        if hasattr(self, "epoch") and self.epoch % 10 == 0:
            print(f"[Epoch {self.epoch}]")
            print(f"Time values (first 5): {t[:5].cpu().detach().numpy().flatten()}")
            print(f"u_t values (first 5): {u_t[:5].cpu().detach().numpy().flatten()}")
            print(f"u values (first 5): {u[:5].cpu().detach().numpy().flatten()}")
        
        # Form input for F network
        F_input = torch.cat([xt, u, u_x, u_t], dim=1)
        F = self.dynamical_F(F_input)
        
        # Compute residual
        f = u_t - F
        
        return u, f

    def train_one_epoch(self, epoch, dataloader):
        self.epoch = epoch
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        loss4_meter = AverageMeter()
        
        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            
            # Compute outputs and losses
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)
            
            # Data loss
            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
            
            # PDE loss - strengthened with custom scaling based on epoch
            f_target = torch.zeros_like(f1).to(device)
            loss2 = 0.5 * self.loss_func(f1, f_target) + 0.5 * self.loss_func(f2, f_target)
            
            # Physics constraint
            loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).mean()
            
            # Physical boundary loss
            loss4 = 0.5 * self.physical_boundary_loss(u1, x1) + 0.5 * self.physical_boundary_loss(u2, x2)
            
            # Total loss - with epoch-dependent scaling of physics terms
            # Gradually increase importance of physics terms
            epoch_factor = min(1.0, epoch / 50.0)  # Ramp up over 50 epochs
            loss = loss1 + \
                   self.alpha * loss2 * (1.0 + epoch_factor) + \
                   self.beta * loss3 * (1.0 + epoch_factor) + \
                   self.gamma * loss4 * epoch_factor
            
            # Optimization step
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer1.step()
            self.optimizer2.step()
            
            # Update meters
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            loss4_meter.update(loss4.item())
            
            if (iter+1) % 50 == 0:
                print(f"[epoch:{epoch} iter:{iter+1}] "
                    f"data loss:{loss1:.6f}, PDE loss:{loss2:.6f}, "
                    f"physics loss:{loss3:.6f}, boundary loss:{loss4:.6f}")
                
                # Log batch-level metrics to wandb
                if hasattr(wandb, 'run') and wandb.run is not None:
                    wandb.log({
                        "batch/data_loss": loss1.item(),
                        "batch/pde_loss": loss2.item(),
                        "batch/physics_loss": loss3.item(),
                        "batch/boundary_loss": loss4.item(),
                        "batch/total_loss": loss.item(),
                        "batch": epoch * len(dataloader) + iter
                    })
        
        # Save training history
        self.train_history['data_loss'].append(loss1_meter.avg)
        self.train_history['pde_loss'].append(loss2_meter.avg)
        self.train_history['physics_loss'].append(loss3_meter.avg)
        self.train_history['boundary_loss'].append(loss4_meter.avg)
        self.train_history['total_loss'].append(loss1_meter.avg + 
                                              self.alpha * loss2_meter.avg + 
                                              self.beta * loss3_meter.avg + 
                                              self.gamma * loss4_meter.avg)
        self.train_history['epochs'].append(epoch)
        self.train_history['lr'].append(self.scheduler.get_lr())
        
        # Log epoch-level metrics to wandb
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({
                "train/data_loss": loss1_meter.avg,
                "train/pde_loss": loss2_meter.avg,
                "train/physics_loss": loss3_meter.avg,
                "train/boundary_loss": loss4_meter.avg,
                "train/total_loss": loss1_meter.avg + self.alpha*loss2_meter.avg + 
                                   self.beta*loss3_meter.avg + self.gamma*loss4_meter.avg,
                "train/learning_rate": self.scheduler.get_lr(),
                "epoch": epoch
            })
            
        # Create training history visualization periodically
        if epoch % 10 == 0 or epoch == 1:
            self.visualize_training_history(
                save_folder=self.args.save_folder if hasattr(self.args, "save_folder") else None
            )
        
        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, loss4_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = float('inf')
        valid_mse = float('inf')
        early_stop = 0
        for e in range(1, self.args.epochs+1):
            early_stop += 1
            loss1, loss2, loss3, loss4 = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()
            info = f'[Train] epoch:{e}, lr:{current_lr:.6f}, total loss:{loss1+self.alpha*loss2+self.beta*loss3+self.gamma*loss4:.6f}'
            info += f' data:{loss1:.6f}, PDE:{loss2:.6f}, physics:{loss3:.6f}, boundary:{loss4:.6f}'
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = f'[Valid] epoch:{e}, MSE: {valid_mse}'
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                r2 = r2_score(true_label, pred_label)
                info = f'[Test] MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}, R²: {r2:.4f}'
                self.logger.info(info)
                
                # Create visualization for best model
                self.visualize_prediction_vs_true(
                    true_label, pred_label, e,
                    save_folder=self.args.save_folder if hasattr(self.args, "save_folder") else None
                )
                
                early_stop = 0
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                    
                    # Log best model metrics to wandb
                    if hasattr(wandb, 'run') and wandb.run is not None:
                        wandb.log({
                            "best/epoch": e,
                            "best/test_mse": MSE,
                            "best/test_mae": MAE,
                            "best/test_mape": MAPE,
                            "best/test_rmse": RMSE,
                            "best/test_r2": r2,
                            "best/valid_mse": valid_mse
                        })
                        
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = f'early stop at epoch {e}'
                self.logger.info(info)
                break
        
        # Final training history visualization
        self.visualize_training_history(
            save_folder=self.args.save_folder if hasattr(self.args, "save_folder") else None
        )
        
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))

if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser('Hyper Parameters')
        parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
        parser.add_argument('--batch_size', type=int, default=512, help='batch size')  # Smaller batch size
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
        
        # Updated scheduler parameters
        parser.add_argument('--epochs', type=int, default=300, help='epoch')  # More epochs
        parser.add_argument('--early_stop', type=int, default=30, help='early stop')  # More patience
        parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epoch')  # Longer warmup
        parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')  # Lower initial lr
        parser.add_argument('--lr', type=float, default=1e-3, help='base lr')  # Lower base lr
        parser.add_argument('--final_lr', type=float, default=1e-5, help='final lr')  # Lower final lr
        parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')  # Lower F lr
        
        # Model parameters
        parser.add_argument('--F_layers_num', type=int, default=4, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=128, help='the hidden dim of F')
        
        # Loss weights
        parser.add_argument('--alpha', type=float, default=1.0, help='PDE loss weight')
        parser.add_argument('--beta', type=float, default=0.5, help='physics constraint weight')
        
        parser.add_argument('--log_dir', type=str, default='training_log.txt', help='log dir')
        parser.add_argument('--save_folder', type=str, default='results', help='save folder')
        
        # Wandb options
        parser.add_argument('--wandb_project', type=str, default='battery-pinn', help='wandb project name')
        parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
        parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
        
        args = parser.parse_args()
        return args

    args = get_args()
    pinn = PINN(args)
    print(pinn.solution_u)
    count_parameters(pinn.solution_u)
    print(pinn.dynamical_F)