import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import wandb
from Model.Model import PINN
from contrastive_Learning.contrastive import ContrastivePINNWrapper
from Model.other_models import MLP, CNN

def load_results(base_dir, model_name):
    """Load prediction results for a model"""
    try:
        if model_name == "PICL":
            true_path = os.path.join(base_dir, "true_label.npy")
            pred_path = os.path.join(base_dir, "pred_label.npy")
        else:
            model_dir = os.path.join(base_dir, f"{model_name}_results")
            true_path = os.path.join(model_dir, "true_label.npy")
            pred_path = os.path.join(model_dir, "pred_label.npy")
            
        if os.path.exists(true_path) and os.path.exists(pred_path):
            true_label = np.load(true_path)
            pred_label = np.load(pred_path)
            return true_label, pred_label
        else:
            print(f"Results for {model_name} not found")
            return None, None
    except Exception as e:
        print(f"Error loading results for {model_name}: {e}")
        return None, None

def calculate_metrics(true_label, pred_label):
    """Calculate performance metrics"""
    if true_label is None or pred_label is None:
        return None, None, None, None
        
    mse = mean_squared_error(true_label, pred_label)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_label, pred_label)
    r2 = r2_score(true_label, pred_label)
    return mse, rmse, mae, r2

def plot_comparison(results_dir, model_results):
    """Create comparison plots for all models"""
    # 1. Bar chart of metrics
    metrics = ["MSE", "RMSE", "MAE", "1-R²"]
    models = list(model_results.keys())
    
    # Extract metrics values
    values = {
        "MSE": [model_results[model]["mse"] if model in model_results else np.nan for model in models],
        "RMSE": [model_results[model]["rmse"] if model in model_results else np.nan for model in models],
        "MAE": [model_results[model]["mae"] if model in model_results else np.nan for model in models],
        "1-R²": [1.0 - model_results[model]["r2"] if model in model_results else np.nan for model in models],
    }
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        "Model": np.repeat(models, len(metrics)),
        "Metric": np.tile(metrics, len(models)),
        "Value": np.concatenate([values[metric] for metric in metrics])
    })
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Use log scale for better visibility of differences
    ax = plt.subplot(111)
    sns.barplot(x="Metric", y="Value", hue="Model", data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_title("Model Performance Comparison (Lower is Better)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Value (log scale)", fontsize=14)
    ax.set_xlabel("Metric", fontsize=14)
    ax.legend(title="Model", fontsize=12, title_fontsize=12)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=9)
    
    plt.tight_layout()
    metrics_path = os.path.join(results_dir, "model_comparison_metrics.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    
    # 2. Scatter plot of predictions
    plt.figure(figsize=(16, 12))
    
    for i, model in enumerate(models):
        if model in model_results and model_results[model]["true"] is not None:
            plt.subplot(2, 2, i+1)
            true = model_results[model]["true"]
            pred = model_results[model]["pred"]
            
            plt.scatter(true, pred, alpha=0.5, s=20)
            
            # Add perfect prediction line
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Add metrics as text
            metrics_text = (
                f"MSE: {model_results[model]['mse']:.6f}\n"
                f"RMSE: {model_results[model]['rmse']:.6f}\n"
                f"MAE: {model_results[model]['mae']:.6f}\n"
                f"R²: {model_results[model]['r2']:.6f}"
            )
            plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            
            plt.title(f"{model} Model", fontsize=14)
            plt.xlabel("True SoH", fontsize=12)
            plt.ylabel("Predicted SoH", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    scatter_path = os.path.join(results_dir, "model_comparison_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    
    # 3. Create a table with metrics
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    for model in models:
        if model in model_results:
            table_data.append([
                model,
                f"{model_results[model]['mse']:.6f}",
                f"{model_results[model]['rmse']:.6f}",
                f"{model_results[model]['mae']:.6f}",
                f"{model_results[model]['r2']:.6f}"
            ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "MSE", "RMSE", "MAE", "R²"],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title("Model Performance Metrics", fontsize=16, fontweight='bold', pad=20)
    
    table_path = os.path.join(results_dir, "model_comparison_table.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log to wandb
    if hasattr(wandb, 'run') and wandb.run is not None:
        wandb.log({
            "comparison/metrics_bar_chart": wandb.Image(metrics_path),
            "comparison/prediction_scatter": wandb.Image(scatter_path),
            "comparison/metrics_table": wandb.Image(table_path),
        })
    
    return metrics_path, scatter_path, table_path

def main():
    parser = argparse.ArgumentParser('Model Comparison')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory with model results')
    parser.add_argument('--wandb_project', type=str, default='battery-health-comparison')
    parser.add_argument('--wandb_entity', type=str, default=None)
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"Model-Comparison-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
        config=vars(args)
    )
    
    # Models to compare
    models = ["PICL", "MLP", "CNN"]
    
    # Load results for each model
    model_results = {}
    for model_name in models:
        true_label, pred_label = load_results(args.results_dir, model_name)
        if true_label is not None and pred_label is not None:
            mse, rmse, mae, r2 = calculate_metrics(true_label, pred_label)
            model_results[model_name] = {
                "true": true_label,
                "pred": pred_label,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            # Log metrics to wandb
            wandb.log({
                f"comparison/{model_name}/mse": mse,
                f"comparison/{model_name}/rmse": rmse,
                f"comparison/{model_name}/mae": mae,
                f"comparison/{model_name}/r2": r2
            })
            
            print(f"{model_name} metrics - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    
    # Create comparison plots
    metrics_path, scatter_path, table_path = plot_comparison(args.results_dir, model_results)
    print(f"Comparison plots saved to: {metrics_path}, {scatter_path}, {table_path}")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()