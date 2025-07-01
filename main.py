from dataloader.dataloader import SimpleLoader
#from dataloader.dataloader import SimpleLoader_Augmented
from Model.Model import PINN
import argparse
import os
import matplotlib.pyplot as plt

def load_data(args):
    data = SimpleLoader(
        csv_path=args.csv_file,
        batch_size=args.batch_size,
        normalization=True,
        normalization_method=args.normalization_method
    )
    loaders = data.load()
    return loaders 
# def load_data_augmented(args):
#     data = SimpleLoader_Augmented(
#         csv_path_augmented=args.csv_file,
#         batch_size=args.batch_size,
#         normalization=True,
#         normalization_method=args.normalization_method
#     )
#    loaders_augmented = data.load()
#    return loaders_augmented
def main():
    args = get_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    setattr(args, "log_dir", args.log_dir)
    setattr(args, "save_folder", args.save_folder)

    dataloader = load_data(args)
   # dataloader_augmented = load_data_augmented(args)
    pinn = PINN(args)

    train_loss_list = []
    val_loss_list = []

    for epoch in range(args.epochs):
        loss1, loss2, loss3 = pinn.train_one_epoch(epoch + 1, dataloader['train'])
        train_loss = loss1 + pinn.alpha * loss2 + pinn.beta * loss3
        train_loss_list.append(train_loss)

        if dataloader.get('valid') is not None:
            val_loss = pinn.Valid(dataloader['valid'])
            val_loss_list.append(val_loss)
        else:
            val_loss_list.append(None)

        print(f"[Epoch {epoch+1}] Train loss: {train_loss:.4f} | Val loss: {val_loss_list[-1]:.4f}")

        pinn.scheduler.step()

    # Plot after training
    plt.figure(figsize=(10,7))
    plt.plot(train_loss_list, label="Train Loss")
    if any(v is not None for v in val_loss_list):
        plt.plot(val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, "loss_curve.png"))
    plt.show()
    true_label, pred_label = pinn.Test(dataloader['test'])
    plt.figure(figsize=(8,8))
    plt.scatter(true_label, pred_label, s=2, alpha=0.5)
    plt.xlabel("True SoH")
    plt.ylabel("Predicted SoH")
    plt.title("Test Set: True vs. Predicted SoH")
    plt.plot([min(true_label), max(true_label)], [min(true_label), max(true_label)], 'k--', lw=1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, "test_scatter.png"))
    plt.show()

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
    
    # More conservative learning parameters
    parser.add_argument('--epochs', type=int, default=300, help='epoch')
    parser.add_argument('--early_stop', type=int, default=30, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.0001, help='warmup lr')  # Reduced
    parser.add_argument('--lr', type=float, default=0.001, help='base lr')  # Reduced
    parser.add_argument('--final_lr', type=float, default=0.00001, help='final lr')  # Reduced
    parser.add_argument('--lr_F', type=float, default=0.0005, help='lr of F')  # Reduced
    
    # Increased model capacity
    parser.add_argument('--F_layers_num', type=int, default=4, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=128, help='the hidden dim of F')
    
    # Adjusted loss weights
    parser.add_argument('--alpha', type=float, default=1.0, help='PDE loss weight')
    parser.add_argument('--beta', type=float, default=0.5, help='physics constraint weight')
    
    parser.add_argument('--log_dir', type=str, default='training_log.txt', help='log dir')
    parser.add_argument('--save_folder', type=str, default='results', help='save folder')
    
    return parser.parse_args()

if __name__ == '__main__':
    main()