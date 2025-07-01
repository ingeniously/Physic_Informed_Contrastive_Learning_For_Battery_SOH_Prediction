import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastivePINNWrapper(nn.Module):
    """
    Wrapper for multi-task learning with PINN and contrastive loss.
    It expects the PINN to have an encoder (e.g., solution_u.encoder).
    """
    def __init__(self, pinn, temperature=0.07, contrastive_weight=1.0, pinn_weight=1.0):
        super().__init__()
        self.pinn = pinn
        self.encoder = pinn.solution_u.encoder  # feature extractor for contrastive learning
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.pinn_weight = pinn_weight
        self.loss_func = nn.MSELoss()

    def nt_xent_loss(self, z_i, z_j):
        """
        NT-Xent (SimCLR-style) contrastive loss between two batches of representations.
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
        z = F.normalize(z, dim=1)
        similarity = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        # mask self-comparisons
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(mask, float('-inf'))

        # Each sample's positive is at index: i + N (for i in 0..N-1), and i-N (for i in N..2N-1)
        positives = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z.device)
        labels = positives

        # For each row i, the positive is at column labels[i]
        loss = F.cross_entropy(similarity, labels)
        return loss

    def forward(self, x1, x2, y1, y2):
        """
        Forward pass for multi-task training.
        x1, x2: original and augmented inputs (or two augmentations per SimCLR)
        y1, y2: corresponding SoH targets
        Returns: total loss (PINN + contrastive), PINN loss, contrastive loss
        """
        # -- Contrastive representations --
        # Use encoder only, do NOT detach, so gradients flow through full PINN
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        contrastive_loss = self.nt_xent_loss(z1, z2)

        # -- PINN loss (apply to both original and augmented) --
        u1, f1 = self.pinn(x1)
        u2, f2 = self.pinn(x2)
        data_loss = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
        pde_loss = 0.5 * self.loss_func(f1, torch.zeros_like(f1)) + 0.5 * self.loss_func(f2, torch.zeros_like(f2))
        physics_loss = self.pinn.relu(torch.mul(u2 - u1, y1 - y2)).mean()
        pinn_loss = data_loss + self.pinn.alpha * pde_loss + self.pinn.beta * physics_loss

        # -- Total loss --
        total_loss = self.contrastive_weight * contrastive_loss + self.pinn_weight * pinn_loss
        return total_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss