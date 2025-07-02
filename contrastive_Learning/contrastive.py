import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Takes encoder outputs and maps them to a representation space for contrastive loss.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),  # Added extra layer for better representation
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class MomentumEncoder(nn.Module):
    """
    Momentum encoder for more stable contrastive learning (MoCo style).
    """
    def __init__(self, encoder, projection_head, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.momentum = momentum
        
        # Create momentum encoder and projector by deep copying the original networks
        self.momentum_encoder = deepcopy(encoder)
        self.momentum_projection = deepcopy(projection_head)
        
        # Disable gradients for momentum networks
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projection.parameters():
            param.requires_grad = False
            
    def update_momentum_encoder(self):
        """Update momentum encoder with moving average"""
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            
            for param_q, param_k in zip(self.projection_head.parameters(), self.momentum_projection.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def forward(self, x):
        """Encode with online encoder"""
        return self.projection_head(self.encoder(x))
    
    def momentum_forward(self, x):
        """Encode with momentum encoder"""
        with torch.no_grad():
            return self.momentum_projection(self.momentum_encoder(x))

class ContrastivePINNWrapper(nn.Module):
    """
    Wrapper for multi-task learning with PINN and contrastive loss.
    It expects the PINN to have an encoder (e.g., solution_u.encoder).
    """
    def __init__(self, pinn, temperature=0.07, contrastive_weight=0.5, pinn_weight=1.0,
                 projection_dim=128, momentum=0.999, queue_size=4096):
        super().__init__()
        self.pinn = pinn
        self.encoder = pinn.solution_u.encoder  # feature extractor for contrastive learning
        
        # Get encoder output dimension
        encoder_dim = 64  # This should match the output of your encoder
        
        # Add projection head for contrastive learning
        self.projection_head = ProjectionHead(input_dim=encoder_dim, output_dim=projection_dim)
        
        # Setup momentum encoder
        self.momentum_encoder = MomentumEncoder(self.encoder, self.projection_head, momentum=momentum)
        
        # Queue for MoCo-style negative samples
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight  # Reduced from 1.0 to 0.5
        self.pinn_weight = pinn_weight
        self.loss_func = nn.MSELoss()
        
        # Add alignment and uniformity weights
        self.alignment_weight = 0.5
        self.uniformity_weight = 0.5
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue with current batch"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace oldest samples in the queue
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Handle queue wraparound
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size-remaining] = keys[remaining:]
        
        # Update queue pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def alignment_and_uniformity(self, z_i, z_j):
        """
        Compute alignment and uniformity losses as in Wang & Isola (2020)
        
        Args:
            z_i, z_j: Normalized embeddings
        Returns:
            alignment, uniformity losses
        """
        # Alignment loss: Expected distance between positive pairs
        # Lower is better (positive pairs should be close)
        alignment = (z_i - z_j).norm(dim=1).pow(2).mean()
        
        # Uniformity loss: Measures how uniformly distributed the embeddings are
        # Lower is better (points should be uniformly distributed)
        t = 2  # Hyperparameter controlling the scale
        uniformity_i = torch.pdist(z_i, p=2).pow(2).mul(-t).exp().mean().log()
        uniformity_j = torch.pdist(z_j, p=2).pow(2).mul(-t).exp().mean().log()
        uniformity = (uniformity_i + uniformity_j) / 2
        
        return alignment, uniformity

    def simclr_loss(self, z_i, z_j):
        """
        Enhanced SimCLR-style contrastive loss with alignment and uniformity.
        
        Args:
            z_i, z_j: Normalized embeddings from the two augmented views
        Returns:
            Loss value
        """
        batch_size = z_i.size(0)
        
        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                               representations.unsqueeze(0), 
                                               dim=2) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Define positive pairs: (2i,2i+1) and (2i+1,2i)
        positives = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(0, batch_size)
        ]).to(z_i.device)
        
        # Basic SimCLR loss (NLL applied to softmax of similarities)
        nce_loss = F.cross_entropy(similarity_matrix, positives)
        
        # Calculate alignment and uniformity
        alignment, uniformity = self.alignment_and_uniformity(z_i, z_j)
        
        # Combined loss
        total_loss = nce_loss + self.alignment_weight * alignment + self.uniformity_weight * uniformity
        
        return total_loss

    def forward(self, x1, x2, y1, y2):
        """
        Forward pass for multi-task training.
        x1, x2: original and augmented inputs (or two augmentations per SimCLR)
        y1, y2: corresponding SoH targets
        Returns: total loss (PINN + contrastive), PINN loss, contrastive loss
        """
        device = x1.device
        
        # Handle different batch sizes by truncating to the smaller batch
        min_batch_size = min(x1.size(0), x2.size(0))
        if x1.size(0) != min_batch_size or x2.size(0) != min_batch_size or y1.size(0) != min_batch_size or y2.size(0) != min_batch_size:
            x1 = x1[:min_batch_size]
            x2 = x2[:min_batch_size]
            y1 = y1[:min_batch_size]
            y2 = y2[:min_batch_size]
        
        # -- PINN loss (apply to both original and augmented) --
        u1, f1 = self.pinn(x1)
        u2, f2 = self.pinn(x2)
        data_loss = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
        pde_loss = 0.5 * self.loss_func(f1, torch.zeros_like(f1)) + 0.5 * self.loss_func(f2, torch.zeros_like(f2))
        physics_loss = self.pinn.relu(torch.mul(u2 - u1, y1 - y2)).mean()
        pinn_loss = data_loss + self.pinn.alpha * pde_loss + self.pinn.beta * physics_loss
        
        # -- Contrastive representations with enhanced SimCLR approach --
        # Get encoder representations
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        # Project to contrastive space
        q1 = self.projection_head(z1)
        q2 = self.projection_head(z2)
        
        # Enhanced SimCLR contrastive loss with alignment and uniformity
        contrastive_loss = self.simclr_loss(q1, q2)
        
        # Update momentum encoder (keep this for consistency)
        self.momentum_encoder.update_momentum_encoder()
        
        # Dynamic contrastive weight based on relative loss magnitudes
        relative_scale = min(1.0, pinn_loss.item() / (contrastive_loss.item() + 1e-6))
        effective_weight = self.contrastive_weight * relative_scale
        
        # -- Total loss with adaptive weighting --
        total_loss = effective_weight * contrastive_loss + self.pinn_weight * pinn_loss
        
        return total_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss