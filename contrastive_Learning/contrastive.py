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
        # This avoids issues with __dict__ containing training flags
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
    def __init__(self, pinn, temperature=0.07, contrastive_weight=1.0, pinn_weight=1.0,
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
        self.contrastive_weight = contrastive_weight
        self.pinn_weight = pinn_weight
        self.loss_func = nn.MSELoss()
        
        # Hard negative mining parameters
        self.hard_negative_weight = 2.0  # Weight for hard negatives
        self.hard_threshold = 0.75  # Cosine similarity threshold for hard negatives
        
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

    def info_nce_loss(self, query, key, negatives):
        """
        InfoNCE loss with hard negative mining and temperature scaling.
        
        Args:
            query: Query embeddings (output from online encoder)
            key: Positive key embeddings (output from momentum encoder)
            negatives: Negative samples (from queue)
        Returns:
            Loss value
        """
        # Positive similarity
        pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        
        # Negative similarity
        neg = torch.einsum('nc,kc->nk', [query, negatives])
        
        # Identify hard negatives (high similarity negatives)
        with torch.no_grad():
            hard_mask = (neg > self.hard_threshold).float()
            # Ensure we have at least some hard negatives
            if hard_mask.sum() == 0:
                topk_values, _ = torch.topk(neg, k=max(1, int(0.1 * neg.shape[1])), dim=1)
                min_topk = topk_values[:, -1].unsqueeze(-1)
                hard_mask = (neg >= min_topk).float()
                
        # Apply hard negative mining by increasing weight of hard negatives
        neg = neg * (1 + (self.hard_negative_weight - 1) * hard_mask)
            
        # All logits (positive and negatives)
        logits = torch.cat([pos, neg], dim=1) / self.temperature
        
        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Calculate cross-entropy loss
        return F.cross_entropy(logits, labels)

    def nt_xent_loss(self, z_i, z_j, z_neg=None):
        """
        NT-Xent (SimCLR-style) contrastive loss with hard negative mining.
        
        Args:
            z_i, z_j: Positive pair embeddings
            z_neg: Optional additional negative samples
        Returns:
            Loss value
        """
        batch_size = z_i.size(0)
        
        # Original positive pairs (i,j) and (j,i)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
        z = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]
        
        # Mask self-comparisons
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(mask, float('-inf'))
        
        # Identify positive pairs: (i,j) and (j,i)
        positives = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        # Get negatives - all except positives and self
        batch_indices = torch.arange(2*batch_size).to(z.device)
        expanded_positives = positives.unsqueeze(1).expand(-1, 2*batch_size)
        expanded_indices = batch_indices.unsqueeze(0).expand(2*batch_size, -1)
        negative_mask = ~((expanded_indices == expanded_positives) | mask)
        
        # Hard negative mining
        with torch.no_grad():
            # Find most similar negatives
            neg_similarities = similarity.clone()
            neg_similarities.masked_fill_(~negative_mask, float('-inf'))
            hard_negatives = (neg_similarities > self.hard_threshold)
            
            # Ensure we have some hard negatives
            if not hard_negatives.any():
                # If no negatives above threshold, take top 10% as hard
                neg_similarities_flat = neg_similarities.reshape(2*batch_size, -1)
                k = max(1, int(0.1 * neg_similarities_flat.shape[1]))
                topk_vals, _ = torch.topk(neg_similarities_flat, k=k, dim=1)
                min_topk = topk_vals[:, -1].unsqueeze(-1)
                hard_negatives = (neg_similarities_flat >= min_topk).view_as(neg_similarities)
                hard_negatives = hard_negatives & negative_mask
            
        # Apply stronger weighting to hard negatives
        similarity = similarity.masked_fill(hard_negatives, similarity * self.hard_negative_weight)
        
        # Compute logits and labels for contrastive loss
        logits = similarity
        labels = positives
        
        # Cross entropy loss
        return F.cross_entropy(logits, labels)

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
        
        # -- Contrastive representations with momentum encoder --
        # Get online encoder representations
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        # Project to contrastive space
        q1 = self.projection_head(z1)  # queries from online encoder
        q2 = self.projection_head(z2)
        
        # Get momentum encoder representations (no gradient)
        with torch.no_grad():
            k1 = self.momentum_encoder.momentum_forward(x1)  # keys from momentum encoder
            k2 = self.momentum_encoder.momentum_forward(x2)
        
        # MoCo contrastive loss
        queue = self.queue.clone().detach()
        contrastive_loss = 0.5 * self.info_nce_loss(q1, k2, queue) + 0.5 * self.info_nce_loss(q2, k1, queue)
        
        # Update queue and momentum encoder
        self._dequeue_and_enqueue(torch.cat([k1, k2], dim=0))
        self.momentum_encoder.update_momentum_encoder()
        
        # -- Total loss --
        total_loss = self.contrastive_weight * contrastive_loss + self.pinn_weight * pinn_loss
        return total_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss