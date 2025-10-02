import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class TransformerProjection(nn.Module):
    """
    BERT-style transformer encoder projection head for contrastive learning
    """
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=128, 
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Initial input projection to transformer dimension (Input Embedding)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer encoder layers (exact BERT structure)
        # For PyTorch 1.7.1, use the compatible parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,  # Standard BERT uses 4x hidden size
            dropout=dropout,
            activation="gelu"  # Use string instead of function for older PyTorch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection (BERT pooler equivalent)
        self.pooler = nn.Linear(hidden_dim, hidden_dim)
        self.pooler_activation = nn.Tanh()  # BERT uses tanh in its pooler
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape input into batch_size x 1 x input_dim to add sequence dimension
        x = x.unsqueeze(1)  # shape: [batch_size, 1, input_dim]
        
        # Initial embedding + normalization (input embedding)
        x = self.input_proj(x)
        x = self.embedding_norm(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # For PyTorch 1.7.1, TransformerEncoder expects [seq_len, batch_size, feat_dim]
        # So we need to transpose from [batch_size, seq_len, feat_dim]
        x = x.transpose(0, 1)  # shape: [1, batch_size, hidden_dim]
        
        # Pass through transformer blocks
        x = self.transformer(x)  # shape: [1, batch_size, hidden_dim]
        
        # Transpose back to [batch_size, seq_len, feat_dim]
        x = x.transpose(0, 1)  # shape: [batch_size, 1, hidden_dim]
        
        # Take the "CLS" token (in our case, the only token)
        x = x.squeeze(1)  # shape: [batch_size, hidden_dim]
        
        # BERT pooler: dense layer with tanh activation
        x = self.pooler(x)
        x = self.pooler_activation(x)
        
        # Final projection to output dimension
        x = self.output_proj(x)
        x = self.output_norm(x)
        
        # L2 normalization for contrastive learning
        return F.normalize(x, dim=1)
class MomentumEncoder(nn.Module):
    """
    Momentum encoder for more stable contrastive learning (MoCo style).
    """
    def __init__(self, encoder, projection_head, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        # Start with lower momentum for faster initial learning
        self.momentum = 0.99  # Lower initial value
        self.target_momentum = momentum
        
        # Create momentum encoder and projector by deep copying the original networks
        self.momentum_encoder = deepcopy(encoder)
        self.momentum_projection = deepcopy(projection_head)
        
        # Disable gradients for momentum networks
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projection.parameters():
            param.requires_grad = False
    def update_momentum(self, epoch, total_epochs):
        """Gradually increase momentum throughout training"""
        self.momentum = min(
            self.target_momentum,
            self.momentum + (self.target_momentum - 0.99) * epoch / (0.5 * total_epochs)
        )      
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
    Enhanced wrapper for multi-task learning with PINN and contrastive loss
    """
    def __init__(self, pinn, temperature=0.07, contrastive_weight=1.0, pinn_weight=1.0,
                 projection_dim=128, momentum=0.999, queue_size=4096):
        super().__init__()
        self.pinn = pinn
        self.encoder = pinn.solution_u.encoder
        
        # Get encoder output dimension (from the actual model)
        encoder_dim = 64  # Should match the output dim of your encoder
        
        self.projection_head = TransformerProjection(
            input_dim=encoder_dim, 
            hidden_dim=projection_dim*2,      # 256 if projection_dim=128
            output_dim=projection_dim,        # 128 by default
            num_heads=4,                      # 4 attention heads
            num_layers=2,                     # 2 transformer layers
            dropout=0.1,                      # Small dropout for regularization
        )

        # Setup momentum encoder
        self.momentum_encoder = MomentumEncoder(
            self.encoder, 
            self.projection_head, 
            momentum=momentum
        )
        
        # Queue for MoCo-style negative samples
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        
        # Parameters with reasonable defaults
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.pinn_weight = pinn_weight
        self.loss_func = nn.MSELoss()
        
        # Add alignment and uniformity weights
        self.alignment_weight = 0.5
        self.uniformity_weight = 0.5
        
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

    def hard_negative_mining(self, z_i, z_j):
        """
        Implement hard negative mining to focus on difficult examples
        """
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Calculate similarity between all embeddings
        similarity = torch.matmul(z_i, z_j.T)
        
        # Create mask for positive pairs (diagonal)
        mask = torch.eye(batch_size, device=similarity.device)
        
        # Find hardest negative for each sample (highest similarity that isn't positive)
        hardest_negatives_i = (similarity * (1 - mask)).max(dim=1)[1]
        hardest_negatives_j = (similarity.T * (1 - mask)).max(dim=1)[1]
        
        # Get embeddings of hardest negatives
        z_i_hard = z_j[hardest_negatives_i]
        z_j_hard = z_i[hardest_negatives_j]
        
        # Mix original and hard negatives (with alpha controlling the ratio)
        alpha = 0.6  # Balance between original and hard negatives
        z_i_mixed = alpha * z_i + (1 - alpha) * z_i_hard
        z_j_mixed = alpha * z_j + (1 - alpha) * z_j_hard
        
        return z_i_mixed, z_j_mixed

    def supervised_contrastive_loss(self, z_i, z_j, y_i, y_j, threshold=0.05):
        """
        Add supervision signal to contrastive loss using SoH values
        Treats samples with similar SoH values as positives
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        features = torch.cat([z_i, z_j], dim=0)
        
        # Create labels: samples with SoH difference < threshold are considered positives
        labels = torch.cat([y_i, y_j], dim=0).flatten()
        similarity_mask = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0)) < threshold
        
        # Compute cosine similarity
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Remove self-contrast cases
        self_mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
        sim_matrix.masked_fill_(self_mask, -9e15)
        
        # For each sample, compute loss against all positive samples
        loss = 0
        n_loss_terms = 0
        
        for i in range(2 * batch_size):
            positive_indices = torch.nonzero(similarity_mask[i] & ~self_mask[i]).flatten()
            
            if len(positive_indices) > 0:
                # Select logits corresponding to positive samples for this anchor
                anchor_dot_contrast = sim_matrix[i]
                
                # Create mask for positive samples
                positive_mask = torch.zeros_like(anchor_dot_contrast).bool()
                positive_mask[positive_indices] = True
                
                # Compute log_prob using log-sum-exp trick for numerical stability
                positive_logits = anchor_dot_contrast[positive_mask]
                
                negative_mask = ~positive_mask & ~self_mask[i]
                negative_logits = anchor_dot_contrast[negative_mask]
                
                if len(negative_logits) > 0:
                    # Cross entropy loss with soft labels
                    logits = torch.cat([positive_logits, negative_logits])
                    labels = torch.zeros(len(logits), device=logits.device)
                    labels[:len(positive_logits)] = 1.0 / len(positive_logits)
                    
                    loss += -(labels * F.log_softmax(logits, dim=0)).sum()
                    n_loss_terms += 1
        
        if n_loss_terms > 0:
            loss = loss / n_loss_terms
        else:
            loss = torch.tensor(0.0, device=z_i.device)
        
        return loss

    def improved_simclr_loss(self, z_i, z_j):
        """
        Enhanced SimCLR loss with better numerical stability
        """
        batch_size = z_i.size(0)
        
        # Normalize the embeddings (important!)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate to create 2N vectors
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Calculate similarity matrix with temperature scaling
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Remove diagonal (self-similarity)
        sim_mask = torch.eye(2 * batch_size, device=similarity_matrix.device).bool()
        similarity_matrix.masked_fill_(sim_mask, -9e15)  # Set to large negative value
        
        # Separate positive pairs: (i,i+N) and (i+N,i)
        positives = torch.zeros((2 * batch_size, 2 * batch_size), device=similarity_matrix.device).bool()
        # Indices i,i+N (first half to second half)
        positives[:batch_size, batch_size:] = torch.eye(batch_size, device=similarity_matrix.device).bool()
        # Indices i+N,i (second half to first half)
        positives[batch_size:, :batch_size] = torch.eye(batch_size, device=similarity_matrix.device).bool()
        
        # For each row, select its positive sample's similarity
        positive_similarities = similarity_matrix[positives].view(2 * batch_size, 1)
        
        # For each row, compute the log-sum-exp of all similarities (except self)
        # Then subtract the positive similarity to get the contrastive loss term
        exp_sim = torch.exp(similarity_matrix)
        # Zero out diagonal
        exp_sim = exp_sim * (~sim_mask).float()
        # Sum over each row
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        # Compute log of denominator
        log_prob = positive_similarities - torch.log(sum_exp_sim + 1e-8)
        # Take negative mean
        loss = -log_prob.mean()
        
        return loss

    def forward(self, x1, x2, y1, y2):
        """
        Forward pass combining PINN and contrastive learning
        """
        device = x1.device
        
        # Handle different batch sizes by truncating
        min_batch_size = min(x1.size(0), x2.size(0))
        if x1.size(0) != min_batch_size or x2.size(0) != min_batch_size:
            x1 = x1[:min_batch_size]
            x2 = x2[:min_batch_size]
            y1 = y1[:min_batch_size]
            y2 = y2[:min_batch_size]
        
        # -- PINN forward pass --
        u1, f1 = self.pinn(x1)
        u2, f2 = self.pinn(x2)
        
        # PINN losses
        data_loss = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
        pde_loss = 0.5 * self.loss_func(f1, torch.zeros_like(f1)) + 0.5 * self.loss_func(f2, torch.zeros_like(f2))
        physics_loss = self.pinn.relu(torch.mul(u2 - u1, y1 - y2)).mean()
        pinn_loss = data_loss + self.pinn.alpha * pde_loss + self.pinn.beta * physics_loss
        
        # -- Contrastive learning part --
        # Get encoder representations
        with torch.no_grad():  # Detach encoder for stability
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
        
        # Apply feature-level augmentation in embedding space for more diversity
        if self.training:
            # Random feature masking
            mask_ratio = 0.1
            feature_mask1 = (torch.rand_like(z1) > mask_ratio).float()
            feature_mask2 = (torch.rand_like(z2) > mask_ratio).float()
            
            z1 = z1 * feature_mask1
            z2 = z2 * feature_mask2
            
            # Add small noise to embeddings for robustness
            noise_level = 0.05
            z1 = z1 + noise_level * torch.randn_like(z1)
            z2 = z2 + noise_level * torch.randn_like(z2)
            
            # Optional: Apply hard negative mining
            z1, z2 = self.hard_negative_mining(z1, z2)
        
        # Project to contrastive space
        q1 = self.projection_head(z1)
        q2 = self.projection_head(z2)
        
        # Normalize for contrastive loss
        q1_norm = F.normalize(q1, dim=1)
        q2_norm = F.normalize(q2, dim=1)
        
        # Update momentum encoder
        self.momentum_encoder.update_momentum_encoder()
        
        # Get momentum projections
        with torch.no_grad():
            k1 = self.momentum_encoder.momentum_forward(x1)
            k2 = self.momentum_encoder.momentum_forward(x2)
            k1_norm = F.normalize(k1, dim=1)
            k2_norm = F.normalize(k2, dim=1)
        
        # Compute contrastive losses - combine multiple techniques
        # 1. Standard InfoNCE/SimCLR loss
        simclr_loss = self.improved_simclr_loss(q1_norm, q2_norm)
        
        # 2. Cross-model contrast (online vs momentum)
        cross_loss = self.improved_simclr_loss(q1_norm, k2_norm) + self.improved_simclr_loss(q2_norm, k1_norm)
        
        # 3. Supervised component using SoH values
        sup_loss = self.supervised_contrastive_loss(q1_norm, q2_norm, y1, y2)
        
        # 4. Alignment and uniformity
        alignment, uniformity = self.alignment_and_uniformity(q1_norm, q2_norm)
        
        # Combined contrastive loss with weights
        contrastive_loss = (
            0.4 * simclr_loss + 
            0.2 * cross_loss + 
            0.2 * sup_loss + 
            self.alignment_weight * alignment + 
            self.uniformity_weight * uniformity
        )
        
        # Enqueue current batch
        if self.training:
            with torch.no_grad():
                # Use momentum representations for queue
                batch_keys = torch.cat([k1_norm, k2_norm], dim=0)
                self._dequeue_and_enqueue(batch_keys)
        
        # Weighted sum of losses
        # Dynamically adjust contrastive weight based on loss magnitudes
        relative_scale = min(1.0, pinn_loss.item() / (contrastive_loss.item() + 1e-6))
        effective_weight = self.contrastive_weight * relative_scale
        
        total_loss = self.pinn_weight * pinn_loss + effective_weight * contrastive_loss
        
        return total_loss, pinn_loss, contrastive_loss, pde_loss, physics_loss