"""
Training Module for scARKIDS
============================

Implements end-to-end training for VAE-DDPM model using the ELBO module.

Mathematical Background:
-----------------------

This module uses the ELBOModule to compute all loss terms rigorously:

SUPERVISED MODE:
- ELBO = Reconstruction - VAE_KL - Diffusion_KL - Terminal_KL
- Reconstruction: E[log p_θ(x|z^(0), b)]
- VAE_KL: D_KL(q_φ(z^(0)|x,b,c*)||p(z^(0)|c*))
- Diffusion_KL: Σ_t D_KL(q(z^(t-1)|z^(t),z^(0))||p_ψ(z^(t-1)|z^(t),c*))
- Terminal_KL: D_KL(q(z^(T)|z^(0))||p(z^(T)))

UNSUPERVISED MODE:
- ELBO = Reconstruction - CellType_KL - VAE_KL - Diffusion_KL - Terminal_KL
- CellType_KL: D_KL(q_ω(c|x,b)||p(c))
- Other terms similar to supervised but averaged over predicted cell types

The training loop:
1. Sample mini-batch from DataLoader
2. Forward pass through ELBOModule to compute loss and all terms
3. Backward pass to compute gradients
4. Gradient clipping for stability
5. Optimizer step to update parameters
6. Learning rate scheduling
7. Periodic logging and checkpointing
"""

from src.utils.logger import Logger
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
import json
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for training.
    
    Attributes:
        supervised: Whether to use supervised (True) or unsupervised (False) mode
        n_epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
        grad_clip_norm: Maximum gradient norm for clipping
        log_interval: Log metrics every N batches
        checkpoint_interval: Save checkpoint every N epochs
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on ('cuda' or 'cpu')
        use_amp: Whether to use automatic mixed precision
        lr_scheduler: Learning rate scheduler type ('cosine', 'step', 'plateau', None)
        lr_warmup_epochs: Number of warmup epochs for learning rate
    """
    supervised: bool
    n_epochs: int
    batch_size: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    log_interval: int = 100
    checkpoint_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    lr_scheduler: Optional[str] = "cosine"
    lr_warmup_epochs: int = 5
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.n_epochs > 0, "n_epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.grad_clip_norm > 0, "grad_clip_norm must be positive"

# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)

# ============================================================================
# Training Module Core
# ============================================================================

class TrainingModuleCore:
    """
    Core training logic for VAE-DDPM model using ELBO module.
    
    This class implements the training loop that:
    1. Uses ELBOModule to compute all loss terms rigorously
    2. Manages optimizer and learning rate scheduler
    3. Handles gradient clipping and mixed precision training
    4. Provides checkpointing and logging functionality
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        likelihood_module,
        prior_module,
        ddpm_forward_module,
        ddpm_backward_module,
        encoder_module,
        variational_posterior_module,
        elbo_module,
        classifier_module=None
    ):
        """
        Initialize training module core.
        
        Args:
            config: Training configuration
            likelihood_module: LikelihoodModule instance
            prior_module: PriorModule instance
            ddpm_forward_module: DDPMForwardModule instance
            ddpm_backward_module: DDPMBackwardModule instance
            encoder_module: VAEEncoder instance
            variational_posterior_module: VariationalPosteriorModule instance
            elbo_module: ELBOModule instance (computes all loss terms)
            classifier_module: ClassifierModule instance (required if unsupervised)
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Store modules
        self.likelihood = likelihood_module
        self.prior = prior_module
        self.ddpm_forward = ddpm_forward_module
        self.ddpm_backward = ddpm_backward_module
        self.encoder = encoder_module
        self.variational_posterior = variational_posterior_module
        self.elbo = elbo_module  # This module computes all loss terms
        self.classifier = classifier_module
        
        # Validate unsupervised mode
        if not config.supervised and classifier_module is None:
            raise ValueError("Unsupervised mode requires classifier_module")
        
        # Move all modules to device
        self._move_to_device()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup AMP scaler if using mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = []
        
        logger.info("Initialized TrainingModuleCore")
        logger.info(f"  Mode: {'Supervised' if config.supervised else 'Unsupervised'}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Using ELBO module for loss computation")
    
    def _move_to_device(self):
        """Move all modules to the specified device."""
        self.likelihood.to(self.device)
        self.prior.to(self.device)
        self.ddpm_forward.to(self.device)
        self.ddpm_backward.to(self.device)
        self.encoder.to(self.device)
        self.variational_posterior.to(self.device)
        self.elbo.to(self.device)
        if self.classifier is not None:
            self.classifier.to(self.device)
        
        logger.info(f"Moved all modules to {self.device}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """
        Setup optimizer for all trainable parameters.
        
        Collects parameters from:
        - Likelihood (decoder networks, dispersion)
        - Prior (cell-type-specific means/vars if learnable)
        - DDPM backward (noise prediction network)
        - Encoder (VAE encoder)
        - Classifier (if unsupervised)
        
        Note: We could also use elbo.parameters() but we collect explicitly
        for clarity and to ensure all atomic modules are included.
        """
        params = []
        
        # Likelihood parameters (θ)
        params.extend(self.likelihood.parameters())
        
        # Prior parameters (learnable priors if any)
        params.extend(self.prior.parameters())
        
        # DDPM backward parameters (ψ - noise prediction network)
        params.extend(self.ddpm_backward.parameters())
        
        # Encoder parameters (φ - VAE encoder)
        params.extend(self.encoder.parameters())
        
        # Classifier parameters (ω - unsupervised only)
        if self.classifier is not None:
            params.extend(self.classifier.parameters())
        
        # Count total parameters
        total_params = sum(p.numel() for p in params if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        # Create optimizer
        optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler with optional warmup."""
        if self.config.lr_scheduler is None:
            return None
        
        if self.config.lr_scheduler == "cosine":
            # Cosine annealing with warmup
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs - self.config.lr_warmup_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.lr_scheduler == "step":
            # Step decay
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.n_epochs // 3,
                gamma=0.1
            )
        elif self.config.lr_scheduler == "plateau":
            # Reduce on plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.config.lr_scheduler}")
        
        logger.info(f"Using {self.config.lr_scheduler} learning rate scheduler")
        return scheduler
    
    def _apply_warmup(self, epoch: int):
        """Apply learning rate warmup for initial epochs."""
        if epoch < self.config.lr_warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.lr_warmup_epochs
            lr = self.config.learning_rate * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            logger.debug(f"Warmup epoch {epoch+1}: lr = {lr:.6e}")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step using ELBO module.

        Args:
            batch: Dictionary containing:
                - 'x': Gene expression (batch_size, n_genes)
                - 'batch_onehot': Batch indicator (batch_size, n_batches)
                - 'batch_indices': Batch indices (batch_size,)
                - 'celltype_onehot': Cell type one-hot (supervised only)
                - 'celltype_indices': Cell type indices (supervised only)
                - 'library_size': Library size (optional)

        Returns:
            Dictionary with loss metrics from ELBO module
        """
        # Set modules to training mode
        self.likelihood.train()
        self.prior.train()  # ← IMPORTANT: Prior has learnable params in supervised mode
        self.encoder.train()
        self.ddpm_backward.train()
        self.elbo.train()
        if self.classifier is not None:
            self.classifier.train()

        # Move batch to device
        x = batch['x'].to(self.device)
        batch_onehot = batch['batch_onehot'].to(self.device)
        library_size = batch.get('library_size', None)
        if library_size is not None:
            library_size = library_size.to(self.device)

        # Get celltype_onehot for supervised mode
        celltype_onehot = None
        if self.config.supervised:
            celltype_onehot = batch['celltype_onehot'].to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute loss using ELBO module
        # The ELBO module handles all loss terms:
        # - Supervised: reconstruction, vae_kl, diffusion_kl, terminal_kl
        # - Unsupervised: reconstruction, celltype_kl, vae_kl, diffusion_kl, terminal_kl
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                loss, metrics = self.elbo(
                    x=x,
                    batch_onehot=batch_onehot,
                    celltype_onehot=celltype_onehot,
                    library_size=library_size
                )
        else:
            loss, metrics = self.elbo(
                x=x,
                batch_onehot=batch_onehot,
                celltype_onehot=celltype_onehot,
                library_size=library_size
            )

        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()

            # Gradient clipping - INCLUDING PRIOR
            self.scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for module in [self.likelihood, self.prior, self.encoder, self.ddpm_backward, self.classifier]
                 if module is not None for p in module.parameters()],
                self.config.grad_clip_norm
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            # Gradient clipping - INCLUDING PRIOR
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for module in [self.likelihood, self.prior, self.encoder, self.ddpm_backward, self.classifier]
                 if module is not None for p in module.parameters()],
                self.config.grad_clip_norm
            )

            # Optimizer step
            self.optimizer.step()

        # Add gradient norm to metrics
        metrics['grad_norm'] = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

        # Update global step
        self.global_step += 1

        return metrics

    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Dictionary with average epoch metrics
        """
        epoch_metrics = {}
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.n_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Training step (uses ELBO module internally)
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            num_batches += 1
            
            # Log periodically
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = epoch_metrics['loss'] / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Detailed logging
                log_msg = f"Batch {batch_idx+1}/{len(train_loader)}:"
                for key, value in metrics.items():
                    log_msg += f" {key}={value:.4f}"
                logger.debug(log_msg)
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Update epoch counter
        self.current_epoch += 1
        
        return epoch_metrics

    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary with average validation metrics
        """
        # Set modules to evaluation mode
        self.likelihood.eval()
        self.prior.eval()
        self.encoder.eval()
        self.ddpm_backward.eval()
        self.elbo.eval()
        if self.classifier is not None:
            self.classifier.eval()
        
        epoch_metrics = {}
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(val_loader, desc="Validation")
        
        # No gradient computation during validation
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                x = batch['x'].to(self.device)
                batch_onehot = batch['batch_onehot'].to(self.device)
                library_size = batch.get('library_size', None)
                if library_size is not None:
                    library_size = library_size.to(self.device)
                
                # Get celltype_onehot for supervised mode
                celltype_onehot = None
                if self.config.supervised:
                    celltype_onehot = batch['celltype_onehot'].to(self.device)
                
                # Compute loss using ELBO module (no gradient)
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, metrics = self.elbo(
                            x=x,
                            batch_onehot=batch_onehot,
                            celltype_onehot=celltype_onehot,
                            library_size=library_size
                        )
                else:
                    loss, metrics = self.elbo(
                        x=x,
                        batch_onehot=batch_onehot,
                        celltype_onehot=celltype_onehot,
                        library_size=library_size
                    )
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
                num_batches += 1
                
                # Update progress bar
                avg_loss = epoch_metrics['loss'] / num_batches
                pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Return to training mode
        self.likelihood.train()
        self.prior.train()
        self.encoder.train()
        self.ddpm_backward.train()
        self.elbo.train()
        if self.classifier is not None:
            self.classifier.train()
        
        return epoch_metrics
    

    def save_checkpoint(
        self,
        filepath: str,
        additional_info: Optional[Dict] = None
    ):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'prior_state_dict': self.prior.state_dict(),
            'ddpm_backward_state_dict': self.ddpm_backward.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.classifier is not None:
            checkpoint['classifier_state_dict'] = self.classifier.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        self.prior.load_state_dict(checkpoint['prior_state_dict'])
        self.ddpm_backward.load_state_dict(checkpoint['ddpm_backward_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        if self.classifier is not None and 'classifier_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {filepath}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

# ============================================================================
# Training Manager (Entry Point)
# ============================================================================

class TrainingManager:
    """
    Manager class for training (single entry point).
    
    This is the main interface instantiated by main.py that:
    1. Takes all atomic objects and composed objects (including ELBO)
    2. Initializes training module core
    3. Provides high-level training API
    4. Uses ELBO module for rigorous loss computation
    """
    
    def __init__(
        self,
        config_dict: Dict,
        likelihood_manager,
        prior_manager,
        ddpm_forward_manager,
        ddpm_backward_manager,
        encoder_manager,
        variational_posterior_manager,
        elbo_manager,
        classifier_manager=None
    ):
        """
        Initialize training manager.
        
        Args:
            config_dict: Training configuration dictionary
            likelihood_manager: LikelihoodManager instance
            prior_manager: PriorManager instance
            ddpm_forward_manager: DDPMForwardManager instance
            ddpm_backward_manager: DDPMBackwardManager instance
            encoder_manager: EncoderManager instance
            variational_posterior_manager: VariationalPosteriorManager instance
            elbo_manager: ELBOManager instance (computes all loss terms)
            classifier_manager: ClassifierManager instance (optional)
        """
        logger.info("Initializing TrainingManager")
        
        # Parse configuration
        self.config = self._parse_config(config_dict)
        
        # Get atomic modules
        likelihood_module = likelihood_manager.get_module()
        prior_module = prior_manager.get_module()
        ddpm_forward_module = ddpm_forward_manager.get_module()
        ddpm_backward_module = ddpm_backward_manager.get_module()
        encoder_module = encoder_manager.get_module()
        variational_posterior_module = variational_posterior_manager.get_module()
        
        # Get ELBO module (this computes all loss terms rigorously)
        elbo_module = elbo_manager.get_module()
        
        # Get classifier module if present
        classifier_module = classifier_manager.get_module() if classifier_manager is not None else None
        
        # Initialize training module core
        self.training_core = TrainingModuleCore(
            config=self.config,
            likelihood_module=likelihood_module,
            prior_module=prior_module,
            ddpm_forward_module=ddpm_forward_module,
            ddpm_backward_module=ddpm_backward_module,
            encoder_module=encoder_module,
            variational_posterior_module=variational_posterior_module,
            elbo_module=elbo_module,
            classifier_module=classifier_module
        )
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("TrainingManager initialized successfully")
        logger.info("Using ELBO module for rigorous loss computation:")
        logger.info("  - Reconstruction term: E[log p_θ(x|z^(0), b)]")
        if self.config.supervised:
            logger.info("  - VAE KL: D_KL(q_φ(z^(0)|x,b,c*)||p(z^(0)|c*))")
        else:
            logger.info("  - Cell Type KL: D_KL(q_ω(c|x,b)||p(c))")
            logger.info("  - VAE KL: D_KL(q_φ(z^(0)|x,b,c)||p(z^(0)))")
        logger.info("  - Diffusion KL: Σ_t D_KL(q(z^(t-1)|z^(t),z^(0))||p_ψ(z^(t-1)|z^(t),c))")
        logger.info("  - Terminal KL: D_KL(q(z^(T)|z^(0))||p(z^(T)))")
    
    def _parse_config(self, config_dict: Dict) -> TrainingConfig:
        """Parse configuration dictionary."""
        try:
            config = TrainingConfig(
                supervised=config_dict['supervised'],
                n_epochs=config_dict['n_epochs'],
                batch_size=config_dict['batch_size'],
                learning_rate=config_dict.get('learning_rate', 1e-3),
                weight_decay=config_dict.get('weight_decay', 1e-5),
                grad_clip_norm=config_dict.get('grad_clip_norm', 1.0),
                log_interval=config_dict.get('log_interval', 100),
                checkpoint_interval=config_dict.get('checkpoint_interval', 1),
                checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints'),
                device=config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                use_amp=config_dict.get('use_amp', False),
                lr_scheduler=config_dict.get('lr_scheduler', 'cosine'),
                lr_warmup_epochs=config_dict.get('lr_warmup_epochs', 5)
            )
            logger.info("Training configuration parsed successfully")
            return config
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Main training loop with optional validation.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        """
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)
        logger.info(f"Training for {self.config.n_epochs} epochs")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Number of training batches: {len(train_loader)}")
        if val_loader is not None:
            logger.info(f"Number of validation batches: {len(val_loader)}")
            logger.info("Validation will be performed after each epoch")

        try:
            for epoch in range(self.training_core.current_epoch, self.config.n_epochs):
                logger.info("-" * 80)
                logger.info(f"Epoch {epoch+1}/{self.config.n_epochs}")
                logger.info("-" * 80)

                # Apply learning rate warmup
                self.training_core._apply_warmup(epoch)

                # ==================== TRAINING ====================
                train_metrics = self.training_core.train_epoch(train_loader)

                # Log training metrics (all terms from ELBO)
                logger.info(f"Training metrics:")
                logger.info(f"  Total Loss: {train_metrics['loss']:.6f}")
                logger.info(f"  ELBO: {train_metrics['elbo']:.6f}")
                logger.info(f"  Reconstruction: {train_metrics['reconstruction']:.6f}")

                if self.config.supervised:
                    logger.info(f"  VAE KL: {train_metrics['vae_kl']:.6f}")
                else:
                    logger.info(f"  Cell Type KL: {train_metrics['celltype_kl']:.6f}")
                    logger.info(f"  VAE KL: {train_metrics['vae_kl']:.6f}")

                logger.info(f"  Diffusion KL: {train_metrics['diffusion_kl']:.6f}")
                logger.info(f"  Terminal KL: {train_metrics['terminal_kl']:.6f}")
                logger.info(f"  Gradient Norm: {train_metrics['grad_norm']:.6f}")

                # ==================== VALIDATION ====================
                val_metrics = None
                if val_loader is not None:
                    logger.info("-" * 40)
                    logger.info("Running validation...")
                    val_metrics = self.training_core.validate_epoch(val_loader)

                    # Log validation metrics
                    logger.info(f"Validation metrics:")
                    logger.info(f"  Total Loss: {val_metrics['loss']:.6f}")
                    logger.info(f"  ELBO: {val_metrics['elbo']:.6f}")
                    logger.info(f"  Reconstruction: {val_metrics['reconstruction']:.6f}")

                    if self.config.supervised:
                        logger.info(f"  VAE KL: {val_metrics['vae_kl']:.6f}")
                    else:
                        logger.info(f"  Cell Type KL: {val_metrics['celltype_kl']:.6f}")
                        logger.info(f"  VAE KL: {val_metrics['vae_kl']:.6f}")

                    logger.info(f"  Diffusion KL: {val_metrics['diffusion_kl']:.6f}")
                    logger.info(f"  Terminal KL: {val_metrics['terminal_kl']:.6f}")
                    logger.info("-" * 40)

                # ==================== LEARNING RATE SCHEDULING ====================
                if self.training_core.scheduler is not None:
                    if isinstance(self.training_core.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        # Use validation loss if available, otherwise training loss
                        metric_for_scheduler = val_metrics['loss'] if val_metrics is not None else train_metrics['loss']
                        self.training_core.scheduler.step(metric_for_scheduler)
                    else:
                        if epoch >= self.config.lr_warmup_epochs:
                            self.training_core.scheduler.step()

                # Current learning rate
                current_lr = self.training_core.optimizer.param_groups[0]['lr']
                logger.info(f"  Learning Rate: {current_lr:.6e}")

                # ==================== CHECKPOINTING ====================
                # Save periodic checkpoint
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"checkpoint_epoch_{epoch+1}.pt"
                    )
                    self.training_core.save_checkpoint(
                        checkpoint_path,
                        additional_info={
                            'train_metrics': train_metrics,
                            'val_metrics': val_metrics
                        }
                    )

                # Save best model based on validation loss (if available) or training loss
                current_loss = val_metrics['loss'] if val_metrics is not None else train_metrics['loss']
                if current_loss < self.training_core.best_loss:
                    self.training_core.best_loss = current_loss
                    best_model_path = os.path.join(
                        self.config.checkpoint_dir,
                        "best_model.pt"
                    )
                    self.training_core.save_checkpoint(
                        best_model_path,
                        additional_info={
                            'train_metrics': train_metrics,
                            'val_metrics': val_metrics,
                            'best_loss_type': 'validation' if val_metrics is not None else 'training'
                        }
                    )
                    loss_type = "validation" if val_metrics is not None else "training"
                    logger.info(f"✓ New best model saved with {loss_type} loss: {self.training_core.best_loss:.6f}")

            logger.info("=" * 80)
            logger.info("Training completed successfully!")
            logger.info(f"Best loss achieved: {self.training_core.best_loss:.6f}")
            logger.info("=" * 80)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save checkpoint on interruption
            interrupt_path = os.path.join(
                self.config.checkpoint_dir,
                "checkpoint_interrupted.pt"
            )
            self.training_core.save_checkpoint(interrupt_path)
            logger.info(f"Saved interrupted checkpoint to {interrupt_path}")
            raise
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            logger.exception("Full traceback:")
            raise

    
    def save_model(self, filepath: str):
        """
        Save trained model parameters.
        
        Args:
            filepath: Path to save model
        """
        self.training_core.save_checkpoint(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model parameters.
        
        Args:
            filepath: Path to model checkpoint
        """
        self.training_core.load_checkpoint(filepath)
        logger.info(f"Model loaded from {filepath}")

# ============================================================================
# Config YAML Schema Documentation
# ============================================================================

"""
Example config.yaml section for training:
-----------------------------------------

training:
  # Mode
  supervised: true  # true for supervised, false for unsupervised
  
  # Training hyperparameters
  n_epochs: 100
  batch_size: 256
  learning_rate: 1.0e-3
  weight_decay: 1.0e-5
  
  # Optimization
  grad_clip_norm: 1.0
  lr_scheduler: "cosine"  # "cosine", "step", "plateau", or null
  lr_warmup_epochs: 5
  
  # Logging and checkpointing
  log_interval: 100
  checkpoint_interval: 1
  checkpoint_dir: "./checkpoints"
  
  # Device
  device: "cuda"  # or "cpu"
  use_amp: false  # Automatic mixed precision

Note on loss weights:
--------------------
Unlike the previous version, we no longer need beta_vae, lambda_ddpm, lambda_class
in the config because the ELBO module computes the mathematically correct ELBO
which already has the proper weighting of all terms. The ELBO is:

SUPERVISED:
  ELBO = Reconstruction - VAE_KL - Diffusion_KL - Terminal_KL

UNSUPERVISED:
  ELBO = Reconstruction - CellType_KL - VAE_KL - Diffusion_KL - Terminal_KL

All terms are computed rigorously by the ELBO module and we maximize the ELBO
(minimize -ELBO) during training.
"""