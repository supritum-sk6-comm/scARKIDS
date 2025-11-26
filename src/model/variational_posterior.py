"""
Variational Posterior Module for scARKIDS
==========================================

Implements the approximate variational posterior q_φ for VAE-DDPM model.

Mathematical Background:
-----------------------

SUPERVISED SETTING (c*_bn known):
---------------------------------
Variational posterior over latents z^(0:T):

q_φ(z_bn^(0:T) | x_bn, b, c*_bn) = q_φ(z_bn^(0) | x_bn, b, c*_bn) · ∏_{t=1}^{T} q(z_bn^(t) | z_bn^(t-1))

Components:
1. VAE Encoder (learnable, parameters φ):
   q_φ(z^(0) | x, b, c*) = N(z^(0) | μ_φ(x, b, c*), diag(σ²_φ(x, b, c*)))

2. Forward Diffusion (fixed):
   q(z^(t) | z^(t-1)) = N(z^(t) | √(1-β_t) z^(t-1), β_t I)

Reparameterization Trick:
z^(0) = μ_φ(x, b, c*) + σ_φ(x, b, c*) ⊙ ε,  ε ~ N(0, I)


UNSUPERVISED SETTING (c_bn unknown):
------------------------------------
Variational posterior over latents z^(0:T) and cell type c:

q_φ,ω(z_bn^(0:T), c_bn | x_bn, b) = 
    q_ω(c_bn | x_bn, b) · q_φ(z_bn^(0) | x_bn, b, c_bn) · ∏_{t=1}^{T} q(z_bn^(t) | z_bn^(t-1))

Components:
1. Cell Type Classifier (learnable, parameters ω):
   q_ω(c | x, b) = Categorical(c | π_ω(x, b))

2. VAE Encoder (learnable, parameters φ):
   q_φ(z^(0) | x, b, c) = N(z^(0) | μ_φ(x, b, c), diag(σ²_φ(x, b, c)))

3. Forward Diffusion (fixed):
   q(z^(t) | z^(t-1)) = N(z^(t) | √(1-β_t) z^(t-1), β_t I)


Key Responsibilities:
--------------------
- Compute z^(0) from encoder (with/without cell type classifier)
- Apply forward diffusion to get z^(0:T)
- Provide log probability computation for ELBO
- Handle both supervised and unsupervised modes seamlessly

Where:
- x: Gene expression vector (n_genes,)
- b: Batch indicator (one-hot, n_batches)
- c or c*: Cell type indicator (one-hot, n_cell_types)
- z^(t): Latent at diffusion step t (latent_dim,)
- μ_φ, σ²_φ: Encoder mean and variance networks
- π_ω: Classifier probability network
- β_t: Fixed diffusion variance schedule
"""

from src.utils.logger import Logger
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VariationalPosteriorConfig:
    """
    Configuration for the Variational Posterior module.
    
    Attributes:
        supervised: If True, use supervised mode (c* known); else unsupervised (c inferred)
        latent_dim: Dimension of latent space z^(0)
        n_diffusion_steps: Number of diffusion timesteps T
        n_cell_types: Number of cell types
    """
    supervised: bool
    latent_dim: int
    n_diffusion_steps: int
    n_cell_types: int
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.n_diffusion_steps > 0, "n_diffusion_steps must be positive"
        assert self.n_cell_types > 0, "n_cell_types must be positive"

# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)

# ============================================================================
# Variational Posterior Module
# ============================================================================

class VariationalPosteriorModule(nn.Module):
    """
    Variational posterior q_φ (supervised) or q_φ,ω (unsupervised).
    
    This module orchestrates:
    1. Cell type inference (if unsupervised)
    2. Encoding to z^(0)
    3. Forward diffusion to z^(0:T)
    
    It provides unified interface for both supervised/unsupervised settings.
    """
    
    def __init__(
        self,
        config: VariationalPosteriorConfig,
        encoder_module,  # VAEEncoder instance
        ddpm_forward_module,  # DDPMForwardModule instance
        classifier_module=None  # Optional: ClassifierModule instance (required if unsupervised)
    ):
        """
        Initialize variational posterior module.
        
        Args:
            config: Configuration object
            encoder_module: Encoder module instance (VAEEncoder)
            ddpm_forward_module: DDPM forward process module (DDPMForwardModule)
            classifier_module: Cell type classifier module (required if unsupervised)
        
        Raises:
            ValueError: If unsupervised mode but no classifier provided
        """
        super().__init__()
        
        self.config = config
        self.encoder = encoder_module
        self.ddpm_forward = ddpm_forward_module
        self.classifier = classifier_module
        
        # Validate configuration
        if not config.supervised and classifier_module is None:
            raise ValueError(
                "Unsupervised mode requires classifier_module. "
                "Please provide a trained classifier."
            )
        
        logger.info("Initialized VariationalPosteriorModule")
        logger.info(f"  Mode: {'Supervised (c* known)' if config.supervised else 'Unsupervised (c inferred)'}")
        logger.info(f"  latent_dim={config.latent_dim}")
        logger.info(f"  n_diffusion_steps={config.n_diffusion_steps}")
        logger.info(f"  n_cell_types={config.n_cell_types}")
    
    def _validate_inputs(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: Optional[torch.Tensor] = None
    ) -> None:
        """
        Validate input tensors.
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types), required if supervised
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Check dimensions
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (batch_size, n_genes), got shape {x.shape}")
        
        if batch_onehot.dim() != 2:
            raise ValueError(f"batch_onehot must be 2D (batch_size, n_batches), got shape {batch_onehot.shape}")
        
        # Check batch size consistency
        batch_size = x.shape[0]
        if batch_onehot.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch. x: {batch_size}, batch_onehot: {batch_onehot.shape[0]}"
            )
        
        # Supervised mode: cell type must be provided
        if self.config.supervised:
            if celltype_onehot is None:
                raise ValueError(
                    "Supervised mode requires celltype_onehot. "
                    "Ground truth cell types (c*) must be provided."
                )
            
            if celltype_onehot.dim() != 2:
                raise ValueError(
                    f"celltype_onehot must be 2D (batch_size, n_cell_types), "
                    f"got shape {celltype_onehot.shape}"
                )
            
            if celltype_onehot.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch. x: {batch_size}, celltype_onehot: {celltype_onehot.shape[0]}"
                )
            
            if celltype_onehot.shape[1] != self.config.n_cell_types:
                raise ValueError(
                    f"Cell type dimension mismatch. Expected {self.config.n_cell_types}, "
                    f"got {celltype_onehot.shape[1]}"
                )
        
        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("x contains NaN or Inf values")
        
        if torch.isnan(batch_onehot).any() or torch.isinf(batch_onehot).any():
            raise ValueError("batch_onehot contains NaN or Inf values")
        
        if celltype_onehot is not None:
            if torch.isnan(celltype_onehot).any() or torch.isinf(celltype_onehot).any():
                raise ValueError("celltype_onehot contains NaN or Inf values")
    
    def infer_cell_type(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infer cell type using classifier (unsupervised mode).
        
        Mathematical formulation:
        q_ω(c | x, b) = Categorical(c | π_ω(x, b))
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
        
        Returns:
            celltype_probs: Cell type probabilities π_ω(x, b), shape (batch_size, n_cell_types)
            celltype_onehot: Sampled cell type (one-hot), shape (batch_size, n_cell_types)
        
        Raises:
            RuntimeError: If classifier not available (supervised mode)
        """
        if self.classifier is None:
            raise RuntimeError(
                "Cell type inference requires classifier. "
                "This should only be called in unsupervised mode."
            )
        
        # Get cell type probabilities from classifier
        batch_indices = batch_onehot.argmax(dim=1)
        celltype_probs = self.classifier(x, batch_indices)
        
        # Sample cell type from categorical distribution
        # c ~ Categorical(π_ω(x, b))
        celltype_dist = torch.distributions.Categorical(probs=celltype_probs)
        celltype_indices = celltype_dist.sample()  # (batch_size,)
        
        # Convert to one-hot encoding
        celltype_onehot = F.one_hot(celltype_indices, num_classes=self.config.n_cell_types).float()
        
        return celltype_probs, celltype_onehot
    
    def encode_to_z0(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: torch.Tensor,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to initial latent z^(0) using VAE encoder.
        
        Mathematical formulation:
        q_φ(z^(0) | x, b, c) = N(z^(0) | μ_φ(x, b, c), diag(σ²_φ(x, b, c)))
        
        Reparameterization:
        z^(0) = μ_φ(x, b, c) + σ_φ(x, b, c) ⊙ ε,  ε ~ N(0, I)
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
            sample: If True, sample using reparameterization; else return mean
        
        Returns:
            z_0: Initial latent representation (batch_size, latent_dim)
            mean: Encoder mean μ_φ (batch_size, latent_dim)
            logvar: Encoder log-variance log(σ²_φ) (batch_size, latent_dim)
        """
        # Encode to latent parameters
        mean, logvar = self.encoder.encode(x, batch_onehot, celltype_onehot)
        
        # Sample or use mean
        if sample:
            z_0 = self.encoder.reparameterize(mean, logvar)
        else:
            z_0 = mean  # Deterministic encoding (e.g., for inference)
        
        return z_0, mean, logvar
    
    def apply_forward_diffusion(
        self,
        z_0: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process to get z^(t).
        
        Mathematical formulation (closed-form):
        q(z^(t) | z^(0)) = N(z^(t) | √ᾱ_t z^(0), (1-ᾱ_t) I)
        
        Implementation:
        z^(t) = √ᾱ_t * z^(0) + √(1-ᾱ_t) * ε,  ε ~ N(0, I)
        
        Args:
            z_0: Initial latent (batch_size, latent_dim)
            t: Timestep indices (batch_size,). If None, sample randomly
        
        Returns:
            z_t: Noised latent at step t (batch_size, latent_dim)
            noise: Sampled Gaussian noise (batch_size, latent_dim)
            t: Timestep indices (batch_size,)
        """
        # Sample timesteps if not provided
        if t is None:
            t = self.ddpm_forward.sample_random_timesteps(z_0.shape[0])
        
        # Apply forward diffusion (closed-form)
        z_t, noise = self.ddpm_forward.add_noise_closed_form(z_0, t)
        
        return z_t, noise, t
    
    def forward(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        sample_z0: bool = True,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through variational posterior.
        
        Pipeline:
        1. Infer cell type (if unsupervised)
        2. Encode to z^(0)
        3. Apply forward diffusion to get z^(t)
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
                             Required if supervised; inferred if unsupervised
            t: Timestep indices (batch_size,). If None, sample randomly
            sample_z0: If True, sample z^(0) using reparameterization; else use mean
            return_all: If True, return all intermediate values; else minimal output
        
        Returns:
            Dictionary containing:
                - 'z_0': Initial latent (batch_size, latent_dim)
                - 'z_t': Noised latent at step t (batch_size, latent_dim)
                - 'noise': Sampled noise (batch_size, latent_dim)
                - 't': Timestep indices (batch_size,)
                - 'encoder_mean': μ_φ (batch_size, latent_dim) [if return_all]
                - 'encoder_logvar': log(σ²_φ) (batch_size, latent_dim) [if return_all]
                - 'celltype_onehot': Cell type (batch_size, n_cell_types) [if return_all]
                - 'celltype_probs': Cell type probabilities (batch_size, n_cell_types) 
                                    [if unsupervised and return_all]
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        self._validate_inputs(x, batch_onehot, celltype_onehot)
        
        # Step 1: Get cell type (infer if unsupervised, use provided if supervised)
        celltype_probs = None
        
        if self.config.supervised:
            # Supervised: use provided ground truth c*
            celltype_used = celltype_onehot
            logger.debug("Using ground truth cell type (supervised mode)")
        
        else:
            # Unsupervised: infer cell type c using classifier
            celltype_probs, celltype_used = self.infer_cell_type(x, batch_onehot)
            logger.debug("Inferred cell type using classifier (unsupervised mode)")
        
        # Step 2: Encode to z^(0) using VAE encoder
        z_0, encoder_mean, encoder_logvar = self.encode_to_z0(
            x, batch_onehot, celltype_used, sample=sample_z0
        )
        
        # Step 3: Apply forward diffusion to get z^(t)
        z_t, noise, t = self.apply_forward_diffusion(z_0, t)
        
        # Prepare output
        output = {
            'z_0': z_0,
            'z_t': z_t,
            'noise': noise,
            't': t,
        }
        
        if return_all:
            output.update({
                'encoder_mean': encoder_mean,
                'encoder_logvar': encoder_logvar,
                'celltype_onehot': celltype_used,
            })
            
            # Add cell type probabilities if unsupervised
            if not self.config.supervised:
                output['celltype_probs'] = celltype_probs
        
        return output
    
    def log_prob_z0(
        self,
        z_0: torch.Tensor,
        encoder_mean: torch.Tensor,
        encoder_logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of z^(0) under encoder posterior.
        
        Mathematical formulation:
        log q_φ(z^(0) | x, b, c) = log N(z^(0) | μ_φ, σ²_φ)
        
        Args:
            z_0: Sampled initial latent (batch_size, latent_dim)
            encoder_mean: Encoder mean μ_φ (batch_size, latent_dim)
            encoder_logvar: Encoder log-variance log(σ²_φ) (batch_size, latent_dim)
        
        Returns:
            log_prob: Log probability, shape (batch_size,)
        """
        # Compute Gaussian log probability
        # log N(z | μ, σ²) = -0.5 * [log(2π) + log(σ²) + (z-μ)²/σ²]
        
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=z_0.device))
        
        # (z - μ)² / σ²
        squared_diff = (z_0 - encoder_mean) ** 2
        var = torch.exp(encoder_logvar)
        normalized_squared_diff = squared_diff / var
        
        # log σ² = logvar
        log_prob_per_dim = -0.5 * (log_2pi + encoder_logvar + normalized_squared_diff)
        
        # Sum over latent dimensions
        log_prob = log_prob_per_dim.sum(dim=-1)  # (batch_size,)
        
        return log_prob
    
    def log_prob_zt_given_z0(
        self,
        z_t: torch.Tensor,
        z_0: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of z^(t) given z^(0) under forward diffusion.
        
        Mathematical formulation:
        log q(z^(t) | z^(0)) = log N(z^(t) | √ᾱ_t z^(0), (1-ᾱ_t) I)
        
        Args:
            z_t: Noised latent at step t (batch_size, latent_dim)
            z_0: Initial latent (batch_size, latent_dim)
            t: Timestep indices (batch_size,)
        
        Returns:
            log_prob: Log probability, shape (batch_size,)
        """
        # Get schedule components at timestep t
        schedule = self.ddpm_forward.get_schedule_at_timestep(t)
        sqrt_alpha_cumprod = schedule['sqrt_alpha_cumprod'].view(-1, 1)  # (batch_size, 1)
        alpha_cumprod = schedule['alpha_cumprod'].view(-1, 1)  # (batch_size, 1)
        
        # Mean: √ᾱ_t * z^(0)
        mean = sqrt_alpha_cumprod * z_0
        
        # Variance: (1 - ᾱ_t) * I
        var = (1.0 - alpha_cumprod)
        
        # Compute Gaussian log probability
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=z_t.device))
        
        squared_diff = (z_t - mean) ** 2
        normalized_squared_diff = squared_diff / var
        
        log_prob_per_dim = -0.5 * (log_2pi + torch.log(var) + normalized_squared_diff)
        
        # Sum over latent dimensions
        log_prob = log_prob_per_dim.sum(dim=-1)  # (batch_size,)
        
        return log_prob
    
    def log_prob_celltype(
        self,
        celltype_onehot: torch.Tensor,
        celltype_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of cell type under classifier (unsupervised only).
        
        Mathematical formulation:
        log q_ω(c | x, b) = log Categorical(c | π_ω(x, b))
        
        Args:
            celltype_onehot: Cell type (one-hot), shape (batch_size, n_cell_types)
            celltype_probs: Cell type probabilities π_ω, shape (batch_size, n_cell_types)
        
        Returns:
            log_prob: Log probability, shape (batch_size,)
        """
        # Extract probability of the selected cell type
        # celltype_onehot is one-hot: [0, 0, 1, 0, ...] for class 2
        # celltype_probs is probability vector: [0.1, 0.2, 0.6, 0.1, ...]
        
        # Select probability corresponding to one-hot encoding
        selected_prob = (celltype_onehot * celltype_probs).sum(dim=-1)  # (batch_size,)
        
        # Log probability
        log_prob = torch.log(selected_prob + 1e-8)  # Add epsilon for numerical stability
        
        return log_prob

# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class VariationalPosteriorManager:
    """
    Manager class for the Variational Posterior module.
    
    This is the single entry point that:
    1. Takes encoder and ddpm_forward atomic objects
    2. Optionally takes classifier object (for unsupervised mode)
    3. Initializes the variational posterior module
    4. Exposes APIs for training/inference
    """
    
    def __init__(
        self,
        encoder_manager,  # EncoderManager instance
        ddpm_forward_manager,  # DDPMForwardManager instance
        classifier_manager=None,  # Optional: ClassifierManager instance
        supervised: bool = True
    ):
        """
        Initialize manager from atomic objects.
        
        Args:
            encoder_manager: EncoderManager instance
            ddpm_forward_manager: DDPMForwardManager instance
            classifier_manager: ClassifierManager instance (required if unsupervised)
            supervised: If True, supervised mode (c* known); else unsupervised (c inferred)
        
        Raises:
            ValueError: If unsupervised mode but no classifier provided
        """
        logger.info("Initializing VariationalPosteriorManager")
        
        self.supervised = supervised
        
        # Get atomic modules
        self.encoder_module = encoder_manager.get_module()
        self.ddpm_forward_module = ddpm_forward_manager.get_module()
        
        self.classifier_module = None
        if not supervised:
            if classifier_manager is None:
                raise ValueError(
                    "Unsupervised mode requires classifier_manager. "
                    "Please provide a trained classifier."
                )
            self.classifier_module = classifier_manager.get_module()
        
        # Create configuration
        self.config = VariationalPosteriorConfig(
            supervised=supervised,
            latent_dim=self.encoder_module.config.latent_dim,
            n_diffusion_steps=self.ddpm_forward_module.config.n_diffusion_steps,
            n_cell_types=self.encoder_module.config.n_cell_types
        )
        
        # Initialize variational posterior module
        self.variational_posterior_module = VariationalPosteriorModule(
            config=self.config,
            encoder_module=self.encoder_module,
            ddpm_forward_module=self.ddpm_forward_module,
            classifier_module=self.classifier_module
        )
        
        logger.info("VariationalPosteriorManager initialized successfully")
    
    def get_module(self) -> VariationalPosteriorModule:
        """
        Get the variational posterior module.
        
        Returns:
            VariationalPosteriorModule instance
        """
        return self.variational_posterior_module
    
    def sample(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from variational posterior (main API for training).
        
        This is the primary interface for training the model.
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
                            Required if supervised; inferred if unsupervised
            t: Timestep indices (batch_size,). If None, sample randomly
            return_all: If True, return all intermediate values
        
        Returns:
            Dictionary containing sampled latents and parameters (see forward() docs)
        """
        return self.variational_posterior_module(
            x=x,
            batch_onehot=batch_onehot,
            celltype_onehot=celltype_onehot,
            t=t,
            sample_z0=True,  # Always sample during training
            return_all=return_all
        )
    
    def encode_mean(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode using mean (deterministic, for inference).
        
        This is useful for generating embeddings or deterministic inference.
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
            t: Timestep indices (batch_size,)
        
        Returns:
            Dictionary containing encoded latents (see forward() docs)
        """
        return self.variational_posterior_module(
            x=x,
            batch_onehot=batch_onehot,
            celltype_onehot=celltype_onehot,
            t=t,
            sample_z0=False,  # Use mean for deterministic encoding
            return_all=True
        )
    
    def compute_log_prob(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute log probabilities for ELBO computation.
        
        Args:
            outputs: Dictionary from sample() or encode_mean() containing:
                - 'z_0', 'z_t', 't', 'encoder_mean', 'encoder_logvar'
                - 'celltype_onehot', 'celltype_probs' (if unsupervised)
        
        Returns:
            Dictionary containing:
                - 'log_q_z0': log q_φ(z^(0) | x, b, c) (batch_size,)
                - 'log_q_zt_given_z0': log q(z^(t) | z^(0)) (batch_size,)
                - 'log_q_c': log q_ω(c | x, b) (batch_size,) [if unsupervised]
        """
        log_probs = {}
        
        # Log probability of z^(0) under encoder posterior
        log_probs['log_q_z0'] = self.variational_posterior_module.log_prob_z0(
            z_0=outputs['z_0'],
            encoder_mean=outputs['encoder_mean'],
            encoder_logvar=outputs['encoder_logvar']
        )
        
        # Log probability of z^(t) given z^(0) under forward diffusion
        log_probs['log_q_zt_given_z0'] = self.variational_posterior_module.log_prob_zt_given_z0(
            z_t=outputs['z_t'],
            z_0=outputs['z_0'],
            t=outputs['t']
        )
        
        # Log probability of cell type (unsupervised only)
        if not self.supervised and 'celltype_probs' in outputs:
            log_probs['log_q_c'] = self.variational_posterior_module.log_prob_celltype(
                celltype_onehot=outputs['celltype_onehot'],
                celltype_probs=outputs['celltype_probs']
            )
        
        return log_probs
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Dictionary of named parameters from encoder (and classifier if unsupervised)
        """
        params = {}
        
        # Encoder parameters (φ)
        for name, param in self.encoder_module.named_parameters():
            params[f'encoder.{name}'] = param
        
        # Classifier parameters (ω) if unsupervised
        if not self.supervised and self.classifier_module is not None:
            for name, param in self.classifier_module.named_parameters():
                params[f'classifier.{name}'] = param
        
        # DDPM forward has no parameters (fixed process)
        
        return params


# ============================================================================
# Usage Documentation
# ============================================================================

"""
Example usage in main.py:
--------------------------

```python
import yaml
from src.model.encoder import EncoderManager
from src.model.ddpm_forward import DDPMForwardManager
from src.model.classifier import ClassifierManager  # For unsupervised mode
from src.model.variational_posterior import VariationalPosteriorManager

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize atomic objects
encoder_manager = EncoderManager(config['encoder'])
ddpm_forward_manager = DDPMForwardManager(config['ddpm_forward'])

# For supervised mode:
vp_manager = VariationalPosteriorManager(
    encoder_manager=encoder_manager,
    ddpm_forward_manager=ddpm_forward_manager,
    supervised=True
)

# For unsupervised mode:
classifier_manager = ClassifierManager(config['classifier'])
vp_manager = VariationalPosteriorManager(
    encoder_manager=encoder_manager,
    ddpm_forward_manager=ddpm_forward_manager,
    classifier_manager=classifier_manager,
    supervised=False
)

# Get module for training
vp_module = vp_manager.get_module()

# Sample from variational posterior (training)
outputs = vp_manager.sample(
    x=gene_expression,
    batch_onehot=batch_indicator,
    celltype_onehot=celltype_gt,  # Only for supervised; None for unsupervised
    t=None,  # Will sample random timesteps
    return_all=True
)

# Outputs contain: z_0, z_t, noise, t, encoder_mean, encoder_logvar, celltype_onehot, etc.

# Compute log probabilities for ELBO
log_probs = vp_manager.compute_log_prob(outputs)

# log_probs contain: log_q_z0, log_q_zt_given_z0, log_q_c (if unsupervised)

# Get parameters for optimization
params = vp_manager.get_parameters()
```

Integration with ELBO:
----------------------

The variational posterior provides the necessary log probabilities for ELBO computation:

ELBO (Supervised):
L = E_{q_φ}[log p(x|z^(0)) + log p(z^(0)|c*) - log q_φ(z^(0)|x,b,c*)]

ELBO (Unsupervised):
L = E_{q_φ,ω}[log p(x|z^(0)) + log p(z^(0)|c) + log p(c) - log q_ω(c|x,b) - log q_φ(z^(0)|x,b,c)]

The VariationalPosteriorManager provides:
- log q_φ(z^(0)|x,b,c) via log_prob_z0()
- log q_ω(c|x,b) via log_prob_celltype() [unsupervised only]
- Sampled z^(0), c for expectation computation

The ELBO module will combine these with likelihood and prior log probabilities.
"""
