"""
ELBO (Evidence Lower Bound) Module for scARKIDS
================================================

Implements ELBO computation for VAE-DDPM model in both supervised and unsupervised settings.

Mathematical Background:
-----------------------

SUPERVISED ELBO:
===============
L^sup(θ, ψ, φ) = E_{q_φ(z^(0)|x,b,c*)}[log p_θ(x|z^(0), b)]
                 - D_KL(q_φ(z^(0)|x, b, c*)||p(z^(0)|c*))
                 - Σ_{t=2}^T E_{q_φ(z^(0))} [D_KL(q(z^(t-1)|z^(t), z^(0))||p_ψ(z^(t-1)|z^(t), c*))]
                 - E_{q_φ(z^(0))} [D_KL(q(z^(T)|z^(0))||p(z^(T)))]

Components:
1. Reconstruction: log p_θ(x|z^(0), b) - data fidelity
2. VAE KL: D_KL(q_φ(z^(0)|x,b,c*)||p(z^(0)|c*)) - regularize encoder to match prior
3. Diffusion KL: train reverse process to denoise correctly
4. Terminal KL: ensure final latent reaches standard Gaussian

UNSUPERVISED ELBO:
==================
L^unsup(θ, ψ, φ, ω) = E_{q_φ(z^(0)|x,b,c)}[log p_θ(x|z^(0), b)]
                      - D_KL(q_ω(c|x, b)||p(c))
                      - E_{q_ω(c|x,b)} [D_KL(q_φ(z^(0)|x, b, c)||p(z^(0)))]
                      - Σ_{t=2}^T E_{q_φ,q_ω} [D_KL(q(z^(t-1)|z^(t), z^(0))||p_ψ(z^(t-1)|z^(t), c))]
                      - E_{q_φ(z^(0))} [D_KL(q(z^(T)|z^(0))||p(z^(T)))]

Components:
1. Reconstruction: Same as supervised
2. Cell Type KL: D_KL(q_ω(c|x,b)||p(c)) - regularize classifier to match prior
3. VAE KL: Regularize encoder to standard Gaussian (averaged over predicted cell types)
4. Diffusion KL: Same as supervised but conditioned on predicted c
5. Terminal KL: Same as supervised

KL Divergence Formulas:
-----------------------
1. Gaussian KL: D_KL(N(μ₁,Σ₁)||N(μ₂,Σ₂)) = 0.5 * [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - k + log(|Σ₂|/|Σ₁|)]
2. Categorical KL: D_KL(Cat(π₁)||Cat(π₂)) = Σ π₁ log(π₁/π₂)
"""

from src.utils.logger import Logger
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ELBOConfig:
    """
    Configuration for the ELBO module.
    
    Attributes:
        supervised: If True, use supervised ELBO (c* known); else unsupervised (c inferred)
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
# KL Divergence Utilities
# ============================================================================

def kl_divergence_gaussians(
    mu1: torch.Tensor,
    logvar1: torch.Tensor,
    mu2: torch.Tensor,
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two diagonal Gaussian distributions.
    
    D_KL(N(μ₁,diag(σ₁²))||N(μ₂,diag(σ₂²))) = 
        0.5 * Σ[σ₁²/σ₂² + (μ₂-μ₁)²/σ₂² - 1 - log(σ₁²/σ₂²)]
    
    Args:
        mu1: Mean of first distribution (batch_size, dim)
        logvar1: Log variance of first distribution (batch_size, dim)
        mu2: Mean of second distribution (batch_size, dim)
        logvar2: Log variance of second distribution (batch_size, dim)
    
    Returns:
        KL divergence (batch_size,)
    """
    # Convert log-variance to variance
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # Compute KL divergence components
    # Term 1: var1 / var2
    term1 = var1 / var2
    
    # Term 2: (mu2 - mu1)² / var2
    mu_diff = mu2 - mu1
    term2 = (mu_diff ** 2) / var2
    
    # Term 3: -1
    term3 = -1.0
    
    # Term 4: -log(var1 / var2) = log(var2) - log(var1)
    term4 = logvar2 - logvar1
    
    # Sum over dimensions and multiply by 0.5
    kl = 0.5 * torch.sum(term1 + term2 + term3 + term4, dim=-1)
    
    return kl


def kl_divergence_gaussian_standard(
    mu: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between Gaussian and standard normal N(0,I).
    
    D_KL(N(μ,diag(σ²))||N(0,I)) = 0.5 * Σ[σ² + μ² - 1 - log(σ²)]
    
    Args:
        mu: Mean (batch_size, dim)
        logvar: Log variance (batch_size, dim)
    
    Returns:
        KL divergence (batch_size,)
    """
    kl = 0.5 * torch.sum(
        torch.exp(logvar) + mu**2 - 1.0 - logvar,
        dim=-1
    )
    return kl


def kl_divergence_categorical(
    probs: torch.Tensor,
    prior_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two categorical distributions.
    
    D_KL(Cat(π)||Cat(π₀)) = Σ π * log(π/π₀)
    
    Args:
        probs: Predicted probabilities (batch_size, n_classes)
        prior_probs: Prior probabilities (n_classes,)
    
    Returns:
        KL divergence (batch_size,)
    """
    # Add small epsilon for numerical stability
    eps = 1e-8
    probs = torch.clamp(probs, min=eps, max=1-eps)
    prior_probs = torch.clamp(prior_probs, min=eps, max=1-eps)
    
    # Expand prior_probs for broadcasting
    prior_probs = prior_probs.unsqueeze(0)  # (1, n_classes)
    
    # Compute KL: Σ p * log(p/q)
    kl = torch.sum(probs * (torch.log(probs) - torch.log(prior_probs)), dim=-1)
    
    return kl

# ============================================================================
# ELBO Module
# ============================================================================

class ELBOModule(nn.Module):
    """
    ELBO computation module.
    
    Computes Evidence Lower Bound for both supervised and unsupervised settings.
    
    The module takes atomic objects (likelihood, prior, ddpm_backward, classifier, 
    variational_posterior) and computes the ELBO loss for training.
    """
    
    def __init__(
        self,
        config: ELBOConfig,
        likelihood_module,      # LikelihoodModule instance
        prior_module,           # PriorModule instance
        ddpm_backward_module,   # DDPMBackwardModule instance
        variational_posterior_module,  # VariationalPosteriorModule instance
        classifier_module=None  # ClassifierModule instance (required for unsupervised)
    ):
        """
        Initialize ELBO module.
        
        Args:
            config: Configuration object
            likelihood_module: Likelihood module instance
            prior_module: Prior module instance
            ddpm_backward_module: DDPM backward process module instance
            variational_posterior_module: Variational posterior module instance
            classifier_module: Classifier module instance (required if unsupervised)
        
        Raises:
            ValueError: If unsupervised mode but no classifier provided
        """
        super().__init__()
        self.config = config
        
        # Store atomic modules
        self.likelihood = likelihood_module
        self.prior = prior_module
        self.ddpm_backward = ddpm_backward_module
        self.variational_posterior = variational_posterior_module
        self.classifier = classifier_module
        
        # Validate configuration
        if not config.supervised and classifier_module is None:
            raise ValueError(
                "Unsupervised mode requires classifier_module. "
                "Please provide a trained classifier."
            )
        
        logger.info("Initialized ELBOModule")
        logger.info(f"  Mode: {'Supervised (c* known)' if config.supervised else 'Unsupervised (c inferred)'}")
        logger.info(f"  latent_dim={config.latent_dim}")
        logger.info(f"  n_diffusion_steps={config.n_diffusion_steps}")
        logger.info(f"  n_cell_types={config.n_cell_types}")
    
    def _validate_inputs(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None
    ) -> None:
        """
        Validate input tensors.
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
            library_size: Library size log(s) (batch_size,)
        
        Raises:
            ValueError: If inputs are invalid
        """
        batch_size = x.shape[0]
        
        # Check x
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (batch_size, n_genes), got shape {x.shape}")
        
        # Check batch_onehot
        if batch_onehot.dim() != 2:
            raise ValueError(f"batch_onehot must be 2D, got shape {batch_onehot.shape}")
        if batch_onehot.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: x={batch_size}, batch_onehot={batch_onehot.shape[0]}")
        
        # Check celltype_onehot (supervised mode)
        if self.config.supervised:
            if celltype_onehot is None:
                raise ValueError("Supervised mode requires celltype_onehot (ground truth c*)")
            if celltype_onehot.dim() != 2:
                raise ValueError(f"celltype_onehot must be 2D, got shape {celltype_onehot.shape}")
            if celltype_onehot.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: x={batch_size}, celltype_onehot={celltype_onehot.shape[0]}")
            if celltype_onehot.shape[1] != self.config.n_cell_types:
                raise ValueError(f"Cell type dimension mismatch: expected {self.config.n_cell_types}, got {celltype_onehot.shape[1]}")
        
        # Check library_size
        if library_size is not None:
            if library_size.dim() != 1:
                raise ValueError(f"library_size must be 1D, got shape {library_size.shape}")
            if library_size.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: x={batch_size}, library_size={library_size.shape[0]}")
        
        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("x contains NaN or Inf values")
        if torch.isnan(batch_onehot).any() or torch.isinf(batch_onehot).any():
            raise ValueError("batch_onehot contains NaN or Inf values")
        if celltype_onehot is not None:
            if torch.isnan(celltype_onehot).any() or torch.isinf(celltype_onehot).any():
                raise ValueError("celltype_onehot contains NaN or Inf values")
        if library_size is not None:
            if torch.isnan(library_size).any() or torch.isinf(library_size).any():
                raise ValueError("library_size contains NaN or Inf values")
    
    def compute_reconstruction_term(
        self,
        x: torch.Tensor,
        z_0: torch.Tensor,
        batch_onehot: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reconstruction term: E[log p_θ(x|z^(0), b)]
        
        This is the data fidelity term (same for supervised and unsupervised).
        
        Args:
            x: Observed counts (batch_size, n_genes)
            z_0: Initial latent (batch_size, latent_dim)
            batch_onehot: Batch indicator (batch_size, n_batches)
            library_size: Library size log(s) (batch_size,) or None
        
        Returns:
            log_p_x: Log likelihood (batch_size,)
            outputs: Dictionary with likelihood outputs
        """
        # Compute log p_θ(x|z^(0), b) using likelihood module
        nll_loss, likelihood_outputs = self.likelihood(
            z=z_0,
            batch_onehot=batch_onehot,
            x=x,
            library_size=library_size
        )
        
        # Extract per-sample log-likelihood
        # likelihood_outputs['log_likelihood'] has shape (batch_size, n_genes)
        log_p_x = likelihood_outputs['log_likelihood'].sum(dim=-1)  # (batch_size,)
        
        return log_p_x, likelihood_outputs
    
    def compute_vae_kl_supervised(
        self,
        z_0: torch.Tensor,
        encoder_mean: torch.Tensor,
        encoder_logvar: torch.Tensor,
        celltype_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VAE KL for supervised mode: D_KL(q_φ(z^(0)|x,b,c*)||p(z^(0)|c*))
        
        Both distributions are Gaussian with learned cell-type-specific priors.
        
        Args:
            z_0: Sampled latent (batch_size, latent_dim)
            encoder_mean: Encoder mean μ_φ (batch_size, latent_dim)
            encoder_logvar: Encoder log-variance log(σ²_φ) (batch_size, latent_dim)
            celltype_indices: Cell type indices (batch_size,)
        
        Returns:
            kl: KL divergence (batch_size,)
        """
        # Get cell-type-specific prior parameters
        # prior.vae_latent_prior.means: (n_cell_types, latent_dim)
        # prior.vae_latent_prior.log_vars: (n_cell_types, latent_dim)
        prior_means = self.prior.vae_latent_prior.means[celltype_indices]  # (batch_size, latent_dim)
        prior_logvars = self.prior.vae_latent_prior.log_vars[celltype_indices]  # (batch_size, latent_dim)
        
        # Compute KL divergence between two Gaussians
        kl = kl_divergence_gaussians(
            mu1=encoder_mean,
            logvar1=encoder_logvar,
            mu2=prior_means,
            logvar2=prior_logvars
        )
        
        return kl
    
    def compute_vae_kl_unsupervised(
        self,
        z_0: torch.Tensor,
        encoder_mean: torch.Tensor,
        encoder_logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VAE KL for unsupervised mode: D_KL(q_φ(z^(0)|x,b,c)||p(z^(0)))
        
        Encoder posterior vs. standard Gaussian prior (averaged over predicted cell types).
        
        Args:
            z_0: Sampled latent (batch_size, latent_dim)
            encoder_mean: Encoder mean μ_φ (batch_size, latent_dim)
            encoder_logvar: Encoder log-variance log(σ²_φ) (batch_size, latent_dim)
        
        Returns:
            kl: KL divergence (batch_size,)
        """
        # In unsupervised mode, prior is N(0, I)
        kl = kl_divergence_gaussian_standard(
            mu=encoder_mean,
            logvar=encoder_logvar
        )
        
        return kl
    
    def compute_celltype_kl(
        self,
        celltype_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cell type KL (unsupervised only): D_KL(q_ω(c|x,b)||p(c))
        
        Regularizes classifier to match prior distribution over cell types.
        
        Args:
            celltype_probs: Predicted cell type probabilities (batch_size, n_cell_types)
        
        Returns:
            kl: KL divergence (batch_size,)
        """
        # Get prior probabilities over cell types
        # prior.cell_type_prior.probabilities: (n_cell_types,)
        prior_probs = self.prior.cell_type_prior.probabilities
        
        # Compute KL divergence
        kl = kl_divergence_categorical(
            probs=celltype_probs,
            prior_probs=prior_probs
        )
        
        return kl
    
    def compute_diffusion_kl(
        self,
        z_t: torch.Tensor,
        z_0: torch.Tensor,
        # z_t_minus_1_from_forward: torch.Tensor,
        t: torch.Tensor,
        celltype_indices: torch.Tensor,
        variance_schedule: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute diffusion KL for timestep t: D_KL(q(z^(t-1)|z^(t), z^(0))||p_ψ(z^(t-1)|z^(t), c))

        This trains the reverse process to match the forward posterior.

        Forward posterior (analytic): q(z^(t-1)|z^(t), z^(0)) = N(μ̃_t, β̃_t I)
        Reverse process (learned): p_ψ(z^(t-1)|z^(t), c) = N(μ_ψ, β_t I)

        Args:
            z_t: Noisy latent at timestep t (batch_size, latent_dim)
            z_0: Clean latent (batch_size, latent_dim)
            z_t_minus_1_from_forward: Sampled z^(t-1) from forward posterior (batch_size, latent_dim)
            t: Timestep indices (batch_size,)
            celltype_indices: Cell type indices (batch_size,)
            variance_schedule: Variance schedule dictionary containing beta, alpha, alpha_cumprod

        Returns:
            kl: KL divergence (batch_size,)
        """
        device = z_t.device
        batch_size = z_t.shape[0]

        # ===================================================================
        # Compute reverse process mean using learned denoising network
        # μ_ψ = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε_ψ(z^(t), t, c))
        # ===================================================================
        reverse_mean = self.ddpm_backward.compute_mean(
            z_t=z_t,
            t=t,
            celltype=celltype_indices
        )  # (batch_size, latent_dim)

        # ===================================================================
        # Compute forward posterior mean (analytic)
        # μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) * z^(0) + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) * z^(t)
        # ===================================================================
        # Extract schedule components
        beta_t = variance_schedule['beta'][t].reshape(-1, 1)  # (batch_size, 1)
        alpha_t = variance_schedule['alpha'][t].reshape(-1, 1)  # (batch_size, 1)
        alpha_cumprod_t = variance_schedule['alpha_cumprod'][t].reshape(-1, 1)  # (batch_size, 1)

        # For t-1, handle boundary case (when t=0, use same values)
        t_minus_1 = torch.clamp(t - 1, min=0)
        alpha_cumprod_t_minus_1 = variance_schedule['alpha_cumprod'][t_minus_1].reshape(-1, 1)

        # Compute forward posterior mean
        # μ̃_t = (√ᾱ_{t-1} * β_t) / (1 - ᾱ_t) * z^(0) + (√α_t * (1 - ᾱ_{t-1})) / (1 - ᾱ_t) * z^(t)
        sqrt_alpha_cumprod_t_minus_1 = torch.sqrt(alpha_cumprod_t_minus_1)
        sqrt_alpha_t = torch.sqrt(alpha_t)

        coef_z0 = (sqrt_alpha_cumprod_t_minus_1 * beta_t) / (1.0 - alpha_cumprod_t)
        coef_zt = (sqrt_alpha_t * (1.0 - alpha_cumprod_t_minus_1)) / (1.0 - alpha_cumprod_t)

        forward_mean = coef_z0 * z_0 + coef_zt * z_t  # (batch_size, latent_dim)

        # ===================================================================
        # Compute variances
        # ===================================================================
        # Forward posterior variance: β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        forward_variance = ((1.0 - alpha_cumprod_t_minus_1) / (1.0 - alpha_cumprod_t)) * beta_t
        forward_logvar = torch.log(forward_variance.clamp(min=1e-20))  # (batch_size, 1)
        forward_logvar = forward_logvar.expand(-1, self.config.latent_dim)  # (batch_size, latent_dim)

        # Reverse variance (typically fixed as β_t)
        reverse_variance = self.ddpm_backward.compute_variance(
            z_t=z_t,
            t=t,
            celltype=celltype_indices
        )  # (batch_size, latent_dim) or (batch_size, 1)
        reverse_logvar = torch.log(reverse_variance.clamp(min=1e-20))
        if reverse_logvar.shape[-1] == 1:
            reverse_logvar = reverse_logvar.expand(-1, self.config.latent_dim)

        # ===================================================================
        # Compute KL divergence between forward and reverse distributions
        # ===================================================================
        kl = kl_divergence_gaussians(
            mu1=forward_mean,
            logvar1=forward_logvar,
            mu2=reverse_mean,
            logvar2=reverse_logvar
        )

        return kl
    
    def compute_terminal_kl(
        self,
        z_T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute terminal diffusion KL: D_KL(q(z^(T)|z^(0))||p(z^(T)))
        
        Ensures that the final latent reaches standard Gaussian.
        Both distributions are N(0, I) in practice, but we compute the KL
        based on the actual z^(T) sampled from forward process.
        
        Args:
            z_T: Terminal latent (batch_size, latent_dim)
        
        Returns:
            kl: KL divergence (batch_size,)
        """
        # Since both q(z^(T)|z^(0)) ≈ N(0, I) and p(z^(T)) = N(0, I),
        # the KL divergence is approximately 0. However, we compute it
        # based on the empirical distribution of z^(T).
        
        # In practice, this term is often negligible and can be approximated as 0
        # or computed using Monte Carlo estimation.
        
        # For simplicity, we compute the log probability under p(z^(T)) = N(0, I)
        # KL ≈ -log p(z^(T)) + constant
        log_p_z_T = self.prior.log_prob_terminal_diffusion(z_T)  # (batch_size,)
        
        # Since entropy of q(z^(T)|z^(0)) is approximately constant,
        # we use -log_p_z_T as a proxy for KL
        # Note: This is a simplification; exact KL would require computing entropy of q
        kl = -log_p_z_T
        
        return kl
    
    def compute_supervised_elbo(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute supervised ELBO.
        
        L^sup = E[log p(x|z^(0), b)] 
                - D_KL(q(z^(0)|x,b,c*)||p(z^(0)|c*))
                - Σ_{t=2}^T D_KL(q(z^(t-1)|z^(t),z^(0))||p(z^(t-1)|z^(t),c*))
                - D_KL(q(z^(T)|z^(0))||p(z^(T)))
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Ground truth cell type (batch_size, n_cell_types)
            library_size: Library size log(s) (batch_size,) or None
        
        Returns:
            elbo: ELBO value (scalar, negative for loss)
            metrics: Dictionary with individual terms
        """
        self._validate_inputs(x, batch_onehot, celltype_onehot, library_size)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Convert cell type one-hot to indices
        celltype_indices = torch.argmax(celltype_onehot, dim=-1)  # (batch_size,)
        
        # Sample from variational posterior: q_φ(z^(0:T)|x, b, c*)
        # This includes: encoder → z^(0), forward diffusion → z^(0:T)
        posterior_outputs = self.variational_posterior(
            x=x,
            batch_onehot=batch_onehot,
            celltype_onehot=celltype_onehot,
            t=None,  # Will sample random t internally
            sample_z0=True,
            return_all=True
        )
        
        z_0 = posterior_outputs['z_0']  # (batch_size, latent_dim)
        z_t = posterior_outputs['z_t']  # (batch_size, latent_dim)
        t = posterior_outputs['t']  # (batch_size,)
        encoder_mean = posterior_outputs['encoder_mean']  # (batch_size, latent_dim)
        encoder_logvar = posterior_outputs['encoder_logvar']  # (batch_size, latent_dim)
        
        # ===================================================================
        # Term 1: Reconstruction - E[log p_θ(x|z^(0), b)]
        # ===================================================================
        log_p_x, likelihood_outputs = self.compute_reconstruction_term(
            x=x,
            z_0=z_0,
            batch_onehot=batch_onehot,
            library_size=library_size
        )
        reconstruction = log_p_x.mean()  # Average over batch
        
        # ===================================================================
        # Term 2: VAE KL - D_KL(q_φ(z^(0)|x,b,c*)||p(z^(0)|c*))
        # ===================================================================
        vae_kl = self.compute_vae_kl_supervised(
            z_0=z_0,
            encoder_mean=encoder_mean,
            encoder_logvar=encoder_logvar,
            celltype_indices=celltype_indices
        )
        vae_kl_mean = vae_kl.mean()
        
        # ===================================================================
        # Term 3: Diffusion KL - Σ_{t=2}^T D_KL(q(z^(t-1)|z^(t),z^(0))||p_ψ(z^(t-1)|z^(t),c*))
        # ===================================================================
        # Note: In practice, we sample a single timestep t and compute KL for that step
        # The expectation over t is achieved through Monte Carlo sampling across batches
        
        diffusion_kl = torch.zeros(batch_size, device=device)
        
        # Only compute diffusion KL for t >= 1 (t=0 has no previous step)
        mask_valid_t = (t >= 1).float()  # (batch_size,)
        
        if mask_valid_t.sum() > 0:
            # Sample z^(t-1) from forward posterior for KL computation
            # z_t_minus_1_outputs = self.ddpm_backward.ddpm_forward.sample_timestep(
            #     z_0=z_0,
            #     t=t - 1
            # )
            # z_t_minus_1 = z_t_minus_1_outputs['z_t']

            variance_schedule = {
                    'beta': self.variational_posterior.ddpm_forward.beta,
                    'alpha': self.variational_posterior.ddpm_forward.alpha,
                    'alpha_cumprod': self.variational_posterior.ddpm_forward.alpha_cumprod,
                }
            
            # Compute KL for this timestep
            diffusion_kl_t = self.compute_diffusion_kl(
                z_t=z_t,
                z_0=z_0,
                # z_t_minus_1_from_forward=z_t_minus_1,
                t=t,
                celltype_indices=celltype_indices,
                variance_schedule=variance_schedule
            )
            
            # Apply mask (only count valid timesteps)
            diffusion_kl = diffusion_kl_t * mask_valid_t
        
        diffusion_kl_mean = diffusion_kl.mean()
        
        # ===================================================================
        # Term 4: Terminal KL - D_KL(q(z^(T)|z^(0))||p(z^(T)))
        # ===================================================================
        # Sample z^(T) from forward process using closed-form
        t_terminal = torch.full(
            (batch_size,), 
            self.config.n_diffusion_steps - 1, 
            dtype=torch.long, 
            device=device
        )

        # Use add_noise_closed_form to get z^(T) from z^(0)
        z_T, _ = self.variational_posterior.ddpm_forward.add_noise_closed_form(
            z_0=z_0,
            t=t_terminal
        )

        terminal_kl = self.compute_terminal_kl(z_T)
        terminal_kl_mean = terminal_kl.mean()

        # ===================================================================
        # Combine terms to get ELBO
        # ===================================================================
        # ELBO = Reconstruction - VAE_KL - Diffusion_KL - Terminal_KL
        elbo = reconstruction - vae_kl_mean - diffusion_kl_mean - terminal_kl_mean
        
        # For training, we want to maximize ELBO, which is equivalent to minimizing -ELBO
        loss = -elbo
        
        # Prepare metrics dictionary
        metrics = {
            'elbo': elbo.item(),
            'loss': loss.item(),
            'reconstruction': reconstruction.item(),
            'vae_kl': vae_kl_mean.item(),
            'diffusion_kl': diffusion_kl_mean.item(),
            'terminal_kl': terminal_kl_mean.item(),
            # Additional likelihood metrics
            'nll': -reconstruction.item(),
        }
        
        return loss, metrics
    
    def compute_unsupervised_elbo(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute unsupervised ELBO.
        
        L^unsup = E[log p(x|z^(0), b)]
                  - D_KL(q_ω(c|x,b)||p(c))
                  - E_c[D_KL(q_φ(z^(0)|x,b,c)||p(z^(0)))]
                  - Σ_{t=2}^T E_c[D_KL(q(z^(t-1)|z^(t),z^(0))||p_ψ(z^(t-1)|z^(t),c))]
                  - D_KL(q(z^(T)|z^(0))||p(z^(T)))
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            library_size: Library size log(s) (batch_size,) or None
        
        Returns:
            elbo: ELBO value (scalar, negative for loss)
            metrics: Dictionary with individual terms
        """
        self._validate_inputs(x, batch_onehot, None, library_size)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Infer cell type using classifier: q_ω(c|x, b)
        celltype_probs, celltype_onehot = self.variational_posterior.infer_cell_type(
            x=x,
            batch_onehot=batch_onehot
        )
        celltype_indices = torch.argmax(celltype_onehot, dim=-1)  # (batch_size,)
        
        # Sample from variational posterior: q_φ,ω(z^(0:T), c|x, b)
        posterior_outputs = self.variational_posterior(
            x=x,
            batch_onehot=batch_onehot,
            celltype_onehot=celltype_onehot,  # Use inferred cell type
            t=None,
            sample_z0=True,
            return_all=True
        )
        
        z_0 = posterior_outputs['z_0']
        z_t = posterior_outputs['z_t']
        t = posterior_outputs['t']
        encoder_mean = posterior_outputs['encoder_mean']
        encoder_logvar = posterior_outputs['encoder_logvar']
        
        # ===================================================================
        # Term 1: Reconstruction - E[log p_θ(x|z^(0), b)]
        # ===================================================================
        log_p_x, likelihood_outputs = self.compute_reconstruction_term(
            x=x,
            z_0=z_0,
            batch_onehot=batch_onehot,
            library_size=library_size
        )
        reconstruction = log_p_x.mean()
        
        # ===================================================================
        # Term 2: Cell Type KL - D_KL(q_ω(c|x,b)||p(c))
        # ===================================================================
        celltype_kl = self.compute_celltype_kl(celltype_probs)
        celltype_kl_mean = celltype_kl.mean()
        
        # ===================================================================
        # Term 3: VAE KL - E_c[D_KL(q_φ(z^(0)|x,b,c)||p(z^(0)))]
        # ===================================================================
        # In unsupervised mode, prior is N(0, I)
        vae_kl = self.compute_vae_kl_unsupervised(
            z_0=z_0,
            encoder_mean=encoder_mean,
            encoder_logvar=encoder_logvar
        )
        vae_kl_mean = vae_kl.mean()

        # ===================================================================
        # Term 4: Diffusion KL - Σ_{t=2}^T D_KL(q(z^(t-1)|z^(t),z^(0))||p_ψ(z^(t-1)|z^(t),c*))
        # ===================================================================
        # Note: In practice, we sample a single timestep t and compute KL for that step
        # The expectation over t is achieved through Monte Carlo sampling across batches
        
        diffusion_kl = torch.zeros(batch_size, device=device)
        
        # Only compute diffusion KL for t >= 1 (t=0 has no previous step)
        mask_valid_t = (t >= 1).float()  # (batch_size,)
        
        if mask_valid_t.sum() > 0:
            # Sample z^(t-1) from forward posterior for KL computation
            # z_t_minus_1_outputs = self.ddpm_backward.ddpm_forward.sample_timestep(
            #     z_0=z_0,
            #     t=t - 1
            # )
            # z_t_minus_1 = z_t_minus_1_outputs['z_t']

            variance_schedule = {
                    'beta': self.variational_posterior.ddpm_forward.beta,
                    'alpha': self.variational_posterior.ddpm_forward.alpha,
                    'alpha_cumprod': self.variational_posterior.ddpm_forward.alpha_cumprod,
                }
            
            # Compute KL for this timestep
            diffusion_kl_t = self.compute_diffusion_kl(
                z_t=z_t,
                z_0=z_0,
                # z_t_minus_1_from_forward=z_t_minus_1,
                t=t,
                celltype_indices=celltype_indices,
                variance_schedule=variance_schedule
            )
            
            # Apply mask (only count valid timesteps)
            diffusion_kl = diffusion_kl_t * mask_valid_t
        
        diffusion_kl_mean = diffusion_kl.mean()
        
        # ===================================================================
        # Term 5: Terminal KL - D_KL(q(z^(T)|z^(0))||p(z^(T)))
        # ===================================================================
        t_terminal = torch.full((batch_size,), self.config.n_diffusion_steps - 1, dtype=torch.long, device=device)
        z_T, _ = self.variational_posterior.ddpm_forward.add_noise_closed_form(
            z_0=z_0,
            t=t_terminal
        )
        
        terminal_kl = self.compute_terminal_kl(z_T)
        terminal_kl_mean = terminal_kl.mean()
        
        # ===================================================================
        # Combine terms to get ELBO
        # ===================================================================
        elbo = reconstruction - celltype_kl_mean - vae_kl_mean - diffusion_kl_mean - terminal_kl_mean
        loss = -elbo
        
        # Prepare metrics
        metrics = {
            'elbo': elbo.item(),
            'loss': loss.item(),
            'reconstruction': reconstruction.item(),
            'celltype_kl': celltype_kl_mean.item(),
            'vae_kl': vae_kl_mean.item(),
            'diffusion_kl': diffusion_kl_mean.item(),
            'terminal_kl': terminal_kl_mean.item(),
            'nll': -reconstruction.item(),
        }
        
        return loss, metrics
    
    def forward(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ELBO (unified interface for both modes).
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
                            Required for supervised, optional for unsupervised
            library_size: Library size log(s) (batch_size,) or None
        
        Returns:
            loss: Negative ELBO (scalar, for minimization)
            metrics: Dictionary with individual terms and metrics
        """
        if self.config.supervised:
            return self.compute_supervised_elbo(
                x=x,
                batch_onehot=batch_onehot,
                celltype_onehot=celltype_onehot,
                library_size=library_size
            )
        else:
            return self.compute_unsupervised_elbo(
                x=x,
                batch_onehot=batch_onehot,
                library_size=library_size
            )

# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class ELBOManager:
    """
    Manager class for the ELBO module.
    
    This is the single entry point that:
    1. Takes atomic objects (likelihood, prior, ddpm_backward, classifier, variational_posterior)
    2. Initializes the ELBO module
    3. Exposes APIs for training/inference
    """
    
    def __init__(
        self,
        likelihood_manager,           # LikelihoodManager instance
        prior_manager,                 # PriorManager instance
        ddpm_backward_manager,         # DDPMBackwardManager instance
        variational_posterior_manager, # VariationalPosteriorManager instance
        classifier_manager=None        # ClassifierManager instance (optional)
    ):
        """
        Initialize manager from atomic objects.
        
        Args:
            likelihood_manager: LikelihoodManager instance
            prior_manager: PriorManager instance
            ddpm_backward_manager: DDPMBackwardManager instance
            variational_posterior_manager: VariationalPosteriorManager instance
            classifier_manager: ClassifierManager instance (required if unsupervised)
        """
        logger.info("Initializing ELBOManager")
        
        # Get atomic modules
        self.likelihood_module = likelihood_manager.get_module()
        self.prior_module = prior_manager.get_module()
        self.ddpm_backward_module = ddpm_backward_manager.get_module()
        self.variational_posterior_module = variational_posterior_manager.get_module()
        
        self.classifier_module = None
        if classifier_manager is not None:
            self.classifier_module = classifier_manager.get_module()
        
        # Determine mode from variational posterior
        supervised = variational_posterior_manager.supervised
        
        # Create configuration
        self.config = ELBOConfig(
            supervised=supervised,
            latent_dim=self.variational_posterior_module.config.latent_dim,
            n_diffusion_steps=self.variational_posterior_module.config.n_diffusion_steps,
            n_cell_types=self.variational_posterior_module.config.n_cell_types
        )
        
        # Initialize ELBO module
        self.elbo_module = ELBOModule(
            config=self.config,
            likelihood_module=self.likelihood_module,
            prior_module=self.prior_module,
            ddpm_backward_module=self.ddpm_backward_module,
            variational_posterior_module=self.variational_posterior_module,
            classifier_module=self.classifier_module
        )
        
        logger.info("ELBOManager initialized successfully")
    
    def get_module(self) -> ELBOModule:
        """
        Get the ELBO module.
        
        Returns:
            ELBOModule instance
        """
        return self.elbo_module
    
    def compute_loss(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ELBO loss (main API for training).
        
        Args:
            x: Gene expression (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
            library_size: Library size log(s) (batch_size,) or None
        
        Returns:
            loss: Negative ELBO (scalar, for minimization)
            metrics: Dictionary with individual terms and metrics
        """
        return self.elbo_module(
            x=x,
            batch_onehot=batch_onehot,
            celltype_onehot=celltype_onehot,
            library_size=library_size
        )
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters from all modules.
        
        Returns:
            Dictionary of named parameters
        """
        params = {}
        
        # Likelihood parameters
        for name, param in self.likelihood_module.named_parameters():
            params[f'likelihood.{name}'] = param
        
        # Prior parameters (if learnable)
        for name, param in self.prior_module.named_parameters():
            params[f'prior.{name}'] = param
        
        # DDPM backward parameters
        for name, param in self.ddpm_backward_module.named_parameters():
            params[f'ddpm_backward.{name}'] = param
        
        # Variational posterior parameters (encoder + optional classifier)
        # These are already included through variational_posterior_manager
        # But we add them here for completeness
        for name, param in self.variational_posterior_module.named_parameters():
            params[f'variational_posterior.{name}'] = param
        
        return params



