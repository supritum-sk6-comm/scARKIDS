"""
scARKIDS DDPM Forward Process Module

Implements the forward (noising) diffusion process for both supervised and
unsupervised VAE-DDPM models. Handles variable gene sets across batches via
union gene set masking.

Mathematical Framework:

Forward Process:
  q(z^(1:T) | z^(0)) = ∏_{t=1}^{T} q(z^(t) | z^(t-1))

Single Step (t → t+1):
  q(z^(t) | z^(t-1)) = N(z^(t) | √(1 - β_t) z^(t-1), β_t I)

Closed-Form (direct sampling from z^(0) to z^(t)):
  q(z^(t) | z^(0)) = N(z^(t) | √(ᾱ_t) z^(0), (1 - ᾱ_t) I)

Where:
  β_t: Variance schedule at timestep t
  ᾱ_t: Cumulative product of (1 - β_t)
  1̄ - ᾱ_t: Cumulative variance

Key: Forward process is FIXED (no learnable parameters).

Variance Schedule Strategies:
  - Linear: β_t = β_min + t/T * (β_max - β_min)
  - Cosine: β_t parameterized by cosine schedule
  - Quadratic: β_t parameterized quadratically

Supported Modes:
  - Supervised: Cell types known; standard forward process
  - Unsupervised: Cell types inferred; standard forward process
  - Both: Union gene set with masking for variable dimensions
"""

from src.utils.logger import Logger
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = Logger.get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class DDPMForwardConfig:
    """Configuration for DDPM forward process with validation."""

    latent_dim: int
    n_diffusion_steps: int
    beta_schedule: str = "linear"
    beta_min: float = 1e-4
    beta_max: float = 2e-2
    device: str = "cpu"

    def __post_init__(self):
        """Validate all parameters."""
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")
        if self.n_diffusion_steps <= 0:
            raise ValueError(
                f"n_diffusion_steps must be > 0, got {self.n_diffusion_steps}"
            )
        if self.beta_schedule not in ["linear", "cosine", "quadratic"]:
            raise ValueError(
                f"beta_schedule must be in ['linear', 'cosine', 'quadratic'], "
                f"got {self.beta_schedule}"
            )
        if not (0 < self.beta_min < self.beta_max < 1):
            raise ValueError(
                f"Must have 0 < beta_min < beta_max < 1, "
                f"got beta_min={self.beta_min}, beta_max={self.beta_max}"
            )

        logger.info(
            f"DDPMForwardConfig: latent_dim={self.latent_dim}, "
            f"n_diffusion_steps={self.n_diffusion_steps}, "
            f"beta_schedule={self.beta_schedule}, "
            f"beta_min={self.beta_min}, beta_max={self.beta_max}"
        )


# ============================================================================
# VARIANCE SCHEDULE BUILDERS
# ============================================================================


class VarianceSchedule:
    """Computes variance schedules for diffusion process."""

    def __init__(self, config: DDPMForwardConfig):
        """
        Initialize variance schedule.

        Args:
            config: DDPMForwardConfig with hyperparameters
        """
        self.config = config
        self.device = torch.device(config.device)

        # Compute schedules based on strategy
        self.beta = self._compute_beta_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_cumprod[:-1]], dim=0
        )

        # Derived quantities for forward process
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alpha)

        logger.info(
            f"VarianceSchedule initialized: "
            f"β_min={self.beta[0]:.6f}, β_max={self.beta[-1]:.6f}"
        )

    def _compute_beta_schedule(self) -> torch.Tensor:
        """
        Compute variance schedule β_t.

        Returns:
            Tensor of shape (n_diffusion_steps,)

        Raises:
            ValueError: If schedule type is invalid
        """
        T = self.config.n_diffusion_steps
        schedule_type = self.config.beta_schedule

        if schedule_type == "linear":
            # Linear interpolation: β_t = β_min + t/T * (β_max - β_min)
            beta = torch.linspace(
                self.config.beta_min,
                self.config.beta_max,
                T,
                device=self.device,
            )
            logger.debug(f"Using linear β schedule")

        elif schedule_type == "cosine":
            # Cosine schedule from Nichol & Dhariwal 2021
            s = 0.008
            steps = torch.arange(T + 1, device=self.device, dtype=torch.float32)
            alphas_cumprod = torch.cos(
                ((steps / T) + s) / (1 + s) * np.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta = torch.clip(betas, 0.0001, 0.9999)
            logger.debug(f"Using cosine β schedule")

        elif schedule_type == "quadratic":
            # Quadratic schedule
            beta = torch.linspace(
                np.sqrt(self.config.beta_min),
                np.sqrt(self.config.beta_max),
                T,
                device=self.device,
            ) ** 2
            logger.debug(f"Using quadratic β schedule")

        else:
            raise ValueError(
                f"Unknown beta_schedule: {schedule_type}. "
                f"Must be one of ['linear', 'cosine', 'quadratic']"
            )

        return beta

    def get_schedule_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return all variance schedule components as dictionary.

        Returns:
            Dict with keys: beta, alpha, alpha_cumprod, alpha_cumprod_prev,
                           sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod,
                           sqrt_beta, sqrt_one_minus_alpha

        """
        return {
            "beta": self.beta,
            "alpha": self.alpha,
            "alpha_cumprod": self.alpha_cumprod,
            "alpha_cumprod_prev": self.alpha_cumprod_prev,
            "sqrt_alpha_cumprod": self.sqrt_alpha_cumprod,
            "sqrt_one_minus_alpha_cumprod": self.sqrt_one_minus_alpha_cumprod,
            "sqrt_beta": self.sqrt_beta,
            "sqrt_one_minus_alpha": self.sqrt_one_minus_alpha,
        }


# ============================================================================
# ABSTRACT BASE CLASS FOR FORWARD PROCESS
# ============================================================================


class ForwardProcess(ABC, nn.Module):
    """Abstract base class for forward diffusion process."""

    def __init__(self, config: DDPMForwardConfig):
        """
        Initialize forward process.

        Args:
            config: DDPMForwardConfig with hyperparameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Initialize variance schedule (pre-computed, not trainable)
        variance_schedule = VarianceSchedule(config)
        schedules = variance_schedule.get_schedule_dict()

        # Register as buffers (not parameters, not updated during training)
        for key, value in schedules.items():
            self.register_buffer(key, value)

    @abstractmethod
    def add_noise_single_step(
        self, z_t_minus_1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise for a single diffusion step (t-1 → t).

        Mathematical: q(z^(t) | z^(t-1)) = N(√(1-β_t) z^(t-1), β_t I)

        Args:
            z_t_minus_1: Latent at step t-1, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), values in [0, T-1]

        Returns:
            Tuple of:
              - z_t: Noised latent at step t, shape (batch_size, latent_dim)
              - noise: Sampled Gaussian noise, shape (batch_size, latent_dim)

        Raises:
            ValueError: If shapes or values are invalid
        """
        pass

    @abstractmethod
    def add_noise_closed_form(
        self, z_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise directly from z^(0) to z^(t) (closed-form).

        Mathematical: q(z^(t) | z^(0)) = N(√ᾱ_t z^(0), (1-ᾱ_t) I)

        This is more efficient than iterative steps for training.

        Args:
            z_0: Initial VAE latent, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), values in [0, T-1]

        Returns:
            Tuple of:
              - z_t: Noised latent at step t, shape (batch_size, latent_dim)
              - noise: Sampled Gaussian noise, shape (batch_size, latent_dim)

        Raises:
            ValueError: If shapes or values are invalid
        """
        pass

    def _validate_inputs(
        self, z: torch.Tensor, t: torch.Tensor, step_name: str
    ) -> None:
        """
        Validate tensor shapes and values.

        Args:
            z: Latent tensor
            t: Timestep tensor
            step_name: Name of calling method for logging

        Raises:
            ValueError: If shapes or values are invalid
        """
        if z.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"{step_name}: latent dimension mismatch. "
                f"Expected {self.config.latent_dim}, got {z.shape[-1]}"
            )
        if len(z.shape) != 2:
            raise ValueError(
                f"{step_name}: z must be 2D tensor (batch_size, latent_dim), "
                f"got shape {z.shape}"
            )
        if len(t.shape) != 1:
            raise ValueError(
                f"{step_name}: t must be 1D tensor (batch_size,), "
                f"got shape {t.shape}"
            )
        if z.shape[0] != t.shape[0]:
            raise ValueError(
                f"{step_name}: batch size mismatch. "
                f"z: {z.shape[0]}, t: {t.shape[0]}"
            )
        if (t < 0).any() or (t >= self.config.n_diffusion_steps).any():
            raise ValueError(
                f"{step_name}: timestep indices out of range [0, {self.config.n_diffusion_steps-1}]. "
                f"Got min={t.min()}, max={t.max()}"
            )
        if z.isnan().any() or z.isinf().any():
            raise ValueError(f"{step_name}: z contains NaN or Inf values")
        if t.isnan().any() or t.isinf().any():
            raise ValueError(f"{step_name}: t contains NaN or Inf values")


# ============================================================================
# UNIFIED FORWARD PROCESS FOR SUPERVISED & UNSUPERVISED
# ============================================================================


class DDPMForwardProcessUnified(ForwardProcess):
    """
    Unified forward diffusion process for both supervised and unsupervised modes.

    The forward process is identical regardless of supervision mode:
    - Only difference is how the VAE latent z^(0) is obtained upstream
    - Supervised: z^(0) ~ q_φ(z^(0) | x_bn, c*) [conditioned on known cell type]
    - Unsupervised: z^(0) ~ q_φ(z^(0) | x_bn) [without cell type conditioning]

    After z^(0) is obtained, the forward process proceeds identically for both cases.
    """

    def __init__(self, config: DDPMForwardConfig):
        """
        Initialize unified forward process.

        Args:
            config: DDPMForwardConfig with hyperparameters
        """
        super().__init__(config)
        logger.info(
            "Initialized DDPMForwardProcessUnified (works for supervised & unsupervised)"
        )

    def add_noise_single_step(
        self, z_t_minus_1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise for a single diffusion step (t-1 → t).

        Mathematical: q(z^(t) | z^(t-1)) = N(√(1-β_t) z^(t-1), β_t I)

        Implementation:
          z^(t) = √(1 - β_t) * z^(t-1) + √β_t * ε
          where ε ~ N(0, I)

        Args:
            z_t_minus_1: Latent at step t-1, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), values in [0, T-2]

        Returns:
            Tuple of (z_t, noise)
              - z_t: Noised latent at step t, shape (batch_size, latent_dim)
              - noise: Sampled Gaussian noise, shape (batch_size, latent_dim)

        Raises:
            ValueError: If shapes or values are invalid
        """
        self._validate_inputs(z_t_minus_1, t, "add_noise_single_step")

        # Sample Gaussian noise: ε ~ N(0, I)
        noise = torch.randn_like(z_t_minus_1)

        # Index the schedule: √(1-β_t) and √β_t
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha[t]  # (batch_size,)
        sqrt_beta_t = self.sqrt_beta[t]  # (batch_size,)

        # Reshape for broadcasting: (batch_size, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.reshape(-1, 1)
        sqrt_beta_t = sqrt_beta_t.reshape(-1, 1)

        # Compute z^(t) = √(1-β_t) * z^(t-1) + √β_t * ε
        z_t = sqrt_one_minus_alpha_t * z_t_minus_1 + sqrt_beta_t * noise

        return z_t, noise

    def add_noise_closed_form(
        self, z_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise directly from z^(0) to z^(t) (closed-form).

        Mathematical: q(z^(t) | z^(0)) = N(√ᾱ_t z^(0), (1-ᾱ_t) I)

        Implementation:
          z^(t) = √ᾱ_t * z^(0) + √(1-ᾱ_t) * ε
          where ε ~ N(0, I)

        This is the primary method used during training (more efficient than
        iterating T steps). It allows direct sampling at any timestep.

        Args:
            z_0: Initial VAE latent, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), values in [0, T-1]

        Returns:
            Tuple of (z_t, noise)
              - z_t: Noised latent at step t, shape (batch_size, latent_dim)
              - noise: Sampled Gaussian noise, shape (batch_size, latent_dim)

        Raises:
            ValueError: If shapes or values are invalid
        """
        self._validate_inputs(z_0, t, "add_noise_closed_form")

        # Sample Gaussian noise: ε ~ N(0, I)
        noise = torch.randn_like(z_0)

        # Index the schedule: √ᾱ_t and √(1-ᾱ_t)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t]  # (batch_size,)
        sqrt_one_minus_alpha_cumprod_t = (
            self.sqrt_one_minus_alpha_cumprod[t]
        )  # (batch_size,)

        # Reshape for broadcasting: (batch_size, 1)
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.reshape(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape(
            -1, 1
        )

        # Compute z^(t) = √ᾱ_t * z^(0) + √(1-ᾱ_t) * ε
        z_t = (
            sqrt_alpha_cumprod_t * z_0
            + sqrt_one_minus_alpha_cumprod_t * noise
        )

        return z_t, noise

    def sample_random_timesteps(
        self, batch_size: int
    ) -> torch.Tensor:
        """
        Sample random timesteps uniformly for a batch.

        Args:
            batch_size: Number of samples

        Returns:
            Tensor of shape (batch_size,) with timesteps in [0, T-1]

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        t = torch.randint(
            0,
            self.config.n_diffusion_steps,
            (batch_size,),
            device=self.device,
        )
        return t

    def get_schedule_at_timestep(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get all variance schedule components at specific timesteps.

        Useful for reverse process, reweighting, etc.

        Args:
            t: Timestep indices, shape (batch_size,)

        Returns:
            Dict with keys: beta, alpha, alpha_cumprod, sqrt_alpha_cumprod,
                           sqrt_one_minus_alpha_cumprod, sqrt_beta, sqrt_one_minus_alpha
            Each value has shape (batch_size,) or (batch_size, 1)

        Raises:
            ValueError: If timesteps are out of range
        """
        if (t < 0).any() or (t >= self.config.n_diffusion_steps).any():
            raise ValueError(
                f"Timestep indices out of range [0, {self.config.n_diffusion_steps-1}]"
            )

        return {
            "beta": self.beta[t],
            "alpha": self.alpha[t],
            "alpha_cumprod": self.alpha_cumprod[t],
            "sqrt_alpha_cumprod": self.sqrt_alpha_cumprod[t],
            "sqrt_one_minus_alpha_cumprod": self.sqrt_one_minus_alpha_cumprod[t],
            "sqrt_beta": self.sqrt_beta[t],
            "sqrt_one_minus_alpha": self.sqrt_one_minus_alpha[t],
        }


# ============================================================================
# FORWARD PROCESS WITH UNION GENE SET MASKING
# ============================================================================


class DDPMForwardProcessWithMasking(DDPMForwardProcessUnified):
    """
    Extended forward process that handles union gene set masking.

    Masks are applied at the likelihood level, not in the diffusion process.
    However, this class provides utilities to manage gene set information
    alongside the diffusion process.

    The forward process itself (adding noise) is unchanged; masking is applied
    downstream during likelihood computation in the reverse process.
    """

    def __init__(
        self,
        config: DDPMForwardConfig,
        union_gene_ids: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """
        Initialize forward process with gene masking support.

        Args:
            config: DDPMForwardConfig with hyperparameters
            union_gene_ids: Dict mapping batch_idx to gene indices (tensor)
                           If None, no masking information is stored

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)

        # Store gene masking information
        self.union_gene_ids = union_gene_ids or {}

        if self.union_gene_ids:
            logger.info(
                f"DDPMForwardProcessWithMasking: "
                f"Managing {len(self.union_gene_ids)} batch gene sets"
            )
        else:
            logger.info("DDPMForwardProcessWithMasking: No gene masking configured")

    def get_gene_mask(
        self, batch_idx: int, full_gene_dim: int
    ) -> torch.Tensor:
        """
        Get binary mask for genes measured in a batch.

        Returns a mask indicating which genes in the union set were measured
        in the specified batch. Shape: (full_gene_dim,)

        Args:
            batch_idx: Batch index
            full_gene_dim: Full dimension of union gene set

        Returns:
            Binary mask tensor of shape (full_gene_dim,), dtype float32
            1.0 for measured genes, 0.0 for unmeasured

        Raises:
            ValueError: If full_gene_dim <= 0
        """
        if full_gene_dim <= 0:
            raise ValueError(f"full_gene_dim must be > 0, got {full_gene_dim}")

        if batch_idx not in self.union_gene_ids:
            # No masking info: return all ones
            mask = torch.ones(
                full_gene_dim, device=self.device, dtype=torch.float32
            )
            logger.debug(
                f"No gene mask for batch_idx={batch_idx}; "
                f"returning all-ones mask"
            )
            return mask

        # Create mask: 1 for measured genes, 0 for unmeasured
        mask = torch.zeros(
            full_gene_dim, device=self.device, dtype=torch.float32
        )
        measured_indices = self.union_gene_ids[batch_idx]

        # Validate indices
        if (measured_indices < 0).any() or (measured_indices >= full_gene_dim).any():
            raise ValueError(
                f"Gene indices out of range for batch_idx={batch_idx}. "
                f"Valid range: [0, {full_gene_dim-1}], "
                f"got indices: {measured_indices}"
            )

        mask[measured_indices] = 1.0

        logger.debug(
            f"Gene mask for batch_idx={batch_idx}: "
            f"{mask.sum().item()}/{full_gene_dim} genes measured"
        )

        return mask

    def get_n_genes_measured(self, batch_idx: int) -> int:
        """
        Get number of genes measured in a batch.

        Args:
            batch_idx: Batch index

        Returns:
            Number of measured genes

        Raises:
            ValueError: If batch_idx not in union_gene_ids
        """
        if batch_idx not in self.union_gene_ids:
            raise ValueError(
                f"batch_idx={batch_idx} not in union_gene_ids. "
                f"Available batches: {list(self.union_gene_ids.keys())}"
            )

        return self.union_gene_ids[batch_idx].shape[0]

    def log_info(self) -> None:
        """Log detailed information about forward process configuration."""
        logger.info(
            f"DDPMForwardProcessWithMasking configuration:\n"
            f"  Latent dim: {self.config.latent_dim}\n"
            f"  Diffusion steps: {self.config.n_diffusion_steps}\n"
            f"  Beta schedule: {self.config.beta_schedule}\n"
            f"  β ∈ [{self.beta[0]:.6f}, {self.beta[-1]:.6f}]\n"
            f"  Gene masking batches: {len(self.union_gene_ids)}"
        )


# ============================================================================
# FORWARD PROCESS MANAGER (UNIFIED INTERFACE)
# ============================================================================


class ForwardProcessManager(nn.Module):
    """
    Unified manager for forward diffusion process.

    Provides a clean interface for both supervised and unsupervised cases,
    with optional gene set masking.

    The manager handles:
    - Single-step and closed-form noise addition
    - Timestep sampling
    - Schedule lookup
    - Gene masking (if configured)
    """

    def __init__(
        self,
        config: DDPMForwardConfig,
        union_gene_ids: Optional[Dict[int, torch.Tensor]] = None,
        enable_masking: bool = False,
    ):
        """
        Initialize forward process manager.

        Args:
            config: DDPMForwardConfig with hyperparameters
            union_gene_ids: Dict mapping batch_idx to gene indices (optional)
            enable_masking: Whether to enable gene set masking utilities

        Raises:
            ValueError: If config is invalid
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Initialize appropriate forward process class
        if enable_masking and union_gene_ids:
            self.forward_process = DDPMForwardProcessWithMasking(
                config, union_gene_ids
            )
            logger.info("ForwardProcessManager: Using forward process with masking")
        else:
            self.forward_process = DDPMForwardProcessUnified(config)
            if enable_masking or union_gene_ids:
                logger.warning(
                    "ForwardProcessManager: enable_masking or union_gene_ids "
                    "provided but not both; using standard forward process"
                )
            logger.info("ForwardProcessManager: Using standard forward process")

    def add_noise(
        self,
        z_0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        use_closed_form: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise to latent codes.

        Args:
            z_0: Initial VAE latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,). If None, sample randomly
            use_closed_form: If True, use closed-form; else use single-step

        Returns:
            Tuple of (z_t, noise, t)
              - z_t: Noised latent, shape (batch_size, latent_dim)
              - noise: Sampled noise, shape (batch_size, latent_dim)
              - t: Timestep indices, shape (batch_size,)

        Raises:
            ValueError: If z_0 shape is invalid
        """
        if t is None:
            t = self.forward_process.sample_random_timesteps(z_0.shape[0])

        if use_closed_form:
            z_t, noise = self.forward_process.add_noise_closed_form(z_0, t)
        else:
            z_t, noise = self.forward_process.add_noise_single_step(z_0, t)

        return z_t, noise, t

    def get_schedule(
        self, t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get variance schedule components.

        Args:
            t: Specific timesteps. If None, return full schedule

        Returns:
            Dict of schedule tensors
        """
        if t is None:
            return self.forward_process.get_schedule_dict()
        else:
            return self.forward_process.get_schedule_at_timestep(t)

    def get_gene_mask(
        self, batch_idx: int, full_gene_dim: int
    ) -> torch.Tensor:
        """
        Get gene mask (if masking is enabled).

        Args:
            batch_idx: Batch index
            full_gene_dim: Full gene dimension

        Returns:
            Gene mask tensor

        Raises:
            RuntimeError: If masking is not enabled
            ValueError: If batch_idx or full_gene_dim invalid
        """
        if not isinstance(
            self.forward_process, DDPMForwardProcessWithMasking
        ):
            raise RuntimeError(
                "Gene masking not enabled. "
                "Initialize with enable_masking=True and union_gene_ids"
            )

        return self.forward_process.get_gene_mask(batch_idx, full_gene_dim)

    def log_info(self) -> None:
        """Log detailed configuration information."""
        logger.info("=" * 70)
        logger.info("ForwardProcessManager Configuration")
        logger.info("=" * 70)
        self.forward_process.log_info()
        logger.info("=" * 70)

    def n_diffusion_steps(self) -> int:
        """Get total number of diffusion steps."""
        return self.config.n_diffusion_steps

    def latent_dim(self) -> int:
        """Get latent dimension."""
        return self.config.latent_dim


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_batch_for_forward(
    z_0: torch.Tensor, t: torch.Tensor, latent_dim: int
) -> None:
    """
    Validate batch tensors for forward process.

    Args:
        z_0: Latent tensor
        t: Timestep tensor
        latent_dim: Expected latent dimension

    Raises:
        ValueError: If validation fails
    """
    if z_0.shape[-1] != latent_dim:
        raise ValueError(
            f"Latent dimension mismatch. Expected {latent_dim}, got {z_0.shape[-1]}"
        )
    if z_0.shape[0] != t.shape[0]:
        raise ValueError(
            f"Batch size mismatch. z_0: {z_0.shape[0]}, t: {t.shape[0]}"
        )
    if (t < 0).any() or (t < 0).any():
        raise ValueError(f"Invalid timesteps: min={t.min()}, max={t.max()}")


def get_noise_schedule_summary(config: DDPMForwardConfig) -> Dict[str, float]:
    """
    Get summary statistics of variance schedule.

    Args:
        config: DDPMForwardConfig

    Returns:
        Dict with schedule statistics
    """
    schedule = VarianceSchedule(config)

    return {
        "beta_min": schedule.beta[0].item(),
        "beta_max": schedule.beta[-1].item(),
        "beta_mean": schedule.beta.mean().item(),
        "alpha_cumprod_min": schedule.alpha_cumprod[-1].item(),
        "alpha_cumprod_max": schedule.alpha_cumprod[0].item(),
    }
