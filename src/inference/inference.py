"""
Inference Module for scARKIDS
==============================

Implements test-time inference algorithm for batch correction and cell type prediction.

Mathematical Background (Inference Algorithm):
----------------------------------------------

Input: New cell (x_new, b_new), optionally c_new if known
Goal: Obtain batch-corrected embedding and reconstruction

Procedure:
1. Cell type prediction (if unknown):
   ĉ_new = argmax_c q_ω(c|x_new, b_new)

2. VAE encoding (deterministic at test time):
   μ_φ, σ_φ = Encoder_φ(x_new, b_new, c_new)
   z_new^(0) = μ_φ  (use mean, no sampling)

3. Optional DDPM refinement:
   • Forward: Add noise to z_new^(0)
   • Reverse: Iteratively denoise using p_ψ(z^(t-1)|z^(t), c_new)

4. Batch correction (decode to target batch):
   x̂_new = Decoder_θ(z_new^(0), b_target)

Output: 
- Batch-invariant latent z_new^(0)
- Predicted cell type ĉ_new
- Corrected expression x̂_new
"""

from src.utils.logger import Logger
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)

# ============================================================================
# Inference Core Module
# ============================================================================

class InferenceModule(nn.Module):
    """
    Core inference logic for test-time prediction and batch correction.
    
    Orchestrates the inference pipeline:
    1. Cell type prediction (if needed)
    2. Deterministic encoding to z^(0)
    3. Optional DDPM refinement
    4. Batch-corrected reconstruction
    """
    
    def __init__(
        self,
        encoder_module,           # VAEEncoder instance
        classifier_module,        # ClassifierModule instance
        ddpm_forward_module,      # DDPMForwardModule instance
        ddpm_backward_module,     # DDPMBackwardModule instance
        likelihood_module,        # LikelihoodModule instance (contains decoder)
        n_batches: int,
        n_cell_types: int,
        n_genes: int,
        latent_dim: int,
        use_ddpm_refinement: bool = False,
        n_refinement_steps: Optional[int] = None
    ):
        """
        Initialize inference module.
        
        Args:
            encoder_module: Trained encoder
            classifier_module: Trained classifier
            ddpm_forward_module: Forward diffusion module
            ddpm_backward_module: Trained backward diffusion module
            likelihood_module: Likelihood module (contains decoder networks)
            n_batches: Number of batches
            n_cell_types: Number of cell types
            n_genes: Number of genes
            latent_dim: Latent dimension
            use_ddpm_refinement: Whether to apply DDPM refinement
            n_refinement_steps: Number of DDPM denoising steps (default: all steps)
        """
        super().__init__()
        
        # Store atomic modules
        self.encoder = encoder_module
        self.classifier = classifier_module
        self.ddpm_forward = ddpm_forward_module
        self.ddpm_backward = ddpm_backward_module
        self.likelihood = likelihood_module
        
        # Configuration
        self.n_batches = n_batches
        self.n_cell_types = n_cell_types
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.use_ddpm_refinement = use_ddpm_refinement
        self.n_refinement_steps = n_refinement_steps or ddpm_forward_module.config.n_diffusion_steps
        
        # Validate configuration
        assert self.n_refinement_steps <= ddpm_forward_module.config.n_diffusion_steps, \
            f"n_refinement_steps ({self.n_refinement_steps}) cannot exceed n_diffusion_steps"
        
        logger.info("Initialized InferenceModule")
        logger.info(f"  n_batches={n_batches}")
        logger.info(f"  n_cell_types={n_cell_types}")
        logger.info(f"  n_genes={n_genes}")
        logger.info(f"  latent_dim={latent_dim}")
        logger.info(f"  use_ddpm_refinement={use_ddpm_refinement}")
        logger.info(f"  n_refinement_steps={self.n_refinement_steps}")
    
    def _validate_inputs(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor,
        cell_type_new: Optional[torch.Tensor] = None,
        batch_target: Optional[torch.Tensor] = None
    ) -> None:
        """
        Validate input tensors.
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Batch indices (batch_size,)
            cell_type_new: Optional cell type indices (batch_size,)
            batch_target: Optional target batch indices (batch_size,)
            
        Raises:
            ValueError: If inputs are invalid
        """
        batch_size = x_new.shape[0]
        
        # Check x_new
        if x_new.dim() != 2:
            raise ValueError(f"x_new must be 2D (batch_size, n_genes), got shape {x_new.shape}")
        if x_new.shape[1] != self.n_genes:
            raise ValueError(f"Gene dimension mismatch: expected {self.n_genes}, got {x_new.shape[1]}")
        
        # Check batch_new
        if batch_new.dim() != 1:
            raise ValueError(f"batch_new must be 1D (batch_size,), got shape {batch_new.shape}")
        if batch_new.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: x_new={batch_size}, batch_new={batch_new.shape[0]}")
        if (batch_new < 0).any() or (batch_new >= self.n_batches).any():
            raise ValueError(f"batch_new out of range [0, {self.n_batches-1}]")
        
        # Check cell_type_new if provided
        if cell_type_new is not None:
            if cell_type_new.dim() != 1:
                raise ValueError(f"cell_type_new must be 1D (batch_size,), got shape {cell_type_new.shape}")
            if cell_type_new.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: x_new={batch_size}, cell_type_new={cell_type_new.shape[0]}")
            if (cell_type_new < 0).any() or (cell_type_new >= self.n_cell_types).any():
                raise ValueError(f"cell_type_new out of range [0, {self.n_cell_types-1}]")
        
        # Check batch_target if provided
        if batch_target is not None:
            if batch_target.dim() != 1:
                raise ValueError(f"batch_target must be 1D (batch_size,), got shape {batch_target.shape}")
            if batch_target.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: x_new={batch_size}, batch_target={batch_target.shape[0]}")
            if (batch_target < 0).any() or (batch_target >= self.n_batches).any():
                raise ValueError(f"batch_target out of range [0, {self.n_batches-1}]")
        
        # Check for NaN/Inf
        if torch.isnan(x_new).any() or torch.isinf(x_new).any():
            raise ValueError("x_new contains NaN or Inf values")
        if torch.isnan(batch_new).any() or torch.isinf(batch_new).any():
            raise ValueError("batch_new contains NaN or Inf values")
    
    def predict_cell_type(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 1: Cell type prediction (if unknown).
        
        Mathematical formulation:
        ĉ_new = argmax_c q_ω(c|x_new, b_new)
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Batch indices (batch_size,)
            
        Returns:
            predicted_cell_type: Predicted cell type indices (batch_size,)
            cell_type_probs: Class probabilities (batch_size, n_cell_types)
        """
        logger.debug("Step 1: Predicting cell type")
        
        # Get cell type probabilities from classifier
        # q_ω(c|x_new, b_new)
        predicted_cell_type, cell_type_probs = self.classifier.predict_class(
            x=x_new,
            batch=batch_new,
            sample=False  # Deterministic: use argmax
        )
        
        logger.debug(f"  Predicted cell types: {predicted_cell_type.tolist()}")
        logger.debug(f"  Mean confidence: {cell_type_probs.max(dim=1)[0].mean().item():.4f}")
        
        return predicted_cell_type, cell_type_probs
    
    def encode_deterministic(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor,
        cell_type_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step 2: VAE encoding (deterministic at test time).
        
        Mathematical formulation:
        μ_φ, σ_φ = Encoder_φ(x_new, b_new, c_new)
        z_new^(0) = μ_φ  (use mean, no sampling)
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Batch indices (batch_size,)
            cell_type_new: Cell type indices (batch_size,)
            
        Returns:
            z_0: Initial latent representation (batch_size, latent_dim)
            mu_phi: Encoder mean (batch_size, latent_dim)
            sigma_phi: Encoder log-variance (batch_size, latent_dim)
        """
        logger.debug("Step 2: Deterministic VAE encoding")
        
        # Convert indices to one-hot
        batch_onehot = F.one_hot(batch_new, num_classes=self.n_batches).float()
        celltype_onehot = F.one_hot(cell_type_new, num_classes=self.n_cell_types).float()
        
        # Encode to latent parameters
        # q_φ(z^(0) | x_new, b_new, c_new)
        mu_phi, sigma_phi = self.encoder.encode(
            x=x_new,
            batch_onehot=batch_onehot,
            celltype_onehot=celltype_onehot
        )
        
        # Use mean (no sampling at test time for stability)
        z_0 = mu_phi
        
        logger.debug(f"  Latent z^(0) shape: {z_0.shape}")
        logger.debug(f"  Latent z^(0) norm: {z_0.norm(dim=1).mean().item():.4f}")
        
        return z_0, mu_phi, sigma_phi
    
    def apply_ddpm_refinement(
        self,
        z_0: torch.Tensor,
        cell_type_new: torch.Tensor
    ) -> torch.Tensor:
        """
        Step 3: Optional DDPM refinement.
        
        Mathematical formulation:
        • Forward: z^(T) = forward_diffusion(z^(0))
        • Reverse: z^(0)_refined = iterative_denoise(z^(T), c_new)
        
        Args:
            z_0: Initial latent (batch_size, latent_dim)
            cell_type_new: Cell type indices (batch_size,)
            
        Returns:
            z_0_refined: Refined latent (batch_size, latent_dim)
        """
        logger.debug("Step 3: Applying DDPM refinement")
        
        batch_size = z_0.shape[0]
        
        # Forward: Add noise to z^(0) to get z^(T)
        t_forward = torch.full(
            (batch_size,), 
            self.n_refinement_steps - 1, 
            dtype=torch.long, 
            device=z_0.device
        )
        z_t, _ = self.ddpm_forward.add_noise_closed_form(z_0, t_forward)
        
        logger.debug(f"  Forward: Added noise to get z^({self.n_refinement_steps-1})")
        logger.debug(f"  Noisy latent norm: {z_t.norm(dim=1).mean().item():.4f}")
        
        # Reverse: Iteratively denoise z^(T) → z^(T-1) → ... → z^(0)
        logger.debug(f"  Reverse: Denoising for {self.n_refinement_steps} steps")
        
        for step_idx in range(self.n_refinement_steps - 1, -1, -1):
            # Current timestep
            t = torch.full((batch_size,), step_idx, dtype=torch.long, device=z_t.device)
            
            # Denoise: z^(t) → z^(t-1)
            # p_ψ(z^(t-1) | z^(t), c_new)
            z_t, _ = self.ddpm_backward.reverse_step(z_t, t, cell_type_new)
            
            # Log progress periodically
            if (step_idx + 1) % 100 == 0 or step_idx == 0:
                logger.debug(
                    f"    Step {self.n_refinement_steps - step_idx}/{self.n_refinement_steps}, "
                    f"t={step_idx}, z_norm={z_t.norm(dim=1).mean().item():.4f}"
                )
        
        z_0_refined = z_t
        
        logger.debug(f"  Refined latent z^(0) norm: {z_0_refined.norm(dim=1).mean().item():.4f}")
        
        return z_0_refined
    
    def decode_batch_corrected(
        self,
        z_0: torch.Tensor,
        batch_target: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step 4: Batch correction (decode to target batch).
        
        Mathematical formulation:
        x̂_new = Decoder_θ(z_new^(0), b_target)
        
        Args:
            z_0: Latent representation (batch_size, latent_dim)
            batch_target: Target batch indices (batch_size,)
            library_size: Optional library size (batch_size,)
            
        Returns:
            x_reconstructed: Reconstructed expression (batch_size, n_genes)
            mu: Mean parameter from decoder (batch_size, n_genes)
            pi: Dropout probability from decoder (batch_size, n_genes)
        """
        logger.debug("Step 4: Batch-corrected reconstruction")
        
        # Convert batch to one-hot
        batch_target_onehot = F.one_hot(batch_target, num_classes=self.n_batches).float()
        
        # Decode: p_θ(x | z^(0), b_target)
        mu = self.likelihood.mean_decoder(z_0, batch_target_onehot, library_size)
        pi = self.likelihood.dropout_decoder(z_0, batch_target_onehot)
        
        # Reconstructed expression: use mean of ZINB distribution
        x_reconstructed = mu
        
        logger.debug(f"  Reconstructed expression shape: {x_reconstructed.shape}")
        logger.debug(f"  Mean expression: {x_reconstructed.mean().item():.4f}")
        
        return x_reconstructed, mu, pi
    
    def forward(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor,
        cell_type_new: Optional[torch.Tensor] = None,
        batch_target: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None,
        return_intermediate: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Complete inference pipeline.
        
        Pipeline:
        1. Cell type prediction (if not provided)
        2. Deterministic encoding to z^(0)
        3. Optional DDPM refinement
        4. Batch-corrected reconstruction
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Source batch indices (batch_size,)
            cell_type_new: Optional cell type indices (batch_size,)
                          If None, will predict using classifier
            batch_target: Optional target batch indices (batch_size,)
                         If None, uses batch_new
            library_size: Optional library size (batch_size,)
            return_intermediate: If True, return all intermediate values
            
        Returns:
            Dictionary containing:
            - 'x_corrected': Batch-corrected expression (batch_size, n_genes)
            - 'z_0': Batch-invariant latent (batch_size, latent_dim)
            - 'cell_type_pred': Predicted/provided cell type (batch_size,)
            - 'cell_type_probs': Cell type probabilities (batch_size, n_cell_types) [if predicted]
            - 'mu': Decoder mean parameter (batch_size, n_genes) [if return_intermediate]
            - 'pi': Decoder dropout probability (batch_size, n_genes) [if return_intermediate]
            - 'encoder_mean': Encoder mean (batch_size, latent_dim) [if return_intermediate]
            - 'encoder_logvar': Encoder log-variance (batch_size, latent_dim) [if return_intermediate]
        """
        # Validate inputs
        self._validate_inputs(x_new, batch_new, cell_type_new, batch_target)
        
        # Set target batch (default: same as source batch)
        if batch_target is None:
            batch_target = batch_new
            logger.debug("No target batch provided, using source batch")
        
        # Step 1: Cell type prediction (if unknown)
        if cell_type_new is None:
            logger.info("Cell type unknown, predicting...")
            cell_type_pred, cell_type_probs = self.predict_cell_type(x_new, batch_new)
        else:
            logger.info("Cell type provided, skipping prediction")
            cell_type_pred = cell_type_new
            cell_type_probs = None
        
        # Step 2: Deterministic VAE encoding
        logger.info("Encoding to latent space...")
        z_0, encoder_mean, encoder_logvar = self.encode_deterministic(
            x_new, batch_new, cell_type_pred
        )
        
        # Step 3: Optional DDPM refinement
        if self.use_ddpm_refinement:
            logger.info(f"Applying DDPM refinement ({self.n_refinement_steps} steps)...")
            z_0 = self.apply_ddpm_refinement(z_0, cell_type_pred)
        else:
            logger.info("DDPM refinement disabled, using VAE latent directly")
        
        # Step 4: Batch-corrected reconstruction
        logger.info("Decoding to batch-corrected expression...")
        x_corrected, mu, pi = self.decode_batch_corrected(z_0, batch_target, library_size)
        
        # Prepare output
        output = {
            'x_corrected': x_corrected,
            'z_0': z_0,
            'cell_type_pred': cell_type_pred,
        }
        
        # Add cell type probabilities if predicted
        if cell_type_probs is not None:
            output['cell_type_probs'] = cell_type_probs
        
        # Add intermediate values if requested
        if return_intermediate:
            output.update({
                'mu': mu,
                'pi': pi,
                'encoder_mean': encoder_mean,
                'encoder_logvar': encoder_logvar,
            })
        
        logger.info("Inference complete")
        
        return output

# ============================================================================
# Inference Manager (Module Entry Point)
# ============================================================================

class InferenceManager:
    """
    Manager class for the Inference module.
    
    This is the single entry point that:
    1. Takes all atomic objects and compound objects
    2. Initializes the inference module
    3. Exposes APIs for test-time inference
    """
    
    def __init__(
        self,
        likelihood_manager,        # LikelihoodManager instance
        prior_manager,             # PriorManager instance (unused but kept for consistency)
        ddpm_forward_manager,      # DDPMForwardManager instance
        ddpm_backward_manager,     # DDPMBackwardManager instance
        encoder_manager,           # EncoderManager instance
        classifier_manager,        # ClassifierManager instance
        variational_posterior_manager,  # VariationalPosteriorManager instance (unused)
        elbo_manager,              # ELBOManager instance (unused but kept for consistency)
        use_ddpm_refinement: bool = False,
        n_refinement_steps: Optional[int] = None
    ):
        """
        Initialize inference manager from atomic and compound objects.
        
        Args:
            likelihood_manager: Likelihood manager
            prior_manager: Prior manager (unused in inference)
            ddpm_forward_manager: DDPM forward manager
            ddpm_backward_manager: DDPM backward manager
            encoder_manager: Encoder manager
            classifier_manager: Classifier manager
            variational_posterior_manager: Variational posterior manager (unused)
            elbo_manager: ELBO manager (unused in inference)
            use_ddpm_refinement: Whether to apply DDPM refinement
            n_refinement_steps: Number of DDPM denoising steps
        """
        logger.info("Initializing InferenceManager")
        
        # Get atomic modules
        self.encoder_module = encoder_manager.get_module()
        if classifier_manager is not None:
            self.classifier_module = classifier_manager.get_module()
        else:
            self.classifier_module = None
        self.ddpm_forward_module = ddpm_forward_manager.get_module()
        self.ddpm_backward_module = ddpm_backward_manager.get_module()
        self.likelihood_module = likelihood_manager.get_module()
        
        # Extract configuration from modules
        n_batches = encoder_manager.config.n_batches
        n_cell_types = encoder_manager.config.n_cell_types
        n_genes = encoder_manager.config.n_genes
        latent_dim = encoder_manager.config.latent_dim
        
        # Initialize inference module
        self.inference_module = InferenceModule(
            encoder_module=self.encoder_module,
            classifier_module=self.classifier_module,
            ddpm_forward_module=self.ddpm_forward_module,
            ddpm_backward_module=self.ddpm_backward_module,
            likelihood_module=self.likelihood_module,
            n_batches=n_batches,
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            latent_dim=latent_dim,
            use_ddpm_refinement=use_ddpm_refinement,
            n_refinement_steps=n_refinement_steps
        )
        
        logger.info("InferenceManager initialized successfully")
    
    def get_module(self) -> InferenceModule:
        """
        Get the inference module.
        
        Returns:
            InferenceModule instance
        """
        return self.inference_module
    
    def predict(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor,
        cell_type_new: Optional[torch.Tensor] = None,
        batch_target: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on new data (main API).
        
        This is the primary interface for test-time inference.
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Source batch indices (batch_size,)
            cell_type_new: Optional cell type indices (batch_size,)
            batch_target: Optional target batch indices (batch_size,)
            library_size: Optional library size (batch_size,)
            return_intermediate: If True, return intermediate values
            
        Returns:
            Dictionary containing inference outputs (see InferenceModule.forward)
        """
        logger.info(f"Running inference on {x_new.shape[0]} cells")
        
        # Set module to eval mode
        self.inference_module.eval()
        
        # Run inference without gradient computation
        with torch.no_grad():
            outputs = self.inference_module(
                x_new=x_new,
                batch_new=batch_new,
                cell_type_new=cell_type_new,
                batch_target=batch_target,
                library_size=library_size,
                return_intermediate=return_intermediate
            )
        
        logger.info("Inference completed successfully")
        
        return outputs
    
    def batch_correct(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor,
        batch_target: torch.Tensor,
        cell_type_new: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batch correction: transform data from source batch to target batch.
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Source batch indices (batch_size,)
            batch_target: Target batch indices (batch_size,)
            cell_type_new: Optional cell type indices (batch_size,)
            library_size: Optional library size (batch_size,)
            
        Returns:
            x_corrected: Batch-corrected expression (batch_size, n_genes)
        """
        logger.info(f"Batch correction: {batch_new[0].item()} → {batch_target[0].item()}")
        
        outputs = self.predict(
            x_new=x_new,
            batch_new=batch_new,
            cell_type_new=cell_type_new,
            batch_target=batch_target,
            library_size=library_size,
            return_intermediate=False
        )
        
        return outputs['x_corrected']
    
    def get_latent_embedding(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor,
        cell_type_new: Optional[torch.Tensor] = None,
        apply_refinement: bool = None
    ) -> torch.Tensor:
        """
        Get batch-invariant latent embedding z^(0).
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Batch indices (batch_size,)
            cell_type_new: Optional cell type indices (batch_size,)
            apply_refinement: Override DDPM refinement setting (None uses default)
            
        Returns:
            z_0: Batch-invariant latent (batch_size, latent_dim)
        """
        # Temporarily override DDPM refinement if specified
        original_setting = self.inference_module.use_ddpm_refinement
        if apply_refinement is not None:
            self.inference_module.use_ddpm_refinement = apply_refinement
        
        try:
            outputs = self.predict(
                x_new=x_new,
                batch_new=batch_new,
                cell_type_new=cell_type_new,
                batch_target=batch_new,  # Same batch (no correction)
                return_intermediate=False
            )
            
            return outputs['z_0']
        
        finally:
            # Restore original setting
            self.inference_module.use_ddpm_refinement = original_setting
    
    def predict_cell_type_only(
        self,
        x_new: torch.Tensor,
        batch_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict cell type only (without full inference).
        
        Args:
            x_new: Gene expression (batch_size, n_genes)
            batch_new: Batch indices (batch_size,)
            
        Returns:
            predicted_cell_type: Predicted cell type indices (batch_size,)
            cell_type_probs: Class probabilities (batch_size, n_cell_types)
        """
        logger.info("Predicting cell types")
        
        self.inference_module.eval()
        
        with torch.no_grad():
            predicted_cell_type, cell_type_probs = self.inference_module.predict_cell_type(
                x_new=x_new,
                batch_new=batch_new
            )
        
        return predicted_cell_type, cell_type_probs
    


