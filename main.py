"""
scARKIDS Main Entry Point
==========================

This is the single entry point to the entire scARKIDS VAE-DDPM pipeline.

It has two modes:
1. Training: Trains and saves the model parameters
2. Inference: Uses the trained model to make predictions

Control Flow:
-------------
1. Parse config.yaml using ConfigManager
2. Initialize data loaders for training/validation/inference
3. Initialize atomic objects (likelihood, prior, ddpm_forward, ddpm_backward, encoder, classifier)
4. Initialize compound objects (variational_posterior, elbo)
5. Initialize training/inference managers
6. Execute training or inference based on mode

Usage:
------
# Training mode
python main.py --mode train --config config.yaml

# Inference mode
python main.py --mode inference --config config.yaml --checkpoint checkpoints/best_model.pt --data_path data/test.h5ad
"""

import argparse
import os
import sys
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import anndata as ad

# Import utilities
from src.utils.config_utils import ConfigManager
from src.utils.logger import Logger

# Import atomic modules (model components)
from src.model.encoder import EncoderManager
from src.model.classifier import ClassifierManager
from src.model.likelihood import LikelihoodManager
from src.model.prior import PriorManager
from src.model.ddpm_forward import DDPMForwardManager
from src.model.ddpm_backward import DDPMBackwardManager

# Import compound modules
from src.model.variational_posterior import VariationalPosteriorManager
from src.model.elbo import ELBOManager

# Import training and inference
from src.training.training import TrainingManager
from src.inference.inference import InferenceManager

# Initialize logger
logger = Logger.get_logger(__name__)

# ============================================================================
# Data Loading
# ============================================================================

class scRNASeqDataset(Dataset):
    """
    PyTorch Dataset for scRNA-seq data loaded from AnnData.
    
    Expected AnnData structure (from screenshot):
    - adata.X: Gene expression matrix (n_obs, n_vars)
    - adata.obs: Cell metadata including 'celltype', 'batch', etc.
    - adata.var: Gene metadata
    
    Dataset returns:
    - x: Gene expression (n_genes,)
    - batch_index: Batch index (scalar)
    - batch_onehot: Batch one-hot encoding (n_batches,)
    - celltype_index: Cell type index (scalar)
    - celltype_onehot: Cell type one-hot encoding (n_cell_types,)
    - library_size: Library size log(sum(x)) (scalar)
    """
    def __init__(
        self,
        adata: ad.AnnData,
        cell_type_key: str = "celltype",
        batch_key: str = "batch",
        n_cell_types: int = None,
        n_batches: int = None,
        supervised: bool = False
    ):
        """
        Initialize dataset from AnnData object with comprehensive data sanitization.

        Args:
            adata: AnnData object containing scRNA-seq data
            cell_type_key: Key in adata.obs for cell type labels
            batch_key: Key in adata.obs for batch labels
            n_cell_types: Number of cell types (inferred if None)
            n_batches: Number of batches (inferred if None)
            supervised: If True, cell types are available; else may be missing
        """
        self.adata = adata
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.supervised = supervised

        # Extract expression matrix (dense format)
        if hasattr(adata.X, 'toarray'):
            # Sparse matrix
            self.expression = adata.X.toarray().astype(np.float32)
        else:
            # Dense matrix
            self.expression = adata.X.astype(np.float32)

        # Sanitize expression matrix: remove NaN/Inf, clip negative values
        logger.info("Sanitizing expression matrix...")
        invalid_mask = ~np.isfinite(self.expression)
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            logger.warning(f"Found {n_invalid} NaN/Inf values in expression matrix. Setting to 0.")
            self.expression[invalid_mask] = 0.0

        # Clip negative values (should not exist in counts, but be safe)
        self.expression = np.clip(self.expression, a_min=0.0, a_max=None)
        logger.info("✓ Expression matrix sanitized")

        # Extract batch labels
        if batch_key not in adata.obs.columns:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

        batch_labels = adata.obs[batch_key].astype('category')
        self.batch_categories = batch_labels.cat.categories
        self.batch_indices = batch_labels.cat.codes.values.astype(np.int64)
        self.n_batches = n_batches or len(self.batch_categories)

        # Extract cell type labels (if available)
        self.has_celltype = cell_type_key in adata.obs.columns
        if self.has_celltype:
            celltype_labels = adata.obs[cell_type_key].astype('category')
            self.celltype_categories = celltype_labels.cat.categories
            self.celltype_indices = celltype_labels.cat.codes.values.astype(np.int64)
            self.n_cell_types = n_cell_types or len(self.celltype_categories)
        else:
            if supervised:
                logger.warning(
                    f"Supervised mode enabled but cell type key '{cell_type_key}' "
                    "not found in adata.obs. Using dummy labels."
                )
            self.celltype_indices = np.zeros(len(adata), dtype=np.int64)
            self.n_cell_types = n_cell_types or 1
            self.celltype_categories = [f"Type{i}" for i in range(self.n_cell_types)]

        # Compute library sizes (sum of counts per cell) with comprehensive sanitization
        logger.info("Computing library sizes...")
        raw_library_sizes = self.expression.sum(axis=1).astype(np.float32)

        # Clip negative values (should not occur after clipping expression, but be safe)
        raw_library_sizes = np.clip(raw_library_sizes, a_min=0.0, a_max=None)

        # Apply log1p transformation
        self.library_sizes = np.log1p(raw_library_sizes)

        # Final sanitization: replace any NaN/Inf with 0.0
        invalid_lib_mask = ~np.isfinite(self.library_sizes)
        if invalid_lib_mask.any():
            n_invalid_lib = invalid_lib_mask.sum()
            logger.warning(f"Found {n_invalid_lib} NaN/Inf values in library sizes. Setting to 0.0.")
            self.library_sizes[invalid_lib_mask] = 0.0

        logger.info("✓ Library sizes computed and sanitized")

        logger.info(f"Dataset initialized:")
        logger.info(f"  n_obs: {len(self)}")
        logger.info(f"  n_genes: {self.expression.shape[1]}")
        logger.info(f"  n_batches: {self.n_batches}")
        logger.info(f"  n_cell_types: {self.n_cell_types}")
        logger.info(f"  has_celltype: {self.has_celltype}")

    # def __init__(
    #     self,
    #     adata: ad.AnnData,
    #     cell_type_key: str = "celltype",
    #     batch_key: str = "batch",
    #     n_cell_types: int = None,
    #     n_batches: int = None,
    #     supervised: bool = False
    # ):
    #     """
    #     Initialize dataset from AnnData object.
        
    #     Args:
    #         adata: AnnData object containing scRNA-seq data
    #         cell_type_key: Key in adata.obs for cell type labels
    #         batch_key: Key in adata.obs for batch labels
    #         n_cell_types: Number of cell types (inferred if None)
    #         n_batches: Number of batches (inferred if None)
    #         supervised: If True, cell types are available; else may be missing
    #     """
    #     self.adata = adata
    #     self.cell_type_key = cell_type_key
    #     self.batch_key = batch_key
    #     self.supervised = supervised
        
    #     # Extract expression matrix (dense format)
    #     if hasattr(adata.X, 'toarray'):
    #         # Sparse matrix
    #         self.expression = adata.X.toarray().astype(np.float32)
    #     else:
    #         # Dense matrix
    #         self.expression = adata.X.astype(np.float32)
        
    #     # Extract batch labels
    #     if batch_key not in adata.obs.columns:
    #         raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
        
    #     batch_labels = adata.obs[batch_key].astype('category')
    #     self.batch_categories = batch_labels.cat.categories
    #     self.batch_indices = batch_labels.cat.codes.values.astype(np.int64)
    #     self.n_batches = n_batches or len(self.batch_categories)
        
    #     # Extract cell type labels (if available)
    #     self.has_celltype = cell_type_key in adata.obs.columns
    #     if self.has_celltype:
    #         celltype_labels = adata.obs[cell_type_key].astype('category')
    #         self.celltype_categories = celltype_labels.cat.categories
    #         self.celltype_indices = celltype_labels.cat.codes.values.astype(np.int64)
    #         self.n_cell_types = n_cell_types or len(self.celltype_categories)
    #     else:
    #         if supervised:
    #             logger.warning(
    #                 f"Supervised mode enabled but cell type key '{cell_type_key}' "
    #                 "not found in adata.obs. Using dummy labels."
    #             )
    #         self.celltype_indices = np.zeros(len(adata), dtype=np.int64)
    #         self.n_cell_types = n_cell_types or 1
    #         self.celltype_categories = [f"Type{i}" for i in range(self.n_cell_types)]
        
    #     # Compute library sizes (sum of counts per cell)
    #     self.library_sizes = np.log1p(self.expression.sum(axis=1)).astype(np.float32)
        
    #     logger.info(f"Dataset initialized:")
    #     logger.info(f"  n_obs: {len(self)}")
    #     logger.info(f"  n_genes: {self.expression.shape[1]}")
    #     logger.info(f"  n_batches: {self.n_batches}")
    #     logger.info(f"  n_cell_types: {self.n_cell_types}")
    #     logger.info(f"  has_celltype: {self.has_celltype}")
    
    def __len__(self):
        return len(self.expression)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            Dictionary containing all required tensors
        """
        # Gene expression
        x = torch.from_numpy(self.expression[idx])
        
        # Batch
        batch_idx = self.batch_indices[idx]
        batch_onehot = F.one_hot(
            torch.tensor(batch_idx, dtype=torch.long),
            num_classes=self.n_batches
        ).float()
        
        # Cell type (if available)
        celltype_idx = self.celltype_indices[idx]
        celltype_onehot = F.one_hot(
            torch.tensor(celltype_idx, dtype=torch.long),
            num_classes=self.n_cell_types
        ).float()
        
        # Library size
        library_size = torch.tensor(self.library_sizes[idx], dtype=torch.float32)
        
        return {
            'x': x,
            'batch_indices': torch.tensor(batch_idx, dtype=torch.long),
            'batch_onehot': batch_onehot,
            'celltype_indices': torch.tensor(celltype_idx, dtype=torch.long),
            'celltype_onehot': celltype_onehot,
            'library_size': library_size
        }


def create_data_loaders(
    config_manager: ConfigManager,
    batch_size: int,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and test sets.
    
    Args:
        config_manager: Configuration manager instance
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    data_config = config_manager.get_data_config()
    global_config = config_manager.global_params
    
    supervised = global_config['supervised']
    cell_type_key = data_config['cell_type_key']
    batch_key = data_config['batch_key']
    n_cell_types = global_config['n_cell_types']
    n_batches = global_config['n_batches']
    
    data_loaders = {}
    
    # Training data
    train_path = data_config.get('train_data_path')
    if train_path and os.path.exists(train_path):
        logger.info(f"Loading training data from {train_path}")
        train_adata = ad.read_h5ad(train_path)
        train_dataset = scRNASeqDataset(
            train_adata,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
            n_cell_types=n_cell_types,
            n_batches=n_batches,
            supervised=supervised
        )
        data_loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"✓ Training DataLoader created: {len(train_dataset)} samples")
    else:
        logger.warning(f"Training data not found at {train_path}")
    
    # Validation data
    val_path = data_config.get('val_data_path')
    if val_path and os.path.exists(val_path):
        logger.info(f"Loading validation data from {val_path}")
        val_adata = ad.read_h5ad(val_path)
        val_dataset = scRNASeqDataset(
            val_adata,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
            n_cell_types=n_cell_types,
            n_batches=n_batches,
            supervised=supervised
        )
        data_loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"✓ Validation DataLoader created: {len(val_dataset)} samples")
    else:
        logger.warning(f"Validation data not found at {val_path}")
    
    # Test data
    test_path = data_config.get('test_data_path')
    if test_path and os.path.exists(test_path):
        logger.info(f"Loading test data from {test_path}")
        test_adata = ad.read_h5ad(test_path)
        test_dataset = scRNASeqDataset(
            test_adata,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
            n_cell_types=n_cell_types,
            n_batches=n_batches,
            supervised=supervised
        )
        data_loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"✓ Test DataLoader created: {len(test_dataset)} samples")
    else:
        logger.warning(f"Test data not found at {test_path}")
    
    return data_loaders


# ============================================================================
# Model Initialization
# ============================================================================

def initialize_models(config_manager: ConfigManager):
    """
    Initialize all atomic and compound model components.
    
    This function:
    1. Initializes atomic objects (encoder, classifier, likelihood, prior, ddpm_forward, ddpm_backward)
    2. Initializes compound objects (variational_posterior, elbo)
    3. Returns managers for training/inference
    
    Args:
        config_manager: Configuration manager instance
    
    Returns:
        Dictionary containing all initialized managers
    """
    logger.info("=" * 80)
    logger.info("Initializing Model Components")
    logger.info("=" * 80)
    
    supervised = config_manager.is_supervised()
    
    # ========================================================================
    # Step 1: Initialize Atomic Objects
    # ========================================================================
    
    logger.info("-" * 80)
    logger.info("Step 1: Initializing Atomic Objects")
    logger.info("-" * 80)
    
    # 1.1 Encoder
    logger.info("Initializing Encoder...")
    encoder_config = config_manager.get_config('encoder')
    encoder_manager = EncoderManager(encoder_config)
    logger.info("✓ Encoder initialized")
    
    # 1.2 Classifier (required for unsupervised mode)
    classifier_manager = None
    if not supervised:
        logger.info("Initializing Classifier (unsupervised mode)...")
        classifier_config = config_manager.get_config('classifier')
        classifier_manager = ClassifierManager(classifier_config)
        logger.info("✓ Classifier initialized")
    else:
        logger.info("Classifier skipped (supervised mode)")
    
    # 1.3 Likelihood
    logger.info("Initializing Likelihood...")
    likelihood_config = config_manager.get_config('likelihood')
    likelihood_manager = LikelihoodManager(likelihood_config)
    logger.info("✓ Likelihood initialized")
    
    # 1.4 Prior
    logger.info("Initializing Prior...")
    prior_config = config_manager.get_config('prior')
    prior_manager = PriorManager(prior_config)
    logger.info("✓ Prior initialized")
    
    # 1.5 DDPM Forward
    logger.info("Initializing DDPM Forward Process...")
    ddpm_forward_config = config_manager.get_config('ddpm_forward')
    ddpm_forward_manager = DDPMForwardManager(ddpm_forward_config)
    variance_schedule = ddpm_forward_manager.get_schedule()
    logger.info("✓ DDPM Forward initialized")
    
    # 1.6 DDPM Backward
    logger.info("Initializing DDPM Backward Process...")
    ddpm_backward_config = config_manager.get_config('ddpm_backward')
    ddpm_backward_manager = DDPMBackwardManager(ddpm_backward_config, variance_schedule)
    logger.info("✓ DDPM Backward initialized")
    
    # ========================================================================
    # Step 2: Initialize Compound Objects
    # ========================================================================
    
    logger.info("-" * 80)
    logger.info("Step 2: Initializing Compound Objects")
    logger.info("-" * 80)
    
    # 2.1 Variational Posterior
    logger.info("Initializing Variational Posterior...")
    variational_posterior_manager = VariationalPosteriorManager(
        encoder_manager=encoder_manager,
        ddpm_forward_manager=ddpm_forward_manager,
        classifier_manager=classifier_manager,
        supervised=supervised
    )
    logger.info("✓ Variational Posterior initialized")
    
    # 2.2 ELBO
    logger.info("Initializing ELBO Module...")
    elbo_manager = ELBOManager(
        likelihood_manager=likelihood_manager,
        prior_manager=prior_manager,
        ddpm_backward_manager=ddpm_backward_manager,
        variational_posterior_manager=variational_posterior_manager,
        classifier_manager=classifier_manager
    )
    logger.info("✓ ELBO Module initialized")
    
    # ========================================================================
    # Step 3: Initialize Training and Inference Managers
    # ========================================================================
    
    logger.info("-" * 80)
    logger.info("Step 3: Initializing Training and Inference Managers")
    logger.info("-" * 80)
    
    # 3.1 Training Manager
    logger.info("Initializing Training Manager...")
    training_config = config_manager.get_config('training')
    training_manager = TrainingManager(
        config_dict=training_config,
        likelihood_manager=likelihood_manager,
        prior_manager=prior_manager,
        ddpm_forward_manager=ddpm_forward_manager,
        ddpm_backward_manager=ddpm_backward_manager,
        encoder_manager=encoder_manager,
        variational_posterior_manager=variational_posterior_manager,
        elbo_manager=elbo_manager,
        classifier_manager=classifier_manager
    )
    logger.info("✓ Training Manager initialized")
    
    # 3.2 Inference Manager
    logger.info("Initializing Inference Manager...")
    inference_config = config_manager.get_config('inference')
    inference_manager = InferenceManager(
        likelihood_manager=likelihood_manager,
        prior_manager=prior_manager,
        ddpm_forward_manager=ddpm_forward_manager,
        ddpm_backward_manager=ddpm_backward_manager,
        encoder_manager=encoder_manager,
        classifier_manager=classifier_manager,
        variational_posterior_manager=variational_posterior_manager,
        elbo_manager=elbo_manager,
        use_ddpm_refinement=inference_config.get('use_ddpm_refinement', False),
        n_refinement_steps=inference_config.get('n_refinement_steps', None)
    )
    logger.info("✓ Inference Manager initialized")
    
    logger.info("=" * 80)
    logger.info("All Model Components Initialized Successfully")
    logger.info("=" * 80)
    
    return {
        # Atomic managers
        'encoder': encoder_manager,
        'classifier': classifier_manager,
        'likelihood': likelihood_manager,
        'prior': prior_manager,
        'ddpm_forward': ddpm_forward_manager,
        'ddpm_backward': ddpm_backward_manager,
        # Compound managers
        'variational_posterior': variational_posterior_manager,
        'elbo': elbo_manager,
        # Training/Inference managers
        'training': training_manager,
        'inference': inference_manager
    }


# ============================================================================
# Training Mode
# ============================================================================

def run_training(
    config_manager: ConfigManager,
    managers: Dict,
    data_loaders: Dict
):
    """
    Execute training mode.
    
    Args:
        config_manager: Configuration manager instance
        managers: Dictionary of initialized managers
        data_loaders: Dictionary of data loaders
    """
    logger.info("=" * 80)
    logger.info("TRAINING MODE")
    logger.info("=" * 80)
    
    training_manager = managers['training']
    
    # Get train and validation loaders
    train_loader = data_loaders.get('train')
    val_loader = data_loaders.get('val')
    
    if train_loader is None:
        raise ValueError("Training data loader not available. Check data paths in config.")
    
    # Run training
    try:
        training_manager.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        logger.info("✓ Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


# ============================================================================
# Inference Mode
# ============================================================================

def run_inference(
    config_manager: ConfigManager,
    managers: Dict,
    data_loaders: Dict,
    checkpoint_path: str,
    output_dir: Optional[str] = None
):
    """
    Execute inference mode.
    
    Args:
        config_manager: Configuration manager instance
        managers: Dictionary of initialized managers
        data_loaders: Dictionary of data loaders
        checkpoint_path: Path to trained model checkpoint
        output_dir: Directory to save inference outputs
    """
    logger.info("=" * 80)
    logger.info("INFERENCE MODE")
    logger.info("=" * 80)
    
    inference_manager = managers['inference']
    training_manager = managers['training']
    inference_config = config_manager.get_config('inference')
    
    # Load trained model
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    training_manager.load_model(checkpoint_path)
    logger.info("✓ Checkpoint loaded successfully")
    
    # Get test loader
    test_loader = data_loaders.get('test')
    if test_loader is None:
        raise ValueError("Test data loader not available. Check data paths in config or provide --data_path.")
    
    # Setup output directory
    if output_dir is None:
        output_dir = inference_config.get('output_dir', './results')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Run inference
    logger.info(f"Running inference on {len(test_loader.dataset)} cells...")
    
    all_results = {
        'latent_embeddings': [],
        'corrected_expression': [],
        'cell_type_predictions': [],
        'cell_type_probs': [],
        'batch_indices': [],
        'true_cell_types': []
    }
    
    device = torch.device(inference_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    default_target_batch = inference_config.get('default_target_batch', 0)
    
    for batch_idx, batch in enumerate(test_loader):
        # Move batch to device
        x_new = batch['x'].to(device)
        batch_new = batch['batch_indices'].to(device)
        celltype_true = batch['celltype_indices'].to(device)
        
        # Target batch (default: batch 0)
        batch_target = torch.full_like(batch_new, default_target_batch)
        
        # Run inference
        # Determine if supervised or unsupervised inference (fetch from config)
        is_supervised = config_manager.get_config("global").get("supervised", False)

        if is_supervised:
            # In supervised case, provide cell type labels to inference module
            cell_type_input = celltype_true
        else:
            # In unsupervised case, cell type labels unknown; let model predict
            cell_type_input = None

        outputs = inference_manager.predict(
            x_new=x_new,
            batch_new=batch_new,
            cell_type_new=cell_type_input,  # Correct behavior for both cases
            batch_target=batch_target,
            return_intermediate=True
        )
        
        # Collect results
        all_results['latent_embeddings'].append(outputs['z_0'].cpu().numpy())
        all_results['corrected_expression'].append(outputs['x_corrected'].cpu().numpy())
        all_results['cell_type_predictions'].append(outputs['cell_type_pred'].cpu().numpy())
        if 'cell_type_probs' in outputs:
            all_results['cell_type_probs'].append(outputs['cell_type_probs'].cpu().numpy())
        all_results['batch_indices'].append(batch_new.cpu().numpy())
        all_results['true_cell_types'].append(celltype_true.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {(batch_idx + 1) * test_loader.batch_size} / {len(test_loader.dataset)} cells")
    
    # Concatenate all results
    for key in all_results:
        if all_results[key]:
            all_results[key] = np.concatenate(all_results[key], axis=0)
    
    logger.info("✓ Inference completed successfully")
    
    # Save results
    logger.info("Saving inference results...")
    
    if inference_config.get('save_latent_embeddings', True):
        latent_path = Path(output_dir) / 'latent_embeddings.npy'
        np.save(latent_path, all_results['latent_embeddings'])
        logger.info(f"✓ Saved latent embeddings to {latent_path}")
    
    if inference_config.get('save_corrected_expression', True):
        corrected_path = Path(output_dir) / 'corrected_expression.npy'
        np.save(corrected_path, all_results['corrected_expression'])
        logger.info(f"✓ Saved batch-corrected expression to {corrected_path}")
    
    if inference_config.get('save_cell_type_predictions', True):
        predictions_path = Path(output_dir) / 'cell_type_predictions.npy'
        np.save(predictions_path, all_results['cell_type_predictions'])
        logger.info(f"✓ Saved cell type predictions to {predictions_path}")
        
        if all_results['cell_type_probs'].size > 0:
            probs_path = Path(output_dir) / 'cell_type_probabilities.npy'
            np.save(probs_path, all_results['cell_type_probs'])
            logger.info(f"✓ Saved cell type probabilities to {probs_path}")
    
    # Compute and log accuracy (if ground truth available)
    if all_results['true_cell_types'].size > 0:
        accuracy = (all_results['cell_type_predictions'] == all_results['true_cell_types']).mean()
        logger.info(f"Cell type prediction accuracy: {accuracy * 100:.2f}%")
    
    logger.info("=" * 80)
    logger.info("Inference completed successfully")
    logger.info("=" * 80)

def run_sequential_training(
    config_manager: ConfigManager,
    managers: Dict,
    data_loaders: Dict
):
    """
    Execute sequential multi-dataset training with checkpoint management.
    
    Two-step approach:
    1. INITIALIZATION: Load config, initialize all models (fresh params)
    2. SEQUENTIAL TRAINING: For each session:
       - Load data
       - For i=1: Use initialized model
       - For i>1: Load best model from session i-1
       - Train
       - Save best model and metrics
    
    Each session has its own:
    - Log file: scARKIDS/logs/session_name_training.log
    - Metrics: scARKIDS/logs/session_name_metrics.json
    - Summary: scARKIDS/logs/session_name_summary.json
    
    Args:
        config_manager: Configuration manager instance
        managers: Dictionary of initialized managers
        data_loaders: Dictionary of data loaders (will be recreated per session)
    """
    
    logger.info("=" * 80)
    logger.info("SEQUENTIAL MULTI-DATASET TRAINING MODE (INITIALIZED ONCE)")
    logger.info("=" * 80)
    
    # ========================================================================
    # STEP 1: INITIALIZATION (Do this once before any training)
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: INITIALIZATION (Fresh Model Parameters)")
    logger.info("=" * 80)
    
    logger.info("Model components initialized with fresh parameters")
    logger.info("Ready for sequential training across all sessions\n")
    
    # Get training sessions from config
    sessions = config_manager.get_training_sessions()
    
    if sessions is None:
        # Fallback to single training run (backward compatibility)
        logger.info("No sessions config found. Running single training session.")
        run_training(config_manager, managers, data_loaders)
        return
    
    num_sessions = len(sessions)
    logger.info(f"Will execute {num_sessions} training sessions sequentially\n")
    
    # ========================================================================
    # STEP 2: SEQUENTIAL TRAINING (One session at a time)
    # ========================================================================
    
    for session_idx, session in enumerate(sessions, 1):
        logger.info("=" * 80)
        logger.info(f"SESSION {session_idx}/{num_sessions}: {session['name']}")
        logger.info("=" * 80)
        
        session_name = session['name']
        data_path = session['data_path']
        supervised = session['supervised']
        n_epochs = session['n_epochs']
        resume_from = session.get('resume_from', None)
        output_checkpoint = session['output_checkpoint']
        
        # ====================================================================
        # Step 2.1: Configure Session-Specific Logging
        # ====================================================================
        
        logger.info(f"\n[STEP 1/5] Configuring logging for session {session_idx}")
        
        # Configure file logging for this session (one log file per session)
        log_file = Logger.configure_file_logging(
            session_name=session_name,
            log_dir="scARKIDS/logs"
        )
        logger.info(f"Session logs: {log_file}")
        
        # Store session name in config for JSON export
        training_manager = managers['training']
        training_manager.session_name = session_name
        
        # ====================================================================
        # Step 2.2: Load Training Data for This Session
        # ====================================================================
        
        logger.info(f"\n[STEP 2/5] Loading training data for session {session_idx}")
        logger.info(f"Data path: {data_path}")
        
        # Validate data exists
        if not os.path.exists(data_path):
            logger.error(f"✗ Data not found: {data_path}")
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        # Update config data path
        data_config = config_manager.get_data_config()
        data_config['train_data_path'] = data_path
        
        # Create fresh data loaders for this session
        training_config = config_manager.get_config('training')
        batch_size = training_config['batch_size']
        
        try:
            session_data_loaders = create_data_loaders(
                config_manager=config_manager,
                batch_size=batch_size,
                num_workers=4
            )
        except Exception as e:
            logger.error(f"✗ Failed to create data loaders: {e}")
            raise
        
        train_loader = session_data_loaders.get('train')
        if train_loader is None:
            raise ValueError(
                f"No training data loaded for session {session_idx} "
                f"(data_path: {data_path})"
            )
        val_loader = session_data_loaders.get('val', None)
        if val_loader is None:
            logger.warning(f"No validation data loaded for session {session_idx}")
            logger.warning(f"Validation metrics will not be computed")
        
        n_train = len(train_loader.dataset)
        logger.info(f"✓ Loaded {n_train} training samples")
        
        # ====================================================================
        # Step 2.3: Handle Model Initialization or Checkpoint Loading
        # ====================================================================
        
        logger.info(f"\n[STEP 3/5] Preparing model for session {session_idx}")
        
        if session_idx == 1:
            # Session 1: Use initialized model (fresh params from STEP 1)
            logger.info(f"Using initialized model (fresh parameters)")
        else:
            # Sessions 2-4: Load best model from previous session
            if resume_from is not None and os.path.exists(resume_from):
                logger.info(f"Loading best model from Session {session_idx - 1}")
                logger.info(f"Checkpoint: {resume_from}")
                try:
                    training_manager.load_model(resume_from)
                    logger.info(f"✓ Model loaded successfully (warm-start)")
                except Exception as e:
                    logger.error(f"✗ Failed to load checkpoint: {e}")
                    raise
            else:
                if resume_from is not None:
                    logger.warning(f"Checkpoint not found: {resume_from}")
                    logger.warning(f"Initializing fresh model")
                else:
                    logger.warning(f"No resume_from specified, using fresh initialization")
        
        # ====================================================================
        # Step 2.4: Update Training Configuration for This Session
        # ====================================================================
        
        logger.info(f"\n[STEP 4/5] Configuring training for session {session_idx}")
        
        # Update global supervised mode
        config_manager.global_params['supervised'] = supervised
        
        # Update number of epochs
        training_config['n_epochs'] = n_epochs
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_checkpoint)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Session Configuration:")
        logger.info(f"  Name: {session_name}")
        logger.info(f"  Mode: {'Supervised' if supervised else 'Unsupervised'}")
        logger.info(f"  Epochs: {n_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Train samples: {n_train}")
        logger.info(f"  Output checkpoint: {output_checkpoint}")
        
        # ====================================================================
        # Step 2.5: Train for This Session
        # ====================================================================
        
        logger.info(f"\n[STEP 5/5] Training on {session_name}")
        logger.info("-" * 80)
        
        try:
            # Reset best loss tracker for this session
            training_manager.training_core.best_loss = float('inf')
            
            # Train on this session's data
            training_manager.train(
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            logger.info("-" * 80)
            logger.info(f"✓ Training completed successfully for session {session_idx}")
            
        except KeyboardInterrupt:
            logger.info("⚠ Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"✗ Training failed for session {session_idx}: {e}")
            raise
        
        # ====================================================================
        # Step 2.6: Save Best Model and Metrics
        # ====================================================================
        
        logger.info(f"\nSaving best model and metrics")
        
        try:
            training_manager.save_model(output_checkpoint)
            logger.info(f"✓ Best model saved to: {output_checkpoint}")
        except Exception as e:
            logger.error(f"✗ Failed to save checkpoint: {e}")
            raise
        
        # Log session summary
        best_loss = training_manager.training_core.best_loss
        logger.info(f"\nSession {session_idx} Summary:")
        logger.info(f"  Name: {session_name}")
        logger.info(f"  Best loss: {best_loss:.6f}")
        logger.info(f"  Total epochs trained: {n_epochs}")
        logger.info(f"  Output checkpoint: {output_checkpoint}")
        logger.info(f"  Log file: {log_file}")
        
        if session_idx < num_sessions:
            next_session = sessions[session_idx]
            logger.info(f"  Next session ({next_session['name']}) will load this checkpoint ✓")
        
        logger.info("=" * 80 + "\n")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("✓ SEQUENTIAL TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Trained on {num_sessions} datasets sequentially")
    logger.info(f"Initialization: Fresh model once before Session 1")
    logger.info(f"Sessions 2-4: Each loaded previous session's best model")
    logger.info(f"Final best model: {sessions[-1]['output_checkpoint']}")
    logger.info(f"All logs saved to: scARKIDS/logs/")
    logger.info("=" * 80)


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for scARKIDS pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="scARKIDS: Single-cell RNA-seq Analysis using VAE-DDPM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'inference', 'sequential'],
        help='Execution mode: train (single), inference, or sequential (multi-dataset)'
    )

    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (required for inference mode)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to data file for inference (overrides config)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for inference results (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides config)'
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("scARKIDS: Single-cell RNA-seq Analysis using VAE-DDPM")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Config: {args.config}")
    
    # Load configuration
    try:
        config_manager = ConfigManager(args.config)
        config_manager.validate()
        logger.info("✓ Configuration loaded and validated successfully")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Set random seed
    seed = args.seed
    if seed is None:
        training_config = config_manager.get_config('training')
        seed = training_config.get('seed', 42)
    set_seed(seed)
    
    # ========================================================================
    # Data Loading
    # ========================================================================
    
    logger.info("-" * 80)
    logger.info("Loading Data")
    logger.info("-" * 80)
    
    try:
        training_config = config_manager.get_config('training')
        batch_size = training_config['batch_size']
        
        # Override test data path if provided
        if args.data_path is not None:
            data_config = config_manager.get_data_config()
            data_config['test_data_path'] = args.data_path
            logger.info(f"Overriding test data path: {args.data_path}")
        
        data_loaders = create_data_loaders(
            config_manager=config_manager,
            batch_size=batch_size,
            num_workers=4
        )
        
        if not data_loaders:
            logger.error("No data loaders created. Check data paths in config.")
            sys.exit(1)
            
        logger.info("✓ Data loaders created successfully")
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Model Initialization
    # ========================================================================
    
    try:
        managers = initialize_models(config_manager)
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Execution
    # ========================================================================
    
    try:
        if args.mode == 'train':
            run_training(
                config_manager=config_manager,
                managers=managers,
                data_loaders=data_loaders
            )
        
        elif args.mode == 'sequential':
            run_sequential_training(
                config_manager=config_manager,
                managers=managers,
                data_loaders=data_loaders
            )
        
        elif args.mode == 'inference':
            # Validate checkpoint path
            checkpoint_path = args.checkpoint
            if checkpoint_path is None:
                # Try to use best model from config
                training_config = config_manager.get_config('training')
                checkpoint_dir = training_config.get('checkpoint_dir', './checkpoints')
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                logger.error("Please provide --checkpoint path for inference mode")
                sys.exit(1)
            
            run_inference(
                config_manager=config_manager,
                managers=managers,
                data_loaders=data_loaders,
                checkpoint_path=checkpoint_path,
                output_dir=args.output_dir
            )
    
    except Exception as e:
        logger.error(f"Execution error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("scARKIDS execution completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
