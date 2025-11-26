"""
Configuration Utility for scARKIDS
==================================

This module provides intelligent configuration management with:
1. Hierarchical configuration with global defaults
2. Automatic parameter inheritance
3. Module-specific overrides
4. Validation and type checking
5. Single source of truth (config.yaml)

Usage:
------
```python
from src.utils.config import ConfigManager

# Load configuration
config_manager = ConfigManager("config.yaml")

# Get module-specific configuration (with inheritance applied)
encoder_config = config_manager.get_config("encoder")
classifier_config = config_manager.get_config("classifier")

# Access global parameters
latent_dim = config_manager.global_params["latent_dim"]
supervised = config_manager.is_supervised()

# Validate entire configuration
config_manager.validate()
```
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from src.utils.logger import Logger

logger = Logger.get_logger(__name__)

# ============================================================================
# Module Configuration Schemas
# ============================================================================

MODULE_SCHEMAS = {
    "encoder": {
        "required": ["n_genes", "n_batches", "n_cell_types", "latent_dim"],
        "optional": {
            "hidden_dims": list,
            "dropout": float,
            "use_batch_norm": bool,
            "use_layer_norm": bool,
            "input_transform": str,
            "eps": float
        },
        "defaults": {
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "use_batch_norm": True,
            "use_layer_norm": False,
            "input_transform": "log1p",
            "eps": 1e-8
        }
    },
    "classifier": {
        "required": ["n_genes", "n_cell_types", "n_batches"],
        "optional": {
            "hidden_dim": int,
            "n_layers": int,
            "batch_embed_dim": int,
            "dropout": float,
            "use_batch_norm": bool
        },
        "defaults": {
            "hidden_dim": 256,
            "n_layers": 3,
            "batch_embed_dim": 32,
            "dropout": 0.2,
            "use_batch_norm": True
        }
    },
    "likelihood": {
        "required": ["latent_dim", "n_genes", "n_batches"],
        "optional": {
            "hidden_dim": int,
            "n_layers": int,
            "dispersion_mode": str,
            "use_library_size": bool,
            "eps": float
        },
        "defaults": {
            "hidden_dim": 128,
            "n_layers": 2,
            "dispersion_mode": "gene",
            "use_library_size": True,
            "eps": 1e-8
        }
    },
    "prior": {
        "required": ["latent_dim", "n_cell_types", "n_batches", "supervised"],
        "optional": {
            "cell_type_prior_probs": (list, type(None)),
            "batch_prior_probs": (list, type(None))
        },
        "defaults": {
            "cell_type_prior_probs": None,
            "batch_prior_probs": None
        }
    },
    "ddpm_forward": {
        "required": ["latent_dim", "n_diffusion_steps"],
        "optional": {
            "beta_schedule": str,
            "beta_min": float,
            "beta_max": float
        },
        "defaults": {
            "beta_schedule": "linear",
            "beta_min": 1e-4,
            "beta_max": 2e-2
        }
    },
    "ddpm_backward": {
        "required": ["latent_dim", "n_diffusion_steps", "n_cell_types"],
        "optional": {
            "variance_type": str,
            "noise_hidden_dim": int,
            "noise_n_layers": int,
            "timestep_embed_dim": int,
            "celltype_embed_dim": int,
            "dropout": float
        },
        "defaults": {
            "variance_type": "fixed",
            "noise_hidden_dim": 128,
            "noise_n_layers": 2,
            "timestep_embed_dim": 64,
            "celltype_embed_dim": 32,
            "dropout": 0.1
        }
    },
    "variational_posterior": {
        "required": ["supervised", "latent_dim", "n_diffusion_steps", "n_cell_types"],
        "optional": {},
        "defaults": {}
    },
    "elbo": {
        "required": ["supervised", "latent_dim", "n_diffusion_steps", "n_cell_types"],
        "optional": {},
        "defaults": {}
    },
    "training": {
        "required": ["supervised", "n_epochs", "batch_size"],
        "optional": {
            "learning_rate": float,
            "weight_decay": float,
            "grad_clip_norm": float,
            "log_interval": int,
            "checkpoint_interval": int,
            "checkpoint_dir": str,
            "device": str,
            "use_amp": bool,
            "lr_scheduler": (str, type(None)),
            "lr_warmup_epochs": int,
            "seed": int
        },
        "defaults": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip_norm": 1.0,
            "log_interval": 100,
            "checkpoint_interval": 1,
            "checkpoint_dir": "./checkpoints",
            "device": "cuda",
            "use_amp": False,
            "lr_scheduler": "cosine",
            "lr_warmup_epochs": 5,
            "seed": 42
        }
    },
    "inference": {
        "required": [],
        "optional": {
            "use_ddpm_refinement": bool,
            "n_refinement_steps": (int, type(None)),
            "default_target_batch": int,
            "save_latent_embeddings": bool,
            "save_corrected_expression": bool,
            "save_cell_type_predictions": bool,
            "output_dir": str
        },
        "defaults": {
            "use_ddpm_refinement": False,
            "n_refinement_steps": None,
            "default_target_batch": 0,
            "save_latent_embeddings": True,
            "save_corrected_expression": True,
            "save_cell_type_predictions": True,
            "output_dir": "./results"
        }
    }
}


# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigManager:
    """
    Intelligent configuration manager with hierarchical inheritance.
    
    Features:
    - Loads config.yaml as single source of truth
    - Applies global parameter inheritance
    - Handles module-specific overrides
    - Validates all configurations
    - Type checking and range validation
    
    Example:
    --------
    ```python
    config_manager = ConfigManager("config.yaml")
    
    # Get resolved configuration for encoder (with inheritance applied)
    encoder_config = config_manager.get_config("encoder")
    # Returns: dict with all required params from global + encoder-specific
    
    # Access raw sections
    global_params = config_manager.global_params
    data_config = config_manager.data_config
    ```
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load raw configuration
        with open(self.config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)
        
        # Extract sections
        self.global_params = self.raw_config.get("global", {})
        self.data_config = self.raw_config.get("data", {})
        self.module_configs = {
            module: self.raw_config.get(module, {})
            for module in MODULE_SCHEMAS.keys()
        }
        
        logger.info(f"Loaded configuration from {config_path}")
        logger.info(f"Mode: {'Supervised' if self.is_supervised() else 'Unsupervised'}")
        logger.info(f"Global parameters: {list(self.global_params.keys())}")
    
    def is_supervised(self) -> bool:
        """Check if model is in supervised mode."""
        return self.global_params.get("supervised", False)
    
    def get_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a module with inheritance applied.
        
        This method:
        1. Starts with module defaults from schema
        2. Applies global parameters
        3. Applies module-specific overrides
        4. Validates required parameters
        
        Args:
            module_name: Name of module ('encoder', 'classifier', etc.)
        
        Returns:
            Complete configuration dictionary for the module
        
        Raises:
            KeyError: If module not found
            ValueError: If required parameters missing
        """
        if module_name not in MODULE_SCHEMAS:
            raise KeyError(f"Unknown module: {module_name}")
        
        schema = MODULE_SCHEMAS[module_name]
        
        # Step 1: Start with defaults
        config = schema["defaults"].copy()
        
        # Step 2: Apply global parameters (for params in required or optional)
        all_module_params = set(schema["required"]) | set(schema["optional"].keys())
        for param in all_module_params:
            if param in self.global_params:
                config[param] = self.global_params[param]
        
        # Step 3: Apply module-specific overrides
        # Step 3: Apply module-specific overrides
            module_config = self.module_configs.get(module_name)
            if module_config is None:
                module_config = {}          # treat empty YAML sections as empty dict
        
            config.update(module_config)
        
        # Step 4: Validate required parameters
        missing = [p for p in schema["required"] if p not in config]
        if missing:
            raise ValueError(
                f"Module '{module_name}' missing required parameters: {missing}. "
                f"These must be specified in either 'global' or '{module_name}' section."
            )
        
        logger.debug(f"Resolved configuration for {module_name}: {config}")
        return config
    
    def validate(self) -> None:
        """
        Validate entire configuration.
        
        Checks:
        - All required parameters present
        - Type correctness
        - Value ranges
        - Cross-module consistency
        
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating configuration...")
        
        # Validate each module
        for module_name in MODULE_SCHEMAS.keys():
            try:
                config = self.get_config(module_name)
                self._validate_module_config(module_name, config)
            except Exception as e:
                logger.error(f"Validation failed for module '{module_name}': {e}")
                raise
        
        # Cross-module consistency checks
        self._validate_cross_module_consistency()
        
        logger.info("✓ Configuration validation passed")
    
    def _validate_module_config(self, module_name: str, config: Dict[str, Any]) -> None:
        """Validate a single module configuration."""
        schema = MODULE_SCHEMAS[module_name]
        
        # Type checking
        for param, value in config.items():
            if param in schema["optional"]:
                expected_type = schema["optional"][param]
                
                # Handle union types (e.g., (list, type(None)))
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param}' in module '{module_name}' has incorrect type. "
                            f"Expected one of {expected_type}, got {type(value)}"
                        )
                else:
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param}' in module '{module_name}' has incorrect type. "
                            f"Expected {expected_type}, got {type(value)}"
                        )
        
        # Value range checks (module-specific)
        self._validate_value_ranges(module_name, config)
    
    def _validate_value_ranges(self, module_name: str, config: Dict[str, Any]) -> None:
        """Validate parameter value ranges."""
        
        # Common validations
        if "latent_dim" in config:
            assert config["latent_dim"] > 0, "latent_dim must be positive"
        
        if "n_diffusion_steps" in config:
            assert config["n_diffusion_steps"] > 0, "n_diffusion_steps must be positive"
        
        if "n_genes" in config:
            assert config["n_genes"] > 0, "n_genes must be positive"
        
        if "n_batches" in config:
            assert config["n_batches"] > 0, "n_batches must be positive"
        
        if "n_cell_types" in config:
            assert config["n_cell_types"] > 0, "n_cell_types must be positive"
        
        if "dropout" in config:
            assert 0 <= config["dropout"] < 1, "dropout must be in [0, 1)"
        
        if "eps" in config:
            assert config["eps"] > 0, "eps must be positive"
        
        # Module-specific validations
        if module_name == "encoder":
            assert config["input_transform"] in ["log1p", "none"], \
                "input_transform must be 'log1p' or 'none'"
            assert not (config["use_batch_norm"] and config["use_layer_norm"]), \
                "Cannot use both batch_norm and layer_norm"
        
        if module_name == "likelihood":
            assert config["dispersion_mode"] == "gene", \
                "Only 'gene' dispersion mode supported"
        
        if module_name == "ddpm_forward":
            assert config["beta_schedule"] in ["linear", "cosine", "quadratic"], \
                "beta_schedule must be 'linear', 'cosine', or 'quadratic'"
            assert 0 < config["beta_min"] < config["beta_max"] < 1, \
                "Must have 0 < beta_min < beta_max < 1"
        
        if module_name == "ddpm_backward":
            assert config["variance_type"] in ["fixed", "learned"], \
                "variance_type must be 'fixed' or 'learned'"
        
        if module_name == "training":
            assert config["n_epochs"] > 0, "n_epochs must be positive"
            assert config["batch_size"] > 0, "batch_size must be positive"
            assert config["learning_rate"] > 0, "learning_rate must be positive"
            assert config["grad_clip_norm"] > 0, "grad_clip_norm must be positive"
            if config["lr_scheduler"] is not None:
                assert config["lr_scheduler"] in ["cosine", "step", "plateau"], \
                    "lr_scheduler must be 'cosine', 'step', 'plateau', or null"
    
    def _validate_cross_module_consistency(self) -> None:
        """Validate consistency across modules."""
        
        # Check that all modules agree on shared parameters
        shared_params = {
            "latent_dim": [],
            "n_diffusion_steps": [],
            "n_genes": [],
            "n_batches": [],
            "n_cell_types": [],
            "supervised": []
        }
        
        for module_name in MODULE_SCHEMAS.keys():
            config = self.get_config(module_name)
            for param in shared_params.keys():
                if param in config:
                    shared_params[param].append((module_name, config[param]))
        
        # Verify consistency
        for param, values in shared_params.items():
            if len(values) > 1:
                first_value = values[0][1]
                for module_name, value in values[1:]:
                    if value != first_value:
                        raise ValueError(
                            f"Inconsistent values for '{param}': "
                            f"{values[0][0]}={first_value}, {module_name}={value}"
                        )
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.data_config
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.raw_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {output_path}")
    
    def __repr__(self) -> str:
        mode = "Supervised" if self.is_supervised() else "Unsupervised"
        return f"ConfigManager(mode={mode}, modules={list(MODULE_SCHEMAS.keys())})"


# ============================================================================
# Example Usage
# ============================================================================

# if __name__ == "__main__":
#     # Example: Load and validate configuration
#     config_manager = ConfigManager("config.yaml")
    
#     # Validate
#     config_manager.validate()
    
#     # Get module configurations (with inheritance applied)
#     encoder_config = config_manager.get_config("encoder")
#     print(f"Encoder config: {encoder_config}")
    
#     classifier_config = config_manager.get_config("classifier")
#     print(f"Classifier config: {classifier_config}")
    
#     # Access global parameters
#     print(f"Latent dim: {config_manager.global_params['latent_dim']}")
#     print(f"Supervised: {config_manager.is_supervised()}")
