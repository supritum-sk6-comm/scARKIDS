scARKIDS: Single-cell RNA Integration via Variational Autoencoders and Denoising Diffusion Models
A principled deep generative framework for batch correction and data integration in single-cell RNA-seq.

🎯 What is scARKIDS?
scARKIDS combines VAE (Variational Autoencoder) and DDPM (Denoising Diffusion Probabilistic Models) to:

Learn batch-invariant latent representations of cells

Perform batch correction while preserving cell-type-specific biology

Support both supervised (with cell type labels) and unsupervised (inferring cell types) modes

Enable transfer learning across multiple datasets through sequential training

📋 Prerequisites
Python: 3.8+

GPU: CUDA-capable GPU recommended (RTX 3090 / A100 / V100 for ~1-2 hours training per session)

OS: Linux/macOS (Windows via WSL2)

🚀 Quick Start
1. Clone Repository
bash
git clone https://github.com/supritum-sk6-comm/scARKIDS.git
cd scARKIDS
2. Create and Activate Virtual Environment
bash
# Create virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows (Command Prompt)
venv\Scripts\activate

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1
3. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
4. Verify Installation
bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
📁 Repository Structure
text
scARKIDS/
├── main.py                          # Entry point (train/inference)
├── config.yaml                      # Configuration file (edit this)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── src/
│   ├── utils/
│   │   ├── config_utils.py         # Configuration manager
│   │   └── logger.py               # Logging utilities
│   ├── model/
│   │   ├── encoder.py              # VAE encoder
│   │   ├── classifier.py           # Cell type classifier
│   │   ├── likelihood.py           # ZINB likelihood (decoder)
│   │   ├── prior.py                # Prior distributions
│   │   ├── ddpm_forward.py         # Forward diffusion process
│   │   ├── ddpm_backward.py        # Reverse diffusion process
│   │   ├── variational_posterior.py # Posterior approximation
│   │   └── elbo.py                 # ELBO loss computation
│   ├── training/
│   │   └── training.py             # Training loop
│   └── inference/
│       └── inference.py            # Inference pipeline
│
├── checkpoints/                    # Saved model checkpoints
├── results/                        # Output directory (embeddings, corrections)
└── data/                           # Input data (H5AD files)
🎓 Understanding the Modes
Supervised Mode
Use when: Cell type labels are available

What it learns: Batch-invariant embeddings conditioned on known cell types

Example dataset: Pancreas with annotated cell types

Unsupervised Mode
Use when: Cell type labels are unknown or missing

What it learns: Cell type predictions + batch-invariant embeddings

Example dataset: Unlabeled lung tissue

🔧 Configuration
Edit Config File
Open config.yaml and modify:

text
global:
  latent_dim: 20                 # Latent space dimension (keep fixed)
  n_diffusion_steps: 400         # Diffusion timesteps (keep fixed)
  n_genes: 2000                  # ADJUST: Number of genes in your data
  n_batches: 8                   # ADJUST: Max number of batches
  n_cell_types: 20               # ADJUST: Max number of cell types
  supervised: true               # true for supervised, false for unsupervised

data:
  train_data_path: "./data/train.h5ad"      # ADJUST: Path to train data
  val_data_path: "./data/val.h5ad"          # ADJUST: Path to validation data
  test_data_path: "./data/test.h5ad"        # ADJUST: Path to test data
  cell_type_key: "cell_type"    # Key in adata.obs for cell type labels
  batch_key: "batch"            # Key in adata.obs for batch labels

training:
  n_epochs: 100                  # Number of epochs
  batch_size: 256                # Batch size (GPU memory dependent)
  learning_rate: 1.0e-3          # Learning rate
  checkpoint_dir: "./checkpoints" # Where to save checkpoints
Data Format
Your data must be in AnnData (.h5ad) format:

python
import anndata as ad

# Minimal structure
adata = ad.AnnData(
    X=expression_matrix,           # (n_cells, n_genes) gene counts
    obs={'batch': batch_labels,    # Batch identifier
         'cell_type': cell_types}, # Cell type annotation (if supervised)
)
adata.write_h5ad('data/train.h5ad')
🏃 Running the Pipeline
Option A: Single Session Training
Supervised training (with cell type labels):

bash
python main.py --mode train --config config.yaml --seed 42
Unsupervised training (without cell type labels):

bash
# First, edit config.yaml: set supervised: false
python main.py --mode train --config config.yaml --seed 42
Option B: Sequential Transfer Learning (4 Sessions)
Session 1 (Supervised, Initialization):

bash
python main.py --mode train --config sess1.yaml --seed 42
Session 2 (Unsupervised, Fine-tuning):

bash
# sess2.yaml has: resume_from: "checkpoints/session_1_best.pt"
python main.py --mode train --config sess2.yaml --seed 42
Session 3 (Supervised, Fine-tuning):

bash
# sess3.yaml has: resume_from: "checkpoints/session_2_best.pt"
python main.py --mode train --config sess3.yaml --seed 42
Session 4 (Unsupervised, Fine-tuning):

bash
# sess4.yaml has: resume_from: "checkpoints/session_3_best.pt"
python main.py --mode train --config sess4.yaml --seed 42
Inference: Batch Correction and Integration
Using trained model:

bash
python main.py --mode inference \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --data_path data/test.h5ad \
  --output_dir results/
Output files:

latent_embeddings.npy: Batch-corrected embeddings (z^(0))

corrected_expression.h5ad: Batch-corrected gene expression

cell_type_predictions.csv: Predicted cell types (unsupervised mode only)

Inference with DDPM Refinement
Add optional diffusion refinement (slower but potentially better):

bash
python main.py --mode inference \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --data_path data/test.h5ad \
  --output_dir results/
Edit config.yaml inference section:

text
inference:
  use_ddpm_refinement: true      # Enable refinement
  n_refinement_steps: 100        # Number of denoising steps (1-400)
  default_target_batch: 0        # Target batch for correction
📊 Monitoring Training
Training logs are automatically saved. Monitor progress:

bash
# Real-time logs (last 50 lines)
tail -f logs/training.log

# Full training history
less logs/training.log

# Extract key metrics
grep "Epoch\|Loss\|KL" logs/training.log
Expected metrics per epoch:

Reconstruction Loss (NLL): ~50-200 (lower is better)

VAE KL: ~0.1-5.0 (regularization)

Diffusion KL: ~5-20 (learning diffusion)

Total ELBO: Should decrease over time

🎯 Command Reference
Training Commands
Task	Command
Train supervised	python main.py --mode train --config config.yaml
Train unsupervised	python main.py --mode train --config config.yaml (with supervised: false)
Resume training	python main.py --mode train --config config.yaml (auto-resumes from checkpoint)
Train with custom seed	python main.py --mode train --config config.yaml --seed 123
Inference Commands
Task	Command
Get embeddings	python main.py --mode inference --config config.yaml --checkpoint checkpoints/best_model.pt --data_path data/test.h5ad
Save to custom dir	python main.py --mode inference --config config.yaml --checkpoint checkpoints/best_model.pt --data_path data/test.h5ad --output_dir my_results/
Batch correction only	python main.py --mode inference --config config.yaml --checkpoint checkpoints/best_model.pt --data_path data/test.h5ad
📋 Typical Workflow
Week 1: Setup & Session 1
bash
# Day 1-2: Setup
git clone https://github.com/supritum-sk6-comm/scARKIDS.git
cd scARKIDS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Day 3-7: Session 1 training (Supervised)
# Edit sess1.yaml with your first dataset paths
python main.py --mode train --config sess1.yaml --seed 42
# Training time: ~1-2 hours on GPU
# Output: checkpoints/session_1_best.pt
Week 2: Sessions 2 & 3
bash
# Session 2 (Unsupervised, ~1 hour)
python main.py --mode train --config sess2.yaml --seed 42
# Output: checkpoints/session_2_best.pt

# Session 3 (Supervised, ~1 hour)
python main.py --mode train --config sess3.yaml --seed 42
# Output: checkpoints/session_3_best.pt
Week 3: Session 4 & Evaluation
bash
# Session 4 (Unsupervised, ~1 hour)
python main.py --mode train --config sess4.yaml --seed 42
# Output: checkpoints/session_4_best.pt

# Inference on test data
python main.py --mode inference \
  --config config.yaml \
  --checkpoint checkpoints/session_4_best.pt \
  --data_path data/test.h5ad \
  --output_dir results/final/
🐛 Troubleshooting
Out of Memory (OOM)
bash
# Reduce batch size in config.yaml
batch_size: 128  # was 256

# Or reduce latent dimension (risky, changes architecture)
latent_dim: 10   # was 20 (NOT RECOMMENDED during transfer learning)
Data Loading Errors
python
# Check data format
import anndata as ad
adata = ad.read_h5ad('data/train.h5ad')
print(adata)
# Should show: AnnData object with n_obs, n_vars, obs with 'batch' and 'cell_type'
Missing Cell Type Labels (Supervised Mode)
python
# Add dummy labels if missing
import anndata as ad
adata = ad.read_h5ad('data/train.h5ad')
adata.obs['cell_type'] = 'Unknown'  # or infer from data
adata.write_h5ad('data/train.h5ad')
Checkpoint Not Found
bash
# Check available checkpoints
ls -lh checkpoints/

# Use explicit path
python main.py --mode inference --checkpoint checkpoints/session_1_best.pt ...
Slow Training
Check GPU usage: nvidia-smi (should show >80% GPU memory)

Increase batch size (if GPU memory allows)

Disable validation during training (edit training.py)

📚 Key Papers & Theory
Our framework combines:

VAE: Kingma & Welling (2013) - Auto-Encoding Variational Bayes

DDPM: Ho et al. (2020) - Denoising Diffusion Probabilistic Models

scRNA-seq: Tran et al. (2020) - Benchmark of batch correction methods

Full mathematical details: See scARKIDS_Paper_Rewritten.pdf

📈 Expected Results
Training Metrics (per epoch)
Reconstruction NLL: Starts ~200 → ~50

VAE KL: Starts ~5 → ~0.5-1.0

Diffusion KL: Starts ~50 → ~10-20

Total ELBO: Consistently decreasing

Validation Metrics (after training)
Batch Silhouette: > 0.3 (good batch separation)

Cell Type Silhouette: > 0.4 (good cell type clustering)

cLISI: 1.0-2.0 (cell type stratification)

bLISI: 2.0-3.0 (batch mixing)

Output Quality
Embeddings: Clean, well-separated cell types

Corrected expression: Batch effects removed, biology preserved

Cell type predictions: >85% accuracy on known cell types (unsupervised)

🤝 Contributing
To contribute:

Fork the repository

Create a feature branch: git checkout -b feature/your-feature

Commit changes: git commit -am 'Add feature'

Push to branch: git push origin feature/your-feature

Submit a pull request

📄 Citation
If you use scARKIDS, please cite:

text
@article{scARKIDS2025,
  title={scARKIDS: Single-cell RNA Integration via Variational Autoencoders and Denoising Diffusion Models},
  author={...},
  journal={...},
  year={2025}
}
📞 Support
Issues: Open on GitHub: https://github.com/supritum-sk6-comm/scARKIDS/issues

Discussions: GitHub Discussions: https://github.com/supritum-sk6-comm/scARKIDS/discussions

Documentation: See scARKIDS_Paper_Rewritten.pdf for mathematical details

📝 License
[Specify your license here, e.g., MIT, Apache 2.0, GPL]

🙏 Acknowledgments
Built with PyTorch, leveraging AnnData for single-cell data handling.

⚡ Quick Cheat Sheet
bash
# Setup (one-time)
git clone https://github.com/supritum-sk6-comm/scARKIDS.git && cd scARKIDS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Training
python main.py --mode train --config config.yaml

# Inference
python main.py --mode inference --config config.yaml --checkpoint checkpoints/best_model.pt --data_path data/test.h5ad

# Check GPU
nvidia-smi

# View logs
tail -f logs/training.log

# Environment details
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
Last Updated: November 2025