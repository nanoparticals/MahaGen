Generative Molecular Design via Multi-Objective RL and VGAE Gating
This repository contains the official implementation of our framework for targeted molecular design. It leverages a Stack-Augmented RNN sequence generator (operating on SELFIES) fine-tuned via Multi-Objective Reinforcement Learning (RL). To ensure chemical validity and adherence to the target chemical space, the RL agent is constrained by a Variational Graph Autoencoder (VGAE) acting as a latent-space gatekeeper.

🌟 Key Features
Robust Representation: Utilizes SELFIES for 100% chemically valid sequence generation.

VGAE Latent Regularization: Prevents the RL agent from exploiting the reward function by constraining generation to a learned latent space (using Mahalanobis/Euclidean distance thresholds).

Multi-Objective Optimization: Integrates Chemprop predictors natively into the reward function to simultaneously optimize for multiple properties (e.g., dipole moment mu, HOMO-LUMO gap, heat capacity cv, and TPSA).

Teacher Forcing & "Pulse" Injection: Supports injecting known high-performing sequences (teacher pool) during RL training to stabilize policy gradients.

Comprehensive Penalties: Built-in penalty systems for sequence length, heavy atom counts, and duplication (with LRU caching) to encourage diverse and compact generation.

📂 Repository Structure
train_rl_multiobj.py: The core Multi-Objective RL training loop using Policy Gradient. Handles soft/hard VGAE gating, Chemprop reward integration, and sequence sampling.

reinforcement.py: Implements the ReLeaSE-style policy gradient logic for the sequence generator.

reward_multiobjective.py: Complex reward calculator handling property scoring, Z-score normalization, duplication penalties, and structural heuristics.

stackRNN.py: The sequence generator architecture (GRU/LSTM with optional Stack augmentation and Z-conditioning).

train_vgae_ref2.py / vgae_prior_GAT.py: Scripts to train the Variational Graph Autoencoder and utilize it as a prior/scorer during RL.

gpu_*.sh: Sample SLURM batch scripts for running the pipeline (training predictors, pre-training generators, RL fine-tuning, and plotting).

⚙️ Installation
We recommend using Conda to manage your environment:

Bash
conda create -n generate310 python=3.10
conda activate generate310

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install cheminformatics and ML dependencies
pip install rdkit selfies chemprop numpy pandas tqdm joblib

🚀 Usage Pipeline
The generative pipeline operates in sequential stages. Sample SLURM scripts are provided for each step.

1. Train the Property Predictor (Chemprop)
Train a directed-message passing neural network to predict your target properties.

Bash
sbatch gpu_chemprop.sh

2. Train the VGAE Prior
Learn the latent space of your training dataset to act as a regularization gate.

Bash
sbatch gpu_vgae_and_z.sh

3. Pre-train the Sequence Generator
Train the base RNN model via supervised learning on your valid SELFIES dataset.

Bash
sbatch gpu_gen_condition_origin.sh

4. Multi-Objective RL Fine-Tuning
Fine-tune the generator to maximize property rewards while staying within the VGAE latent boundaries.

Bash
sbatch gpu_rl_vgae_soft_congen.sh
Note: You can adjust the gating strategy (--vgae-gate), predictor weights (--weights), and duplication penalties directly in the shell script.

5. Analysis and Plotting
Generate paper-ready figures, radar charts, and chemical space visualizations (t-SNE/UMAP).

Bash
sbatch gpu_aizynth_3.sh

