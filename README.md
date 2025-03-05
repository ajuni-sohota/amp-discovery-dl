# amp-discovery-dl
Deep learning to predict and generate novel antimicrobial peptides (AMPs)

# AMP-Discovery-DL: Deep Learning for Antimicrobial Peptide Discovery

This project implements a deep learning framework for the prediction and generation of novel antimicrobial peptides (AMPs) using TensorFlow/Keras. AMPs are short peptides that can kill or inhibit the growth of microorganisms, making them promising candidates for addressing the global crisis of antimicrobial resistance.

## Features

- **AMP Prediction**: CNN-LSTM hybrid model for predicting if a peptide sequence has antimicrobial activity
- **Novel AMP Generation**: Variational Autoencoder (VAE) model for generating new peptide sequences with antimicrobial potential
- **Comprehensive Analysis**: Tools for calculating and visualizing peptide properties and model performance
- **Flexible Pipeline**: Command-line interface for training, prediction, and generation modes

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Biopython
- Requests

## Installation

```bash
# Clone the repository
git clone https://github.com/ajuni-sohota/amp-discovery-dl.git
cd amp-discovery-dl

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
python amp_discovery.py --mode train --epochs 50 --batch_size 32 --latent_dim 32 --generate 10 --min_prob 0.8
```

This will:
1. Download or generate AMP and non-AMP sequences
2. Preprocess the data
3. Train the prediction and VAE models
4. Evaluate model performance
5. Generate novel AMPs

### Loading Pre-trained Models and Generating AMPs

```bash
python amp_discovery.py --mode generate --generate 20 --min_prob 0.7 --output my_generated_amps.csv --load_models
```

### Command-line Arguments

- `--mode`: Mode of operation (`train`, `predict`, `generate`)
- `--positive`: Path or URL to positive (AMP) sequences in FASTA format
- `--negative`: Path or URL to negative (non-AMP) sequences in FASTA format
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--latent_dim`: Latent dimension size for VAE
- `--generate`: Number of peptides to generate
- `--min_prob`: Minimum AMP probability threshold for generated peptides
- `--output`: Output file for results
- `--load_models`: Load pre-trained models instead of training new ones

## Model Architecture

### AMP Prediction Model

The prediction model uses a hybrid CNN-LSTM architecture:
- Convolutional layers to capture local sequence patterns
- Bidirectional LSTM layers to model long-range dependencies
- Dense layers for final classification

### AMP Generation Model (VAE)

The Variational Autoencoder consists of:
- An encoder network that compresses peptide sequences into a latent space
- A decoder network that generates peptide sequences from latent vectors
- A sampling mechanism that enables controlled generation of novel sequences

## Data Sources

The framework can download data from:
- [DBAASP Database](https://dbaasp.org/) for positive AMP examples
- For negative examples, it can either use provided non-AMP sequences or generate synthetic examples

If no data sources are provided, the framework will generate synthetic data for demonstration.

## Results

The system produces:
- Performance metrics for the prediction model (ROC, Precision-Recall curves, confusion matrix)
- Novel peptide sequences with high predicted antimicrobial activity
- Analysis of physico-chemical properties of generated peptides

## Example

```python
from amp_discovery import AMPDiscovery

# Initialize the framework
amp_framework = AMPDiscovery()

# Download data
pos_seqs, neg_seqs = amp_framework.download_data()

# Preprocess data
X_train, y_train, X_val, y_val, X_test, y_test = amp_framework.preprocess_data(pos_seqs, neg_seqs)

# Train prediction model
amp_framework.build_prediction_model()
history = amp_framework.train_prediction_model(X_train, y_train, X_val, y_val)

# Train VAE model
amp_framework.build_vae_model()
vae_history = amp_framework.train_vae_model(X_train, X_val)

# Generate novel peptides
novel_peptides = amp_framework.generate_peptides(n_samples=10)

# Analyze properties
properties = amp_framework.calculate_peptide_properties(novel_peptides)
print(properties)
```

## Future Development

- Integration with molecular dynamics simulation for structural analysis
- Addition of more advanced peptide property calculations
- Reinforcement learning components to optimize specific properties
- Web interface for user-friendly interaction

## Citation

If you use this code in your research, please cite:

```
@software{amp_discovery_2025,
  author = {Ajuni Sohota},
  title = {AMP-Discovery-DL: Deep Learning for Antimicrobial Peptide Discovery},
  year = {2025},
  url = {https://github.com/ajuni-sohota/amp-discovery-dl}
}
```

## License

MIT License

## Contact

Ajuni Sohota - ajunisohota@gmail.com

Project Link: [https://github.com/ajuni-sohota/amp-discovery-dl](https://github.com/ajuni-sohota/amp-discovery-dl)
