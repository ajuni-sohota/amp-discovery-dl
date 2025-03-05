#!/usr/bin/env python
# AMP-Discovery-DL: Deep Learning for Antimicrobial Peptide Prediction and Design
# Author: Ajuni Sohota (ajunisohota@gmail.com)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from Bio import SeqIO
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import random
import requests
from io import StringIO
import argparse

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class AMPDiscovery:
    """
    A deep learning framework for antimicrobial peptide discovery using TensorFlow/Keras.
    
    This class implements:
    1. Data preprocessing for peptide sequences
    2. CNN-LSTM model for AMP prediction
    3. Variational autoencoder (VAE) for novel AMP generation
    4. Visualization of peptide properties and model performance
    5. Utility functions for sequence analysis
    """
    
    def __init__(self, max_length=50):
        """
        Initialize the AMP discovery framework
        
        Args:
            max_length (int): Maximum peptide sequence length for padding/truncating
        """
        self.max_length = max_length
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_dict = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.prediction_model = None
        self.vae_model = None
        self.encoder = None
        self.decoder = None
        
    def download_data(self, positive_url=None, negative_url=None):
        """
        Download AMP datasets or use default sources
        
        Args:
            positive_url (str): URL for positive AMP examples
            negative_url (str): URL for negative non-AMP examples
            
        Returns:
            tuple: (positive_sequences, negative_sequences)
        """
        # Default data sources if none provided
        if positive_url is None:
            # Default to downloading from DBAASP database
            positive_url = "https://dbaasp.org/downloads/content/peptides/natural"
            
        if negative_url is None:
            # UniProt-derived non-AMPs could be used
            # For demonstration, we'll generate synthetic negative examples
            pass
        
        try:
            # Load positive examples (real AMPs)
            print("Downloading positive AMP examples...")
            if positive_url.startswith('http'):
                response = requests.get(positive_url)
                if response.status_code == 200:
                    positive_data = response.text
                    positive_sequences = self._parse_fasta(StringIO(positive_data))
                else:
                    raise Exception(f"Failed to download data: {response.status_code}")
            else:
                # Assume local file
                positive_sequences = self._parse_fasta(positive_url)
                
            # For demonstration, we'll generate synthetic negative examples
            # In a real project, you'd download real non-AMPs from UniProt or similar
            print("Generating synthetic negative examples...")
            negative_sequences = self._generate_negative_examples(len(positive_sequences))
            
            print(f"Downloaded/generated {len(positive_sequences)} positive and {len(negative_sequences)} negative examples")
            return positive_sequences, negative_sequences
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            # Generate sample data for demonstration
            return self._generate_sample_data()
    
    def _parse_fasta(self, fasta_file):
        """Parse FASTA file and extract sequences"""
        if isinstance(fasta_file, str):
            if fasta_file.startswith('http'):
                response = requests.get(fasta_file)
                sequences = SeqIO.parse(StringIO(response.text), 'fasta')
            else:
                sequences = SeqIO.parse(fasta_file, 'fasta')
        else:
            sequences = SeqIO.parse(fasta_file, 'fasta')
            
        return [str(record.seq) for record in sequences]
    
    def _generate_sample_data(self, n_samples=1000):
        """Generate synthetic data for demonstration"""
        # AMPs are typically rich in cationic and hydrophobic residues
        amp_weights = {'K': 0.12, 'R': 0.12, 'H': 0.08, 'L': 0.10, 'I': 0.08, 
                       'W': 0.06, 'F': 0.06, 'A': 0.08, 'C': 0.04, 'G': 0.04,
                       'V': 0.06, 'M': 0.02, 'S': 0.02, 'T': 0.02, 'P': 0.02,
                       'N': 0.02, 'Q': 0.02, 'D': 0.02, 'E': 0.02, 'Y': 0.02}
        
        # Non-AMPs have more balanced amino acid distribution
        non_amp_weights = {aa: 0.05 for aa in self.amino_acids}
        
        amp_sequences = []
        for _ in range(n_samples):
            length = random.randint(10, 40)  # AMPs usually 10-40 residues
            seq = ''.join(random.choices(list(amp_weights.keys()), 
                                         weights=list(amp_weights.values()), 
                                         k=length))
            amp_sequences.append(seq)
            
        non_amp_sequences = []
        for _ in range(n_samples):
            length = random.randint(10, 50)  # Non-AMPs can be longer
            seq = ''.join(random.choices(list(non_amp_weights.keys()), 
                                         weights=list(non_amp_weights.values()), 
                                         k=length))
            non_amp_sequences.append(seq)
            
        return amp_sequences, non_amp_sequences
    
    def _generate_negative_examples(self, n_samples):
        """Generate synthetic negative examples"""
        # Uniform distribution of amino acids for non-AMPs
        weights = {aa: 1/20 for aa in self.amino_acids}
        
        non_amp_sequences = []
        for _ in range(n_samples):
            length = random.randint(10, 50)  # Non-AMPs can be longer
            seq = ''.join(random.choices(list(weights.keys()), 
                                         weights=list(weights.values()), 
                                         k=length))
            non_amp_sequences.append(seq)
            
        return non_amp_sequences
    
    def preprocess_data(self, positive_seqs, negative_seqs):
        """
        Preprocess sequences for model training
        
        Args:
            positive_seqs (list): List of positive AMP sequences
            negative_seqs (list): List of negative non-AMP sequences
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Filter out sequences with non-standard amino acids
        valid_chars = set(self.amino_acids)
        positive_seqs = [seq for seq in positive_seqs if all(aa in valid_chars for aa in seq)]
        negative_seqs = [seq for seq in negative_seqs if all(aa in valid_chars for aa in seq)]
        
        # Balance dataset sizes if needed
        min_size = min(len(positive_seqs), len(negative_seqs))
        positive_seqs = positive_seqs[:min_size]
        negative_seqs = negative_seqs[:min_size]
        
        # Create labels
        X = positive_seqs + negative_seqs
        y = [1] * len(positive_seqs) + [0] * len(negative_seqs)
        
        # Create train/validation/test splits
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # One-hot encode sequences
        X_train_encoded = self._encode_sequences(X_train)
        X_val_encoded = self._encode_sequences(X_val)
        X_test_encoded = self._encode_sequences(X_test)
        
        print(f"Training set: {len(X_train)} sequences")
        print(f"Validation set: {len(X_val)} sequences")
        print(f"Test set: {len(X_test)} sequences")
        
        return X_train_encoded, np.array(y_train), X_val_encoded, np.array(y_val), X_test_encoded, np.array(y_test)
    
    def _encode_sequences(self, sequences):
        """One-hot encode amino acid sequences"""
        # Initialize array of zeros
        X = np.zeros((len(sequences), self.max_length, len(self.amino_acids)), dtype=np.float32)
        
        # Fill in one-hot encoded values
        for i, seq in enumerate(sequences):
            seq = seq[:self.max_length]  # Truncate if too long
            for j, aa in enumerate(seq):
                if aa in self.aa_dict:
                    X[i, j, self.aa_dict[aa]] = 1
                    
        return X
    
    def _decode_one_hot(self, one_hot_matrix):
        """Convert one-hot encoded matrix back to amino acid sequence"""
        sequences = []
        for seq_matrix in one_hot_matrix:
            seq = ""
            for pos in seq_matrix:
                if np.max(pos) > 0:  # Check if position contains an amino acid
                    aa_index = np.argmax(pos)
                    seq += self.amino_acids[aa_index]
            sequences.append(seq)
        return sequences
    
    def build_prediction_model(self):
        """
        Build CNN-LSTM hybrid model for AMP prediction
        
        Returns:
            keras.Model: Compiled model for AMP prediction
        """
        # Input layer
        inputs = keras.Input(shape=(self.max_length, len(self.amino_acids)))
        
        # Convolutional layers to capture local patterns
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 7, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Bidirectional LSTM to capture sequence dependencies
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, 
                                             dropout=0.3, 
                                             recurrent_dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3))(x)
        
        # Dense layers for classification
        x = layers.Dense(64, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile model
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.AUC(name='auc'),
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )
        
        self.prediction_model = model
        return model
    
    def build_vae_model(self, latent_dim=32):
        """
        Build Variational Autoencoder for peptide generation
        
        Args:
            latent_dim (int): Dimension of latent space
            
        Returns:
            tuple: (vae_model, encoder_model, decoder_model)
        """
        # === ENCODER ===
        encoder_inputs = keras.Input(shape=(self.max_length, len(self.amino_acids)))
        
        # Convolutional layers
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        
        # Flatten and dense
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # VAE latent space parameters
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        
        # Sampling function as a Lambda layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
        z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        
        # Create encoder model
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # === DECODER ===
        latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
        
        # Reshape to the right dimensions
        x = layers.Dense(128, activation='relu')(latent_inputs)
        x = layers.Dense((self.max_length // 4) * 32, activation='relu')(x)
        x = layers.Reshape((self.max_length // 4, 32))(x)
        
        # Upsampling with Conv1D transpose
        x = layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer - one-hot amino acid probabilities
        decoder_outputs = layers.Conv1D(len(self.amino_acids), 3, activation='softmax', padding='same')(x)
        
        # Create decoder model
        decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        
        # === COMPLETE VAE MODEL ===
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = keras.Model(encoder_inputs, outputs, name='vae')
        
        # VAE loss function
        def vae_loss(inputs, outputs):
            # Reconstruction loss
            recon_loss = keras.losses.categorical_crossentropy(
                tf.reshape(inputs, [-1, len(self.amino_acids)]),
                tf.reshape(outputs, [-1, len(self.amino_acids)])
            )
            recon_loss = tf.reduce_sum(recon_loss, axis=0)
            
            # KL divergence regularization
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
            
            # Total loss
            return tf.reduce_mean(recon_loss + kl_loss)
        
        # Compile VAE model
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=vae_loss)
        
        self.vae_model = vae
        self.encoder = encoder
        self.decoder = decoder
        
        return vae, encoder, decoder
        
    def train_prediction_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the AMP prediction model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            keras.callbacks.History: Training history
        """
        if self.prediction_model is None:
            self.build_prediction_model()
            
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train the model
        history = self.prediction_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def train_vae_model(self, X_train, X_val, epochs=50, batch_size=32, latent_dim=32):
        """
        Train the VAE model for peptide generation
        
        Args:
            X_train: Training data (one-hot encoded sequences)
            X_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            latent_dim (int): Dimension of latent space
            
        Returns:
            keras.callbacks.History: Training history
        """
        if self.vae_model is None:
            self.build_vae_model(latent_dim=latent_dim)
            
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train the model
        history = self.vae_model.fit(
            X_train, X_train,  # VAE input = output (autoencoder)
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_prediction_model(self, X_test, y_test):
        """
        Evaluate the AMP prediction model
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            dict: Performance metrics
        """
        if self.prediction_model is None:
            raise ValueError("Prediction model has not been trained yet")
            
        # Get predictions
        y_pred_prob = self.prediction_model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Metrics
        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
        
        return metrics
    
    def plot_metrics(self, metrics):
        """
        Plot evaluation metrics
        
        Args:
            metrics (dict): Performance metrics from evaluate_prediction_model
        """
        # Create figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot ROC curve
        axs[0].plot(metrics['fpr'], metrics['tpr'], lw=2, 
                   label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        axs[0].plot([0, 1], [0, 1], 'k--', lw=2)
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.05])
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver Operating Characteristic (ROC)')
        axs[0].legend(loc="lower right")
        
        # Plot Precision-Recall curve
        axs[1].plot(metrics['recall'], metrics['precision'], lw=2,
                   label=f'PR curve (AUC = {metrics["pr_auc"]:.3f})')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].set_title('Precision-Recall Curve')
        axs[1].legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig('amp_model_performance.png', dpi=300)
        plt.show()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-AMP', 'AMP'],
                   yticklabels=['Non-AMP', 'AMP'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('amp_confusion_matrix.png', dpi=300)
        plt.show()
    
    def generate_peptides(self, n_samples=10, min_amp_probability=0.8):
        """
        Generate novel peptide sequences using the VAE model
        
        Args:
            n_samples (int): Number of peptide sequences to generate
            min_amp_probability (float): Minimum probability threshold for AMPs
            
        Returns:
            list: Generated peptide sequences that pass the AMP threshold
        """
        if self.decoder is None or self.prediction_model is None:
            raise ValueError("Both VAE and prediction models must be trained first")
        
        # Generate from random points in latent space
        z_sample = np.random.normal(size=(100, self.encoder.output[0].shape[1]))
        
        # Decode the latent vectors
        decoded = self.decoder.predict(z_sample)
        
        # Convert one-hot to sequences
        generated_sequences = []
        for seq in decoded:
            peptide = ""
            for pos in seq:
                aa_index = np.argmax(pos)
                peptide += self.amino_acids[aa_index]
            generated_sequences.append(peptide)
        
        # Filter sequences by AMP prediction probability
        valid_sequences = []
        for seq in generated_sequences:
            # One-hot encode the sequence
            encoded_seq = np.zeros((1, self.max_length, len(self.amino_acids)))
            for i, aa in enumerate(seq[:self.max_length]):
                if aa in self.aa_dict:
                    encoded_seq[0, i, self.aa_dict[aa]] = 1
            
            # Predict AMP probability
            amp_prob = self.prediction_model.predict(encoded_seq)[0][0]
            
            if amp_prob >= min_amp_probability:
                valid_sequences.append((seq, amp_prob))
                
                if len(valid_sequences) >= n_samples:
                    break
        
        return valid_sequences
    
    def calculate_peptide_properties(self, sequences):
        """
        Calculate physico-chemical properties of peptides
        
        Args:
            sequences (list): List of peptide sequences
            
        Returns:
            pd.DataFrame: DataFrame with peptide properties
        """
        properties = []
        
        for seq in sequences:
            if isinstance(seq, tuple):
                # If input is (sequence, probability) tuple from generation
                seq, amp_prob = seq
                prop = {'sequence': seq, 'amp_probability': amp_prob}
            else:
                prop = {'sequence': seq, 'amp_probability': None}
            
            # Length
            prop['length'] = len(seq)
            
            # Amino acid composition
            aa_count = Counter(seq)
            for aa in self.amino_acids:
                prop[f'{aa}_count'] = aa_count.get(aa, 0)
                prop[f'{aa}_percent'] = aa_count.get(aa, 0) / len(seq) * 100
                
            # Charge (approximation)
            positive_charge = aa_count.get('K', 0) + aa_count.get('R', 0) + aa_count.get('H', 0)
            negative_charge = aa_count.get('D', 0) + aa_count.get('E', 0)
            prop['net_charge'] = positive_charge - negative_charge
            
            # Hydrophobic amino acids
            hydrophobic = aa_count.get('A', 0) + aa_count.get('V', 0) + aa_count.get('L', 0) + \
                         aa_count.get('I', 0) + aa_count.get('M', 0) + aa_count.get('F', 0) + \
                         aa_count.get('W', 0)
            prop['hydrophobic_percent'] = hydrophobic / len(seq) * 100
            
            properties.append(prop)
            
        return pd.DataFrame(properties)
    
    def save_models(self, prediction_path='amp_prediction_model.h5', 
                   vae_path='amp_vae_model.h5',
                   encoder_path='amp_encoder_model.h5',
                   decoder_path='amp_decoder_model.h5'):
        """
        Save trained models to files
        
        Args:
            prediction_path (str): Path to save prediction model
            vae_path (str): Path to save VAE model
            encoder_path (str): Path to save encoder model
            decoder_path (str): Path to save decoder model
        """
        if self.prediction_model is not None:
            self.prediction_model.save(prediction_path)
            print(f"Prediction model saved to {prediction_path}")
            
        if self.vae_model is not None:
            self.vae_model.save(vae_path)
            print(f"VAE model saved to {vae_path}")
            
        if self.encoder is not None:
            self.encoder.save(encoder_path)
            print(f"Encoder model saved to {encoder_path}")
            
        if self.decoder is not None:
            self.decoder.save(decoder_path)
            print(f"Decoder model saved to {decoder_path}")
            
    def load_models(self, prediction_path='amp_prediction_model.h5', 
                   vae_path='amp_vae_model.h5',
                   encoder_path='amp_encoder_model.h5',
                   decoder_path='amp_decoder_model.h5'):
        """
        Load trained models from files
        
        Args:
            prediction_path (str): Path to prediction model
            vae_path (str): Path to VAE model
            encoder_path (str): Path to encoder model
            decoder_path (str): Path to decoder model
        """
        if os.path.exists(prediction_path):
            self.prediction_model = keras.models.load_model(prediction_path)
            print(f"Prediction model loaded from {prediction_path}")
            
        if os.path.exists(vae_path):
            self.vae_model = keras.models.load_model(vae_path, 
                                                    custom_objects={'vae_loss': None})
            print(f"VAE model loaded from {vae_path}")
            
        if os.path.exists(encoder_path):
            self.encoder = keras.models.load_model(encoder_path)
            print(f"Encoder model loaded from {encoder_path}")
            
        if os.path.exists(decoder_path):
            self.decoder = keras.models.load_model(decoder_path)
            print(f"Decoder model loaded from {decoder_path}")

def main():
    """Main function to demonstrate the AMP discovery pipeline"""
    print("AMP-Discovery-DL: Deep Learning for Antimicrobial Peptide Discovery")
    print("Author: Ajuni Sohota (ajunisohota@gmail.com)")
    print("GitHub: https://github.com/ajuni-sohota/amp-discovery-dl")
    print("\n")
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='AMP-Discovery-DL: Deep Learning for Antimicrobial Peptide Prediction and Design')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'generate'],
                        help='Mode: train models, predict on sequences, or generate novel AMPs')
    parser.add_argument('--positive', type=str, default=None, 
                        help='Path or URL to positive (AMP) sequences in FASTA format')
    parser.add_argument('--negative', type=str, default=None,
                        help='Path or URL to negative (non-AMP) sequences in FASTA format')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension size for VAE')
    parser.add_argument('--generate', type=int, default=10, help='Number of peptides to generate')
    parser.add_argument('--min_prob', type=float, default=0.8, 
                        help='Minimum AMP probability threshold for generated peptides')
    parser.add_argument('--output', type=str, default='generated_amps.csv',
                        help='Output file for results')
    parser.add_argument('--load_models', action='store_true',
                        help='Load pre-trained models instead of training new ones')
    args = parser.parse_args()
    
    # Initialize the AMP discovery framework
    amp_discovery = AMPDiscovery(max_length=50)
    
    if args.mode == 'train':
        print("=== Training mode ===")
        
        if args.load_models:
            # Load pre-trained models
            amp_discovery.load_models()
        else:
            # Download or generate data
            pos_seqs, neg_seqs = amp_discovery.download_data(
                positive_url=args.positive,
                negative_url=args.negative
            )
            
            # Preprocess data
            X_train, y_train, X_val, y_val, X_test, y_test = amp_discovery.preprocess_data(
                pos_seqs, neg_seqs
            )
            
            # Build and train prediction model
            print("\n--- Training AMP prediction model ---")
            amp_discovery.build_prediction_model()
            history = amp_discovery.train_prediction_model(
                X_train, y_train, X_val, y_val,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            # Evaluate model
            print("\n--- Evaluating AMP prediction model ---")
            metrics = amp_discovery.evaluate_prediction_model(X_test, y_test)
            amp_discovery.plot_metrics(metrics)
            
            # Build and train VAE model
            print("\n--- Training VAE model for peptide generation ---")
            amp_discovery.build_vae_model(latent_dim=args.latent_dim)
            vae_history = amp_discovery.train_vae_model(
                X_train, X_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                latent_dim=args.latent_dim
            )
            
            # Save models
            amp_discovery.save_models()
        
        # Generate novel peptides
        print("\n--- Generating novel antimicrobial peptides ---")
        generated_peptides = amp_discovery.generate_peptides(
            n_samples=args.generate,
            min_amp_probability=args.min_prob
        )
        
        # Analyze generated peptides
        if generated_peptides:
            properties_df = amp_discovery.calculate_peptide_properties(generated_peptides)
            print("\nGenerated peptides:")
            print(properties_df[['sequence', 'amp_probability', 'length', 'net_charge', 
                              'hydrophobic_percent']].to_string(index=False))
            
            # Save to file
            properties_df.to_csv(args.output, index=False)
            print(f"\nGenerated peptides saved to {args.output}")
        else:
            print("No peptides meeting the criteria were generated. Try lowering the min_prob threshold.")
            
    elif args.mode == 'predict':
        # Load models for prediction
        amp_discovery.load_models()
        
        # TODO: Implement prediction on user-provided sequences
        print("Prediction mode is not fully implemented yet.")
        
    elif args.mode == 'generate':
        # Load models for generation
        amp_discovery.load_models()
        
        # Generate novel peptides
        print("--- Generating novel antimicrobial peptides ---")
        generated_peptides = amp_discovery.generate_peptides(
            n_samples=args.generate,
            min_amp_probability=args.min_prob
        )
        
        # Analyze generated peptides
        if generated_peptides:
            properties_df = amp_discovery.calculate_peptide_properties(generated_peptides)
            print("\nGenerated peptides:")
            print(properties_df[['sequence', 'amp_probability', 'length', 'net_charge', 
                              'hydrophobic_percent']].to_string(index=False))
            
            # Save to file
            properties_df.to_csv(args.output, index=False)
            print(f"\nGenerated peptides saved to {args.output}")
        else:
            print("No peptides meeting the criteria were generated. Try lowering the min_prob threshold.")

if __name__ == "__main__":
    main()
